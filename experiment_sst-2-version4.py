# pylint: disable=g-bad-file-header
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import matplotlib.pyplot as plt
import argparse
from typing import NamedTuple, Optional
from tqdm import tqdm
import pandas as pd
import haiku as hk
import numpy as np
import jax
import jax.numpy as jnp
from jax.random import PRNGKey

import optax
from datasets import load_dataset
from enn.networks.bert.bert_v2 import BertConfigCustom, make_bert_enn
from huggingface_haiku_loader import (
    load_pretrained_bert_weights,
    embedding_params_fix,
    layer_params_fix,
)
from plots_and_files import plot_loss_curve, plot_loss_and_accuracy_curve, save_results_to_file, plot_all_runs_with_mean
from bert_processing import postprocess_key, tokenize_input, preprocess_dataset
from bert_enn import BertInput, ArrayBatch
########################################################################

# Some light postprocessing to make the parameter keys a bit more concise

def update_step(params, state, frozen_params, opt_state, batch, rng_key, apply_fn, optimizer, indexer):
    # Combine frozen BERT with trainable classifier
    combined_params = hk.data_structures.merge(frozen_params, params)
    
    def loss_fn_inner(trainable_params):
        combined = hk.data_structures.merge(frozen_params, trainable_params)
        #logits, _ = apply_fn(combined, rng_key, batch.x)
        index = indexer(rng_key)
        output, new_state = apply_fn(combined, state, batch.x, index)
        #output, new_state = apply_fn(combined, state, rng_key, batch.x, index)
        logits = output.extra['classification_logits']

        #probs = jax.nn.softmax(logits, axis=-1)
        #preds = jnp.argmax(probs, axis=-1)
        
       # labels = jax.nn.one_hot(batch.y, 2)
       # labels = labels.astype(jnp.int32)
        labels = batch.y.astype(jnp.int32)
        #return optax.softmax_cross_entropy(logits, labels).mean()
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    grads = jax.grad(loss_fn_inner)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    # Compute loss and state update again
    combined = hk.data_structures.merge(frozen_params, new_params)
    index = indexer(rng_key)
    output, new_state = apply_fn(combined, state, batch.x, index)
    logits = output.extra['classification_logits']
    labels = batch.y.astype(jnp.int32)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    return new_params, new_state, new_opt_state, loss

update_step = jax.jit(update_step, static_argnames=["apply_fn", "optimizer", "indexer"])


# BERT base fine tune on SST2 
class BERTENNFineTuneSST2:
    def __init__(
        self,
        model,
        dataset,
        tokenizer,
        rng,
        batch_size,
        criterion: str,
        num_steps: Optional[int],
        learning_rate,
        train_all: bool = False,
        pretrained_params=None
    ):
        self.model = model 
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.rng = hk.PRNGSequence(rng)
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.train_all = train_all
        self.indexer = model.indexer

        if criterion == 'num_steps':
            if num_steps is None:
                print("Error, selected num_steps criterion but"
                      " did not provide num_steps")
                exit()
        elif criterion == 'num_epochs':
            if num_epochs is None:
                print("Error, selected num_epochs criterion but"
                      " did not provide num_epochs")
                exit()
        else:
            print("Error in criterion")
            exit()

        dummy_input = BertInput(
            token_ids=jnp.zeros((1, 128), dtype=jnp.int32),
            segment_ids=jnp.zeros((1, 128), dtype=jnp.int32),
            input_mask=jnp.ones((1, 128), dtype=jnp.int32)
        )
        dummy_index = self.indexer(key=next(self.rng))
        self.params, self.state = self.model.init(next(self.rng), dummy_input, dummy_index)
        
        if pretrained_params is not None:
            self.params = pretrained_params

        self.params = hk.data_structures.to_immutable_dict(self.params)
        self.labels = jnp.array([ex['label'] for ex in self.dataset])

        def print_tree(params, prefix=""):
            for k, v in params.items():
                if isinstance(v, dict):
                    print_tree(v, prefix + k + "/")
                else:
                    print(prefix + k)
        
       # print_tree(self.params)

        def is_trainable(module_name, name, value):
            if self.train_all:
                return True
            return (
                "classifier_head" in module_name
                or "pooler_dense" in module_name
                or "attention_output_11" in module_name
                or "intermediate_output_11" in module_name
                or "query_11" in module_name
                or "values_11" in module_name
                or "layer_output_11" in module_name
                or "attention_output_10" in module_name
                or "intermediate_output_10" in module_name
                or "query_10" in module_name
                or "values_10" in module_name
                or "layer_output_10" in module_name
                or "train_epinet" in module_name
            )
        
        self.frozen_params, self.trainable_params = hk.data_structures.partition(
            lambda mod, name, val: not is_trainable(mod, name, val),
            self.params
        )

        print("Trainable params:")
        for k in self.trainable_params.keys():
            print(k)
        
        self.opt = optax.adam(learning_rate)
        self.opt_state = self.opt.init(self.trainable_params)

        print(f"DEBUG: params before learning: \n{self.trainable_params['BERT/classifier_head']['w']}")

    def tokenize(self, texts):
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="np",
        )
        return tokens


    def compute_accuracy(self, logits: jnp.ndarray, labels: jnp.ndarray) -> float:
        predictions = jnp.argmax(logits, axis=-1)
        return jnp.mean(predictions == labels)

    def run(self):
        train_labels = jnp.array([item['label'] for item in self.dataset])
        all_indices = jnp.arange(len(train_labels))
        
        losses = []
        accuracies = []
        indexer_fn = lambda key: self.indexer(key=key)

        for step in tqdm(range(self.num_steps)):
            batch_indices = jax.random.choice(
            key=next(self.rng),
            a=jnp.arange(len(self.dataset)),
            shape=(self.batch_size,),
            replace=True,
        )
            batch_input_ids = jnp.stack([jnp.array(self.dataset[int(i)]['input_ids']) for i in batch_indices])
            batch_token_type_ids = jnp.stack([jnp.array(self.dataset[int(i)]['token_type_ids']) for i in batch_indices])
            batch_attention_mask = jnp.stack([jnp.array(self.dataset[int(i)]['attention_mask']) for i in batch_indices])
 
#            batch_labels = jnp.array([self.dataset[int(i)]['label'] for i in batch_indices])
            batch_labels = self.labels[batch_indices]
        
            input = BertInput(
                token_ids=batch_input_ids,
                segment_ids=batch_token_type_ids,
                input_mask=batch_attention_mask,
            )
            batch = ArrayBatch(x=input, y=batch_labels)

            ##############################################

            self.trainable_params, self.state, self.opt_state, loss = update_step(
                self.trainable_params,
                self.state,
                frozen_params=self.frozen_params,
                opt_state=self.opt_state,
                batch=batch,
                rng_key=next(self.rng),
#                apply_fn=base_model.apply,
                apply_fn=self.model.apply,
                optimizer=self.opt,
                indexer=indexer_fn,
            )
    
            # Compute accuracy
            combined_params = hk.data_structures.merge(self.frozen_params, self.trainable_params)
            #output, _ = base_model.apply(combined_params, next(self.rng), batch.x)
            index = self.indexer(next(self.rng))
#            output, _ = self.model.apply(combined_params, self,state, next(self.rng), batch.x, index=index)
            output, _ = self.model.apply(
                params=combined_params, 
                state=self.state,
                inputs=batch.x, 
                index=index
            )
            logits = output.extra['classification_logits']
            acc = float(self.compute_accuracy(logits, batch.y))
            losses.append(float(loss))
            accuracies.append(acc)
    
        return losses, accuracies



if __name__ == "__main__":
    print("\n\nFine tuning of BERT ENN model for SST2 classification\n\n")

    parser = argparse.ArgumentParser(description="Fine-tune BERT on SST2")
    parser.add_argument('--train_all', action='store_true', help='If set, trains all model parameters.')
    parser.add_argument('--test', action='store_true', help='If set, runs a faster test on 50 training steps and 3 number of repetitions')
    parser.add_argument('--suffix', type=str, default="", help="Optional suffix for output file names.")
    parser.add_argument('--save_params', action='store_true', help='If set, saves the final trained parameters.')
    parser.add_argument('--load_params', action='store_true', help='If set, loads previously saved parameters.')
#    parser.add_argument('--params_path', type=str, default='saved_params.npz', help='Path to load/save model parameters.')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate for optimizer.') 
    args = parser.parse_args()

    SEED = 0
    rng = hk.PRNGSequence(SEED)

    print("Downloading BERT base model from Hugging Face 🤗...")
    # Load HuggingFace model + weights
    tokenizer, config, hf_weights = load_pretrained_bert_weights()
    bert_config_custom = BertConfigCustom(config)
    print("...done")

    epinet_config = {  #see C.2.4 fine tuning paper
        "epinet_hiddens": [50, 50] ,
        "num_classes": 2,
        "index_dim": 10, 
        "prior_scale": 1,
    }
    print("Building ENN model from BERT base...")
    base_model = make_bert_enn(
        bert_config=config,
        epinet_config=epinet_config,
        is_training=True
    )
    print("..done")

    # Create dummy input
    dummy_input = BertInput(
        token_ids=jnp.zeros((1, 128), dtype=jnp.int32),
        segment_ids=jnp.zeros((1, 128), dtype=jnp.int32),
        input_mask=jnp.ones((1, 128), dtype=jnp.int32),
    )
    
    print(f"Extracting indexer...")
    indexer = base_model.indexer
    print(f"...done")

    # Sampling index
    index = indexer(key=next(rng))

    # Initialize random Haiku parameters
    params, state = base_model.init(
        jax.random.PRNGKey(0), 
        dummy_input,
#        next(rng)
        index,
    )
    
    # Apply embedding + encoder weights
    params = embedding_params_fix(params, hf_weights)
    for layer in range(config.num_hidden_layers):
        params = layer_params_fix(params, hf_weights, layer)

   
    print("Loading SST2 dataset...")
    dataset = load_dataset(
        "nyu-mll/glue", 
        "sst2",
    )
    print("...done") 
 
    print("Pretokenizing the dataset...")
    dataset = dataset.map(lambda x: {'sentence': x['sentence'], 'label': x['label']})
    dataset = preprocess_dataset(dataset['train'], tokenizer)
    print("...done")


    config = {
        'batch_size': 32,
        'num_steps': 1000,
        'epochs': 1,
        'learning_rate': args.learning_rate,
        'seed': rng,
        'num_runs': 5,
    }
    if args.test:
        config['num_steps'] = 50
        config['num_runs'] = 3

    all_loss_df = pd.DataFrame()
    all_acc_df = pd.DataFrame()

    if args.load_params and os.path.exists(args.params_path):
        print(f"🔁 Loading pretrained parameters from: {args.params_path}")
        with np.load(args.params_path, allow_pickle=True) as f:
            loaded_params = f['params'].item()
        params = hk.data_structures.to_immutable_dict(loaded_params)
    else:
        print("🚀 Using default initialized parameters (with preloaded encoder weights)")
    
    print("Running experiment...")
    for i in range(config['num_runs']):
        print(f"\n▶️ Run {i + 1}/{config['num_runs']}") 

        experiment = BERTENNFineTuneSST2(
            model=base_model,
            dataset = dataset,
            tokenizer = tokenizer,
            rng = next(config['seed']),
            batch_size = config['batch_size'],
            criterion = 'num_steps',
            num_steps = config['num_steps'],
            learning_rate = config['learning_rate'],
            train_all = args.train_all,
            pretrained_params=params
        )
        losses, acc = experiment.run()

        all_loss_df[f'run_{i+1}'] = losses
        all_acc_df[f'run_{i+1}'] = acc

        if args.save_params:
            print(f"💾 Saving final trained parameters to: {args.params_path}")
            final_combined_params = hk.data_structures.merge(experiment.frozen_params, experiment.trainable_params)
            np.savez(args.params_path, params=final_combined_params)

    suffix = f"_{args.suffix}" if args.suffix else ""

    if args.train_all:
        losses_file_name = f"outputs/all_params_losses{suffix}.csv"
        accs_file_name = f"outputs/all_accs_losses{suffix}.csv"
        img_file_name = f"outputs/all_params_BERT-base{suffix}"
    else:
        losses_file_name = f"outputs/subset_params_losses{suffix}.csv"
        accs_file_name = f"outputs/subset_params_accs{suffix}.csv"
        img_file_name = f"outputs/subset_params_BERT-base{suffix}"
        all_loss_df.to_csv(losses_file_name , index_label="step")
        all_acc_df.to_csv(accs_file_name , index_label="step")
   

   # plot_loss_and_accuracy_curve(losses_file_name, accs_file_name, img_file_name)
    plot_all_runs_with_mean(losses_file_name, accs_file_name, img_file_name) 
    save_results_to_file(losses, acc)
    print(f"DEBUG: params after learning: \n{experiment.trainable_params['BERT/classifier_head']['w']}")
