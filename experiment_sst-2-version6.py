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
import json
import time
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
import jax.profiler
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
from enn.active_learning import priorities 
from enn.networks import forwarders

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
        val_dataset,
        tokenizer,
        rng,
        batch_size,
        criterion: str,
        num_steps: Optional[int],
        learning_rate,
        train_all: bool = False,
        pretrained_params=None,
        priority_criterion: str='uniform',
        priority_fn_enn_samples: int=10,
        batch_gradient_enn_samples: int=10,
        NB: int = 40,
        nb: int = 4,
    ):
        self.model = model 
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.rng = hk.PRNGSequence(rng)
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.train_all = train_all
        self.indexer = model.indexer
        self.priority_fn_enn_samples = priority_fn_enn_samples
        self.batch_gradient_enn_samples = batch_gradient_enn_samples
        self.NB = NB
        self.nb = nb

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

        # preprocessing of the dataset
        self.input_ids = jnp.stack([jnp.array(ex["input_ids"]) for ex in self.dataset])
        self.token_type_ids = jnp.stack([jnp.array(ex["token_type_ids"]) for ex in self.dataset])
        self.attention_masks = jnp.stack([jnp.array(ex["attention_mask"]) for ex in self.dataset])

        self.val_input_ids = jnp.stack([jnp.array(ex["input_ids"]) for ex in self.val_dataset])
        self.val_token_type_ids = jnp.stack([jnp.array(ex["token_type_ids"]) for ex in self.val_dataset])
        self.val_attention_masks = jnp.stack([jnp.array(ex["attention_mask"]) for ex in self.val_dataset])



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
        
        if self.train_all:
            print("\n\nTraining all parameters of model\n")
        else:
            print("\n\nTraining subset of parameters of the model\n")

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
        # Priority functions setup
        print("Constructing priority function...")
        priority_fn_ctor = priorities.get_priority_fn_ctor(priority_criterion)
        print("...done")
        print(f"[DEBUG] Using priority: {priority_criterion}")

        print("Setting up enn batch forwarder...")
        self.enn_batch_fwd = forwarders.make_batch_fwd(
            self.model, num_enn_samples=self.priority_fn_enn_samples 
        )
        print("...done")
        print("Constructing priority function...")
        self.priority_fn = priority_fn_ctor(self.enn_batch_fwd)
        print("...done")
    
    def tokenize(self, texts):
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="np",
        )
        return tokens

#    def select_uncertain_samples(self, pool, num_samples=10):
#        tokens = self.tokenize(pool['sentence'])
#        input_ids = tokens['input_ids']
#        batch = datasets.ArrayBatch(x=input_ids, y=np.array(pool['label']))
#        scores, _ = self.priority_fn(self.params, self.state, batch, next(self.rng))
#        selected_indices = jnp.argsort(scores)[-num_samples:]
#        return [pool[i] for i in selected_indices]


    def evaluate_validation_loss(self):
        val_labels = jnp.array([item['label'] for item in self.val_dataset])
        batch_size = 32  # or 64 or whatever your GPU can handle
        num_batches = len(val_labels) // batch_size + (len(val_labels) % batch_size != 0)
    
        losses = []
    
        for i in range(num_batches):
            batch_indices = jnp.arange(i * batch_size, min((i + 1) * batch_size, len(val_labels)))
    
            batch_input_ids = jnp.stack([jnp.array(self.val_dataset[int(i)]['input_ids']) for i in batch_indices])
            batch_token_type_ids = jnp.stack([jnp.array(self.val_dataset[int(i)]['token_type_ids']) for i in batch_indices])
            batch_attention_mask = jnp.stack([jnp.array(self.val_dataset[int(i)]['attention_mask']) for i in batch_indices])
            batch_labels = val_labels[batch_indices]
    
            input = BertInput(
                token_ids=batch_input_ids,
                segment_ids=batch_token_type_ids,
                input_mask=batch_attention_mask,
            )
            batch = ArrayBatch(x=input, y=batch_labels)
    
            combined_params = hk.data_structures.merge(self.frozen_params, self.trainable_params)
            output, _ = self.model.apply(combined_params, self.state, batch.x, index=self.indexer(next(self.rng)))
            logits = output.extra['classification_logits']
    
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch.y).mean()
            losses.append(loss)
    
        mean_val_loss = jnp.mean(jnp.array(losses))
        return float(mean_val_loss)



    def compute_accuracy(self, logits: jnp.ndarray, labels: jnp.ndarray) -> float:
        predictions = jnp.argmax(logits, axis=-1)
        return jnp.mean(predictions == labels)

    def run(self):
        run_start = time.time()
        train_labels = jnp.array([item['label'] for item in self.dataset])
        all_indices = jnp.arange(len(train_labels))
        
        losses = []
        accuracies = []

        indexer_fn = lambda key: self.indexer(key=key)

        for step in tqdm(range(self.num_steps)):
            candidate_indices = jax.random.choice(
                key=next(self.rng),
                a=jnp.arange(len(self.dataset)),
                shape=(self.NB,),
                replace=False,
            )
            
            # Build candidate batch
            candidate_input_ids = self.input_ids[candidate_indices]
            candidate_token_type_ids = self.token_type_ids[candidate_indices]
            candidate_attention_mask = self.attention_masks[candidate_indices]
 
            candidate_input = BertInput(
                token_ids=candidate_input_ids,
                segment_ids=candidate_token_type_ids,
                input_mask=candidate_attention_mask,
            )

            # Build ArrayBatch for the candidates
            candidate_batch = ArrayBatch(
                x=candidate_input,
                y=self.labels[candidate_indices]
            )
            
            # Score candidate examples with the priority function
            scores, _ = self.priority_fn(
                params=hk.data_structures.merge(self.frozen_params, self.trainable_params),
                state=self.state,
                batch=candidate_batch,
                key=next(self.rng)
            )
            
            # Select the top self.batch_size samples
            selected_indices_in_pool = jnp.argsort(scores)[-self.nb:]
            batch_indices = candidate_indices[selected_indices_in_pool]

            batch_input_ids = self.input_ids[batch_indices]
            batch_token_ids = self.token_type_ids[batch_indices]
            batch_attention_mask = self.attention_masks[batch_indices]
 
#            batch_labels = jnp.array([self.dataset[int(i)]['label'] for i in batch_indices])
            batch_labels = self.labels[batch_indices]
        
            input = BertInput(
                token_ids=batch_input_ids,
                segment_ids=batch_token_ids,
                input_mask=batch_attention_mask,
            )
            batch = ArrayBatch(x=input, y=batch_labels)

            ##############################################
            # ======= UPDATE STEP ======= 
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
            
            losses.append(float(loss))

            # === VALIDATION EVALUATION every 100 steps ===
            if step % 100 == 0:
                val_loss = self.evaluate_validation_loss()
                #val_losses.append(val_loss)
            
            # Compute accuracy
#            combined_params = hk.data_structures.merge(self.frozen_params, self.trainable_params)
#            #output, _ = base_model.apply(combined_params, next(self.rng), batch.x)
#            index = self.indexer(next(self.rng))
##            output, _ = self.model.apply(combined_params, self,state, next(self.rng), batch.x, index=index)
#            output, _ = self.model.apply(
#                params=combined_params, 
#                state=self.state,
#                inputs=batch.x, 
#                index=index
#            )
#            logits = output.extra['classification_logits']
#            acc = float(self.compute_accuracy(logits, batch.y))
#            losses.append(float(loss))
#            accuracies.append(acc)
        run_end = time.time()
        print(f"🕒 Run {i+1} duration: {run_end - run_start:.2f} seconds")    
        return losses, accuracies



if __name__ == "__main__":
    start_time = time.time()
    print("\n\nFine tuning of BERT ENN model for SST2 classification\n\n")
    parser = argparse.ArgumentParser(description="Fine-tune BERT on SST2")
    parser.add_argument('--train_all', action='store_true', help='If set, trains all model parameters.')
    parser.add_argument('--test', action='store_true', help='If set, runs a faster test on 50 training steps and 3 number of repetitions')
    parser.add_argument('--suffix', type=str, default="", help="Optional suffix for output file names.")
    parser.add_argument('--save_params', action='store_true', help='If set, saves the final trained parameters.')
    parser.add_argument('--load_params', action='store_true', help='If set, loads previously saved parameters.')
    parser.add_argument('--load_params_path', type=str, default=None, help='Path to loadmodel parameters.')
    parser.add_argument('--save_params_path', type=str, default=None, help='Path to save model parameters.')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate for optimizer.') 
    parser.add_argument('--priority', type=str, default='uniform', help='Priority criterion. Default: "uniform". Available: "uniform", "variance", "entropy", "margin", "bald"') 
    parser.add_argument('--NB', type=int, default=40, help='Candidate pool size.') 
    parser.add_argument('--nb', type=int, default=4, help='Number of samples selected for training.')
    parser.add_argument('--num_runs', type=int, default=2, help='Number of seeds to run')
    args = parser.parse_args()

    SEED = args.seed if hasattr(args, 'seed') else 0
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
    
    full_dataset = dataset
    split = int(0.9*len(full_dataset))
    train_dataset = full_dataset.select(range(split))
    val_dataset = full_dataset.select(range(split, len(full_dataset)))
    print("...done")


    config = {
        'batch_size': 32,
        'num_steps': 1000,
        'epochs': 1,
        'learning_rate': args.learning_rate,
        'seed': rng,
        'num_runs': args.num_runs,
    }

    if args.test:
        config['num_steps'] = 50
        config['num_runs'] = 3

    config_dict = {
            "train_all": args.train_all,
            "test": args.test,
            "suffix": args.suffix,
            "save_params": args.save_params,
            "load_params": args.load_params,
            "load_params_path": args.load_params_path,
            "save_params_path": args.save_params_path,
            "learning_rate": args.learning_rate,
            "priority": args.priority,
            "batch_size": config['batch_size'],
            "NB": args.NB,
            "nb": args.nb,
            "num_steps": config['num_steps'],
            "num_runs": config['num_runs'],
            "seed": SEED,  
        }


    all_loss_df = pd.DataFrame()
    all_acc_df = pd.DataFrame()

    if args.load_params and args.load_params_path is not None:
        print(f"🔁 Loading pretrained parameters from: {args.load_params_path}")
        with np.load(args.load_params_path, allow_pickle=True) as f:
            loaded_params = f['params'].item()
        params = hk.data_structures.to_immutable_dict(loaded_params)
    else:
        print("🚀 Using default initialized parameters (with preloaded encoder weights)")

    
    suffix = f"_{args.suffix}" if args.suffix else ""
    # Create an output directory per job ID
    output_dir = f"outputs/job{suffix}"
    print(f"\n\nMaking directory {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Saving all outputs to {output_dir}")
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    print(f"Saved config to {config_path}")

    print("Running experiment...")
    for i in range(config['num_runs']):
        print(f"\n▶️ Run {i + 1}/{config['num_runs']}") 
        seed = SEED + i
        rng = hk.PRNGSequence(seed)

        experiment = BERTENNFineTuneSST2(
            model=base_model,
            dataset = train_dataset,
            val_dataset = val_dataset,
            tokenizer = tokenizer,
            rng = next(rng),
            batch_size = config['batch_size'],
            criterion = 'num_steps',
            num_steps = config['num_steps'],
            learning_rate = config['learning_rate'],
            train_all = args.train_all,
            pretrained_params=params,
            priority_criterion = args.priority,
        )
        jax.profiler.start_trace(f"/tmp/jax_trace_{suffix}")
        losses, acc = experiment.run()
        jax.profiler.stop_trace()

        all_loss_df[f'run_{i+1}'] = losses
        all_acc_df[f'run_{i+1}'] = acc

        if args.save_params and args.save_params_path is not None:
            print(f"🔁 Saving pretrained parameters to: {args.save_params_path}")
            final_combined_params = hk.data_structures.merge(experiment.frozen_params, experiment.trainable_params)
            np.savez(args.save_params_path, params=final_combined_params)

#        if args.save_params:
#            print(f"💾 Saving final trained parameters to: {args.params_path}")
#            final_combined_params = hk.data_structures.merge(experiment.frozen_params, experiment.trainable_params)
#            np.savez(args.params_path, params=final_combined_params)

    losses_file_name = f"{output_dir}/losses{suffix}.csv"
    accs_file_name = f"{output_dir}/accs{suffix}.csv"
    img_file_name = f"{output_dir}/img{suffix}.png"
  
    all_loss_df.to_csv(losses_file_name, index_label="step")
    all_acc_df.to_csv(accs_file_name, index_label="step")

   # plot_loss_and_accuracy_curve(losses_file_name, accs_file_name, img_file_name)
    #plot_all_runs_with_mean(losses_file_name, img_file_name) 
    save_results_to_file(losses, acc)
    print(f"DEBUG: params after learning: \n{experiment.trainable_params['BERT/classifier_head']['w']}")
    end_time = time.time()
    elapsed = end_time - start_time
    mins, secs = divmod(int(elapsed), 60)
    hours, mins = divmod(mins, 60)
    
    print(f"\n⏱️ Total run time: {hours}h {mins}m {secs}s")
