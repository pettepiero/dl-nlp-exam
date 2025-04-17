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
import functools
from typing import NamedTuple, Optional
import joblib
from tqdm import tqdm
import requests
from acme.utils import loggers
import chex
import pandas as pd
#from enn import datasets
#from enn.networks import forwarders
#from enn.networks.bert.bert_classification_enn_v2 import (
#    create_enn_bert_for_classification_ft,
#    make_bert_base
#)
#from enn.networks.bert.base import BertInput, BertEnn
import haiku as hk
import numpy as np
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from neural_testbed import base as testbed_base
from neural_testbed import generative
from neural_testbed import likelihood

import optax
from io import BytesIO
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel
from datasets import load_dataset
from types import SimpleNamespace
from enn.networks.bert.bert_v2 import BertConfigCustom, make_bert_base
# Define the tokenizer from Huggingface
huggingface_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
huggingface_roberta = BertModel.from_pretrained(
    "bert-base-uncased", output_hidden_states=True, return_dict=False
)

########################################################################
# Functions to load the pretrained weights


# Some light postprocessing to make the parameter keys a bit more concise
def postprocess_key(key):
    return (
        key.replace("model/featurizer/bert/", "").replace(":0", "").replace("self/", "")
    )

class BertInput(NamedTuple):
    token_ids: jnp.ndarray
    segment_ids: jnp.ndarray
    input_mask: jnp.ndarray

class ArrayBatch(NamedTuple):
    x: BertInput
    y: jnp.ndarray

#def update_step(params, frozen_params, opt_state, batch, rng_key, apply_fn, loss_fn, optimizer):
def update_step(params, frozen_params, opt_state, batch, rng_key, apply_fn, optimizer):
    # Combine frozen BERT with trainable classifier
    combined_params = hk.data_structures.merge(frozen_params, params)
    
    def loss_fn_inner(trainable_params):
        combined = hk.data_structures.merge(frozen_params, trainable_params)
        #logits, _ = apply_fn(combined, rng_key, batch.x)
        output = apply_fn(combined, rng_key, batch.x)
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
    loss = loss_fn_inner(new_params)
    return new_params, new_opt_state, loss

#update_step = jax.jit(update_step, static_argnames=["apply_fn", "loss_fn", "optimizer"])
update_step = jax.jit(update_step, static_argnames=["apply_fn", "optimizer"])


# DEBUG
def test_on_logits(tokenize_fn, apply_fn, params, rng):
    # Example text
    sentence = "I love this movie!"
    
    # Tokenize using HuggingFace
    tok = tokenize_fn(sentence)

    input = BertInput(
        token_ids=jnp.array(tok["input_ids"]),
        segment_ids=jnp.array(tok["token_type_ids"]),
        input_mask=jnp.array(tok["attention_mask"]),
    )
    label = 1 

    batch = ArrayBatch(x=input, y=label)
    #self.trainable_params, self.opt_state, loss = self.update(
    #    self.trainable_params, self.opt_state, batch
    #)
    print(f"DEBUG: batch.x.token_ids.shape = {batch.x.token_ids.shape}")    

    # Forward pass
    output = apply_fn(params, rng, batch.x)
    logits = output.extra['classification_logits']
    probs = jax.nn.softmax(logits, axis=-1)
    preds = jnp.argmax(probs, axis=-1)
 
    print("DEBUG: test_on_logits")    
    print("Logits:", logits)
    print("Shape:", logits.shape)  # Should be (1, num_classes), e.g., (1, 2) for SST-2
    print("probs:", probs)
    print("preds:", preds)


    print("############################################\n\n")


# BERT base fine tune on SST2 
class BERTFineTuneSST2:
    def __init__(
        self,
        model,
        dataset,
        tokenizer,
        rng,
        batch_size,
        criterion: str,
        num_steps: Optional[int],
        learning_rate
    ):
        self.model = model 
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.rng = hk.PRNGSequence(rng)
        self.batch_size = batch_size
        self.num_steps = num_steps

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

        # Init model
        dummy_input = BertInput(
            token_ids=jnp.zeros((1, 128), dtype=jnp.int32),
            segment_ids=jnp.zeros((1, 128), dtype=jnp.int32),
            input_mask=jnp.ones((1, 128), dtype=jnp.int32)
        )
        #self.params, _ = model.init(next(self.rng), dummy_input)
        self.params = self.model.init(next(self.rng), dummy_input)
        self.params = hk.data_structures.to_immutable_dict(self.params)
        self.labels = jnp.array([ex['label'] for ex in self.dataset])

        def print_tree(params, prefix=""):
            for k, v in params.items():
                if isinstance(v, dict):
                    print_tree(v, prefix + k + "/")
                else:
                    print(prefix + k)
        
       # print_tree(self.params)

#        def is_trainable(module_name, name, value):
#            return (
#                "classifier_head" in module_name
#                or "pooler_dense" in module_name
#            )
        def is_trainable(module_name, name, value):
            return True
           # return (
           #     "classifier_head" in module_name
           #     or "pooler_dense" in module_name
           #     or "attention_output_11" in module_name
           #     or "intermediate_output_11" in module_name
           #     or "query_11" in module_name
           #     or "values_11" in module_name
           #     or "layer_output_11" in module_name
           #     or "attention_output_10" in module_name
           #     or "intermediate_output_10" in module_name
           #     or "query_10" in module_name
           #     or "values_10" in module_name
           #     or "layer_output_10" in module_name
           # )
       # self.trainable_params = hk.data_structures.filter(
       #     lambda m, n, p: 'classifier_head' in m, self.params
       # )
       # self.frozen_params = hk.data_structures.filter(
       #     lambda m, n, p: "classifier_head" not in m, self.params
       # )        
        
        self.frozen_params, self.trainable_params = hk.data_structures.partition(
            lambda mod, name, val: not is_trainable(mod, name, val),
            self.params
        )

        print("Trainable params:")
        for k in self.trainable_params.keys():
            print(k)
        
        print("\nFrozen params:")
        for k in self.frozen_params.keys():
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

    #def loss_fn(self, params, batch: ArrayBatch):
    #    
#   #     logits, _ = self.model.apply(
    #    logits = self.model.apply(
    #        {**self.params, **params}, # combine frozen + trainable params
    #        next(self.rng),
    #        batch.x,
    #        is_training=True,
    #    )
    #    #labels = jax.nn.one_hot(batch.y, 2)
    #    labels = batch.y.astype(jnp.int32)
    #    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


#    @jax.jit
#    def update(self, params, opt_state, batch):
#        grads = jax.grad(self.loss_fn)(params, batch)
#        updates, new_opt_state = self.opt.update(grads, opt_state)
#        new_params = optax.applu_updates(params, updates)
#        loss = self.loss_fn(new_params, batch)
#    
#        return new_params, new_opt_state, loss
#    def get_accuracy(self, rng):
#        n_samples = 50
#        test_labels = jnp.array(self.dataset['test'][:n_samples]['label']
#        indices = jnp.arange(n_samples)
#        test_sentences = self.dataset['test'][:n_samples]['sentence']
#        tok = self.tokenize(batch_sentences)
#        input = BertInput(
#            token_ids=jnp.array(tok['input_ids']),
#            segment_ids=jnp.array(tok['token_type_ids']),
#            input_mask=jnp.array(tok['attention_mask']),
#        )
#        batch = ArrayBatch(x=input, y=batch_labels)
#
#        predictions = self.model.apply(
#            params, rng, 

    def compute_accuracy(self, logits: jnp.ndarray, labels: jnp.ndarray) -> float:
        predictions = jnp.argmax(logits, axis=-1)
        return jnp.mean(predictions == labels)

    def run(self):
        train_labels = jnp.array([item['label'] for item in self.dataset])
        all_indices = jnp.arange(len(train_labels))
        
        losses = []
        accuracies = []
    
        for step in tqdm(range(self.num_steps)):
        #    batch_indices = jax.random.choice(
        #        key=next(self.rng),
        #        a=all_indices,
        #        shape=(self.batch_size,),
        #        replace=True,
        #    )
        #    
        #    batch_sentences = [
        #        self.dataset["train"][int(i)]["sentence"] for i in batch_indices 
        #    ]
        #    tokens = self.tokenize(batch_sentences)
        #    token_ids = jnp.array(tokens["input_ids"])
        #    segment_ids = jnp.array(tokens["token_type_ids"])
        #    input_mask = jnp.array(tokens["attention_mask"])

        #    if token_ids.ndim == 1:
        #        token_ids = token_ids[None, :]
        #    if segment_ids.ndim == 1:
        #        segment_ids = segment_ids[None, :]
        #    if input_mask.ndim == 1:
        #        input_mask = input_mask[None, :]

        #    input = BertInput(
        #        token_ids=token_ids,
        #        segment_ids=segment_ids,
        #        input_mask=input_mask,
        #    )
        #    batch_labels = train_labels[batch_indices]
        #    batch = ArrayBatch(x=input, y=batch_labels)

            ############################################
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
            self.trainable_params, self.opt_state, loss = update_step(
                self.trainable_params,
                frozen_params=self.frozen_params,
                opt_state=self.opt_state,
                batch=batch,
                rng_key=next(self.rng),
                apply_fn=base_model.apply,
                optimizer=self.opt
            )
    
            # Compute accuracy
            combined_params = hk.data_structures.merge(self.frozen_params, self.trainable_params)
            output = base_model.apply(combined_params, next(self.rng), batch.x)
            logits = output.extra['classification_logits']
            acc = float(self.compute_accuracy(logits, batch.y))
    
            losses.append(float(loss))
            accuracies.append(acc)
    
        return losses, accuracies


def plot_loss_curve(file: str, save_path: str):
    """
    Plot the losses and save to image
    """
    available_priorities = ["BERT-base", "uniform", "entropy", "margin", "bald", "variance"]
    matched_priority = next((p for p in available_priorities if p in save_path), None)
    if matched_priority:
        title = f"Training Loss over steps using {matched_priority}"
    else:
        print(
            "Error in plot_loss_curve, save_path must contain"
            "one of the available priority functions"
        )
        return
    df = pd.read_csv(file)
    plt.figure(figsize=(8, 5))
    plt.xlabel("Step")
    plt.ylabel("Loss")
    df.mean(axis=1).plot(title=title)
#    plt.title(title)
#    plt.grid(True)
#    plt.tight_layout()
    plt.savefig(save_path)
#    plt.show()
#    plt.close()
    print(f"✅ Loss curve saved to {save_path}")



def plot_loss_and_accuracy_curve(file: str, save_path: str):
    """
    Plot the training loss and accuracy curves from a CSV file.
    """
    available_priorities = ["BERT-base", "uniform", "entropy", "margin", "bald", "variance"]
    matched_priority = next((p for p in available_priorities if p in save_path), None)
    
    if matched_priority:
        title = f"Training Metrics over Steps using {matched_priority}"
    else:
        print(
            "Error in plot_loss_and_accuracy_curve: save_path must contain "
            "one of the available priority functions"
        )
        return

    df = pd.read_csv(file)

    if not {"loss", "accuracy"}.issubset(df.columns):
        print("Error: CSV file must contain 'loss' and 'accuracy' columns.")
        return

    plt.figure(figsize=(10, 6))
    
    # Plot loss
    plt.plot(df["loss"], label="Loss")
    
    # Plot accuracy
    plt.plot(df["accuracy"], label="Accuracy")
    
    plt.xlabel("Step")
    plt.ylabel("Metric Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Loss and accuracy curve saved to {save_path}")

def save_loss_to_file(losses: list, save_path: str):
    """
    Save the list of losses to a file.

    Args:
    losses (list): List of loss values.
    save_path (str): Path to the file where losses should be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(f'./{save_path}'), exist_ok=True)

    # Write losses to the file
    with open(save_path, "w") as f:
        for loss in losses:
            f.write(f"{loss}\n")

    print(f"Losses saved to {save_path}")


def tokenize_input(
    text: str, tokenizer: BertTokenizer, max_length: int = 128 
) -> BertInput:
    """Tokenizes input text and converts it into BertInput format."""
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",  # Convert to NumPy arrays
    )

    return BertInput(
        token_ids=encoding["input_ids"],
        segment_ids=encoding["token_type_ids"],
        input_mask=encoding["attention_mask"],
    )


def preprocess_dataset(dataset, tokenizer, max_length=128):
    def tokenize(example):
        encoding = tokenizer(
            example['sentence'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='np'
        )
        return {
            'input_ids': encoding['input_ids'][0],
            'token_type_ids': encoding['token_type_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'label': example['label']
        }

    return dataset.map(tokenize)

if __name__ == "__main__":
    print("\n\nFine tuning of BERT base model for SST2 classification\n\n")
    print("Creating base model....")
    base_model = make_bert_base()
    print("..done")
   
    print("Loading SST2 dataset...")
    dataset = load_dataset(
        "nyu-mll/glue", 
        "sst2",
    )
    print("...done") 
    print(dataset)
 
    print("Loading Hugginface's tokenizer 🤗...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print("...done")
    
    print("Pretokenizing the dataset...")
    dataset = dataset.map(lambda x: {'sentence': x['sentence'], 'label': x['label']})
    dataset = preprocess_dataset(dataset['train'], tokenizer)
    print("...done")

    SEED = 0
    rng = hk.PRNGSequence(SEED)

    config = {
        'batch_size': 32,
        'num_steps': 1000,
#        'num_steps': 300,
        'epochs': 1,
        'learning_rate': 3e-5,
        'seed': rng
    }
    
     
    print("Creating experiment...")
    experiment = BERTFineTuneSST2(
        model=base_model,
        dataset = dataset,
        tokenizer = tokenizer,
        rng = next(config['seed']),
        batch_size = config['batch_size'],
        criterion = 'num_steps',
        num_steps = config['num_steps'],
        learning_rate = config['learning_rate']
    )
    print("...done")
    
    all_losses = []
    print("Running experiment...")
    # Test on one run only
    losses, acc = experiment.run()
    df = pd.DataFrame({
        "loss": losses,
        "accuracy": acc
    })
    df.to_csv("single_losses_test.csv", index_label="step")
   
    plot_loss_and_accuracy_curve("single_losses_test.csv", 'BERT-base')
    save_loss_to_file(losses, 'BERT-base')
    print(f"DEBUG: params after learning: \n{experiment.trainable_params['BERT/classifier_head']['w']}")
