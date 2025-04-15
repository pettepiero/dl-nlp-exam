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
from typing import NamedTuple
import joblib
from tqdm import tqdm
import requests
from acme.utils import loggers
import chex
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



# BERT base fine tune on SST2 
class BERTFineTuneSST2:

    def __init__(
        self,
        bert,
        learning_rate=1e-5,
        batch_size=16,
        learning_batch_size: int = 4,
        num_steps: int = 100,
        seed=0,
    ):
        self.tokenizer = huggingface_tokenizer
        self.dataset = load_dataset("nyu-mll/glue", "sst2")
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.learning_batch_size = learning_batch_size
        self.rng = hk.PRNGSequence(seed)
        self.num_steps = num_steps
        self.replay = []

        self.bert=bert

        # Take a real sentence from the dataset
        example = self.dataset["train"][0]["sentence"]

        # Tokenize with huggingface tokenizer
        tokenized = self.tokenizer(
            example,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="np",
        )
        # Handle missing segment_ids (token_type_ids)
        if "token_type_ids" not in tokenized:
            tokenized["token_type_ids"] = np.zeros_like(tokenized["input_ids"])

        # Build BertInput
#        dummy_input = BertInput(
#            token_ids=tokenized["input_ids"],
#            segment_ids=tokenized["token_type_ids"],
#            input_mask=tokenized["attention_mask"],
#        )

        self.params, self.state = self.network.init(
            next(self.rng), dummy_input 
         )
            
        self.trainable_params, self.non_trainable_params = hk.data_structures.partition(
            lambda m, n, p: m=='BERT/classifier_head', self.params
        )

#        print("DEBUG: --")
        self.opt_state = optax.adam(learning_rate).init(self.params)
        entropy_priority_ctor = priorities.get_priority_fn_ctor(priority_fn)

#        print(f"DEBUG: created priority function constructor")

        # Prepare ENN forwarder and priority function
        self.network_batch_fwd = forwarders.make_batch_fwd(
            self.network, num_enn_samples=self.num_enn_samples
        )

#        print(f"DEBUG: prepared ENN forwarder")

        self.priority_fn = entropy_priority_ctor(self.network_batch_fwd)

#        print(f"DEBUG: Finished ActiveLearningSST2.__init__()")
        def make_mask(params):
            """ Created to freeze BERT base parameters and train only
                the classification head """
            def mask_fn(path, _):
                return path[0] == 'BERT/classifier_head'
        
            return jax.tree_util.tree_map_with_path(mask_fn, params)
        self.mask = make_mask(self.params)
        self.tx = optax.masked(optax.adam(self.learning_rate), self.mask)
        self.opt_state = self.tx.init(self.params)


    def tokenize(self, texts):
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="np",
        )
        # Handle missing segment_ids (token_type_ids)
        if "token_type_ids" not in tokens:
            tokens["token_type_ids"] = np.zeros_like(tokens["input_ids"])

        return tokens

    def loss_fn(self, trainable_params, non_trainable_params, state, batch):
        params = hk.data_structures.merge(trainable_params, non_trainable_params)

        # Averages the loss over epistemic indices
        def single_index_loss(index):
            net_out, new_state = self.network.apply(params, state, batch["x"], index)
            logits = net_out.train
            labels = jax.nn.one_hot(batch["y"], 2)
#            print(f"DEBUG: in loss_fn: logits.shape = {logits.shape}")
#            print(f"DEBUG: in loss_fn: labels.shape = {labels.shape}")
            loss = optax.softmax_cross_entropy(logits, labels).mean()
            return loss

        indices = [
            self.network.indexer(next(self.rng)) for _ in range(self.num_enn_samples)
        ]
        loss_values = jax.vmap(single_index_loss)(jnp.array(indices))
        mean_loss = jnp.mean(loss_values)
        return mean_loss, state


    def update(self, batch):
        # 5: compute minibatch gradient only for trainable_params
        
        grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
        (loss, state), grads = grad_fn(self.params, self.non_trainable_params, self.state, batch)
        updates, self.opt_state = self.tx.update(grads, self.opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)
        #updates, self.opt_state = optax.adam(self.learning_rate).update(
        #    grads, self.opt_state
        #    grads, self.opt_state, self.trainable_params
        #)
        # 6: update only trainable parameters
        #self.trainable_params = optax.apply_updates(self.trainable_params, updates)
        self.state = state
        return loss

    def select_uncertain_samples(self, pool, num_samples=10):
        input_ids, _ = self.tokenize(pool["sentence"])
        batch = datasets.ArrayBatch(x=input_ids, y=np.array(pool["label"]))
        scores, _ = self.priority_fn(self.params, self.state, batch, next(self.rng))
        selected_indices = jnp.argsort(scores)[-num_samples:]
        return [pool[i] for i in selected_indices]

    def fine_tune_base(self):
        """ Fine tune the base BERT model for SST-2 classification without epinet
            and without active learning."""
        
        train_labels = jnp.array(self.dataset['train']['label'])
        num_samples = len(train_labels)
        all_indices = jnp.arange(num_samples)
        losses = []

        for step in tqdm(range(self.num_steps)):
            batch_indices = jax.random.choice(
                key=next(self.rng),
                a=all_indices,
                shape=(self.batch_size,),
                replace=False
            )
            
            batch_sentences = [self.dataset['train'][int(i)]['sentence'] for i in batch_indices]
            batch_input = self.tokenize(batch_sentences)
            
            input = BertInput(
                token_ids=batch_input['input_ids'],
                segment_ids=batch_input['token_type_ids'],
                input_mask=batch_input['attention_mask'],
            )
            batch_labels = train_labels[batch_indices]
            train_batch = datasets.ArrayBatch(x=input, y=batch_labels)
            
            loss = self.update(train_batch)
            losses.append(loss)

        return losses        

    def run(self):
        # Numbered comments are based on algorithm 1 of Fine tuning paper.
        # Pre-load all data into JAX-compatible arrays
        train_labels = jnp.array(self.dataset["train"]["label"])
        num_samples = len(train_labels)
        all_indices = jnp.arange(num_samples)

#        print(f"DEBUG: setup of loop done")
        losses = []

        for step in tqdm(range(self.num_steps)):
            # 2: Sample batch_size candidate indices without replacement
            cand_indices = jax.random.choice(
                key=next(self.rng),
                a=all_indices,
                shape=(self.batch_size,),
                replace=False,
            )

            cand_indices_list = [int(i) for i in cand_indices]

#            print(f"DEBUG: sampled candidate indices")
#            print(f"DEBUG: candidate indices: {cand_indices_list}")
            # Get batch directly using JAX array indexing
            batch_sentences = [
                self.dataset["train"][i]["sentence"] for i in cand_indices_list
            ]
#            print(f"DEBUG: candidate batch_sentences: \n{batch_sentences}")
            cand_input_ids = self.tokenize(batch_sentences)
#            print(f"DEBUG: cand_input_ids = \n{cand_input_ids}")
#            print(f"DEBUG: Tokenization done")

            input = BertInput(
                token_ids=cand_input_ids["input_ids"],
                segment_ids=cand_input_ids["token_type_ids"],
                input_mask=cand_input_ids["attention_mask"],
            )

            # Create ArrayBatch
            cand_batch = datasets.ArrayBatch(x=input, y=train_labels[cand_indices])

#            print(f"DEBUG: Created ArrayBatch")
#            print(f"\n\nDEBUG: candidate batch dimensions:")
#            print(f"DEBUG: cand_batch.x = \n{cand_batch.x}")
            print(
#                f"DEBUG: cand_batch.x.token_ids.shape = \n{cand_batch.x.token_ids.shape}"
            )

            # 3: select the learning_batch_size indices with highest priority
            batch_priorities, _ = self.priority_fn(
                self.params,
                self.state,
                cand_batch,
                next(self.rng),
            )

#            print(f"DEBUG: Got batch priorities")

            # Select top-k highest priority samples
            top_k_indices = jnp.argsort(-batch_priorities)[: self.learning_batch_size]
#            print(f"\n\nDEBUG: top_k_indices = \n{top_k_indices}")
            selected_input = BertInput(
                token_ids=cand_input_ids["input_ids"][top_k_indices],
                segment_ids=cand_input_ids["token_type_ids"][top_k_indices],
                input_mask=cand_input_ids["attention_mask"][top_k_indices],
            )
            selected_labels = train_labels[cand_indices][top_k_indices]

            train_batch = datasets.ArrayBatch(x=selected_input, y=selected_labels)
            loss = self.update(train_batch)
            losses.append(loss)
            #print(f"Step {step}: Loss = {loss:.4f}")
        
        return losses


def _clean_results(results: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    """Cleans the results for logging (can't log jax arrays)."""

    def clean_result(value: tp.Any) -> tp.Any:
        value = loggers.to_numpy(value)
        if isinstance(value, chex.ArrayNumpy) and value.size == 1:
            value = float(value)
        return value

    for key, value in results.items():
        results[key] = clean_result(value)

    return results


def plot_loss_curve(losses: list, save_path: str):
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
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Training Loss", color="blue")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Loss curve saved to {save_path}")


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
    text: str, tokenizer: BertTokenizer, max_length: int = 512
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

if __name__ == "__main__":
    print("Creating base model....")
    base_model = make_bert_base()
    print("..done")

    def model_fn(inputs: BertInput, is_training: bool):
        cls_output = bert_forward(inputs.token_ids, inputs.segment_ids, inputs.input_mask)
    
        return classifier_head(cls_output, is_training)

