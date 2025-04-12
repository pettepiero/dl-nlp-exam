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
"""Active learning evaluation of ENN agent on SST-2 by Piero"""
import os
import matplotlib.pyplot as plt
import functools
import typing as tp
from tqdm import tqdm
import joblib
import requests
from acme.utils import loggers
import chex
from enn import datasets
from enn.networks import forwarders
from enn.networks.bert.bert_classification_enn import (
    create_enn_bert_for_classification_ft,
)
from enn.networks.bert.base import BertInput, BertEnn
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

from bert_enn import Embedding, TransformerBlock
from enn.active_learning import priorities, prioritized
from enn.networks.forwarders import make_batch_fwd
from types import SimpleNamespace

# Define the tokenizer from Huggingface
huggingface_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
huggingface_roberta = BertModel.from_pretrained(
    "bert-base-uncased", output_hidden_states=True, return_dict=False
)
# huggingface_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# huggingface_roberta = RobertaModel.from_pretrained(
# "roberta-base", output_hidden_states=True, return_dict=False
# )

########################################################################
# Define the transformer model
class RobertaFeaturizer(hk.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(name="Transformer")
        self.config = config

    def __call__(self, token_ids, training=False):
        x = Embedding(self.config)(token_ids, training=training)
        mask = (token_ids == self.config["mask_id"]).astype(jnp.float32)
        for layer_num, layer in enumerate(range(self.config["n_layers"])):
            x = TransformerBlock(self.config, layer_num=layer_num)(
                x, mask, training=training
            )
        return x

########################################################################
# Functions to load the pretrained weights

# Some light postprocessing to make the parameter keys a bit more concise
def postprocess_key(key):
    return (
        key.replace("model/featurizer/bert/", "").replace(":0", "").replace("self/", "")
    )

# Cache the downloaded file to go easy on the tubes
@functools.lru_cache()
def get_pretrained_weights():
    # We'll use the weight dictionary from the Roberta encoder at https://github.com/IndicoDataSolutions/finetune
    remote_url = "https://bendropbox.s3.amazonaws.com/roberta/roberta-model-sm-v2.jl"
    weights = joblib.load(BytesIO(requests.get(remote_url).content))

    weights = {postprocess_key(key): value for key, value in weights.items()}

    # We use huggingface's word embedding matrix because their token IDs mapping varies slightly from the
    # format used in our joblib file above
    weights["embeddings/word_embeddings"] = (
        huggingface_roberta.get_input_embeddings().weight.detach().numpy()
    )
    return weights


# Active Learning Experiment
class ActiveLearningSST2:

    def __init__(
        self,
        enn: BertEnn,
        # model_fn,
        priority_fn: str = "uniform",
        learning_rate=1e-3,
        batch_size=40,
        learning_batch_size: int = 4,
        num_enn_samples: int = 10,
        num_steps: int = 100,
        seed=0,
    ):
        self.tokenizer = huggingface_tokenizer
        self.dataset = load_dataset("nyu-mll/glue", "sst2")
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.learning_batch_size = learning_batch_size
        self.rng = hk.PRNGSequence(seed)
        # self.model_fn = hk.transform(model_fn)
        self.num_enn_samples = num_enn_samples
        self.num_steps = num_steps
        self.replay = []

        # Initialize ENN
        self.enn = enn
        # dummy_input = jnp.ones([1, 512], dtype=jnp.int32)

        # Take a real sentence from the dataset
        example = self.dataset["train"][0]["sentence"]

        # Tokenize with huggingface tokenizer
        tokenized = self.tokenizer(
            example,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="np"
        )
        # Handle missing segment_ids (token_type_ids)
        if "token_type_ids" not in tokenized:
            tokenized["token_type_ids"] = np.zeros_like(tokenized["input_ids"])

        # Build BertInput
        dummy_input = BertInput(
            token_ids=tokenized["input_ids"],
            segment_ids=tokenized["token_type_ids"],
            input_mask=tokenized["attention_mask"],
        )

#        print("DEBUG: calling self.enn.init(...)")
        self.params, self.state = self.enn.init(
            next(self.rng), dummy_input, self.enn.indexer(next(self.rng))
        )
#        print("DEBUG: --")
        self.opt_state = optax.adam(learning_rate).init(self.params)
        entropy_priority_ctor = priorities.get_priority_fn_ctor(priority_fn)

#        print(f"DEBUG: created priority function constructor")

        # Prepare ENN forwarder and priority function
        self.enn_batch_fwd = forwarders.make_batch_fwd(
            self.enn, num_enn_samples=self.num_enn_samples
        )

#        print(f"DEBUG: prepared ENN forwarder")

        self.priority_fn = entropy_priority_ctor(self.enn_batch_fwd)

#        print(f"DEBUG: Finished ActiveLearningSST2.__init__()")

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

    def loss_fn(self, params, state, batch):
        # Averages the loss over epistemic indices
        def single_index_loss(index):
            net_out, new_state = self.enn.apply(params, state, batch["x"], index)
            logits = net_out.train
            labels = jax.nn.one_hot(batch["y"], 2)
#            print(f"DEBUG: in loss_fn: logits.shape = {logits.shape}")
#            print(f"DEBUG: in loss_fn: labels.shape = {labels.shape}")
            loss = optax.softmax_cross_entropy(logits, labels).mean()
            return loss

        indices = [self.enn.indexer(next(self.rng)) for _ in range(self.num_enn_samples)]
        loss_values = jax.vmap(single_index_loss)(jnp.array(indices))
        mean_loss = jnp.mean(loss_values)
        return mean_loss, state

    def update(self, batch):
        # 5: compute minibatch gradient

        grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
        (loss, state), grads = grad_fn(self.params, self.state, batch)
        updates, self.opt_state = optax.adam(self.learning_rate).update(
            grads, self.opt_state
        )
        # 6: update parameters
        self.params = optax.apply_updates(self.params, updates)
        self.state = state
        return loss

    def select_uncertain_samples(self, pool, num_samples=10):
        input_ids, _ = self.tokenize(pool["sentence"])
        batch = datasets.ArrayBatch(x=input_ids, y=np.array(pool["label"]))
        scores, _ = self.priority_fn(self.params, self.state, batch, next(self.rng))
        selected_indices = jnp.argsort(scores)[-num_samples:]
        return [pool[i] for i in selected_indices]

    def run(self) -> list:
        # Numbered comments are based on algorithm 1 of Fine tuning paper.
        # Pre-load all data into JAX-compatible arrays
        train_labels = jnp.array(self.dataset['train']['label'])
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
                replace=False
            )

            cand_indices_list = [int(i) for i in cand_indices]

#            print(f"DEBUG: sampled candidate indices")
#            print(f"DEBUG: candidate indices: {cand_indices_list}")
            # Get batch directly using JAX array indexing
            batch_sentences = [self.dataset['train'][i]['sentence'] for i in cand_indices_list]
#            print(f"DEBUG: candidate batch_sentences: \n{batch_sentences}")
            cand_input_ids = self.tokenize(batch_sentences)
#            print(f"DEBUG: cand_input_ids = \n{cand_input_ids}")
#            print(f"DEBUG: Tokenization done")

            input = BertInput(
                token_ids=cand_input_ids['input_ids'],
                segment_ids=cand_input_ids['token_type_ids'],
                input_mask=cand_input_ids['attention_mask'],
            )

            # Create ArrayBatch
            cand_batch = datasets.ArrayBatch(
                x=input, 
                y=train_labels[cand_indices]
            )

#            print(f"DEBUG: Created ArrayBatch")
#            print(f"\n\nDEBUG: candidate batch dimensions:")
#            print(f"DEBUG: cand_batch.x = \n{cand_batch.x}")
#            print(f"DEBUG: cand_batch.x.token_ids.shape = \n{cand_batch.x.token_ids.shape}")

            # 3: select the learning_batch_size indices with highest priority
            batch_priorities, _ = self.priority_fn(
                self.params,
                self.state,
                cand_batch,
                next(self.rng),
            )

#            print(f"DEBUG: Got batch priorities")

            # Select top-k highest priority samples
            top_k_indices = jnp.argsort(-batch_priorities)[:self.learning_batch_size]
#            print(f"\n\nDEBUG: top_k_indices = \n{top_k_indices}")
            selected_input = BertInput(
                token_ids=cand_input_ids["input_ids"][top_k_indices],
                segment_ids=cand_input_ids["token_type_ids"][top_k_indices],
                input_mask=cand_input_ids["attention_mask"][top_k_indices],
            )
            selected_labels = train_labels[cand_indices][top_k_indices]

            train_batch = datasets.ArrayBatch(
                x=selected_input, 
                y=selected_labels
            )
            loss = self.update(train_batch)
            losses.append(loss)
        
        return losses

def plot_loss_curve(losses: list, save_path: str):
    """
        Plot the losses and save to image
    """
    available_priorities = [
        'uniform',
        'entropy',
        'margin',
        'bald',
        'variance'
    ]
    matched_priority = next((p for p in available_priorities if p in save_path), None)
    if matched_priority:
        title = f"Training Loss over steps using {matched_priority}"    
    else:
        print("Error in plot_loss_curve, save_path must contain" 
            "one of the available priority functions")
        return
    plt.figure(figsize=(8,5))
    plt.plot(losses, label='Training Loss', color='blue')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim((0, 3))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_loss_to_file(losses: list, save_path: str):
    """
    Save the list of losses to a file.

    Args:
    losses (list): List of loss values.
    save_path (str): Path to the file where losses should be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Write losses to the file
    with open(save_path, 'w') as f:
        for loss in losses:
            f.write(f"{loss}\n")


if __name__ == "__main__":
    print(f"JAX devices list: ", jax.devices())
    print(f"JAX devices:", jax.devices()[0].device_kind)

    print("Creating ENN BERT for classification...")

    print("Creating ENN BERT for classification...")

    enn_model, haiku_params, haiku_state = create_enn_bert_for_classification_ft()

    print("... done")

    priorities_to_test = [
        "uniform",
        "entropy",
        "margin",
        "bald",
        "variance",
    ]

    for priority in priorities_to_test:
        print(f"\nTraining using '{priority}' priority function")
        experiment = ActiveLearningSST2(
            enn=enn_model,
            priority_fn=priority,
        )
        losses = experiment.run()
        plot_loss_curve(losses, priority)
        save_loss_to_file(losses, f'./{priority}.txt')

