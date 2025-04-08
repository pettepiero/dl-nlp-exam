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

import functools
import typing as tp
import joblib
import requests
from acme.utils import loggers
import chex
from enn import base as enn_base
from enn import datasets
from enn import losses
from enn import networks
from enn.networks import forwarders
from enn.networks.bert.bert_classification_enn import (
    create_enn_bert_for_classification_ft,
)
from enn.networks.bert.base import BertInput
import haiku as hk
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from neural_testbed import agents
from neural_testbed import base as testbed_base
from neural_testbed import generative
from neural_testbed import likelihood
from neural_testbed.bandit import replay
from neural_testbed.leaderboard import sweep
import optax
from io import BytesIO
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel
from datasets import load_dataset

from bert_enn import Embedding, TransformerBlock
from enn import active_learning as al
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


########################################################################
# Create transformed model

def featurizer_fn(tokens, training=False):
    contextual_embeddings = RobertaFeaturizer(config)(tokens, training=training)
    return contextual_embeddings


########################################################################
# Active Learning Experiment
class ActiveLearningSST2:
    def __init__(
        self,
        enn_config: agents.VanillaEnnConfig,
        model_fn,
        learning_rate=1e-3,
        batch_size=32,
        seed=0,
    ):
        self.tokenizer = huggingface_tokenizer
        self.dataset = load_dataset("nyu-mll/glue", "sst2")
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.rng = hk.PRNGSequence(seed)
        self.model_fn = hk.transform(model_fn)
        self.replay = []

        # Initialize ENN
        self.enn = enn_config.enn_ctor()
        dummy_input = jnp.ones([1, 512], dtype=jnp.int32)
        self.params, self.state = self.enn.init(
            next(self.rng), dummy_input, self.enn.indexer(next(self.rng))
        )
        self.opt_state = optax.adam(learning_rate).init(self.params)

        # Prepare ENN forwarder and priority function
        self.enn_batch_fwd = forwarders.make_batch_fwd(self.enn, num_enn_samples=20)
        self.priority_fn = priorities.make_priority_fn_ctor(priorities.entropy_priority)(
            self.enn_batch_fwd
        )

    def tokenize(self, texts):
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="np"
        )
        return tokens["input_ids"], tokens["attention_mask"]

    def loss_fn(self, params, state, batch):
        index = self.enn.indexer(next(self.rng))
        net_out, state = self.enn.apply(params, state, batch["input_ids"], index)
        logits = net_out.output
        labels = jax.nn.one_hot(batch["labels"], 2)
        loss = optax.softmax_cross_entropy(logits, labels).mean()
        return loss, state

    def update(self, batch):
        grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
        (loss, state), grads = grad_fn(self.params, self.state, batch)
        updates, self.opt_state = optax.adam(self.learning_rate).update(
            grads, self.opt_state
        )
        self.params = optax.apply_updates(self.params, updates)
        self.state = state
        return loss

    def select_uncertain_samples(self, pool, num_samples=10):
        input_ids, _ = self.tokenize(pool["sentence"])
        batch = datasets.ArrayBatch(x=input_ids, y=np.array(pool["label"]))
        scores, _ = self.priority_fn(self.params, self.state, batch, next(self.rng))
        selected_indices = jnp.argsort(scores)[-num_samples:]
        return [pool[i] for i in selected_indices]

    def run(self, num_steps=100):
        for step in range(num_steps):
            # Select a batch with highest uncertainty from full training set
            raw_batch = self.select_uncertain_samples(
                self.dataset["train"], self.batch_size
            )

            # Tokenize and format batch
            input_ids, _ = self.tokenize([ex["sentence"] for ex in raw_batch])
            labels = jnp.array([ex["label"] for ex in raw_batch])
            batch = {"input_ids": input_ids, "labels": labels}

            # Train on selected batch
            loss = self.update(batch)
            print(f"Step {step}: Loss = {loss:.4f}")


def _make_test_problem(
    logit_fn: generative.LogitFn,
    prior: testbed_base.PriorKnowledge,
    input_dim: int,
    key: chex.PRNGKey,
    num_classes: int = 2,
) -> likelihood.SampleBasedTestbed:
    """Makes the test environment."""
    sampler_key, kl_key = jax.random.split(key)
    # Defining dummy values for x_train_generator and num_train. These values are
    # not used as we only use data_sampler to make test data.
    dummy_x_train_generator = generative.make_gaussian_sampler(input_dim)
    dummy_num_train = 10
    data_sampler = generative.ClassificationEnvLikelihood(
        logit_fn=logit_fn,
        x_train_generator=dummy_x_train_generator,  # UNUSED
        x_test_generator=generative.make_gaussian_sampler(input_dim),
        num_train=dummy_num_train,  # UNUSED
        key=sampler_key,
        tau=1,
    )
    sample_based_kl = likelihood.CategoricalKLSampledXSampledY(
        num_test_seeds=1000,
        num_enn_samples=1000,
        key=kl_key,
        num_classes=num_classes,
    )
    sample_based_kl = likelihood.add_classification_accuracy_ece(
        sample_based_kl,
        num_test_seeds=1000,
        num_enn_samples=100,
        num_classes=num_classes,
    )
    return likelihood.SampleBasedTestbed(data_sampler, sample_based_kl, prior)


def _random_argmax(vals: chex.Array, key: chex.PRNGKey, scale: float = 1e-5) -> int:
    """Selects argmax with additional random noise."""
    noise = jax.random.normal(key, vals.shape)
    return jnp.argmax(
        vals + scale * noise, axis=0
    )  # pytype: disable=bad-return-type  # jnp-type


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


if __name__ == "__main__":

    # 1. Load the SST-2 dataset (train pool)
    sst2 = load_dataset("glue", "sst2")
    train_texts = sst2['train']['sentence']
    train_labels = jnp.array(sst2['train']['label'])

    # Just take a small pool for now
    N = 256
    texts_pool = train_texts[:N]
    labels_pool = train_labels[:N]

    # 2. Create tokenizer and batch inputs
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def encode(texts):
        tok_out = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="np"
        )
        return tok_out["input_ids"]

    x_pool = encode(texts_pool)
    y_pool = labels_pool

    encoded = huggingface_tokenizer(
        texts_pool,
        padding="max_length",   
        truncation=True,
        max_length=128,
        return_tensors="np"  # return NumPy arrays to convert easily to jnp
    )

    bert_input = SimpleNamespace(
        token_ids=jnp.array(encoded["input_ids"]),
        segment_ids=jnp.array(encoded["token_type_ids"]),
        input_mask=jnp.array(encoded["attention_mask"]),
    )

    # Make batch
    batch = datasets.ArrayBatch(x=jnp.array(x_pool), y=jnp.array(y_pool))

    # 3. Create the ENN model
    enn_model, haiku_params, haiku_state = create_enn_bert_for_classification_ft()

    # Initialize model
    key = jax.random.PRNGKey(0)
    print(f"\n\nDEBUG: x_pool = {x_pool}\n\n")

    # params, state = enn_model.init(key, x_pool[:1], enn_model.indexer(key))
    params, state = enn_model.init(key, bert_input, enn_model.indexer(key))

    # 4. Create forwarder
    enn_batch_fwd = make_batch_fwd(enn_model, num_enn_samples=100, seed=42)

    # 5. Define acquisition strategy (e.g., predictive variance)
    per_example_priority = priorities.get_per_example_priority('variance')
    priority_fn_ctor = priorities.make_priority_fn_ctor(per_example_priority)

    # 6. Create PrioritizedBatcher
    batcher = prioritized.PrioritizedBatcher(
        enn_batch_fwd=enn_batch_fwd,
        acquisition_size=32,
        priority_fn_ctor=priority_fn_ctor
    )

    print(f"\n\nDEBUG: batch = \n{batch}\n\n")

    # 7. Run 1 acquisition step
    key = jax.random.PRNGKey(1)
    acquired_batch = batcher.sample_batch(params, state, batch, key)

    # 8. Print acquired data
    print("Acquired x shape:", acquired_batch.x.shape)
    print("Acquired y shape:", acquired_batch.y.shape)

    # returned = create_enn_bert_for_classification_ft()
    # print(type(returned))
    # enn_model, haiku_params, haiku_state = returned
    # print(f"Created enn_model: \n\n{enn_model}")

    # # Step 2: Create a forwarder
    # enn_fwd = forwarders.make_evaluation_forwarder(enn)

    # # Step 3: Choose acquisition function
    # priority_fn_ctor = make_priority_fn_ctor(priorities.predictive_entropy)

    # # Step 4: Create active learner
    # active_learner = PrioritizedBatcher(
    #     enn_batch_fwd=enn_fwd,
    #     acquisition_size=64,
    #     priority_fn_ctor=priority_fn_ctor
    # )

    # # Step 5: Assume you have a pool batch (e.g. from SST-2)
    # # Example dummy batch
    # batch = datasets.ArrayBatch(
    #     x=jnp.zeros((100, 128), dtype=jnp.int32),  # 100 tokenized sentences
    #     y=jnp.zeros((100,), dtype=jnp.int32)
    # )

    # # Step 6: Sample a batch using active learner
    # key = jax.random.PRNGKey(0)
    # params, state = enn.init(key, batch.x)
    # acquired_batch = active_learner.sample_batch(params, state, batch, key)

    # # Get first batch of data
    # dataset = load_dataset("glue", "sst2")
    # train_data = dataset["train"]
    # sample = train_data[0]
    # print(f"\n\nDEBUG: sample:\n{sample} \n\n")
    # text = sample["sentence"]
    # label = sample["label"]

    # # Tokenize input (assuming max length 128, can change)
    # encoded = huggingface_tokenizer(
    #     text,
    #     padding="max_length",
    #     truncation=True,
    #     max_length=128,
    #     return_tensors="np"  # return NumPy arrays to convert easily to jnp
    # )

    # # Create a BertInput-like object (mimicking your base.BertInput)
    # bert_input = SimpleNamespace(
    #     token_ids=jnp.array(encoded["input_ids"]),
    #     segment_ids=jnp.array(encoded["token_type_ids"]),
    #     input_mask=jnp.array(encoded["attention_mask"]),
    # )

    # # Just to verify
    # print("Token IDs shape:", bert_input.token_ids.shape)
    # print("Segment IDs shape:", bert_input.segment_ids.shape)
    # print("Input Mask shape:", bert_input.input_mask.shape)
    # print("Label:", label)
    # print("Sample sentence: ", text)

    # ctor_margin = al.make_priority_fn_ctor()

    # al.PrioritizedBatcher(first_batch, priorities.margin_per_example)

    # pretrained = get_pretrained_weights()
    # # Define the configuration for the model
    # config = {
    #     "pretrained": pretrained,
    #     "max_length": 512,
    #     "embed_dropout_rate": 0.1,
    #     "fully_connected_drop_rate": 0.1,
    #     "attention_drop_rate": 0.1,
    #     "hidden_size": 768,
    #     "intermediate_size": 3072,
    #     "n_heads": 12,
    #     "n_layers": 12,
    #     "mask_id": 1,
    #     "weight_stddev": 0.02,
    #     # For use later in finetuning
    #     "n_classes": 2,
    #     "classifier_drop_rate": 0.1,
    #     "learning_rate": 1e-5,
    #     "max_grad_norm": 1.0,
    #     "l2": 0.1,
    #     "n_epochs": 5,
    #     "batch_size": 4,
    # }

    # roberta = hk.transform(featurizer_fn, apply_rng=True)
    # # Test the model
    # sample_text = "This was a lot less painful than re-implementing a tokenizer"
    # encoded = huggingface_tokenizer.batch_encode_plus(
    #     [sample_text, sample_text], pad_to_max_length=True, max_length=config["max_length"]
    # )
    # sample_tokens = encoded["input_ids"]
    # print(sample_tokens[0][:20])

    # rng = PRNGKey(42)
    # sample_tokens = jnp.asarray(sample_tokens)
    # params = roberta.init(rng, sample_tokens, training=False)
    # contextual_embedding = jax.jit(roberta.apply)(params, rng, sample_tokens)
    # print(contextual_embedding.shape)
