import jax
import jax.numpy as jnp
import haiku as hk
from transformers import BertModel, BertTokenizer, BertConfig
from enn.networks.bert.base import BertInput
from enn.networks.bert.bert_v2 import make_bert_base, BertConfigCustom


def torch_to_jax(torch_tensor):
    return jnp.array(torch_tensor.detach().cpu().numpy())


def embedding_params_fix(haiku_params, jax_bert_params):
    haiku_params["BERT/word_embeddings"]["embeddings"] = jax_bert_params[
        "embeddings.word_embeddings.weight"
    ]
    haiku_params["BERT/token_type_embeddings"]["embeddings"] = jax_bert_params[
        "embeddings.token_type_embeddings.weight"
    ]
    haiku_params["BERT/position_embeddings"]["embeddings"] = jax_bert_params[
        "embeddings.position_embeddings.weight"
    ]
    haiku_params["BERT/embeddings_ln"]["scale"] = jax_bert_params[
        "embeddings.LayerNorm.weight"
    ]
    haiku_params["BERT/embeddings_ln"]["offset"] = jax_bert_params[
        "embeddings.LayerNorm.bias"
    ]
    return haiku_params


def layer_params_fix(haiku_params, jax_bert_params, layer_id):
    conversion_dict = {
        "query_": "attention.self.query",
        "keys_": "attention.self.key",
        "values_": "attention.self.value",
        "attention_output_dense_": "attention.output.dense",
        "attention_output_ln_": "attention.output.LayerNorm",
        "intermediate_output_": "intermediate.dense",
        "layer_output_": "output.dense",
        "layer_output_ln_": "output.LayerNorm",
    }

    for name, hf_name in conversion_dict.items():
        haiku_name = f"BERT/~_bert_layer/{name}{layer_id}"
        jax_name = f"encoder.layer.{layer_id}.{hf_name}"

        if "dense" in hf_name or "query" in hf_name or "key" in hf_name or "value" in hf_name:
            haiku_params[haiku_name]["w"] = jax_bert_params[jax_name + ".weight"].T
        else:
            haiku_params[haiku_name]["w"] = jax_bert_params[jax_name + ".weight"]
        haiku_params[haiku_name]["b"] = jax_bert_params[jax_name + ".bias"]
    return haiku_params


def load_pretrained_bert_weights(pretrained_model_name="bert-base-uncased"):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    torch_bert = BertModel.from_pretrained(pretrained_model_name)
    config = BertConfig.from_pretrained(pretrained_model_name)
    jax_weights = {k: torch_to_jax(v) for k, v in torch_bert.state_dict().items() if "pooler" not in k}
    return tokenizer, config, jax_weights

