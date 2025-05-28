import jax
import jax.numpy as jnp
import haiku as hk
from transformers import BertModel, BertTokenizer, BertConfig
from enn.networks.bert.base import BertInput
from enn.networks.bert.bert_v2 import make_bert_base, BertConfigCustom


def th_tensor_to_jnp_array(torch_tensor):
    return jnp.array(torch_tensor.detach().cpu().numpy())


def embedding_params_fix(haiku_params, hf_params):
    if haiku_params["BERT/word_embeddings"]["embeddings"].shape !=  hf_params["embeddings.word_embeddings.weight"].shape:
        print("ERROR in embedding_params_fix (1)")
        exit(1)
    else:
        print("Passed")
    haiku_params["BERT/word_embeddings"]["embeddings"] = hf_params[
        "embeddings.word_embeddings.weight"
    ]

    if haiku_params["BERT/token_type_embeddings"]["embeddings"].shape !=  hf_params["embeddings.token_type_embeddings.weight"].shape:
        print("ERROR in embedding_params_fix (2)")
        exit(1)
    else:
        print("Passed")
    haiku_params["BERT/token_type_embeddings"]["embeddings"] = hf_params[
        "embeddings.token_type_embeddings.weight"
    ]

    if haiku_params["BERT/position_embeddings"]["embeddings"].shape !=  hf_params["embeddings.position_embeddings.weight"].shape:
        print("ERROR in embedding_params_fix (3)")
        exit(1)
    else:
        print("Passed")
    haiku_params["BERT/position_embeddings"]["embeddings"] = hf_params[
        "embeddings.position_embeddings.weight"
    ]
    if haiku_params["BERT/embeddings_ln"]["scale"].shape !=  hf_params["embeddings.LayerNorm.weight"].shape:
        print("ERROR in embedding_params_fix (4)")
        exit(1)
    else:
        print("Passed")
    haiku_params["BERT/embeddings_ln"]["scale"] = hf_params[
        "embeddings.LayerNorm.weight"
    ]
    if haiku_params["BERT/embeddings_ln"]["offset"].shape !=  hf_params["embeddings.LayerNorm.bias"].shape:
        print("ERROR in embedding_params_fix (5)")
        exit(1)
    else:
        print("Passed")
    haiku_params["BERT/embeddings_ln"]["offset"] = hf_params[
        "embeddings.LayerNorm.bias"
    ]
    return haiku_params


def layer_params_fix(haiku_params, hf_params, layer_id):
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
        hf_name = f"encoder.layer.{layer_id}.{hf_name}"

        if "dense" in hf_name or "query" in hf_name or "key" in hf_name or "value" in hf_name:
            haiku_params[haiku_name]["w"] = hf_params[hf_name + ".weight"].T
        else:
            haiku_params[haiku_name]["w"] = hf_params[hf_name + ".weight"]
        haiku_params[haiku_name]["b"] = hf_params[hf_name + ".bias"]
    return haiku_params


def load_pretrained_bert_weights(pretrained_model_name="bert-base-uncased"):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    torch_bert = BertModel.from_pretrained(pretrained_model_name)
    config = BertConfig.from_pretrained(pretrained_model_name)
    hf_weights = {k: th_tensor_to_jnp_array(v) for k, v in torch_bert.state_dict().items() if "pooler" not in k}
    return tokenizer, config, hf_weights 

