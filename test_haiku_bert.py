import matplotlib.pyplot as plt
from enn.networks.bert.bert_v2 import BertConfigCustom, make_bert_base
from huggingface_haiku_loader import (
    load_pretrained_bert_weights,
    embedding_params_fix,
    layer_params_fix,
)
from typing import NamedTuple
import jax.numpy as jnp
import jax

import numpy as np
from transformers import BertModel, BertTokenizer, BertConfig
import torch


class BertInput(NamedTuple):
    token_ids: jnp.ndarray
    segment_ids: jnp.ndarray
    input_mask: jnp.ndarray

def compare_embeddings(tokens, torch_embeddings, jax_embeddings, atol=1e-4):
    print(f"{'Token':<12} {'L2 Diff':>10}  {'Equal?':>8}")
    print("-" * 34)
    
    for token, torch_emb, jax_emb in zip(tokens, torch_embeddings, jax_embeddings):
        # Ensure both are NumPy arrays
        torch_vec = torch_emb.detach().cpu().numpy()
        jax_vec = np.array(jax_emb)
        
        # Compute L2 norm of difference
        diff = np.linalg.norm(torch_vec - jax_vec)
        close = np.allclose(torch_vec, jax_vec, atol=atol)
        
        print(f"{token:<12} {diff:10.6f}  {str(close):>8}")

if __name__ == "__main__":

    ######################################################################
    # Haiku part
    ######################################################################

    print("Creating base model....")
    # Load HuggingFace model + weights
    tokenizer, haiku_config, hf_weights = load_pretrained_bert_weights()
    
    # build haiku model
    haiku_base_model = make_bert_base(
        bert_config = haiku_config,
        is_training=False
    )

    # Create dummy input
    haiku_dummy_input = BertInput(
        token_ids=jnp.zeros((1, 128), dtype=jnp.int32),
        segment_ids=jnp.zeros((1, 128), dtype=jnp.int32),
        input_mask=jnp.ones((1, 128), dtype=jnp.int32),
    )

    # Initialize random Haiku parameters
    haiku_params = haiku_base_model.init(jax.random.PRNGKey(0), haiku_dummy_input)
    
    # Apply embedding + encoder weights
    haiku_params = embedding_params_fix(haiku_params, hf_weights)
    for layer in range(haiku_config.num_hidden_layers):
        haiku_params = layer_params_fix(haiku_params, hf_weights, layer)
    print("Conversion successful\n")


    sentence = "The cat sat on the mat."
    #hf_tokens = tokenizer(sentence, padding="max_length", truncation=True, max_length=128, return_tensors='np')
    hf_tokens = tokenizer(sentence, truncation=True, max_length=128, return_tensors='np')
    inputs = BertInput(
        token_ids=jnp.array(hf_tokens["input_ids"]),
        segment_ids=jnp.array(hf_tokens["token_type_ids"]),
        input_mask=jnp.array(hf_tokens["attention_mask"]),
    )
    print(f"inputs: \n{inputs}")
    
    haiku_outputs = haiku_base_model.apply(haiku_params, jax.random.PRNGKey(0), inputs)
    haiku_ids = inputs.token_ids[0].tolist()
 
    tokens = tokenizer.convert_ids_to_tokens(inputs.token_ids[0].tolist())

    last_layer = haiku_outputs.extra['last_layer']  # shape: (1, 128, 768)
    print("\n\nJAX/HAIKU:")

    print("last_layer shape:", last_layer.shape)

    for token, embedding in zip(tokens, last_layer[0]):
        print(f"{token:12} -> {embedding[:5]}")
    

    ###################################################################àà
    # Hugging face part
    ###################################################################

    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = BertModel.from_pretrained("google-bert/bert-base-uncased")
    hf_config = BertConfig.from_pretrained("google-bert/bert-base-uncased")
    
    sentence = "The cat sat on the mat."
    inputs = tokenizer(sentence, return_tensors='pt')
    
    with torch.no_grad():
        torch_outputs = model(**inputs)
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    hf_ids = inputs['input_ids'][0].tolist()
    print("\n\nHugging Face:")

    for token, embedding in zip(tokens, torch_outputs.last_hidden_state[0]):
        print(f"{token:12} -> {embedding[:5]}")  # Print first 5 dimensions for brevity


    ######################################################################
    # Comparison part
    ######################################################################
    print("\n **************** Beginning test ***************")
    print(">>>> Token IDS match:\n", hf_ids == haiku_ids)
    print("\n")

    print(">>>> Test on output embeddings:\n")
    compare_embeddings(tokens, torch_outputs.last_hidden_state[0], haiku_outputs.extra['last_layer'][0])
    print("\n")

    print(">>>> Comparing config dicts:\n ")
    #print(f"hf_config:\n{hf_config}")
    #print(f"haiku_config:\n{haiku_config}")
    print(f"Dicts are same: ", hf_config == haiku_config)
    print("\n")

    print(f"DEBUG: hf_config = \n{hf_config}")
    print(f"DEBUG: haiku_config = \n{haiku_config}")
    
    print(f">>>> Manually inspecting some weights:\n")
    # debug prints
#    print(f"DEBUG: hf_weights.keys():")
#    for key in hf_weights.keys():
#        print(f"{key}")
#    print("\n")
#    print(f"\nDEBUG: haiku_params.keys()")
#    for key in haiku_params.keys():
#        print(f"{key}")
#
#    print("\n")
#    print(f"haiku_layer_2 query weights:\n{haiku_params['BERT/~_bert_layer/query_2']['w'].T}")
#    print(f"\nhf_weights['encoder.layer.2.attention.self.query.weight'] =\n{hf_weights['encoder.layer.2.attention.self.query.weight']}")
#    print("\n\n")
#    
#    print(f"haiku_params['BERT/~_bert_layer/query_2']['w'].shape = {haiku_params['BERT/~_bert_layer/query_2']['w'].shape}")
#    print(f"hf_weights['encoder.layer.2.attention.self.query.weight'].shape = {hf_weights['encoder.layer.2.attention.self.query.weight'].shape}")
    print(np.allclose(haiku_params['BERT/~_bert_layer/query_2']['w'].T, hf_weights['encoder.layer.2.attention.self.query.weight'], atol=1e-05))
    print("\n")

    torch_weights = hf_weights['encoder.layer.1.attention.output.dense.weight']
    haiku_weights = haiku_params['BERT/~_bert_layer/attention_output_dense_1']['w']
    
   # print(f"Shapes: encoder.layer.1.intermediate.dense.weight.shape = {torch_weights.shape} | BERT/~_bert_layer/attention_output_dense_1['w'].shape = {haiku_weights.shape}") 
    print(f"Testing some params -> test passed: {np.allclose(torch_weights.T, haiku_weights, atol=1e-05)}")


