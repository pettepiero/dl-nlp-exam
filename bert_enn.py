import haiku as hk
import jax.numpy as jnp
import jax


class Scope(object):
    """
    A tiny utility to help make looking up into our dictionary cleaner.
    There's no haiku magic here.
    """

    def __init__(self, weights, prefix):
        self.weights = weights
        self.prefix = prefix

    def __getitem__(self, key):
        return self.weights[self.prefix + key]


class PositionEmbeddings(hk.Module):
    """
    A position embedding of shape [n_seq, n_hidden]
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # The Roberta position embeddings are offset by 2
        self.offset = 2

    def __call__(self):
        pretrained_position_embedding = self.config["pretrained"][
            "embeddings/position_embeddings"
        ]
        position_weights = hk.get_parameter(
            "position_embeddings",
            pretrained_position_embedding.shape,
            init=hk.initializers.Constant(pretrained_position_embedding),
        )

        return position_weights[self.offset : self.offset + self.config["max_length"]]


class MultiHeadAttention(hk.Module):

    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.n = layer_num

    def _split_into_heads(self, x):
        return jnp.reshape(
            x,
            [
                x.shape[0],
                x.shape[1],
                self.config["n_heads"],
                x.shape[2] // self.config["n_heads"],
            ],
        )

    def __call__(self, x, mask, training=False):
        """
        x: tensor of shape (batch, seq, n_hidden)
        mask: tensor of shape (batch, seq)
        """
        scope = Scope(self.config["pretrained"], f"encoder/layer_{self.n}/attention/")

        # Project to queries, keys, and values
        # Shapes are all [batch, sequence_length, hidden_size]
        queries = hk.Linear(
            output_size=self.config["hidden_size"],
            w_init=hk.initializers.Constant(scope["query/kernel"]),
            b_init=hk.initializers.Constant(scope["query/bias"]),
        )(x)
        keys = hk.Linear(
            output_size=self.config["hidden_size"],
            w_init=hk.initializers.Constant(scope["key/kernel"]),
            b_init=hk.initializers.Constant(scope["key/bias"]),
        )(x)
        values = hk.Linear(
            output_size=self.config["hidden_size"],
            w_init=hk.initializers.Constant(scope["value/kernel"]),
            b_init=hk.initializers.Constant(scope["value/bias"]),
        )(x)

        # Reshape our hidden state to group into heads
        # New shape are [batch, sequence_length, n_heads, size_per_head]
        queries = self._split_into_heads(queries)
        keys = self._split_into_heads(keys)
        values = self._split_into_heads(values)

        # Compute per head attention weights
        # b: batch
        # s: source sequence
        # t: target sequence
        # n: number of heads
        # h: per-head hidden state

        # Note -- we could also write this with jnp.reshape and jnp.matmul, but I'm becoming
        # a fan of how concise opting to use einsum notation for this kind of operation is.
        # For more info, see https://rockt.github.io/2018/04/30/einsum and
        # https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/
        attention_logits = jnp.einsum("bsnh,btnh->bnst", queries, keys) / jnp.sqrt(
            queries.shape[-1]
        )
        # Add logits of mask tokens with a large negative number to prevent attending to those terms.
        # attention_logits += jnp.reshape(mask * -2**32, [mask.shape[0], 1, 1, mask.shape[1]])
        attention_logits += jnp.reshape(
            mask * -(2**16), [mask.shape[0], 1, 1, mask.shape[1]]
        )
        # Instead of -2**32, use -1e9 or -jnp.inf which are handled safely by softmax
        # attention_logits = jnp.where(mask[..., None, :], -1e9, attention_logits)
        attention_weights = jax.nn.softmax(attention_logits, axis=-1)
        per_head_attention_output = jnp.einsum(
            "btnh,bnst->bsnh", values, attention_weights
        )
        attention_output = jnp.reshape(
            per_head_attention_output,
            [
                per_head_attention_output.shape[0],
                per_head_attention_output.shape[1],
                per_head_attention_output.shape[2] * per_head_attention_output.shape[3],
            ],
        )

        # Apply dense layer to output of attention operation
        attention_output = hk.Linear(
            output_size=self.config["hidden_size"],
            w_init=hk.initializers.Constant(scope["output/dense/kernel"]),
            b_init=hk.initializers.Constant(scope["output/dense/bias"]),
        )(attention_output)

        # Apply dropout at training time
        if training:
            attention_output = hk.dropout(
                rng=hk.next_rng_key(),
                rate=self.config["attention_drop_rate"],
                x=attention_output,
            )

        return attention_output


class Embedding(hk.Module):
    """
    Embeds tokens and positions into an array of shape [n_batch, n_seq, n_hidden]
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def __call__(self, token_ids, training=False):
        """
        token_ids: ints of shape (batch, n_seq)
        """
        word_embeddings = self.config["pretrained"]["embeddings/word_embeddings"]

        # We have to flatten our tokens before passing them to the hk.Embed module,
        # as arrays with more than one dimension are interpreted as multi-dimensional indexes
        flat_token_ids = jnp.reshape(
            token_ids, [token_ids.shape[0] * token_ids.shape[1]]
        )
        flat_token_embeddings = hk.Embed(
            vocab_size=word_embeddings.shape[0],
            embed_dim=word_embeddings.shape[1],
            # Here we're using hk.initializers.Constant to supply pre-trained embeddings
            # to our hk.Embed module
            w_init=hk.initializers.Constant(
                self.config["pretrained"]["embeddings/word_embeddings"]
            ),
        )(flat_token_ids)

        # After we've embedded our token IDs, we reshape to recover our batch dimension
        token_embeddings = jnp.reshape(
            flat_token_embeddings,
            [token_ids.shape[0], token_ids.shape[1], word_embeddings.shape[1]],
        )

        # Combine our token embeddings with a set of learned positional embeddings
        embeddings = token_embeddings + PositionEmbeddings(self.config)()
        embeddings = hk.LayerNorm(
            axis=-1,
            create_scale=True,
            create_offset=True,
            # The layer norm parameters are also pretrained, so we have to take care to
            # use a constant initializer for these as well
            scale_init=hk.initializers.Constant(
                self.config["pretrained"]["embeddings/LayerNorm/gamma"]
            ),
            offset_init=hk.initializers.Constant(
                self.config["pretrained"]["embeddings/LayerNorm/beta"]
            ),
        )(embeddings)

        # Dropout is will be applied later when we finetune our Roberta implementation
        # to solve a classification task. For now we'll set `training` to False.
        if training:
            embeddings = hk.dropout(
                # Haiku magic -- we'll explicitly provide a RNG key to haiku later to make this function
                hk.next_rng_key(),
                rate=self.config["embed_dropout_rate"],
                x=embeddings,
            )

        return embeddings


def gelu(x):
    """
    We use this in place of jax.nn.relu because the approximation used
    produces a non-trivial difference in the output state
    """
    return x * 0.5 * (1.0 + jax.scipy.special.erf(x / jnp.sqrt(2.0)))


class TransformerMLP(hk.Module):

    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.n = layer_num

    def __call__(self, x, training=False):
        # Project out to higher dim
        scope = Scope(self.config["pretrained"], f"encoder/layer_{self.n}/")
        intermediate_output = hk.Linear(
            output_size=self.config["intermediate_size"],
            w_init=hk.initializers.Constant(scope["intermediate/dense/kernel"]),
            b_init=hk.initializers.Constant(scope["intermediate/dense/bias"]),
        )(x)

        # Apply gelu nonlinearity
        intermediate_output = gelu(intermediate_output)

        # Project back down to hidden size
        output = hk.Linear(
            output_size=self.config["hidden_size"],
            w_init=hk.initializers.Constant(scope["output/dense/kernel"]),
            b_init=hk.initializers.Constant(scope["output/dense/bias"]),
        )(intermediate_output)

        # Apply dropout at training time
        if training:
            output = hk.dropout(
                rng=hk.next_rng_key(),
                rate=self.config["fully_connected_drop_rate"],
                x=output,
            )

        return output


class TransformerBlock(hk.Module):

    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.n = layer_num

    def __call__(self, x, mask, training=False):
        scope = Scope(self.config["pretrained"], f"encoder/layer_{self.n}/")
        # Feed our input through a multi-head attention operation
        attention_output = MultiHeadAttention(self.config, self.n)(
            x, mask, training=training
        )

        # Add a residual connection with the input to the layer
        residual = attention_output + x

        # Apply layer norm to the combined output
        attention_output = hk.LayerNorm(
            axis=-1,
            create_scale=True,
            create_offset=True,
            scale_init=hk.initializers.Constant(
                scope["attention/output/LayerNorm/gamma"]
            ),
            offset_init=hk.initializers.Constant(
                scope["attention/output/LayerNorm/beta"]
            ),
        )(residual)

        # Project out to a larger dim, apply a gelu, and then project back down to our hidden dim
        mlp_output = TransformerMLP(self.config, self.n)(
            attention_output, training=training
        )

        # Residual connection to the output of the attention operation
        output_residual = mlp_output + attention_output

        # Apply another LayerNorm
        layer_output = hk.LayerNorm(
            axis=-1,
            create_scale=True,
            create_offset=True,
            scale_init=hk.initializers.Constant(scope["output/LayerNorm/gamma"]),
            offset_init=hk.initializers.Constant(scope["output/LayerNorm/beta"]),
        )(output_residual)
        return layer_output


def test_code():

    # 1. Load the SST-2 dataset (train pool)
    sst2 = load_dataset("glue", "sst2")
    train_texts = sst2["train"]["sentence"]
    train_labels = jnp.array(sst2["train"]["label"])

    # Just take a small pool for now
    N = 64
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
            return_tensors="np",
        )
        return tok_out["input_ids"]

    x_pool = encode(texts_pool)
    y_pool = labels_pool

    encoded = huggingface_tokenizer(
        texts_pool,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np",  # return NumPy arrays to convert easily to jnp
    )

    bert_input = SimpleNamespace(
        token_ids=jnp.array(encoded["input_ids"]),
        segment_ids=jnp.array(encoded["token_type_ids"]),
        input_mask=jnp.array(encoded["attention_mask"]),
    )

    # bert_input = {
    #     "token_ids": jnp.array(encoded["input_ids"]),
    #     "segment_ids": jnp.array(encoded["token_type_ids"]),
    #     "input_mask": jnp.array(encoded["attention_mask"]),
    # }

    init_input = SimpleNamespace(
        token_ids=bert_input.token_ids[:1],
        segment_ids=bert_input.segment_ids[:1],
        input_mask=bert_input.input_mask[:1],
    )

    # Note: Create a list of SimpleNamespace for each example
    # x_full = [
    #     SimpleNamespace(
    #         token_ids=jnp.array(encoded["input_ids"][i]),
    #         segment_ids=jnp.array(encoded["token_type_ids"][i]),
    #         input_mask=jnp.array(encoded["attention_mask"][i])
    #     )
    #     for i in range(len(encoded["input_ids"]))
    # ]

    x_full = [
        BertInput(
            token_ids=jnp.array(encoded["input_ids"][i]),
            segment_ids=jnp.array(encoded["token_type_ids"][i]),
            input_mask=jnp.array(encoded["attention_mask"][i]),
        )
        for i in range(len(encoded["input_ids"]))
    ]

    # Replace the SimpleNamespace with a dictionary or tuple structure
    # x_full = {
    #     'token_ids': jnp.array(encoded["input_ids"]),
    #     'segment_ids': jnp.array(encoded["token_type_ids"]),
    #     'input_mask': jnp.array(encoded["attention_mask"])
    # }

    # Create the batch
    # batch = datasets.ArrayBatch(x=x_full, y=jnp.array(y_pool))

    # When creating your batch, use BertInput instead of SimpleNamespace
    batch = BertInput(
        token_ids=jnp.array(encoded["input_ids"]),  # Convert to JAX array
        segment_ids=jnp.array(encoded["token_type_ids"]),
        input_mask=jnp.array(encoded["attention_mask"]),
        extra={},  # Empty dict unless you have additional inputs
    )

    # Then create your ArrayBatch with this input
    batch = datasets.ArrayBatch(x=batch, y=jnp.array(y_pool))

    # Make batch
    # batch = datasets.ArrayBatch(x=jnp.array(x_pool), y=jnp.array(y_pool))

    # 3. Create the ENN model
    enn_model, haiku_params, haiku_state = create_enn_bert_for_classification_ft()

    # Initialize model
    key = jax.random.PRNGKey(0)
    print(f"\n\nDEBUG: x_pool = {x_pool}\n\n")

    # params, state = enn_model.init(key, x_pool[:1], enn_model.indexer(key))
    params, state = enn_model.init(key, init_input, enn_model.indexer(key))

    print(f"DEBUG: enn_model.init done!")
    # 4. Create forwarder
    enn_batch_fwd = make_batch_fwd(enn_model, num_enn_samples=100, seed=42)
    print("make_batch_fwd done!")
    # 5. Define acquisition strategy (e.g., predictive variance)
    per_example_priority = priorities.get_per_example_priority("variance")
    print("get_per_example_priority done!")
    priority_fn_ctor = priorities.make_priority_fn_ctor(per_example_priority)
    print("make_priority_fn_ctor done!")

    # 6. Create PrioritizedBatcher
    batcher = prioritized.PrioritizedBatcher(
        enn_batch_fwd=enn_batch_fwd,
        acquisition_size=32,
        priority_fn_ctor=priority_fn_ctor,
    )

    print(f"\n\nDEBUG: type(batch) = {type(batch)}")
    print(f"\n\nDEBUG: len(batch.x) = \n{len(batch.x)}\n\n")
    print(f"\n\nDEBUG: len(batch.y) = \n{len(batch.y)}\n\n")
    print(f"\n\nDEBUG: bert_input = \n{bert_input}\n\n")
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
