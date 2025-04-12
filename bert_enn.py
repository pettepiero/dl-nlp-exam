import haiku as hk
from transformers import RobertaModel
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
