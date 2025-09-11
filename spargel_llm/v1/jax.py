from dataclasses import dataclass

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from spargel_llm.layers.jax import (
    DotProductAttention,
    Embedding,
    FeedForward,
    LayerNorm,
    Linear,
    PositionEncodingStrategy,
    StaticPositionEncoding,
    all_you_need_init,
)


@dataclass
class MicroLMConfig:
    vocab_size: int
    embed_dim: int
    qk_dim: int
    hidden_dim: int
    block_size: int
    layer_count: int

    position_encoding: PositionEncodingStrategy = PositionEncodingStrategy.ALL_YOU_NEED


class MicroLM(nnx.Module):
    attentions: list[DotProductAttention]
    feed_forwards: list[FeedForward]

    def __init__(
        self,
        config: MicroLMConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.token_embed = Embedding(config.vocab_size, config.embed_dim, rngs=rngs)

        match config.position_encoding:
            case PositionEncodingStrategy.NONE:
                self.pos_embed = None
            case PositionEncodingStrategy.LEARNED:
                self.pos_embed = Embedding(
                    config.block_size, config.embed_dim, rngs=rngs
                )
            case PositionEncodingStrategy.ALL_YOU_NEED:
                self.pos_embed = StaticPositionEncoding(
                    config.block_size, config.embed_dim, all_you_need_init
                )
            case _:
                raise Exception("unimplmented")

        self.embed_normalization = LayerNorm(config.embed_dim, rngs=rngs)

        self.attentions = []
        self.feed_forwards = []

        for _ in range(config.layer_count):
            self.attentions.append(
                DotProductAttention(
                    config.embed_dim,
                    config.embed_dim,
                    config.embed_dim,
                    config.embed_dim,
                    config.qk_dim,
                    rngs=rngs,
                )
            )
            self.feed_forwards.append(
                FeedForward(
                    config.embed_dim, config.embed_dim, config.hidden_dim, rngs=rngs
                )
            )
        self.lm_head = Linear(config.embed_dim, config.vocab_size, rngs=rngs)

    def __call__(self, input: jax.Array, padding_mask: jax.Array):
        """
        Args:
            input: shape (...batch, block_size). an array of token ids
        """
        seq_len = input.shape[-1]
        assert seq_len == self.config.block_size
        assert seq_len == padding_mask.shape[-1]

        mask = padding_mask[:, jnp.newaxis] & padding_mask[..., jnp.newaxis]

        x = self.token_embed(input)

        if self.pos_embed is not None:
            x += self.pos_embed(jnp.arange(seq_len))

        for i in range(self.config.layer_count):
            x_norm = self.embed_normalization(x)
            x += self.attentions[i](x_norm, x_norm, x_norm, mask=mask)
            x_norm = self.embed_normalization(x)
            x += self.feed_forwards[i](x_norm)

        x_norm = self.embed_normalization(x)
        return self.lm_head(x_norm)
