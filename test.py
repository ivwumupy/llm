import torch
import torch.nn as nn
from torch import Tensor


def scaled_dot_product(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    *,
    mask: Tensor,
) -> Tensor:
    """
    Args:
        Q: (..., cnt_q, d_key)
        K: (..., cnt_k, d_key)
        V: (..., cnt_k, d_value)

        mask: (..., cnt_q, cnt_k)
    Returns:
        (..., cnt_q, d_value)
    """

    torch._assert(Q.size(-1) == K.size(-1), "d_key not matched")
    torch._assert(K.size(-2) == V.size(-2), "cnt_k not matched")

    cnt_q, cnt_k = Q.size(-2), K.size(-2)

    torch._assert(mask.dtype == torch.bool, "mask dtype != bool")
    torch._assert(mask.size(-2) == cnt_q, "bad mask size (-2)")
    torch._assert(mask.size(-1) == cnt_k, "bad mask size (-1)")

    # (..., cnt_q, cnt_k)
    scores = torch.einsum("...ik, ...jk -> ...ij", Q, K)

    # native code bug in PyTorch library
    # scores = do_mask(scores, mask)
    scores = scores.masked_fill(mask, -1e9)

    return scores @ V


class Attention(nn.Module):
    """Multihead scaled dot-product attention

    Args:
        x: (..., seq_len, d_in)
        mask (Optional): (..., seq_len), dtype=bool
    Returns:
        (..., seq_len, d_out)
    """

    def __init__(
        self,
        cnt_h: int,
        d_in: int,
        d_out: int,
        d_key: int,
        d_value: int,
    ):
        """
        Args:
            cnt_h: number of heads
            d_in: input dimension
            d_out: output dimension
            d_key: key dimension
            d_value: value dimension
        """
        super().__init__()

        self._cnt_h = cnt_h

        self._W_q = nn.Parameter(torch.rand(cnt_h, d_in, d_key))
        self._W_k = nn.Parameter(torch.rand(cnt_h, d_in, d_key))
        self._W_v = nn.Parameter(torch.rand(cnt_h, d_in, d_value))
        self._W_o = nn.Parameter(torch.rand(cnt_h, d_value, d_out))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        seq_len = x.size(-2)
        torch._assert(mask.size(-1) == seq_len, "bad mask size")

        # x: (..., seq_len, d_in)

        # W_q: (cnt_h, d_in, d_key)
        Q = torch.einsum(
            "ijk, ...lj -> ...ilk", self._W_q, x
        )  # (..., cnt_h, seq_len, d_key)

        # W_k: (cnt_h, d_in, d_key)
        K = torch.einsum(
            "ijk, ...lj -> ...ilk", self._W_k, x
        )  # (..., cnt_h, seq_len, d_key)

        # W_v: (cnt_h, d_in, d_value)
        V = torch.einsum(
            "ijk, ...lj -> ...ilk", self._W_v, x
        )  # (..., cnt_h, seq_len, d_value)

        mask = mask.unsqueeze(-2) | mask.unsqueeze(-1)
        # mask: (..., seq_len, seq_len) | (seq_len, seq_len) | None
        mask = mask.unsqueeze(-3)
        # mask: (..., 1, seq_len, seq_len) | (1, seq_len, seq_len) | None

        values = scaled_dot_product(Q, K, V, mask=mask)

        # values: (..., cnt_h, seq_len, d_value)
        # W_o: (cnt_h, d_value, d_out)
        return torch.einsum("ijk, ...ilj -> ...lk", self._W_o, values)


class TransformerBlock(nn.Module):
    """One transformer block

    This module consists of self-attention and feed-forward layers.

    Args:
        x: (..., seq_len, dim)
        mask: (..., seq_len), dtype=bool
    Returns:
        (..., seq_len, dim)
    """

    def __init__(self, cnt_h: int, dim: int, d_key: int, d_value: int, d_hidden: int):
        super().__init__()

        self._attention = Attention(
            cnt_h=cnt_h,
            d_in=dim,
            d_out=dim,
            d_key=d_key,
            d_value=d_value,
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        y = x
        x = self._attention(x, mask)

        # NOTE(tianjiao): This is essential to the BUG.
        x += y

        return x


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.A = nn.Parameter(torch.rand(vocab_size, embed_dim))

    def forward(self, x):
        # (...,seq_len)
        return self.A[x]


class LLM(nn.Module):
    """The full LLM

    Args:
        tokens: (..., seq_len), dtype=int
        mask: (..., seq_len), dtype=bool
    Returns:
        (..., vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        cnt_h: int,
        dim: int,
        d_key: int,
        d_value: int,
        d_hidden: int,
    ):
        super().__init__()

        self._token_embedding = Embedding(vocab_size, dim)
        self._transformer = TransformerBlock(
            cnt_h=cnt_h, dim=dim, d_key=d_key, d_value=d_value, d_hidden=d_hidden
        )
        self._head = nn.Linear(dim, vocab_size)

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        x = self._token_embedding(tokens)
        x = self._transformer(x, mask)
        x = self._head(x)
        return x


def calculate_loss(
    model: LLM,
    input: Tensor,
    mask: Tensor,
    target: Tensor,
    pad_index: int,
) -> Tensor:
    """
    Args:
        input: (..., seq_len), dtype=int
        mask: (..., seq_len), dtype=bool
        target: (..., seq_len), dtype=int
    """

    torch._assert(input.shape == target.shape, "shape not matched")

    logits: Tensor = model(input, mask)  # (..., seq_len, vocab_size)
    loss = nn.functional.cross_entropy(
        logits.flatten(0, -2), target.flatten(0, -1), ignore_index=pad_index
    )

    return loss


# %%
@torch.compile
def train_step(model: LLM, inputs: Tensor, masks: Tensor, targets: Tensor):
    loss = calculate_loss(model, inputs, masks, targets, pad_index=0)
    loss.backward()


# %%
max_seq_len = 4

model = LLM(
    vocab_size=2,
    cnt_h=1,
    dim=8,
    d_key=1,
    d_value=1,
    d_hidden=1,
)

batch_size = 2

inputs = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 0]])
masks = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
targets = torch.zeros(batch_size, max_seq_len, dtype=torch.int)

# train
model.train()

print(inputs)

train_step(model, inputs, masks, targets)
