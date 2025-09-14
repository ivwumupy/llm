import torch
import torch.nn as nn
from torch import Tensor

# The problematic kernel is `cpp_fused_add_index_put_new_zeros_4`.


class Attention(nn.Module):
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

        self._W_q = nn.Parameter(torch.rand(cnt_h, d_in, d_key))
        self._W_k = nn.Parameter(torch.rand(cnt_h, d_in, d_key))
        self._W_v = nn.Parameter(torch.rand(cnt_h, d_in, d_value))
        self._W_o = nn.Parameter(torch.rand(cnt_h, d_value, d_out))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., seq_len, d_in)
        Returns:
            (..., seq_len, d_out)
        """
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

        # (..., cnt_q, cnt_k)
        scores = torch.einsum("...ik, ...jk -> ...ij", Q, K)

        # values = scores @ V
        values = torch.einsum("...qk,...kv->...qv", scores, V)

        # values: (..., cnt_h, seq_len, d_value)
        # W_o: (cnt_h, d_value, d_out)
        return torch.einsum("ijk, ...ilj -> ...lk", self._W_o, values)


class TransformerBlock(nn.Module):
    def __init__(self, cnt_h: int, dim: int, d_key: int, d_value: int):
        super().__init__()

        self._attention = Attention(
            cnt_h=cnt_h,
            d_in=dim,
            d_out=dim,
            d_key=d_key,
            d_value=d_value,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., seq_len, dim)
        Returns:
            (..., seq_len, dim)
        """
        y = x
        x = self._attention(x)

        # NOTE(tianjiao): This is essential to the BUG.
        x = y + x

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

    def __init__(self, vocab_size: int, cnt_h: int, dim: int, d_key: int, d_value: int):
        super().__init__()

        self._token_embedding = Embedding(vocab_size, dim)
        self._transformer = TransformerBlock(
            cnt_h=cnt_h, dim=dim, d_key=d_key, d_value=d_value
        )
        self._head = nn.Linear(dim, vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        x = self._token_embedding(tokens)
        x = self._transformer(x)
        x = self._head(x)
        return x


# %%


def train_step(model: LLM, inputs: Tensor):
    loss = model(inputs).mean()
    loss.backward()


# %%
max_seq_len = 4

model = LLM(
    vocab_size=2,
    cnt_h=1,
    dim=8,
    d_key=1,
    d_value=1,
)

batch_size = 2

inputs = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 0]])
print(inputs)

# train
model.train()
torch.compile(train_step)(model, inputs)
# train_step(model, inputs)
