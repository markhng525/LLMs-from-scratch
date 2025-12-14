import torch
import torch.nn.functional as F
from torch import nn


def attention(embedded_tokens: torch.Tensor):
    attn_scores = embedded_tokens @ embedded_tokens.T
    attn_weights = F.softmax(attn_scores, dim=-1)
    # Check weights all sum to 1
    print(attn_weights.sum(dim=-1))
    context = attn_weights @ embedded_tokens
    return context


class SelfAttentionV2(nn.Module):
    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor):
        query = self.W_q(x)  # seq_len, d_in x d_in, d_out = seq_len x d_out
        key = self.W_k(x)
        value = self.W_v(x)
        attn_weights = F.softmax(
            query @ key.T / key.shape[-1] ** 0.5, dim=-1
        )  # seq_len x d_in, d_in, seq_len
        return (
            attn_weights @ value
        )  # seq_len, seq_len x seq_len x d_out = seq_len x d_outh


class CausalAttentionV1(nn.Module):
    mask: torch.Tensor

    def __init__(
        self,
        d_in: int,
        d_out: int,
        ctx_len: int,
        dropout: float,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.d_out = d_out
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(ctx_len, ctx_len), diagonal=1)
        )

    def forward(self, x: torch.Tensor):
        b, num_tokens, d_in = x.shape
        keys: torch.Tensor = self.W_k(x)
        queries: torch.Tensor = self.W_q(x)
        values: torch.Tensor = self.W_v(x)
        attn_scores = queries @ keys.transpose(1, 2)
        mask = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.d_out**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return attn_weights @ values


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        ctx_len: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttentionV1(d_in, d_out, ctx_len, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )
        self.out_proj = nn.Linear(d_out * num_heads, d_out * num_heads)

    def forward(self, x):
        return self.out_proj(torch.cat([head(x) for head in self.heads], dim=-1))


class MultiheadAttention(nn.Module):
    mask: torch.Tensor

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        is_causal: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            "`embed_dim` must be divisible by `num_heads`"
        )
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_causal = is_causal
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
            persistent=False,
        )
        # pre-compute scaling factor
        self.scale = self.head_dim**-0.5

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        # batch_size, seq_len, embed_dim -> batch_size, seq_len, num_heads, head_dim
        # q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # transpose so that we get [batch_size, num_heads, seq_len, seq_len]
        # [batch_size, num_heads, seq_len, head_dim] x [batch_size, num_heads, head_dim, seq_len]
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        attn_weights: torch.Tensor = q @ k.transpose(-2, -1) * self.scale

        # Now mask out
        if self.is_causal:
            attn_weights.masked_fill_(self.mask[:seq_len, :seq_len], -torch.inf)

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [batch_size, num_heads, seq_len, seq_len] x [batch_size, num_heads, seq_len, head_dim]
        # attn_output = (attn_weights @ v).reshape(batch_size, seq_len, embed_dim)
        attn_output: torch.Tensor = attn_weights @ v
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, embed_dim)
        )
        return self.out_proj(attn_output)


class FusedMHA(nn.Module):
    mask: torch.Tensor

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        is_causal: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            "`embed_dim` must be divisible by `num_heads`"
        )
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_causal = is_causal
        # Stack 3 Tensors so that a single batch [1, embed_dim] -> [3, embed_dim]
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
            persistent=False,
        )
        # pre-compute scaling factor
        self.scale = self.head_dim**-0.5

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        # batch_size, seq_len, embed_dim -> batch_size, seq_len, num_heads, head_dim
        # q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # transpose so that we get [batch_size, num_heads, seq_len, seq_len]
        # [batch_size, num_heads, seq_len, head_dim] x [batch_size, num_heads, head_dim, seq_len]

        # [batch_size, seq_len, 3 * embed_dim] = [batch_size, seq_len, 3 * (num_heads x head_dim)] -> [batch_size, seq_len, 3*num_heads, 3*head_dim]
        # B, S, 3 * E -> B, S, 3, H, D
        qkv: torch.Tensor = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        # Permute so that we have [3, B, H, S, D] - (2, 0, 3, 1, 4)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn_weights = q @ k.transpose(-2, -1) * self.scale
        # Now mask out
        if self.is_causal:
            attn_weights.masked_fill_(self.mask[:seq_len, :seq_len], -torch.inf)

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [batch_size, num_heads, seq_len, seq_len] x [batch_size, num_heads, seq_len, head_dim]
        # attn_output = (attn_weights @ v).reshape(batch_size, seq_len, embed_dim)
        attn_output: torch.Tensor = attn_weights @ v
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, embed_dim)
        )
        return self.out_proj(attn_output)


class FusedMHAEinsum(nn.Module):
    mask: torch.Tensor

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        is_causal: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            "`embed_dim` must be divisible by `num_heads`"
        )
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_causal = is_causal
        # Stack 3 Tensors so that a single batch [1, embed_dim] -> [3, embed_dim]
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
            persistent=False,
        )
        # pre-compute scaling factor
        self.scale = self.head_dim**-0.5

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        # batch_size, seq_len, embed_dim -> batch_size, seq_len, num_heads, head_dim
        # q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # transpose so that we get [batch_size, num_heads, seq_len, seq_len]
        # [batch_size, num_heads, seq_len, head_dim] x [batch_size, num_heads, head_dim, seq_len]

        # [batch_size, seq_len, 3 * embed_dim] = [batch_size, seq_len, 3 * (num_heads x head_dim)] -> [batch_size, seq_len, 3*num_heads, 3*head_dim]
        # B, S, 3 * E -> B, S, 3, H, D
        qkv: torch.Tensor = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        # Permute so that we have [3, B, H, S, D] - (2, 0, 3, 1, 4)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn_weights = torch.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        # Now mask out
        if self.is_causal:
            attn_weights.masked_fill_(self.mask[:seq_len, :seq_len], -torch.inf)

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [batch_size, num_heads, seq_len, seq_len] x [batch_size, num_heads, seq_len, head_dim]
        # attn_output = (attn_weights @ v).reshape(batch_size, seq_len, embed_dim)
        attn_output: torch.Tensor = torch.einsum("bhij,bhjk->bhik", attn_weights, v)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, embed_dim)
        )
        return self.out_proj(attn_output)


if __name__ == "__main__":
    # batch_size = 2, ctx_len = 6, embed_dim = 4
    batch = torch.randn(2, 6, 4)
    batch_size, seq_len, embed_dim = batch.shape
    mha = MultiheadAttention(embed_dim, 2, seq_len)
    ctx_vecs = mha(batch)
    print(ctx_vecs)
    print(f"`ctx_vecs` shape: {ctx_vecs.shape}")
