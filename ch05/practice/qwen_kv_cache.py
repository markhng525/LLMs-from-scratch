# Optimized KV Cache Scaled Dot Product Attention
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TransformerConfig:
    embed_dim: int
    n_head: int
    max_seq_len: int
    dropout: float = 0.0
    is_causal: bool = True
    bias: bool = False
    intermediate_size: int | None = None  # Defaults to 4 * embed_dim if None

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.embed_dim


@dataclass
class Qwen3Config(TransformerConfig):
    vocab_size: int = 151_936
    n_layers: int = 28
    n_kv_group: int = 4
    qk_norm: bool = False
    dtype: torch.dtype | None = None
    head_dim: int | None = None
    rope_base: float = 10_000
    tie_embeddings: bool = True


class MHAAttention(nn.Module):
    k_cache: torch.Tensor | None
    v_cache: torch.Tensor | None

    # B, S, E, dropout, is_causal
    def __init__(
        self,
        max_seq_len: int,
        n_head: int,
        embed_dim: int,
        dropout: float = 0.0,
        is_causal: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert embed_dim % n_head == 0, (
            "num_head must least common multiple of embed_dim"
        )
        self.n_head = n_head
        self.head_dim = embed_dim // n_head
        self.is_causal = is_causal
        self.max_seq_len = max_seq_len
        # E -> 3 E -> reshape -> 3, E -> 3, H, D
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.dropout = dropout
        # kv cache
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)
        self.cur_pos = 0

    def forward(self, x: torch.Tensor, use_cache: bool) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        qkv: torch.Tensor = self.qkv(x).reshape(
            batch_size, seq_len, 3, self.n_head, self.head_dim
        )
        # (3, B, H, S, D)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if use_cache:
            if self.k_cache is None:
                # Allocate cache for max_seq_len, not just current seq_len
                self.k_cache = torch.zeros(
                    batch_size,
                    self.n_head,
                    self.max_seq_len,
                    self.head_dim,
                    device=x.device,
                    dtype=x.dtype,
                )
                self.v_cache = torch.zeros_like(self.k_cache)

            assert self.k_cache is not None and self.v_cache is not None

            self.k_cache[:, :, self.cur_pos : self.cur_pos + seq_len, :] = k
            self.v_cache[:, :, self.cur_pos : self.cur_pos + seq_len, :] = v
            k = self.k_cache[:, :, : self.cur_pos + seq_len]
            v = self.v_cache[:, :, : self.cur_pos + seq_len]
            self.cur_pos += seq_len

        attn_weights = F.scaled_dot_product_attention(
            query=q, key=k, value=v, dropout_p=self.dropout, is_causal=True
        )
        # Transpose back to (B, S, H, D) then reshape to (B, S, E)
        attn_weights = attn_weights.transpose(1, 2).reshape(
            batch_size, seq_len, embed_dim
        )
        return self.out_proj(attn_weights)

    def reset_cache(self) -> None:
        self.k_cache = None
        self.v_cache = None
        self.cur_pos = 0


# FFN w/ SwiGLU
class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, intermediate_dim: int, bias: bool) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, intermediate_dim, bias)
        self.up_proj = nn.Linear(embed_dim, intermediate_dim, bias)
        self.down_proj = nn.Linear(intermediate_dim, embed_dim, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


def compute_rope_params(
    head_dim: int,
    theta_base: float = 10_000,
    context_len: int = 4096,
    dtype=torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 2 == 0, "embed_dim must be even"
    # \theta_d = \theta_b ** (-d / D)
    freqs = torch.arange(0, head_dim, 2, dtype=dtype) / head_dim
    inv_freq = theta_base**-freqs
    positions = torch.arange(context_len, dtype=dtype)
    # (context_len, 1) x (1, head_dim // 2) = (context_len, head_dim // 2)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    # (context_len, head_dim)
    angles = torch.cat([angles, angles], dim=1)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0
) -> torch.Tensor:
    _, _, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "head_dim must be even"

    # (batch_size, n_heads, seq_len, head_dim // 2)
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    # (1, 1, seq_len, head_dim)
    cos = cos[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)

    # (batch_size, n_heads, seq_len, head_dim) == x.shape
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


class GroupedQueryAttention(nn.Module):
    q_norm: nn.Module | None
    k_norm: nn.Module | None

    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        n_kv_groups: int,
        head_dim: int | None = None,
        bias: bool = False,
        qk_norm: bool = False,
        dtype=None,
    ) -> None:
        super().__init__()
        assert n_head % n_kv_groups == 0, "n_head must be divisible by n_kv_groups"

        self.n_head = n_head
        self.n_kv_groups = n_kv_groups
        self.group_size = n_head // n_kv_groups

        if head_dim is None:
            assert embed_dim % n_head == 0, "`embed_dim` must be divisible by `n_head`"
            head_dim = embed_dim // n_head

        self.head_dim = head_dim
        # Q stays: (B, S, E)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias, dtype=dtype)
        # K, V reduced to: (B, S, g x D) where g x D < E
        self.k_proj = nn.Linear(embed_dim, n_kv_groups * head_dim, bias, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, n_kv_groups * head_dim, bias, dtype=dtype)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias, dtype=dtype)

        if qk_norm:
            self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos=0,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, embed_dim = x.shape
        # (B, S, H, D) -> (B, H, S, D)
        q = (
            self.q_proj(x)
            .reshape(batch_size, seq_len, self.n_head, self.head_dim)
            .transpose(1, 2)
        )
        # (B, S, G, D) -> (B, S, G, D)
        k = (
            self.k_proj(x)
            .reshape(batch_size, seq_len, self.n_kv_groups, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .reshape(batch_size, seq_len, self.n_kv_groups, self.head_dim)
            .transpose(1, 2)
        )
        if self.q_norm:
            q = self.q_norm(q)
        if self.k_norm:
            k = self.k_norm(k)

        # Apply RoPE with correct offset
        q = apply_rope(q, cos, sin, offset=start_pos)
        k = apply_rope(k, cos, sin, offset=start_pos)

        # Update KV cache
        if kv_cache:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=-2)
            v = torch.cat([v_cache, v], dim=-2)

        next_kv_cache = (k, v)
        # (B, G, S, D) repeat G dim to match H
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        attn_weights: torch.Tensor = q @ k.transpose(-2, -1)
        attn_weights = attn_weights / (self.head_dim**0.5)
        attn_weights = attn_weights.masked_fill(mask, -torch.inf)
        attn_outputs = F.softmax(attn_weights, dim=-1) @ v
        attn_outputs = attn_outputs.transpose(1, 2).reshape(
            batch_size, seq_len, embed_dim
        )
        return self.out_proj(attn_outputs), next_kv_cache


class TranformerBlock(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.embed_dim, eps=1e-6)
        self.attn = GroupedQueryAttention(
            embed_dim=cfg.embed_dim,
            n_head=cfg.n_head,
            n_kv_groups=cfg.n_kv_group,
            qk_norm=cfg.qk_norm,
            dtype=cfg.dtype,
        )
        self.ffn_norm = nn.RMSNorm(cfg.embed_dim, eps=1e-6)
        self.ffn = FeedForward(cfg.embed_dim, 4 * cfg.embed_dim, cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x,
        mask: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, next_kv_cache = self.attn(
            self.attn_norm(x), mask, cos, sin, start_pos=start_pos, kv_cache=kv_cache
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, next_kv_cache


class KVCache:
    def __init__(self, n_layers: int):
        self.cache: list[torch.Tensor | None] = [None] * n_layers

    def get(self, layer_idx: int):
        return self.cache[layer_idx]

    def update(self, layer_idx: int, value: torch.Tensor):
        self.cache[layer_idx] = value

    def get_all(self):
        return self.cache

    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None


class Qwen3(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, dtype=cfg.dtype)
        # transformer blocks
        self.layers = nn.ModuleList([TranformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.dropout = nn.Dropout(cfg.dropout)
        self.final_norm = nn.RMSNorm(cfg.embed_dim, eps=1e-6)
        self.lm_head = nn.Linear(
            cfg.embed_dim, cfg.vocab_size, cfg.bias, dtype=cfg.dtype
        )
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_embed.weight

        if cfg.head_dim:
            head_dim = cfg.head_dim
        else:
            head_dim = cfg.embed_dim // cfg.n_head

        cos, sin = compute_rope_params(head_dim, cfg.rope_base, cfg.max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.cur_pos = 0

    def forward(
        self, x: torch.Tensor, kv_caches: KVCache | None = None
    ) -> torch.Tensor:
        _, seq_len = x.shape

        x = self.tok_embed(x).to(x.device)

        # Track position before updating
        start_pos = self.cur_pos

        if kv_caches:
            # Create causal mask for current position
            # Full mask size is (cur_pos + seq_len) x (cur_pos + seq_len)
            # But we only need the slice for the new tokens
            mask = torch.triu(
                torch.ones(
                    self.cur_pos + seq_len,
                    self.cur_pos + seq_len,
                    device=x.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )[self.cur_pos : self.cur_pos + seq_len, : self.cur_pos + seq_len]
            self.cur_pos += seq_len
        else:
            self.cur_pos = 0
            start_pos = 0
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
        mask = mask[None, None, ...]

        for i, layer in enumerate(self.layers):
            layer_kv_cache = kv_caches.get(i) if kv_caches else None
            x, new_kv_cache = layer(
                x,
                mask,
                self.cos,
                self.sin,
                start_pos=start_pos,
                kv_cache=layer_kv_cache,
            )
            if kv_caches:
                kv_caches.update(i, new_kv_cache)

        return self.lm_head(self.final_norm(x).to(self.cfg.dtype))

    def reset_cache(self):
        self.cur_pos = 0
