import torch
from torch import nn

from ch03.practice.self_attention_simple import FusedMHA


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["embed_dim"], 4 * cfg["embed_dim"]),
            GELU(),
            nn.Linear(cfg["embed_dim"], 4 * cfg["embed_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class ExampleDeepNeuralNetwork(nn.Module):
    """Example deep neural network with optional residual connections.

    Args:
        layer_sizes: List of layer dimensions (e.g., [input_dim, hidden1, hidden2, ..., output_dim])
        use_shortcut: Whether to use residual/shortcut connections where dimensions match
    """

    def __init__(self, layer_sizes: list[int], use_shortcut: bool = True):
        super().__init__()
        self.use_shortcut = use_shortcut

        # Build layers: Linear -> GELU -> Linear -> GELU -> ...
        # Note: No activation after final layer
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            # Add GELU activation except after the last layer
            if i < len(layer_sizes) - 2:
                layers.append(GELU())

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connections.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        for layer in self.layers:
            # Compute layer output
            layer_output = layer(x)

            # Apply residual connection if enabled and shapes match
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output  # Residual connection
            else:
                x = layer_output

        return x


class ResidualBlock(nn.Module):
    """Residual block with two linear layers and GELU activation.

    Following the pattern used in transformers (e.g., HuggingFace, Meta's LLaMA).
    This is similar to the feedforward block in transformers.

    Args:
        hidden_size: Dimension of the hidden layer
        intermediate_size: Dimension of the intermediate layer (typically 4x hidden_size)
    """

    def __init__(self, hidden_size: int, intermediate_size: int | None = None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        # Two-layer MLP with residual connection
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        # Residual connection
        return hidden_states + residual


from dataclasses import dataclass


@dataclass
class TransformerConfig:
    embed_dim: int
    num_heads: int
    max_seq_len: int
    dropout: float = 0.0
    is_causal: bool = True
    bias: bool = False
    intermediate_size: int | None = None  # Defaults to 4 * embed_dim if None

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.embed_dim


class TransformerBlock(nn.Module):
    def __init__(
        self,
        cfg: dict,
    ):
        super().__init__()
        self.att = FusedMHA(
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            max_seq_len=cfg["max_seq_len"],
            dropout=cfg["dropout"],
            is_causal=cfg["is_causal"],
            bias=cfg["bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["embed_dim"])
        self.norm2 = LayerNorm(cfg["embed_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x) + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x) + shortcut
        return x


class MLP(nn.Module):
    """Standard transformer MLP/FFN block (no internal residual).

    Used by TransformerBlock which handles residual connections externally.
    """

    def __init__(self, embed_dim: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, intermediate_size, bias=bias)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(intermediate_size, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlockV2(nn.Module):
    """Clean Pre-LN transformer block following frontier lab conventions.

    Architecture: x → norm → sublayer → dropout → + residual

    Args:
        cfg: TransformerConfig with model hyperparameters
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(cfg.embed_dim)
        self.attn = FusedMHAWithKV(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            max_seq_len=cfg.max_seq_len,
            dropout=cfg.dropout,
            is_causal=cfg.is_causal,
            bias=cfg.bias,
        )
        self.ffn_norm = nn.LayerNorm(cfg.embed_dim)
        intermediate_size = cfg.intermediate_size or 4 * cfg.embed_dim
        self.ffn = MLP(cfg.embed_dim, intermediate_size, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """Forward pass with Pre-LN and residual connections.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            use_cache: Whether to use KV cache (for inference)

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        x = x + self.dropout(self.attn(self.attn_norm(x), use_cache=use_cache))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


@dataclass
class GPTConfig(TransformerConfig):
    vocab_size: int = 50257  # GPT-2 default
    n_layers: int = 12


GPT_CONFIG_124M = GPTConfig(
    max_seq_len=256,
    embed_dim=768,
    num_heads=12,
    dropout=0.1,
    bias=False,
)


class GPTModel(nn.Module):
    """GPT-2 style decoder-only transformer with Pre-LN and weight tying."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.dropout)
        # nn.ModuleList for extensibility (e.g., gradient checkpointing)
        self.layers = nn.ModuleList(
            [TransformerBlockV2(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = nn.LayerNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        # Weight tying: share weights between token embeddings and lm_head
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token indices of shape (batch_size, seq_len)

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        _, seq_len = input_ids.shape
        tok_emb = self.tok_emb(input_ids)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=input_ids.device))
        x = self.drop(tok_emb + pos_emb)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.lm_head(x)


@torch.no_grad()
def generate_simple_text(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    context_len: int,
) -> torch.Tensor:
    """Greedy decoding: always pick the most likely next token."""
    # input_ids: [seq_len, embed_dim]
    output_ids = input_ids
    for _ in range(max_new_tokens):
        # Truncate to context window
        context = output_ids[:, -context_len:]
        logits = model(context)
        # Get logits for the last position only
        next_token_logits = logits[:, -1, :]
        # Greedy: pick highest probability token
        next_id = next_token_logits.argmax(dim=-1, keepdim=True)
        output_ids = torch.cat([output_ids, next_id], dim=1)
    return output_ids


# Updating for KV-Cache
class FusedMHAWithKV(nn.Module):
    mask: torch.Tensor
    cached_k: torch.Tensor | None
    cached_v: torch.Tensor | None

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        is_causal: bool = True,
        bias: bool = False,
        window_len: int | None = None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            "`embed_dim` must be divisible by `num_heads`"
        )
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_causal = is_causal
        self.window_len = window_len or max_seq_len
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
            persistent=False,
        )
        self.scale = self.head_dim**-0.5
        self.register_buffer("cached_k", None, persistent=False)
        self.register_buffer("cached_v", None, persistent=False)
        self.current_pos = 0

    def forward(self, x: torch.Tensor, use_cache: bool = False):
        batch_size, seq_len, embed_dim = x.shape
        # Fused QKV projection: [B, S, E] -> [B, S, 3, H, D] -> [3, B, H, S, D]
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if use_cache:
            if self.cached_k is None or self.cached_k.size(0) != batch_size:
                self.cached_k = torch.zeros(
                    batch_size,
                    self.num_heads,
                    self.window_len,
                    self.head_dim,
                    device=x.device,
                )
                self.cached_v = torch.zeros_like(self.cached_k)
                self.current_pos = 0

            # Type narrowing: after the check above, cached_k and cached_v are not None
            assert self.cached_k is not None and self.cached_v is not None

            # Once reach the window_len max evict the length of incoming tokens
            if self.current_pos + seq_len > self.window_len:
                overflow = self.current_pos + seq_len - self.window_len
                self.cached_k[:, :, :-overflow, :] = self.cached_k[
                    :, :, overflow:, :
                ].clone()
                self.cached_v[:, :, :-overflow, :] = self.cached_v[
                    :, :, overflow:, :
                ].clone()
                self.current_pos -= overflow
            self.cached_k[:, :, self.current_pos : self.current_pos + seq_len, :] = k
            self.cached_v[:, :, self.current_pos : self.current_pos + seq_len, :] = v
            k = self.cached_k[:, :, : self.current_pos + seq_len, :]
            v = self.cached_v[:, :, : self.current_pos + seq_len, :]
            self.current_pos += seq_len

        attn_weights = q @ k.transpose(-2, -1) * self.scale
        K = attn_weights.size(-1)
        offset = K - seq_len
        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, K, device=x.device),
                diagonal=offset + 1,
            ).bool()
            attn_weights.masked_fill_(
                causal_mask,
                -torch.inf,
            )
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D] -> [B, S, E]
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, embed_dim)
        return self.out_proj(attn_output)

    def reset_cache(self):
        self.cached_k, self.cached_v = None, None
        self.current_pos = 0


class GPTModelV2(nn.Module):
    """GPT-2 style decoder-only transformer with Pre-LN and weight tying."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.dropout)
        # nn.ModuleList for extensibility (e.g., gradient checkpointing)
        self.layers = nn.ModuleList(
            [TransformerBlockV2(cfg) for _ in range(cfg.n_layers)]
        )
        self.current_pos = 0
        self.final_norm = nn.LayerNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        # Weight tying: share weights between token embeddings and lm_head
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            use_cache: Whether to use KV cache for inference (default False for training)

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        _, seq_len = input_ids.shape
        tok_emb = self.tok_emb(input_ids)
        if use_cache:
            pos_ids = torch.arange(
                self.current_pos,
                self.current_pos + seq_len,
                dtype=torch.long,
                device=input_ids.device,
            )
            self.current_pos += seq_len
        else:
            pos_ids = torch.arange(
                seq_len,
                dtype=torch.long,
                device=input_ids.device,
            )
        pos_emb = self.pos_emb(pos_ids)
        x = self.drop(tok_emb + pos_emb)
        for layer in self.layers:
            x = layer(x, use_cache=use_cache)
        x = self.final_norm(x)
        return self.lm_head(x)

    def reset_cache(self):
        for layer in self.layers:
            layer.attn.reset_cache()  # type: ignore[attr-defined]
        self.current_pos = 0


@torch.no_grad()
def generate_simple_text_with_cache(
    model: GPTModelV2,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    context_len: int,
    use_cache: bool = True,
) -> torch.Tensor:
    """Greedy decoding: always pick the most likely next token.

    Args:
        model: GPT model with KV cache support
        input_ids: Token indices of shape (batch_size, seq_len)
        max_new_tokens: Number of tokens to generate
        context_len: Maximum context window size
        use_cache: Whether to use KV cache for faster generation

    Returns:
        Token indices of shape (batch_size, seq_len + max_new_tokens)
    """
    model.reset_cache()  # Always start fresh for generation
    output_ids = input_ids
    for i in range(max_new_tokens):
        if use_cache and i != 0:
            # With cache, only need to process the last token
            context = output_ids[:, -1:]
        else:
            context = output_ids[:, -context_len:]
        logits = model(context, use_cache=use_cache)
        # Get logits for the last position only
        next_token_logits = logits[:, -1, :]
        # Greedy: pick highest probability token
        next_id = next_token_logits.argmax(dim=-1, keepdim=True)
        output_ids = torch.cat([output_ids, next_id], dim=1)
    return output_ids


if __name__ == "__main__":
    import tiktoken

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Hello, I am"
    # Create tensor directly on target device (avoids CPU→GPU copy)
    token_ids = torch.tensor(
        tokenizer.encode(start_context), dtype=torch.long, device=device
    ).unsqueeze(0)
    model = GPTModel(GPT_CONFIG_124M).to(device)
    # Below is important! Otherwise, dropout happens on forward pass
    model.eval()
    output_ids = generate_simple_text(
        model=model,
        input_ids=token_ids,
        max_new_tokens=6,
        context_len=GPT_CONFIG_124M.max_seq_len,
    )
    print(f"Output token ids {output_ids}")
    output_text = tokenizer.decode(output_ids.squeeze(0).tolist())
    print(f"Output text {output_text}")
