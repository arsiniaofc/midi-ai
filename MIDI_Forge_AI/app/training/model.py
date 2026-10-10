"""
MIDI Forge AI - Model Architecture Module
Autoregressive Transformer/GPT architecture combining techniques from aiMIDI and Theremin MIDI (minGPT).
Features causal multi-head self-attention, learnable positional embeddings, Pre-LN residual blocks,
and efficient generation.
"""

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


@dataclass
class GPTConfig:
    vocab_size: int = 512
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GPTConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


if torch is not None:
    class CausalSelfAttention(nn.Module):
        """Multi-head causal self-attention with scaled dot-product."""

        def __init__(self, config: GPTConfig):
            super().__init__()
            assert config.n_embd % config.n_head == 0
            self.n_head = config.n_head
            self.n_embd = config.n_embd
            self.dropout = config.dropout

            # Key, Query, Value projections in a single linear layer
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            # Output projection
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.attn_dropout = nn.Dropout(config.dropout)
            self.resid_dropout = nn.Dropout(config.dropout)

            # Causal mask (tril)
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                )
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, T, C = x.size()  # Batch, Sequence Length, Embedding Dimension

            # Calculate Q, K, V
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

            # Scaled dot-product attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, T, C)

            return self.resid_dropout(self.c_proj(y))


    class MLP(nn.Module):
        """Position-wise Feed Forward Network with GELU."""

        def __init__(self, config: GPTConfig):
            super().__init__()
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.c_fc(x)
            x = self.gelu(x)
            x = self.c_proj(x)
            return self.dropout(x)


    class TransformerBlock(nn.Module):
        """Standard Pre-LN Transformer Block."""

        def __init__(self, config: GPTConfig):
            super().__init__()
            self.ln_1 = nn.LayerNorm(config.n_embd)
            self.attn = CausalSelfAttention(config)
            self.ln_2 = nn.LayerNorm(config.n_embd)
            self.mlp = MLP(config)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x


    class MusicGPT(nn.Module):
        """Full Autoregressive Music Transformer Model."""

        def __init__(self, config: GPTConfig):
            super().__init__()
            self.config = config

            self.transformer = nn.ModuleDict(dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            ))
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

            # Weight tying
            self.lm_head.weight = self.transformer.wte.weight

            # Init weights
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

        def forward(
            self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            device = idx.device
            b, t = idx.size()
            assert t <= self.config.block_size, f"Comprimento da sequência {t} excede o limite {self.config.block_size}"

            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

            # Forward transformer embeddings
            tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
            pos_emb = self.transformer.wpe(pos)  # (1, t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)

            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x)  # (b, t, vocab_size)

            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            return logits, loss

        def get_num_params(self) -> int:
            """Calculates total trainable parameter count."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

else:
    # Minimal fallback placeholder when PyTorch is unavailable
    class MusicGPT:
        def __init__(self, config: GPTConfig):
            self.config = config

        def get_num_params(self) -> int:
            return 0
