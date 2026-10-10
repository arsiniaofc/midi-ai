"""
MIDI Forge AI - Sampler Module
Provides autoregressive sampling controls:
- Temperature scaling
- Top-k filtering
- Top-p (nucleus) filtering
- Repetition penalty
- Random seed reproducibility
"""

import random
from typing import List, Optional

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


def top_k_top_p_filtering(
    logits: "torch.Tensor",
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -float("Inf")
) -> "torch.Tensor":
    """Filters logits using top-k and/or nucleus (top-p) thresholds."""
    if top_k > 0:
        top_k = min(max(top_k, 1), logits.size(-1))
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


@torch.no_grad()
def sample_sequence(
    model: "torch.nn.Module",
    prompt_ids: List[int],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    eos_id: Optional[int] = None,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> List[int]:
    """Generates tokens autoregressively from a prompt prefix."""
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    model.eval()
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Crop context to model's block_size
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]

        logits, _ = model(idx_cond)
        # Take logits at final position
        logits = logits[:, -1, :]  # shape: (1, vocab_size)

        # Apply repetition penalty on recent tokens
        if repetition_penalty > 1.0 and idx.size(1) > 1:
            recent_tokens = set(idx[0, -32:].tolist())
            for token_id in recent_tokens:
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # Temperature scaling
        if temperature > 0:
            logits = logits / temperature

        # Top-k and Top-p filtering
        filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(filtered_logits, dim=-1)

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)
        next_token_id = next_token.item()

        idx = torch.cat((idx, next_token), dim=1)

        # Stop if EOS is generated
        if eos_id is not None and next_token_id == eos_id:
            break

    return idx[0].tolist()
