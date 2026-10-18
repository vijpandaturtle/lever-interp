"""Logit lens for the LM half of a VLM.

The cheapest first look at what an intermediate state "means": pretend the model stopped there and
decode it. For VLMs it's especially useful at *image-token* positions, which have no next-token
target, so it's the only direct way to ask "does this patch's representation look like the word
'car' yet?" It's biased toward late layers (early states aren't in the unembedding's basis);
tuned lenses fix that, and are a later addition if the plain lens turns out too noisy.
"""

from __future__ import annotations

import torch

from lever.models import VLM


@torch.no_grad()
def logit_lens(vlm: VLM, resid: torch.Tensor) -> torch.Tensor:
    """resid [..., d_model] -> log-probs [..., vocab], through the model's own final norm + lm_head.

    Casts to the head's dtype and device so cached CPU/fp32 activations can be fed back in; without
    that, `record(to_cpu=True)` output wouldn't be lens-able. Applied to the last layer it exactly
    reproduces the model's logits (tested), which is the check that we picked the right norm.
    """
    x = resid.to(vlm.device, dtype=next(vlm.lm_head.parameters()).dtype)
    return vlm.lm_head(vlm.final_norm(x)).float().log_softmax(-1)


def top_tokens(vlm: VLM, logprobs: torch.Tensor, k: int = 5) -> list[list[tuple[str, float]]]:
    """logprobs [n, vocab] -> per-row [(token_str, prob)]. For printing and JSON dumps: the point of
    a lens is being human-readable, and raw ids aren't."""
    vals, idx = logprobs.topk(k, dim=-1)
    return [list(zip(vlm.decode(i), v.exp().tolist())) for i, v in zip(idx, vals)]
