"""Behavioural counterfactuals: does the prediction move when one thing about the input changes?

Why this exists: before asking *which* heads carry a shortcut (direct effects, ablation,
patching), you need to know the model is sensitive to it at all. That's a plain comparison:
score the clean inputs, score the same inputs with one thing edited (text stamped on, background
swapped), and count how often the prediction changes. Run it with a control edit too (the true
content changed, the suspect left alone) and the two flip rates side by side say whether the
model follows the suspect or the content.

One function for CLIP and VLMs: a scorer maps one input to logits over a fixed set of candidate
answers (CLIP's labels; a VLM's answer tokens), so "prediction" is always the argmax over
candidates and "flip" is that argmax changing. `vlm_scorer` and `clip_scorer` build the two.

Caveats:
  - This is description, not mechanism: a high flip rate says the model is sensitive to the edit,
    not how. Whether an edit changed only the suspect is the caller's job to check (a stamp also
    hides part of the object).
  - Candidates are a closed set. A VLM that would answer something outside them still gets an
    argmax over them; pick candidates that cover the answers the edit could plausibly cause.
  - A VLM candidate is its answer's first token, as everywhere else in lever.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from lever.clip import CLIP, label_probs
from lever.models import VLM

Scorer = Callable[[Any], torch.Tensor]


@dataclass
class Condition:
    """One edited set, compared item by item with the clean set."""

    predictions: torch.Tensor   # [n] argmax candidate per item
    flipped: torch.Tensor       # [n] bool: prediction differs from the clean one
    prob_drop: torch.Tensor     # [n] clean prediction's probability, clean minus edited

    @property
    def flip_rate(self) -> float:
        return self.flipped.float().mean().item()

    @property
    def mean_prob_drop(self) -> float:
        return self.prob_drop.mean().item()

    def flip_rate_to(self, candidate: int) -> float:
        """How often the edit moves the prediction to one candidate, e.g. the label a typographic
        attack wrote on the image. A high flip rate spread over many candidates is noise; one
        concentrated on the stamped label is the shortcut."""
        return (self.flipped & (self.predictions == candidate)).float().mean().item()


@dataclass
class CounterfactualEval:
    clean_predictions: torch.Tensor   # [n]
    clean_probs: torch.Tensor         # [n, n_candidates]
    conditions: dict[str, Condition]

    def summary(self) -> dict[str, dict[str, float]]:
        """JSON-safe per-condition numbers, for results and reports."""
        return {name: {"flip_rate": round(c.flip_rate, 4), "mean_prob_drop": round(c.mean_prob_drop, 4)}
                for name, c in self.conditions.items()}


@torch.no_grad()
def counterfactual_eval(score: Scorer, clean: Sequence, conditions: Mapping[str, Sequence]) -> CounterfactualEval:
    """Score `clean` and each edited set with `score`, pairing items by position.

    `conditions` maps a name ("shortcut", "control") to inputs the same length as `clean`, item i
    being clean item i with one thing changed.
    """
    if not clean:
        raise ValueError("no clean inputs")
    for name, items in conditions.items():
        if len(items) != len(clean):
            raise ValueError(f"condition {name!r} has {len(items)} inputs for {len(clean)} clean ones; "
                             "each edited input pairs with the clean input at the same position")
    base = torch.stack([score(x).float().softmax(-1) for x in clean])
    pred = base.argmax(-1)
    rows = torch.arange(len(clean))
    out = {}
    for name, items in conditions.items():
        probs = torch.stack([score(x).float().softmax(-1) for x in items])
        if probs.shape != base.shape:
            raise ValueError(f"condition {name!r} scored {probs.shape[-1]} candidates, clean scored {base.shape[-1]}")
        p = probs.argmax(-1)
        out[name] = Condition(p, p != pred, base[rows, pred] - probs[rows, pred])
    return CounterfactualEval(pred, base, out)


def vlm_scorer(vlm: VLM, candidate_ids: Sequence[int]) -> Scorer:
    """Score prepared VLM inputs (e.g. from `vlm.prepare`) by the answer position's logits at
    `candidate_ids`."""
    ids = torch.tensor(list(candidate_ids))

    def score(inputs: dict) -> torch.Tensor:
        return vlm.model(**inputs).logits[0, -1].float().cpu()[ids]

    return score


def clip_scorer(clip: CLIP, labels: Sequence[str] | torch.Tensor) -> Scorer:
    """Score preprocessed images (`clip.pixels(image)`) against candidate labels, as text or as
    precomputed unit-norm text embeddings."""
    text = labels if isinstance(labels, torch.Tensor) else clip.encode_text(list(labels))

    def score(pixel_values: torch.Tensor) -> torch.Tensor:
        return label_probs(clip, clip.encode_image(pixel_values), text)[0].float().cpu().log()

    return score
