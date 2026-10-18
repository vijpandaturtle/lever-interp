"""Representational similarity between two models (or two layers) on the same inputs.

Why: model diffing needs a way to compare representations whose dimensions don't correspond:
different widths, bases, training runs. Linear CKA (Kornblith et al. 2019) asks "do these two
feature sets induce the same similarity structure over the inputs?" It is invariant to rotations
and isotropic scaling, so it compares *what* is represented, not *how it's coordinatized*.
Used now for CLIP vs SigLIP, and later for base Qwen-VL vs its driving fine-tune.
"""

import torch


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """x [n, d1], y [n, d2], rows = the same n inputs -> similarity in [0, 1].

    Caveat: dominated by high-variance directions, so a few outlier dimensions (common in
    transformer residual streams) can drive the score. Compare trends, not absolute values.
    """
    x = x.double() - x.double().mean(0)
    y = y.double() - y.double().mean(0)
    # ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F): feature-space form, never builds n x n Gram matrices.
    cross = (y.T @ x).norm() ** 2
    return float(cross / ((x.T @ x).norm() * (y.T @ y).norm()))
