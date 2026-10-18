"""Forward-hook primitives that every experiment is built from: read activations, overwrite them.

Why our own instead of TransformerLens / nnsight: TransformerLens doesn't wrap these vision-language
architectures, and nnsight's tracing proxy adds indirection that makes debugging hard for new
model code. Plain `register_forward_hook` works on any nn.Module in any model we'll touch. The value
here is that hooks are always removed (context managers), so a failed experiment can't leave a
model silently patched for the next cell.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager

import torch
from torch import nn


def _main(output):
    # HF blocks disagree on return type (tensor vs (hidden, attn_weights, ...)); the residual/hidden
    # state is always first, and that's the thing interp code wants.
    return output[0] if isinstance(output, tuple) else output


@contextmanager
def record(points: Mapping[str, nn.Module], to_cpu: bool = False) -> Iterator[dict[str, torch.Tensor]]:
    """Capture module outputs from a normal forward pass, so analysis runs on a plain dict.

    Exists so experiments never re-implement hook bookkeeping, and so the model's own forward
    (with its real preprocessing, masking and position handling) is what gets measured rather than
    a re-implementation of it. `to_cpu` matters for 3B+ models, where caching every layer on MPS/GPU
    costs more memory than the model itself. Only the last call's outputs are kept per name.

        with record({k: vlm.named_hook_points()[k] for k in names}) as cache:
            vlm.model(**inputs)
        cache["lm.12"]  # [batch, seq, d_model]
    """
    cache: dict[str, torch.Tensor] = {}
    handles = []
    for name, module in points.items():
        def hook(_m, _inp, out, name=name):
            t = _main(out).detach()
            cache[name] = t.float().cpu() if to_cpu else t
        handles.append(module.register_forward_hook(hook))
    try:
        yield cache
    finally:
        for h in handles:
            h.remove()


@contextmanager
def patch_output(module: nn.Module, fn: Callable[[torch.Tensor], torch.Tensor]) -> Iterator[None]:
    """Intervene on one module's output: the causal half of interp, next to `record`'s observational half.

    Ablation (fn=torch.zeros_like / mean), activation patching (fn=lambda _: clean_cache[name]) and
    steering (fn=lambda x: x + v) are all this one operation. Everything downstream of the module
    recomputes normally, so it measures total effect, unlike the direct-effect-only arithmetic
    in `clip.decompose_image_embedding`.
    """

    def hook(_m, _inp, out):
        if isinstance(out, tuple):
            return (fn(out[0]), *out[1:])
        return fn(out)

    h = module.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()
