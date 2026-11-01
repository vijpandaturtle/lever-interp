"""The model-agnostic interface every lever technique is written against.

Why this exists: a technique (patching sweep, probe, logit lens, CKA) should be written once and run
on GPT-2, LLaVA, Qwen-VL, a VLA model or a dual encoder unchanged. That needs three things no
existing library gives together for all of those model kinds:

  1. Canonical hook names that mean the same thing in every model, across every tower:
         {tower}.blocks.{i}.{site}     e.g. "lm.blocks.12.resid_post", "vision.blocks.3.attn.z"
         {tower}.{site}                e.g. "projector.out", "lm.final_norm"
     Towers: "vision", "text", "audio", "projector", "lm" (any subset). `*` in the layer slot
     expands to every layer. Spectra (the closest prior art) only names "lm.*"; vision towers and
     projectors are where encoder-level questions (CLIP vs SigLIP, typographic attacks) live.
  2. Adapters that map those names onto the real HF modules. They are declarative (module paths in
     a TowerSpec), so adding an architecture is a few lines, and they are selected by
     `config.model_type`, not by checkpoint id, so fine-tunes (e.g. driving models built on
     Qwen-VL) work without registration.
  3. Capability flags, so a technique that needs, say, an unembedding fails with a clear message on
     a model without one instead of producing garbage.

Backend: plain forward hooks on the unmodified model, so the model's own forward (masking, RoPE,
token merging) is what gets measured. TransformerLens fits the same scheme without a separate
backend layer: its HookPoints are nn.Modules, so a TL adapter is just a TowerSpec whose paths point
at them. nnsight's proxy model does not fit this scheme and is deliberately out of scope here.

Caveats:
  - "attn.z" (per-head outputs) is the *input* to the attention output projection, reshaped to
    [..., n_heads, head_dim]. It assumes that projection is a single linear over concatenated
    heads, which holds for every HF architecture targeted so far; adapters must not claim it otherwise.
  - "resid_pre"/"resid_post" are the block's input/output. In models that inject features between
    blocks (Qwen3-VL DeepStack) block i's output is not block i+1's input; such adapters must leave
    out the "strict_residual" capability so techniques that assume it can refuse.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
from torch import nn

Capability = Literal[
    "per_head",            # "attn.z" is valid for every tower that declares n_heads
    "strict_residual",     # block i's resid_post is exactly block i+1's resid_pre
    "unembed",             # Adapter.unembed maps lm residuals to vocabulary logits
    "modality_positions",  # Adapter.modality_masks distinguishes image/audio tokens from text
]


class CapabilityError(RuntimeError):
    """A technique asked for something this model's adapter doesn't provide."""


# ── hook sites ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Site:
    """Where a block-level hook attaches: which submodule of the block, and before or after it.
    `per_head` sites are reshaped to [..., n_heads, head_dim] on the way in and flattened back."""

    target: Literal["block", "attn", "o_proj", "mlp"]
    kind: Literal["pre", "post"]
    per_head: bool = False


BLOCK_SITES: dict[str, Site] = {
    "resid_pre": Site("block", "pre"),
    "resid_post": Site("block", "post"),
    "attn.out": Site("attn", "post"),
    "attn.z": Site("o_proj", "pre", per_head=True),
    "mlp.out": Site("mlp", "post"),
}

# Tower-level sites are named modules on the TowerSpec (all "post"): embed, out, final_norm.
TOWER_SITES = ("embed", "out", "final_norm")

_NAME = re.compile(r"^(?P<tower>[a-z]+)\.(?:blocks\.(?P<layer>\d+|\*)\.(?P<site>[a-z_.]+)|(?P<tsite>[a-z_]+))$")


@dataclass(frozen=True)
class TowerSpec:
    """Declarative description of one tower: module paths from the model root (`blocks`, `embed`,
    `out`, `final_norm`) or from each block (`attn`, `o_proj`, `mlp`). Paths rather than getter
    methods so an adapter is data that can be read at a glance and checked by a generic test."""

    blocks: str | None = None
    attn: str | None = None
    o_proj: str | None = None
    mlp: str | None = None
    n_heads: int | None = None
    embed: str | None = None
    out: str | None = None
    final_norm: str | None = None


# ── adapters ──────────────────────────────────────────────────────────────────


class Adapter:
    """Per-architecture knowledge, and nothing else: where things are, what positions mean, which
    readouts exist. Subclasses set `model_types` and `capabilities` and implement `towers()`;
    the optional methods are only needed for the capabilities they declare."""

    model_types: ClassVar[tuple[str, ...]] = ()
    capabilities: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, model: nn.Module):
        self.model = model

    def towers(self) -> dict[str, TowerSpec]:
        raise NotImplementedError

    def modality_masks(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Bool masks [batch, seq] over lm positions, one per modality ("text", "image", ...).
        Masks, not (first, last) ranges: image tokens aren't contiguous in every model (SmolVLM),
        and batches differ per row."""
        raise CapabilityError(f"{type(self).__name__} does not provide modality_positions")

    def unembed(self, resid: torch.Tensor) -> torch.Tensor:
        """Final norm + unembedding applied to an lm residual: the readout a logit lens needs."""
        raise CapabilityError(f"{type(self).__name__} does not provide unembed")


_REGISTRY: dict[str, type[Adapter]] = {}


def register(cls: type[Adapter]) -> type[Adapter]:
    """Class decorator: make an adapter discoverable by every `config.model_type` it lists."""
    for mt in cls.model_types:
        if mt in _REGISTRY:
            raise ValueError(f"model_type {mt!r} already registered by {_REGISTRY[mt].__name__}")
        _REGISTRY[mt] = cls
    return cls


def adapter_for(model: nn.Module) -> Adapter:
    """Pick the adapter from the model's own config, so any checkpoint of a known architecture works."""
    mt = getattr(getattr(model, "config", None), "model_type", None)
    if mt not in _REGISTRY:
        raise NotImplementedError(f"no lever adapter for model_type={mt!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[mt](model)


# ── the handle ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HookPoint:
    name: str
    module: nn.Module
    kind: Literal["pre", "post"]
    n_heads: int | None  # set only for per-head sites


class Model:
    """A real HF (or TL) model plus its adapter. Not a wrapper with its own forward: you still call
    `m.model(**inputs)`, so nothing measured can come from a wrapper bug.

        m = lever.wrap(hf_model)
        with m.record(["lm.blocks.*.resid_post", "projector.out"]) as cache:
            m.model(**inputs)
        with m.patch({"lm.blocks.5.attn.z": lambda z: z * 0}):
            m.model(**inputs)
    """

    def __init__(self, model: nn.Module, adapter: Adapter | None = None):
        self.model = model
        self.adapter = adapter or adapter_for(model)
        self._towers = self.adapter.towers()

    def has(self, cap: Capability) -> bool:
        return cap in self.adapter.capabilities

    def require(self, *caps: Capability) -> None:
        """Call at the top of a technique: fail before running anything, with the reason."""
        missing = [c for c in caps if not self.has(c)]
        if missing:
            raise CapabilityError(f"{type(self.adapter).__name__} lacks {missing}")

    def n_layers(self, tower: str) -> int:
        spec = self._spec(tower)
        if spec.blocks is None:
            raise KeyError(f"tower {tower!r} has no blocks")
        return len(self.model.get_submodule(spec.blocks))

    def hook_names(self) -> list[str]:
        """Every concrete name this model supports, for discovery and for agents listing options."""
        names = []
        for tower, spec in self._towers.items():
            names += [f"{tower}.{s}" for s in TOWER_SITES if getattr(spec, s)]
            if spec.blocks:
                sites = [s for s, site in BLOCK_SITES.items() if self._site_ok(spec, site)]
                names += [f"{tower}.blocks.{i}.{s}" for i in range(self.n_layers(tower)) for s in sites]
        return names

    def resolve(self, pattern: str) -> list[HookPoint]:
        """Canonical name (with optional `*` layer) -> concrete hook points."""
        m = _NAME.match(pattern)
        if not m:
            raise ValueError(f"bad hook name {pattern!r}; expected 'tower.blocks.i.site' or 'tower.site'")
        tower, spec = m["tower"], self._spec(m["tower"])
        if m["tsite"]:
            path = getattr(spec, m["tsite"], None) if m["tsite"] in TOWER_SITES else None
            if not path:
                raise KeyError(f"{tower!r} has no tower-level site {m['tsite']!r}")
            return [HookPoint(pattern, self.model.get_submodule(path), "post", None)]
        site = BLOCK_SITES.get(m["site"])
        if site is None or not self._site_ok(spec, site):
            raise KeyError(f"{tower!r} blocks have no site {m['site']!r}; known: {list(BLOCK_SITES)}")
        layers = range(self.n_layers(tower)) if m["layer"] == "*" else [int(m["layer"])]
        blocks = self.model.get_submodule(spec.blocks)
        out = []
        for i in layers:
            block = blocks[i]
            mod = block if site.target == "block" else block.get_submodule(getattr(spec, site.target))
            out.append(HookPoint(f"{tower}.blocks.{i}.{m['site']}", mod, site.kind,
                                 spec.n_heads if site.per_head else None))
        return out

    @contextmanager
    def record(self, names: list[str], to_cpu: bool = False) -> Iterator[dict[str, torch.Tensor]]:
        """Cache activations at canonical names during whatever forward pass runs inside the block.
        Per-head sites come back as [..., n_heads, head_dim]. Only the last call per name is kept."""
        cache: dict[str, torch.Tensor] = {}

        def store(hp: HookPoint, t: torch.Tensor):
            t = _split_heads(t.detach(), hp.n_heads)
            cache[hp.name] = t.float().cpu() if to_cpu else t

        with _hooks([hp for n in names for hp in self.resolve(n)], lambda hp, t: store(hp, t)):
            yield cache

    @contextmanager
    def patch(self, edits: Mapping[str, Callable[[torch.Tensor], torch.Tensor]]) -> Iterator[None]:
        """Replace activations during the forward pass: fn(activation) -> new activation, with the
        same shape `record` returns for that name. Ablation, activation patching and steering are
        all this one operation; downstream modules recompute, so effects are total, not direct."""
        fns = {}
        for name, fn in edits.items():
            for hp in self.resolve(name):
                fns[hp] = fn

        def edit(hp: HookPoint, t: torch.Tensor) -> torch.Tensor:
            return _merge_heads(fns[hp](_split_heads(t, hp.n_heads)), hp.n_heads)

        with _hooks(list(fns), edit):
            yield

    def _spec(self, tower: str) -> TowerSpec:
        if tower not in self._towers:
            raise KeyError(f"{type(self.adapter).__name__} has no tower {tower!r}; has {list(self._towers)}")
        return self._towers[tower]

    def _site_ok(self, spec: TowerSpec, site: Site) -> bool:
        if site.target != "block" and getattr(spec, site.target) is None:
            return False
        return not site.per_head or (spec.n_heads is not None and self.has("per_head"))


def wrap(model: nn.Module) -> Model:
    return Model(model)


# ── hook mechanics ────────────────────────────────────────────────────────────


def _split_heads(t: torch.Tensor, n_heads: int | None) -> torch.Tensor:
    return t if n_heads is None else t.unflatten(-1, (n_heads, -1))


def _merge_heads(t: torch.Tensor, n_heads: int | None) -> torch.Tensor:
    return t if n_heads is None else t.flatten(-2)


@contextmanager
def _hooks(points: list[HookPoint], fn: Callable[[HookPoint, torch.Tensor], torch.Tensor | None]):
    """Attach `fn` at each point and always detach, so a failed experiment can't leave the model
    silently patched. `fn` sees the main tensor only: the first element of tuple outputs (HF blocks
    disagree on tensor vs tuple returns), or the hidden_states input of a pre-hook whether it was
    passed positionally or by keyword. A non-None return replaces that tensor."""
    handles = []
    for hp in points:
        if hp.kind == "post":
            def post(_m, _args, out, hp=hp):
                main = out[0] if isinstance(out, tuple) else out
                new = fn(hp, main)
                if new is None:
                    return None
                return (new, *out[1:]) if isinstance(out, tuple) else new
            handles.append(hp.module.register_forward_hook(post))
        else:
            def pre(_m, args, kwargs, hp=hp):
                if args:
                    new = fn(hp, args[0])
                    return None if new is None else ((new, *args[1:]), kwargs)
                new = fn(hp, kwargs["hidden_states"])
                return None if new is None else (args, {**kwargs, "hidden_states": new})
            handles.append(hp.module.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
