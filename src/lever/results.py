"""A saved, self-describing record of one technique run, so tools built on lever (seesaw's
datasheets, a playground) can render results without re-running the model or importing internals.

Why this exists: an interp number is only meaningful alongside the conditions that produced it.
The CLIP vs SigLIP confound in this repo came from preprocessing that nobody recorded next to the
result. So every Result carries a Provenance captured from the live model and processor at run
time, not typed in by hand afterwards.

Why one generic container rather than a class per technique: lever stays free of opinions about
what a "patching result" must contain. A technique names itself in `kind`, puts its inputs in
`params` and its outputs in `data`; consumers dispatch on `kind`. `schema_version` covers the
container only, not what each kind puts inside it.

On disk a Result is a directory: `result.json` (everything except tensors) + `tensors.safetensors`.
Safetensors, not pickle, so loading someone else's result can't execute code.

Caveats:
  - Provenance records what the objects report about themselves. A model built from a config, or
    loaded from a local path, has no hub revision; `revision` is then None, not a guess.
  - Preprocessing is the processor's own `to_dict()`. Values JSON can't hold are stored as their
    `repr`, which is fine to read but not to reconstruct a processor from.
  - Tensors in `data` must be top-level values (not inside lists/dicts) to be saved as tensors;
    nested ones fail loudly rather than being silently stringified.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from lever.core import Model

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Provenance:
    """The conditions a result was produced under. Fields are the ones that have changed a
    conclusion in practice (weights, dtype, preprocessing, which hooks) plus the library versions
    needed to rerun it."""

    model_id: str | None            # hub id or local path, from config._name_or_path
    revision: str | None            # hub commit hash; None for local or config-built models
    model_type: str | None          # config.model_type, i.e. what picked the adapter
    adapter: str
    capabilities: list[str]
    dtype: str
    hooks: list[str]                # canonical names the technique read or patched
    preprocessing: dict[str, Any] | None
    seed: int | None
    versions: dict[str, str]
    created_at: str

    @classmethod
    def capture(cls, m: Model, hooks: list[str], processor: Any = None, seed: int | None = None) -> Provenance:
        """Read provenance off the live objects. Pass the processor that actually produced the
        inputs; without it `preprocessing` is None and consumers should treat it as unknown."""
        cfg = getattr(m.model, "config", None)
        return cls(
            model_id=getattr(cfg, "_name_or_path", None) or None,
            revision=getattr(cfg, "_commit_hash", None),
            model_type=getattr(cfg, "model_type", None),
            adapter=type(m.adapter).__name__,
            capabilities=sorted(m.adapter.capabilities),
            dtype=str(next(m.model.parameters()).dtype).removeprefix("torch."),
            hooks=list(hooks),
            preprocessing=_jsonable(processor.to_dict()) if processor is not None else None,
            seed=seed,
            versions=_versions(),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )


@dataclass
class Result:
    kind: str                       # e.g. "activation_patching"; consumers dispatch on this
    provenance: Provenance
    params: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tensors = {k: v.detach().cpu().contiguous() for k, v in self.data.items() if isinstance(v, torch.Tensor)}
        rest = {k: v for k, v in self.data.items() if k not in tensors}
        meta = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "provenance": asdict(self.provenance),
            "params": self.params,
            "data": rest,
            "tensor_keys": sorted(tensors),
        }
        # No `default=`: a tensor nested inside params/data should fail here, not become a string.
        (path / "result.json").write_text(json.dumps(meta, indent=2))
        if tensors:
            save_file(tensors, path / "tensors.safetensors")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Result:
        path = Path(path)
        meta = json.loads((path / "result.json").read_text())
        if meta["schema_version"] > SCHEMA_VERSION:
            raise ValueError(f"{path} uses schema v{meta['schema_version']}; this lever reads up to v{SCHEMA_VERSION}")
        data = meta["data"]
        if meta["tensor_keys"]:
            data |= load_file(path / "tensors.safetensors")
        return cls(meta["kind"], Provenance(**meta["provenance"]), meta["params"], data, meta["schema_version"])


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("lever-interp", "torch", "transformers"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return repr(x)
