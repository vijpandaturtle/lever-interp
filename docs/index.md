# lever

Interpretability tooling for vision-language models, built for safety research.

`lever` lets you look inside and intervene on CLIP, SigLIP, LLaVA and Qwen-VL with one small API,
so the same experiment runs on a vision encoder and on the VLMs built on top of it.

> **Status: 0.1, research-grade.** The API will change.

## Install

Requires Python 3.11–3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/vijpandaturtle/lever-interp.git
cd lever-interp
uv sync
```

Runs on CUDA, Apple silicon (MPS) or CPU. Weights download from Hugging Face on first use.

## Quickstart

### Look inside a VLM

```python
import torch
from PIL import Image
from lever import load_vlm, record, patch_output, logit_lens, top_tokens

vlm = load_vlm("llava-hf/llava-interleave-qwen-0.5b-hf")   # or "Qwen/Qwen2.5-VL-3B-Instruct"
inputs = vlm.prepare(Image.open("scene.jpg"), "Is it safe to turn left?")

# Observe: cache every LM layer's output.
layers = {f"lm.{i}": l for i, l in enumerate(vlm.lm_layers)}
with torch.no_grad(), record(layers) as cache:
    vlm.model(**inputs)

# What do the image tokens decode to at layer 12?
img = vlm.image_mask(inputs["input_ids"])[0]
print(top_tokens(vlm, logit_lens(vlm, cache["lm.12"][0, img]), k=3)[:5])

# Intervene: remove all visual input.
with torch.no_grad(), patch_output(vlm.projector, torch.zeros_like):
    no_vision = vlm.model(**inputs).logits
```

### Break a CLIP prediction into heads

```python
from lever import load_clip, decompose_image_embedding

clip = load_clip("openai/clip-vit-base-patch16")
pv = clip.pixels(Image.open("apple.jpg"))
comps, names = decompose_image_embedding(clip, pv)   # sums exactly to the real embedding
```

## The API in four functions

| Function | What it does |
|---|---|
| `record(points)` | Context manager that caches module outputs during a forward pass. |
| `patch_output(module, fn)` | Context manager that replaces a module's output with `fn(output)`. Ablation, patching and steering are all this. |
| `logit_lens(vlm, resid)` | Decodes any LM residual state into vocabulary log-probs. |
| `decompose_image_embedding(clip, pv)` | Splits a CLIP image embedding into per-head and per-MLP contributions. |

Hooks are always removed on exit, and wrappers never replace the model's own forward pass: you
always call the real Hugging Face model.

## Supported models

| Family | Status |
|---|---|
| CLIP | ✅ tested, run on real weights |
| SigLIP | ✅ run on real weights (via `lever.dual_encoders`) |
| LLaVA | ✅ tested on tiny models |
| Qwen2.5-VL, Qwen3-VL | ✅ tested on tiny models |

## Know before you trust

- On Qwen3-VL, zeroing `projector` does **not** remove all visual input (DeepStack bypasses it).
- `decompose_image_embedding` measures **direct** effects only; confirm with `patch_output`.
- `record` keeps only the last forward call.

## Development

```bash
uv run pytest
```

Tests use tiny random-weight models, so they need no downloads.
