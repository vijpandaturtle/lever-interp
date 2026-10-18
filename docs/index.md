# lever

Mechanistic interpretability tooling for vision-language models, built for safety research.

`lever` gives you one way to observe and intervene on the internals of CLIP, SigLIP, LLaVA and
Qwen-VL models, so that **the same experiment runs unchanged as you move from a vision encoder to
the VLMs built on it**. It's the code behind the research in this repo, which works in stages:
CLIP → LLaVA → Qwen-VL → Qwen-based driving models.

> **Status: 0.1, research-grade.** The API will change. Everything is tested on tiny random-weight
> models; see [Supported models](#supported-models) for what has also run on real checkpoints.

**Contents:**
[Why this exists](#why-this-exists) ·
[Install](#install) ·
[Core ideas](#core-ideas) ·
[Quickstart](#quickstart) ·
[API reference](#api-reference) ·
[Supported models](#supported-models) ·
[Caveats](#caveats-and-known-limitations) ·
[Experiments](#experiments) ·
[Roadmap](#roadmap) ·
[Development](#development)

---

## Why this exists

Interpretability tools for language models are mature (TransformerLens, nnsight). For multimodal
models, the tools "cover narrow fields," and there's no unified way to evaluate them
([Lin et al., 2025](https://arxiv.org/abs/2502.17516)). In practice, three things get in the way:

1. **Every VLM family puts its parts in different places.** The vision blocks, the projector and the
   language model live under different module paths in LLaVA and Qwen-VL, and the image-token layout
   differs too. Experiment code ends up hard-wired to one model.
2. **Vision backbones are used two ways.** A dual encoder like CLIP is used directly through its
   final embedding. A VLM throws that readout away and consumes patch tokens from the middle of the
   encoder. Findings about one don't automatically transfer to the other, so you need tools for both.
3. **Safety questions need attacks and interpretability in the same place.** "Why does this model
   follow text written in the image?" means generating the attack *and* tracing it through the model.

`lever` addresses these with architecture-neutral hook names, thin wrappers that never replace the
model's own forward pass, and attack stimuli that live next to the analysis tools.

## Install

Requires Python 3.11–3.12. The project uses [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/vijpandaturtle/lever-interp.git
cd lever-interp
uv sync
```

It runs on CUDA, Apple silicon (MPS) or CPU; the device is picked automatically. Model weights
download from Hugging Face on first use.

Rough memory guide (16 GB Apple-silicon Mac):

| Fits locally | Needs a GPU box |
|---|---|
| CLIP B/16, L/14; SigLIP B/16 | LLaVA-1.5-7B |
| `llava-hf/llava-interleave-qwen-0.5b-hf` | Qwen-VL 7B and larger |
| `Qwen/Qwen2.5-VL-3B-Instruct` (bf16) | most driving fine-tunes |

## Core ideas

### 1. Hook points have the same names across architectures

Every wrapper exposes `named_hook_points()`, a dict from a stable short name to an `nn.Module`.
`"lm.12"` means "output of language-model block 12" whether the model is LLaVA or Qwen, so caches
and results from different models can be compared key by key.

**VLMs** (`VLM.named_hook_points()`):

| Name | What it is | LLaVA | Qwen2.5-VL / Qwen3-VL |
|---|---|---|---|
| `vit.{i}` | vision encoder block *i* | `model.vision_tower[.vision_model].encoder.layers[i]` (CLIP or SigLIP) | `model.visual.blocks[i]` |
| `projector` | maps image features into the LM's space | `model.multi_modal_projector` (2-layer MLP) | `model.visual.merger` (2×2 patch merge + MLP) |
| `lm.{i}` | LM decoder block *i*; its output is the residual stream | `model.language_model.layers[i]` | `model.language_model.layers[i]` |
| `final_norm` | LM final norm, before the unembedding | `model.language_model.norm` | `model.language_model.norm` |

**CLIP** (`CLIP.named_hook_points()`):

| Name | What it is |
|---|---|
| `vit.embed` | residual stream entering layer 0 (after `pre_layrnorm`) |
| `vit.{i}` | residual stream after layer *i* |
| `vit.{i}.attn` | what layer *i*'s attention writes into the residual stream |
| `vit.{i}.mlp` | what layer *i*'s MLP writes into the residual stream |

### 2. Observe with `record`, intervene with `patch_output`

These two context managers are the whole hook API. Both always remove their hooks on exit, so a
crashed experiment can't leave a model silently patched.

- `record(points)` caches module outputs during a normal forward pass.
- `patch_output(module, fn)` replaces a module's output with `fn(output)`. Ablation, activation
  patching and steering are all this one operation.

### 3. Wrappers hold references, never replace the forward pass

`VLM`, `CLIP` and `DualEncoder` are thin records of the Hugging Face model plus the submodules
interp code needs. You always call the real model (`vlm.model(**inputs)`), so nothing you measure
can come from a wrapper bug.

### 4. Two readouts for two uses

- **Language-model side:** `logit_lens` decodes any residual-stream state into vocabulary
  probabilities through the model's own final norm and unembedding.
- **Dual-encoder side:** CLIP has no vocabulary. Its "readout" is cosine similarity with text
  embeddings, so `CLIP.lens` projects any residual state into the joint image–text space instead.

## Quickstart

### Look inside a VLM

```python
import torch
from PIL import Image
from lever import load_vlm, record, patch_output, logit_lens, top_tokens

vlm = load_vlm("llava-hf/llava-interleave-qwen-0.5b-hf")       # or "Qwen/Qwen2.5-VL-3B-Instruct"
inputs = vlm.prepare(Image.open("scene.jpg"), "Is it safe to turn left?")

# Observe: cache the residual stream at every LM layer.
layers = {f"lm.{i}": l for i, l in enumerate(vlm.lm_layers)}
with torch.no_grad(), record(layers) as cache:
    vlm.model(**inputs)

# What do the image tokens "mean" at layer 12?
img = vlm.image_mask(inputs["input_ids"])[0]
print(top_tokens(vlm, logit_lens(vlm, cache["lm.12"][0, img]), k=3)[:5])

# Intervene: remove all visual input and compare the next-token logits.
with torch.no_grad(), patch_output(vlm.projector, torch.zeros_like):
    no_vision = vlm.model(**inputs).logits
```

### Decompose a CLIP prediction into heads

```python
from lever import load_clip, decompose_image_embedding

clip = load_clip("openai/clip-vit-base-patch16")
pv = clip.pixels(Image.open("apple_with_ipod_note.jpg"))
text = clip.encode_text(["a photo of an apple", "a photo of an iPod"])

comps, names = decompose_image_embedding(clip, pv)      # [1, n_components, d], exact sum
emb = comps[0].sum(0)
contrib = clip.logit_scale * comps[0] @ (text[1] - text[0]) / emb.norm()
for i in contrib.argsort(descending=True)[:5]:
    print(names[i], f"{contrib[i]:+.2f}")                   # components pushing toward "iPod"
```

### Compare CLIP and SigLIP

```python
from lever.dual_encoders import load_dual_encoder
from lever.similarity import linear_cka

clip = load_dual_encoder("openai/clip-vit-base-patch16")
siglip = load_dual_encoder("google/siglip-base-patch16-224")
fa, fb = clip.encode_images(images), siglip.encode_images(images)   # list of PIL images
print(linear_cka(fa.patch_means[10], fb.patch_means[10]))            # layer-10 similarity
```

## API reference

### `lever.models`: VLMs

**`load_vlm(model_id="llava-hf/llava-interleave-qwen-0.5b-hf", device=None, dtype=torch.bfloat16, max_pixels=512*28*28) -> VLM`**
Loads a LLaVA or Qwen-VL checkpoint, including driving fine-tunes built on them. `max_pixels` (Qwen
only) caps the image resolution. Qwen keeps native resolution, so an unscaled dashcam frame would
otherwise become thousands of image tokens.

**`wrap_model(model, processor=None) -> VLM`**
Wraps an already-built model: tiny test models, quantized models, PEFT-merged models. It dispatches
on `config.model_type` (`llava`, `qwen2_5_vl`, `qwen3_vl`) and raises `NotImplementedError` for
anything else rather than guessing module paths.

**`class VLM`**

| Member | Description |
|---|---|
| `model`, `processor`, `family` | the HF model, its processor, and `"llava"` or `"qwen"` |
| `vision_blocks`, `projector`, `lm_layers`, `final_norm`, `lm_head` | resolved submodules |
| `image_token_id`, `device`, `n_layers` | basics |
| `named_hook_points()` | see [Core ideas](#1-hook-points-have-the-same-names-across-architectures) |
| `prepare(image, prompt)` | builds inputs with the model's own chat template, including family-specific extras (Qwen's `image_grid_thw`, `mm_token_type_ids`), on the model's device |
| `image_mask(input_ids)` | boolean mask of image-token positions; use it to split any LM activation into vision and text parts |
| `image_grid(inputs)` | `(h, w)` of the LM-side image-token grid for the first image, for reshaping per-token results into a spatial map |
| `decode(ids)` | per-token strings, aligned 1:1 with lens rows |

**`pick_device()`**: CUDA, then MPS, then CPU.

### `lever.hooks`

**`record(points: Mapping[str, nn.Module], to_cpu=False)`**: a context manager that yields a
`dict[str, Tensor]` of each module's primary output (the first element if it returns a tuple).
`to_cpu=True` moves outputs to CPU as fp32, which you need when caching every layer of a 3B+ model.
Only the last forward call's outputs are kept per name.

**`patch_output(module, fn)`**: a context manager that replaces the module's primary output with
`fn(output)` for forward passes inside the block. Common `fn`s:

```python
torch.zeros_like                      # zero ablation
lambda x: mean_act.expand_as(x)      # mean ablation (mean_act precomputed over a dataset)
lambda _: clean_cache["lm.12"]        # activation patching from a clean run
lambda x: x + alpha * direction       # steering
```

Everything downstream recomputes, so this measures **total** effect. Contrast with
`decompose_image_embedding`, which measures **direct** effect only.

### `lever.lens`

**`logit_lens(vlm, resid) -> log-probs [..., vocab]`**: applies the model's final norm and `lm_head`.
It accepts cached CPU/fp32 tensors and casts them to the head's device and dtype. At the last layer
it reproduces the model's own logits exactly (tested).

**`top_tokens(vlm, logprobs, k=5) -> [[(token, prob), ...], ...]`**: human-readable top-k per row.

### `lever.clip`: CLIP internals

**`load_clip(model_id="openai/clip-vit-base-patch32", device=None, dtype=torch.float32) -> CLIP`**
The default is fp32, because the decomposition sums hundreds of terms and its exactness degrades in
bf16.

**`class CLIP`**

| Member | Description |
|---|---|
| `vision`, `layers`, `n_layers`, `n_heads`, `device` | vision transformer and basics |
| `named_hook_points()` | see [Core ideas](#1-hook-points-have-the-same-names-across-architectures) |
| `pixels(images)` | preprocess with the checkpoint's own resize/crop/normalize |
| `encode_text(texts)` / `encode_image(pixel_values)` | unit-norm embeddings `[n, d_joint]` |
| `lens(resid)` | any residual state → `post_layernorm` → projection → unit-norm, i.e. "what would the embedding be if the model stopped here?" Equals `encode_image` on the final CLS (tested). |
| `logit_scale` | learned temperature (≈100 for OpenAI CLIP) |

**`decompose_image_embedding(clip, pixel_values) -> (components [b, n, d_joint], names)`**
Exact additive decomposition of the un-normalized image embedding into `embed`, every attention head
`L{i}.H{h}`, each layer's `L{i}.attn_bias` and `L{i}.mlp`, and the final `ln_bias`.
`components.sum(1)` equals the real embedding (tested). The final LayerNorm is linearized by freezing
its per-sample mean and std, which is exact for that input. Any component's **direct** effect on a
label's logit is `component @ text_embedding`.

**`label_probs(clip, image_emb, text_emb) -> [b, n_labels]`**: zero-shot softmax. It takes embeddings
rather than images, so real, lensed and ablated embeddings are scored the same way.

### `lever.dual_encoders`: CLIP and SigLIP side by side

**`load_dual_encoder(model_id, device=None) -> DualEncoder`**: the family is read from the config
(`clip` or `siglip`), not guessed from the name.

**`class DualEncoder`**

| Member | Description |
|---|---|
| `encode_text(texts)` | unit-norm text embeddings. Handles SigLIP's required `max_length=64` padding; getting that wrong silently degrades embeddings. |
| `encode_images(images, batch_size=32) -> ImageFeatures` | final embeddings plus per-layer states in one pass |
| `label_scores(image_embeds, text_embeds)` | scores **as each model was trained to produce them**: softmax over labels for CLIP, independent sigmoid with the learned bias for SigLIP |

**`ImageFeatures`**: `embeds [n, d_joint]` (unit-norm); `patch_means [n_layers+1, n, d_vis]` (mean
over patch tokens at every layer, the representation comparable across families); and `cls`
(CLIP only, else `None`).

### `lever.similarity`

**`linear_cka(x [n, d1], y [n, d2]) -> float`**: linear CKA
([Kornblith et al., 2019](https://arxiv.org/abs/1905.00414)), invariant to rotation and isotropic
scaling (tested). It's dominated by high-variance directions, so **check for outlier dimensions
first**: CLIP-B/16 has one at layer 8 that flips the conclusion. Report CKA both raw and after
per-dimension standardization.

### `lever.images`: attack stimuli

**`stamp_text(image, text, frac) -> Image`**: renders `text` in a white-backed label below center,
with font height `frac` × image height. It's programmatic so attacks are reproducible and easy to
sweep.

## Supported models

| Family | Architectures | Tested on tiny models | Run on real weights |
|---|---|---|---|
| CLIP | `CLIPModel` | ✅ | ✅ ViT-B/16 (CLIP 04) |
| SigLIP | `SiglipModel` (via `DualEncoder`) | — | ✅ B/16-224 (CLIP 04) |
| LLaVA | `LlavaForConditionalGeneration` with a CLIP or SigLIP tower | ✅ | processor and inputs verified for llava-interleave-0.5B; full forward pending |
| Qwen2.5-VL | `Qwen2_5_VLForConditionalGeneration` | ✅ | pending |
| Qwen3-VL | `Qwen3VLForConditionalGeneration` | ✅ | pending |

Not supported yet: LLaVA-NeXT / OneVision "anyres" tiling (`image_grid` raises an error), cross-attention
VLMs (Llama 3.2 Vision, Flamingo-style), and resampler VLMs (BLIP-2).

## Caveats and known limitations

- **Qwen3-VL DeepStack bypasses `projector`.** Qwen3-VL also injects ViT features into early LM
  layers through `visual.deepstack_merger_list`. `patch_output(vlm.projector, torch.zeros_like)`
  therefore does **not** remove all visual information for Qwen3-VL. A fix is planned.
- **Direct vs total effect.** `decompose_image_embedding` measures direct effects only. A head that
  matters by feeding later layers shows up small; confirm with `patch_output`.
- **CLIP's patch lens is off-distribution.** Patch tokens are never projected during training, so
  `CLIP.lens` maps over patches are suggestive only.
- **Logit lens favors late layers.** Early states aren't expressed in the unembedding's basis.
- **`record` keeps the last call only.** With `generate()`, you get the final decoding step's outputs.
- **SigLIP scores aren't probabilities over labels.** Never port a CLIP confidence threshold to them.

## Experiments

Numbered, self-contained scripts that write to `outputs/` (git-ignored).

| Script | Question |
|---|---|
| `experiments/clip/01_residual_lens.py` | At which layer does the class become readable from CLS? Where spatially? |
| `experiments/clip/02_direct_effects.py` | Which heads and MLPs write the class into the embedding? Does removing them flip it? |
| `experiments/clip/03_typographic_attack.py` | Which components let written text beat what's pictured? |
| `experiments/clip/04_clip_vs_siglip_diff.py` | CLIP vs SigLIP (same ViT-B/16): behavior, geometry, layerwise CKA and probes, text-reading |
| `experiments/vlm/01_image_token_logit_lens.py` | At which LM layers do image tokens decode to object words, and where? |

Findings are logged in [`notes/ROADMAP.md`](https://github.com/vijpandaturtle/lever-interp/blob/main/notes/ROADMAP.md).

## Roadmap

Planned features, mapped to the method taxonomy of
[A Survey on Mechanistic Interpretability for Multi-Modal Foundation Models](https://arxiv.org/abs/2502.17516)
(Lin et al., 2025). Several are inspired by
[LVLM-Interpret](https://arxiv.org/abs/2404.03118) (Stan et al., CVPR 2024 XAI4CV workshop).

| Feature | Survey section | Status |
|---|---|---|
| Logit lens | 4.2 | ✅ |
| CLIP representation decomposition | 4.4 | ✅ |
| SigLIP attention-pooling-head decomposition | 4.4 | next |
| Text↔image attention capture and summaries (LVLM-Interpret's raw-attention view) | 4.4 / tools | next |
| Activation-patching sweeps over (layer × image/text positions) | 4.3 | planned |
| Relevancy maps through the LLM and ViT, plus an **image-vs-text reliance score** | 5.5 | planned |
| Linear-probing helper | 4.1 | planned |
| Attack generators: adversarial patches, visual prompt injection | safety (6.4) | planned |
| Steering vectors / defenses | 4.5 | planned |
| TextSpan-style head descriptions | 5.1 | planned |
| Sparse autoencoders | 4.6 | later |
| Interactive front end (Gradio, as in LVLM-Interpret) | tools | later |
| Qwen3-VL DeepStack-aware visual ablation | — | planned fix |

## Development

```bash
uv run pytest
```

Tests build tiny random-weight CLIP, LLaVA, Qwen2.5-VL and Qwen3-VL models, so they need no downloads
and finish in a few seconds. Key invariants under test:

- the logit lens at the last layer reproduces the model's logits;
- the CLIP decomposition sums exactly to the real embedding;
- `CLIP.lens` on the final CLS equals `encode_image`;
- ablating the projector changes the output;
- CKA is 1 for identical and rotated/scaled features, and low for unrelated ones.

Conventions: docstrings say *why* a function exists and how far to trust its output, not just what
it does. New experiments get a number, write to `outputs/`, and log their findings in the roadmap.
