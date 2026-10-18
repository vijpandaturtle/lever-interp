# lever

Mechanistic interpretability and safety research on vision-language models, building up in stages:

**CLIP → LLaVA → Qwen-VL → Qwen-VL-based driving models**

Each stage adds one piece of machinery: CLIP gives the vision encoder alone, LLaVA adds the
simplest possible bridge (CLIP/SigLIP ViT + MLP projector + LM), Qwen-VL adds native-resolution
patches, 2×2 token merging, M-RoPE (and DeepStack in Qwen3-VL), and the driving model adds a task
where mistakes are safety-critical. Insights and tools carry forward from one stage to the next.

**Library docs:** [docs/index.md](docs/index.md)

## Layout

```
src/lever/
  clip.py        CLIP wrapper: residual lens into joint space, exact per-head/MLP decomposition
  dual_encoders.py  CLIP + SigLIP behind one interface (for cross-encoder comparisons)
  similarity.py  linear CKA (model diffing)
  images.py      synthetic stimuli (text stamping)
  models.py      LLaVA + Qwen-VL behind one VLM interface (vit.{i}, projector, lm.{i}, final_norm)
  hooks.py       record activations / patch module outputs
  lens.py        logit lens for the LM side
experiments/
  clip/          stage 1
  vlm/           stages 2-3 (same scripts, different --model)
notes/           roadmap + research log
tests/           tiny random-weight CLIP / LLaVA / Qwen2.5-VL / Qwen3-VL, no downloads
```

## Setup

```bash
uv sync
uv run pytest
```

Local (16 GB Apple silicon, MPS): `openai/clip-vit-base-patch32` / `-large-patch14`,
`llava-hf/llava-interleave-qwen-0.5b-hf`, `Qwen/Qwen2.5-VL-3B-Instruct`.
Need a CUDA box: `llava-hf/llava-1.5-7b-hf` (the canonical LLaVA), 7B+ Qwen, driving fine-tunes.

## Experiments

| Stage | # | Question | Script |
|---|---|---|---|
| CLIP | 01 | At which layer does the class become readable from CLS? Where spatially? | `experiments/clip/01_residual_lens.py` |
| CLIP | 02 | Which heads/MLPs directly write the class into the embedding? Does removing them flip it? | `experiments/clip/02_direct_effects.py` |
| CLIP | 03 | Typographic attacks: which components let written text beat what's pictured? | `experiments/clip/03_typographic_attack.py` |
| CLIP | 04 | Model diff CLIP vs SigLIP (same ViT-B/16): behavior, embedding geometry, layerwise CKA + probes, text-reading | `experiments/clip/04_clip_vs_siglip_diff.py` |
| VLM | 01 | At which LM layers do image tokens decode to object words, and where? | `experiments/vlm/01_image_token_logit_lens.py` |

See [notes/ROADMAP.md](notes/ROADMAP.md).
