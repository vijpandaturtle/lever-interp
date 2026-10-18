"""VLM 01 — What do image tokens "mean" as they flow through the LM? (LLaVA and Qwen-VL)

Logit-lens every image-token position at every LM layer. Two outputs:
  1. top-k decoded tokens per image patch at selected layers (JSON), and
  2. for each concept word (e.g. "car", "pedestrian", "red"), a spatial heatmap of
     P(concept) over the image-token grid, per layer (PNG).

Prior work (Neo et al. 2024, "Towards Interpreting Visual Information Processing in VLMs")
finds object-specific image tokens decode to the object's name in middle-late layers.
Run it on LLaVA first, then Qwen-VL, then the driving fine-tune: do the same layers light up?

    uv run python experiments/vlm/01_image_token_logit_lens.py --image scene.jpg --concepts car pedestrian truck
    uv run python experiments/vlm/01_image_token_logit_lens.py --image scene.jpg --model Qwen/Qwen2.5-VL-3B-Instruct
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from lever import load_vlm, logit_lens, record, top_tokens


def concept_token_ids(vlm, word: str) -> int:
    """Map a concept word to the single vocab id the lens is scored on.

    Leading space: BPE vocabularies encode mid-sentence words with it, and that's the variant a model
    would actually predict. Only the first sub-token is used, so multi-token words ("pedestrian"
    may split) are approximated. Check `top_tokens.json` before trusting a flat heatmap.
    """
    return vlm.processor.tokenizer.encode(" " + word, add_special_tokens=False)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", default="Describe the driving scene.")
    p.add_argument("--model", default="llava-hf/llava-interleave-qwen-0.5b-hf")
    p.add_argument("--concepts", nargs="+", default=["car", "road", "person"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out = Path(args.out or f"outputs/vlm_01_logit_lens/{args.model.split('/')[-1]}")
    out.mkdir(parents=True, exist_ok=True)
    vlm = load_vlm(args.model)
    inputs = vlm.prepare(Image.open(args.image).convert("RGB"), args.prompt)

    layers = {f"lm.{i}": l for i, l in enumerate(vlm.lm_layers)}
    with torch.no_grad(), record(layers) as cache:
        vlm.model(**inputs)

    mask = vlm.image_mask(inputs["input_ids"])[0]
    gh, gw = vlm.image_grid(inputs)
    concept_ids = {c: concept_token_ids(vlm, c) for c in args.concepts}

    tops = {}
    concept_maps = {c: [] for c in args.concepts}  # [layer] -> [gh, gw]
    for i in range(vlm.n_layers):
        lp = logit_lens(vlm, cache[f"lm.{i}"][0, mask])  # [n_img, vocab]
        if i % 4 == 0 or i == vlm.n_layers - 1:
            tops[i] = top_tokens(vlm, lp, args.k)
        for c, tid in concept_ids.items():
            concept_maps[c].append(lp[:, tid].exp().reshape(gh, gw).cpu())

    (out / "top_tokens.json").write_text(json.dumps({"grid": [gh, gw], "layers": tops}, ensure_ascii=False, indent=1))

    # Most common top-1 token over image patches, per layer — quick read of where semantics emerge.
    for i, rows in tops.items():
        firsts = [r[0][0] for r in rows]
        common = sorted(set(firsts), key=firsts.count, reverse=True)[:8]
        print(f"layer {i:>2}: " + ", ".join(f"{t!r}×{firsts.count(t)}" for t in common))

    image = Image.open(args.image).convert("RGB")
    show = [l for l in range(vlm.n_layers) if l % max(1, vlm.n_layers // 6) == 0][:6] + [vlm.n_layers - 1]
    for c, maps in concept_maps.items():
        fig, axes = plt.subplots(1, len(show) + 1, figsize=(3 * (len(show) + 1), 3))
        axes[0].imshow(image)
        axes[0].set_title("input")
        for ax, l in zip(axes[1:], show):
            ax.imshow(maps[l], cmap="magma")
            ax.set_title(f"L{l} P({c!r})\nmax={maps[l].max():.2g}")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"concept_{c.replace(' ', '_')}.png", dpi=120)
        plt.close(fig)
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
