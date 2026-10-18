"""CLIP 02: which heads and MLPs write the class into the image embedding?

Exact decomposition of the image embedding into embed + per-head + per-MLP + bias terms
(Gandelsman et al. 2023). Each term's dot product with a label's text embedding is its *direct*
contribution to that label's logit. We report the contrast top label vs runner-up, then check causally:
remove the top-k contributing components' direct effect and see if the prediction flips.

Expect (per the paper): late-layer attention heads dominate; early layers ~ no direct effect.

    uv run python experiments/clip/02_direct_effects.py --image scene.jpg \
        --labels "a photo of a car" "a photo of a truck" "a photo of a bus"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from lever import decompose_image_embedding, label_probs, load_clip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--model", default="openai/clip-vit-base-patch32")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", default="outputs/clip_02_direct_effects")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    clip = load_clip(args.model)
    pv = clip.pixels(Image.open(args.image).convert("RGB"))
    text = clip.encode_text(args.labels)
    comps, names = decompose_image_embedding(clip, pv)
    comps = comps[0]  # [n_comp, d]
    emb = comps.sum(0)
    norm = emb.norm()

    probs = label_probs(clip, (emb / norm)[None], text)[0]
    a, b = probs.topk(2).indices.tolist()
    print(f"top: {args.labels[a]!r} ({probs[a]:.2f})  runner-up: {args.labels[b]!r} ({probs[b]:.2f})")

    # Direct contribution of each component to logit(a) - logit(b).
    contrib = (clip.logit_scale * comps @ (text[a] - text[b]) / norm).cpu()
    order = contrib.abs().argsort(descending=True)
    print(f"\ntop {args.k} components for {args.labels[a]!r} vs {args.labels[b]!r}  (sum={contrib.sum():.2f})")
    for i in order[: args.k].tolist():
        print(f"  {names[i]:>14}: {contrib[i]:+.3f}")

    # Layer x head heatmap + MLP column.
    L, H = clip.n_layers, clip.n_heads
    grid = torch.zeros(L, H + 1)
    for i, n in enumerate(names):
        if ".H" in n:
            l, h = n[1:].split(".H")
            grid[int(l), int(h)] = contrib[i]
        elif n.endswith(".mlp"):
            grid[int(n[1:].split(".")[0]), H] = contrib[i]
    lim = grid.abs().max()
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(H + 1), [f"H{h}" for h in range(H)] + ["MLP"], fontsize=7)
    ax.set_ylabel("layer")
    ax.set_title(f"direct effect on logit({args.labels[a]!r}) - logit({args.labels[b]!r})", fontsize=9)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out / "direct_effects.png", dpi=120)

    # Causal check on the direct path: drop the top-k positive contributors.
    pos = [i for i in order.tolist() if contrib[i] > 0][: args.k]
    keep = torch.ones(len(names), dtype=torch.bool, device=comps.device)
    keep[pos] = False
    ablated = comps[keep].sum(0)
    probs2 = label_probs(clip, (ablated / ablated.norm())[None], text)[0]
    print(f"\nafter removing top {len(pos)} positive components' direct effect:")
    print("  " + ", ".join(f"{l!r}={p:.2f}" for l, p in zip(args.labels, probs2.tolist())))
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
