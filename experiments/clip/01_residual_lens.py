"""CLIP 01: at which layer does the image's class become readable?

Push the CLS residual stream after every ViT layer through the final LN + projection, then
zero-shot classify against your labels. Also project each *patch* token the same way to get a
per-layer spatial map of which patches align with the top label.

    uv run python experiments/clip/01_residual_lens.py --image scene.jpg \
        --labels "a photo of a car" "a photo of a pedestrian" "a photo of a traffic light" "a photo of an empty road"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from lever import label_probs, load_clip, record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--model", default="openai/clip-vit-base-patch32")
    p.add_argument("--out", default="outputs/clip_01_residual_lens")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    clip = load_clip(args.model)
    image = Image.open(args.image).convert("RGB")
    pv = clip.pixels(image)
    text = clip.encode_text(args.labels)

    points = {"embed": clip.vision.pre_layrnorm} | {f"L{i}": l for i, l in enumerate(clip.layers)}
    with torch.no_grad(), record(points) as cache:
        clip.vision(pixel_values=pv)
    names = list(points)

    cls_probs = torch.stack([label_probs(clip, clip.lens(cache[n][:, 0]), text)[0] for n in names]).cpu()
    final = cls_probs[-1]
    print("final:", ", ".join(f"{l!r}={p:.2f}" for l, p in zip(args.labels, final.tolist())))
    for n, row in zip(names, cls_probs):
        print(f"{n:>6}: top={args.labels[int(row.argmax())]!r} p={row.max():.2f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    for j, lab in enumerate(args.labels):
        ax.plot(range(len(names)), cls_probs[:, j], marker="o", label=lab)
    ax.set_xticks(range(len(names)), names, rotation=45)
    ax.set_ylabel("zero-shot P(label) via lens on CLS")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "cls_lens.png", dpi=120)

    # Patch-token lens: cosine of each patch (projected) with the top label's text embedding.
    top = int(final.argmax())
    side = int((cache["L0"].shape[1] - 1) ** 0.5)
    show = names[-6:]
    fig, axes = plt.subplots(1, len(show) + 1, figsize=(3 * (len(show) + 1), 3))
    axes[0].imshow(image)
    axes[0].set_title("input")
    for ax, n in zip(axes[1:], show):
        sims = (clip.lens(cache[n][0, 1:]) @ text[top]).reshape(side, side).cpu()
        ax.imshow(sims, cmap="magma")
        ax.set_title(f"{n} cos(patch, {args.labels[top]!r})", fontsize=7)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "patch_lens.png", dpi=120)
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
