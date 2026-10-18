"""CLIP 03: typographic attacks — which components let written text override what's pictured?

CLIP famously labels an apple with a paper "iPod" note as an iPod (Goh et al. 2021, "Multimodal
Neurons"). Driving version: a sticker reading "STOP" / "GREEN LIGHT" / "NO PEDESTRIANS" in the scene.
We render text onto the image, compare zero-shot predictions, and diff the per-component
direct effects (clean vs attacked) on the attack label: those components are the ones reading
the text.

    uv run python experiments/clip/03_typographic_attack.py --image apple.jpg \
        --labels "a photo of an apple" "a photo of an iPod" --text iPod
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from lever import decompose_image_embedding, label_probs, load_clip
from lever.images import stamp_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--labels", nargs="+", required=True, help="include both the true label and the attack label")
    p.add_argument("--text", required=True, help="text to write onto the image")
    p.add_argument("--size", type=float, default=0.12, help="font height as fraction of image height")
    p.add_argument("--model", default="openai/clip-vit-base-patch32")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", default="outputs/clip_03_typographic")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    clip = load_clip(args.model)
    clean_img = Image.open(args.image).convert("RGB")
    attacked_img = stamp_text(clean_img, args.text, args.size)
    attacked_img.save(out / "attacked.png")
    text = clip.encode_text(args.labels)

    results = {}
    for tag, img in [("clean", clean_img), ("attacked", attacked_img)]:
        comps, names = decompose_image_embedding(clip, clip.pixels(img))
        emb = comps[0].sum(0)
        probs = label_probs(clip, (emb / emb.norm())[None], text)[0]
        # direct contribution of each component to every label's logit: [n_comp, n_labels]
        results[tag] = (probs.cpu(), (clip.logit_scale * comps[0] @ text.T / emb.norm()).cpu())
        print(f"{tag:>9}: " + ", ".join(f"{l!r}={p:.2f}" for l, p in zip(args.labels, probs.tolist())))

    target = int(results["attacked"][0].argmax())
    if target == int(results["clean"][0].argmax()):
        print("\nprediction did not flip; try a larger --size or more label-like --text")
        target = int((results["attacked"][0] - results["clean"][0]).argmax())
    others = [j for j in range(len(args.labels)) if j != target]

    # Contrast against the other labels rather than using the raw target logit: components that
    # raise every label equally (generic "photo-ness") shouldn't be credited with the attack.
    def margin(c):
        return c[:, target] - c[:, others].mean(1)

    delta = margin(results["attacked"][1]) - margin(results["clean"][1])
    print(f"\ncomponents whose direct push toward {args.labels[target]!r} grew most under attack:")
    for i in delta.argsort(descending=True)[: args.k].tolist():
        print(f"  {names[i]:>14}: {delta[i]:+.3f}")
    torch.save({"names": names, "delta": delta, "results": results}, out / "results.pt")
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
