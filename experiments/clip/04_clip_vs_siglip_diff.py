"""CLIP 04: model diffing CLIP vs SigLIP. Same architecture, different objective, different readout.

Both defaults are ViT-B/16 at 224px (12 layers, width 768), so architecture is held fixed and
what differs is: training loss (softmax contrastive vs pairwise sigmoid), data (WIT-400M vs WebLI),
and pooling (CLS token vs attention-pooling head). Differences found here are candidates for
"what the objective/data does to a vision encoder". Data is a confounder we can't remove.

Four questions, one section each:
  A. Behavior: do they get the same images right? How confident, in each model's own terms?
  B. Geometry: how far apart are image and text embeddings (modality gap)? How spread out?
  C. Layers: where do the two encoders' representations agree (CKA), and where does class
     information become linearly decodable (probes)? For CLIP, CLS vs patch-mean.
  D. Text reading: how strongly does written text in the image drive each model's output? Swept
     over stamp size, with text-only images at each size, so "can it read" and "does it follow"
     are measured at the same text size.

Section letters here are the summary.json keys. In the blog post, A/B/D appear as A1-A3 ("as
models in their own right") and C as B2-B3 ("as VLM backbones").

Data: Imagenette (10 easy ImageNet classes), val split, `--per-class` images each.

    uv run python experiments/clip/04_clip_vs_siglip_diff.py
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.datasets import Imagenette

from mmsafety.dual_encoders import load_dual_encoder
from mmsafety.images import stamp_text
from mmsafety.similarity import linear_cka

TEMPLATES = ["a photo of a {}.", "a blurry photo of a {}.", "a close-up photo of a {}.", "a photo of the {}.", "an image of a {}."]


def center_square(im: Image.Image) -> Image.Image:
    """Center-crop to a square before anything else touches the image.

    Why: the two processors handle non-square inputs differently. CLIP resizes the short side and
    center-crops; SigLIP squashes the whole image to 224x224. Most Imagenette images are 213x160,
    so without this CLIP would see a cropped view (clipping the sides of long stamped labels) and
    SigLIP a horizontally squashed one: different stimuli, confounding the attack comparison.
    On a square image both reduce to the same bicubic resize, so both models see the same content.
    """
    s = min(im.size)
    left, top = (im.width - s) // 2, (im.height - s) // 2
    return im.crop((left, top, left + s, top + s))


def load_images(per_class: int, seed: int):
    """Balanced subset so per-class numbers and probes aren't skewed by class frequency.
    Square-cropped (see center_square) so preprocessing can't differ between models."""
    ds = Imagenette("data", split="val", size="160px", download=not Path("data/imagenette2-160").exists())
    classes = [names[0] for names in ds.classes]
    by_class: dict[int, list[int]] = {}
    for i, (_, y) in enumerate(ds._samples):  # private, but avoids decoding ~4k images just to read labels
        by_class.setdefault(y, []).append(i)
    rng = random.Random(seed)
    idx = [i for y in sorted(by_class) for i in rng.sample(by_class[y], per_class)]
    images = [center_square(ds[i][0].convert("RGB")) for i in idx]
    labels = torch.tensor([ds[i][1] for i in idx])
    return images, labels, classes


def class_text_embeds(enc, classes):
    """Prompt-ensembled class embeddings (mean over templates), the standard zero-shot recipe.
    Averaging reduces sensitivity to any single template, which would otherwise add noise to the diff."""
    embs = torch.stack([enc.encode_text([t.format(c) for t in TEMPLATES]).mean(0) for c in classes])
    return (embs / embs.norm(dim=-1, keepdim=True)).cpu()


def probe_accuracy(feats: torch.Tensor, labels: torch.Tensor) -> float:
    """5-fold linear-probe accuracy: "is class linearly decodable from this layer?"
    Standardized, because raw residual streams have a few huge-magnitude dims that dominate otherwise."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return float(cross_val_score(clf, feats.numpy(), labels.numpy(), cv=5).mean())


def modality_gap(img: torch.Tensor, txt: torch.Tensor, labels: torch.Tensor) -> dict:
    """Liang et al. 2022: CLIP's image and text embeddings occupy separate cones. Gap = distance
    between the two centroids on the unit sphere. Anisotropy = mean pairwise cosine within a
    modality (1 = everything points the same way). Both shape how usable the joint space is."""
    def mean_pairwise_cos(e):
        s = e @ e.T
        n = len(e)
        return float((s.sum() - n) / (n * (n - 1)))

    return {
        "gap": float((img.mean(0) - txt.mean(0)).norm()),
        "mean_cos_matched": float((img * txt[labels]).sum(-1).mean()),
        "image_anisotropy": mean_pairwise_cos(img),
        "text_anisotropy": mean_pairwise_cos(txt),
    }


def bootstrap_ci(stat, n: int, n_boot: int = 2000, seed: int = 0) -> list[float]:
    """95% percentile-bootstrap CI over images for stat(idx) -> float.

    Why: 500 images and one seed; the post compares rates like 57% vs 31%, and readers need to
    know whether a gap is outside resampling noise. Resampling the *same* indices for both models
    gives a paired CI on differences, which is much tighter than comparing two independent CIs.
    Covers image sampling only, not the choice of classes, stamp style or seed.
    """
    g = torch.Generator().manual_seed(seed)
    vals = torch.tensor([stat(torch.randint(0, n, (n,), generator=g)) for _ in range(n_boot)])
    return [round(float(v), 3) for v in vals.quantile(torch.tensor([0.025, 0.975], dtype=vals.dtype))]


def zscore(x: torch.Tensor) -> torch.Tensor:
    """Per-dimension standardization. Needed because raw CKA is dominated by the highest-variance
    dimensions: CLIP develops a single outlier dim at layer 8 holding ~half the variance, which
    swamps raw CKA. Reporting both views keeps one dimension from dictating the conclusion."""
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def top_dim_share(x: torch.Tensor) -> float:
    """Fraction of total variance in the single largest-variance dimension: an outlier detector."""
    v = x.var(0)
    return float(v.max() / v.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs=2, default=["openai/clip-vit-base-patch16", "google/siglip-base-patch16-224"])
    p.add_argument("--per-class", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    # Text height as a fraction of image height. The headline numbers use --stamp-size; the sweep
    # shows whether the CLIP/SigLIP gap is a property of the models or of one chosen size.
    # Sweep floor is 0.08: stamp_text clamps fonts to >=12px, so smaller sizes on 160px images would
    # be mislabeled on the x-axis.
    p.add_argument("--stamp-size", type=float, default=0.15)
    p.add_argument("--stamp-sweep", type=float, nargs="+", default=[0.08, 0.1, 0.12, 0.15, 0.2])
    # The single illustrative image for the write-up. Index is into the loaded subset, so it only
    # points at the same photo for the default --per-class/--seed. 25 = tench pile, no people in frame.
    p.add_argument("--example-idx", type=int, default=25)
    p.add_argument("--example-text", default="gas pump")
    p.add_argument("--out", default="outputs/clip_04_clip_vs_siglip")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    images, labels, classes = load_images(args.per_class, args.seed)
    n_cls = len(classes)

    # Typographic stimuli (section D), built once so both models see identical pixels.
    rng = random.Random(args.seed)
    stamped_cls = torch.tensor([(int(y) + rng.randint(1, n_cls - 1)) % n_cls for y in labels])  # always a wrong class
    sizes = sorted(set(args.stamp_sweep) | {args.stamp_size})
    attacked = {s: [stamp_text(im, classes[c], s) for im, c in zip(images, stamped_cls.tolist())] for s in sizes}
    # Pure text on flat backgrounds, same canvas size as the photos so a given `s` gives the same
    # text height in model input: isolates reading from any object evidence, at matched size.
    side = images[0].height
    text_only = {s: [stamp_text(Image.new("RGB", (side, side), (g, g, g)), c, s) for c in classes for g in (60, 120, 180, 230)]
                 for s in sizes}
    text_only_labels = torch.tensor([i for i in range(n_cls) for _ in range(4)])
    # Fixed-text example, separate from the random-wrong-class attack set: a write-up needs one
    # concrete, checkable case, and recording both models' predictions on it keeps the prose honest.
    example = stamp_text(images[args.example_idx], args.example_text, args.stamp_size)
    example.save(out / "example_attacked.png")

    feats, summary = {}, {}
    for model_id in args.models:
        enc = load_dual_encoder(model_id)
        txt = class_text_embeds(enc, classes)
        f = enc.encode_images(images)
        feats[enc.name] = f

        scores = enc.label_scores(f.embeds, txt)
        pred = scores.argmax(-1)
        sweep = {}
        for s in sizes:
            pred_att = enc.label_scores(enc.encode_images(attacked[s]).embeds, txt).argmax(-1)
            pred_txt = enc.label_scores(enc.encode_images(text_only[s]).embeds, txt).argmax(-1)
            sweep[s] = {
                "attack_acc": float((pred_att == labels).float().mean()),
                "attack_success": float((pred_att == stamped_cls).float().mean()),  # followed the text
                "text_only_acc": float((pred_txt == text_only_labels).float().mean()),
                "_followed": pred_att == stamped_cls,
            }
        head = sweep[args.stamp_size]
        ex = enc.label_scores(enc.encode_images([images[args.example_idx], example]).embeds, txt)

        geo = modality_gap(f.embeds, txt, labels)
        summary[enc.name] = {
            "family": enc.family,
            "A_zero_shot_acc": float((pred == labels).float().mean()),
            "A_per_class_acc": {c: float((pred[labels == i] == i).float().mean()) for i, c in enumerate(classes)},
            # Confidence in each model's own output semantics (softmax vs sigmoid); see DualEncoder.label_scores.
            "A_mean_top_score": float(scores.max(-1).values.mean()),
            "A_frac_top_score_below_0.5": float((scores.max(-1).values < 0.5).float().mean()),
            "B_geometry": geo,
            "D_stamp_size": args.stamp_size,
            "D_attack_acc": head["attack_acc"],
            "D_attack_success": head["attack_success"],
            "D_attack_success_ci95": bootstrap_ci(lambda i, v=head["_followed"]: v[i].float().mean(), len(labels)),
            "D_text_only_acc": head["text_only_acc"],
            "D_example": {
                f"{k}_pred": classes[int(v.argmax())] for k, v in zip(("clean", "stamped"), ex)
            } | {f"{k}_top_score": round(float(v.max()), 3) for k, v in zip(("clean", "stamped"), ex)},
            "D_sweep": {s: {k: v for k, v in d.items() if not k.startswith("_")} for s, d in sweep.items()},
            "_pred": pred,
            "_followed": head["_followed"],
        }
        del enc  # free the first model before loading the second; both at once is tight on 16 GB
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    a, b = list(summary)
    pa, pb = summary[a].pop("_pred"), summary[b].pop("_pred")
    fol_a, fol_b = summary[a].pop("_followed"), summary[b].pop("_followed")
    # Paired: same images, same stamps, so resample indices jointly (see bootstrap_ci).
    summary["D_attack_success_diff"] = {
        "a_minus_b": float(fol_a.float().mean() - fol_b.float().mean()),
        "ci95": bootstrap_ci(lambda i: fol_a[i].float().mean() - fol_b[i].float().mean(), len(labels)),
    }
    summary["A_agreement"] = {
        "same_prediction": float((pa == pb).float().mean()),
        f"only_{a}_right": float(((pa == labels) & (pb != labels)).float().mean()),
        f"only_{b}_right": float(((pa != labels) & (pb == labels)).float().mean()),
        "both_wrong": float(((pa != labels) & (pb != labels)).float().mean()),
    }

    # C1. Cross-model CKA, every layer of A vs every layer of B (mean-pooled patch states),
    # raw and per-dim standardized (see zscore for why both).
    fa, fb = feats[a].patch_means, feats[b].patch_means
    views = {"raw": (fa, fb), "standardized": (torch.stack([zscore(x) for x in fa]), torch.stack([zscore(x) for x in fb]))}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (view, (xa, xb)) in zip(axes, views.items()):
        cka = torch.tensor([[linear_cka(xa[i], xb[j]) for j in range(len(xb))] for i in range(len(xa))])
        summary[f"C_cka_diagonal_{view}"] = [round(float(cka[i, i]), 3) for i in range(min(len(xa), len(xb)))]
        im = ax.imshow(cka, vmin=0, vmax=1, cmap="viridis", origin="lower")
        ax.set_xlabel(f"{b} layer (0 = embeddings)")
        ax.set_ylabel(f"{a} layer")
        ax.set_title(f"linear CKA ({view}), mean-pooled patches", fontsize=9)
    fig.colorbar(im, ax=axes)
    fig.savefig(out / "cka.png", dpi=130, bbox_inches="tight")
    summary["C_cka_final_embeds"] = linear_cka(feats[a].embeds, feats[b].embeds)
    summary["C_top_dim_variance_share"] = {k: [round(top_dim_share(x), 3) for x in f.patch_means] for k, f in feats.items()}

    # C2. Linear probes per layer. For CLIP, CLS vs patch-mean: LLaVA consumes patches, not CLS.
    probes = {}
    for name, f in feats.items():
        probes[f"{name} patch-mean"] = [probe_accuracy(x, labels) for x in f.patch_means]
        if f.cls is not None:
            probes[f"{name} CLS"] = [probe_accuracy(x, labels) for x in f.cls]
    summary["C_probe_acc_by_layer"] = {k: [round(v, 3) for v in vs] for k, vs in probes.items()}

    fig, ax = plt.subplots(figsize=(7, 4))
    for k, vs in probes.items():
        ax.plot(range(len(vs)), vs, marker="o", label=k)
    ax.axhline(1 / n_cls, color="gray", ls=":", label="chance")
    ax.set_xlabel("layer (0 = embeddings)")
    ax.set_ylabel("5-fold linear probe acc")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "probes.png", dpi=130)

    # D. Dose-response: following (solid) vs reading (dashed) at each text size. If both models read
    # perfectly at a size where they follow at different rates, the gap is in weighting, not reading.
    fig, ax = plt.subplots(figsize=(7, 4))
    for k in (a, b):
        sw = summary[k]["D_sweep"]
        xs = sorted(sw)
        line, = ax.plot(xs, [sw[x]["attack_success"] for x in xs], marker="o", label=f"{k}: follows stamped text")
        ax.plot(xs, [sw[x]["text_only_acc"] for x in xs], marker="s", ls="--", color=line.get_color(),
                label=f"{k}: reads text-only image")
    ax.axvline(args.stamp_size, color="gray", ls=":", lw=1)
    ax.set_xlabel("text height (fraction of image height)")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "attack_sweep.png", dpi=130)

    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for k in (a, b):
        s = summary[k]
        print(f"\n== {k} ({s['family']})")
        print(f"  A zero-shot acc {s['A_zero_shot_acc']:.3f} | mean top score {s['A_mean_top_score']:.3f} "
              f"| top score < 0.5 on {s['A_frac_top_score_below_0.5']:.0%} of images")
        g = s["B_geometry"]
        print(f"  B modality gap {g['gap']:.3f} | cos(img, own class text) {g['mean_cos_matched']:.3f} "
              f"| anisotropy img {g['image_anisotropy']:.3f} txt {g['text_anisotropy']:.3f}")
        print(f"  D attacked acc {s['D_attack_acc']:.3f} | followed stamped text {s['D_attack_success']:.3f} "
              f"{s['D_attack_success_ci95']} | text-only acc {s['D_text_only_acc']:.3f}")
        print(f"  D example ({args.example_text!r} on image {args.example_idx}): {s['D_example']}")
        for sz, d in s["D_sweep"].items():
            print(f"    size {sz:.2f}: follows {d['attack_success']:.3f} | reads text-only {d['text_only_acc']:.3f}")
    print(f"\nagreement: {summary['A_agreement']}")
    print(f"attack success {a} - {b}: {summary['D_attack_success_diff']}")
    print(f"CKA diagonal raw:          {summary['C_cka_diagonal_raw']}")
    print(f"CKA diagonal standardized: {summary['C_cka_diagonal_standardized']}")
    print(f"CKA final embeds: {summary['C_cka_final_embeds']:.3f}")
    for k, vs in summary["C_top_dim_variance_share"].items():
        print(f"top-dim variance share {k:>28}: {vs}")
    for k, vs in summary["C_probe_acc_by_layer"].items():
        print(f"probe {k:>40}: {vs}")
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
