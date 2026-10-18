"""CLIP and SigLIP behind one interface, for experiments that compare contrastive encoders.

Why separate from `clip.CLIP`: that class is built around CLIP internals (CLS token, pre-LN,
exact decomposition). Comparing encoders needs the opposite: only what *every* dual encoder
has (image/text embeddings, per-layer states, a zero-shot score), with the family differences
hidden. The differences that matter here:

    CLIP    CLS token -> LN -> linear proj;   softmax over labels (contrastive over the batch)
    SigLIP  no CLS; attention-pooling head;  independent sigmoid per (image, text) pair + learned bias
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lever.models import pick_device


@dataclass
class ImageFeatures:
    embeds: torch.Tensor  # [n, d_joint], unit-norm: what zero-shot classification uses
    patch_means: torch.Tensor  # [n_layers + 1, n, d_vis], mean over patch tokens: comparable across families
    cls: torch.Tensor | None  # [n_layers + 1, n, d_vis], CLIP only: the token CLIP's readout actually uses


@dataclass
class DualEncoder:
    model: torch.nn.Module
    processor: object
    family: str  # "clip" | "siglip"
    name: str

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Unit-norm text embeddings [n, d_joint].

        SigLIP must be padded to max_length (it was trained that way, and its text pooling reads the
        last position); CLIP pools at the EOS token, so dynamic padding is fine. Getting this wrong
        silently gives degraded embeddings instead of an error, which is why it's handled once here.
        """
        pad = dict(padding="max_length", max_length=64) if self.family == "siglip" else dict(padding=True)
        tok = self.processor.tokenizer(texts, return_tensors="pt", truncation=True, **pad).to(self.device)
        emb = self.model.get_text_features(**tok).pooler_output
        return emb / emb.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_images(self, images, batch_size: int = 32) -> ImageFeatures:
        """Final embeddings and per-layer states in one pass per batch.

        Per-layer states are mean-pooled over patch tokens because the two families don't share
        token layouts (patch size, CLS or not); a per-image vector is the common unit that
        representational comparisons (CKA, probes) can run on. For CLIP the CLS token is also kept,
        since whether class info lives in CLS or in patches matters for LLaVA, which drops CLS.
        """
        embeds, means, clss = [], [], []
        for i in range(0, len(images), batch_size):
            pv = self.processor(images=images[i : i + batch_size], return_tensors="pt")["pixel_values"].to(self.device)
            out = self.model.vision_model(pixel_values=pv, output_hidden_states=True)
            pooled = out.pooler_output
            emb = self.model.visual_projection(pooled) if self.family == "clip" else pooled
            embeds.append(emb / emb.norm(dim=-1, keepdim=True))
            hs = torch.stack(out.hidden_states)  # [L+1, b, tokens, d]
            patches = hs[:, :, 1:] if self.family == "clip" else hs
            means.append(patches.mean(2))
            if self.family == "clip":
                clss.append(hs[:, :, 0])
        return ImageFeatures(
            embeds=torch.cat(embeds).cpu(),
            patch_means=torch.cat(means, 1).cpu(),
            cls=torch.cat(clss, 1).cpu() if clss else None,
        )

    @torch.no_grad()
    def label_scores(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """Per-(image, label) probabilities *as each model was trained to produce them*.

        CLIP: softmax across labels (rows sum to 1). SigLIP: independent sigmoid per pair (rows need
        not sum to 1; "none of these" is expressible). Argmax agrees with cosine either way. The
        distinction exists for confidence comparisons, where using the wrong one misleads.
        """
        logits = image_embeds @ text_embeds.T * self.model.logit_scale.exp().item()
        if self.family == "siglip":
            return torch.sigmoid(logits + self.model.logit_bias.item())
        return logits.softmax(-1)


def load_dual_encoder(model_id: str, device=None) -> DualEncoder:
    """Load CLIP or SigLIP by HF id; family is read from the config, not guessed from the name."""
    from transformers import AutoConfig, AutoModel, AutoProcessor

    device = torch.device(device) if device is not None else pick_device()
    family = AutoConfig.from_pretrained(model_id).model_type
    if family not in ("clip", "siglip"):
        raise NotImplementedError(f"model_type={family!r}")
    model = AutoModel.from_pretrained(model_id, dtype=torch.float32).to(device).eval()
    return DualEncoder(model, AutoProcessor.from_pretrained(model_id), family, model_id.split("/")[-1])
