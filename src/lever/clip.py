"""CLIP tooling: residual-stream lens into the joint embedding space, and exact direct-effect
decomposition of the image embedding into per-head / per-MLP contributions.

Why CLIP gets its own module instead of going through `VLM`: CLIP has no LM and no vocabulary.
Its "readout" is cosine similarity with text embeddings, so both the lens and the attribution
target differ. Why study it at all: it's the vision tower inside LLaVA-1.5 and the ancestor of
the ViTs in the later models, so circuits found here (e.g. text-reading heads) are hypotheses to
test downstream.

Layout (transformers CLIPModel):
    model.vision_model.embeddings -> pre_layrnorm -> encoder.layers[i] -> post_layernorm (CLS) -> visual_projection
    encoder.layers[i]: x + self_attn(layer_norm1(x)), then x + mlp(layer_norm2(x))
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from lever.hooks import record
from lever.models import pick_device


@dataclass
class CLIP:
    """Handle to a CLIPModel + processor. Like `VLM`, only holds references; the HF model does the compute."""

    model: nn.Module
    processor: object | None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def vision(self) -> nn.Module:
        return self.model.vision_model

    @property
    def layers(self) -> nn.ModuleList:
        return self.vision.encoder.layers

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def n_heads(self) -> int:
        return self.layers[0].self_attn.num_heads

    def named_hook_points(self) -> dict[str, nn.Module]:
        """vit.embed = residual stream entering layer 0; vit.{i} = residual after layer i;
        vit.{i}.attn / vit.{i}.mlp = what each sublayer writes into the residual stream."""
        points = {"vit.embed": self.vision.pre_layrnorm}
        for i, layer in enumerate(self.layers):
            points[f"vit.{i}"] = layer
            points[f"vit.{i}.attn"] = layer.self_attn
            points[f"vit.{i}.mlp"] = layer.mlp
        return points

    @torch.no_grad()
    def pixels(self, images) -> torch.Tensor:
        """Preprocess with the checkpoint's own resize/crop/normalize: wrong normalization quietly
        degrades accuracy and would contaminate every result."""
        return self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Unit-norm text embeddings [n, d_joint]: the "readout directions" every CLIP experiment
        projects onto, playing the role the unembedding plays in an LM."""
        tok = self.processor.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        emb = self.model.text_projection(self.model.text_model(**tok).pooler_output)
        return emb / emb.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Unit-norm image embeddings [b, d_joint]: the ground truth that lens/decomposition results
        are checked against."""
        emb = self.model.visual_projection(self.vision(pixel_values=pixel_values).pooler_output)
        return emb / emb.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def lens(self, resid: torch.Tensor) -> torch.Tensor:
        """Map a residual-stream vector (any layer, any token) into the joint space as if it
        were the final CLS state: post_layernorm -> visual_projection -> unit-normalize.

        CLIP's analogue of the logit lens: tells you at which depth the image "already means"
        a label. On the final CLS it equals `encode_image` (tested). On patch tokens it's off-distribution
        (they're never projected in training), so treat patch maps as suggestive only.
        """
        emb = self.model.visual_projection(self.vision.post_layernorm(resid))
        return emb / emb.norm(dim=-1, keepdim=True)

    @property
    def logit_scale(self) -> float:
        # Learned temperature (~100 for OpenAI CLIP). Needed to report probabilities and logit
        # contributions on the same scale the model actually classifies with.
        return float(self.model.logit_scale.exp())


@torch.no_grad()
def decompose_image_embedding(clip: CLIP, pixel_values: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
    """Exact additive decomposition of the (un-normalized) image embedding.

    Turns "which parts of the ViT produce this prediction?" from an ablation search into a single
    forward pass. Any component's direct effect on any label is just `component @ text_emb`. That's the
    basis for finding label-specific heads (Gandelsman et al. 2023) and text-reading heads for
    typographic attacks. Caveat: direct effect only. A head that matters by feeding later layers
    shows up small here; confirm with `patch_output`.

    CLS residual after the last layer = embed + sum_i (attn_i + mlp_i), and attn_i = sum_h head_h + bias.
    Freezing the final LayerNorm's per-sample mean/std (a standard, exact-for-this-input
    linearization), each part maps linearly to the joint space. Returns
    components [b, n_components, d_joint] and their names; components.sum(1) == visual_projection(pooler_output).
    """
    vision, n_heads = clip.vision, clip.n_heads
    head_in: dict[int, torch.Tensor] = {}
    handles = [
        layer.self_attn.out_proj.register_forward_pre_hook(
            lambda _m, args, i=i: head_in.__setitem__(i, args[0][:, 0].detach())
        )
        for i, layer in enumerate(clip.layers)
    ]
    points = {k: v for k, v in clip.named_hook_points().items() if k == "vit.embed" or k.endswith(".mlp")}
    try:
        with record(points) as cache:
            vision(pixel_values=pixel_values)
    finally:
        for h in handles:
            h.remove()

    parts, names = [cache["vit.embed"][:, 0]], ["embed"]
    for i, layer in enumerate(clip.layers):
        out_proj = layer.self_attn.out_proj
        z = head_in[i]  # [b, d] = concat of per-head outputs before out_proj
        d_head = z.shape[-1] // n_heads
        for h in range(n_heads):
            sl = slice(h * d_head, (h + 1) * d_head)
            parts.append(z[:, sl] @ out_proj.weight[:, sl].T)
            names.append(f"L{i}.H{h}")
        parts.append(out_proj.bias.expand_as(parts[-1]) if out_proj.bias is not None else torch.zeros_like(parts[-1]))
        names.append(f"L{i}.attn_bias")
        parts.append(cache[f"vit.{i}.mlp"][:, 0])
        names.append(f"L{i}.mlp")

    comps = torch.stack(parts, 1)  # [b, n, d_vis]
    x = comps.sum(1)
    ln = vision.post_layernorm
    std = (x.var(-1, unbiased=False, keepdim=True) + ln.eps).sqrt()
    normed = (comps - comps.mean(-1, keepdim=True)) / std.unsqueeze(1) * ln.weight
    comps = torch.cat([normed, ln.bias.expand(x.shape[0], 1, -1)], 1)
    names.append("ln_bias")
    return clip.model.visual_projection(comps), names


def load_clip(model_id: str = "openai/clip-vit-base-patch32", device=None, dtype=torch.float32) -> CLIP:
    """One-call load. Default fp32: CLIP-B is small, and decomposition sums hundreds of terms whose
    tested exactness would degrade in bf16."""
    from transformers import CLIPModel, CLIPProcessor

    device = torch.device(device) if device is not None else pick_device()
    model = CLIPModel.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return CLIP(model, CLIPProcessor.from_pretrained(model_id))


def label_probs(clip: CLIP, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    """Zero-shot softmax over labels. Both inputs unit-norm: [b, d], [n_labels, d] -> [b, n_labels].

    Takes embeddings rather than images so the same function scores real, lensed, and
    ablated/reconstructed embeddings; that's how experiments compare them.
    """
    return (clip.logit_scale * image_emb @ text_emb.T).softmax(-1)
