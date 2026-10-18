"""Loading VLMs (LLaVA, Qwen2.5-VL, Qwen3-VL) behind one interface for interp code.

Why: the research plan is to run the *same* experiment across LLaVA -> Qwen-VL -> a driving
fine-tune and compare. That only works if experiment code never touches architecture-specific module
paths. All per-family differences (where the blocks live, how the image-token grid is shaped)
are resolved here once.

All three share the shape: vision encoder -> projector -> decoder LM, with image features
spliced into the token sequence at `image_token_id` positions.

    family      vision blocks                       projector                         LM
    llava       model.vision_tower[.vision_model]   model.multi_modal_projector       model.language_model
                .encoder.layers  (CLIP / SigLIP)    (2-layer MLP)
    qwen*_vl    model.visual.blocks                 model.visual.merger (2x2 merge+MLP) model.language_model
                (Qwen3-VL also has visual.deepstack_merger_list -> early LM layers)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


def pick_device() -> torch.device:
    """Same scripts should run on the Mac (MPS) and a rented GPU box without edits."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class VLM:
    """Handle to a loaded VLM plus the submodules interp code reads and writes.

    Deliberately a thin record of references, not a wrapper with its own forward: `vlm.model(**inputs)`
    stays the real HF model, so nothing we measure can come from a wrapper bug.
    """

    model: nn.Module
    processor: object | None
    family: str
    vision_blocks: nn.ModuleList
    projector: nn.Module
    lm_layers: nn.ModuleList
    final_norm: nn.Module
    lm_head: nn.Module
    image_token_id: int

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def n_layers(self) -> int:
        return len(self.lm_layers)

    def named_hook_points(self) -> dict[str, nn.Module]:
        """Architecture-neutral names -> modules. "lm.12" means the same thing for LLaVA and Qwen, so
        results/caches from different models can be compared key-by-key."""
        points = {f"vit.{i}": b for i, b in enumerate(self.vision_blocks)}
        points["projector"] = self.projector
        points |= {f"lm.{i}": l for i, l in enumerate(self.lm_layers)}
        points["final_norm"] = self.final_norm
        return points

    def image_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Which sequence positions hold image features: needed to slice every LM-side activation
        into "vision" vs "text" parts."""
        return input_ids == self.image_token_id

    def image_grid(self, inputs: dict) -> tuple[int, int]:
        """(h, w) of the LM-side image-token grid for the first image in `inputs`.

        Needed to reshape per-token results back into a spatial map over the image. The grid isn't
        stored anywhere uniform: Qwen reports pre-merge patch counts (divide by the merge size),
        LLaVA only implies it through the token count.
        """
        if self.family == "qwen":
            m = self.model.config.vision_config.spatial_merge_size
            _, h, w = inputs["image_grid_thw"][0].tolist()
            return h // m, w // m
        n = int(self.image_mask(inputs["input_ids"][:1]).sum())
        side = math.isqrt(n)
        if side * side != n:
            raise ValueError(f"{n} image tokens is not a square grid (anyres/tiling models need custom handling)")
        return side, side

    @torch.no_grad()
    def prepare(self, image, prompt: str) -> dict[str, torch.Tensor]:
        """Build model inputs exactly as the model's own chat template does.

        Interp results are only meaningful on in-distribution inputs; a hand-rolled prompt with the
        wrong special tokens can change behavior. This also yields the family-specific extras
        (Qwen's image_grid_thw, mm_token_type_ids) that the forward pass requires.
        """
        if self.processor is None:
            raise RuntimeError("No processor attached; build inputs manually.")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    def decode(self, ids) -> list[str]:
        """Per-token strings (not one joined string) so they line up 1:1 with lens rows. Falls back
        to ids so tiny test models without a tokenizer still work."""
        tok = getattr(self.processor, "tokenizer", None)
        if tok is None:
            return [str(int(i)) for i in ids]
        return [tok.decode([int(i)]) for i in ids]


def _vit_layers(tower: nn.Module) -> nn.ModuleList:
    # LLaVA's tower is sometimes the bare transformer (CLIP) and sometimes a *Model wrapper around
    # one (SigLIP in llava-interleave); both expose .encoder.layers one level apart.
    inner = getattr(tower, "vision_model", tower)
    return inner.encoder.layers


def wrap_model(model: nn.Module, processor=None) -> VLM:
    """Resolve submodules on an already-built model. Split from `load_vlm` so tests can use tiny
    random-weight models, and so it works on models loaded any other way (quantized, PEFT-merged).
    Fails loudly on unknown families rather than guessing paths."""
    cfg, inner = model.config, model.model
    lm = inner.language_model
    common = dict(
        model=model,
        processor=processor,
        lm_layers=lm.layers,
        final_norm=lm.norm,
        lm_head=model.lm_head,
        image_token_id=cfg.image_token_id,
    )
    if cfg.model_type in ("qwen2_5_vl", "qwen3_vl"):
        return VLM(family="qwen", vision_blocks=inner.visual.blocks, projector=inner.visual.merger, **common)
    if cfg.model_type == "llava":
        return VLM(family="llava", vision_blocks=_vit_layers(inner.vision_tower), projector=inner.multi_modal_projector, **common)
    raise NotImplementedError(f"model_type={cfg.model_type!r}")


def load_vlm(
    model_id: str = "llava-hf/llava-interleave-qwen-0.5b-hf",
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    max_pixels: int = 512 * 28 * 28,
) -> VLM:
    """Load a LLaVA or Qwen-VL checkpoint (including driving fine-tunes built on them) in one call.

    Default is the smallest real LLaVA that fits comfortably on a 16 GB Mac. `max_pixels` (Qwen only)
    exists because Qwen keeps native resolution: an unscaled dashcam frame becomes thousands of
    image tokens, and caching every layer at every position would exhaust memory.
    """
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    device = torch.device(device) if device is not None else pick_device()
    kw = {"max_pixels": max_pixels} if AutoConfig.from_pretrained(model_id).model_type.startswith("qwen") else {}
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return wrap_model(model, AutoProcessor.from_pretrained(model_id, **kw))
