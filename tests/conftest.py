"""Tiny random-weight models so the tooling is testable without downloading checkpoints."""

import pytest
import torch
from transformers import (
    CLIPConfig,
    CLIPModel,
    LlavaConfig,
    LlavaForConditionalGeneration,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)

from lever import CLIP, wrap_model

TEXT = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2)
TINY_VIT = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=32, patch_size=8)


def _qwen25():
    cfg = Qwen2_5_VLConfig(
        text_config=TEXT | dict(vocab_size=152064, rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2]}),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, fullatt_block_indexes=[1]),
    )
    return Qwen2_5_VLForConditionalGeneration(cfg)


def _qwen3():
    cfg = Qwen3VLConfig(
        text_config=TEXT | dict(vocab_size=152064, head_dim=16, rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": True}),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, deepstack_visual_indexes=[0], num_position_embeddings=64),
    )
    return Qwen3VLForConditionalGeneration(cfg)


def _llava():
    cfg = LlavaConfig(
        vision_config=TINY_VIT | dict(model_type="clip_vision_model"),
        text_config=TEXT | dict(model_type="llama", vocab_size=32064),
        image_token_id=32000,
    )
    return LlavaForConditionalGeneration(cfg)


def qwen_inputs(model, grid=(1, 4, 4)):
    cfg, vc = model.config, model.config.vision_config
    n_patches = grid[0] * grid[1] * grid[2]
    n_img = n_patches // vc.spatial_merge_size**2
    ids = torch.tensor([[1, 2, cfg.vision_start_token_id] + [cfg.image_token_id] * n_img + [cfg.vision_end_token_id, 5, 6]])
    return dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        mm_token_type_ids=(ids == cfg.image_token_id).int(),
        pixel_values=torch.randn(n_patches, 3 * vc.temporal_patch_size * vc.patch_size**2),
        image_grid_thw=torch.tensor([grid]),
    )


def llava_inputs(model):
    vc = model.config.vision_config
    n_img = (vc.image_size // vc.patch_size) ** 2  # CLS dropped by the "default" feature strategy
    ids = torch.tensor([[1, 2] + [model.config.image_token_id] * n_img + [5, 6]])
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=torch.randn(1, 3, vc.image_size, vc.image_size))


@pytest.fixture(params=["llava", "qwen2_5_vl", "qwen3_vl"])
def tiny(request):
    torch.manual_seed(0)
    build, make_inputs = {"llava": (_llava, llava_inputs), "qwen2_5_vl": (_qwen25, qwen_inputs), "qwen3_vl": (_qwen3, qwen_inputs)}[request.param]
    model = build().eval()
    return wrap_model(model), make_inputs(model)


@pytest.fixture
def tiny_clip():
    torch.manual_seed(0)
    cfg = CLIPConfig(
        text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2),
        vision_config=TINY_VIT | dict(num_attention_heads=4),
        projection_dim=16,
    )
    model = CLIPModel(cfg).eval()
    # Random init leaves LN at identity/zero; perturb so the decomposition test exercises every term.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "layer_norm" in n or "layrnorm" in n or n.endswith("bias"):
                p.add_(0.1 * torch.randn_like(p))
    return CLIP(model, None), torch.randn(2, 3, 32, 32)
