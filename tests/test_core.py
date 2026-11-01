"""lever.core: canonical names resolve to the right modules, record/patch are faithful to the real
forward pass, and capability/registry errors are loud.

The adapter here is provisional (real adapters arrive with the step-2 port) and is passed to
`Model` directly rather than registered, so it can't collide with the real one later.
"""

import pytest
import torch
from conftest import _qwen25, qwen_inputs
from torch import nn

from lever.core import Adapter, CapabilityError, Model, TowerSpec, adapter_for, register


class _Qwen25(Adapter):
    capabilities = frozenset({"per_head", "strict_residual"})

    def towers(self):
        cfg = self.model.config
        return {
            "vision": TowerSpec(blocks="model.visual.blocks", attn="attn", o_proj="attn.proj", mlp="mlp",
                                n_heads=cfg.vision_config.num_heads),
            "projector": TowerSpec(out="model.visual.merger"),
            "lm": TowerSpec(blocks="model.language_model.layers", attn="self_attn", o_proj="self_attn.o_proj",
                            mlp="mlp", n_heads=cfg.text_config.num_attention_heads,
                            embed="model.language_model.embed_tokens", final_norm="model.language_model.norm"),
        }


@pytest.fixture
def qwen():
    torch.manual_seed(0)
    hf = _qwen25().eval()
    return Model(hf, _Qwen25(hf)), qwen_inputs(hf)


def test_resid_post_matches_hf_hidden_states(qwen):
    # Ground truth is HF's own output_hidden_states: hidden_states[i + 1] is block i's output (the
    # last entry is post-final-norm, so it's excluded). Catches wrong paths and tuple handling.
    m, inputs = qwen
    with torch.no_grad(), m.record(["lm.blocks.*.resid_post"]) as cache:
        hs = m.model(**inputs, output_hidden_states=True).hidden_states
    n = m.n_layers("lm")
    assert len(cache) == n
    for i in range(n - 1):
        torch.testing.assert_close(cache[f"lm.blocks.{i}.resid_post"], hs[i + 1])


def test_resid_pre_is_previous_resid_post(qwen):
    m, inputs = qwen
    with torch.no_grad(), m.record(["lm.blocks.*.resid_pre", "lm.blocks.*.resid_post"]) as cache:
        m.model(**inputs)
    for i in range(1, m.n_layers("lm")):
        torch.testing.assert_close(cache[f"lm.blocks.{i}.resid_pre"], cache[f"lm.blocks.{i - 1}.resid_post"])


def test_attn_z_is_per_head_and_zeroing_it_equals_zeroing_attn_out(qwen):
    # Qwen2's o_proj has no bias, so o_proj(0) == 0: zeroing every head's z must give exactly the
    # same logits as zeroing the attention output. Checks the pre-hook path and head reshaping.
    m, inputs = qwen
    cfg = m.model.config.text_config
    with torch.no_grad(), m.record(["lm.blocks.1.attn.z", "vision.blocks.0.attn.z"]) as cache:
        base = m.model(**inputs).logits
    z = cache["lm.blocks.1.attn.z"]
    assert z.shape[-2:] == (cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads)
    assert cache["vision.blocks.0.attn.z"].shape[-2] == m.model.config.vision_config.num_heads
    with torch.no_grad():
        with m.patch({"lm.blocks.1.attn.z": torch.zeros_like}):
            via_z = m.model(**inputs).logits
        with m.patch({"lm.blocks.1.attn.out": torch.zeros_like}):
            via_out = m.model(**inputs).logits
    assert not torch.allclose(base, via_z)
    torch.testing.assert_close(via_z, via_out)


def test_identity_patch_is_a_noop_and_hooks_are_removed(qwen):
    m, inputs = qwen
    with torch.no_grad():
        base = m.model(**inputs).logits
        with m.patch({"lm.blocks.*.attn.z": lambda z: z, "projector.out": lambda x: x}):
            same = m.model(**inputs).logits
        torch.testing.assert_close(base, same)
        with m.patch({"projector.out": torch.zeros_like}):
            m.model(**inputs)
        torch.testing.assert_close(m.model(**inputs).logits, base)  # nothing left attached


def test_hook_names_and_errors(qwen):
    m, _ = qwen
    names = m.hook_names()
    assert "lm.blocks.0.attn.z" in names and "projector.out" in names and "lm.final_norm" in names
    assert "projector.blocks.0.resid_post" not in names
    with pytest.raises(ValueError):
        m.resolve("lm.12")
    with pytest.raises(KeyError):
        m.resolve("audio.out")
    with pytest.raises(KeyError):
        m.resolve("lm.blocks.0.nonexistent")
    m.require("per_head")
    with pytest.raises(CapabilityError):
        m.require("unembed")
    with pytest.raises(CapabilityError):
        m.adapter.unembed(torch.zeros(1))


def test_registry_selects_by_model_type():
    class _Cfg:
        model_type = "lever_test_arch"

    class _M(nn.Module):
        config = _Cfg()

    @register
    class _A(Adapter):
        model_types = ("lever_test_arch",)

        def towers(self):
            return {}

    assert isinstance(adapter_for(_M()), _A)
    with pytest.raises(ValueError):
        register(type("_Dup", (Adapter,), {"model_types": ("lever_test_arch",)}))
    _Cfg.model_type = "unknown_arch"
    with pytest.raises(NotImplementedError):
        adapter_for(_M())
