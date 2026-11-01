"""lever.results: provenance is read off the live objects, and a Result survives save/load intact."""

import json

import pytest
import torch
from conftest import _qwen25, qwen_inputs
from test_core import _Qwen25
from transformers import CLIPImageProcessor

from lever.core import Model
from lever.results import SCHEMA_VERSION, Provenance, Result


@pytest.fixture
def qwen():
    torch.manual_seed(0)
    hf = _qwen25().eval()
    return Model(hf, _Qwen25(hf)), qwen_inputs(hf)


def test_provenance_reads_live_objects(qwen):
    m, _ = qwen
    proc = CLIPImageProcessor(crop_size=224, image_mean=[0.5, 0.5, 0.5])
    p = Provenance.capture(m, ["lm.blocks.*.resid_post"], processor=proc, seed=0)
    assert p.model_type == "qwen2_5_vl" and p.adapter == "_Qwen25"
    assert p.revision is None  # config-built model: no hub revision, and none invented
    assert p.dtype == "float32" and p.capabilities == ["per_head", "strict_residual"]
    assert p.preprocessing["image_mean"] == [0.5, 0.5, 0.5]
    assert "torch" in p.versions
    json.dumps(p.preprocessing)  # whatever the processor held is now JSON-safe
    assert Provenance.capture(m, []).preprocessing is None


def test_save_load_round_trip(qwen, tmp_path):
    m, inputs = qwen
    with torch.no_grad(), m.record(["lm.blocks.0.attn.z"]) as cache:
        m.model(**inputs)
    r = Result(
        kind="record",
        provenance=Provenance.capture(m, ["lm.blocks.0.attn.z"]),
        params={"prompt": "a photo"},
        data={"z": cache["lm.blocks.0.attn.z"], "n_layers": m.n_layers("lm"), "labels": ["a", "b"]},
    )
    back = Result.load(r.save(tmp_path / "run"))
    assert back.kind == "record" and back.provenance == r.provenance and back.params == r.params
    assert back.data["n_layers"] == r.data["n_layers"] and back.data["labels"] == ["a", "b"]
    torch.testing.assert_close(back.data["z"], r.data["z"])


def test_scalar_only_result_writes_no_tensor_file(qwen, tmp_path):
    m, _ = qwen
    path = Result("cka", Provenance.capture(m, []), data={"score": 0.9}).save(tmp_path / "cka")
    assert not (path / "tensors.safetensors").exists()
    assert Result.load(path).data == {"score": 0.9}


def test_nested_tensor_and_newer_schema_fail_loudly(qwen, tmp_path):
    m, _ = qwen
    prov = Provenance.capture(m, [])
    with pytest.raises(TypeError):
        Result("x", prov, data={"nested": [torch.zeros(2)]}).save(tmp_path / "bad")
    path = Result("x", prov).save(tmp_path / "future")
    meta = json.loads((path / "result.json").read_text())
    meta["schema_version"] = SCHEMA_VERSION + 1
    (path / "result.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        Result.load(path)
