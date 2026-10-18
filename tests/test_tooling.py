import torch

from lever import logit_lens, patch_output, record


def test_hook_points_resolve(tiny):
    vlm, _ = tiny
    points = vlm.named_hook_points()
    assert {"vit.0", "vit.1", "projector", "lm.0", "lm.2", "final_norm"} <= points.keys()
    assert vlm.n_layers == 3


def test_record_captures_residual_stream(tiny):
    vlm, inputs = tiny
    with record(vlm.named_hook_points()) as cache:
        vlm.model(**inputs)
    seq = inputs["input_ids"].shape[1]
    for i in range(vlm.n_layers):
        assert cache[f"lm.{i}"].shape == (1, seq, 64)
    n_img = int(vlm.image_mask(inputs["input_ids"]).sum())
    assert cache["projector"].numel() == n_img * 64  # projector output = LM-space image embeddings
    h, w = vlm.image_grid(inputs)
    assert h * w == n_img


def test_logit_lens_on_last_layer_matches_model_logits(tiny):
    vlm, inputs = tiny
    with record({"last": vlm.lm_layers[-1]}) as cache:
        out = vlm.model(**inputs)
    lens = logit_lens(vlm, cache["last"])
    torch.testing.assert_close(lens, out.logits.float().log_softmax(-1), atol=1e-4, rtol=1e-4)


def test_ablating_image_tokens_changes_output(tiny):
    vlm, inputs = tiny
    clean = vlm.model(**inputs).logits
    with patch_output(vlm.projector, torch.zeros_like):
        ablated = vlm.model(**inputs).logits
    assert not torch.allclose(clean[0, -1], ablated[0, -1])
