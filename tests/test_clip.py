import torch

from lever import decompose_image_embedding, record


def test_decomposition_sums_to_image_embedding(tiny_clip):
    clip, pv = tiny_clip
    comps, names = decompose_image_embedding(clip, pv)
    assert comps.shape[:2] == (2, len(names))
    assert len([n for n in names if ".H" in n]) == clip.n_layers * clip.n_heads
    expected = clip.model.visual_projection(clip.vision(pixel_values=pv).pooler_output)
    torch.testing.assert_close(comps.sum(1), expected, atol=1e-5, rtol=1e-4)


def test_lens_on_final_cls_equals_image_embedding(tiny_clip):
    clip, pv = tiny_clip
    with record({"last": clip.layers[-1]}) as cache:
        clip.vision(pixel_values=pv)
    torch.testing.assert_close(clip.lens(cache["last"][:, 0]), clip.encode_image(pv))
