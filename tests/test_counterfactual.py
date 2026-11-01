"""lever.counterfactual: an edit that changes nothing flips nothing, and flips are counted against
the clean prediction for both VLMs and CLIP."""

import pytest
import torch

from lever.counterfactual import counterfactual_eval, clip_scorer, vlm_scorer


def _fixed(logits):
    """A scorer over pre-baked candidate logits, so flip counts can be checked exactly."""
    return lambda i: torch.tensor(logits[i])


def test_counts_flips_against_the_clean_prediction():
    score = _fixed({0: [2.0, 0.0], 1: [0.0, 2.0], 2: [0.0, 2.0], 3: [0.0, 2.0], 4: [2.0, 0.0]})
    r = counterfactual_eval(score, [0, 1], {"shortcut": [2, 4], "control": [0, 1]})
    assert r.clean_predictions.tolist() == [0, 1]
    assert r.conditions["shortcut"].flipped.tolist() == [True, True]
    assert r.conditions["shortcut"].flip_rate_to(1) == 0.5
    assert r.conditions["control"].flip_rate == 0.0
    assert r.conditions["control"].mean_prob_drop == pytest.approx(0.0)
    assert r.summary()["shortcut"]["flip_rate"] == 1.0


def test_mismatched_lengths_fail():
    with pytest.raises(ValueError, match="same position"):
        counterfactual_eval(_fixed({0: [1.0, 0.0]}), [0], {"x": [0, 0]})


def test_vlm_unchanged_inputs_flip_nothing(tiny):
    vlm, inputs = tiny
    other = dict(inputs, pixel_values=torch.randn_like(inputs["pixel_values"]))
    r = counterfactual_eval(vlm_scorer(vlm, [5, 6, 7]), [inputs], {"same": [inputs], "other": [other]})
    assert r.conditions["same"].flip_rate == 0.0
    assert r.conditions["same"].prob_drop.abs().max() < 1e-5
    assert r.clean_probs.shape == (1, 3)
    # The scorer reads the model's own answer-position logits at the candidate ids.
    expected = vlm.model(**inputs).logits[0, -1, [5, 6, 7]].float().softmax(-1)
    torch.testing.assert_close(r.clean_probs[0], expected, atol=1e-5, rtol=1e-4)


def test_clip_scorer_matches_label_probs(tiny_clip):
    clip, pv = tiny_clip
    labels = torch.nn.functional.normalize(torch.randn(3, 16), dim=-1)
    score = clip_scorer(clip, labels)
    images = [pv[:1], pv[1:]]
    r = counterfactual_eval(score, images, {"same": images, "swapped": images[::-1]})
    assert r.conditions["same"].flip_rate == 0.0
    expected = (clip.logit_scale * clip.encode_image(pv[:1]) @ labels.T).softmax(-1)[0]
    torch.testing.assert_close(r.clean_probs[0], expected, atol=1e-5, rtol=1e-4)
