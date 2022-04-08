"""Tests for adversarial.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

import adversarial


def rescale(img):
    img -= img.min()
    img /= img.max()
    return img


@pytest.mark.parametrize("epsilon", np.linspace(1, 0, 10, endpoint=False))
def test_generate_adversarial_example(
    teddy_bear_image, model, inverted_word_map, epsilon
):
    teddy_bear_out, i = model(teddy_bear_image)

    target = teddy_bear_out.argmax(-1)

    adversarial_noise = adversarial.generate_adversarial_noise(
        teddy_bear_image, model, target, epsilon
    )
    assert adversarial_noise.shape == teddy_bear_image.shape
    assert adversarial_noise.max() == epsilon
    assert adversarial_noise.min() == -epsilon

    adversarial_sample = teddy_bear_image + adversarial_noise
    adversarial_out, i = model(adversarial_sample)

    adversarial_sentence = " ".join(
        inverted_word_map[int(idx)] for idx in adversarial_out[0].argmax(-1)
    )
    teddy_bear_sentence = " ".join(
        inverted_word_map[int(idx)] for idx in teddy_bear_out[0].argmax(-1)
    )

    assert adversarial_sentence != teddy_bear_sentence


def test_adversarial_inference(batch_size, teddy_bear_image, model):
    original_sentences, adversarial_sentences = adversarial.inference(
        torch.cat(batch_size * [teddy_bear_image]), model, 0.0
    )
    torch.testing.assert_allclose(original_sentences, adversarial_sentences)
