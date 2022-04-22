"""Tests for adversarial.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torchvision.utils import save_image

import adversarial


def rescale(img):
    img -= img.min()
    img /= img.max()
    return img


@pytest.mark.parametrize("epsilon", np.linspace(1, 0, 4, endpoint=False))
def test_generate_adversarial_example(image, model, inverted_word_map, epsilon):
    image, filename = image
    normal_image_out, i = model(image)

    target = normal_image_out.argmax(-1)

    adversarial_noise = adversarial.generate_adversarial_noise(
        image, model, target, epsilon
    )
    assert adversarial_noise.shape == image.shape
    assert adversarial_noise.max() == epsilon
    assert adversarial_noise.min() == -epsilon

    adversarial_sample = image + adversarial_noise
    adversarial_out, i = model(adversarial_sample)

    adversarial_sentence = " ".join(
        inverted_word_map[int(idx)] for idx in adversarial_out[0].argmax(-1)
    )
    normal_sentence = " ".join(
        inverted_word_map[int(idx)] for idx in normal_image_out[0].argmax(-1)
    )

    save_image(
        rescale(adversarial_sample.detach()),
        f"samples/adv_{epsilon:.2f}_{filename}",
    )

    assert adversarial_sentence != normal_sentence


def test_adversarial_inference(batch_size, teddy_bear_image, model):
    original_sentences, adversarial_sentences = adversarial.inference(
        torch.cat(batch_size * [teddy_bear_image]), model, 0.0
    )
    torch.testing.assert_allclose(original_sentences, adversarial_sentences)
