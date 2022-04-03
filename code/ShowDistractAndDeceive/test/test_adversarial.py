"""Tests for adversarial.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from adversarial import generate_adversarial_noise


def rescale(img):
    img -= img.min()
    img /= img.max()
    return img


@pytest.mark.parametrize("epsilon", np.linspace(0, 2, 21))
def test_generate_adversarial_example(
    teddy_bear_image, model, device, inverted_word_map, epsilon
):
    teddy_bear_out, i = model(teddy_bear_image.unsqueeze(0))
    # Always predict: 'a'
    # target = torch.tensor([[1] * 50], dtype=torch.long).to(device)

    target = teddy_bear_out.argmax(-1)

    adversarial_noise = generate_adversarial_noise(
        teddy_bear_image, model, target, epsilon
    )
    assert adversarial_noise.shape == teddy_bear_image.shape
    assert adversarial_noise.max() == epsilon
    assert adversarial_noise.min() == -epsilon

    adversarial_sample = teddy_bear_image - adversarial_noise
    adversarial_out, i = model(adversarial_sample.unsqueeze(0))

    adversarial_sentence = " ".join(
        inverted_word_map[int(idx)] for idx in adversarial_out[0].argmax(-1)
    )
    teddy_bear_sentence = " ".join(
        inverted_word_map[int(idx)] for idx in teddy_bear_out[0].argmax(-1)
    )

    plt.imsave(
        f"samples/teddy_bear_image.jpg",
        rescale(teddy_bear_image.detach().cpu().permute(1, 2, 0)).numpy(),
    )
    with open(f"samples/teddy_bear_sample.txt", "w") as file:
        file.write(teddy_bear_sentence)

    plt.imsave(
        f"samples/adversarial_noise_{epsilon:2f}.jpg",
        rescale(adversarial_noise.detach().cpu().permute(1, 2, 0)).numpy(),
    )
    plt.imsave(
        f"samples/adversarial_sample_{epsilon:2f}.jpg",
        rescale(adversarial_sample.detach().cpu().permute(1, 2, 0)).numpy(),
    )
    with open(f"samples/adversarial_sample_{epsilon:2f}.txt", "w") as file:
        file.write(adversarial_sentence)
        file.write(
            f"""
            \rmax: {adversarial_sample.max()}\n
            \rmin: {adversarial_sample.min()}\n
            \rnoise: {epsilon:2d}
            """
        )

    assert adversarial_sentence != teddy_bear_sentence or epsilon == 0
