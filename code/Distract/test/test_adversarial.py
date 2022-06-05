"""Tests for adversarial.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torchvision.utils import save_image

import adversarial
import utils
from utils import rescale


@pytest.fixture(params=[0.05, 0.1, 0.2, 0.3, 1])
def epsilon(request):
    return request.param


@pytest.fixture(params=[True, False])
def targeted(request):
    return request.param


@pytest.fixture(
    params=[
        adversarial.FastGradientSignAdversarial,
        adversarial.GaussianAdversarial,
        adversarial.GaussianSignAdversarial,
    ]
)
def method(request):
    return request.param


@pytest.fixture()
def adversarial_method(model, method, targeted):
    return method(model, targeted)


def test_generate_adversarial_example(
    image, model, inverted_word_map, adversarial_method, epsilon
):

    image, filename = image
    normal_image_out, i, _ = model(image)

    target = normal_image_out.argmax(-1)

    adversarial_noise = adversarial_method(image, target, epsilon) - image
    assert adversarial_noise.shape == image.shape
    assert (
        adversarial_noise.max() <= epsilon + 1e-6
    ), f"Max seen value: {adversarial_noise.max()} is bigger then epsilon ({epsilon})"
    assert (
        adversarial_noise.min() >= -epsilon - 1e-6
    ), f"Min seen value: {adversarial_noise.min()} is smaller then -epsilon ({-epsilon})"

    adversarial_sample = image + adversarial_noise
    adversarial_out, i, _ = model(adversarial_sample)

    adversarial_sentence = " ".join(
        inverted_word_map[int(idx)] for idx in adversarial_out[0].argmax(-1)
    )
    normal_sentence = " ".join(
        inverted_word_map[int(idx)] for idx in normal_image_out[0].argmax(-1)
    )

    save_image(rescale(image), f"samples/{filename}")

    save_image(
        rescale(adversarial_noise),
        f"samples/adv_noise_{epsilon:.3f}_{filename}",
    )

    save_image(
        rescale(adversarial_sample), f"samples/adv_{epsilon:.3f}_{filename}"
    )

    with open(f"samples/text_{epsilon:.3f}_{filename}.txt", "w") as file:
        file.write(f"original: {normal_sentence}\n")
        file.write(f"adversarial: {adversarial_sentence}")

    assert adversarial_sentence != normal_sentence


def test_adversarial_inference_to_target_sentence(
    model, image, word_map, device, inverted_word_map, epsilon
):
    image, filename = image
    adversarial_method = adversarial.IterativeAdversarial(
        adversarial_method=adversarial.FastGradientSignAdversarial(
            model=model, targeted=True
        ),
        iterations=100,
        alpha_multiplier=20,
    )


    prediction, adv_prediction, adv_image, att, adv_att = adversarial.adversarial_inference(
        adversarial_method, image, None, epsilon
    )
    predicted_sentence = utils.decode_prediction(inverted_word_map, prediction)
    adv_predicted_sentence = utils.decode_prediction(
        inverted_word_map, adv_prediction
    )

    save_image(rescale(image), f"samples/{filename}")

    save_image(
        rescale((adv_image - image)),
        f"samples/target_noise_{epsilon:.3f}_{filename}",
    )

    save_image(rescale(adv_image), f"samples/target_{epsilon:.3f}_{filename}")

    with open(f"samples/target_text_{epsilon:.3f}_{filename}.txt", "w") as file:
        file.write(f"original: {predicted_sentence}")
        file.write(f"adversarial: {adv_predicted_sentence}")

    assert att != adv_att


# def test_adversarial_inference(batch_size, teddy_bear_image, model):
#     original_sentences, adversarial_sentences, _ = adversarial.inference(
#         torch.cat(batch_size * [teddy_bear_image]), model, 0.0
#     )
#     torch.testing.assert_allclose(original_sentences, adversarial_sentences)
