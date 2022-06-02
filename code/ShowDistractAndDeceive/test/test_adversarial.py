"""Tests for adversarial.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torchvision.utils import save_image

import adversarial
import utils


def rescale(img):
    img -= img.min()
    img /= img.max()
    return img


@pytest.fixture(params=[0.1, 1])
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
    ],
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
    normal_image_out, i = model(image)

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
    adversarial_out, i = model(adversarial_sample)

    adversarial_sentence = " ".join(
        inverted_word_map[int(idx)] for idx in adversarial_out[0].argmax(-1)
    )
    normal_sentence = " ".join(
        inverted_word_map[int(idx)] for idx in normal_image_out[0].argmax(-1)
    )

    save_image(
        rescale(image),
        f"samples/{filename}"
    )
    
    save_image(
        rescale(adversarial_noise.detach()),
        f"samples/adv_noise_{epsilon:.2f}_{filename}"
    )

    save_image(
        rescale(adversarial_sample.detach()),
        f"samples/adv_{epsilon:.2f}_{filename}",
    )

    with open(f"samples/text_{epsilon:.2f}_{filename}.txt", "w") as file:
        file.write(f"original: {normal_sentence}\n")
        file.write(f"adversarial: {adversarial_sentence}")

    assert adversarial_sentence != normal_sentence


def test_adversarial_inference_to_target_sentence(
    model, teddy_bear_image, word_map, device, inverted_word_map, epsilon
):
    adversarial_method = adversarial.IterativeAdversarial(
        adversarial_method=adversarial.FastGradientSignAdversarial(
            model=model,
            targeted=True,
        ),
        iterations=1000,
        alpha_multiplier=2,
    )

    adversarial_sentence = "this is an attack on show attend and tell"

    target_sentence = utils.pad_target_sentence(
        utils.sentence_to_tokens(adversarial_sentence, word_map).to(device),
        word_map,
        model.max_sentence_length,
    ).unsqueeze(0)


    adversarial_image = adversarial_method(
        teddy_bear_image, target_sentence, epsilon
    )

    prediction, _ = model(adversarial_image)
    predicted_sentence = utils.decode_prediction(inverted_word_map, prediction)

    assert adversarial_sentence == predicted_sentence[0]


# def test_adversarial_inference(batch_size, teddy_bear_image, model):
#     original_sentences, adversarial_sentences, _ = adversarial.inference(
#         torch.cat(batch_size * [teddy_bear_image]), model, 0.0
#     )
#     torch.testing.assert_allclose(original_sentences, adversarial_sentences)
