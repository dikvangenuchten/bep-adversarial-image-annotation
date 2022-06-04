import pytest
import torch

import utils


def test_load_model(model, device):
    random_input = torch.rand(1, 3, 256, 256).to(device)

    encoded_sentence = model(random_input)

    assert encoded_sentence


def test_inference_on_teddy_bear(
    model,
    teddy_bear_image,
    inverted_word_map,
):
    input_ = teddy_bear_image
    scores, i = model(input_)
    # Based on caption.py from ShowAttendAndTell
    expected_string = "a group of stuffed animals sitting on top of a couch"
    predicted_sentence = utils.decode_prediction(inverted_word_map, scores)[0]
    # decoded_sentence = [inverted_word_map[token] for token in scores.argmax(1)]

    assert (
        predicted_sentence == expected_string
    ), f"expected string:\n{expected_string}\ngot:\n{predicted_sentence}"


def test_backpropagation(model, teddy_bear_image, device):
    teddy_bear_image.requires_grad = True
    output, i = model(teddy_bear_image)
    target = output.argmax(-1)
    loss = torch.nn.functional.cross_entropy(output.transpose(2, 1), target)
    loss.backward()
    assert teddy_bear_image.grad is not None


@pytest.mark.parametrize("i", range(1, 5))
def test_sentence_length(benchmark, model, teddy_bear_image, device, i):
    model.max_sentence_length = i
    input_ = teddy_bear_image.to(device)
    encoded_sentence, length = benchmark(model, input_)

    assert length == i - 1
