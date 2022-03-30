import pytest
import torch


def test_load_model(model, device):
    random_input = torch.rand(1, 3, 256, 256).to(device)

    encoded_sentence = model(random_input)

    assert encoded_sentence


def test_inference_on_teddy_bear(
    model,
    teddy_bear_image,
    inverted_word_map,
):
    input_ = teddy_bear_image.unsqueeze(0)
    scores, i = model(input_)
    # Based on caption.py from ShowAttendAndTell
    expected_string = "a group of stuffed animals sitting on top of a couch"
    words = []
    for token in scores.argmax(-1)[0]:
        word = inverted_word_map[int(token)]
        if word == "<end>":
            break
        words.append(word)
    # decoded_sentence = [inverted_word_map[token] for token in scores.argmax(1)]

    assert (
        " ".join(words) == expected_string
    ), f"expected string:\n{expected_string}\ngot:\n{' '.join(words)}"


def test_backpropagation(model, teddy_bear_image, device):
    teddy_bear_image.requires_grad = True
    output, i = model(teddy_bear_image.unsqueeze(0))
    target = torch.tensor([1] * (i + 1), dtype=torch.long).to(device)
    loss = torch.nn.functional.cross_entropy(
        output.transpose(1, 2), target.unsqueeze(0)
    )
    loss.backward()
    assert teddy_bear_image.grad is not None


@pytest.mark.parametrize("i", range(1, 5))
def test_sentence_length(benchmark, model, teddy_bear_image, device, i):
    pytest.skip()
    model.max_sentence_length = i
    input_ = teddy_bear_image.unsqueeze(0).to(device)
    encoded_sentence, length = benchmark(model, input_)

    assert length == i - 1
