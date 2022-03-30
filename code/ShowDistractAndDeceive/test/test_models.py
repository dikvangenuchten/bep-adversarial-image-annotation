import json
import pytest
import torch
import torchvision
import models
import imageio

MODEL_PATH = "data/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar"
WORD_MAP_PATH = "data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"


@pytest.fixture()
def word_map():
    with open(WORD_MAP_PATH, "r", encoding="utf-8") as word_map_file:
        word_map = json.load(word_map_file)
    yield word_map


@pytest.fixture()
def inverted_word_map(word_map):
    return {v: k for k, v in word_map.items()}


@pytest.fixture()
def teddy_bear_image():
    path = "test/tedy_bear.jpg"
    teddy_bear = torchvision.transforms.functional.resize(
        torch.FloatTensor(imageio.imread(path).transpose(2, 0, 1)),
        (256, 256),
    )
    # Retrieved from caption.py from ShowAttendAndTell
    normalized_teddy_bear = torchvision.transforms.functional.normalize(
        teddy_bear / 255,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return normalized_teddy_bear.to(models.DEVICE)


@pytest.fixture()
def model(word_map):
    return models.ShowAttendAndTell.load(MODEL_PATH, word_map)


def test_load_model(word_map, model):
    random_input = torch.rand(1, 3, 256, 256).to(models.DEVICE)

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


def test_backpropagation(model, teddy_bear_image):
    teddy_bear_image.requires_grad = True
    output, i = model(teddy_bear_image.unsqueeze(0))
    pass


@pytest.mark.parametrize("i", range(1, 5))
def test_sentence_length(benchmark, model, teddy_bear_image, i):
    pytest.skip()
    model.max_sentence_length = i
    input_ = teddy_bear_image.unsqueeze(0).to(models.DEVICE)
    encoded_sentence, length = benchmark(model, input_)

    assert length == i - 1
