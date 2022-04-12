import json
import os

import imageio
import pytest
import torch
import torchvision

import models

MODEL_PATH = "data/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar"
WORD_MAP_PATH = "data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"


@pytest.fixture(params=["cuda", "cpu"], scope="module")
def device(request):
    if request.param == "cpu":
        pytest.skip("skipping cpu test as it is slow.")
    device_ = torch.device(request.param)
    yield device_


@pytest.fixture(scope="module")
def word_map():
    with open(WORD_MAP_PATH, "r", encoding="utf-8") as word_map_file:
        word_map = json.load(word_map_file)
    yield word_map


@pytest.fixture()
def inverted_word_map(word_map):
    return {v: k for k, v in word_map.items()}


@pytest.fixture(
    params=[
        "clean_samples/baseball.jpg",
        "clean_samples/elephant.jpg",
        "clean_samples/kitchen_oven.jpg",
        "clean_samples/tedy_bear.jpg",
    ]
)
def image(request, device):
    return load_image(request.param, device), os.path.basename(request.param)


@pytest.fixture()
def teddy_bear_image(device):
    path = "clean_samples/tedy_bear.jpg"
    return load_image(path, device)


def load_image(path: str, device):
    raw_image = torchvision.transforms.functional.resize(
        torch.FloatTensor(imageio.imread(path).transpose(2, 0, 1)),
        (256, 256),
    )
    # Retrieved from caption.py from ShowAttendAndTell
    normalized_image = torchvision.transforms.functional.normalize(
        raw_image / 255,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    image = normalized_image.to(device)
    return image.unsqueeze(0)


@pytest.fixture()
def model(word_map, device):
    return models.ShowAttendAndTell.load(MODEL_PATH, word_map, device)


@pytest.fixture(params=[1, 2, 4], scope="module")
def batch_size(request):
    yield request.param
