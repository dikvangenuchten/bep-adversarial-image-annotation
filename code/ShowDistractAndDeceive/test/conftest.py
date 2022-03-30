import json

import imageio
import pytest
import torch
import torchvision

import models

MODEL_PATH = "data/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar"
WORD_MAP_PATH = "data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"


@pytest.fixture(params=["cuda", "cpu"])
def device(request):
    # device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ = torch.device(request.param)
    return device_


@pytest.fixture()
def word_map():
    with open(WORD_MAP_PATH, "r", encoding="utf-8") as word_map_file:
        word_map = json.load(word_map_file)
    yield word_map


@pytest.fixture()
def inverted_word_map(word_map):
    return {v: k for k, v in word_map.items()}


@pytest.fixture()
def teddy_bear_image(device):
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
    return normalized_teddy_bear.to(device)


@pytest.fixture()
def model(word_map, device):
    return models.ShowAttendAndTell.load(MODEL_PATH, word_map, device)
