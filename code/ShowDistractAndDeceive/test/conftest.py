import os

import pytest
import torch

import utils

MODEL_PATH = "data/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar"
WORD_MAP_PATH = "data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"


@pytest.fixture(params=["cuda", "cpu"], scope="module")
def device(request):
    # if request.param == "cpu":
    #     pytest.skip("skipping cpu test as it is slow.")
    device_ = torch.device(request.param)
    yield device_


@pytest.fixture(scope="module")
def word_map():
    yield utils.load_word_map(WORD_MAP_PATH)


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
    return utils.load_image(request.param, device), os.path.basename(
        request.param
    )


@pytest.fixture()
def teddy_bear_image(device):
    path = "clean_samples/tedy_bear.jpg"
    return utils.load_image(path, device)


@pytest.fixture()
def model(word_map, device):
    return utils.load_model(MODEL_PATH, word_map, device)


@pytest.fixture(params=[1, 2, 4], scope="module")
def batch_size(request):
    yield request.param
