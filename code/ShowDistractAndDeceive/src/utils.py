import json
from typing import Union

import imageio
import torch
import torchvision

from models import ShowAttendAndTell


def load_model(
    model_path: str, word_map: Union[str, dict], device: str
) -> ShowAttendAndTell:
    if not isinstance(word_map, dict):
        word_map = load_word_map(word_map)
    return ShowAttendAndTell.load(model_path, word_map, device)


def load_word_map(word_map_path: str):
    with open(word_map_path, "r", encoding="utf-8") as word_map_file:
        word_map = json.load(word_map_file)
    return word_map


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
