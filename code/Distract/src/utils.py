import json
from typing import Union

import imageio
import torch
import torchvision

from models import ShowAttendAndTell


def rescale(img):
    img = img.detach()
    img -= img.min()
    img /= img.max()
    return img


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
        torch.FloatTensor(imageio.imread(path).transpose(2, 0, 1)), (256, 256)
    )
    # Retrieved from caption.py from ShowAttendAndTell
    normalized_image = raw_image / 255
    image = normalized_image.to(device)
    return image.unsqueeze(0)


def sentence_to_tokens(sentence: str, word_map: dict):
    return torch.tensor(
        list(word_map[word] for word in sentence.lower().split(" ")),
        dtype=torch.int64,
    )


def invert_word_map(word_map):
    return {v: k for k, v in word_map.items()}


def decode_prediction(inverted_word_map, scores):
    sentences = []
    for sentence_scores in scores:
        words = []
        for token in sentence_scores.argmax(-1):
            word = inverted_word_map[int(token)]
            if word == "<end>":
                break
            words.append(word)
        sentences.append(" ".join(words))
    return sentences


def pad_target_sentence(
    target_sentence: str, word_map: dict, sentence_length: int
):
    end_token = word_map["<end>"]
    target_length = len(target_sentence)

    padded_target = torch.nn.functional.pad(
        target_sentence, (0, sentence_length - target_length), value=end_token
    )

    return torch.tensor(padded_target)


def decode_label(inverted_word_map, encoded_label):
    words = []
    for token in encoded_label:
        word = inverted_word_map[int(token)]
        if word == "<end>":
            break
        words.append(word)
    return " ".join(words)
