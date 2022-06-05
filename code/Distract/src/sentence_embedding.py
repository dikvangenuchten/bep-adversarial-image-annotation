"""Sentence Embedding to 
"""
from typing import List

import tensorflow as tf
import tensorflow_hub as hub
import torch

tf.config.set_visible_devices(tf.config.list_logical_devices("CPU"))
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def cosine_similarity(orig_sentences: List[str], adv_sentences: List[str]):
    orig_features = torch.tensor(embed(orig_sentences).numpy())
    adv_features = torch.tensor(embed(adv_sentences).numpy())
    return torch.nn.functional.cosine_similarity(orig_features, adv_features)
