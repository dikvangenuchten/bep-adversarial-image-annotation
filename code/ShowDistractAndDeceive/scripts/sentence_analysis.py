from typing import List
from collections import Counter

# import tensorflow as tf
# import tensorflow_hub as hub
# import torch


def load_data():
    data = []
    for i in range(11):

        with open(f"output/sentence_e:{i/10:.2f}.txt") as file:
            lines = [line.strip() for line in file.readlines()]
        data.append(lines)
    return data


# embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def cosine_similarity(orig_sentences: List[str], adv_sentences: List[str]):
    orig_features = torch.tensor(embed(orig_sentences).numpy())
    adv_features = torch.tensor(embed(adv_sentences).numpy())
    return torch.nn.functional.cosine_similarity(orig_features, adv_features)


if __name__ == "__main__":
    data = load_data()
    for lines in data:
        print(
            Counter(
                word for line in lines for word in line.strip().split(" ")
            ).most_common(10)
        )
