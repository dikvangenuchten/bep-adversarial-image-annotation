import json
import nltk
import random
import tensorflow as tf
import tensorflow_hub as hub


with open("/data/captions_val2017.json") as annotation_file:
    annotations = json.load(annotation_file)["annotations"]

merged_captions = {}
for annotation in annotations:
    captions = merged_captions.get(annotation["image_id"], [])
    captions.append(annotation["caption"])
    merged_captions[annotation["image_id"]] = captions

for id in merged_captions.keys():
    merged_captions[id] = merged_captions[id][:5]

hypothesis = []
references = []
for i in range(5):
    for sentences in merged_captions.values():
        sentences_ = sentences.copy()
        hypothesis.append(sentences_.pop(i))
        references.append(sentences_)

human_score_corpus = nltk.bleu_score.corpus_bleu(references, hypothesis)
print(f"human_score_corpus={human_score_corpus}")

embedding_model = hub.load(
    "https://tfhub.dev/google/universal-sentence-encoder/4"
)


references_transposed = list(zip(*references))
hypothesis_embed = embedding_model(hypothesis)
max_sim = None
for i in range(4):
    references_embed = embedding_model(references_transposed[i])
    cosine_sim = -tf.keras.losses.cosine_similarity(
        hypothesis_embed, references_embed
    )
    if max_sim is None:
        max_sim = cosine_sim

    max_sim = tf.math.maximum(max_sim, cosine_sim)

    print(f"human_cosine_similarity={tf.reduce_mean(cosine_sim)}")
print(f"max human_cosine_similarity={tf.reduce_mean(max_sim)}")
