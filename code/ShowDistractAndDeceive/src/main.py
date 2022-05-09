import argparse

import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
from matplotlib import pyplot as plt

import data_loader
import sentence_embedding
import utils
import wandb
import adversarial
from models import ShowAttendAndTell

ADV_METHOD = adversarial.generate_gaussian_sample

def epoch():
    pass


def main(model: ShowAttendAndTell, dataloader, word_map):
    inverted_word_map = utils.invert_word_map(word_map)

    wandb.init(job_type="Adversarial Image Caption")
    bleu_scores = []
    all_cosine_similarities = []
    epsilons = []

    for epsilon in tqdm(np.linspace(0, 1, 11)):
        similarities = []
        samples = []
        all_labels = []
        all_adv_sentences = []
        for image, labels in tqdm(dataloader, leave=False):
            orig_pred, adv_pred, adv_img = adversarial.inference(image, model, epsilon, ADV_METHOD)
            orig_sentences = utils.decode_prediction(
                inverted_word_map, orig_pred
            )
            adv_sentences = utils.decode_prediction(inverted_word_map, adv_pred)

            similartity = sentence_embedding.cosine_similarity(
                orig_sentences, adv_sentences
            )

            # labels are transposed for some reason
            all_labels.extend(zip(*labels))
            all_adv_sentences.extend(adv_sentences)

            if len(samples) < 40:
                samples.extend(
                    wandb.Image(
                        img,
                        caption=f"""original: {ori_caption}
                        \nadversarial: {adv_caption}
                        \ncosine similarity: {cos_sim:.3f}""",
                    )
                    for img, ori_caption, adv_caption, cos_sim in zip(
                        adv_img, orig_sentences, adv_sentences, similartity
                    )
                )
            similarities.append(similartity)

        cosine_similarities = torch.concat(similarities).numpy()
        bleu_score = corpus_bleu(all_labels, all_adv_sentences)
        wandb.log(
            {
                "epsilon": epsilon,
                "cosine similarities": wandb.Histogram(
                    np_histogram=np.histogram(
                        cosine_similarities,
                        bins=list(np.linspace(0, 1, 11, endpoint=True)),
                    )
                ),
                "adversarial samples": samples,
                "bleu score": bleu_score,
            }
        )
        epsilons.append(epsilon)
        bleu_scores.append(bleu_score)
        all_cosine_similarities.append(cosine_similarities)

    fig, ax = plt.subplots()
    ax.violinplot(all_cosine_similarities, epsilons, widths=0.1)
    ax.set_title("Cosine Similarity distribution over epsilon")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("cosine similarity")
    fig.savefig("violin_cosine_plot")
    fig.clf()

    fig, ax = plt.subplots()
    heatmap, xedges, yedges = np.histogram2d(
        *list(
            zip(
                *[
                    (sim, eps)
                    for (sims, eps) in zip(all_cosine_similarities, epsilons)
                    for sim in sims
                ]
            )
        ),
        bins=(11, 11),
    )
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    ax.imshow(heatmap, extent=extent, origin="lower")
    ax.set_title("Heatmap cosine similarity vs epsilon")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("cosine similarity")
    fig.savefig("heatmap_cosine_vs_epsilon")
    fig.clf()

    fig, ax = plt.subplots()
    ax.plot(epsilons, bleu_scores)
    ax.set_title("Bleu Score vs epsilon")
    ax.set_xbound(0, 1)
    ax.set_ybound(0, 1)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Bleu Score")
    ax.set_xticks(epsilons)
    fig.savefig("bleu_score_plot")
    fig.clf()

    fig, ax = plt.subplots()
    ax.plot(epsilons, [np.mean(x) for x in all_cosine_similarities])
    ax.set_title("Average Cosine Similarity vs epsilon")
    ax.set_xbound(0, 1)
    ax.set_ybound(0, 1)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("cosine similarity")
    ax.set_xticks(epsilons)
    fig.savefig("average_cosine_sim_plot")
    fig.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ShowDistractandDeceive: Adversarial Image annotation tool"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="data/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar",
        help="Path to the show attend and tell model checkpoint.",
    )
    parser.add_argument(
        "--word-map",
        type=str,
        default="data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json",
        help="Path to the json wordmap of the model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Wheter to use the gpu, default true",
    )
    args = parser.parse_args()
    word_map = utils.load_word_map(args.word_map)
    model = utils.load_model(args.model_path, word_map, args.device)
    dataset = data_loader.get_data_loader(args.device, batch_size=6)

    main(model, dataset, word_map)
