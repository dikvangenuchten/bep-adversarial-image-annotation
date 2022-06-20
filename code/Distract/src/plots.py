import matplotlib as mtp
from matplotlib import pyplot as plt
import numpy as np


def cosine_similarity_violin_plot(name, all_cosine_similarities, epsilons):
    fig, ax = plt.subplots()
    ax.violinplot(all_cosine_similarities, epsilons, widths=0.1)
    ax.set_xscale("symlog", base=2, linthresh=0.01)
    ax.xaxis.set_major_formatter(mtp.ticker.ScalarFormatter())
    ax.set_title("Cosine Similarity distribution over epsilon")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("cosine similarity")
    fig.savefig(name)
    fig.clf()


def cosine_similarity_heatmap(name, all_cosine_similarities, epsilons):
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
    fig.savefig(name)
    fig.clf()


def plot_bleu_scores(name, bleu_scores, epsilons):
    fig, ax = plt.subplots()
    ax.plot(epsilons, bleu_scores)
    ax.set_xscale("symlog", base=2, linthresh=0.01)
    ax.xaxis.set_major_formatter(mtp.ticker.ScalarFormatter())
    ax.set_title("Bleu Score vs epsilon")
    ax.set_xbound(min(epsilons), max(epsilons))
    ax.set_ybound(0, 1)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Bleu Score")
    ax.set_xticks(epsilons)
    fig.savefig(name)
    fig.clf()


def plot_average_cosine_similarity(name, all_cosine_similarities, epsilons):
    fig, ax = plt.subplots()
    ax.plot(epsilons, [np.mean(x) for x in all_cosine_similarities])
    ax.set_title("Average Cosine Similarity vs epsilon")
    ax.set_xscale("symlog", base=2, linthresh=0.01)
    ax.xaxis.set_major_formatter(mtp.ticker.ScalarFormatter())
    ax.set_xbound(min(epsilons), max(epsilons))
    ax.set_ybound(0, 1)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("cosine similarity")
    ax.set_xticks(epsilons)
    fig.savefig(name)
    fig.clf()
