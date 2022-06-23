import os
from matplotlib import pyplot as plt
import matplotlib as mtp

import numpy as np
from PIL import Image
import skimage.transform


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


def visualize_att(img, seq, alphas, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image: pytorch tensor of image for captioning
    :param seq: caption
    :param alphas: weights
    :param smooth: smooth weights?
    """
    image = Image.fromarray(
        np.moveaxis((img * 255).cpu().numpy().astype(np.uint8), 0, -1)
    )
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = seq.split(" ")
    # figure = plt.figure()
    plt.clf()
    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(int(np.ceil(len(words) / 5.0)), 5, t + 1)

        plt.text(
            0,
            -50,
            "%s" % (words[t]),
            color="black",
            backgroundcolor="white",
            fontsize=12,
        )
        plt.imshow(image)
        current_alpha = alphas[t, :]
        current_alpha = current_alpha.reshape(14, 14)
        if smooth:
            alpha = skimage.transform.pyramid_expand(
                current_alpha.numpy(), upscale=24, sigma=8
            )
        else:
            alpha = skimage.transform.resize(
                current_alpha.numpy(), [14 * 24, 14 * 24]
            )
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(mtp.cm.Greys_r)
        plt.axis("off")
    return plt

def plot_attention_heatmap(name, attention, epsilon):

    fig, ax = plt.subplots()

    im, cbar = heatmap(
        attention / attention.sum(), ax=ax, cbarlabel="Attention"
    )

    ax.set_title(f"Average Attention for \u03B5: {epsilon:.3f}")
    fig.tight_layout()
    fig.savefig(name)
    fig.clf()


def heatmap(data, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar