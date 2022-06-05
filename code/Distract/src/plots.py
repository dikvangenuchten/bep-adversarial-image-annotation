from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True

def cosine_similarity_violin_plot(name, all_cosine_similarities, epsilons):
    fig, ax = plt.subplots()
    ax.violinplot(all_cosine_similarities, epsilons, widths=0.1)
    ax.set_title("Cosine Similarity distribution over epsilon")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("cosine similarity")
    ax.set_xticks(epsilons)
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
    ax.set_xbound(min(epsilons), max(epsilons))
    ax.set_ybound(0, 1)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("cosine similarity")
    ax.set_xticks(epsilons)
    fig.savefig(name)
    fig.clf()


def plot_attention_heatmap(name, attention, epsilon):

    fig, ax = plt.subplots()

    im, cbar = heatmap(attention / attention.sum(), ax=ax,
                   cbarlabel="Attention")

    ax.set_title(f"Average Attention for $\epsilon: {epsilon:.3f}$")
    fig.tight_layout()
    fig.savefig(name)
    fig.clf()

def heatmap(data, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
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

    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar