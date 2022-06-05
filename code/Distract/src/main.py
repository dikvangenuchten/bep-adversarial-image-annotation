import argparse
from typing import Callable

import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

import plots
import data_loader
import sentence_embedding
import utils
import wandb
import adversarial
from models import ShowAttendAndTell

ADV_METHODS = {
    "gaussian": adversarial.GaussianAdversarial,
    "gaussian sign": adversarial.GaussianSignAdversarial,
    "fast gradient": adversarial.FastGradientSignAdversarial,
}


def main(
    model: ShowAttendAndTell,
    dataloader,
    word_map,
    adversarial_method: Callable,
    epsilons,
    target=None,
):
    inverted_word_map = utils.invert_word_map(word_map)

    # Prepare target
    if target is not None:
        target = utils.pad_target_sentence(
            target, word_map, model.max_sentence_length
        )

    wandb.init(
        project="Bachelor End Project",
        tags=[adversarial_method.__class__.__name__],
        name=f"Adversarial Image Caption: {adversarial_method.__class__.__name__}",
    )
    bleu_scores = []
    all_cosine_similarities = []

    for epsilon in tqdm(epsilons):
        cosine_similarities, bleu_score, samples, noise, or_att, ad_att = epoch(
            dataloader=dataloader,
            inverted_word_map=inverted_word_map,
            epsilon=epsilon,
            adv_func=adversarial_method,
            target=target,
        )
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
                "noise": noise,
                "attention": [or_att, ad_att],
                "bleu score": bleu_score,
            }
        )
        bleu_scores.append(bleu_score)
        all_cosine_similarities.append(cosine_similarities)

    plots.cosine_similarity_violin_plot(
        f"cosine_similarity_violin_plot_{adversarial_method.__class__.__name__}.jpg",
        all_cosine_similarities,
        epsilons,
    )
    plots.cosine_similarity_heatmap(
        f"cosine_similarity_heatmap_{adversarial_method.__class__.__name__}.jpg",
        all_cosine_similarities,
        epsilons,
    )
    plots.plot_bleu_scores(
        f"plot_bleu_scores_{adversarial_method.__class__.__name__}.jpg",
        bleu_scores,
        epsilons,
    )
    plots.plot_average_cosine_similarity(
        f"plot_average_cosine_similarity_{adversarial_method.__class__.__name__}.jpg",
        all_cosine_similarities,
        epsilons,
    )


def epoch(dataloader, inverted_word_map, epsilon, adv_func, target=None):
    similarities = []
    samples = []
    noise = []
    all_labels = []
    all_adv_sentences = []
    original_attention = None
    adversarial_attention = None
    if target is not None:
        target_sentence = utils.decode_label(inverted_word_map, target)
        base_target = target
    for image, labels in tqdm(dataloader, leave=False):

        if target is not None:
            target = base_target.repeat([image.size(0), 1])
            target_sentences = [target_sentence] * image.size(0)

        (
            orig_pred,
            adv_pred,
            adv_img,
            attention,
            adv_attention,
        ) = adversarial.adversarial_inference(adv_func, image, target, epsilon)

        if original_attention is None:
            original_attention = torch.zeros_like(attention).sum(dim=[0, 1])
            adversarial_attention = torch.zeros_like(adv_attention).sum(
                dim=[0, 1]
            )

        original_attention += attention.sum(dim=[0, 1])
        adversarial_attention += adv_attention.sum(dim=[0, 1])

        if target is None:
            target_sentences = utils.decode_prediction(
                inverted_word_map, orig_pred
            )

        adv_sentences = utils.decode_prediction(inverted_word_map, adv_pred)

        similartity = sentence_embedding.cosine_similarity(
            target_sentences, adv_sentences
        )

        if target is None:
            # labels are transposed to ensure batch is in the correct spot
            all_labels.extend(zip(*labels))
        else:
            all_labels.extend(zip(*[target]))
        all_adv_sentences.extend(adv_sentences)

        if len(samples) < 40:
            samples.extend(
                wandb.Image(
                    img,
                    caption=f"original: {ori_caption}\n"
                    f"adversarial: {adv_caption}\n"
                    f"cosine similarity: {cos_sim:.3f}",
                )
                for img, ori_caption, adv_caption, cos_sim in zip(
                    adv_img, target_sentences, adv_sentences, similartity
                )
            )

            noise.extend(
                wandb.Image(
                    utils.rescale(adv_image - img),
                    caption=f"epsilon: {epsilon}",
                )
                for img, adv_image in zip(image, adv_img)
            )
        similarities.append(similartity)

    with open(
        f"output/sentence_e:{epsilon:.3f}.txt", "w", encoding="utf-8"
    ) as sentence_file:
        sentence_file.write("\n".join(all_adv_sentences))

    # Reshape attention
    or_att = wandb.Image(
        torch.reshape(original_attention, (14, 14)) / original_attention.max(),
        caption="Average Attention Clean images",
    )

    ad_att = wandb.Image(
        torch.reshape(adversarial_attention, (14, 14))
        / adversarial_attention.max(),
        caption="Average Attention Adversarial images",
    )

    cosine_similarities = torch.concat(similarities).numpy()
    bleu_score = corpus_bleu(all_labels, all_adv_sentences)
    return cosine_similarities, bleu_score, samples, noise, or_att, ad_att


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ShowDistractandDeceive: Adversarial Image annotation tool"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar",
        help="Path to the show attend and tell model checkpoint.",
    )
    parser.add_argument(
        "--word-map",
        type=str,
        default="/data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json",
        help="Path to the json wordmap of the model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Wheter to use the gpu, default true.",
    )
    parser.add_argument(
        "--adversarial-method",
        type=str,
        default="fast gradient",
        help="Which adversarial noise generation method should be used."
        """One of:
        gaussian: Generates gaussian noise,
        fast gradient: Uses the fast gradient method to generate 
            adversarial noise to let the model predict as differently as possible
        target fast gradient: Must also give --target sentence option.
            Steers the model in the direction of target sentence
        """,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        required=False,
        default=1,
        help="How many iterations of the adversarial attack should be done.\n"
        "Applies `epsilon/iteration` of noise for `iteration` amount of times.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=False,
        default=1,
        help="The alpha multiplier used for the iterative method.\n"
        "The epsilon per step is calculated as: epsilon * alpha / iterations.\n"
        "The result is clipped after each iteration to ensure a max deviation of epsilon.",
    )
    parser.add_argument(
        "--target-sentence",
        type=str,
        required=False,
        # default="this is an attack on show attend and tell",
        help="Only works in combination with fast gradient adversarial method."
        "If given this will be used as target sentence during generating of the noise.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        required=False,
        default=None,
        help="Limit the dataset to n amount of samples.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=2,
        help="The batch size used during inference.",
    )
    args = parser.parse_args()
    word_map = utils.load_word_map(args.word_map)
    model = utils.load_model(args.model_path, word_map, args.device)
    dataset = data_loader.get_data_loader(
        args.device, batch_size=args.batch_size, size=args.limit_samples
    )
    assert (
        args.adversarial_method in ADV_METHODS
    ), f"Unknown adversarial method: {args.adversarial_method}.\n"
    f"Must be one of {list(ADV_METHODS.keys())}"
    target_sentence = args.target_sentence
    if target_sentence is not None:
        target_sentence = utils.sentence_to_tokens(
            target_sentence, word_map
        ).to(args.device)

    adversarial_method_class = ADV_METHODS.get(args.adversarial_method)

    targeted = args.target_sentence is not None

    adv_method = adversarial_method_class(model, targeted)

    if args.iterations > 1:
        adv_method = adversarial.IterativeAdversarial(
            adversarial_method=adv_method,
            iterations=args.iterations,
            alpha_multiplier=args.alpha,
        )

    epsilons = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]

    print(f"Starting run with the following args:\n{args}")
    main(
        model=model,
        dataloader=dataset,
        word_map=word_map,
        adversarial_method=adv_method,
        epsilons=epsilons,
        target=target_sentence,
    )
