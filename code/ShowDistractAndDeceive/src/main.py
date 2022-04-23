import argparse
from adversarial import inference

import utils
from models import ShowAttendAndTell
import data_loader


def main(model: ShowAttendAndTell, dataloader, word_map):
    inverted_word_map = utils.invert_word_map(word_map)

    for image, labels in dataloader:
        print(labels)
        orig_pred, adv_pred = inference(image, model, 0.1)
        orig_sentences = utils.decode_prediction(inverted_word_map, orig_pred)
        adv_sentences = utils.decode_prediction(inverted_word_map, adv_pred)

        print(orig_sentences)
        print(adv_sentences)


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
    dataset = data_loader.get_data_loader(args.device, batch_size=2)

    main(model, dataset, word_map)
