import argparse

import utils
from models import ShowAttendAndTell


def main(model: ShowAttendAndTell):
    pass


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
    model = utils.load_model(args.model_path, args.word_map, args.device)
    main(model)
