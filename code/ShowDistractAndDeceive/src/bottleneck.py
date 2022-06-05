import torch

import adversarial
import utils

MODEL_PATH = "/data/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar"
WORD_MAP_PATH = "/data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"


def device(device):
    device_ = torch.device(device)
    return device_


def word_map():
    return utils.load_word_map(WORD_MAP_PATH)


def inverted_word_map(word_map):
    return utils.invert_word_map(word_map)


def teddy_bear_image(device):
    path = "test/clean_samples/tedy_bear.jpg"
    return utils.load_image(path, device)


def model(word_map, device):
    return utils.load_model(MODEL_PATH, word_map, device)


def test_adversarial_inference_to_target_sentence(
    model, teddy_bear_image, word_map, device, inverted_word_map
):
    adversarial_method = adversarial.IterativeAdversarial(
        adversarial_method=adversarial.FastGradientSignAdversarial(
            model=model, targeted=True, epsilon=1
        ),
        iterations=10,
        alpha_multiplier=2,
    )

    adversarial_sentence = "this is an attack on show attend and tell"

    target_sentence = utils.pad_target_sentence(
        utils.sentence_to_tokens(adversarial_sentence, word_map).to(device),
        word_map,
        model.max_sentence_length,
    ).unsqueeze(0)

    adversarial_image = adversarial_method(teddy_bear_image, target_sentence)

    prediction, _ = model(adversarial_image)
    predicted_sentence = utils.decode_prediction(inverted_word_map, prediction)

    # assert adversarial_sentence == predicted_sentence[0]


if __name__ == "__main__":
    _device = device("cuda")
    _word_map = word_map()
    _model = model(_word_map, _device)
    _image = teddy_bear_image(_device)
    _inverted_word_map = inverted_word_map(_word_map)
    for _ in range(10):
        test_adversarial_inference_to_target_sentence(
            _model, _image, _word_map, _device, _inverted_word_map
        )
