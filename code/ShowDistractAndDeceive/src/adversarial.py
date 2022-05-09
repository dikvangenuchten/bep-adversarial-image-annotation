"""Tools to generate adversarial examples.
"""
from typing import Callable
import torch


def generate_gaussian_sample(image, model, target, epsilon: float = 0.1):
    """Generate an adversarial sample based on the fast gradient method."""
    return epsilon * torch.randn_like(image)

def generate_adversarial_noise(image, model, target, epsilon: float = 0.1):
    """Generate adversarial noise with a max value of epsilon"""
    image.requires_grad = True
    prediction, _ = model(image)
    adversarial_loss = torch.nn.functional.cross_entropy(
        prediction.transpose(2, 1), target
    )
    adversarial_loss.backward()
    return epsilon * torch.sign(image.grad)


def inference(image, model, epsilon, adv_method):
    """Do inference on a image batch and the adversarial image batch"""
    prediction, _ = model(image)
    target = prediction.argmax(-1)
    noise = adv_method(image, model, target, epsilon)
    adversarial_prediction, _ = model(image + noise)
    return prediction, adversarial_prediction, image + noise
