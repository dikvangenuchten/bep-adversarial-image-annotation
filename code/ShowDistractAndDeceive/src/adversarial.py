"""Tools to generate adversarial examples.
"""
import torch


def generate_adversarial_noise(image, model, target, epsilon: float = 0.1):
    """Generate adversarial noise with a max value of epsilon"""
    image.requires_grad = True
    prediction, _ = model(image.unsqueeze(0))
    adversarial_loss = torch.nn.functional.cross_entropy(
        prediction.squeeze(0), target.squeeze(0)
    )
    adversarial_loss.backward()
    return epsilon * torch.sign(image.grad)
