"""Tools to generate adversarial examples.
"""
from abc import abstractmethod, ABC
import torch
from models import ShowAttendAndTell


class AbstractAdversarial(ABC):
    def __init__(self, model: ShowAttendAndTell, targeted: bool):
        """Abstract Interface of"""
        self.model = model
        self.targeted = targeted

    def __call__(self, images, target=None, epsilon=0):
        """Generates the adversarial image."""
        return images - self._generate_noise(images, target, epsilon)

    @abstractmethod
    def _generate_noise(self, images, target, epsilon):
        """Actual implementation that generates noise."""


class GaussianAdversarial(AbstractAdversarial):
    def _generate_noise(self, images, target, epsilon):
        return _clip((epsilon / 2) * torch.randn_like(images), epsilon)


class GaussianSignAdversarial(AbstractAdversarial):
    def _generate_noise(self, images, target, epsilon):
        return epsilon * torch.sign(torch.randn_like(images))


class FastGradientSignAdversarial(AbstractAdversarial):
    def _generate_noise(self, images, target, epsilon):
        """Generate adversarial noise with a max value of epsilon"""
        images.requires_grad = True

        prediction, _, attention = self.model(images)
        attention = attention.reshape(-1, 196)
        adversarial_loss = torch.nn.functional.cross_entropy(
            attention,
            torch.ones(
                attention.shape[0],
                device=attention.device,
                dtype=torch.long,
            ),
        )

        loss = adversarial_loss.mean()
        loss.backward()
        return epsilon * torch.sign(images.grad)


class IterativeAdversarial:
    def __init__(
        self,
        adversarial_method: AbstractAdversarial,
        iterations: int = 10,
        alpha_multiplier: float = 1,
    ):
        self.adversarial_method = adversarial_method
        self.alpha_multiplier = alpha_multiplier
        self.iterations = iterations

    def __call__(self, images, target, epsilon):
        or_images = images.clone().detach()
        acc_noise = torch.zeros_like(images)
        for _ in range(self.iterations):
            alpha = (epsilon * self.alpha_multiplier) / self.iterations
            images = torch.clamp(or_images + acc_noise, min=0, max=1)
            acc_noise += self.adversarial_method(images, target, alpha)
            acc_noise = _clip(acc_noise, epsilon)
        return acc_noise

    @property
    def model(self):
        return self.adversarial_method.model


def adversarial_inference(method, images, target, epsilon):
    noise = method(images, target, epsilon)

    prediction, _, attention = method.model(images)
    adv_images = torch.clamp(images + noise, min=0, max=1)
    adv_prediction, _, adv_attention = method.model(adv_images)
    return (
        prediction.detach(),
        adv_prediction.detach(),
        adv_images.detach(),
        attention.detach(),
        adv_attention.detach(),
    )


def _clip(adv_images, epsilon):
    return torch.clamp(adv_images, min=-epsilon, max=epsilon)


def _calculate_loss_weights(target, end_token):
    # Include first end_token
    weight = torch.roll(target != end_token, 1, -1)
    # Include first word that got overwritten by torch.roll
    weight[:, 0] = True
    return weight
