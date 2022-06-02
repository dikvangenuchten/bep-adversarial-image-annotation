"""Tools to generate adversarial examples.
"""
from abc import abstractmethod, ABC
import torch
from models import ShowAttendAndTell


class AbstractAdversarial(ABC):
    def __init__(self, model: ShowAttendAndTell,  targeted: bool):
        """Abstract Interface of"""
        self.model = model
        self.targeted = targeted

    def __call__(self, images, target=None, epsilon=0):
        """Generates the adversarial image."""
        if self.targeted:
            return images - self._generate_noise(images, target, epsilon)
        if target is None:
            target = self.model(images)[0].argmax(-1)
        return images + self._generate_noise(images, target, epsilon)

    @abstractmethod
    def _generate_noise(self, images, target, epsilon):
        """Actual implementation that generates noise."""


class GaussianAdversarial(AbstractAdversarial):
    def _generate_noise(self, images, target, epsilon):
        return _clip(
            (epsilon / 2) * torch.randn_like(images),
            0,
            epsilon,
        )


class GaussianSignAdversarial(AbstractAdversarial):
    def _generate_noise(self, images, target, epsilon):
        return epsilon * torch.sign(torch.randn_like(images))


class FastGradientSignAdversarial(AbstractAdversarial):
    def _generate_noise(self, images, target, epsilon):
        """Generate adversarial noise with a max value of epsilon"""
        images.requires_grad = True

        prediction, _ = self.model(images)
        adversarial_loss = torch.nn.functional.cross_entropy(
            prediction.transpose(2, 1), target, reduction="none"
        )

        if self.targeted:
            # Todo don't hardcode
            loss_weights = _calculate_loss_weights(target, 9489)
            adversarial_loss *= loss_weights

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
        or_image = images.clone().detach()
        for _ in range(self.iterations):
            alpha =  (
            epsilon * self.alpha_multiplier
        ) / self.iterations
            images = _clip(
                self.adversarial_method(images, target, alpha),
                or_image,
                epsilon,
            ).detach()
        return images

def adversarial_inference(method, images, target, epsilon):
    noise = method(images, target, epsilon)
    
    prediction, _ = method.model(images)
    adv_images = images + noise
    adv_prediction, _ = method.model(adv_images)
    return prediction, adv_prediction, adv_images
    

def _clip(adv_images, original_images, epsilon):
    return torch.clamp(
        adv_images,
        min=original_images - epsilon,
        max=original_images + epsilon,
    )


def _calculate_loss_weights(target, end_token):
    # Include first end_token
    weight = torch.roll(target != end_token, 1, -1)
    # Include first word that got overwritten by torch.roll
    weight[:, 0] = True
    return weight
