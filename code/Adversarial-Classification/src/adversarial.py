from typing import Callable
import torch
import torchvision
import wandb

from model import MnistModel


def generate_adversarial_sample(
    model: torch.nn.Module,
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    target: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Generate an adversarial sample based on the fast gradient method."""
    x.requires_grad = True
    output = model(x)
    loss = loss_func(output, target)
    loss.backward()

    return x + epsilon * torch.sign(x.grad)


def generate_log_image(x_or, x_ad, pred_or, pred_ad, y) -> wandb.Image:
    orig = torch.argmax(pred_or)
    caption = f"""Original prediction: {orig} (score: {pred_or[orig]:0.3f})\n"""
    ad = torch.argmax(pred_ad)
    caption += f"""Adversarial prediction: {ad} (score: {pred_ad[ad]:0.3f})\n"""
    caption += f"""Target: {y}\nLeft Original, Right Adversarial"""

    combined_image = torch.cat([x_or, x_ad], dim=-1)
    return wandb.Image(combined_image, caption=caption)


def main(model_path):
    model = MnistModel.load(model_path)
    model.eval()
    test_data = torchvision.datasets.mnist.MNIST(
        root="data/mnist",
        download=True,
        transform=torchvision.transforms.ToTensor(),
        train=False,
    )

    data_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=100,
        shuffle=False,
    )

    captioned_samples = []
    for x_orig, y in data_loader:
        # Model returns log_softmax values
        output_orig = torch.exp(model(x_orig))
        x_ad = generate_adversarial_sample(
            model, torch.nn.functional.nll_loss, x_orig, y, 0.25
        )
        output_ad = torch.exp(model(x_ad))
        captioned_samples += [
            generate_log_image(x_or_s, x_ad_s, pred_or_s, pred_ad_s, y_s)
            for x_or_s, x_ad_s, pred_or_s, pred_ad_s, y_s in zip(
                x_orig, x_ad, output_orig, output_ad, y
            )
        ]
        break

    wandb.log({"adversarial_samples": captioned_samples})


if __name__ == "__main__":
    wandb.init("Adversarial Evaluation")
    main("data/model.pth")
