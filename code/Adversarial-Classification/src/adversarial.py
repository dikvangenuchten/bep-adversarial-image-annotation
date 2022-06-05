from typing import Callable
import torch
import torchvision
import wandb

import numpy as np
from model import MnistModel
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import tqdm


def generate_fast_gradient_adversarial_sample(
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


def generate_gaussian_sample(
    model: torch.nn.Module,
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    target: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Generate an adversarial sample based on the fast gradient method."""
    return x + epsilon * torch.randn_like(x)


def generate_fast_gradient_targeted_adversarial_sample(
    model: torch.nn.Module,
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    target: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Generate an adversarial sample based on the fast gradient method with output target."""
    x.requires_grad = True
    output = model(x)
    loss = loss_func(output, target)
    loss.backward()

    return x - epsilon * torch.sign(x.grad)


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

    data_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

    for epsilon in tqdm.tqdm(np.linspace(0, 1, num=11)):
        captioned_samples = []
        all_confidence_orig = []
        all_confidence_ad = []
        for x_orig, y in data_loader:
            # Model returns log_softmax values
            output_orig = torch.exp(model(x_orig))
            x_ad = generate_gaussian_sample(
                model, torch.nn.functional.nll_loss, x_orig, y, epsilon
            )
            output_ad = torch.exp(model(x_ad))

            all_confidence_orig.append(output_orig)
            all_confidence_ad.append(output_ad)

            if len(captioned_samples) < 100:
                captioned_samples += [
                    generate_log_image(x_or_s, x_ad_s, pred_or_s, pred_ad_s, y_s)
                    for x_or_s, x_ad_s, pred_or_s, pred_ad_s, y_s in zip(
                        x_orig, x_ad, output_orig, output_ad, y
                    )
                ]
            # # TODO calculate accuracy, average confidence
            # if len(all_confidence_ad) > 1:
            #     break

        all_confidence_ad = torch.cat(all_confidence_ad)
        all_confidence_orig = torch.cat(all_confidence_orig)

        adversarial_violin = make_violin_plot(all_confidence_ad.detach())
        original_violin = make_violin_plot(all_confidence_orig.detach())

        # wandb.log(
        #     {
        #         "adversarial_samples": captioned_samples,
        #         "epsilon": epsilon,
        #         "histogram average original score": original_violin,
        #         "histogram average adversar score": adversarial_violin,
        #     }
        # )


def make_violin_plot(confidences):
    fig = go.Figure()
    for label, trace in enumerate(confidences.T):
        fig.add_trace(go.Violin(y=trace, name=f"{label}"))
    return fig


if __name__ == "__main__":
    # wandb.init("Adversarial Evaluation")
    main("data/model.pth")
