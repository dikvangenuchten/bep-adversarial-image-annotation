"""Load datasets for pytorch
"""
import torch
import torchvision


def transform_to_device(device):
    """Move tensor to device"""

    def to_device(tensor):
        return tensor.to(device)

    return to_device


def limit_to_n_labels(n):
    def _limit_to_n(labels):
        return labels[:n]

    return _limit_to_n


def get_data_loader(device, batch_size, size=None):
    """Loads the coco dataset"""
    dataset = torchvision.datasets.CocoCaptions(
        root="/data/val2017/",
        annFile="/data/captions_val2017.json",
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                transform_to_device(device),
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        ),
        target_transform=torchvision.transforms.Compose([limit_to_n_labels(5)]),
    )

    if size is not None:
        dataset = torch.utils.data.Subset(dataset, torch.arange(size))

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)
    return loader
