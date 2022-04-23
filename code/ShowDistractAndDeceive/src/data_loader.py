"""Load datasets for pytorch
"""
import torch
import torchvision


def transform_to_device(device):
    """Move tensor to device"""

    def to_device(tensor):
        return tensor.to(device)

    return to_device


def get_data_loader(device, batch_size):
    """Loads the coco dataset"""
    dataset = torchvision.datasets.CocoCaptions(
        root="../../coco_dataset/val2017/",
        annFile="../../coco_dataset/captions_val2017.json",
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
        target_transform=torchvision.transforms.Compose([]),
    )

    # Remove all captions that do not have exactly 5
    assert False, "TODO"

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)
    return loader
