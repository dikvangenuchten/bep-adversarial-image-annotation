import data_loader


def test_dataset_creation(device):
    loader = data_loader.get_data_loader(device)

    image, captions = next(iter(loader))

    assert image.shape == (1, 3, 256, 256)
    assert image.device.type == device.type
    assert len(captions) == 5
