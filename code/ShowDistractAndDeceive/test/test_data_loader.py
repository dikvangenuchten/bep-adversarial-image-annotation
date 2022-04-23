import data_loader


def test_dataset_creation(device, batch_size):
    loader = data_loader.get_data_loader(device, batch_size)

    image, captions = next(iter(loader))

    assert image.shape == (batch_size, 3, 256, 256)
    assert image.device.type == device.type
    assert len(captions) == 5
