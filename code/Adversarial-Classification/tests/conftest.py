import wandb


def pytest_configure():
    """Start of pytest run"""
    # Don't log wandb during tests.
    wandb.init(mode="disabled", reinit=False)
