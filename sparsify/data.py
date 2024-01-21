import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sparsify.configs import Config
from pathlib import Path
from datasets.load import load_dataset

def load_data_to_device(data, device):
    """Recursively load a batch of data samples to the given device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [load_data_to_device(d, device) for d in data]
    elif isinstance(data, dict):
        return {k: load_data_to_device(v, device) for k, v in data.items()}
    else:
        raise TypeError(f"Unsupported type: {type(data)}")


def data_preprocessing(data, config):
    if config.data.dataset == "mnist":
        data = data.view(data.shape[0], -1)
    return data

def get_num_minibatch_elements(data):
    if isinstance(data, torch.Tensor):
        return data.shape[0]
    elif isinstance(data, (list, tuple)):
        return get_num_minibatch_elements(data[0])
    elif isinstance(data, dict):
        return get_num_minibatch_elements(data[list(data.keys())[0]])
    else:
        raise TypeError(f"Unsupported type: {type(data)}")


def get_data(config: Config):
    if config.data.dataset == "mnist":
        transform = transforms.ToTensor()
        train_data = datasets.MNIST(
            root=Path(__file__).parent.parent / ".data", train=True, download=True, transform=transform
        )
        train_loader = DataLoader(train_data, batch_size=config.train.batch_size, shuffle=True)
        test_data = datasets.MNIST(
            root=Path(__file__).parent.parent / ".data", train=False, download=True, transform=transform
        )
        test_loader = DataLoader(test_data, batch_size=config.train.batch_size, shuffle=False)
    elif config.data.dataset == "pile-10k":
        train_data = load_dataset("NeelNanda/pile-10k", split="train") # No test exists
        train_loader = DataLoader(train_data, batch_size=config.train.batch_size, shuffle=True)
        test_loader = None
    return train_loader, test_loader
    