import torch



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