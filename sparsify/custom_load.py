from collections import OrderedDict

def map_state_dict_names_mlp(state_dict: OrderedDict) -> OrderedDict:
    """Map the MLP state dict names to the MLPMod state dict names."""
    new_state_dict = OrderedDict()
    for name, param in state_dict.items():
        if name.startswith("layers."):
            new_name = name.replace("layers.", "orig.layers")
            new_state_dict[new_name] = param
        else:
            new_state_dict[name] = param
    return new_state_dict
