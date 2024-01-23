"""Defines a Hook object and a HookedModel class for adding hooks to PyTorch models.

A Hook object defines a hook function and the hook point to add the hook to. The HookedModel class
is a wrapper around a PyTorch model that allows hooks to be added and removed.
"""

import inspect
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Optional, Union

import torch

from sparsify.utils import get_model_attr


@dataclass
class Hook:
    """Defines a hook object that can be added to a model.

    After initialization, the hook_type is created and stored as an attribute based on whether
    fn contains an output argument.


    Attributes:
        name: Name of the hook. This is useful for identifying hooks when two hooks have the
            same module_name (e.g. a forward and pre_forward hook).
        data_key: The key or keys to index the data in HookedModel.hooked_data.
        fn: The hook function to run at the hook point.
        module_name: String representing the attribute of the model to add the hook to.
            Nested attributes are split by periods (e.g. "layers.linear_0").
        fn_kwargs: Additional keyword arguments to pass to the hook function.
    """

    name: str
    data_key: Union[str, list[str]]
    fn: Callable
    module_name: str
    fn_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set the hook_type attribute based on whether fn contains an output argument.

        Also verify that the name of the function contains 'forward' or 'pre_forward', depending
        on which type is inferred.
        """
        fn_args = list(inspect.signature(self.fn).parameters.keys())
        assert fn_args[:2] == [
            "module",
            "inputs",
        ], f"Hook function must have signature (module, inputs, ...), got {fn_args}"
        if len(fn_args) > 2 and fn_args[2] == "output":
            self.hook_type = "forward"
            assert (
                "forward" in self.fn.__name__ and "pre_forward" not in self.fn.__name__
            ), f"Hook name must contain 'forward' for forward hooks, got {self.fn.__name__}"
        else:
            self.hook_type = "pre_forward"
            assert "pre_forward" in self.fn.__name__, (
                f"Hook name must contain 'pre_forward' for pre_forward hooks, got "
                f"{self.fn.__name__}"
            )


class HookedModel(torch.nn.Module):
    """A wrapper around a PyTorch model that allows hooks to be added and removed.

    Example:
        >>> model = torch.nn.Sequential()
        >>> model.add_module("linear_0", torch.nn.Linear(3, 2))
        >>> hooked_model = HookedModel(model)
        >>> hook = Hook(name="forward_linear_0", data_key="gram", fn=gram_forward_hook_fn,
            module_name="linear_0")
        >>> hooked_model(torch.randn(6, 3), hooks=[hook])
        >>> hooked_model.hooked_data["linear_0"]["gram"]
        tensor([[ 1.2023, -0.0311],
                [-0.0311,  0.9988]])
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self.hooked_data: dict[str, Any] = {}

    def __call__(self, *args, hooks: Optional[list[Hook]] = None, **kwargs) -> Any:
        return self.forward(*args, hooks=hooks, **kwargs)

    def forward(self, *args, hooks: Optional[list[Hook]] = None, **kwargs) -> Any:
        """Run the forward pass of the model and remove all hooks."""
        if hooks is not None:
            self.add_hooks(hooks)
        try:
            output = self.model(*args, **kwargs)
        finally:
            self.remove_hooks()
        return output

    def add_hooks(self, hooks: list[Hook]) -> None:
        """Add a hook to the model at each of the specified hook points."""
        for hook in hooks:
            hook_module = get_model_attr(self.model, hook.module_name)
            hook_fn_partial: partial = partial(
                hook.fn,
                hooked_data=self.hooked_data,
                hook_name=hook.name,
                data_key=hook.data_key,
                **hook.fn_kwargs,
            )
            if hook.hook_type == "forward":
                handle = hook_module.register_forward_hook(hook_fn_partial)
            elif hook.hook_type == "pre_forward":
                handle = hook_module.register_forward_pre_hook(hook_fn_partial)
            self.hook_handles.append(handle)

    def remove_hooks(self) -> None:
        """Remove all hooks from the model."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def clear_hooked_data(self) -> None:
        """Clear all data stored in the hooked_data attribute."""
        self.hooked_data = {}
