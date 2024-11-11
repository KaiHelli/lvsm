import torch
from typing import Callable
from functools import partial


def get_activation_fn(activation: str) -> Callable[..., torch.nn.Module]:
    """Return an activation function given its name."""
    activations = {str(a).lower(): a for a in torch.nn.modules.activation.__all__}

    if activation in activations:
        activation_fn = getattr(torch.nn, activations[activation])
        return activation_fn
    else:
        raise ValueError(f"Activation function '{activation}' not found.")


def get_weight_init_fn(activation: str) -> Callable[[torch.Tensor], None]:
    """Return a weight initialization function given the activation function name."""

    weight_inits = {
        "tanh": partial(torch.nn.init.xavier_normal_, gain=5 / 3),
        "sigmoid": partial(torch.nn.init.xavier_normal_, gain=1.0),
        "relu": partial(torch.nn.init.kaiming_normal_, nonlinearity="relu"),
        "leakyrelu": partial(torch.nn.init.kaiming_normal_, nonlinearity="leaky_relu"),
        "elu": partial(torch.nn.init.kaiming_normal_, nonlinearity="relu"),
        "gelu": partial(torch.nn.init.kaiming_normal_, nonlinearity="relu"),
        "selu": partial(torch.nn.init.kaiming_normal_, nonlinearity="selu"),
    }

    if activation in weight_inits:
        return weight_inits[activation]
    else:
        raise ValueError(f"Weight initialization for activation '{activation}' not found.")
