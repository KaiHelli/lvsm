from typing import Any
import torch


class Identity(torch.nn.Module):
    """
    An identity layer that returns whatever you put in.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, *args):
        return args
