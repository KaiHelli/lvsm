from typing import Optional
import torch


def model_on_gpu(module: torch.nn.Module) -> bool:
    """
    Check if a model (module) is on the GPU.

    Args:
        module (nn.Module): A PyTorch module.

    Returns:
        bool: True if the model's parameters are on the GPU, False otherwise.
    """
    # Handle the case where the module might have no parameters
    return next(module.parameters()).is_cuda


def tensor_on_gpu(tensor: torch.Tensor) -> bool:
    """
    Check if a tensor is on the GPU.

    Args:
        tensor (Tensor): A PyTorch tensor.

    Returns:
        bool: True if the tensor is on the GPU, False otherwise.
    """
    return tensor.is_cuda
