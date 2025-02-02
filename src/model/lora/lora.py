"""
Implementation adapted from minLoRA:
https://github.com/cccntu/minLoRA

Other References:
1) the official LoRA implementation released by Microsoft:
https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
"""
import math
from functools import partial
import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn
from dataclasses import dataclass, asdict
from typing import Literal


class LoRAParametrization(nn.Module):
    """
    This class replaces a given layer's weight W with (W + BA), where
        - B and A are low-rank (r) matrices learned during training.
        - The combination (W + BA) is then used wherever `layer.weight` is accessed.

    Args:
        fan_in (int):  The input dimension of the weight (in_features).
        fan_out (int): The output dimension of the weight (out_features).
        fan_in_fan_out (bool): If True, treat `W` as (in_features, out_features),
            e.g. for embeddings. Otherwise, treat `W` as (out_features, in_features).
        lora_rank (int): The rank r of LoRA.
        lora_dropout_p (float): Dropout probability for the LoRA update branch.
        lora_alpha (float): The LoRA “alpha” scaling factor.
    """

    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        fan_in_fan_out: bool,
        lora_rank: int,
        lora_dropout_p: float = 0.0,
        lora_alpha: float = 1.0,
    ):
        super().__init__()

        # Decide whether to swap dimensions (needed for embeddings which store weights as (vocab_size, hidden_dim))
        # 'swap' simply flips (fan_out, fan_in) -> (fan_in, fan_out) when fan_in_fan_out=True.
        self.swap = (lambda x: (x[1], x[0])) if fan_in_fan_out else (lambda x: x)

        # Create LoRA parameters A and B:
        #   - A has shape (r, fan_in) or (fan_in, r) after applying `swap`.
        #   - B has shape (fan_out, r) or (r, fan_out) after applying `swap`.
        self.lora_A = nn.Parameter(torch.zeros(self.swap((lora_rank, fan_in))))
        self.lora_B = nn.Parameter(torch.zeros(self.swap((fan_out, lora_rank))))

        # Initialize A with a Kaiming uniform scheme; B stays zero by default
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # Rank-stabilized scaling: alpha / sqrt(r) instead of alpha / r
        self.scaling = lora_alpha / math.sqrt(lora_rank)

        # Setup dropout for LoRA branch
        # Official LoRA typically does dropout on the *input activations*.
        # Here, we do "parameter dropout" on A by multiplying with a mask.
        self.lora_dropout = nn.Dropout(p=lora_dropout_p) if lora_dropout_p > 0 else nn.Identity()

        # We'll store a dropout mask of shape (1, fan_in) or (fan_in, 1), then broadcast
        # across the entire LoRA A matrix.
        self.register_buffer(
            "lora_dropout_mask", torch.ones(self.swap((1, fan_in)), dtype=self.lora_A.dtype), persistent=False
        )

        # We define how to forward: can be disabled or enabled at runtime.
        self.forward_fn = self.lora_forward

    def _dropout(self, A: torch.Tensor) -> torch.Tensor:
        """
        Apply the dropout mask to A, effectively zeroing out entire columns of A
        with probability p. This attempts to mimic "A @ dropout(x)" with a simpler
        parameter-level dropout mask.
        """
        # 'dropout(ones)' creates the random 0/1 mask. Broadcasting it over A means
        # an entire input dimension is dropped or kept, akin to column dropout.
        return A * self.lora_dropout(self.lora_dropout_mask)

    def lora_forward(self, original_weight: torch.Tensor) -> torch.Tensor:
        """
        Return W + (B @ A). The 'original_weight' argument is PyTorch's
        way of passing in the unparametrized weight (W).

        After this, the layer's forward pass will do: (W + B A) x
        where x are the *actual* input activations.
        """
        # dropout_fn(A) => apply dropout to A if needed
        # swap((B, dropout_fn(A))) => ensures correct multiplication order
        lora_update = torch.matmul(*self.swap((self.lora_B, self._dropout(self.lora_A))))

        # Reshape the LoRA update to the same shape as W, then scale by alpha/sqrt(r)
        lora_update = lora_update.view(original_weight.shape) * self.scaling

        return original_weight + lora_update

    def forward(self, original_weight: torch.Tensor) -> torch.Tensor:
        """
        The parametrize API calls this during forward passes. Typically:
            param_weight = self.forward_fn(original_weight)
        """
        return self.forward_fn(original_weight)

    def disable_lora(self):
        """Make the parametrization a no‐op (returns original weight only)."""
        self.forward_fn = lambda weight: weight

    def enable_lora(self):
        """Enable the LoRA update again."""
        self.forward_fn = self.lora_forward

    @classmethod
    def from_linear(cls, layer: nn.Linear, lora_rank: int, lora_dropout_p: int = 0.0, lora_alpha: int = 1.0):
        """
        Convenience to build a LoRAParametrization for a linear layer (W is (fan_out, fan_in)).
        """
        fan_out, fan_in = layer.weight.shape
        lora = cls(
            fan_in,
            fan_out,
            fan_in_fan_out=False,
            lora_rank=lora_rank,
            lora_dropout_p=lora_dropout_p,
            lora_alpha=lora_alpha,
        )
        return lora.to(layer.weight.device, layer.weight.dtype)

    @classmethod
    def from_conv2d(cls, layer: nn.Conv2d, lora_rank: int, lora_dropout_p: int = 0.0, lora_alpha: int = 1.0):
        """
        Convenience to build a LoRAParametrization for a Conv2D layer.
        We flatten the kernel dimensions so it's effectively (fan_out, fan_in).
        """
        fan_out, fan_in = layer.weight.view(layer.weight.shape[0], -1).shape
        lora = cls(
            fan_in,
            fan_out,
            fan_in_fan_out=False,
            lora_rank=lora_rank,
            lora_dropout_p=lora_dropout_p,
            lora_alpha=lora_alpha,
        )
        return lora.to(layer.weight.device, layer.weight.dtype)

    @classmethod
    def from_embedding(cls, layer: nn.Embedding, lora_rank: int, lora_dropout_p: int = 0.0, lora_alpha: int = 1.0):
        """
        Convenience to build a LoRAParametrization for an embedding layer, which
        internally has shape (num_embeddings, embedding_dim) = (fan_in, fan_out).
        So we set fan_in_fan_out=True to interpret that as (fan_in, fan_out).
        """
        fan_in, fan_out = layer.weight.shape
        lora = cls(
            fan_in,
            fan_out,
            fan_in_fan_out=True,
            lora_rank=lora_rank,
            lora_dropout_p=lora_dropout_p,
            lora_alpha=lora_alpha,
        )
        return lora.to(layer.weight.device, layer.weight.dtype)


@dataclass
class LoRALayerConfig:
    lora_rank: int = 4
    lora_dropout_p: float = 0.0
    lora_alpha: float = 1.0


@dataclass
class LoRAConfig:
    layer_config: dict[str, dict[str, LoRALayerConfig]]

    def __post_init__(self):
        supported_modules = {"Linear", "Conv2d", "Embedding"}
        for module in self.layer_config:
            assert module in supported_modules, f"Unsupported module: {module} - Supported: {supported_modules}"

        # Resolve the module str to the actual module class
        self.layer_config = {getattr(nn, module): config for module, config in self.layer_config.items()}

        base_fn = {
            nn.Linear: LoRAParametrization.from_linear,
            nn.Conv2d: LoRAParametrization.from_conv2d,
            nn.Embedding: LoRAParametrization.from_embedding,
        }

        # Replace the LoRALayerConfig dataclass with the parametrization function
        for module, attr_dict in self.layer_config.items():
            for attr_name, lora_config in attr_dict.items():
                attr_dict[attr_name] = partial(base_fn[module], **asdict(lora_config))


def apply_lora(layer, lora_config: LoRAConfig, register=True, merge=False):
    """
    Attach or remove LoRA parametrization from a given `layer`.
    This is designed to be used with `model.apply(...)`.
      - If `register=True`, we add LoRA. If `register=False`, we remove it.
      - `merge=True` means remove the parametrization but *bake* its effect
        permanently into the underlying weight.
    """
    if register:
        if type(layer) in lora_config.layer_config:
            for attr_name, parametrization_fn in lora_config.layer_config[type(layer)].items():
                parametrize.register_parametrization(layer, attr_name, parametrization_fn(layer))
    else:
        # remove any existing parametrizations
        if hasattr(layer, "parametrizations"):
            for attr_name in list(layer.parametrizations.keys()):
                parametrize.remove_parametrizations(layer, attr_name, leave_parametrized=merge)


def add_lora(model: nn.Module, lora_config: LoRAConfig):
    """
    Add LoRA parametrizations to *all* layers in `model` that match the config.
    Calling this twice will add two sets of LoRA parameters.
    """
    model.apply(partial(apply_lora, lora_config=lora_config))


def add_lora_by_name(model: nn.Module, target_module_names, lora_config: LoRAConfig):
    """
    Add LoRA parametrizations only to layers whose name contains any string in `target_module_names`.
    """
    for name, layer in model.named_modules():
        if any(t in name for t in target_module_names):
            add_lora(layer, lora_config=lora_config)


def merge_lora(model: nn.Module):
    """
    Merge (W + BA) into the underlying weight W in all layers, then remove the parametrizations.
    This makes the LoRA update permanent.
    """
    model.apply(partial(apply_lora, register=False, merge=True))


def remove_lora(model: nn.Module):
    """
    Remove the LoRA parametrizations but *do not* merge them. The original weights are kept,
    discarding the LoRA updates entirely.
    """
    model.apply(partial(apply_lora, register=False, merge=False))
