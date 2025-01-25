import torch
import functools

from src.misc.utils import tensor_on_gpu


class LayerNorm(torch.nn.LayerNorm):
    """
    Custom LayerNorm module that handles FP16 inputs.
    """

    @functools.wraps(torch.nn.LayerNorm.__init__)
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform layer normalization on the input tensor.

        Args:
            x: The input tensor.

        Returns:
            The layer normalized tensor.
        """

        if x.dtype == torch.float16 and sum(self.normalized_shape) < 512:
            with torch.amp.autocast("cuda" if tensor_on_gpu(x) else "cpu", enabled=False):
                return super().forward(x)
        return super().forward(x)


# From https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_utils.py#L141
# Itself from https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
class LayerNorm2d(torch.nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class QKNormV2(torch.nn.Module):
    """
    Applies QKNorm to Q and K matrices.

    However, this version follows the version of Dehghani et al. (2023).

    This version uses plain LayerNorm instead of the QKScaleUp.
    Instead of having a learnable parameter per head, in this version, we have head_dim learnable parameters in the LayerNorm, shared across all heads.

    @misc{dehghani2023scalingvisiontransformers22,
        title={Scaling Vision Transformers to 22 Billion Parameters},
        author={Mostafa Dehghani and Josip Djolonga and Basil Mustafa and Piotr Padlewski and Jonathan Heek and Justin Gilmer and Andreas Steiner and Mathilde Caron and Robert Geirhos and Ibrahim Alabdulmohsin and Rodolphe Jenatton and Lucas Beyer and Michael Tschannen and Anurag Arnab and Xiao Wang and Carlos Riquelme and Matthias Minderer and Joan Puigcerver and Utku Evci and Manoj Kumar and Sjoerd van Steenkiste and Gamaleldin F. Elsayed and Aravindh Mahendran and Fisher Yu and Avital Oliver and Fantine Huot and Jasmijn Bastings and Mark Patrick Collier and Alexey Gritsenko and Vighnesh Birodkar and Cristina Vasconcelos and Yi Tay and Thomas Mensink and Alexander Kolesnikov and Filip Pavetić and Dustin Tran and Thomas Kipf and Mario Lučić and Xiaohua Zhai and Daniel Keysers and Jeremiah Harmsen and Neil Houlsby},
        year={2023},
        eprint={2302.05442},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2302.05442},
    }
    """

    def __init__(self, head_dim: int, bias: bool):
        super(QKNormV2, self).__init__()

        self.q_layer_norm = LayerNorm(head_dim, bias=bias)
        self.k_layer_norm = LayerNorm(head_dim, bias=bias)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)

        return q, k


class QKNorm(torch.nn.Module):
    """
    Applies QKNorm to Q and K matrices.

    @inproceedings{henry-etal-2020-query,
        title = "Query-Key Normalization for Transformers",
        author = "Henry, Alex and Dachapally, Prudhvi Raj  and Pawar, Shubham Shantaram  and Chen, Yuxuan",
        booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
        year = "2020",
    }
    """

    def __init__(self, num_heads, scale):
        super(QKNorm, self).__init__()

        self.qk_scaling = QKScaleUp(num_heads, scale)

    def forward(self, q, k):
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        k = torch.nn.functional.normalize(k, p=2, dim=-1)

        # In the paper we do: (g * (QK^T)) * V
        # We do ((gQ)*K^T)*V due to fused attention kernel that don't allow the first variant.
        q = self.qk_scaling(q)

        return q, k


class QKScaleUp(torch.nn.Module):
    """
    Learned pararmeter used to scale up QK^T before taking the softmax.

    See: https://github.com/CyndxAI/QKNorm/blob/c628cb5d21f1475ba95db779a175748ff9efe940/QKNorm/layers.py#L8C1-L16C30
    """

    def __init__(self, num_heads, scale):
        super(QKScaleUp, self).__init__()

        self.num_heads = num_heads
        self.weight = torch.nn.Parameter(torch.tensor([float(scale)] * num_heads))

    def forward(self, x):
        assert x.shape[2] == self.num_heads, "Shape mismatch in head dimension."

        return x * self.weight[None, None, :, None]
