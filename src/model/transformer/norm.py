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

    def __init__(self, scale):
        super(QKNorm, self).__init__()

        self.qk_scaling = QKScaleUp(scale)

    def forward(self, q, k):
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        k = torch.nn.functional.normalize(k, p=2, dim=-1)

        # In the paper we do: (g * (QK^T)) * V
        # We do ((gQ)*K^T)*V due to
        q = self.qk_scaling(q)

        return q, k


class QKScaleUp(torch.nn.Module):
    """
    Learned pararmeter used to scale up QK^T before taking the softmax.

    See: https://github.com/CyndxAI/QKNorm/blob/c628cb5d21f1475ba95db779a175748ff9efe940/QKNorm/layers.py#L8C1-L16C30
    """

    def __init__(self, scale):
        super(QKScaleUp, self).__init__()

        self.weight = torch.nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x):
        return x * self.weight
