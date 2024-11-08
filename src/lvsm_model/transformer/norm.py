import torch
import functools


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
            with torch.amp.autocast("cuda" if x.is_cuda else "cpu", enabled=False):
                return super().forward(x)
        return super().forward(x)
