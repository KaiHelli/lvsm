from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample, BatchedViewsRGBD
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: BatchedViewsRGBD,
        ground_truth: BatchedViewsRGBD,
        batch: BatchedExample,
        global_step: int,
    ) -> Float[Tensor, ""]:
        delta = prediction["color"] - ground_truth["color"]
        return self.cfg.weight * (delta**2).mean()
