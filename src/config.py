from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from .dataset.data_module import DataLoaderCfg, DatasetCfg
from .loss import LossCfgWrapper
from .model.model_wrapper import OptimizerCfg, TestCfg, TrainCfg, LossCfg
from .model.transformer import TransformerCfg
from .model.lvsm import LVSMCfg


@dataclass
class CheckpointingCfg:
    load: Optional[str]  # Not a path, since it could be something like wandb://...
    every_n_train_steps: int
    save_top_k: int
    pretrained_model: Optional[str]
    resume: Optional[bool] = True


@dataclass
class TrainerCfg:
    max_steps: int
    check_val_every_n_epoch: int | None
    val_check_interval: int | float | None
    gradient_clip_val: int | float | None
    accumulate_grad_batches: int
    num_sanity_val_steps: int
    precision: str
    log_every_n_steps: int
    num_nodes: Optional[int] = 1


@dataclass
class RootCfg:
    wandb: dict
    mode: Literal["train", "test"]
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    lvsm_cfg: LVSMCfg
    optimizer: OptimizerCfg
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    loss: LossCfg
    test: TestCfg
    train: TrainCfg
    seed: int


TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    # Resolve all interpolations before typing.
    OmegaConf.resolve(cfg)

    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )


def separate_loss_cfg_wrappers(joined: dict) -> list[LossCfgWrapper]:
    # The dummy allows the union to be converted.
    @dataclass
    class Dummy:
        dummy: LossCfgWrapper

    return [load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy for k, v in joined.items()]


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg,
        RootCfg,
        {list[LossCfgWrapper]: separate_loss_cfg_wrappers},
    )
