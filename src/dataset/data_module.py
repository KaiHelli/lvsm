import random
from dataclasses import dataclass
from typing import Callable, List, Any

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch import Generator, nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, default_collate

from ..misc.step_tracker import StepTracker
from . import DatasetCfg, get_dataset
from .types import DataShim, Stage
from .validation_wrapper import ValidationWrapper
from .shims.plucker_rays import generate_rays_batch


def get_data_shim(model: nn.Module) -> DataShim:
    """Get functions that modify the batch. It's sometimes necessary to modify batches
    outside the data loader because GPU computations are required to modify the batch or
    because the modification depends on something outside the data loader.
    """

    shims: list[DataShim] = []
    if hasattr(model, "get_data_shim"):
        shims.append(model.get_data_shim())

    def combined_shim(batch):
        for shim in shims:
            batch = shim(batch)
        return batch

    return combined_shim


@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool
    seed: int | None


#@dataclass
#class DataLoaderShimCfg:
#    calculate_rays: bool


@dataclass
class DataLoaderCfg:
    train: DataLoaderStageCfg
    test: DataLoaderStageCfg
    val: DataLoaderStageCfg
#    shims: DataLoaderShimCfg


DatasetShim = Callable[[Dataset, Stage], Dataset]


#class RayCollateFn:
#    """Callable class for the ray collate function, to ensure it is pickleable."""

#    def __init__(self) -> None:
#        pass

#    def __call__(self, batch: List[dict[str, Any]]) -> dict[str, Any]:
#        # Stack the batch into tensors
#        collated_batch = default_collate(batch)
#        # Generate rays for the 'context' and 'target' views
#        collated_batch = generate_rays_batch(collated_batch)

#        return collated_batch


def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


class DataModule(LightningDataModule):
    dataset_cfg: DatasetCfg
    data_loader_cfg: DataLoaderCfg
    step_tracker: StepTracker | None
    dataset_shim: DatasetShim
    global_rank: int

    def __init__(
        self,
        dataset_cfg: DatasetCfg,
        data_loader_cfg: DataLoaderCfg,
        step_tracker: StepTracker | None = None,
        dataset_shim: DatasetShim = lambda dataset, _: dataset,
        global_rank: int = 0,
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.data_loader_cfg = data_loader_cfg
        self.step_tracker = step_tracker
        self.dataset_shim = dataset_shim
        self.global_rank = global_rank
        self.collate_fn = None

#        if self.data_loader_cfg.shims.calculate_rays:
#            self.collate_fn = RayCollateFn()

    def get_persistent(self, loader_cfg: DataLoaderStageCfg) -> bool | None:
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers

    def get_generator(self, loader_cfg: DataLoaderStageCfg) -> torch.Generator | None:
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator

    def train_dataloader(self):
        dataset = get_dataset(self.dataset_cfg, "train", self.step_tracker)
        dataset = self.dataset_shim(dataset, "train")
        return DataLoader(
            dataset,
            self.data_loader_cfg.train.batch_size,
            shuffle=not isinstance(dataset, IterableDataset),
            num_workers=self.data_loader_cfg.train.num_workers,
            collate_fn=self.collate_fn,
            generator=self.get_generator(self.data_loader_cfg.train),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.train),
        )

    def val_dataloader(self):
        dataset = get_dataset(self.dataset_cfg, "val", self.step_tracker)
        dataset = self.dataset_shim(dataset, "val")
        return DataLoader(
            ValidationWrapper(dataset, 1),
            self.data_loader_cfg.val.batch_size,
            num_workers=self.data_loader_cfg.val.num_workers,
            collate_fn=self.collate_fn,
            generator=self.get_generator(self.data_loader_cfg.val),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.val),
        )

    def test_dataloader(self, dataset_cfg=None):
        dataset = get_dataset(
            self.dataset_cfg if dataset_cfg is None else dataset_cfg,
            "test",
            self.step_tracker,
        )
        dataset = self.dataset_shim(dataset, "test")
        return DataLoader(
            dataset,
            self.data_loader_cfg.test.batch_size,
            num_workers=self.data_loader_cfg.test.num_workers,
            collate_fn=self.collate_fn,
            generator=self.get_generator(self.data_loader_cfg.test),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.test),
            shuffle=False,
        )
