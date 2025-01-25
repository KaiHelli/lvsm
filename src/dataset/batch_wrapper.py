from torch.utils.data import IterableDataset
from math import ceil
import random

from .dataset_re10k import DatasetRE10k


class BatchWrapper(IterableDataset):
    def __init__(self, dataset: DatasetRE10k, batch_size: int, drop_last: bool = False):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def sample_num_context_views(self) -> int:
        if self.dataset.view_sampler.cfg.match_random_num_context_views_per_batch:
            self.dataset.view_sampler.sample_num_context_views()

    def __iter__(self):
        data_iter = iter(self.dataset)

        while True:
            batch = []
            self.sample_num_context_views()

            # Collect items up to batch_size
            for _ in range(self.batch_size):
                try:
                    item = next(data_iter)
                except StopIteration:
                    # We've exhausted the underlying dataset
                    break
                batch.append(item)

            # If we didn't get anything, there's nothing left to yield
            if len(batch) == 0:
                break

            # If we should drop the last partial batch and we don't have enough, stop
            if self.drop_last and len(batch) < self.batch_size:
                break

            yield batch

    def __len__(self):
        d_len = len(self.dataset)
        return (d_len // self.batch_size) if self.drop_last else ceil(d_len / self.batch_size)
