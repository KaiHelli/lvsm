from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import math
from typing import List
import warnings

class WarmupCosineLR(LRScheduler):
    """
    A custom learning rate scheduler that linearly warms up for a specified number of steps,
    then applies cosine decay for the remainder of the training.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): The number of steps to linearly increase the learning rate.
        max_steps (int): The total number of steps for training. The cosine decay is applied from
                         warmup_steps to max_steps.
        initial_lr (float): The initial learning rate at the end of the warmup phase, which will be
                            the starting value for the cosine decay.
        min_lr (float): The minimum learning rate to decay to.
        last_epoch (int): The index of the last epoch when resuming training. Default: -1, which 
                          starts a new schedule.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        initial_lr: float = 0.0,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ) -> None:
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.initial_lr = initial_lr
        self.min_lr = min_lr

        assert self.warmup_steps < self.max_steps, "Warmup steps must be less than max steps."
        
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute the current learning rate based on the current step (self.last_epoch).
        
        Returns:
            List[float]: A list of learning rates for each parameter group.
        """

        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.", UserWarning)

        ## Linear warmup phase
        if self.last_epoch < self.warmup_steps:
            lrs = []

            pos_in_warmup = self.last_epoch / (self.warmup_steps - 1)

            # Calculate the new learning rate for each parameter group
            for base_lr in self.base_lrs:
                # Calculate delta between initial and base learning rate
                delta_lr = base_lr - self.initial_lr
                # Update the learning rate for this group
                lrs.append(self.initial_lr + delta_lr * pos_in_warmup)

            # Return the new learning rates
            return lrs

        ## Cosine annealing phase
        # Calculate the current position in the cosine cycle
        step_in_cos_cycle = self.last_epoch - self.warmup_steps + 1
        steps_in_cos_cycle = self.max_steps - self.warmup_steps
        
        cur_pos_in_cos_cycle = step_in_cos_cycle / steps_in_cos_cycle

        # Calculate the decay factor for the current and previous positions in the cosine cycle
        cur_decay_factor = 0.5 * (1 + math.cos(math.pi * cur_pos_in_cos_cycle))

        lrs = []
        for base_lr in self.base_lrs:
            # Calculate delta between minimum and base learning rate
            delta_lr = base_lr - self.min_lr

            # Calculate the new learning rate
            lr = self.min_lr + delta_lr * cur_decay_factor

            lrs.append(lr)

        return lrs