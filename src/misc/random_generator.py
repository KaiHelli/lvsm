import torch
from dataclasses import dataclass

@dataclass
class RandomGeneratorCfg:
    min_num_context_views: int
    max_num_context_views: int
    context_views_sampler: str
    percentage_treshold: int

class RandomGenerator:
    @classmethod
    def from_cfg(cls, cfg: RandomGeneratorCfg) -> "RandomGenerator":
        """
        Args:
            cfg (RandomGeneratorCfg): Configuration object containing initialization parameters.

        Returns:
            RandomGenerator: An initialized RandomGenerator instance.
        """
        return cls(
            min_num_context_views=cfg.min_num_context_views,
            max_num_context_views=cfg.max_num_context_views,
            context_views_sampler=cfg.context_views_sampler,
            percentage_treshold=cfg.percentage_treshold,
        )
    
    def __init__(self, min_num_context_views: int, max_num_context_views: int, context_views_sampler: str, percentage_treshold: int):
        """
        Initialize the RandomGenerator class.

        Args:
            min_num_context_views (int): The minimum number of context views.
            max_num_context_views (int): The maximum number of context views.
            context_views_sampler (str): The sampling strategy, e.g., 'uniform'.
        """
        self.min_num_context_views = min_num_context_views
        self.max_num_context_views = max_num_context_views
        self.context_views_sampler = context_views_sampler
        self.percentage_treshold = percentage_treshold

    def generate(self, global_step: int, total_global_steps: int | None = None) -> int:
        """
        Generate a random number of context views based on the sampling strategy.

        Args:
            global_step (int): The current global step.
            total_global_steps (int): The total number of global steps.

        Returns:
            int: A random number of context views.
        """
        if self.context_views_sampler == 'uniform' or total_global_steps is None:
            return torch.randint(
                self.min_num_context_views,
                self.max_num_context_views,
                size=(1,)
            ).item()
        elif self.context_views_sampler == 'schedular':
            # Determine the percentage of global steps to vary the sampling
            threshold_step = int(total_global_steps * self.percentage_threshold)

            if global_step < threshold_step:               
                return self.min_num_context_views
            else:
                # Gradually include more variability from min_num_context_views to max_num_context_views
                range_size = self.max_num_context_views - self.min_num_context_views
                interpolation = (global_step - threshold_step) / (total_global_steps - threshold_step)
                variable_view_count = int(self.min_num_context_views + (range_size * interpolation))
                return torch.randint(
                    self.min_num_context_views,
                    variable_view_count + 1,
                    size=(1,)
                ).item()
        else:
            raise ValueError(f"Unknown context_views_sampler: {self.context_views_sampler}")