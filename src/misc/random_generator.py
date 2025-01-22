import torch
from dataclasses import dataclass

@dataclass
class RandomGeneratorCfg:
    min_num_context_views: int
    max_num_context_views: int
    context_views_sampler: str

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
        )
    
    def __init__(self, min_num_context_views: int, max_num_context_views: int, context_views_sampler: str):
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

    def generate(self) -> int:
        """
        Generate a random number of context views based on the sampling strategy.

        Returns:
            int: A random number of context views.
        """
        if self.context_views_sampler == 'uniform':
            return torch.randint(
                self.min_num_context_views, 
                self.max_num_context_views, 
                size=(1,)
            ).item()
        else:
            raise ValueError(f"Unknown context_views_sampler: {self.context_views_sampler}")