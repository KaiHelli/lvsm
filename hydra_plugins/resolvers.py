from omegaconf import OmegaConf
from typing import List


# (dataset.image_shape^2 / model.lvsm.patch_size^2) * (dataset.view_sampler.num_context_views + dataset.view_sampler.num_target_views)
def calculate_exp_seq_len(
    image_shape: List[int], patch_size: int, num_context_views: int, num_target_views: int
) -> int:
    assert (
        image_shape[0] % patch_size == 0 and image_shape[1] % patch_size == 0
    ), "Image shape must be divisible by patch size in height and width."

    num_patches = (image_shape[0] * image_shape[1]) // (patch_size**2)
    num_images = num_context_views + num_target_views

    return num_patches * num_images


OmegaConf.register_new_resolver("calc_exp_seq_len", calculate_exp_seq_len)
