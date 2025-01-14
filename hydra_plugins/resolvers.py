from omegaconf import OmegaConf
from typing import List


def calculate_exp_seq_len(
    image_shape: List[int], patch_size: int, num_context_views: int, num_target_views: int, vae_cfg: dict
) -> int:
    if vae_cfg is not None:
        downsample_factor = vae_cfg.get("downsample_factor", 1)
    
        assert downsample_factor > 0, "Downsample factor must be greater than 0."
        assert all(dim % downsample_factor == 0 for dim in image_shape), "Image shape must be divisible by downsample factor."

        image_shape = [dim // downsample_factor for dim in image_shape]

    assert (
        image_shape[0] % patch_size == 0 and image_shape[1] % patch_size == 0
    ), "Image shape must be divisible by patch size in height and width."


    num_patches = (image_shape[0] * image_shape[1]) // (patch_size**2)
    num_images = num_context_views + num_target_views

    return num_patches * num_images


OmegaConf.register_new_resolver("calc_exp_seq_len", calculate_exp_seq_len)
