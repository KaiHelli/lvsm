import torch
from colorspacious import cspace_convert
from einops import rearrange
from jaxtyping import Float
from matplotlib import cm
from torch import Tensor
from typing import Tuple

def apply_color_map(
    x: Float[Tensor, " *batch"],
    color_map: str = "inferno",
) -> Float[Tensor, "*batch 3"]:
    cmap = cm.get_cmap(color_map)

    # Convert to NumPy so that Matplotlib color maps can be used.
    mapped = cmap(x.detach().clip(min=0, max=1).cpu().numpy())[..., :3]

    # Convert back to the original format.
    return torch.tensor(mapped, device=x.device, dtype=torch.float32)


def apply_color_map_to_image(
    image: Float[Tensor, "*batch height width"],
    color_map: str = "inferno",
) -> Float[Tensor, "*batch 3 height with"]:
    image = apply_color_map(image, color_map)
    return rearrange(image, "... h w c -> ... c h w")


def apply_color_map_2d(
    x: Float[Tensor, "*#batch"],
    y: Float[Tensor, "*#batch"],
) -> Float[Tensor, "*batch 3"]:
    red = cspace_convert((189, 0, 0), "sRGB255", "CIELab")
    blue = cspace_convert((0, 45, 255), "sRGB255", "CIELab")
    white = cspace_convert((255, 255, 255), "sRGB255", "CIELab")
    x_np = x.detach().clip(min=0, max=1).cpu().numpy()[..., None]
    y_np = y.detach().clip(min=0, max=1).cpu().numpy()[..., None]

    # Interpolate between red and blue on the x axis.
    interpolated = x_np * red + (1 - x_np) * blue

    # Interpolate between color and white on the y axis.
    interpolated = y_np * interpolated + (1 - y_np) * white

    # Convert to RGB.
    rgb = cspace_convert(interpolated, "CIELab", "sRGB1")
    return torch.tensor(rgb, device=x.device, dtype=torch.float32).clip(min=0, max=1)


def apply_color_map_3d(vectors: Float[Tensor, "*#batch h w 3"], min_brightness: float = 0.3) -> Float[Tensor, "*#batch h w 3"]:
    """
    Apply a color map to 3D vectors, encoding direction as color (RGB)
    and magnitude as brightness, with a lower boundary for brightness.
    
    Args:
        vectors (torch.Tensor): Tensor of shape [..., 3] where the last dimension
                                represents 3D vectors (x, y, z).
        min_brightness (float): Minimum brightness value to ensure visibility of low-magnitude vectors.
                                Should be in the range [0, 1].
                                
    Returns:
        torch.Tensor: Tensor of shape [..., 3], where each vector is mapped to an RGB color.
    """
    # Compute magnitudes and normalize directions
    magnitudes = torch.norm(vectors, dim=-1, keepdim=True)
    directions_normalized = vectors / (magnitudes + 1e-8)  # Avoid division by zero

    # Map directions to colors (scaled to [0, 1])
    colors = (directions_normalized + 1) / 2  # Scale [-1, 1] to [0, 1]

    # Modulate brightness by magnitude with interpolation
    max_magnitude = magnitudes.max()  # Find global max for normalization
    normalized_magnitude = magnitudes / (max_magnitude + 1e-8)
    brightness = normalized_magnitude * (1 - min_brightness) + min_brightness  # Interpolate between min_brightness and 1

    # Apply brightness to colors
    colors = colors * brightness

    return colors


def plucker_to_colormaps(plucker_tensor: Float[Tensor, "*#batch 6 h w"], min_brightness: float = 0.3) -> Tuple[Float[Tensor, "*#batch h w 3"], Float[Tensor, "*#batch h w 3"]]:
    """
    Convert a Plücker ray tensor into color maps for directions and momentum.
    
    Args:
        plucker_tensor (torch.Tensor): A tensor of shape [b, n, 6, h, w],
                                       where the last dimension splits into:
                                       - [0:3]: normalized directions (x, y, z)
                                       - [3:6]: momentum (x, y, z)
        min_brightness (float): Minimum brightness value to ensure visibility of low-magnitude vectors.
                                Should be in the range [0, 1].
    
    Returns:
        tuple: (direction_colors, momentum_colors)
               - direction_colors: Tensor of shape [b, n, 3, h, w] (RGB for directions).
               - momentum_colors: Tensor of shape [b, n, 3, h, w] (RGB for momentum).
    """
    assert plucker_tensor.shape[-3] == 6, "Expecting plucker coordinate dimension at index -3"

    # Split into directions and momenta
    directions = rearrange(plucker_tensor[..., :3, :, :], "... c h w -> ... h w c")  # Shape [b, n, h, w, 3]
    momentum = rearrange(plucker_tensor[..., 3:, :, :], "... c h w -> ... h w c")  

    # Apply color mapping
    direction_colors = apply_color_map_3d(directions, min_brightness)
    momentum_colors = apply_color_map_3d(momentum, min_brightness)

    direction_colors = rearrange(direction_colors, "... h w c -> ... c h w")  # Shape [b, n, 3, h, w]
    momentum_colors = rearrange(momentum_colors, "... h w c -> ... c h w")

    return direction_colors, momentum_colors