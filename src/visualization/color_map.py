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


def apply_color_map_3d_cielab(vectors: Float[Tensor, "*#batch h w 3"]) -> Float[Tensor, "*#batch h w 3"]:
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
    # Calculate the maximum norm of all vectors
    norm = vectors.norm(dim=-1).max()

    # Normalize all vector to be within a sphere of [-1, 1]
    unit_vectors = vectors / (norm + 1e-8)

    # Adjust scaling for y to range [0, 1]
    unit_vectors[..., 1] = (unit_vectors[..., 1] + 1) / 2  # Normalize y to range [0, 1]

    # Clamp brightness to be within 20 and 100
    unit_vectors[..., 1] = 0.2 + unit_vectors[..., 1] * (0.8 - 0.2)

    # Define scaling factors for x, y, z dimensions
    scales = torch.tensor([100.0, 100.0, 100.0])  # Scaling factors for x, y, z respectively

    # Rescale each vector component by its respective range
    vectors_scaled = unit_vectors * scales

    # Swap y to be the brightness in CIELab
    vectors_scaled = vectors_scaled[..., [1, 0, 2]]

    # Apply brightness to colors (back to RGB)
    colors_rgb = cspace_convert(vectors_scaled.cpu().numpy(), "CIELab", "sRGB1")
    colors_rgb = torch.tensor(colors_rgb, dtype=torch.float32)

    return colors_rgb


def apply_color_map_3d(
    vectors: Float[Tensor, "*#batch h w 3"], min_brightness: float = 0.1
) -> Float[Tensor, "*#batch h w 3"]:
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
    # Scale the spherical space a bit down
    colors = 0.1 + colors * (0.9 - 0.1)

    # Modulate brightness by magnitude with interpolation
    max_magnitude = magnitudes.max()  # Find global max for normalization
    normalized_magnitude = magnitudes / (max_magnitude + 1e-8)
    brightness = (
        normalized_magnitude * (1 - min_brightness) + min_brightness
    )  # Interpolate between min_brightness and 1

    # Apply brightness to colors
    colors = colors * brightness

    return colors


def plucker_to_colormaps(
    plucker_tensor: Float[Tensor, "*#batch 6 h w"], min_brightness: float = 0.1
) -> Tuple[Float[Tensor, "*#batch 3 h w"], Float[Tensor, "*#batch 3 h w"]]:
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
