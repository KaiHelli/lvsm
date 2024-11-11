from math import prod

import torch
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Bool, Float, Int64
from torch import Tensor


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(
    vectors: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched vectors (xyz) to (xyz0)."""
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_rigid(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    transformation: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Apply a rigid-body transformation to points or vectors."""
    return einsum(transformation, homogeneous_coordinates, "... i j, ... j -> ... i")


def transform_cam2world(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Transform points from 3D camera coordinates to 3D world coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics)


def transform_world2cam(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Transform points from 3D world coordinates to 3D camera coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics.inverse())


def project_camera_space(
    points: Float[Tensor, "*#batch dim"],
    intrinsics: Float[Tensor, "*#batch dim dim"],
    epsilon: float = torch.finfo(torch.float32).eps,
    infinity: float = 1e8,
) -> Float[Tensor, "*batch dim-1"]:
    points = points / (points[..., -1:] + epsilon)
    points = points.nan_to_num(posinf=infinity, neginf=-infinity)
    points = einsum(intrinsics, points, "... i j, ... j -> ... i")
    return points[..., :-1]


def project(
    points: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
    intrinsics: Float[Tensor, "*#batch dim dim"],
    epsilon: float = torch.finfo(torch.float32).eps,
) -> tuple[
    Float[Tensor, "*batch dim-1"],  # xy coordinates
    Bool[Tensor, " *batch"],  # whether points are in front of the camera
]:
    points = homogenize_points(points)
    points = transform_world2cam(points, extrinsics)[..., :-1]
    in_front_of_camera = points[..., -1] >= 0
    return project_camera_space(points, intrinsics, epsilon=epsilon), in_front_of_camera


def unproject(
    coordinates: Float[Tensor, "*#batch dim"],
    z: Float[Tensor, "*#batch"],
    intrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
) -> Float[Tensor, "*batch dim+1"]:
    """Unproject 2D camera coordinates with the given Z values."""

    # Apply the inverse intrinsics to the coordinates.
    coordinates = homogenize_points(coordinates)
    ray_directions = einsum(intrinsics.inverse(), coordinates, "... i j, ... j -> ... i")

    # Apply the supplied depth values.
    return ray_directions * z[..., None]


def get_world_rays(
    coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim+2 dim+2"],
    intrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
) -> tuple[
    Float[Tensor, "*batch dim+1"],  # origins
    Float[Tensor, "*batch dim+1"],  # directions
]:
    # Get camera-space ray directions.
    directions = unproject(
        coordinates,
        torch.ones_like(coordinates[..., 0]),
        intrinsics,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Transform ray directions to world coordinates.
    directions = homogenize_vectors(directions)
    directions = transform_cam2world(directions, extrinsics)[..., :-1]

    # Tile the ray origins to have the same shape as the ray directions.
    origins = extrinsics[..., :-1, -1].broadcast_to(directions.shape)

    return origins, directions


def calculate_plucker_rays(
    image: Float[Tensor, "*#batch c h w"], extrinsics: Float[Tensor, "*#batch 4 4"], intrinsics: Float[Tensor, "*#batch 3 3"]
) -> Float[Tensor, "*batch 6 h w"]:
    b, v, _, *grid_shape = image.shape

    # Generate image grid coordinates
    coordinates, _ = sample_image_grid(tuple(grid_shape), device=image.device)

    # Get world rays using the get_world_rays function
    origins, directions = get_world_rays(rearrange(coordinates, "... d -> ... () () d"), extrinsics, intrinsics)

    # Reshape origins and directions back to the original batch and view dimensions
    origins = rearrange(origins, "h w ... c -> ... h w c")
    directions = rearrange(directions, "h w ... c -> ... h w c")

    # Calculate Plücker coordinates
    L = directions
    M = torch.cross(origins, directions, dim=-1)

    # Concatenate L and M along the last dimension and permute to match the desired output shape
    plucker_rays = torch.cat((L, M), dim=-1)  # Shape: (b, n, h, w, 6)
    plucker_rays = rearrange(plucker_rays, "... h w c -> ... c h w")  # Shape: (b, n, 6, h, w)

    return plucker_rays


def plucker_to_point_direction(plucker_rays: Float[Tensor, "*batch 6 h w"], normalize_moment=True) -> Float[Tensor, "*batch 6 h w"]:
    """
    Convert Plücker rays <D, OxD> to point-direction representation <O, D>. 

    Args:
        plucker_rays: A tensor representing Plücker rays, shape: (b, n, 6, h, w)

    Returns:
        A tensor representing point-direction rays, shape: (b, n, 6, h, w)

    Source:
        @InProceedings{zhang2024raydiffusion,
            title={Cameras as Rays: Pose Estimation via Ray Diffusion},
            author={Zhang, Jason Y and Lin, Amy and Kumar, Moneish and Yang, Tzu-Hsuan and Ramanan, Deva and Tulsiani, Shubham},
            booktitle={International Conference on Learning Representations (ICLR)},
            year={2024}
        }
        https://github.com/jasonyzhang/RayDiffusion/blob/main/ray_diffusion/utils/rays.py#L129
    """
    # Extract direction (L) and origin (M) from Plücker rays
    L = plucker_rays[..., :3]
    M = plucker_rays[..., 3:]

    direction = torch.nn.functional.normalize(L, dim=-1)

    if normalize_moment:
        c = torch.norm(L, dim=-1, keepdim=True)
        M = M / c
    
    points = torch.cross(L, M, dim=-1)

    return torch.cat((points, direction), dim=-1)


# TODO: Currently implemented incorrectly
def plot_plucker_rays(image, plucker_rays, step=16):
    """
    Plots Plücker rays on top of the given image.

    Args:
        image: A tensor representing the image, shape: (b, n, c, h, w)
        plucker_rays: A tensor representing Plücker rays, shape: (b, n, 6, h, w)
        step: Step size to reduce the density of rays for better visualization.
    """
    pass
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from einops import rearrange
    import uuid

    # Convert torch tensor image to numpy for visualization
    image_np = image.cpu().numpy()[0, 0]  # Assuming batch size is 1 for visualization
    image_np = np.transpose(image_np, (1, 2, 0))  # (h, w, c)

    # Create a figure and axis object
    fig, ax = plt.subplots()
    ax.imshow(image_np)
    ax.axis('off')

    # Extract origin (M) and direction (L) from Plücker rays
    L = plucker_rays[0, 0, :3]  # (3, h, w)
    M = plucker_rays[0, 0, 3:]  # (3, h, w)

    # Get image dimensions
    img_height, img_width, _ = image_np.shape

    # Use step to reduce the number of rays for better visualization
    h, w = L.shape[1], L.shape[2]
    for i in range(0, h, step):
        for j in range(0, w, step):
            # The origin is the pixel position (i + 0.5, j + 0.5) scaled to the image dimensions to get the center of the pixel
            origin_x = (j + 0.5) / w * img_width
            origin_y = (i + 0.5) / h * img_height

            # Extract direction at the given (i, j) position
            direction = L[:, i, j]

            # Convert direction to pixel scale for visualization
            direction_x = direction[0].cpu().numpy() * img_width
            direction_y = direction[1].cpu().numpy() * img_height

            # Define start and end points for the ray (scale down direction for better visualization)
            start = (origin_x, origin_y)
            end = (origin_x + direction_x * 0.05, origin_y + direction_y * 0.05)  # Scale down the direction

            # Plotting a line for each ray
            ax.plot([start[0], end[0]], [start[1], end[1]], color='r', alpha=0.6)

    # Generate a random ID for the filename
    random_id = str(uuid.uuid4())
    filename = f"plucker_rays_{random_id}.png"
    
    # Save the plot with the generated filename
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    print(f"Plot saved as {filename}")
    """

def sample_image_grid(
    shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
) -> tuple[
    Float[Tensor, "*shape dim"],  # float coordinates (xy indexing)
    Int64[Tensor, "*shape dim"],  # integer indices (ij indexing)
]:
    """Get normalized (range 0 to 1) coordinates and integer indices for an image."""

    # Each entry is a pixel-wise integer coordinate. In the 2D case, each entry is a
    # (row, col) coordinate.
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)

    # Each entry is a floating-point coordinate in the range (0, 1). In the 2D case,
    # each entry is an (x, y) coordinate.
    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)

    return coordinates, stacked_indices


def sample_training_rays(
    image: Float[Tensor, "batch view channel ..."],
    intrinsics: Float[Tensor, "batch view dim dim"],
    extrinsics: Float[Tensor, "batch view dim+1 dim+1"],
    num_rays: int,
) -> tuple[
    Float[Tensor, "batch ray dim"],  # origins
    Float[Tensor, "batch ray dim"],  # directions
    Float[Tensor, "batch ray 3"],  # sampled color
]:
    device = extrinsics.device
    b, v, _, *grid_shape = image.shape

    # Generate all possible target rays.
    xy, _ = sample_image_grid(tuple(grid_shape), device)
    origins, directions = get_world_rays(
        rearrange(xy, "... d -> ... () () d"),
        extrinsics,
        intrinsics,
    )
    origins = rearrange(origins, "... b v xy -> b (v ...) xy", b=b, v=v)
    directions = rearrange(directions, "... b v xy -> b (v ...) xy", b=b, v=v)
    pixels = rearrange(image, "b v c ... -> b (v ...) c")

    # Sample random rays.
    num_possible_rays = v * prod(grid_shape)
    ray_indices = torch.randint(num_possible_rays, (b, num_rays), device=device)
    batch_indices = repeat(torch.arange(b, device=device), "b -> b n", n=num_rays)

    return (
        origins[batch_indices, ray_indices],
        directions[batch_indices, ray_indices],
        pixels[batch_indices, ray_indices],
    )


def intersect_rays(
    origins_x: Float[Tensor, "*#batch 3"],
    directions_x: Float[Tensor, "*#batch 3"],
    origins_y: Float[Tensor, "*#batch 3"],
    directions_y: Float[Tensor, "*#batch 3"],
    eps: float = 1e-5,
    inf: float = 1e10,
) -> Float[Tensor, "*batch 3"]:
    """Compute the least-squares intersection of rays. Uses the math from here:
    https://math.stackexchange.com/a/1762491/286022
    """

    # Broadcast the rays so their shapes match.
    shape = torch.broadcast_shapes(
        origins_x.shape,
        directions_x.shape,
        origins_y.shape,
        directions_y.shape,
    )
    origins_x = origins_x.broadcast_to(shape)
    directions_x = directions_x.broadcast_to(shape)
    origins_y = origins_y.broadcast_to(shape)
    directions_y = directions_y.broadcast_to(shape)

    # Detect and remove batch elements where the directions are parallel.
    parallel = einsum(directions_x, directions_y, "... xyz, ... xyz -> ...") > 1 - eps
    origins_x = origins_x[~parallel]
    directions_x = directions_x[~parallel]
    origins_y = origins_y[~parallel]
    directions_y = directions_y[~parallel]

    # Stack the rays into (2, *shape).
    origins = torch.stack([origins_x, origins_y], dim=0)
    directions = torch.stack([directions_x, directions_y], dim=0)
    dtype = origins.dtype
    device = origins.device

    # Compute n_i * n_i^T - eye(3) from the equation.
    n = einsum(directions, directions, "r b i, r b j -> r b i j")
    n = n - torch.eye(3, dtype=dtype, device=device).broadcast_to((2, 1, 3, 3))

    # Compute the left-hand side of the equation.
    lhs = reduce(n, "r b i j -> b i j", "sum")

    # Compute the right-hand side of the equation.
    rhs = einsum(n, origins, "r b i j, r b j -> r b i")
    rhs = reduce(rhs, "r b i -> b i", "sum")

    # Left-matrix-multiply both sides by the pseudo-inverse of lhs to find p.
    result = torch.linalg.lstsq(lhs, rhs).solution

    # Handle the case of parallel lines by setting depth to infinity.
    result_all = torch.ones(shape, dtype=dtype, device=device) * inf
    result_all[~parallel] = result
    return result_all


def get_fov(intrinsics: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 2"]:
    intrinsics_inv = intrinsics.inverse()

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)
