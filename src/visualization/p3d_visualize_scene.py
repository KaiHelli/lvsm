from jaxtyping import Float, Int
from typing import List, Optional, Tuple

import warnings
import io
import torch
import numpy as np
from einops import rearrange, repeat
from PIL import Image
from plotly.graph_objects import Figure
from plotly.io import to_image

from pytorch3d.renderer import PerspectiveCameras, RayBundle
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import TexturesAtlas
from pytorch3d.vis.plotly_vis import AxisArgs, plot_scene
from pytorch3d.utils import cameras_from_opencv_projection

from ..geometry.projection import get_world_rays, sample_image_grid, get_world_points


def visualize_cameras(
    extrinsics: Float[torch.Tensor, "n 4 4"],
    intrinsics: Float[torch.Tensor, "n 3 3"],
    image_size: Int[torch.Tensor, "n 2"],
) -> PerspectiveCameras:
    """
    Visualize multiple cameras in a scene using PyTorch3D.

    Args:
        extrinsics (Float[torch.Tensor, "n 4 4"]): A tensor of shape (n, 4, 4) containing the extrinsic matrices of the cameras.
        intrinsics (Float[torch.Tensor, "n 3 3"]): A tensor of shape (n, 3, 3) containing the intrinsic matrices of the cameras.
        image_size (Int[torch.Tensor, "n 2"]): A tensor of shape (n, 2) containing the image sizes for each camera.
        device (torch.device): The device for computations (default: CUDA if available).

    Returns:
        PerspectiveCameras: A batch of PyTorch3D cameras.
    """
    assert intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3), "Intrinsics should be of shape (n, 3, 3)"
    assert extrinsics.ndim == 3 and extrinsics.shape[1:] == (4, 4), "Extrinsics should be of shape (n, 4, 4)"
    assert extrinsics.shape[0] == intrinsics.shape[0], "Number of intrinsics and extrinsics must match"

    extrinsics_inv = extrinsics.inverse()

    # Convert extrinsics to rotation (R) and translation (tvec)
    R = extrinsics_inv[:, :3, :3]  # Shape: (n, 3, 3)
    tvec = extrinsics_inv[:, :3, 3]  # Shape: (n, 3)

    # Convert to PyTorch3D's PerspectiveCameras
    cameras = cameras_from_opencv_projection(R=R, tvec=tvec, camera_matrix=intrinsics, image_size=image_size)

    return cameras


def draw_rays(
    extrinsics: Float[torch.Tensor, "n 4 4"],
    intrinsics: Float[torch.Tensor, "n 3 3"],
    image_size: List[int],
    ray_length: float = 1.0,
) -> List[RayBundle]:
    """
    Draw camera rays in the scene.

    Args:
        extrinsics (Float[torch.Tensor, "n 4 4"]): A tensor of shape (n, 4, 4) containing the extrinsic matrices of the cameras.
        intrinsics (Float[torch.Tensor, "n 3 3"]): A tensor of shape (n, 3, 3) containing the intrinsic matrices of the cameras.
        image_size (List[int]): A list containing the height and width of the images.
        ray_length (float, optional): Length of the rays to be visualized.

    Returns:
        List[RayBundle]: A list of RayBundle objects representing rays for each camera.
    """
    coordinates, _ = sample_image_grid(tuple(image_size), device=extrinsics.device)
    origins, directions = get_world_rays(rearrange(coordinates, "... d -> ... () d"), extrinsics, intrinsics)

    # Subsample the rays for visualization
    keep_every_x, keep_every_y = max(image_size[1] // 16, 1), max(image_size[0] // 16, 1)
    origins = origins[::keep_every_y, ::keep_every_x]
    directions = directions[::keep_every_y, ::keep_every_x]

    # Flatten the origins and directions tensors
    origins, directions = rearrange(origins, "h w n d -> n (h w) d"), rearrange(directions, "h w n d -> n (h w) d")

    lengths = torch.ones_like(origins[..., :2]) * ray_length
    lengths[..., 0] = 0

    num_views = origins.shape[0]

    ray_bundles = []
    for i in range(num_views):
        ray_bundle = RayBundle(origins=origins[i], directions=directions[i], lengths=lengths[i], xys=directions[i])
        ray_bundles.append(ray_bundle)

    return ray_bundles


def draw_images_on_planes(
    images: Float[torch.Tensor, "n c h w"],
    extrinsics: Float[torch.Tensor, "n 4 4"],
    intrinsics: Float[torch.Tensor, "n 3 3"],
) -> Meshes:
    """
    Draw images in the scene associated with each camera.

    Args:
        images (Float[torch.Tensor, "n c h w"]): A tensor of shape (n, c, h, w) representing images associated with each camera.
        extrinsics (Float[torch.Tensor, "n 4 4"]): A tensor of shape (n, 4, 4) containing the extrinsic matrices of the cameras.
        intrinsics (Float[torch.Tensor, "n 3 3"]): A tensor of shape (n, 3, 3) containing the intrinsic matrices of the cameras.

    Returns:
        Meshes: A PyTorch3D Meshes object for rendering image planes.
    """
    assert images.ndim == 4, "Images should be of shape (n, c, h, w)"

    N, _, H, W = images.shape

    # Get a grid containing coordinates of corner pixels, normalized.
    coordinates, _ = sample_image_grid((H, W), device=images.device, sampling_type="corner")

    # Project the grid into world space
    verts = get_world_points(rearrange(coordinates, "... d -> ... () d"), extrinsics, intrinsics)
    verts = rearrange(verts, "h w n p -> n (h w) p")

    faces = torch.full((1, H * W * 2, 3), -1, device=images.device, dtype=torch.long)
    face_idx = 0
    for y in range(H):
        for x in range(W):
            # Index the four corners of the current pixel square
            v0 = y * (W + 1) + x
            v1 = v0 + 1
            v2 = v0 + (W + 1)
            v3 = v2 + 1

            faces[0, face_idx] = torch.tensor([v0, v1, v2], dtype=torch.long)
            faces[0, face_idx + 1] = torch.tensor([v2, v1, v3], dtype=torch.long)
            face_idx += 2

    # The face mapping is the same for all images
    faces = repeat(faces, "1 f i -> n f i", n=N)

    # Flip images to match the face ordering**
    images_flipped = torch.flip(images, dims=[2, 3])  # Flip along the height and width dimension

    # Each pixel color needs to be duplicated as it consists of two triangles
    atlas_colors = repeat(images_flipped, "n c h w -> n (h w 2) 1 1 c")

    textures = TexturesAtlas(atlas_colors)
    mesh = Meshes(verts=verts, faces=faces, textures=textures)

    return mesh


def compute_combined_aabb(
    cameras: PerspectiveCameras, ray_bundles: List[RayBundle], meshes: Meshes
) -> Tuple[Float[torch.Tensor, "3"], Float[torch.Tensor, "3"]]:
    """
    Compute the axis-aligned bounding box that includes camera origins, rays, and mesh vertices.
    Args:
        cameras: A PerspectiveCameras object.
        ray_bundles: A list of RayBundle objects containing rays.
        meshes: A Meshes object containing 3D meshes.
    Returns:
        bbox_min: Tensor of shape (3,) representing the minimum x, y, z coordinates.
        bbox_max: Tensor of shape (3,) representing the maximum x, y, z coordinates.
    """
    # Get camera origins
    camera_origins = cameras.get_camera_center()

    # Get all ray points from all RayBundles (includes ray origins and directions)
    all_ray_points = []
    for ray_bundle in ray_bundles:
        ray_points = (
            ray_bundle.origins[:, None, :]
            + rearrange(ray_bundle.lengths, "n l -> n l 1") * ray_bundle.directions[:, None, :]
        )
        ray_points = rearrange(ray_points, "n l p -> (n l) p")

        all_ray_points.append(ray_points)

    all_ray_points = torch.cat(all_ray_points, dim=0)

    # Get all mesh vertices
    mesh_verts = meshes.verts_packed()

    # Combine all points (camera origins, ray points, mesh vertices)
    all_points = torch.cat([camera_origins, all_ray_points, mesh_verts], dim=0)

    # Compute the combined AABB
    bbox_min = all_points.min(dim=0).values
    bbox_max = all_points.max(dim=0).values

    return bbox_min, bbox_max


def generate_rotation_gif(
    fig: Figure, aabb_min: Float[torch.Tensor, "3"], aabb_max: Float[torch.Tensor, "3"], num_frames: int = 36
) -> io.BytesIO:
    """
    Generates a rotating GIF for a 3D Plotly figure around a given AABB.

    Parameters:
        fig (Figure): A Plotly 3D figure.
        aabb_min (torch.Tensor): The minimum point of the AABB (shape [3]).
        aabb_max (torch.Tensor): The maximum point of the AABB (shape [3]).
        num_frames (int): Number of frames for the rotation.

    Returns:
        Image: A pillow image.
    """
    warnings.warn("Check the rotation implementation before using this function to create gifs from Plotly figures.")

    # Calculate the center and radius of the bounding box
    center = (aabb_min + aabb_max) / 2  # Center of the AABB
    radius = 0.75 * torch.norm(aabb_max - center)  # Radius for the camera

    # Generate rotation angles
    angles = torch.linspace(0, 2 * torch.pi, num_frames + 1)[:-1]

    # Store frames in memory
    frames = []
    for angle in angles:
        # Calculate camera position
        x = center[0] + radius * torch.cos(angle)
        y = center[1] + radius * 0.5
        z = center[2] + radius * torch.sin(angle)

        # Update camera
        fig.update_layout(
            scene_camera=dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=center[0].item(), y=center[1].item(), z=center[2].item()),
                eye=dict(x=x.item(), y=y.item(), z=z.item()),
            )
        )

        # Save frame to a BytesIO object
        img_bytes = io.BytesIO()
        fig.write_image(img_bytes, format="png", width=800, height=800)
        img_bytes.seek(0)  # Reset pointer for reading
        frames.append(Image.open(img_bytes))

    # Create GIF in memory
    gif_bytes = io.BytesIO()
    frames[0].save(
        gif_bytes,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=500,  # Duration between frames in ms
        loop=0,  # Loop indefinitely
    )
    gif_bytes.seek(0)  # Reset pointer for reading

    return gif_bytes


@torch.no_grad
def visualize_scene(
    images: Float[torch.Tensor, "n c h w"],
    extrinsics: Float[torch.Tensor, "n 4 4"],
    intrinsics: Float[torch.Tensor, "n 3 3"],
    device: torch.device | str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    generate_gif: bool = False,
) -> Figure | Tuple[Figure, io.BytesIO]:
    """
    Visualize the entire scene, including cameras, rays, and images, using PyTorch3D.

    Args:
        images (Float[torch.Tensor, "n c h w"]): Tensor representing images associated with each camera.
        extrinsics (Float[torch.Tensor, "n 4 4"]): Tensor of shape (n, 4, 4) containing the extrinsic matrices of the cameras.
        intrinsics (Float[torch.Tensor, "n 3 3"]): Tensor of shape (n, 3, 3) containing the intrinsic matrices of the cameras.
        device (torch.device): The device for computations.
    """
    assert images.ndim == 4, "Input images should be of shape (n, c, h, w)"
    assert intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3), "Intrinsics should be of shape (n, 3, 3)"
    assert extrinsics.ndim == 3 and extrinsics.shape[1:] == (4, 4), "Extrinsics should be of shape (n, 4, 4)"

    if images.shape[1] == 1:  # Grayscale
        images = repeat(images, "n 1 h w -> n c h w", c=3)

    # Set the device for computations
    extrinsics = extrinsics.to(device)
    intrinsics = intrinsics.to(device)
    images = images.to(device)

    # Create image sizes for each camera
    n, _, *image_size = images.shape
    image_size_normalized = repeat(torch.tensor([1, 1], device=images.device), "s -> n s", n=n)

    # Visualize cameras
    cameras = visualize_cameras(extrinsics, intrinsics, image_size_normalized)
    rays = draw_rays(extrinsics, intrinsics, image_size, 2**0.5)
    images_mesh = draw_images_on_planes(images, extrinsics, intrinsics)

    # Use plotly to visualize the scene
    scene = {}
    for i in range(n):
        scene[f"image_{i}"] = images_mesh[i]
        scene[f"camera_{i}"] = cameras[i]
        scene[f"rays_{i}"] = rays[i]

    fig = plot_scene(
        {"scene": scene},
        axis_args=AxisArgs(showgrid=True, showline=True, zeroline=True, showticklabels=True, showaxeslabels=True),
        xaxis={"backgroundcolor": "rgb(200, 200, 230)"},
        yaxis={"backgroundcolor": "rgb(230, 200, 200)"},
        zaxis={"backgroundcolor": "rgb(200, 230, 200)"},
    )

    for item in fig["data"]:
        name = item["name"]
        group = name.split("_")[1]

        item["showlegend"] = True
        item["legendgroup"] = group
        item["legendgrouptitle"] = {"text": f"view_{group}"}

        item["visible"] = "legendonly" if "rays" in name else True

    fig.update_legends({"groupclick": "toggleitem"})

    # Calculate the ranges to display the scene in plotly
    margin = 1
    bbox_min, bbox_max = compute_combined_aabb(cameras, rays, images_mesh)

    bbox_min_margin = bbox_min - margin
    bbox_max_margin = bbox_max + margin

    # Calculate axis ranges
    range_xyz = bbox_max - bbox_min
    # Find the maximum range to make sure all squares are the same length
    max_range = range_xyz.max()

    # Calculate the aspect ratio
    aspect_ratio = dict(
        x=(range_xyz[0] / max_range).item(), y=(range_xyz[1] / max_range).item(), z=(range_xyz[2] / max_range).item()
    )

    # Set the aspect ratio to have an undistorted view on the scene
    fig["layout"]["scene"]["aspectmode"] = "manual"

    fig["layout"]["scene"]["xaxis"]["range"] = [bbox_min_margin[0].item(), bbox_max_margin[0].item()]
    fig["layout"]["scene"]["yaxis"]["range"] = [bbox_min_margin[1].item(), bbox_max_margin[1].item()]
    fig["layout"]["scene"]["zaxis"]["range"] = [bbox_min_margin[2].item(), bbox_max_margin[2].item()]

    fig["layout"]["scene"]["aspectratio"] = aspect_ratio

    if generate_gif:
        gif_bytes = generate_rotation_gif(fig, bbox_min, bbox_max, 16)
        return fig, gif_bytes

    return fig


# Example usage
if __name__ == "__main__":
    from scipy.spatial.transform import Rotation as R
    from torchvision import datasets, transforms

    n_cameras = 5
    radius = 2.0  # Fixed distance from the origin
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create intrinsic matrices (n, 3, 3)
    intrinsics = torch.eye(3).unsqueeze(0).repeat(n_cameras, 1, 1).to(device)
    intrinsics[:, 0, 0] = 0.8  # fx
    intrinsics[:, 1, 1] = 0.8  # fy
    intrinsics[:, 0, 2] = 0.5  # cx
    intrinsics[:, 1, 2] = 0.5  # cy

    # Create extrinsic matrices (n, 4, 4)
    extrinsics = torch.eye(4).unsqueeze(0).repeat(n_cameras, 1, 1).to(device)

    # Distribute cameras evenly around the y-axis on a circle
    for i in range(n_cameras):
        angle = (2 * np.pi / n_cameras) * i  # Calculate angle for each camera
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)

        # Set the translation part of the extrinsic matrix
        extrinsics[i, :3, 3] = torch.tensor([x, 0, z], device=device)

        # Calculate the rotation so that the camera looks towards the origin
        direction = torch.tensor([-x, 0, -z], device=device, dtype=torch.float)  # Point towards the origin
        direction = direction / torch.norm(direction)  # Normalize direction vector

        # Create a rotation matrix to align the camera's z-axis with the direction vector
        up = torch.tensor([0, 1, 0], device=device, dtype=torch.float)  # Define the up direction
        right = torch.cross(up, direction, dim=-1)  # Calculate the right direction
        right = right / torch.norm(right)  # Normalize right vector
        up = torch.cross(direction, right, dim=-1)  # Recalculate the up vector to ensure orthogonality

        # Set the rotation part of the extrinsic matrix
        rotation_matrix = torch.stack((right, up, direction), dim=1)
        extrinsics[i, :3, :3] = rotation_matrix

    # Load MNIST samples - Shape [5, 1, 28, 28]
    transform = transforms.ToTensor()
    mnist_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    images = torch.stack([mnist_dataset[i][0] for i in range(n_cameras)])  # Stack the images into a single tensor

    # Make them rectangular
    images = images[:, :, :, 4:24]

    # Visualize the cameras, rays, and images
    fig = visualize_scene(images, extrinsics, intrinsics)

    fig.show()
