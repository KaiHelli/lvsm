import open3d as o3d
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from einops import rearrange, repeat

from ..geometry.projection import get_world_rays, sample_image_grid, get_world_points


def visualize_cameras(
    extrinsics: Float[np.ndarray, "n 4 4"],
    intrinsics: Float[np.ndarray, "n 3 3"],
    view_width_px: int = 1,
    view_height_px: int = 1,
    scale: float = 1.0,
) -> list:
    """
    Visualize multiple cameras in a scene using Open3D.

    Args:
        extrinsics (numpy.ndarray): A numpy array of shape (n, 4, 4) containing the extrinsic matrices of the cameras.
        intrinsics (numpy.ndarray): A numpy array of shape (n, 3, 3) containing the intrinsic matrices of the cameras.
        view_width_px (int, optional): Width of the camera view for visualization.
        view_height_px (int, optional): Height of the camera view for visualization.
        scale (float, optional): Scaling factor for the visualization of the cameras.

    Returns:
        list: A list of Open3D geometries representing the cameras.
    """
    assert intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3), "Intrinsics should be of shape (n, 3, 3)"
    assert extrinsics.ndim == 3 and extrinsics.shape[1:] == (4, 4), "Extrinsics should be of shape (n, 4, 4)"
    assert extrinsics.shape[0] == intrinsics.shape[0], "Number of intrinsics and extrinsics must match"

    # For whatever reason, Open3D expects the inverse of the extrinsics matrix for camera visualization
    extrinsics = np.linalg.inv(extrinsics)

    camera_geometries = []
    for i in range(intrinsics.shape[0]):
        # Create a LineSet representation of the camera using Open3D's built-in function
        camera = o3d.geometry.LineSet.create_camera_visualization(
            view_width_px, view_height_px, intrinsics[i], extrinsics[i], scale
        )
        camera.paint_uniform_color(
            [np.random.rand(), np.random.rand(), np.random.rand()]
        )  # Assign random color to each camera
        camera_geometries.append({"name": f"camera_{i}", "geometry": camera, "group": f"image_{i}"})

    return camera_geometries


def draw_rays(
    origins: Float[np.ndarray, "num_views num_rays 3"],
    directions: Float[np.ndarray, "num_views num_rays 3"],
    ray_length: float = 1.0,
) -> list:
    """
    Draw arbitrary rays in the scene.

    Args:
        origins (numpy.ndarray): A numpy array of shape (num_views, num_rays, 3), where each row represents the origin of a ray (x, y, z).
        directions (numpy.ndarray): A numpy array of shape (num_views, num_rays, 3), where each row represents the direction of a ray (dx, dy, dz).
        ray_length (float, optional): Length of the rays to be visualized.

    Returns:
        list: A list of Open3D geometries representing the rays.
    """
    assert origins.shape[2] == 3, "Input origins should have shape (num_views, num_rays, 3)"
    assert directions.shape[2] == 3, "Input directions should have shape (num_views, num_rays, 3)"
    assert origins.shape[0] == directions.shape[0], "Origins and directions should have the same number of views"
    assert origins.shape[1] == directions.shape[1], "Origins and directions should have the same number of rays"

    endpoints = origins + directions * ray_length

    num_views = origins.shape[0]
    num_rays = origins.shape[1]

    ray_geometries = []
    for i in range(num_views):
        # Create interleaved array of origins and endpoints
        interleaved_points = np.empty((num_rays * 2, 3))
        interleaved_points[0::2] = origins[i]
        interleaved_points[1::2] = endpoints[i]
        line_indices = np.arange(num_rays * 2).reshape(-1, 2)

        ray_line_set = o3d.geometry.LineSet()
        ray_line_set.points = o3d.utility.Vector3dVector(interleaved_points)
        ray_line_set.lines = o3d.utility.Vector2iVector(line_indices)

        # Define material for visualization
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "unlitLine"
        material.base_color = [0.25, 0.25, 0.25, 0.75]
        material.line_width = 1.0

        ray_geometries.append(
            {"name": f"rays_{i}", "geometry": ray_line_set, "material": material, "group": f"image_{i}"}
        )

    return ray_geometries


def draw_images_on_planes(images: Float[np.ndarray, "n c h w"], corners: Float[np.ndarray, "n 4 3"]) -> list:
    """
    Draw images in the scene associated with each camera.

    Args:
        images (numpy.ndarray): A numpy array of shape (n, c, h, w) representing images associated with each camera.
        corners (numpy.ndarray): A numpy array of shape (n, 4, 3) representing the corners of the image planes in 3D space.

    Returns:
        list: A list of Open3D geometries representing the images as planes in the scene.

    See https://github.com/isl-org/Open3D/discussions/5742
    """
    assert images.ndim == 4, "Images should be of shape (n, c, h, w)"
    n = images.shape[0]

    image_geometries = []
    for i in range(n):
        image_np = images[i]
        # Convert image to proper shape for Open3D (height, width, channels)
        if image_np.shape[0] == 3:  # RGB
            image_np = rearrange(image_np, "c h w -> h w c")
        elif image_np.shape[0] == 1:  # Grayscale
            image_np = repeat(image_np, "1 h w -> h w c", c=3)
        else:
            raise ValueError("Image must have 1 (grayscale) or 3 (RGB) channels")

        image_np = image_np.astype(np.float32)  # Ensure correct dtype for Open3D Image
        img = o3d.t.geometry.Image(o3d.core.Tensor(image_np))

        # Create mesh plane to display the image
        triangle_mesh = o3d.t.geometry.TriangleMesh()
        triangle_mesh.vertex.positions = o3d.core.Tensor(corners[i], o3d.core.float32)
        triangle_mesh.triangle.indices = o3d.core.Tensor([[0, 1, 3], [3, 2, 0]], o3d.core.int64)
        triangle_mesh.vertex.texture_uvs = o3d.core.Tensor(
            [[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]], o3d.core.float32
        )

        # Add image to frame as the albedo
        material = triangle_mesh.material
        material.material_name = "defaultLitTransparency"
        material.texture_maps["albedo"] = img

        image_geometries.append({"name": f"image_{i}", "geometry": triangle_mesh, "group": f"image_{i}"})

    return image_geometries


@torch.no_grad
def visualize_scene(
    images: Float[Tensor, "n c h w"], extrinsics: Float[Tensor, "n 4 4"], intrinsics: Float[Tensor, "n 3 3"]
) -> None:
    """
    Visualize the entire scene, including cameras, rays, and images, using Open3D.

    Args:
        images (torch.Tensor): Tensor representing images associated with each camera.
        extrinsics (torch.Tensor): Tensor of shape (n, 4, 4) containing the extrinsic matrices of the cameras.
        intrinsics (torch.Tensor): Tensor of shape (n, 3, 3) containing the intrinsic matrices of the cameras.
    """
    assert images.ndim == 4, "Input images should be of shape (n, c, h, w)"
    assert intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3), "Intrinsics should be of shape (n, 3, 3)"
    assert extrinsics.ndim == 3 and extrinsics.shape[1:] == (4, 4), "Extrinsics should be of shape (n, 4, 4)"

    n, _, *grid_shape = images.shape
    coordinates, _ = sample_image_grid(tuple(grid_shape), device=images.device)
    origins, directions = get_world_rays(rearrange(coordinates, "... d -> ... () d"), extrinsics, intrinsics)

    # Subsample the rays for visualization
    keep_every_x, keep_every_y = max(grid_shape[1] // 16, 1), max(grid_shape[0] // 16, 1)
    origins = origins[::keep_every_y, ::keep_every_x]
    directions = directions[::keep_every_y, ::keep_every_x]

    # Flatten the origins and directions tensors for Open3D
    origins, directions = rearrange(origins, "h w n d -> n (h w) d"), rearrange(directions, "h w n d -> n (h w) d")

    # Calculate corners of the image planes
    corners = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float32, device=images.device)
    corners = get_world_points(rearrange(corners, "... d -> ... () d"), extrinsics, intrinsics).permute(1, 0, 2)

    # Upscale image resolution for sharp texture visualization
    target_height, target_width = 2048, 2048
    scale_factor_x, scale_factor_y = max((target_width + grid_shape[1] - 1) // grid_shape[1], 1), max(
        (target_height + grid_shape[0] - 1) // grid_shape[0], 1
    )
    images = repeat(images, "n c h w -> n c (h h_res) (w w_res)", h_res=scale_factor_y, w_res=scale_factor_x)

    # Convert tensors to numpy arrays
    images, extrinsics, intrinsics, origins, directions, corners = [
        x.cpu().numpy() for x in [images, extrinsics, intrinsics, origins, directions, corners]
    ]

    # Visualize the scene
    camera_geometries = visualize_cameras(extrinsics, intrinsics, 1, 1, 0.25)
    ray_geometries = draw_rays(origins, directions, 1.0)
    image_geometries = draw_images_on_planes(images, corners)

    all_geometries = camera_geometries + ray_geometries + image_geometries

    # Add coordinate frame to the scene
    world_coordinate_frame = {
        "name": "world_coordinate_frame",
        "geometry": o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0]),
    }
    all_geometries.append(world_coordinate_frame)

    # Open3D visualization
    o3d.visualization.draw(all_geometries, show_ui=True, title="Scene Visualization")


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
    visualize_scene(images, extrinsics, intrinsics)
