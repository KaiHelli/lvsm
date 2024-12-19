from ..types import BatchedExample, BatchedViews
from src.geometry.projection import calculate_plucker_rays
from einops import rearrange, einsum, repeat
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
import torch


def encode_relative_poses_batch_fc(batch: BatchedExample) -> BatchedExample:
    """Encode the camera poses relative to the first context view."""

    # Get the w2c matrix of the reference context view
    w2c_reference = batch["context"]["extrinsics"][:, 0].inverse()
    w2c_reference = rearrange(w2c_reference, "b i j -> b 1 i j")

    # Combine the c2w extrinsic matrices with the w2c reference matrix to rotate/translate into the reference context view
    batch["context"]["extrinsics"] = w2c_reference @ batch["context"]["extrinsics"]
    batch["target"]["extrinsics"] = w2c_reference @ batch["target"]["extrinsics"]

    return batch


def encode_relative_poses_batch_avgc(batch: BatchedExample) -> BatchedExample:
    """
    Encode the camera poses relative to the average position and rotation of all context views.

    Source for averaging rotation matrices:
    http://tbirdal.blogspot.com/2019/10/i-allocate-this-post-to-providing.html
    """
    # Extract all extrinsics: shape (B, N, 4, 4) where N is the number of context views
    extrinsics = batch["context"]["extrinsics"]  # c2w transforms

    if extrinsics.shape[1] == 1:
        # If there is only one context view, we encode relative poses w.r.t. that view
        return encode_relative_poses_batch_fc(batch)

    # Separate rotation and translation
    R = extrinsics[..., :3, :3]  # (B, N, 3, 3)
    t = extrinsics[..., :3, 3]  # (B, N, 3)

    # Compute the mean translation across all context views
    t_mean = t.mean(dim=1)  # (B, 3)

    # Convert rotation matrices to quaternions for averaging.
    q = matrix_to_quaternion(R)  # (B, N, 4)

    # Ensure quaternions have a consistent "handedness" by flipping those with a negative w-component
    # q[..., 0] is shape (B, N), so we create a mask
    mask = q[..., 0] < 0
    q[mask] = -q[mask]

    # Compute the averaging matrix A = (1/N) * Σ q_n q_n^T over all context quaternions
    N = q.shape[1]
    A = (1 / N) * einsum(q, q, "b n i, b n j -> b i j")  # (B, 4, 4)

    # Perform eigen-decomposition of A to find the best representative quaternion
    evals, evecs = torch.linalg.eigh(A)  # vals: (B,4), evecs: (B,4,4)

    # The principal component (eigenvector with the largest eigenvalue) gives the average quaternion
    q_mean = evecs[..., -1]  # (B, 4)

    # Normalize the averaged quaternions to unit length
    q_mean = q_mean / q_mean.norm(dim=-1, keepdim=True)

    # Convert the averaged quaternion back to a rotation matrix
    R_mean = quaternion_to_matrix(q_mean)  # (B, 3, 3)

    # Construct the average camera-to-world matrix from R_mean and t_mean
    c2w_mean = torch.eye(4, device=extrinsics.device).unsqueeze(0).repeat(R_mean.shape[0], 1, 1)
    c2w_mean[:, :3, :3] = R_mean
    c2w_mean[:, :3, 3] = t_mean

    # Invert to get w2c reference
    w2c_reference = torch.inverse(c2w_mean)  # (B, 4, 4)
    w2c_reference = rearrange(w2c_reference, "b i j -> b 1 i j")

    # Transform all context and target extrinsics into this new reference frame
    batch["context"]["extrinsics"] = w2c_reference @ batch["context"]["extrinsics"]
    batch["target"]["extrinsics"] = w2c_reference @ batch["target"]["extrinsics"]

    # Normalize to bbox [-1, -1, -1] up to [1, 1, 1]
    context_positions = batch["context"]["extrinsics"][..., :3, 3]  # (B, N, 3)

    # Determine min/max of the bounding box from the context camera positions
    pos_min = context_positions.min(dim=1)[0]  # (B, 3)
    pos_max = context_positions.max(dim=1)[0]  # (B, 3)

    # Compute the center of the bounding box
    center = 0.5 * (pos_min + pos_max)  # midpoint (B, 3)
    # Compute scaling and translation factors to normalize positions into the range [-1, 1].
    # We keep the aspect ratio between the axes by scaling w.r.t. the largest range.
    ranges = pos_max - pos_min  # (B, 3)
    max_range = ranges.max(dim=1, keepdim=True)[0]  # (B, 1)
    # Compute scale factor to fit the largest dimension into [-1,1]
    # The full width for that dimension after scaling should be 2.0
    scale = 2.0 / max_range  # (B, 1)

    def normalize_extrinsics(extrinsics, scale, center):
        # Extract old translations: (B, N, 3)
        old_translations = extrinsics[..., :3, 3]

        # Use einops to broadcast scale and mid properly
        # scale: (B,1) -> (B,N,1) via repeat
        B, N = extrinsics.shape[:2]
        scale_3d = repeat(scale, "b 1 -> b n 1", n=N)  # (B, N, 1)
        mid_expanded = repeat(center, "b c -> b n c", n=N)  # (B, N, 3)

        # Apply normalization
        new_translations = scale_3d * (old_translations - mid_expanded)  # (B, N, 3)

        # Assign back to extrinsics
        extrinsics[..., :3, 3] = new_translations
        return extrinsics

    # Apply normalization to both context and target
    batch["context"]["extrinsics"] = normalize_extrinsics(batch["context"]["extrinsics"], scale, center)
    batch["target"]["extrinsics"] = normalize_extrinsics(batch["target"]["extrinsics"], scale, center)

    return batch
