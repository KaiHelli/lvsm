from ..types import BatchedExample, BatchedViews
from src.geometry.projection import calculate_plucker_rays
from einops import rearrange


def encode_relative_poses_batch(batch: BatchedExample) -> BatchedExample:
    """Encode the camera poses relative to the first context view."""

    # Get the w2c matrix of the reference context view
    w2c_reference = batch["context"]["extrinsics"][:, 0].inverse()
    w2c_reference = rearrange(w2c_reference, "b i j -> b 1 i j")

    # Combine the c2w extrinsic matrices with the w2c reference matrix to rotate/translate into the reference context view
    batch["context"]["extrinsics"] = w2c_reference @ batch["context"]["extrinsics"]
    batch["target"]["extrinsics"] = w2c_reference @ batch["target"]["extrinsics"]

    return batch
