from ..types import BatchedExample, BatchedViews
from src.geometry.projection import calculate_plucker_rays


def generate_rays_views(views: BatchedViews) -> BatchedViews:
    """Calculate plucker rays for the views."""

    plucker_rays = calculate_plucker_rays(
        image=views["image"], intrinsics=views["intrinsics"], extrinsics=views["extrinsics"]
    )

    return {
        **views,
        "plucker_rays": plucker_rays,
    }


def generate_rays_batch(batch: BatchedExample) -> BatchedExample:
    """Calculate plucker rays for the entire batch."""
    return {
        **batch,
        "context": generate_rays_views(batch["context"]),
        "target": generate_rays_views(batch["target"]),
    }
