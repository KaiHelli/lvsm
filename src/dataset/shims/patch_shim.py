from ..types import BatchedExample, BatchedViews
from einops import rearrange


def apply_patch_shim_to_views(views: BatchedViews, patch_size: int) -> BatchedViews:
    _, _, _, h, w = views["image"].shape

    # Image size must be even so that naive center-cropping does not cause misalignment.
    assert h % 2 == 0 and w % 2 == 0

    h_new = (h // patch_size) * patch_size
    row = (h - h_new) // 2
    w_new = (w // patch_size) * patch_size
    col = (w - w_new) // 2

    # Center-crop the image.
    image = views["image"][:, :, :, row : row + h_new, col : col + w_new]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = views["intrinsics"].clone()
    intrinsics[:, :, 0, 0] *= w / w_new  # fx
    intrinsics[:, :, 1, 1] *= h / h_new  # fy

    return {
        **views,
        "image": image,
        "intrinsics": intrinsics,
    }


def apply_patch_shim(batch: BatchedExample, patch_size: int) -> BatchedExample:
    """Crop images in the batch so that their dimensions are cleanly divisible by the
    specified patch size.
    """
    return {
        **batch,
        "context": apply_patch_shim_to_views(batch["context"], patch_size),
        "target": apply_patch_shim_to_views(batch["target"], patch_size),
    }


def patchify_views(views: BatchedViews, patch_size: int) -> BatchedViews:
    """Split the images in the views into non-overlapping patches of the given patch size."""
    b, n, c, h, w = views["image"].shape  # batch, num_views, channels, height, width
    assert h % patch_size == 0 and w % patch_size == 0, "Image dimensions must be divisible by patch size."

    # Use einops to create patches.
    patches = rearrange(views["image"], "b n c (h ph) (w pw) -> b n (h w) c ph pw", ph=patch_size, pw=patch_size)
    # patches = rearrange(views["image"], 'b n c (h ph) (w pw) -> b n h w (c ph pw)', ph=patch_size, pw=patch_size)

    return {
        **views,
        "image_patches": patches,
    }


def patchify_batch(batch: BatchedExample, patch_size: int) -> BatchedExample:
    """Apply the patchify operation to the entire batch."""
    return {
        **batch,
        "context": patchify_views(batch["context"], patch_size),
        "target": patchify_views(batch["target"], patch_size),
    }
