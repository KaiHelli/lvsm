import io
from pathlib import Path
from typing import Union
import imageio
import os

import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]


def fig_to_image(
    fig: Figure,
    dpi: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "3 height width"]:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="raw", dpi=dpi)
    buffer.seek(0)
    data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    h = int(fig.bbox.bounds[3])
    w = int(fig.bbox.bounds[2])
    data = rearrange(data, "(h w c) -> c h w", h=h, w=w, c=4)
    buffer.close()
    return (torch.tensor(data, device=device, dtype=torch.float32) / 255)[:3]


def prep_image(image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()


def save_image(
    image: FloatImage,
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    Image.fromarray(prep_image(image)).save(path)


def load_image(
    path: Union[Path, str],
) -> Float[Tensor, "3 height width"]:
    return tf.ToTensor()(Image.open(path))[:3]


def save_video(
    images: list[FloatImage], path: Union[Path, str], ffmpeg_path: Union[Path, str] = "/usr/local/bin/"
) -> None:
    """Save a video from a list of images (assumed to be in range 0-1) using ImageIO.

    Parameters:
        images: List of images as tensors.
        path: Output video file path.
        ffmpeg_path: Directory or full path to the ffmpeg executable.
    """
    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Prepare frames.
    frames = [prep_image(image) for image in images]

    # Set ffmpeg executable for imageio_ffmpeg.
    # If ffmpeg_path is a directory, assume the executable is named "ffmpeg" inside it.
    ffmpeg_exe = Path(ffmpeg_path)
    if ffmpeg_exe.is_dir():
        ffmpeg_exe = ffmpeg_exe / "ffmpeg"
    print(f"Using ffmpeg at {ffmpeg_exe}")

    # Set the IMAGEIO_FFMPEG_EXE environment variable to force using a specific ffmpeg executable.
    os.environ["IMAGEIO_FFMPEG_EXE"] = str(ffmpeg_exe)

    # Write video using imageio.
    # Note: fps=30 is passed to imageio, and we add "-framerate 30" to inform FFmpeg about the input.
    with imageio.get_writer(
        str(path),
        fps=30,
        codec="libx264",
        ffmpeg_params=[
            "-r",
            "30",
            # Remove the explicit pix_fmt option to avoid duplicate settings:
            # "-pix_fmt", "yuv420p",
            "-crf",
            "21",
            "-vf",
            "setpts=1.*PTS",
        ],
    ) as writer:
        for frame in frames:
            writer.append_data(frame)
