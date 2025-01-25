from pathlib import Path
import argparse
import glob
import sys
import torch
from io import BytesIO
from PIL import Image
from torch import Tensor
from tqdm import tqdm
import shutil
import json
import torchvision.transforms as tf
from jaxtyping import Float, UInt8
from einops import repeat

from src.dataset.shims.crop_shim import rescale_and_crop
from src.model.vae import VAE

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_input", type=str, help="dataset source directory", default="datasets/re10k")
parser.add_argument("--dataset_output", type=str, help="dataset target directory", default="datasets/re10k_scaled")

parser.add_argument("--resize", type=bool, help="whether to resize the images", default=True)
parser.add_argument("--target_height", type=int, help="length of height to reduce to", default=192)
parser.add_argument("--target_width", type=int, help="length of width to reduce to", default=192)

parser.add_argument("--vae_encode", type=bool, help="whether to preencode resized images using vae", default=True)
parser.add_argument(
    "--vae_hf_model_id", type=str, help="huggingface id to vae model", default="stabilityai/stable-diffusion-3.5-large"
)
parser.add_argument("--vae_hf_subfolder", type=str, help="huggingface id to vae model", default="vae")
parser.add_argument("--vae_hf_gated", type=bool, help="whether the vae model is gated", default=True)
parser.add_argument("--vae_num_latent_channels", type=int, help="number of latent channels in vae model", default=16)
parser.add_argument("--vae_downsample_factor", type=int, help="downsample factor in vae model", default=8)
parser.add_argument("--num_images_in_parallel", type=int, help="number of images in parallel in vae model", default=1)

args = parser.parse_args()

DATASET_INPUT = Path(args.dataset_input)
DATASET_OUTPUT = Path(args.dataset_output)


def convert_images(
    self,
    images: list[UInt8[Tensor, "..."]],
) -> Float[Tensor, "batch 3 height width"]:
    torch_images = []
    for image in images:
        image = Image.open(BytesIO(image.numpy().tobytes()))
        torch_images.append(self.to_tensor(image))
    return torch.stack(torch_images)


def bytes_to_image(
    images: list[UInt8[Tensor, "..."]],
    expected_shape: tuple[int, int],
) -> list[UInt8[Tensor, "..."]]:
    images_out = []
    format = None

    bad_images = []

    to_tensor = tf.ToTensor()
    for i, image in enumerate(images):
        # Open the image from raw bytes
        image = Image.open(BytesIO(image.numpy().tobytes()))

        # Grab the original format; if None, fall back to something (e.g., "PNG")
        found_format = image.format if image.format is not None else "JPEG"

        # Initialize the format if it's not set
        format = format or found_format
        assert format == found_format, "All images must have the same format"

        image = to_tensor(image)

        # Filter out images that don't match the expected shape
        # Would cause issues with torch.stack
        if image.shape != expected_shape:
            bad_images.append(i)
            continue

        images_out.append(image)

    if len(images) - len(bad_images) > 0:
        images_out = torch.stack(images_out)
    else:
        images_out = None

    return images_out, format, bad_images


def image_to_bytes(images: list[UInt8[Tensor, "..."]], format: str) -> list[UInt8[Tensor, "..."]]:
    images_out = []
    to_pil_image = tf.ToPILImage()

    for image in images:
        # Save back to raw bytes in the same format
        buf = BytesIO()
        pil_image = to_pil_image(image)
        pil_image.save(buf, format=format)

        # Convert bytes to a torch.Tensor(dtype=torch.uint8)
        image_bytes = torch.frombuffer(buf.getbuffer(), dtype=torch.uint8)
        images_out.append(image_bytes)

    return images_out


def pose_to_intrinsics(
    poses: Float[Tensor, "batch 18"],
) -> tuple[Float[Tensor, "batch 4 4"], Float[Tensor, "batch 3 3"],]:  # extrinsics  # intrinsics
    b, _ = poses.shape

    # Convert the intrinsics to a 3x3 normalized K matrix.
    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    return intrinsics


def update_pose_intrinsics(
    poses: Float[Tensor, "batch 18"],
    intrinsics: Float[Tensor, "batch 3 3"],
) -> Float[Tensor, "batch 18"]:
    """Update the fx, fy, cx, cy entries in `poses` with the values from `intrinsics`."""

    # fx, fy, cx, cy come from the main diagonal and [0, 2], [1, 2] of the intrinsics.
    poses[:, 0] = intrinsics[:, 0, 0]  # fx
    poses[:, 1] = intrinsics[:, 1, 1]  # fy
    poses[:, 2] = intrinsics[:, 0, 2]  # cx
    poses[:, 3] = intrinsics[:, 1, 2]  # cy

    return poses


if __name__ == "__main__":
    # Create the output directory
    DATASET_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Load and update the dataset metadata
    with (DATASET_INPUT / "meta.json").open("r") as f:
        metadata = json.load(f)

        expected_shape = tuple(metadata["expected_shape"])

        if args.resize:
            num_channels, height, width = metadata["expected_shape"]
            # Update the expected shape
            metadata["expected_shape"] = [num_channels, args.target_height, args.target_width]
            print(f"Updated expected shape: {metadata['expected_shape']}")

        if args.vae_encode:
            metadata["vae_encoded"] = True
            metadata["vae_hf_model_id"] = args.vae_hf_model_id
            print(f"Updated VAE model: {metadata['vae_hf_model_id']}")

        json.dump(metadata, (DATASET_OUTPUT / "meta.json").open("w"), indent=2)
        print(f"Updated metadata: {metadata}")

    # If the path has a pattern, use glob to find matching files
    paths = glob.glob(str(DATASET_INPUT / "**/*.torch"), recursive=True)
    if not paths:
        print("No dataset files matched the pattern. Exit.")
        sys.exit(0)

    paths = [Path(path) for path in paths]

    if args.vae_encode:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the VAE model
        vae = VAE(
            args.vae_hf_model_id,
            args.vae_hf_subfolder,
            hf_gated=args.vae_hf_gated,
            num_latent_channels=args.vae_num_latent_channels,
            downsample_factor=args.vae_downsample_factor,
        )
        vae = vae.to(device)
        vae.eval()

    for path in (f_tbar := tqdm(paths, position=0)):
        f_tbar.set_postfix(path=path)

        output_path = DATASET_OUTPUT / path.relative_to(DATASET_INPUT)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if file already exists
        if output_path.exists():
            print(f"\nSkipping {path} as {output_path} already exists.")
            continue

        # Load the dataset
        dataset = torch.load(path)

        bad_scenes = []

        # Iterate over the dataset
        for i, scene in enumerate((s_pbar := tqdm(dataset, position=1))):
            s_pbar.set_postfix(scene=scene["key"])

            images, format, bad_images = bytes_to_image(scene["images"], expected_shape=expected_shape)

            if images is None:
                print(f"\n-- All images in {scene['key']} have unexpected shape. Skipping and removing scene.")
                bad_scenes.append(i)
                continue

            if bad_images:
                mask = torch.ones(len(images), dtype=torch.bool)
                mask[bad_images] = False

                print(f"\n-- Removing {len(bad_images)} images with unexpected shape from {scene['key']}")
                # Bad images were already filtered out, so we can just update the scene
                # images = images[mask]
                scene["timestamps"] = scene["timestamps"][mask]
                scene["cameras"] = scene["cameras"][mask]

            intrinsics = pose_to_intrinsics(scene["cameras"])

            if args.resize:
                images, intrinsics = rescale_and_crop(images, intrinsics, (args.target_height, args.target_width))

                assert images.shape[2:] == (
                    args.target_height,
                    args.target_width,
                ), "Images have unexpected shape after resizing."

            if args.vae_encode:
                # Encode the images using the VAE
                # Slice the images into smaller chunks if necessary, handling remainders
                images = images.to(device)

                num_images = images.shape[0]
                num_slices = (num_images + args.num_images_in_parallel - 1) // args.num_images_in_parallel
                slices = torch.chunk(images, num_slices, dim=0)

                # Encode each slice
                means = []
                stds = []
                with torch.no_grad(), torch.autocast(device_type=device):
                    for slice in slices:
                        mean, std = vae.encode(slice, sample=False)

                        means.append(mean)
                        stds.append(std)

                # Concatenate the latents
                mean = torch.cat(means, dim=0)
                std = torch.cat(stds, dim=0)

                scene["vae_latents"] = {"mean": mean, "std": std}

                images = images.cpu()

            scene["images"] = image_to_bytes(images, format)
            scene["cameras"] = update_pose_intrinsics(scene["cameras"], intrinsics)

        if bad_scenes:
            mask = torch.ones(len(dataset), dtype=torch.bool)
            mask[bad_scenes] = False

            print(f"\n# Removing {len(bad_scenes)} scene(s) with unexpected shape.")
            dataset = [scene for scene, keep in zip(dataset, mask) if keep]

        # Save the dataset
        torch.save(dataset, output_path)

    # Finally copy over auxiliary files, excluding the .torch files
    print("Copying auxiliary files...")
    for path in DATASET_INPUT.rglob("*"):
        # Skip the .torch files as well as meta.json
        if path.is_file() and path.suffix != ".torch" and path.name != "meta.json":
            output_path = DATASET_OUTPUT / path.relative_to(DATASET_INPUT)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output_path)
            print(f"  {path} -> {output_path}")

    print(f"\nDone. Preprocessed dataset saved to {DATASET_OUTPUT}.")
