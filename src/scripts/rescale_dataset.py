from pathlib import Path
import argparse
import glob
import sys
import torch
from io import BytesIO
from PIL import Image
from jaxtyping import UInt8
from torch import Tensor
from tqdm import tqdm
import shutil
import json

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_input", type=str, help="dataset source directory", default="datasets/re10k")
parser.add_argument("--dataset_output", type=str, help="dataset target directory", default="datasets/re10k_scaled")
parser.add_argument("--min_size", type=int, help="minimum length of either width or height to reduce to", default=32)
args = parser.parse_args()

DATASET_INPUT = Path(args.dataset_input)
DATASET_OUTPUT = Path(args.dataset_output)


def get_target_size(width: int, height: int, min_size: int) -> tuple[int, int]:
    if width < height:
        target_width = min_size
        target_height = int(min_size * height / width)
    else:
        target_width = int(min_size * width / height)
        target_height = min_size

    return target_width, target_height


def convert_images(
    images: list[UInt8[Tensor, "..."]],
    min_size: int,
) -> list[UInt8[Tensor, "..."]]:
    images_out = []
    for image in images:
        # Open the image from raw bytes
        pil_img = Image.open(BytesIO(image.numpy().tobytes()))

        # Grab the original format; if None, fall back to something (e.g., "PNG")
        fmt = pil_img.format if pil_img.format is not None else "JPEG"

        # Get the original width and height
        width, height = pil_img.size

        assert min_size <= min(width, height), "Minimum size must be smaller than the image itself."

        # Determine the target width and height
        target_width, target_height = get_target_size(width, height, min_size)

        # Resize the image
        pil_img = pil_img.resize((target_width, target_height), resample=Image.LANCZOS)

        # Save back to raw bytes in the same format
        buf = BytesIO()
        pil_img.save(buf, format=fmt)

        # Convert bytes back to a torch.Tensor(dtype=torch.uint8)
        new_image_tensor = torch.frombuffer(buf.getbuffer(), dtype=torch.uint8)
        images_out.append(new_image_tensor)

    return images_out


if __name__ == "__main__":
    # Create the output directory
    DATASET_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Load and update the dataset metadata
    with (DATASET_INPUT / "meta.json").open("r") as f:
        metadata = json.load(f)

        num_channels, height, width = metadata["expected_shape"]

        target_width, target_height = get_target_size(width, height, args.min_size)

        # Update the expected shape
        metadata["expected_shape"] = [num_channels, target_height, target_width]

        json.dump(metadata, (DATASET_OUTPUT / "meta.json").open("w"), indent=2)
        print(f"Updated metadata: {metadata}")

    # If the path has a pattern, use glob to find matching files
    paths = glob.glob(str(DATASET_INPUT / "**/*.torch"), recursive=True)
    if not paths:
        print("No dataset files matched the pattern. Exit.")
        sys.exit(0)

    paths = [Path(path) for path in paths]

    for path in (f_tbar := tqdm(paths, position=0)):
        f_tbar.set_postfix(path=path)

        # Load the dataset
        dataset = torch.load(path)

        # Iterate over the dataset and resize the images
        for scene in (s_pbar := tqdm(dataset, position=1)):
            s_pbar.set_postfix(scene=scene["key"])

            scene["images"] = convert_images(scene["images"], args.min_size)

        # Save the dataset
        output_path = DATASET_OUTPUT / path.relative_to(DATASET_INPUT)
        output_path.parent.mkdir(parents=True, exist_ok=True)
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

    print(f"\nDone. Rescaled dataset saved to {DATASET_OUTPUT}.")
