from pathlib import Path
import argparse
import glob
import sys
import torch
from io import BytesIO
from tqdm import tqdm
import shutil
import blosc2

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_input", type=str, help="dataset source directory", default="datasets/re10k")
parser.add_argument("--dataset_output", type=str, help="dataset target directory", default="datasets/re10k_scaled")
args = parser.parse_args()

DATASET_INPUT = Path(args.dataset_input)
DATASET_OUTPUT = Path(args.dataset_output)

if __name__ == "__main__":
    # Create the output directory
    DATASET_OUTPUT.mkdir(parents=True, exist_ok=True)

    # If the path has a pattern, use glob to find matching files
    paths = glob.glob(str(DATASET_INPUT / "**/*.torch"), recursive=True)
    if not paths:
        print("No dataset files matched the pattern. Exit.")
        sys.exit(0)

    paths = [Path(path) for path in paths]

    # Copy over auxiliary files, excluding the .torch files
    print("\nCopying auxiliary files...")
    for path in DATASET_INPUT.rglob("*"):
        # Skip the .torch files
        if path.is_file() and path.suffix != ".torch":
            output_path = DATASET_OUTPUT / path.relative_to(DATASET_INPUT)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output_path)
            print(f"  {path} -> {output_path}")

    # Keep track of total savings in bytes
    total_savings_bytes = 0

    for path in (pbar := tqdm(paths, desc="Compressing", unit="file", position=0)):
        # Prepare the output path
        output_path = DATASET_OUTPUT / path.relative_to(DATASET_INPUT)
        # Switch .torch to .bl2
        output_path = output_path.with_suffix(".bl2")

        # Skip if file already exists
        if output_path.exists():
            print(f"\nSkipping {path} as {output_path} already exists.")
            continue

        # Create the output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load the dataset
        dataset = torch.load(path)

        # Serialize the dataset into a BytesIO buffer
        with BytesIO() as saved_data:
            torch.save(dataset, saved_data)
            # The pointer is now at the end of the BytesIO buffer
            uncompressed_size = saved_data.tell()
            # Reset pointer to the beginning
            saved_data.seek(0)

            # Compress pickled bytes using Blosc
            compressed_data = blosc2.compress2(saved_data.read())
            compressed_size = len(compressed_data)

        # Calculate how many bytes we've saved in this iteration
        saved_bytes = uncompressed_size - compressed_size
        total_savings_bytes += saved_bytes

        # Write compressed data to disk
        with open(output_path, "wb") as f:
            f.write(compressed_data)

        # Update progress bar postfix
        saved_gb = total_savings_bytes / (1024 ** 3)
        pbar.set_postfix(
            path=str(path),
            saved_GB=f"{saved_gb:.2f}"
        )



    print(f"\nDone. Compressed dataset saved to {DATASET_OUTPUT}.")
