from pathlib import Path
import os
import glob
from typing import Union, Optional

import wandb


def version_to_int(artifact) -> int:
    """Convert versions of the form vX to X. For example, v12 to 12."""
    return int(artifact.version[1:])


def download_checkpoint(
    run_id: str,
    download_dir: Path,
    version: str | None,
) -> Path:
    api = wandb.Api()
    run = api.run(run_id)

    # Find the latest saved model checkpoint.
    chosen = None
    for artifact in run.logged_artifacts():
        if artifact.type != "model" or artifact.state != "COMMITTED":
            continue

        # If no version is specified, use the latest.
        if version is None:
            if chosen is None or version_to_int(artifact) > version_to_int(chosen):
                chosen = artifact

        # If a specific verison is specified, look for it.
        elif version == artifact.version:
            chosen = artifact
            break

    # Download the checkpoint.
    download_dir.mkdir(exist_ok=True, parents=True)
    root = download_dir / run_id
    chosen.download(root=root)
    return root / "model.ckpt"

def update_checkpoint_path(path: Optional[str], wandb_cfg: dict) -> Optional[Path]:
    if path is None:
        return None

    # Check if path is a WandB URL
    if path.startswith("wandb://"):
        run_id, *version = path[len("wandb://"):].split(":")
        if len(version) == 0:
            version = None
        elif len(version) == 1:
            version = version[0]
        else:
            raise ValueError("Invalid version specifier!")

        project = wandb_cfg["project"]
        return download_checkpoint(
            f"{project}/{run_id}",
            Path("checkpoints"),
            version,
        )

    # Handle paths that do not start with "wandb://"
    checkpoint_path = Path(path)
    
    # If the path does not contain regex patterns and exists as a single file
    if checkpoint_path.is_file() and checkpoint_path.suffix == '.ckpt':
        return checkpoint_path

    # If the path has a pattern, use glob to find matching files
    matches = glob.glob(str(checkpoint_path))
    if not matches:
        print("No checkpoints matched the pattern. Skipping.")
        return None
    
    # Sort matches by creation time and get the newest file
    newest_file = max(matches, key=lambda x: os.path.getctime(x))

    # Print match
    print(f"Found checkpoint: {newest_file}. Loading.")

    return Path(newest_file)