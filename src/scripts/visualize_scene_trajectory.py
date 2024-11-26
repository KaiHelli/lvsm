from pathlib import Path
from random import randrange
from dataclasses import dataclass

import hydra
import torch
from jaxtyping import install_import_hook
from omegaconf import DictConfig
from einops import pack
from lightning_fabric.utilities.apply_func import move_data_to_device

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_config
    from src.dataset.data_module import DataModule
    from src.dataset import DatasetCfg
    from src.dataset.data_module import DataLoaderCfg, DataModule
    from src.dataset.shims.plucker_rays import generate_rays_views
    from src.global_cfg import set_cfg
    from src.misc.step_tracker import StepTracker
    from src.visualization.annotation import add_label
    from src.visualization.layout import add_border, hcat
    from src.visualization.p3d_visualize_scene import visualize_scene
    from src.visualization.color_map import plucker_to_colormaps
    from src.misc.LocalLogger import LocalLogger



@dataclass
class RootCfg:
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    seed: int

@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="visualize_scene_trajectory",
)
def visualize_scene_trajectory(cfg_dict: DictConfig):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Boilerplate configuration stuff like in the main file...
    cfg = load_typed_config(cfg_dict, RootCfg)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)
    data_module = DataModule(cfg.dataset, cfg.data_loader, StepTracker())
    
    # dataset = iter(data_module.train_dataloader())
    dataset = iter(data_module.test_dataloader())

    # Load all frames of the scene
    all_frames = next(dataset)["context"]

    # Offload work to the gpu if possible
    all_frames = move_data_to_device(all_frames, device)

    # Calculate the plucker ray embeddings for this scene
    all_frames = generate_rays_views(all_frames)

    # Remove the outermost singleton batch dimension
    all_frames = {key: all_frames[key][0] for key in ["image", "extrinsics", "intrinsics", "plucker_rays"]}

    # Subsample frames for the 3D visualization
    sub_frames = {key: all_frames[key][::10] for key in ["image", "extrinsics", "intrinsics", "plucker_rays"]}

    # Visualize the scene in 3D
    figure = visualize_scene(sub_frames["image"], sub_frames["extrinsics"], sub_frames["intrinsics"], device=device, generate_gif=False)
    figure.show()

    # Visualize the scene in a 2D video
    directions_cm, momentum_cm = plucker_to_colormaps(all_frames["plucker_rays"])

    images = [
        add_border(
            hcat(
                add_label(output, "View"),
                add_label(directions, "Plucker Directions"),
                add_label(momentum, "Plucker Momentum")
            )
        )
        for output, directions, momentum in zip(all_frames["image"], directions_cm, momentum_cm)
    ]

    video = torch.stack(images)
    video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
    
    # Loop the video in reverse
    video = pack([video, video[::-1][1:-1]], "* c h w")[0]
    
    logger = LocalLogger()

    logger.log_video(
        f"video/{cfg.dataset.overfit_to_scene}",
        [video],
        step=0,
        caption=[f"scene {cfg.dataset.overfit_to_scene}"],
        fps=[30],
        format=["mp4"]
    )


if __name__ == "__main__":
    visualize_scene_trajectory()
