from dataclasses import dataclass
import hydra
import torch
from pathlib import Path
from jaxtyping import install_import_hook
from omegaconf import DictConfig
from einops import pack, rearrange
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
    from src.dataset.shims.relative_poses import encode_relative_poses_batch_avgc
    from src.global_cfg import set_cfg
    from src.misc.step_tracker import StepTracker
    from src.visualization.annotation import add_label
    from src.visualization.layout import add_border, hcat
    from src.visualization.p3d_visualize_scene import visualize_scene
    from src.visualization.color_map import plucker_to_colormaps
    from src.misc.LocalLogger import LocalLogger

NUM_SCENES = 5
LOG_PATH = Path("outputs/local")


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

    dataset = iter(data_module.train_dataloader())
    # dataset = iter(data_module.test_dataloader())

    scene_list = []
    scene_names = []
    for _ in range(NUM_SCENES):
        # Load all frames of the scene
        ds = next(dataset)

        # Skip scenes that don't contain enough frames
        while ds["context"]["image"].shape[1] < 50:
            ds = next(dataset)

        ds = encode_relative_poses_batch_avgc(ds)

        scene_list += [ds["context"]]
        scene_names += [ds["scene"]]

    def collate_fn(batch, keep_frames=10):
        batch_by_key = {
            key: [elem[key][0, :: elem[key].shape[1] // keep_frames] for elem in batch] for key in batch[0].keys()
        }

        scene_indices = [
            i * torch.ones((elem.shape[0]), dtype=torch.long) for i, elem in enumerate(batch_by_key["image"])
        ]
        scene_indices = torch.cat(scene_indices, dim=0)

        batch_out = {key: torch.cat(batch_by_key[key], dim=0) for key in batch_by_key.keys()}

        return batch_out, scene_indices

    scenes, scene_indices = collate_fn(scene_list)
    # Subsample frames for the 3D visualization
    sub_scenes, sub_scene_indices = collate_fn(scene_list, 5)

    # Offload work to the gpu if possible
    scenes = move_data_to_device(scenes, device)
    scene_indices = move_data_to_device(scene_indices, device)

    sub_scenes = move_data_to_device(sub_scenes, device)
    sub_scene_indices = move_data_to_device(sub_scene_indices, device)

    scene_ids = torch.unique(scene_indices)

    # Calculate the plucker ray embeddings for this scene
    scenes = generate_rays_views(scenes)

    # Visualize the scene in 3D
    figure = visualize_scene(
        sub_scenes["image"], sub_scenes["extrinsics"], sub_scenes["intrinsics"], device=device, generate_gif=False
    )

    # Visualize the scene in a 2D video
    directions_cm, momentum_cm = plucker_to_colormaps(scenes["plucker_rays"])

    logger = LocalLogger()

    # Logger initialization cleans outputs/local therefore write html after.
    path = LOG_PATH / "scene.html"
    path.parent.mkdir(exist_ok=True, parents=True)
    figure.write_html(path)

    for scene_id in scene_ids:
        scene_images = scenes["image"][scene_indices == scene_id]
        scene_directions = directions_cm[scene_indices == scene_id]
        scene_momentum = momentum_cm[scene_indices == scene_id]

        images = [
            add_border(
                hcat(
                    add_label(output, "View"),
                    add_label(directions, "Plucker Directions"),
                    add_label(momentum, "Plucker Momentum"),
                )
            )
            for output, directions, momentum in zip(scene_images, scene_directions, scene_momentum)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()

        # Loop the video in reverse
        video = pack([video, video[::-1][1:-1]], "* c h w")[0]

        logger.log_video(
            f"video/{scene_names[scene_id.item()]}",
            [video],
            step=0,
            caption=[f"scene {scene_names[scene_id.item()]}"],
            fps=[30],
            format=["mp4"],
        )


if __name__ == "__main__":
    visualize_scene_trajectory()
