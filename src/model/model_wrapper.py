from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json
from colorama import Fore

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample, BatchedViewsRGBD
from ..dataset import DatasetCfg
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import get_losses, LossCfgWrapper
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization import layout
from ..geometry.p3d_visualize_scene import visualize_scene

from .transformer.norm import LayerNorm
from .lvsm import LVSM, LVSMCfg
from .lr_scheduler import WarmupCosineLR
import plotly.io as pio
from wandb import Html


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    beta_1: float
    beta_2: float
    weight_decay: float
    initial_lr: float
    min_lr: float


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int


@dataclass
class TrainCfg:
    extended_visualization: bool
    print_log_every_n_steps: int


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],
        Float[Tensor, "batch view 3 3"],
    ]:  # extrinsics  # intrinsics
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    model_cfg: LVSMCfg
    loss_cfg: list[LossCfgWrapper]
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        model_cfg: LVSMCfg,
        loss_cfg: list[LossCfgWrapper],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()

        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.model = LVSM.from_cfg(model_cfg)

        self.data_shim = get_data_shim(self.model)
        self.losses = nn.ModuleList(get_losses(loss_cfg))

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        # TODO: Still needed?
        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"model": 0}

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # Run the model.
        output = self.model(
            batch["context"]["image"], batch["context"]["plucker_rays"], batch["target"]["plucker_rays"]
        )

        # Type the output.
        output = BatchedViewsRGBD({"color": output, "depth": None})

        target_gt = batch["target"]["image"]

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output["color"], "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, self.global_step)
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss
        self.log("loss/total", total_loss)

        if self.global_rank == 0 and self.global_step % self.train_cfg.print_log_every_n_steps == 0:
            print(
                Fore.RED
                + f"train"
                + Fore.RESET
                + f" | {self.global_step/self.trainer.max_steps*100:>6.2f}% [ep {self.current_epoch} | step {self.global_step}] | "
                f"loss = {total_loss:.6f} | "
                f"scene = {[x[:20] for x in batch['scene']]} | "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}] | "
                f"context = {batch['context']['index'].tolist()} | "
                f"target = {batch['target']['index'].tolist()}"
            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1

        # Run the model.
        with self.benchmarker.time("model"):
            output = self.model(
                batch["context"]["image"], batch["context"]["plucker_rays"], batch["target"]["plucker_rays"]
            )

        # Type the output.
        output = BatchedViewsRGBD({"color": output, "depth": None})

        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        images_prob = output["color"][0]
        rgb_gt = batch["target"]["image"][0]

        # Save images.
        if self.test_cfg.save_image:
            for index, color in zip(batch["target"]["index"][0], images_prob):
                save_image(color, path / scene / f"color/{index:0>6}.png")

        # save video
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in images_prob],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["model"] += 1  # TODO: += v?

            rgb = images_prob

            if f"psnr" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr"] = []
            if f"ssim" not in self.test_step_outputs:
                self.test_step_outputs[f"ssim"] = []
            if f"lpips" not in self.test_step_outputs:
                self.test_step_outputs[f"lpips"] = []

            self.test_step_outputs[f"psnr"].append(compute_psnr(rgb_gt, rgb).mean().item())
            self.test_step_outputs[f"ssim"].append(compute_ssim(rgb_gt, rgb).mean().item())
            self.test_step_outputs[f"lpips"].append(compute_lpips(rgb_gt, rgb).mean().item())

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        out_dir = self.test_cfg.output_path / name
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
            self.benchmarker.dump_memory(self.test_cfg.output_path / name / "peak_memory.json")
            self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                Fore.YELLOW
                + f"val"
                + Fore.RESET
                + f"   | {self.global_step/self.trainer.max_steps*100:>6.2f}% [ep {self.current_epoch} | step {self.global_step}] | "
                f"scene = {[a[:20] for a in batch['scene']]} | "
                f"context = {batch['context']['index'].tolist()} | "
                f"target = {batch['target']['index'].tolist()}"
            )

        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1

        # Run the model.
        output = self.model(
            batch["context"]["image"], batch["context"]["plucker_rays"], batch["target"]["plucker_rays"]
        )

        # Type the output.
        output = BatchedViewsRGBD({"color": output, "depth": None})

        rgb_out = output["color"][0]

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        for tag, rgb in zip(("val",), (rgb_out,)):
            psnr = compute_psnr(rgb_gt, rgb).mean()
            self.log(f"val/psnr_{tag}", psnr)
            lpips = compute_lpips(rgb_gt, rgb).mean()
            self.log(f"val/lpips_{tag}", lpips)
            ssim = compute_ssim(rgb_gt, rgb).mean()
            self.log(f"val/ssim_{tag}", ssim)

        # Construct comparison image.
        comparison = hcat(
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_out), "Target (Predicted)"),
        )
        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        # Visualize scene.
        images = torch.stack((*batch["context"]["image"][0], *batch["target"]["image"][0]), dim=0)
        intrinsics = torch.stack((*batch["context"]["intrinsics"][0], *batch["target"]["intrinsics"][0]), dim=0)
        extrinsics = torch.stack((*batch["context"]["extrinsics"][0], *batch["target"]["extrinsics"][0]), dim=0)
        with torch.amp.autocast("cuda" if images.is_cuda else "cpu", enabled=False):
            fig = visualize_scene(images, extrinsics, intrinsics, generate_gif=False)

        html_str = pio.to_html(fig, auto_play=False)
        html = wandb.Html(html_str)
        self.logger.log_table("scene", columns=["scene_html"], data=[[html]], step=self.global_step)

        # Rendering gif takes up too much time in validation
        # self.logger.log_video(
        #    "scene_rendered",
        #    [gif_bytes],
        #    step=self.global_step,
        #    caption=batch["scene"]
        # )

        # TODO: Look at this.
        # Run video validation step.
        # self.render_video_interpolation(batch)
        # self.render_video_wobble(batch)
        # if self.train_cfg.extended_visualization:
        #    self.render_video_interpolation_exaggerated(batch)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (batch["context"]["extrinsics"][0, 1] if v == 2 else batch["target"]["extrinsics"][0, 0]),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (batch["context"]["intrinsics"][0, 1] if v == 2 else batch["target"]["intrinsics"][0, 0]),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (batch["context"]["extrinsics"][0, 1] if v == 2 else batch["target"]["extrinsics"][0, 0]),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (batch["context"]["intrinsics"][0, 1] if v == 2 else batch["target"]["intrinsics"][0, 0]),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False)
        # gaussians_det = self.encoder(batch["context"], self.global_step, True)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth")
        images_prob = [vcat(rgb, depth) for rgb, depth in zip(output_prob["color"][0], depth_map(output_prob.depth[0]))]
        # output_det = self.decoder.forward(
        #     gaussians_det, extrinsics, intrinsics, near, far, (h, w), "depth"
        # )
        # images_det = [
        #     vcat(rgb, depth)
        #     for rgb, depth in zip(output_det["color"][0], depth_map(output_det.depth[0]))
        # ]
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Softmax"),
                    # add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")}

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(str(dir / f"{self.global_step:0>6}.mp4"), logger=None)

    def configure_optimizers(self):
        """
        Weight-decay exclusion adapted from minGPT:
        https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L215

        Some discussion on the topic:
        https://github.com/karpathy/minGPT/pull/24#issuecomment-679316025
        https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994
        """
        decay = set()
        no_decay = set()

        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, LayerNorm, nn.Embedding)

        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = f"{module_name}.{param_name}" if module_name else param_name

                if param_name.endswith("bias"):
                    # Exclude biases from weight decay.
                    no_decay.add(full_param_name)
                elif param_name == "weight" and isinstance(module, whitelist_weight_modules):
                    # Include weights of certain modules in weight decay.
                    decay.add(full_param_name)
                elif param_name == "weight" and isinstance(module, blacklist_weight_modules):
                    # Exclude weights of certain modules from weight decay.
                    no_decay.add(full_param_name)

        # validate that we considered every parameter
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"parameters {param_dict.keys() - union_params} were not separated into either decay/no_decay set!"

        # create optimizer groups
        optim_groups = [
            {
                "params": [param_dict[param_name] for param_name in sorted(list(decay))],
                "weight_decay": self.optimizer_cfg.weight_decay,
            },
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = optim.AdamW(
            optim_groups, lr=self.optimizer_cfg.lr, betas=(self.optimizer_cfg.beta_1, self.optimizer_cfg.beta_2)
        )

        lr_scheduler = WarmupCosineLR(
            optimizer,
            warmup_steps=self.optimizer_cfg.warm_up_steps,
            max_steps=self.trainer.max_steps,
            initial_lr=self.optimizer_cfg.initial_lr,
            min_lr=self.optimizer_cfg.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
