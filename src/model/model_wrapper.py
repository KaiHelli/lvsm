from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from functools import lru_cache
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
from packaging.version import parse as parse_version
from time import perf_counter

from ..geometry.projection import calculate_plucker_rays
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
from ..misc.utils import tensor_on_gpu, model_on_gpu
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image, plucker_to_colormaps
from ..visualization.layout import add_border, hcat, vcat
from ..visualization import layout
from ..visualization.p3d_visualize_scene import visualize_scene

from .transformer.norm import LayerNorm, LayerNorm2d, QKScaleUp, QKNorm
from .lora.lora import add_lora, LoRAConfig
from .lora.lora_utils import get_lora_params, get_lora_state_dict, get_lora_original_state_dict
from .lvsm import LVSM, LVSMCfg
from .lr_scheduler import WarmupCosineLR
import plotly.io as pio
from wandb import Html

if parse_version(torch.__version__) >= parse_version("2.5.0"):
    from torch.nn.attention.flex_attention import create_block_mask

    create_block_mask = torch.compile(create_block_mask)


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
    ffmpeg_path: Path


@dataclass
class TrainCfg:
    extended_visualization: bool
    print_log_every_n_steps: int
    val_every_n_batches: int
    vis_every_n_validations: int
    decode_vae_latents: bool
    finetune_vae_decoder: Optional[LoRAConfig]


@dataclass
class LossCfg:
    vis_loss: list[LossCfgWrapper] = field(default_factory=list)
    vis_loss_weight: float = 0.0
    latent_loss: list[LossCfgWrapper] = field(default_factory=list)
    latent_loss_weight: float = 0.0

    def __post_init__(self):
        if len(self.vis_loss) > 0:
            assert self.vis_loss_weight >= 0.0, "vis_loss_weight must be non-negative"
        if len(self.latent_loss) > 0:
            assert self.latent_loss_weight >= 0.0, "latent_loss_weight must be non-negative"

        assert len(self.vis_loss) > 0 or len(self.latent_loss) > 0, "At least one loss must be specified"


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[Float[Tensor, "batch view 4 4"], Float[Tensor, "batch view 3 3"],]:  # extrinsics  # intrinsics
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
        loss_cfg: LossCfg,
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()

        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.model_cfg = model_cfg
        self.sdpa_kernel = model_cfg.transformer_cfg.sdpa_kernel

        # Track that the model is not configured yet.
        self.model_configured = False

        self.loss_cfg = loss_cfg
        self.vis_losses = nn.ModuleList(get_losses(loss_cfg.vis_loss))
        self.latent_losses = nn.ModuleList(get_losses(loss_cfg.latent_loss))

        # Track the number of calls to validation_step()
        # self.register_buffer("num_validations", torch.tensor(0, dtype=torch.int64))
        self.num_validations = 0

        # Track wether validation just generated visuals to pick that up in the next training_step()
        self.val_generated_vis = False

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"model": 0}

    def configure_model(self):
        if self.model_configured:
            return

        self.model = LVSM.from_cfg(self.model_cfg)
        self.data_shim = get_data_shim(self.model)

        self.vae_decoder_requires_grad = False

        if self.model.vae is not None and self.train_cfg.finetune_vae_decoder is not None:
            assert self.train_cfg.decode_vae_latents, "Finetuning the VAE decoder requires decoding the latents."

            self.vae_decoder_requires_grad = True

            def enable_lora_finetuning(module):
                print("Finetuning the VAE decoder with LoRA.")

                # Add LoRA params to the VAE decoder.
                add_lora(module.vae.vae.decoder, self.train_cfg.finetune_vae_decoder)

                # First freeze all parameters of the model.
                module.requires_grad_(False)

                # Enable gradients for all added LoRA parameters.
                for param in get_lora_params(module.vae.vae.decoder):
                    param.requires_grad_(True)

                # Set all to eval mode first.
                module.eval()

                # Set the decoder to train mode
                module.vae.vae.decoder.train()

                # Print all parameters that are not frozen.
                ft_params = [name for name, param in module.named_parameters() if param.requires_grad]
                print("Finetuning the following parameters:", ft_params)

                # Print all modules in train mode:
                # train_modules = [name for name, mod in module.named_modules() if mod.training]
                # print("Modules in train mode:", train_modules)

            enable_lora_finetuning(self.model)

        # Compile the model to achieve some speedup.
        # For now only compile if a GPU is available.
        if torch.cuda.is_available():
            print("Using torch.compile() to speed up model.")
            self.model = torch.compile(self.model, fullgraph=True)

        def patch_state_dict(
            module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            """
            A pre-hook to fix or rename parameters before PyTorch performs strict matching.
            Removes '_orig_mod.' prefixes for parameters from compiled checkpoints when loaded on uncompiled models.
            Adds '_orig_mod.' prefixes for parameters from uncompiled checkpoints when loaded on compiled models.
            """
            from torch._dynamo import OptimizedModule

            is_lvsm = (isinstance(module, OptimizedModule) and isinstance(module._orig_mod, LVSM)) or isinstance(
                module, LVSM
            )

            if not is_lvsm:
                return

            # Remove Dynamo/compiled '_orig_mod.' prefixes if needed
            if not isinstance(module, OptimizedModule):
                unwanted_prefix = "_orig_mod."
                unwanted_match = prefix + unwanted_prefix

                for old_key in list(state_dict.keys()):
                    # If a key starts with e.g. "model._orig_mod." but we want "model."
                    if old_key.startswith(unwanted_match):
                        new_key = old_key.replace(unwanted_match, prefix, 1)
                        state_dict[new_key] = state_dict.pop(old_key)
                        print(f"[load_state_dict_pre_hook] Renamed from compiled checkpoint: {old_key} -> {new_key}")

            # Add Dynamo/compiled '_orig_mod.' prefixes if needed
            if isinstance(module, OptimizedModule):
                wanted_prefix = "_orig_mod."
                wanted_match = prefix + wanted_prefix

                print(f"{prefix=}, {wanted_match=}")

                for old_key in list(state_dict.keys()):
                    # Otherwise, insert "_orig_mod." right after "model."
                    if old_key.startswith(prefix) and not old_key.startswith(wanted_match):
                        new_key = old_key.replace(prefix, prefix + wanted_prefix, 1)
                        state_dict[new_key] = state_dict.pop(old_key)
                        print(f"[load_state_dict_pre_hook] Renamed from uncompiled checkpoint: {old_key} -> {new_key}")

            # In case we load a checkpoint that was saved without LoRA parametrization, we need to match the keys.
            if self.train_cfg.finetune_vae_decoder:
                lora_params = get_lora_original_state_dict(module)

                for key in lora_params:
                    if key in state_dict:
                        continue

                    full_key = prefix + key
                    split = full_key.split(".")
                    key_without_lora = ".".join(split[:-3]) + "." + split[-2]
                    if key_without_lora in state_dict:
                        state_dict[full_key] = state_dict.pop(key_without_lora)
                        print(
                            f"[load_state_dict_pre_hook] Matched missing LoRA key with unexpected key: {key_without_lora} -> {full_key}"
                        )

        def ignore_lora_keys(module, incompatible_keys):
            """
            A post-hook to ignore LoRA parameters that are not present in the checkpoint.
            This makes it possible to load checkpoints that were saved without LoRA parametrization in order to finetune parts of the model.
            """
            from torch._dynamo import OptimizedModule

            is_lvsm = (isinstance(module, OptimizedModule) and isinstance(module._orig_mod, LVSM)) or isinstance(
                module, LVSM
            )

            if not is_lvsm:
                return

            from torch._dynamo import OptimizedModule

            lora_params = get_lora_state_dict(module)
            compiled = isinstance(module, OptimizedModule)
            num_prefixes = 2 if compiled else 1
            missing_keys = incompatible_keys.missing_keys
            # Go in reverse order to avoid changing the indices of the keys.
            for i in range(len(missing_keys) - 1, -1, -1):
                if missing_keys[i].split(".", num_prefixes)[-1] in lora_params:
                    key = missing_keys.pop(i)
                    print(f"[load_state_dict_post_hook] Ignored missing LoRA key: {key}")

            # Workaround for old checkpoints that still had lora_dropout_mask

            unexpected_keys = incompatible_keys.unexpected_keys
            for i in range(len(unexpected_keys) - 1, -1, -1):
                if unexpected_keys[i].endswith("lora_dropout_mask"):
                    key = unexpected_keys.pop(i)
                    print(f"[load_state_dict_post_hook] Ignored unexpected LoRA key: {key}")

        self.model.register_load_state_dict_pre_hook(patch_state_dict)
        self.model.register_load_state_dict_post_hook(ignore_lora_keys)

        # Subsequent calls to configure_model() will be ignored.
        self.model_configured = True

    def training_step(self, batch, batch_idx):
        step_time_start = perf_counter()

        batch: BatchedExample = self.data_shim(batch)

        b, n_src, _, h, w = batch["context"]["image"].shape
        _, n_tgt, _, _, _ = batch["target"]["image"].shape
        device = batch["target"]["image"].device

        n_tkn_per_view = self.model.get_num_tkn_per_view(h, w)

        # Get the right mask
        attn_mask = self._get_mask(
            num_src_views=n_src, num_tgt_views=n_tgt, num_tkn_per_view=n_tkn_per_view, device=device
        )

        # Run the model.
        decode_latents = self.train_cfg.decode_vae_latents or self.val_generated_vis
        vis_pred, latent_pred = self(
            batch["context"]["image"],
            batch["context"]["plucker_rays"],
            batch["target"]["plucker_rays"],
            attn_mask,
            vae_latents=batch["context"].get("vae_latents", None),
            decode_latents=decode_latents,
            decoder_requires_grad=self.vae_decoder_requires_grad,
        )

        assert vis_pred is not None or not decode_latents, "vis_pred cannot be None if latents are decoded."
        assert latent_pred is not None or self.model.vae is None, "latent_pred cannot be None if vae is used."

        # Output can be None if we don't want to decode the latents.
        # In this case, we don't compute the PSNR metric.
        if vis_pred is not None:
            vis_gt = batch["target"]["image"]

            # Compute metrics.
            psnr_probabilistic = compute_psnr(
                rearrange(vis_gt, "b v c h w -> (b v) c h w"),
                rearrange(vis_pred, "b v c h w -> (b v) c h w"),
            )
            self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        total_loss = self._calc_and_log_loss(vis_pred, latent_pred, batch, "train")

        # Log the time taken for the step.
        step_time = perf_counter() - step_time_start

        # Following each image generated in validation, we also want to generate the corresponding output of a training sample.
        if self.global_rank == 0 and self.val_generated_vis:
            # Reset state
            self.val_generated_vis = False

            # Construct comparison image.
            comparison_rgb = hcat(
                add_label(vcat(*batch["context"]["image"][0]), "Context"),
                add_label(vcat(*batch["target"]["image"][0]), "Target (Ground Truth)"),
                add_label(vcat(*vis_pred[0]), "Target (Predicted)"),
            )
            self.logger.log_image(
                "comparison/train/rgb",
                [prep_image(add_border(comparison_rgb))],
                step=self.global_step,
                caption=[f"scene {batch['scene'][0]} | step {self.global_step}"],
            )

            directions_ctx_cm, momentum_ctx_cm = plucker_to_colormaps(batch["context"]["plucker_rays"][0])
            directions_tgt_cm, momentum_tgt_cm = plucker_to_colormaps(batch["target"]["plucker_rays"][0])

            comparison_plucker = hcat(
                add_label(vcat(*directions_ctx_cm), "Context (Direction)"),
                add_label(vcat(*momentum_ctx_cm), "Context (Momentum)"),
                add_label(vcat(*directions_tgt_cm), "Target (Direction)"),
                add_label(vcat(*momentum_tgt_cm), "Target (Momentum)"),
            )
            self.logger.log_image(
                "comparison/train/rays",
                [prep_image(add_border(comparison_plucker))],
                step=self.global_step,
                caption=[f"scene {batch['scene'][0]} | step {self.global_step}"],
            )

        if self.global_rank == 0 and self.global_step % self.train_cfg.print_log_every_n_steps == 0:
            print(
                Fore.RED
                + f"train"
                + Fore.RESET
                + f" | {self.global_step/self.trainer.max_steps*100:>6.2f}% [ep {self.current_epoch} | step {self.global_step}]"
                f" | time = {step_time:.6f}s"
                f" | loss = {total_loss:.6f}"
                f" | num_batch = {b}"
                f" | num_src = {n_src}"
                f" | num_tgt = {n_tgt}"
                # f" | scene = {[x[:6] for x in batch['scene']]}"
                # f" | bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                # f"{batch['context']['far'].detach().cpu().numpy().mean()}]"
                # f" | context = {batch['context']['index'].tolist()}"
                # f" | target = {batch['target']['index'].tolist()}"
            )
        # self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        # self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        b, n_src, _, h, w = batch["context"]["image"].shape
        _, n_tgt, _, _, _ = batch["target"]["image"].shape
        device = batch["target"]["image"].device

        n_tkn_per_view = self.model.get_num_tkn_per_view(h, w)

        assert b == 1

        # Split along the batch dimension into chunks of size `chunk_size`
        # target_chunks = torch.split(batch["target"]["plucker_rays"], split_size_or_sections=90, dim=1)

        # predicted_chunks = []
        # for chunk in target_chunks:
        #    n_chunk_tgt = chunk.shape[1]

        #    attn_mask = self._get_mask(
        #        num_src_views=n_src, num_tgt_views=n_chunk_tgt, num_tkn_per_view=n_tkn_per_view, device=device
        #    )

        #    vis_pred, _ = self.model(
        #        batch["context"]["image"],
        #        batch["context"]["plucker_rays"],
        #        chunk,
        #        attn_mask,
        #        vae_latents=batch["context"].get("vae_latents", None),
        #    )

        #    predicted_chunks.append(vis_pred)

        # vis_pred = torch.cat(predicted_chunks, dim=1)

        # Get the right mask
        attn_mask = self._get_mask(
            num_src_views=n_src, num_tgt_views=n_tgt, num_tkn_per_view=n_tkn_per_view, device=device
        )

        # Run the model.
        with self.benchmarker.time("model"):
            vis_pred, _ = self.model(
                batch["context"]["image"],
                batch["context"]["plucker_rays"],
                batch["target"]["plucker_rays"],
                attn_mask,
                vae_latents=batch["context"].get("vae_latents", None),
            )

        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        rgb_pred = vis_pred[0]
        rgb_gt = batch["target"]["image"][0]

        # Save images.
        if self.test_cfg.save_image:
            for index, color in zip(batch["context"]["index"][0], batch["context"]["image"][0]):
                save_image(color, path / scene / f"color/src_{index:0>6}.png")

            for index, color in zip(batch["target"]["index"][0], rgb_pred):
                save_image(color, path / scene / f"color/tgt_{index:0>6}.png")

            for index, color in zip(batch["target"]["index"][0], batch["target"]["image"][0]):
                save_image(color, path / scene / f"color/gt_{index:0>6}.png")

        # save video
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in rgb_pred],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
                ffmpeg_path=self.test_cfg.ffmpeg_path,
            )

        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["model"] += 1

            if f"psnr" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr"] = []
            if f"ssim" not in self.test_step_outputs:
                self.test_step_outputs[f"ssim"] = []
            if f"lpips" not in self.test_step_outputs:
                self.test_step_outputs[f"lpips"] = []

            self.test_step_outputs[f"psnr"].append(compute_psnr(rgb_gt, rgb_pred).mean().item())
            self.test_step_outputs[f"ssim"].append(compute_ssim(rgb_gt, rgb_pred).mean().item())
            self.test_step_outputs[f"lpips"].append(compute_lpips(rgb_gt, rgb_pred).mean().item())

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

        b, n_src, _, h, w = batch["context"]["image"].shape
        _, n_tgt, _, _, _ = batch["target"]["image"].shape
        device = batch["target"]["image"].device

        n_tkn_per_view = self.model.get_num_tkn_per_view(h, w)

        assert b == 1

        # Get the right mask
        attn_mask = self._get_mask(
            num_src_views=n_src, num_tgt_views=n_tgt, num_tkn_per_view=n_tkn_per_view, device=device
        )

        # Run the model.
        vis_pred, latent_pred = self.model(
            batch["context"]["image"],
            batch["context"]["plucker_rays"],
            batch["target"]["plucker_rays"],
            attn_mask,
            vae_latents=batch["context"].get("vae_latents", None),
        )

        assert vis_pred is not None, "vis_pred cannot be None."
        assert latent_pred is not None or self.model.vae is None, "latent_pred cannot be None if vae is used."

        self._calc_and_log_loss(vis_pred, latent_pred, batch, "val")

        # Compute validation metrics.
        rgb_pred = vis_pred[0]
        rgb_gt = batch["target"]["image"][0]
        for tag, rgb in zip(("val",), (rgb_pred,)):
            psnr = compute_psnr(rgb_gt, rgb).mean()
            self.log(f"val/psnr_{tag}", psnr)
            lpips = compute_lpips(rgb_gt, rgb).mean()
            self.log(f"val/lpips_{tag}", lpips)
            ssim = compute_ssim(rgb_gt, rgb).mean()
            self.log(f"val/ssim_{tag}", ssim)

        # Don't upload images and videos in every validation step.
        if self.num_validations % self.train_cfg.vis_every_n_validations == 0:
            # Notify train_step() that visuals have been generated
            self.val_generated_vis = True

            # Construct comparison image.
            comparison = hcat(
                add_label(vcat(*batch["context"]["image"][0]), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Predicted)"),
            )

            self.logger.log_image(
                "comparison/val/rgb",
                [prep_image(add_border(comparison))],
                step=self.global_step,
                caption=[f"scene {batch['scene'][0]} | step {self.global_step}"],
            )

            directions_ctx_cm, momentum_ctx_cm = plucker_to_colormaps(batch["context"]["plucker_rays"][0])
            directions_tgt_cm, momentum_tgt_cm = plucker_to_colormaps(batch["target"]["plucker_rays"][0])

            comparison_plucker = hcat(
                add_label(vcat(*directions_ctx_cm), "Context (Direction)"),
                add_label(vcat(*momentum_ctx_cm), "Context (Momentum)"),
                add_label(vcat(*directions_tgt_cm), "Target (Direction)"),
                add_label(vcat(*momentum_tgt_cm), "Target (Momentum)"),
            )
            self.logger.log_image(
                "comparison/val/rays",
                [prep_image(add_border(comparison_plucker))],
                step=self.global_step,
                caption=[f"scene {batch['scene'][0]} | step {self.global_step}"],
            )

            # Visualize scene.
            if self.train_cfg.extended_visualization:
                images = torch.stack((*batch["context"]["image"][0], *batch["target"]["image"][0]), dim=0)
                intrinsics = torch.stack((*batch["context"]["intrinsics"][0], *batch["target"]["intrinsics"][0]), dim=0)
                extrinsics = torch.stack((*batch["context"]["extrinsics"][0], *batch["target"]["extrinsics"][0]), dim=0)
                with torch.amp.autocast("cuda" if tensor_on_gpu(images) else "cpu", enabled=False):
                    fig = visualize_scene(images, extrinsics, intrinsics, generate_gif=False)

                html_str = pio.to_html(fig, auto_play=False)
                html = wandb.Html(html_str)
                self.logger.log_table("scene/val", columns=["scene_html"], data=[[html]], step=self.global_step)

                # Rendering gif takes up too much time in validation
                # self.logger.log_video(
                #    "scene_rendered",
                #    [gif_bytes],
                #    step=self.global_step,
                #    caption=batch["scene"]
                # )

            # Run video validation step.
            self._render_video_interpolation(batch)
            self._render_video_wobble(batch)
            if self.train_cfg.extended_visualization:
                self._render_video_interpolation_exaggerated(batch)

    @rank_zero_only
    def on_validation_epoch_end(self):
        self.num_validations += 1

    @rank_zero_only
    def _render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, -1, :3, 3]
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

        return self._render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def _render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (batch["context"]["extrinsics"][0, -1]),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (batch["context"]["intrinsics"][0, -1]),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self._render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def _render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, -1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (batch["context"]["extrinsics"][0, -1]),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (batch["context"]["intrinsics"][0, -1]),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self._render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def _render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        b, n_src, _, h, w = batch["context"]["image"].shape
        _, n_tgt, _, _, _ = batch["target"]["image"].shape
        device = batch["target"]["image"].device

        assert b == 1, "For now only a batch size of 1 is supported."
        assert (
            num_frames % n_tgt == 0
        ), "For now we need to have the number of frames being divisible by the number of target views."

        with torch.amp.autocast("cuda" if tensor_on_gpu(batch["context"]["image"]) else "cpu", enabled=False):
            t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
            if smooth:
                t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

            extrinsics, intrinsics = trajectory_fn(t)
            plucker_rays = calculate_plucker_rays(
                img_height=h, img_width=w, extrinsics=extrinsics, intrinsics=intrinsics
            )

        n_tkn_per_view = self.model.get_num_tkn_per_view(h, w)

        # Get the right mask
        attn_mask = self._get_mask(
            num_src_views=n_src, num_tgt_views=n_tgt, num_tkn_per_view=n_tkn_per_view, device=device
        )

        num_frame_batches = num_frames // n_tgt
        plucker_rays_batched = rearrange(
            plucker_rays, "b (nf ntgt) p h w -> b nf ntgt p h w", nf=num_frame_batches, ntgt=n_tgt
        )

        outputs = []
        for i in range(num_frame_batches):
            # Run the model batch-wise.
            vis_pred, _ = self.model(
                batch["context"]["image"],
                batch["context"]["plucker_rays"],
                plucker_rays_batched[:, i],
                attn_mask,
                vae_latents=batch["context"].get("vae_latents", None),
            )
            outputs.append(vis_pred)

        outputs = torch.cat(outputs, dim=1)

        directions_cm, momentum_cm = plucker_to_colormaps(plucker_rays)

        images = [
            add_border(
                hcat(
                    add_label(output, "Output"),
                    add_label(directions, "Plucker Directions"),
                    add_label(momentum, "Plucker Momentum"),
                )
            )
            for output, directions, momentum in zip(outputs[0], directions_cm[0], momentum_cm[0])
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()

        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]

        self.logger.log_video(
            f"video/{name}",
            [video],
            step=self.global_step,
            caption=[f"scene {batch['scene'][0]} | step {self.global_step}"],
            fps=[30],
            format=["mp4"],
        )

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

        whitelist_weight_modules = (nn.Linear, nn.Conv2d)
        blacklist_weight_modules = (nn.LayerNorm, LayerNorm, LayerNorm2d, nn.GroupNorm, QKNorm, QKScaleUp, nn.Embedding)

        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = f"{module_name}.{param_name}" if module_name else param_name

                # Handle parametrizations
                if param_name.startswith("parametrizations"):
                    param_type = param_name.split(".")[1]
                else:
                    param_type = param_name

                if param_type.endswith("bias"):
                    no_decay.add(full_param_name)
                elif param_type == "weight" and isinstance(module, whitelist_weight_modules):
                    decay.add(full_param_name)
                elif param_type == "weight" and isinstance(module, blacklist_weight_modules):
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

    def _calc_and_log_loss(self, vis_pred, latent_pred, batch, stage):
        assert vis_pred is not None or (
            vis_pred is None and len(self.vis_losses) == 0
        ), "vis_pred cannot be None if losses on it are used."
        assert latent_pred is not None or (
            latent_pred is None and len(self.latent_losses) == 0
        ), "latent_pred cannot be None if losses on it are used."

        vis_gt = batch["target"]["image"]

        total_vis_loss = 0
        total_latent_loss = 0

        if vis_pred is not None:
            vis_pred_typed = BatchedViewsRGBD({"color": vis_pred, "depth": None})
            vis_gt_typed = BatchedViewsRGBD({"color": vis_gt, "depth": None})

            # Compute and log loss.
            for loss_fn in self.vis_losses:
                loss = loss_fn(vis_pred_typed, vis_gt_typed, batch, self.global_step)
                self.log(f"loss/{stage}/vis_{loss_fn.name}", loss)
                total_vis_loss = total_vis_loss + loss
            self.log(f"loss/{stage}/vis_total", total_vis_loss)

        # If latent space is used, compute metrics for the latent space.
        if latent_pred is not None:
            vae_latents = batch["target"].get("vae_latents", False)
            if vae_latents:
                latent_gt = self.model.vae.sample(vae_latents["mean"], vae_latents["std"])
            else:
                latent_gt = self.model.vae.encode(vis_gt)

            latent_pred_typed = BatchedViewsRGBD({"color": latent_pred, "depth": None})
            latent_gt_typed = BatchedViewsRGBD({"color": latent_gt, "depth": None})

            for loss_fn in self.latent_losses:
                loss = loss_fn(latent_pred_typed, latent_gt_typed, batch, self.global_step)
                self.log(f"loss/{stage}/latent_{loss_fn.name}", loss)
                total_latent_loss = total_latent_loss + loss
            self.log(f"loss/{stage}/latent_total", total_latent_loss)

        # Add the latent loss to the total loss.
        total_loss = (
            self.loss_cfg.vis_loss_weight * total_vis_loss + self.loss_cfg.latent_loss_weight * total_latent_loss
        )
        self.log(f"loss/{stage}/total", total_loss)

        return total_loss

    def _get_mask(self, num_src_views: int, num_tgt_views: int, num_tkn_per_view: int, device: torch.device | str):
        use_flex_attn = self.sdpa_kernel == "flex-attention" or (
            self.sdpa_kernel == "auto"
            and torch.cuda.is_available()
            and parse_version(torch.__version__) >= parse_version("2.5.0")
        )

        if use_flex_attn:
            return ModelWrapper._get_block_mask(num_src_views, num_tgt_views, num_tkn_per_view, device)
        else:
            return ModelWrapper._get_full_mask(num_src_views, num_tgt_views, num_tkn_per_view, device)

    @lru_cache
    @staticmethod
    def _get_block_mask(num_src_views: int, num_tgt_views: int, num_tkn_per_view: int, device: torch.device | str):
        num_src_tkn = num_src_views * num_tkn_per_view

        total_views = num_src_views + num_tgt_views
        num_tkn = total_views * num_tkn_per_view

        # All context views get the same id.
        src_view_ids = torch.zeros(num_src_tkn, device=device)

        # For the target views, each view gets its own id.
        tgt_view_ids = torch.arange(1, num_tgt_views + 1, device=device).repeat_interleave(num_tkn_per_view)

        view_ids = torch.cat([src_view_ids, tgt_view_ids])

        def document_mask(b, h, q_idx, kv_idx):
            # Allow attention within views.
            # Remember context views all have the same id, allowing attention between context views.
            attend_within_views = view_ids[q_idx] == view_ids[kv_idx]
            # Allow attention from target views to context views.
            attend_tgt_to_src = view_ids[kv_idx] == 0

            return attend_within_views | attend_tgt_to_src

        # Compute the floor of the log2 of num_tkn_per_view to find the lower bound power of 2
        log2_floor = np.floor(np.log2(num_tkn_per_view))

        # Calculate the sqrt of the overall number, but in the exponent.
        # Max with 2^7=128, as this is the minimum multiple currently needed in flex-attention.
        max_value = max(log2_floor // 2, 7)

        # Calculate the final block size.
        block_size = 1 << int(max_value)

        block_mask = create_block_mask(document_mask, None, None, num_tkn, num_tkn, device, BLOCK_SIZE=block_size)

        print(f"Built flex-attention block mask:\n{block_mask.to_string(grid_size=num_src_views+num_tgt_views)}")

        return block_mask

    @lru_cache
    @staticmethod
    def _get_full_mask(num_src_views: int, num_tgt_views: int, num_tkn_per_view: int, device: torch.device | str):
        seq_len = (num_src_views + num_tgt_views) * num_tkn_per_view
        num_src_tkn = num_src_views * num_tkn_per_view

        ## Create attention mask
        # Step 1: Initialize the final attention mask with zeros
        # Create an attention mask of shape [(total number of tokens), (total number of tokens)].
        final_attn_mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

        # Step 2: Fill in intra-target attention (block diagonal for each target view)
        # Target tokens can fully attend to tokens within the same target view but not to tokens in other target views.
        ones_block = torch.ones((num_tkn_per_view, num_tkn_per_view), dtype=torch.bool, device=device)
        intra_target_mask = torch.block_diag(*[ones_block for _ in range(num_tgt_views)])
        final_attn_mask[num_src_tkn:, num_src_tkn:] = intra_target_mask

        # Step 3: Fill in target-to-source and source-to-source attention.
        # Target tokens attend to all source tokens for contextual information.
        final_attn_mask[:, :num_src_tkn] = True

        # Example with 1 source view (2 tokens) and 2 target views (each with 2 tokens):
        # [ 1 1 | 0 0 0 0 ]  <- Source tokens do only attend to themselves.
        # [ 1 1 | 0 0 0 0 ]
        # -------------------
        # [ 1 1 | 1 1 0 0 ]  <- Target tokens from view 1 attend to themselves and source tokens.
        # [ 1 1 | 1 1 0 0 ]
        # -------------------
        # [ 1 1 | 0 0 1 1 ]  <- Target tokens from view 2 attend to themselves and source tokens.
        # [ 1 1 | 0 0 1 1 ]
        #
        # - The upper left block (zero_mask) indicates that source tokens do not attend to any tokens.
        # - The lower left blocks (tgt_src_attn_mask) indicate that target tokens attend to all source tokens.
        # - The lower right blocks (tgt_attn_mask) represent intra-target attention, where tokens in the same group can attend to each other.
        return final_attn_mask

    def forward(self, *args, **kwargs):
        """
        NOTE: This is a workaround to allow for registered forward-hooks to be called when
        watching the parameters of this module.

        Only use this function during training_step(), to log parameters during training.
        """
        return self.model(*args, **kwargs)
