import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from einops import rearrange
from pytorch_lightning import LightningModule
from tqdm import tqdm

from ..geometry.epipolar_lines import project_rays
from ..geometry.projection import get_world_rays, sample_image_grid
from ..misc.image_io import save_image
from ..visualization.annotation import add_label
from ..visualization.layout import add_border, hcat
from .evaluation_index_generator import EvaluationIndexGeneratorCfg, IndexEntry


class EvaluationIndexGeneratorV2(LightningModule):
    generator: torch.Generator
    cfg: EvaluationIndexGeneratorCfg
    index: dict[str, Optional[IndexEntry]]

    def __init__(self, cfg: EvaluationIndexGeneratorCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.generator = torch.Generator()
        self.generator.manual_seed(cfg.seed)
        self.index = {}

    def compute_overlap(
        self,
        idx_a: int,
        idx_b: int,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        h: int,
        w: int,
        device: torch.device,
    ) -> float:
        """
        Computes the bidirectional overlap between the views at idx_a and idx_b.
        Returns a float in [0, 1].
        """
        xy, _ = sample_image_grid((h, w), device=device)

        # Rays from A
        origins_a, dirs_a = get_world_rays(
            rearrange(xy, "h w xy -> (h w) xy"),
            extrinsics[idx_a],
            intrinsics[idx_a],
        )
        # Rays from B
        origins_b, dirs_b = get_world_rays(
            rearrange(xy, "h w xy -> (h w) xy"),
            extrinsics[idx_b],
            intrinsics[idx_b],
        )

        # Project A's rays onto B
        proj_ab = project_rays(origins_a, dirs_a, extrinsics[idx_b], intrinsics[idx_b])
        overlap_ab = proj_ab["overlaps_image"].float().mean()

        # Project B's rays onto A
        proj_ba = project_rays(origins_b, dirs_b, extrinsics[idx_a], intrinsics[idx_a])
        overlap_ba = proj_ba["overlaps_image"].float().mean()

        return float(min(overlap_ab, overlap_ba))

    def pick_context_views(
        self, extrinsics: torch.Tensor, intrinsics: torch.Tensor, h: int, w: int, device: torch.device
    ) -> Optional[list[int]]:
        """
        Try to pick a left and right context view (with indices i and j) that satisfy the following:
          - Their index difference is within [min_ctx_distance, max_ctx_distance]
          - Their mutual overlap is within [min_ctx_overlap, max_ctx_overlap]
        If more than 2 context views are needed, we fill in intermediate views from
        between i and j.
        """
        v = extrinsics.shape[0]
        valid_pair_found = False
        chosen_left, chosen_right = None, None

        left_candidates = list(range(v - 1))
        # Randomly permute left candidates.
        left_candidates = [
            left_candidates[i] for i in torch.randperm(len(left_candidates), generator=self.generator).tolist()
        ]

        for left in tqdm(left_candidates, "Finding context pair"):
            right_candidates = list(range(left + 1, v))
            # Randomly permute right candidates.
            right_candidates = [
                right_candidates[i] for i in torch.randperm(len(right_candidates), generator=self.generator).tolist()
            ]
            for right in right_candidates:
                delta = right - left
                if delta < self.cfg.min_ctx_distance or delta > self.cfg.max_ctx_distance:
                    continue
                ovlp = self.compute_overlap(left, right, extrinsics, intrinsics, h, w, device)
                if ovlp < self.cfg.min_ctx_overlap or ovlp > self.cfg.max_ctx_overlap:
                    continue
                chosen_left, chosen_right = left, right
                valid_pair_found = True
                break
            if valid_pair_found:
                break

        if not valid_pair_found:
            return None

        context = [chosen_left, chosen_right]

        # If only two context views are needed, return immediately.
        if self.cfg.num_context_views == 2:
            return sorted(context)

        # For more than two context views, randomly sample intermediate views.
        required = self.cfg.num_context_views - 2
        num_candidates = chosen_right - chosen_left - 1  # available indices between left and right
        if num_candidates < required:
            return None

        # Use torch.arange and torch.randperm to select the intermediate indices.
        chosen_intermediates = torch.arange(chosen_left + 1, chosen_right)[
            torch.randperm(num_candidates, generator=self.generator)
        ][:required].tolist()

        context = [chosen_left] + chosen_intermediates + [chosen_right]
        context.sort()

        return context

    def pick_target_views(self, context_indices: list[int], v: int) -> Optional[list[int]]:
        """
        Pick cfg.num_target_views from indices strictly between the minimum and maximum context indices,
        ensuring they are at least min_tgt_to_ctx_distance away from any context view.
        """
        lower_bound = min(context_indices)
        upper_bound = max(context_indices)
        # Candidate indices in between, excluding the context views themselves.
        candidates = [i for i in range(lower_bound + 1, upper_bound) if i not in context_indices]
        if len(candidates) < self.cfg.num_target_views:
            return None

        perm = torch.randperm(len(candidates), generator=self.generator).tolist()
        chosen_targets = []
        for i in perm:
            idx = candidates[i]
            if any(abs(idx - c) < self.cfg.min_tgt_to_ctx_distance for c in context_indices):
                continue
            chosen_targets.append(idx)
            if len(chosen_targets) == self.cfg.num_target_views:
                break

        if len(chosen_targets) < self.cfg.num_target_views:
            return None

        chosen_targets.sort()
        return chosen_targets

    def generate_preview(self, batch: dict, context: list[int], targets: list[int], scene: str) -> None:
        """
        Saves a combined preview image that shows all context views (and optionally targets)
        with overlap info relative to the *first* context view, for example.
        """
        # Create output folder.
        preview_path = self.cfg.output_path / "previews"
        preview_path.mkdir(exist_ok=True, parents=True)

        images_to_show = []
        first_ctx = context[0]

        # For each context view, label overlap with the first context.
        extrinsics = batch["target"]["extrinsics"][0]
        intrinsics = batch["target"]["intrinsics"][0]
        _, _, _, h, w = batch["target"]["image"].shape

        def get_frame_image(idx: int):
            img = batch["target"]["image"][0, idx]  # CxHxW.
            return img

        for ci in context:
            img = get_frame_image(ci)
            overlap_val = self.compute_overlap(first_ctx, ci, extrinsics, intrinsics, h, w, device=self.device) * 100.0
            label_text = f"Context {ci} (ovlp {overlap_val:.1f}%)"
            labeled = add_label(img, label_text)
            images_to_show.append(labeled)

        for ti in targets:
            img = get_frame_image(ti)
            labeled = add_label(img, f"Target {ti}")
            images_to_show.append(labeled)

        row_vis = hcat(*images_to_show)
        row_vis = add_border(row_vis, width=2, color=0)
        save_image(row_vis, preview_path / f"{scene}.png")

    def test_step(self, batch, batch_idx):
        """
        Called once per scene/batch. We attempt to generate a valid set of context/target indices.
        If none can be found, we store None.
        """
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1, "Batch size must be 1 for this generator."
        extrinsics = batch["target"]["extrinsics"][0]
        intrinsics = batch["target"]["intrinsics"][0]
        scene = batch["scene"][0]

        context_indices = self.pick_context_views(extrinsics, intrinsics, h, w, device=self.device)
        if context_indices is None:
            print(f"\nCould not find a valid context set for {scene}.")
            self.index[scene] = None
            return

        target_indices = self.pick_target_views(context_indices, v)
        if target_indices is None:
            print(f"\nCould not find valid targets for {scene}.")
            self.index[scene] = None
            return

        chosen_entry = IndexEntry(
            context=tuple(context_indices),
            target=tuple(target_indices),
        )

        if self.cfg.save_previews:
            self.generate_preview(batch, context_indices, target_indices, scene)

        self.index[scene] = chosen_entry

    def save_index(self) -> None:
        self.cfg.output_path.mkdir(exist_ok=True, parents=True)
        print(f"Found {sum(v is not None for v in self.index.values())} valid sets.")

        # Build an incremental dictionary of filtered contexts per scene.
        # Start with the minimal context: first and last.
        incr = {}
        for scene, entry in self.index.items():
            if entry is None:
                continue
            ctx = list(entry.context)  # assumed sorted
            incr[scene] = [ctx[0], ctx[-1]]

        # Write out the minimal (2-context) index.
        out_path = self.cfg.output_path / "evaluation_index_num_src_2.json"
        filtered_index = {
            scene: None if entry is None else IndexEntry(context=tuple(incr[scene]), target=entry.target)
            for scene, entry in self.index.items()
        }
        with out_path.open("w") as f:
            json.dump({k: None if v is None else asdict(v) for k, v in filtered_index.items()}, f, indent=2)

        # For each additional context view from 3 up to num_context_views,
        # add one new context view from the interior, keeping the previous ones.
        for i in range(3, self.cfg.num_context_views + 1):
            out_path = self.cfg.output_path / f"evaluation_index_num_src_{i}.json"
            filtered_index = {}
            for scene, entry in self.index.items():
                if entry is None:
                    filtered_index[scene] = None
                    continue
                full_ctx = list(entry.context)
                # Interior candidates: those between the first and last not already chosen.
                interior = [c for c in full_ctx[1:-1] if c not in incr[scene]]
                if interior:
                    # Pick one new candidate at random.
                    new_cand = interior[torch.randint(0, len(interior), (1,), generator=self.generator).item()]
                    incr[scene].append(new_cand)
                    incr[scene].sort()
                # Otherwise, keep the previous selection.
                filtered_index[scene] = IndexEntry(context=tuple(incr[scene]), target=entry.target)
            with out_path.open("w") as f:
                json.dump({k: None if v is None else asdict(v) for k, v in filtered_index.items()}, f, indent=2)
