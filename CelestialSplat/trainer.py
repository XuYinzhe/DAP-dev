"""
Stage-1 Trainer for MonoCelestialSplat.
Single-GPU training with adjacent-frame novel-view consistency.

Training flow per pair (I_t, D_t, P_t) and (I_t+1, D_t+1, P_t+1):
  1. Estimate Gaussians G_t and G_t+1 from respective frames.
  2. Render G_t in its input view -> reconstruction + constraint losses.
  3. Render G_t+1 in its input view -> reconstruction + constraint losses.
  4. Transform G_t to P_t+1 camera frame, render novel view -> loss vs I_t+1, D_t+1.
  5. Transform G_t+1 to P_t camera frame, render novel view -> loss vs I_t, D_t.
  6. Backprop total loss.

Pose convention: camera-to-world (c2w), matching load_tartanair_poses().
Novel-view relative transform: T = P_tgt^{-1} @ P_src.
"""

import os
import sys
import math
import time
import argparse
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CelestialSplat.mono_model import (
    MonoCelestialSplat,
    MonoCelestialSplatConfig,
    OmniGaussianRender,
    Gaussians3D,
)
from CelestialSplat.loss import CelestialSplatLoss, LossConfig
from CelestialSplat.mono_dataset import TartanAir360MonoDataset
from CelestialSplat.utils.visualization import LossTracker


# ------------------------------------------------------------------------------
# Quaternion / rotation helpers
# ------------------------------------------------------------------------------

def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion (w, x, y, z). Supports [3,3] and [B,3,3]."""
    if R.dim() == 2:
        R = R.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    batch = R.shape[0]
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = torch.zeros(batch, 4, device=R.device, dtype=R.dtype)

    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2
        q[mask1, 0] = 0.25 * s
        q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
        q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
        q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s

    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
        q[mask2, 1] = 0.25 * s
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s

    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
        q[mask3, 2] = 0.25 * s
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s

    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
        q[mask4, 3] = 0.25 * s

    q = F.normalize(q, dim=-1)
    if squeeze:
        q = q.squeeze(0)
    return q


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product. Both (w, x, y, z). Broadcasts over last dim."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def transform_gaussians(gaussians: Gaussians3D, T: torch.Tensor) -> Gaussians3D:
    """
    Rigidly transform Gaussians from source camera frame to target camera frame.

    Args:
        gaussians: Gaussians3D with fields of shape [B, P, C].
        T: [B, 4, 4] or [4, 4] rigid transform matrix (source -> target).
    Returns:
        Transformed Gaussians3D in target camera frame.
    """
    R = T[..., :3, :3]   # [B,3,3] or [3,3]
    t = T[..., :3, 3]    # [B,3] or [3]

    mean_vectors = gaussians.mean_vectors  # [B, P, 3]
    if R.dim() == 2:
        new_means = torch.matmul(mean_vectors, R.T) + t
    else:
        new_means = torch.matmul(mean_vectors, R.transpose(-2, -1)) + t.unsqueeze(-2)

    # Recompute metric depths (norm of positions)
    new_depths = torch.norm(new_means, dim=-1, keepdim=True).clamp(min=1e-2)

    # Recompute projected ERP coords from new positions
    x, y, z = new_means[..., 0], new_means[..., 1], new_means[..., 2]
    lon = torch.atan2(x, z)
    lat = torch.atan2(-y, torch.sqrt(x ** 2 + z ** 2))
    lon_ndc = lon / math.pi
    lat_ndc = lat / (math.pi / 2.0)
    new_projected_coords = torch.stack([lon_ndc, -lat_ndc], dim=-1)  # [B, P, 2]

    # Transform base_positions
    new_base_positions = None
    if gaussians.base_positions is not None:
        if R.dim() == 2:
            new_base_positions = torch.matmul(gaussians.base_positions, R.T) + t
        else:
            new_base_positions = torch.matmul(
                gaussians.base_positions, R.transpose(-2, -1)
            ) + t.unsqueeze(-2)

    # Rotate quaternions
    q_rot = rotation_matrix_to_quaternion(R)  # [B,4] or [4]
    if q_rot.dim() == 1:
        q_rot = q_rot.unsqueeze(0).unsqueeze(1)  # [1,1,4]
    else:
        q_rot = q_rot.unsqueeze(1)  # [B,1,4]
    q_rot = q_rot.expand_as(gaussians.quaternions)
    new_quaternions = quaternion_multiply(q_rot, gaussians.quaternions)

    return Gaussians3D(
        mean_vectors=new_means,
        singular_values=gaussians.singular_values,
        quaternions=new_quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
        base_positions=new_base_positions,
        projected_coords=new_projected_coords,
        depths=new_depths,
    )


# ------------------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------------------

class Stage1Trainer:
    """
    Stage-1 trainer: supervised learning on synthetic data.
    Adjacent-frame novel-view consistency + input-view constraints.
    """

    def __init__(
        self,
        model_config: MonoCelestialSplatConfig,
        loss_config: LossConfig,
        device: str = "cuda:0",
    ):
        self.device = torch.device(device)
        self.model = MonoCelestialSplat(model_config).to(self.device)
        self.rasterizer = OmniGaussianRender(model_config).to(self.device)
        self.criterion = CelestialSplatLoss(loss_config).to(self.device)
        self.loss_tracker = LossTracker()

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.global_step = 0

    def build_optimizer(self, lr: float = 1.6e-4) -> None:
        """Adam optimizer with selective DAP unfreezing.

        Policy:
        - Freeze DINOv3Adapter (pretrained feature extractor): domain gap makes
          finetuning risky and memory-heavy.
        - Freeze mask_head: TartanAir's unrectified horizon causes mask_head to
          predict all-sky; finetuning it is not productive.
        - depth_head: initially frozen for Phase-1 stability; may be unfrozen
          later via unfreeze_depth_head() for Phase-2 joint fine-tuning.
        - All parameters are registered in the optimizer; gradients are controlled
          by requires_grad. This allows dynamic unfreezing without re-creating
          the optimizer.
        """
        trainable_params = []
        for name, param in self.model.named_parameters():
            if 'dap_core.pretrained' in name:
                param.requires_grad = False
            elif 'dap_core.mask_head' in name:
                param.requires_grad = False
            elif 'dap_core.depth_head' in name:
                param.requires_grad = False  # Phase 1: frozen
            else:
                param.requires_grad = True
                trainable_params.append(param)
        self.optimizer = torch.optim.Adam(
            trainable_params,
            lr=lr,
            betas=(0.9, 0.999),
        )

    def unfreeze_depth_head(self, lr_scale: float = 0.1) -> None:
        """Unfreeze depth_head for Phase-2 joint fine-tuning.

        Adds a new parameter group with scaled LR so depth_head does not
        disrupt the already-stable GS parameters.
        """
        depth_params = []
        for name, param in self.model.named_parameters():
            if 'dap_core.depth_head' in name and not param.requires_grad:
                param.requires_grad = True
                depth_params.append(param)
        if depth_params:
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.optimizer.add_param_group({
                "params": depth_params,
                "lr": current_lr * lr_scale,
                "betas": (0.9, 0.999),
            })
            print(f"[Step {self.global_step}] Unfroze depth_head with lr={current_lr * lr_scale:.2e}")

    def build_scheduler(self, num_iterations: int, warmup_iterations: int = 100) -> None:
        """Cosine decay with linear warmup."""
        if self.optimizer is None:
            raise RuntimeError("Call build_optimizer() first")

        def lr_lambda(step: int) -> float:
            if step < warmup_iterations:
                return step / max(1, warmup_iterations)
            progress = (step - warmup_iterations) / max(1, num_iterations - warmup_iterations)
            # Decay from 1.0 down to 0.1 (i.e. lr final = 0.1 * initial = 1.6e-5 when initial=1.6e-4)
            return 0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def compute_render_loss(
        self,
        gaussians: Gaussians3D,
        target_image: torch.Tensor,
        target_depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pred_depth: Optional[torch.Tensor] = None,
        is_novel_view: bool = False,
        nonsky_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Rasterize Gaussians and compute composite loss.
        Gaussian must already be expressed in the target camera frame.
        """
        rendered_img, rendered_depth, rendered_alpha = self.rasterizer(gaussians)

        outputs = {
            "image": rendered_img,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "positions": gaussians.mean_vectors,
            "base_positions": gaussians.base_positions,
            "projected_coords": gaussians.projected_coords,
            "opacities": gaussians.opacities,
            "depths": gaussians.depths,
            "scales": gaussians.singular_values,
            "mask": mask,
        }
        targets = {
            "image": target_image,
            "depth": target_depth,
        }
        total_loss, loss_dict = self.criterion(
            outputs, targets,
            pred_depth=pred_depth,
            is_novel_view=is_novel_view,
            nonsky_mask=nonsky_mask,
        )
        return total_loss, loss_dict

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step on one adjacent-frame pair."""
        image_0 = batch["image_0"].to(self.device)
        depth_0 = batch["depth_0"].to(self.device)
        mask_0 = batch["mask_0"].to(self.device)
        pose_0 = batch["pose_0"].to(self.device)
        image_1 = batch["image_1"].to(self.device)
        depth_1 = batch["depth_1"].to(self.device)
        mask_1 = batch["mask_1"].to(self.device)
        pose_1 = batch["pose_1"].to(self.device)

        # ---- Forward: estimate Gaussians for both frames ----
        gaussians_0, extras_0 = self.model(image_0, gt_depth=depth_0, return_extras=True)
        gaussians_1, extras_1 = self.model(image_1, gt_depth=depth_1, return_extras=True)

        # Dataset already provides reliable GT-based mask (depth > 0 & depth < threshold).
        # We ignore DAP's predicted nonsky_mask because TartanAir's unrectified domain gap
        # causes mask_head to predict all-sky (all False) on most frames.

        # ---- Input-view losses (reconstruction + constraints) ----
        # ml-sharp: L_depth on Layer-1, L_tv on Layer-2, both only on input view.
        # Perceptual loss is disabled on input views (paper: only on novel views).
        loss_0_in, dict_0_in = self.compute_render_loss(
            gaussians_0, image_0, depth_0, mask=mask_0,
            pred_depth=extras_0["aligned_depth"],
            is_novel_view=False,
            nonsky_mask=extras_0["nonsky_mask"],
        )
        loss_1_in, dict_1_in = self.compute_render_loss(
            gaussians_1, image_1, depth_1, mask=mask_1,
            pred_depth=extras_1["aligned_depth"],
            is_novel_view=False,
            nonsky_mask=extras_1["nonsky_mask"],
        )

        # ---- Novel view: t -> t+1 ----
        # Novel view only computes reconstruction losses (color, perceptual, alpha).
        # L_depth and L_tv are NOT applied to novel views per ml-sharp paper.
        T_0_to_1 = torch.matmul(torch.inverse(pose_1), pose_0)
        g_0_in_1 = transform_gaussians(gaussians_0, T_0_to_1)
        loss_0_nov, dict_0_nov = self.compute_render_loss(
            g_0_in_1, image_1, depth_1, mask=mask_1,
            is_novel_view=True,
        )

        # ---- Novel view: t+1 -> t ----
        T_1_to_0 = torch.matmul(torch.inverse(pose_0), pose_1)
        g_1_in_0 = transform_gaussians(gaussians_1, T_1_to_0)
        loss_1_nov, dict_1_nov = self.compute_render_loss(
            g_1_in_0, image_0, depth_0, mask=mask_0,
            is_novel_view=True,
        )

        # ---- Total loss ----
        total_loss = loss_0_in + loss_1_in + loss_0_nov + loss_1_nov

        log_dict = {
            "total": total_loss.item() if torch.isfinite(total_loss) else float('nan'),
            "in_0": loss_0_in.item() if torch.isfinite(loss_0_in) else float('nan'),
            "in_1": loss_1_in.item() if torch.isfinite(loss_1_in) else float('nan'),
            "nov_0->1": loss_0_nov.item() if torch.isfinite(loss_0_nov) else float('nan'),
            "nov_1->0": loss_1_nov.item() if torch.isfinite(loss_1_nov) else float('nan'),
        }
        for k, v in dict_0_in.items():
            if k != "total":
                log_dict[f"in0_{k}"] = v.item() if torch.isfinite(v) else float('nan')

        # Safety: skip backward if loss is NaN/Inf (common when depth_head is
        # freshly unfrozen and predicts near-zero depths occasionally).
        if not torch.isfinite(total_loss):
            print(f"[WARN] Step {self.global_step}: non-finite loss ({total_loss.item()}), skipping backward.")
            self.global_step += 1
            return log_dict

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1
        return log_dict

    @torch.no_grad()
    def validate(self, dataloader: DataLoader, num_steps: int = 10) -> Dict[str, float]:
        """Quick validation loop."""
        self.model.eval()
        data_iter = iter(dataloader)
        totals = {}
        for _ in range(num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            log_dict = self._eval_step(batch)
            for k, v in log_dict.items():
                totals[k] = totals.get(k, 0.0) + v
        self.model.train()
        return {k: v / max(num_steps, 1) for k, v in totals.items()}

    def _eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluation step (no backward)."""
        image_0 = batch["image_0"]
        depth_0 = batch["depth_0"]
        mask_0 = batch["mask_0"]
        pose_0 = batch["pose_0"]
        image_1 = batch["image_1"]
        depth_1 = batch["depth_1"]
        mask_1 = batch["mask_1"]
        pose_1 = batch["pose_1"]

        gaussians_0, _ = self.model(image_0, gt_depth=depth_0, return_extras=True)
        gaussians_1, _ = self.model(image_1, gt_depth=depth_1, return_extras=True)

        loss_0_in, _ = self.compute_render_loss(gaussians_0, image_0, depth_0, mask_0)
        loss_1_in, _ = self.compute_render_loss(gaussians_1, image_1, depth_1, mask_1)

        T_0_to_1 = torch.matmul(torch.inverse(pose_1), pose_0)
        g_0_in_1 = transform_gaussians(gaussians_0, T_0_to_1)
        loss_0_nov, _ = self.compute_render_loss(g_0_in_1, image_1, depth_1, mask_1)

        T_1_to_0 = torch.matmul(torch.inverse(pose_0), pose_1)
        g_1_in_0 = transform_gaussians(gaussians_1, T_1_to_0)
        loss_1_nov, _ = self.compute_render_loss(g_1_in_0, image_0, depth_0, mask_0)

        total_loss = loss_0_in + loss_1_in + loss_0_nov + loss_1_nov
        return {
            "total": total_loss.item(),
            "in_0": loss_0_in.item(),
            "in_1": loss_1_in.item(),
            "nov_0->1": loss_0_nov.item(),
            "nov_1->0": loss_1_nov.item(),
        }

    @torch.no_grad()
    def visualize_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        out_dir: str = "outputs/visualizations",
    ) -> None:
        """Generate and save visualization for current step."""
        self.model.eval()
        image_0 = batch["image_0"].to(self.device)
        depth_0 = batch["depth_0"].to(self.device)
        pose_0 = batch["pose_0"].to(self.device)
        image_1 = batch["image_1"].to(self.device)
        depth_1 = batch["depth_1"].to(self.device)
        pose_1 = batch["pose_1"].to(self.device)

        gaussians_0, extras_0 = self.model(image_0, gt_depth=depth_0, return_extras=True)
        rendered_rgb_0, rendered_depth_0, rendered_alpha_0 = self.rasterizer(gaussians_0)

        T_0_to_1 = torch.matmul(torch.inverse(pose_1), pose_0)
        g_0_in_1 = transform_gaussians(gaussians_0, T_0_to_1)
        rendered_rgb_1_nov, rendered_depth_1_nov, rendered_alpha_1_nov = self.rasterizer(g_0_in_1)

        from CelestialSplat.utils.visualization import save_training_visualization
        saved_path = save_training_visualization(
            out_dir=out_dir,
            step=step,
            gt_rgb_0=image_0,
            rendered_rgb_0=rendered_rgb_0,
            gt_depth_0=depth_0,
            rendered_depth_0=rendered_depth_0,
            rendered_alpha_0=rendered_alpha_0,
            pred_depth_0=extras_0["pred_depth"],
            pred_mask_0=extras_0.get("pred_mask"),
            nonsky_mask_0=extras_0["nonsky_mask"],
            gt_rgb_1=image_1,
            rendered_rgb_1_nov=rendered_rgb_1_nov,
            rendered_depth_1_nov=rendered_depth_1_nov,
            rendered_alpha_1_nov=rendered_alpha_1_nov,
        )
        print(f"  [vis] Saved to {saved_path}")
        self.model.train()

    def train(
        self,
        dataloader: DataLoader,
        num_iterations: int,
        log_interval: int = 10,
        val_dataloader: Optional[DataLoader] = None,
        val_interval: Optional[int] = None,
        save_interval: Optional[int] = None,
        vis_dir: str = "outputs/visualizations",
        unfreeze_depth_at_step: Optional[int] = None,
    ) -> None:
        """Main training loop."""
        self.model.train()
        data_iter = iter(dataloader)

        for step in range(num_iterations):
            # Phase 2: unfreeze depth_head at specified step
            if unfreeze_depth_at_step is not None and step == unfreeze_depth_at_step:
                self.unfreeze_depth_head(lr_scale=0.1)

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            t0 = time.time()

            log_dict = self.train_step(batch)

            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            elapsed = time.time() - t0

            self.loss_tracker.update(step, log_dict)

            if save_interval is not None and step % save_interval == 0:
                self.visualize_step(batch, step, out_dir=vis_dir)
                self.loss_tracker.save_plot(f"{vis_dir}/loss_curves.png", smooth_window=5)

            if step % log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                mem = (
                    torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                    if torch.cuda.is_available()
                    else 0.0
                )
                msg = (
                    f"[Step {step:04d}/{num_iterations}] "
                    f"loss={log_dict['total']:.4f} "
                    f"(in={log_dict['in_0']:.3f}+{log_dict['in_1']:.3f} "
                    f"nov={log_dict['nov_0->1']:.3f}+{log_dict['nov_1->0']:.3f}) "
                    f"lr={lr:.2e} time={elapsed:.3f}s mem={mem:.2f}GB"
                )
                if val_dataloader is not None and val_interval is not None and step % val_interval == 0 and step > 0:
                    val_dict = self.validate(val_dataloader, num_steps=10)
                    msg += f" | val={val_dict['total']:.3f}"
                print(msg)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(self.device)


# ------------------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage-1 MonoCelestialSplat training")
    parser.add_argument("--data_dir", type=str, default="/homes/shaun/main/dataset/tartanair360/")
    parser.add_argument("--num_sequences", type=int, default=1, help="Scenes to use (for quick test)")
    parser.add_argument("--num_iterations", type=int, default=100, help="Total training iterations")
    parser.add_argument("--warmup_iterations", type=int, default=10, help="LR warmup steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.6e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=None, help="Validation every N steps")
    parser.add_argument("--save_interval", type=int, default=None, help="Save visualization every N steps")
    parser.add_argument("--vis_dir", type=str, default="outputs/visualizations", help="Visualization output directory")
    parser.add_argument("--image_height", type=int, default=512, help="Input height (512 needs ~22GB, 256 needs ~9GB)")
    parser.add_argument("--image_width", type=int, default=1024, help="Input width")
    parser.add_argument("--rectify", type=int, default=1, help="1=enable horizon rectification, 0=disable")
    parser.add_argument("--depth_threshold", type=float, default=100.0, help="Max valid depth in meters (sky is ~4188m)")
    parser.add_argument("--frame_skip", type=int, default=1, help="Frame skip for adjacent pair (1=consecutive, 5=every 5th frame)")
    parser.add_argument("--unfreeze_depth_at_step", type=int, default=None, help="Step to unfreeze depth_head (Phase 2). None=keep frozen.")
    args = parser.parse_args()

    model_config = MonoCelestialSplatConfig(
        depth_scaler_mode="global_median",
        image_height=args.image_height,
        image_width=args.image_width,
    )
    loss_config = LossConfig(
        image_height=args.image_height,
        image_width=args.image_width,
    )

    print("Loading dataset...")
    dataset = TartanAir360MonoDataset(
        root_dir=args.data_dir,
        image_size=(args.image_height, args.image_width),
        num_sequences=args.num_sequences,
        rectify=bool(args.rectify),
        depth_threshold=args.depth_threshold,
        frame_skip=args.frame_skip,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = None
    if args.val_interval is not None:
        # Use same dataset for quick validation (no shuffle)
        val_dataset = TartanAir360MonoDataset(
            root_dir=args.data_dir,
            image_size=(args.image_height, args.image_width),
            num_sequences=args.num_sequences,
            rectify=bool(args.rectify),
            depth_threshold=args.depth_threshold,
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            pin_memory=True, drop_last=False,
        )

    print("Initializing trainer...")
    trainer = Stage1Trainer(model_config, loss_config, device=args.device)
    trainer.build_optimizer(lr=args.lr)
    trainer.build_scheduler(
        num_iterations=args.num_iterations,
        warmup_iterations=args.warmup_iterations,
    )

    print(f"Starting training: {args.num_iterations} iters, bs={args.batch_size}, lr={args.lr}, rectify={bool(args.rectify)}")
    trainer.train(
        dataloader,
        num_iterations=args.num_iterations,
        log_interval=args.log_interval,
        val_dataloader=val_dataloader,
        val_interval=args.val_interval,
        save_interval=args.save_interval,
        vis_dir=args.vis_dir,
        unfreeze_depth_at_step=args.unfreeze_depth_at_step,
    )
    print("Training completed!")


if __name__ == "__main__":
    main()
