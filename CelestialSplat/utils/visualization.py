"""Training visualization utilities for MonoCelestialSplat.

Saves rendered RGB/depth/alpha and DAP predictions as PNG grids for inspection.
"""

import numpy as np
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_numpy(t, batch_idx: int = 0):
    """Convert tensor to numpy [H, W] or [H, W, C]."""
    if t is None:
        return None
    t = t.detach().cpu()
    if t.dim() == 4:  # [B, C, H, W]
        t = t[batch_idx]
    if t.dim() == 3:
        if t.shape[0] in (1, 3, 4):  # [C, H, W]
            t = t.permute(1, 2, 0)
    if t.dim() == 3 and t.shape[2] == 1:
        t = t.squeeze(2)
    return t.numpy()


def _depth_vmin_vmax(depth_np: np.ndarray):
    """Compute reasonable vmin/vmax for depth visualization."""
    valid = depth_np[depth_np > 0]
    if len(valid) == 0:
        return 0.1, 10.0
    vmin = float(np.percentile(valid, 1))
    vmax = float(np.percentile(valid, 99))
    return max(vmin, 0.1), max(vmax, vmin + 0.1)


def save_training_visualization(
    out_dir: Path,
    step: int,
    *,
    gt_rgb_0,
    rendered_rgb_0,
    gt_depth_0,
    rendered_depth_0,
    rendered_alpha_0,
    pred_depth_0: Optional,
    pred_mask_0: Optional,
    nonsky_mask_0: Optional,
    gt_rgb_1,
    rendered_rgb_1_nov,
    rendered_depth_1_nov,
    rendered_alpha_1_nov,
):
    """Save a comprehensive training visualization grid.

    Args:
        out_dir: Root directory for visualizations.
        step: Current training step (used for subfolder naming).
        *_0: Tensors for input view (frame t).
        *_1_nov: Tensors for novel view (frame t rendered in t+1 pose).
        pred_depth_0: [B, 2, H, W] DAP predicted two-layer depth.
        pred_mask_0: [B, 1, H, W] DAP raw probability mask.
        nonsky_mask_0: [B, 1, H, W] Binary non-sky mask.
    """
    out_dir = Path(out_dir) / f"step_{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute shared depth range for consistent coloring
    dmin, dmax = _depth_vmin_vmax(_to_numpy(gt_depth_0))

    # --- Helper: save individual image ---
    def save_img(data, name, cmap=None, vmin=None, vmax=None):
        data_np = _to_numpy(data)
        if data_np is None:
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        if cmap is not None:
            ax.imshow(data_np, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            ax.imshow(np.clip(data_np, 0, 1))
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}.jpg", dpi=150, bbox_inches="tight", pad_inches=0.0)
        plt.close()

    # --- Input view (frame t) ---
    save_img(gt_rgb_0, "input_0_gt_rgb")
    save_img(rendered_rgb_0, "input_0_rendered_rgb")
    save_img(gt_depth_0, "input_0_gt_depth", cmap="turbo", vmin=dmin, vmax=dmax)
    save_img(rendered_depth_0, "input_0_rendered_depth", cmap="turbo", vmin=dmin, vmax=dmax)
    save_img(rendered_alpha_0, "input_0_rendered_alpha", cmap="gray", vmin=0, vmax=1)

    if pred_depth_0 is not None:
        save_img(pred_depth_0[:, 0:1], "input_0_pred_depth_l1", cmap="turbo", vmin=dmin, vmax=dmax)
        save_img(pred_depth_0[:, 1:2], "input_0_pred_depth_l2", cmap="turbo", vmin=dmin, vmax=dmax)
    if pred_mask_0 is not None:
        save_img(pred_mask_0, "input_0_pred_mask", cmap="gray", vmin=0, vmax=1)
    if nonsky_mask_0 is not None:
        save_img(nonsky_mask_0, "input_0_nonsky_mask", cmap="gray", vmin=0, vmax=1)

    # --- Novel view (t -> t+1) ---
    save_img(gt_rgb_1, "novel_0to1_gt_rgb")
    save_img(rendered_rgb_1_nov, "novel_0to1_rendered_rgb")
    save_img(rendered_depth_1_nov, "novel_0to1_rendered_depth", cmap="turbo", vmin=dmin, vmax=dmax)
    save_img(rendered_alpha_1_nov, "novel_0to1_rendered_alpha", cmap="gray", vmin=0, vmax=1)

    # --- Combined grid for quick inspection ---
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    def show_on_ax(ax, data, title, cmap=None, vmin=None, vmax=None):
        data_np = _to_numpy(data)
        if data_np is None:
            ax.set_visible(False)
            return
        if cmap is not None:
            ax.imshow(data_np, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            ax.imshow(np.clip(data_np, 0, 1))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    show_on_ax(axes[0], gt_rgb_0, "GT RGB (t)")
    show_on_ax(axes[1], rendered_rgb_0, "Rendered RGB (t)")
    show_on_ax(axes[2], gt_depth_0, "GT Depth (t)", cmap="turbo", vmin=dmin, vmax=dmax)
    show_on_ax(axes[3], rendered_depth_0, "Rendered Depth (t)", cmap="turbo", vmin=dmin, vmax=dmax)

    show_on_ax(axes[4], rendered_alpha_0, "Rendered Alpha (t)", cmap="gray", vmin=0, vmax=1)
    show_on_ax(axes[5], pred_depth_0[:, 0:1] if pred_depth_0 is not None else None,
               "Pred Depth L1 (t)", cmap="turbo", vmin=dmin, vmax=dmax)
    show_on_ax(axes[6], pred_depth_0[:, 1:2] if pred_depth_0 is not None else None,
               "Pred Depth L2 (t)", cmap="turbo", vmin=dmin, vmax=dmax)
    show_on_ax(axes[7], pred_mask_0, "DAP Mask (t)", cmap="gray", vmin=0, vmax=1)

    show_on_ax(axes[8], gt_rgb_1, "GT RGB (t+1)")
    show_on_ax(axes[9], rendered_rgb_1_nov, "Rendered RGB (t→t+1)")
    show_on_ax(axes[10], rendered_depth_1_nov, "Rendered Depth (t→t+1)", cmap="turbo", vmin=dmin, vmax=dmax)
    show_on_ax(axes[11], rendered_alpha_1_nov, "Rendered Alpha (t→t+1)", cmap="gray", vmin=0, vmax=1)

    plt.tight_layout()
    plt.savefig(out_dir / "combined_grid.jpg", dpi=150, bbox_inches="tight")
    plt.close()

    return str(out_dir)



class LossTracker:
    """Track and plot training loss curves.

    Usage:
        tracker = LossTracker()
        for step, log_dict in enumerate(training_loop):
            tracker.update(step, log_dict)
            if step % save_interval == 0:
                tracker.save_plot("outputs/loss_curves.png")
    """

    def __init__(self, keys=None):
        """Args:
            keys: If provided, only track these keys. If None, track all keys.
        """
        self.keys = keys
        self.history = {}  # key -> {"steps": [], "values": []}

    def update(self, step, loss_dict):
        """Append a new step's loss values."""
        for k, v in loss_dict.items():
            if self.keys is not None and k not in self.keys:
                continue
            if k not in self.history:
                self.history[k] = {"steps": [], "values": []}
            self.history[k]["steps"].append(step)
            # Handle both scalar floats and torch tensors
            val = float(v.item()) if hasattr(v, "item") else float(v)
            self.history[k]["values"].append(val)

    def save_plot(
        self,
        out_path,
        figsize=(14, 10),
        max_subplots=12,
        smooth_window=1,
    ):
        """Save a multi-panel loss-curve figure.

        Args:
            out_path: Path to save the PNG.
            figsize: Matplotlib figure size.
            max_subplots: Maximum number of subplots (excess keys are skipped).
            smooth_window: Moving-average window for smoothing curves.
        """
        if not self.history:
            return

        keys = list(self.history.keys())[:max_subplots]
        n = len(keys)
        ncols = 3
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for idx, key in enumerate(keys):
            ax = axes[idx]
            steps = np.array(self.history[key]["steps"])
            values = np.array(self.history[key]["values"])

            # Optional moving-average smoothing
            if smooth_window > 1 and len(values) >= smooth_window:
                kernel = np.ones(smooth_window) / smooth_window
                smoothed = np.convolve(values, kernel, mode="valid")
                smoothed_steps = steps[smooth_window - 1:]
                ax.plot(steps, values, alpha=0.3, color="C0", label="raw")
                ax.plot(smoothed_steps, smoothed, color="C1", label=f"ma{smooth_window}")
            else:
                ax.plot(steps, values, color="C0")

            ax.set_title(key, fontsize=10)
            ax.set_xlabel("step")
            ax.set_ylabel("loss")
            ax.grid(True, alpha=0.3)
            if smooth_window > 1:
                ax.legend(fontsize=7)

        # Hide unused subplots
        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    def get_latest(self, key):
        """Return the most recent value for a given key."""
        if key not in self.history or not self.history[key]["values"]:
            return None
        return self.history[key]["values"][-1]

    def get_summary(self):
        """Return min/max/mean/last for each tracked key."""
        summary = {}
        for key, hist in self.history.items():
            vals = np.array(hist["values"])
            summary[key] = {
                "min": float(vals.min()),
                "max": float(vals.max()),
                "mean": float(vals.mean()),
                "last": float(vals[-1]),
                "count": len(vals),
            }
        return summary
