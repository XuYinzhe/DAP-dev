import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, Dict, List

from CelestialSplat.utils.file_loader import (
    load_tartanair_poses,
    load_tartanair_rgb,
    load_tartanair_depth,
    save_tartanair_depth,
)
from CelestialSplat.geometry import HorizonRectifier


def preprocess_sequence(
    seq_path: Path,
    image_size: Tuple[int, int] = (512, 1024),
    device: str = 'cpu',
) -> None:
    """
    Preprocess one TartanAir sequence: resize + horizon rectification.

    Creates per-sequence output dirs:
        image_lcam_custom0_equirect_{W}/
        depth_lcam_custom0_equirect_{W}/
        pose_lcam_front_rectified.txt

    If they already exist, skips.
    """
    image_dir = seq_path / 'image_lcam_custom0_equirect'
    depth_dir = seq_path / 'depth_lcam_custom0_equirect'
    pose_file = seq_path / 'pose_lcam_front.txt'

    out_w = image_size[1]
    out_image_dir = seq_path / f'image_lcam_custom0_equirect_{out_w}'
    out_depth_dir = seq_path / f'depth_lcam_custom0_equirect_{out_w}'
    out_pose_file = seq_path / 'pose_lcam_front_rectified.txt'

    if out_image_dir.exists() and out_depth_dir.exists() and out_pose_file.exists():
        print(f"  [preprocess] Already done for {seq_path.name}")
        return

    if not image_dir.exists() or not depth_dir.exists() or not pose_file.exists():
        raise FileNotFoundError(f"Missing raw data in {seq_path}")

    poses = load_tartanair_poses(str(pose_file), frame_indices=None)
    num_frames = len(poses)
    rectifier = HorizonRectifier(img_h=image_size[0], img_w=image_size[1], device=device)

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_depth_dir.mkdir(parents=True, exist_ok=True)

    rectified_poses = []
    print(f"  [preprocess] Processing {num_frames} frames for {seq_path.name} ...")

    for t in range(num_frames):
        img_path = image_dir / f"{t:06d}_lcam_image_custom0_equirect.png"
        depth_path = depth_dir / f"{t:06d}_lcam_depth_custom0_equirect.png"

        rgb = load_tartanair_rgb(str(img_path))
        rgb = cv2.resize(rgb, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        depth = load_tartanair_depth(str(depth_path))
        depth = cv2.resize(depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
        depth = torch.from_numpy(depth).unsqueeze(0).float()

        pose = torch.from_numpy(poses[t]).float()

        with torch.no_grad():
            rgb_rect, depth_rect, pose_rect = rectifier.rectify(
                rgb.unsqueeze(0), depth.unsqueeze(0), pose.unsqueeze(0)
            )
        rgb_rect = rgb_rect.squeeze(0)
        depth_rect = depth_rect.squeeze(0) if depth_rect is not None else None
        pose_rect = pose_rect.squeeze(0)

        # Save RGB
        rgb_np = (rgb_rect.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cv2.imwrite(
            str(out_image_dir / f"{t:06d}_lcam_image_custom0_equirect.png"),
            cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR),
        )

        # Save depth as 4-channel PNG (float32 encoded in RGBA)
        save_tartanair_depth(
            str(out_depth_dir / f"{t:06d}_lcam_depth_custom0_equirect.png"),
            depth_rect.squeeze(0).numpy(),
        )

        # Save pose as 3x4 matrix row (12 floats)
        rectified_poses.append(pose_rect[:3, :].flatten().numpy())

    # Write pose file: one row per frame, 12 floats (3x4 matrix)
    np.savetxt(str(out_pose_file), np.array(rectified_poses, dtype=np.float32))
    print(f"  [preprocess] Done: {out_image_dir}, {out_depth_dir}, {out_pose_file}")


class TartanAir360MonoDataset(Dataset):
    """
    Stage-1 mono dataset for TartanAir360.
    Loads adjacent frame pairs: (I_t, D_t, P_t) and (I_t+1, D_t+1, P_t+1).

    Supports preprocessed horizon-rectified data (fast load) or on-the-fly
    rectification (slower but no disk usage).
    """

    def __init__(
        self,
        root_dir: str,
        image_size: Tuple[int, int] = (512, 1024),
        num_sequences: Optional[int] = None,
        rectify: bool = True,
        depth_threshold: float = 100.0,
        frame_skip: int = 1,
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size  # (H, W)
        self.rectify = rectify
        self.depth_threshold = depth_threshold
        self.frame_skip = frame_skip
        self.pairs: List[Dict] = []

        # On-the-fly rectifier (only used if preprocessed dirs not found)
        self.rectifier = None
        if self.rectify:
            self.rectifier = HorizonRectifier(
                img_h=image_size[0], img_w=image_size[1], device='cpu'
            )

        scenes = self._scan_scenes(num_sequences)
        for scene_path in scenes:
            self._load_sequence_pairs(scene_path)

        print(f"TartanAir360MonoDataset: {len(scenes)} scenes, {len(self.pairs)} frame pairs")
        print(f"  rectify={rectify}, depth_threshold={depth_threshold}")

    def _scan_scenes(self, num_sequences: Optional[int]) -> List[Path]:
        scenes = []
        for scene_dir in sorted(self.root_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            for difficulty in ['Data_easy', 'Data_hard']:
                diff_dir = scene_dir / difficulty
                if not diff_dir.exists():
                    continue
                for subdir in sorted(diff_dir.iterdir()):
                    if not subdir.is_dir() or not subdir.name.startswith('P'):
                        continue
                    image_dir = subdir / 'image_lcam_custom0_equirect'
                    depth_dir = subdir / 'depth_lcam_custom0_equirect'
                    pose_file = subdir / 'pose_lcam_front.txt'
                    if image_dir.exists() and depth_dir.exists() and pose_file.exists():
                        scenes.append(subdir)
                        break
                if num_sequences is not None and len(scenes) >= num_sequences:
                    return scenes
        return scenes

    def _has_preprocessed(self, seq_path: Path) -> bool:
        out_w = self.image_size[1]
        return (
            (seq_path / f'image_lcam_custom0_equirect_{out_w}').exists() and
            (seq_path / f'depth_lcam_custom0_equirect_{out_w}').exists() and
            (seq_path / 'pose_lcam_front_rectified.txt').exists()
        )

    def _load_sequence_pairs(self, seq_path: Path):
        out_w = self.image_size[1]
        use_preprocessed = self._has_preprocessed(seq_path)

        if use_preprocessed:
            image_dir = seq_path / f'image_lcam_custom0_equirect_{out_w}'
            depth_dir = seq_path / f'depth_lcam_custom0_equirect_{out_w}'
            poses = load_tartanair_poses(str(seq_path / 'pose_lcam_front_rectified.txt'), None)
            self_rectify = False
        else:
            image_dir = seq_path / 'image_lcam_custom0_equirect'
            depth_dir = seq_path / 'depth_lcam_custom0_equirect'
            poses = load_tartanair_poses(str(seq_path / 'pose_lcam_front.txt'), frame_indices=None)
            self_rectify = self.rectify

        num_frames = len(poses)
        max_skip = min(self.frame_skip, num_frames - 1)
        for t in range(num_frames - max_skip):
            self.pairs.append({
                'image_dir': image_dir,
                'depth_dir': depth_dir,
                'frame_0': t,
                'frame_1': t + max_skip,
                'pose_0': torch.from_numpy(poses[t]).float(),
                'pose_1': torch.from_numpy(poses[t + max_skip]).float(),
                'self_rectify': self_rectify,
            })

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_image(self, path: Path) -> torch.Tensor:
        rgb = load_tartanair_rgb(str(path))
        if rgb is None:
            raise RuntimeError(f"Failed to load image: {path}")
        rgb = cv2.resize(rgb, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return rgb

    def _load_depth(self, path: Path) -> torch.Tensor:
        if path.suffix == '.npy':
            depth = np.load(str(path))
        else:
            depth = load_tartanair_depth(str(path))
        if depth is None:
            raise RuntimeError(f"Failed to load depth: {path}")
        depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        depth = torch.from_numpy(depth).unsqueeze(0).float()
        return depth

    def _compute_mask(self, depth: torch.Tensor) -> torch.Tensor:
        """Valid mask: depth > 0 and depth < threshold (sky is ~4188m)."""
        return ((depth > 0.0) & (depth < self.depth_threshold)).float()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]

        img_name_0 = f"{pair['frame_0']:06d}_lcam_image_custom0_equirect.png"
        img_name_1 = f"{pair['frame_1']:06d}_lcam_image_custom0_equirect.png"
        depth_name_0 = f"{pair['frame_0']:06d}_lcam_depth_custom0_equirect.png"
        depth_name_1 = f"{pair['frame_1']:06d}_lcam_depth_custom0_equirect.png"

        image_0 = self._load_image(pair['image_dir'] / img_name_0)
        depth_0 = self._load_depth(pair['depth_dir'] / depth_name_0)
        image_1 = self._load_image(pair['image_dir'] / img_name_1)
        depth_1 = self._load_depth(pair['depth_dir'] / depth_name_1)

        pose_0 = pair['pose_0']
        pose_1 = pair['pose_1']

        # ---- Runtime rectification (only if not preprocessed) ----
        if pair['self_rectify'] and self.rectifier is not None:
            image_0, depth_0, pose_0 = self.rectifier.rectify(
                image_0.unsqueeze(0), depth_0.unsqueeze(0), pose_0.unsqueeze(0)
            )
            image_0 = image_0.squeeze(0)
            depth_0 = depth_0.squeeze(0) if depth_0 is not None else None
            pose_0 = pose_0.squeeze(0)

            image_1, depth_1, pose_1 = self.rectifier.rectify(
                image_1.unsqueeze(0), depth_1.unsqueeze(0), pose_1.unsqueeze(0)
            )
            image_1 = image_1.squeeze(0)
            depth_1 = depth_1.squeeze(0) if depth_1 is not None else None
            pose_1 = pose_1.squeeze(0)

        mask_0 = self._compute_mask(depth_0)
        mask_1 = self._compute_mask(depth_1)

        return {
            'image_0': image_0,       # [3, H, W]
            'depth_0': depth_0,       # [1, H, W]
            'mask_0': mask_0,         # [1, H, W]
            'pose_0': pose_0,         # [4, 4]
            'image_1': image_1,
            'depth_1': depth_1,
            'mask_1': mask_1,
            'pose_1': pose_1,
        }
