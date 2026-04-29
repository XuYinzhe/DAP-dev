import os
import cv2
import torch
import numpy as np

def quat_to_matrix(qx, qy, qz, qw):
    """Convert quaternion to rotation matrix."""
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])


def load_tartanair_poses(pose_file, frame_indices=None):
    """Load poses from TartanAir pose file.
    
    Supports two formats:
    - Original: 7 values per line (tx, ty, tz, qx, qy, qz, qw)
      DROID-SLAM permutation [1, 2, 0, 4, 5, 3, 6] is applied.
    - Preprocessed rectified: 12 values per line (3x4 c2w matrix, flattened row-major)
      No conversion needed; already in c2w format.
    
    Args:
        pose_file: Path to pose txt file.
        frame_indices: Set/list of frame indices to load, or None to load all.
    """
    all_poses = []
    with open(pose_file, 'r') as f:
        for i, line in enumerate(f):
            if frame_indices is None or i in frame_indices:
                values = list(map(float, line.strip().split()))
                if len(values) == 12:
                    # Preprocessed rectified format: 3x4 c2w matrix
                    pose = np.eye(4, dtype=np.float32)
                    pose[:3, :] = np.array(values, dtype=np.float32).reshape(3, 4)
                    all_poses.append(pose)
                elif len(values) == 7:
                    # Original TartanAir format: tx, ty, tz, qx, qy, qz, qw
                    # DROID-SLAM permutation: [1, 2, 0, 4, 5, 3, 6]
                    tx, ty, tz = values[1], values[2], values[0]
                    qx, qy, qz, qw = values[4], values[5], values[3], values[6]
                    
                    R = quat_to_matrix(qx, qy, qz, qw)
                    pose = np.eye(4, dtype=np.float32)
                    pose[:3, :3] = R
                    pose[:3, 3] = [tx, ty, tz]
                    all_poses.append(pose)
                else:
                    raise ValueError(
                        f"Unexpected pose format in {pose_file}: "
                        f"line {i} has {len(values)} values (expected 7 or 12)"
                    )
    return np.array(all_poses, dtype=np.float32)


def load_tartanair_depth(depth_path):
    """Read TartanAir depth from PNG (float32 encoded in RGBA)."""
    depth_rgba = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_rgba is None:
        return None
    depth = depth_rgba.view("<f4")
    return np.squeeze(depth, axis=-1)


def save_tartanair_depth(depth_path, depth):
    """Save depth as PNG with float32 encoded in RGBA.
    
    Args:
        depth_path: Output PNG path.
        depth: [H, W] or [1, H, W] numpy float32 array.
    """
    depth = np.squeeze(depth)  # Remove singleton dimensions (e.g. [1,H,W] -> [H,W])
    depth = np.ascontiguousarray(depth.astype(np.float32))
    depth_rgba = depth.view(np.uint8).reshape(depth.shape[0], depth.shape[1], 4)
    cv2.imwrite(depth_path, depth_rgba)


def load_tartanair_rgb(rgb_path):
    """Read TartanAir RGB image."""
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb is None:
        return None
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return rgb
