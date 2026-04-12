"""
Keyframe selection strategies for CelestialSplat.

Supports:
1. Uniform spatial sampling (based on camera position distribution)
2. Translation distance-based sampling

JSON Format Specification:
---------------------------
The chunk assignment JSON file contains the following structure:

{
    "metadata": {
        "pose_file": "/path/to/pose_lcam_front.txt",
        "image_dir": "/path/to/image_lcam_custom0_equirect",
        "image_pattern": "{index:06d}_lcam_image_custom0_equirect.png",
        "depth_dir": "/path/to/depth_lcam_custom0_equirect",
        "depth_pattern": "{index:06d}_lcam_depth_custom0_equirect.png",
        "strategy": "uniform_spatial",
        "num_keyframes": 200,
        "chunk_size": 4,
        "stride": 2,
        "total_frames": 2723,
        "total_keyframes": 200,
        "total_chunks": 99
    },
    "keyframes": {
        "indices": [0, 45, 64, 76, ...]
    },
    "chunks": [
        {
            "chunk_id": 0,
            "frame_indices": [0, 45, 64, 76],
            "frame_info": [
                {"index": 0, "filename": "000000_lcam_image_custom0_equirect.png", "pose": [[...], [...], [...], [...]]},
                {"index": 45, "filename": "000045_lcam_image_custom0_equirect.png", "pose": [[...], [...], [...], [...]]},
                {"index": 64, "filename": "000064_lcam_image_custom0_equirect.png", "pose": [[...], [...], [...], [...]]},
                {"index": 76, "filename": "000076_lcam_image_custom0_equirect.png", "pose": [[...], [...], [...], [...]]}
            ],
            "pose_range": {
                "x": [x_min, x_max],
                "y": [y_min, y_max],
                "z": [z_min, z_max]
            },
            "trajectory_length": 7.25
        },
        ...
    ],
    "statistics": {
        "all_positions_range": {
            "x": [x_min, x_max],
            "y": [y_min, y_max],
            "z": [z_min, z_max]
        },
        "bounding_box": [x_range, y_range, z_range],
        "total_trajectory_length": 550.18,
        "avg_frame_distance": 0.20,
        "avg_chunk_trajectory_length": 8.10,
        "chunk_trajectory_length_stats": {
            "mean": 8.10,
            "std": 2.74,
            "min": 5.02,
            "max": 18.35
        }
    }
}

Field Descriptions:
-------------------
metadata:
  - pose_file: Path to the pose file used for keyframe selection (use pose_lcam_front.txt, not pose_lcam_custom0_equirect.txt)
  - image_dir: Directory containing equirect images (image_lcam_custom0_equirect)
  - image_pattern: Filename pattern for generating image filenames
  - depth_dir: Directory containing equirect depth maps (depth_lcam_custom0_equirect)
  - depth_pattern: Filename pattern for generating depth filenames
  - strategy: Keyframe selection strategy used
  - num_keyframes: Target number of keyframes (for uniform_spatial)
  - chunk_size: Number of frames per chunk
  - stride: Stride between consecutive chunks
  - total_frames: Total number of frames in the sequence
  - total_keyframes: Actual number of keyframes selected
  - total_chunks: Total number of chunks created

keyframes:
  - indices: List of all selected keyframe indices

chunks (list):
  - chunk_id: Sequential chunk identifier (0-indexed)
  - frame_indices: List of frame indices in this chunk
  - frame_info: List of {index, filename, pose} for each frame
    - pose: 4x4 transformation matrix (camera-to-world) after conversion
  - pose_range: Min/max camera position in each axis
  - trajectory_length: Sum of distances between consecutive frames in chunk

statistics:
  - all_positions_range: Min/max of all camera positions
  - bounding_box: 3D bounding box dimensions [x_range, y_range, z_range]
  - total_trajectory_length: Sum of all frame-to-frame distances
  - avg_frame_distance: Average distance between consecutive frames
  - avg_chunk_trajectory_length: Average chunk internal trajectory length
  - chunk_trajectory_length_stats: Statistics of chunk trajectory lengths
"""

import torch
import numpy as np
import json
import os
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Default matplotlib backend (can be overridden by caller)
def set_matplotlib_backend(backend: Optional[str] = None):
    """
    Set matplotlib backend. Should be called before importing pyplot.
    
    Args:
        backend: 'Qt5Agg', 'TkAgg', 'Agg', or None for auto-detection
    """
    if backend is None:
        # Auto-detect based on DISPLAY environment variable
        if os.environ.get('DISPLAY'):
            try:
                matplotlib.use('Qt5Agg')
            except ImportError:
                try:
                    matplotlib.use('TkAgg')
                except ImportError:
                    pass
        else:
            matplotlib.use('Agg')
    else:
        matplotlib.use(backend)


def uniform_spatial_sampling(
    poses: torch.Tensor,
    num_keyframes: int,
    return_indices: bool = True
) -> Tuple[torch.Tensor, List[int]]:
    """
    Uniform spatial sampling based on camera position distribution.
    
    Strategy:
    - Use k-means++ style initialization to select spatially diverse keyframes
    - This ensures more uniform coverage of the trajectory, regardless of speed variations
    
    Args:
        poses: [N, 4, 4] camera poses (world-to-camera)
        num_keyframes: desired number of keyframes
        return_indices: whether to return selected indices
    
    Returns:
        selected_poses: [K, 4, 4] selected keyframe poses
        indices: list of selected indices
    """
    N = len(poses)
    positions = poses[:, :3, 3]  # [N, 3]
    
    if num_keyframes >= N:
        return poses, list(range(N))
    
    # K-means++ initialization for spatial diversity
    selected_indices = []
    
    # 1. First keyframe: random or first frame
    first_idx = 0
    selected_indices.append(first_idx)
    
    # 2. Iteratively select farthest point from existing set
    for _ in range(num_keyframes - 1):
        # Compute min distance to any selected point
        selected_positions = positions[selected_indices]  # [M, 3]
        
        # Pairwise distances: [N, M]
        dists = torch.cdist(positions, selected_positions)
        min_dists = dists.min(dim=1)[0]  # [N]
        
        # Select point with max min-distance
        next_idx = min_dists.argmax().item()
        
        # Avoid selecting too close points (optional)
        if min_dists[next_idx] < 0.01:  # threshold for "too close"
            # Fall back to uniform sampling if clustering fails
            remaining = [i for i in range(N) if i not in selected_indices]
            if remaining:
                next_idx = remaining[len(remaining) // 2]
        
        if next_idx not in selected_indices:
            selected_indices.append(next_idx)
    
    # Sort indices to maintain temporal order (important for sequence!)
    selected_indices = sorted(selected_indices)
    
    selected_poses = poses[selected_indices]
    
    if return_indices:
        return selected_poses, selected_indices
    return selected_poses


def translation_distance_sampling(
    poses: torch.Tensor,
    min_translation: float = 0.5,
    return_indices: bool = True
) -> Tuple[torch.Tensor, List[int]]:
    """
    Select keyframes based on minimum translation distance.
    
    Args:
        poses: [N, 4, 4] camera poses
        min_translation: minimum distance between consecutive keyframes
        return_indices: whether to return selected indices
    
    Returns:
        selected_poses: [K, 4, 4] selected keyframe poses
        indices: list of selected indices
    """
    N = len(poses)
    positions = poses[:, :3, 3]  # [N, 3]
    
    selected_indices = [0]  # Always include first frame
    last_position = positions[0]
    
    for i in range(1, N):
        dist = torch.norm(positions[i] - last_position)
        if dist >= min_translation:
            selected_indices.append(i)
            last_position = positions[i]
    
    # Always include last frame for complete coverage
    if selected_indices[-1] != N - 1:
        selected_indices.append(N - 1)
    
    selected_poses = poses[selected_indices]
    
    if return_indices:
        return selected_poses, selected_indices
    return selected_poses


def create_chunks_from_keyframes(
    keyframe_indices: List[int],
    chunk_size: int = 4,
    stride: Optional[int] = None,
) -> List[List[int]]:
    """
    Create chunks from selected keyframes.
    
    Args:
        keyframe_indices: list of selected keyframe indices
        chunk_size: number of frames per chunk
        stride: stride between chunks (None = non-overlapping)
    
    Returns:
        chunks: list of chunk indices
    """
    if stride is None:
        stride = chunk_size  # Non-overlapping
    
    chunks = []
    for start in range(0, len(keyframe_indices) - chunk_size + 1, stride):
        chunk = keyframe_indices[start:start + chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(chunk)
    
    return chunks


def visualize_keyframe_selection(
    poses: torch.Tensor,
    keyframe_indices: List[int],
    chunk_indices: Optional[List[List[int]]] = None,
    title: str = "Keyframe Selection",
    save_path: Optional[str] = None,
):
    """
    Visualize keyframe selection in 3D.
    
    Args:
        poses: [N, 4, 4] all camera poses
        keyframe_indices: selected keyframe indices
        chunk_indices: optional chunks to visualize with different colors
        title: plot title
        save_path: optional path to save figure
    """
    positions = poses[:, :3, 3].cpu().numpy()
    keyframe_positions = positions[keyframe_indices]
    
    fig = plt.figure(figsize=(12, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all camera positions (faint)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c='lightgray', alpha=0.3, s=10, label='All frames')
    
    # Plot trajectory line
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            'b-', alpha=0.2, linewidth=0.5)
    
    # Plot keyframes
    ax.scatter(keyframe_positions[:, 0], keyframe_positions[:, 1], keyframe_positions[:, 2],
               c='red', s=100, marker='o', edgecolors='black', linewidth=2,
               label=f'Keyframes (N={len(keyframe_indices)})', zorder=5)
    
    # Plot keyframe indices
    for i, idx in enumerate(keyframe_indices):
        ax.text(keyframe_positions[i, 0], keyframe_positions[i, 1], keyframe_positions[i, 2],
                str(idx), fontsize=8, color='darkred', fontweight='bold')
    
    # Plot chunks with different colors if provided
    if chunk_indices is not None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(chunk_indices)))
        for i, chunk in enumerate(chunk_indices):
            chunk_positions = positions[chunk]
            ax.scatter(chunk_positions[:, 0], chunk_positions[:, 1], chunk_positions[:, 2],
                      c=[colors[i]], s=50, marker='^', alpha=0.7,
                      label=f'Chunk {i}' if i < 5 else None)
            # Connect chunk frames
            ax.plot(chunk_positions[:, 0], chunk_positions[:, 1], chunk_positions[:, 2],
                   '--', color=colors[i], alpha=0.5)
    
    # Draw camera frustums for keyframes (optional, simplified)
    for idx in keyframe_indices[::max(1, len(keyframe_indices)//10)]:  # Subsample for clarity
        pose = poses[idx]
        R = pose[:3, :3].cpu().numpy()
        t = pose[:3, 3].cpu().numpy()
        
        # Simple frustum visualization (just forward direction)
        forward = -R[:, 2] * 0.3  # Camera looks at -Z
        ax.quiver(t[0], t[1], t[2], forward[0], forward[1], forward[2],
                 length=0.5, normalize=True, color='green', alpha=0.5, arrow_length_ratio=0.3)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # Equal aspect ratio
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                          positions[:, 1].max() - positions[:, 1].min(),
                          positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    
    # Only show interactive plot if not using Agg backend
    if matplotlib.get_backend() != 'Agg':
        plt.show()
    
    return fig


def _infer_image_info_from_pose_file(pose_file: str) -> tuple:
    """
    Infer image directory and filename pattern from pose file path.
    
    Supports TartanAir format:
    - pose: .../pose_lcam_front.txt (correct poses, not pose_lcam_custom0_equirect.txt)
    - images: .../image_lcam_custom0_equirect (equirect images converted from cubemaps)/{index:06d}_lcam_image_custom0_equirect.png
    
    Returns:
        (image_dir, image_pattern)
    """
    pose_dir = os.path.dirname(pose_file)
    pose_basename = os.path.basename(pose_file)
    
    # Check if this is a front camera pose file (correct poses for equirect)
    # pose_lcam_front.txt or pose_front.txt -> image_lcam_custom0_equirect
    if 'lcam_front' in pose_basename or pose_basename == 'pose_front.txt':
        # Look for equirect image directory
        # Try image_lcam_custom0_equirect first
        equirect_dir = os.path.join(pose_dir, 'image_lcam_custom0_equirect')
        if os.path.exists(equirect_dir):
            image_dir = equirect_dir
            pattern = "{index:06d}_lcam_image_custom0_equirect.png"
        else:
            # Fallback to standard front image directory
            image_dir = os.path.join(pose_dir, 'image_lcam_front')
            pattern = "{index:06d}_lcam_image_front.png"
    elif 'pose_' in pose_basename:
        # Replace 'pose_' with 'image_' and remove '.txt' to get image directory
        image_dir_name = pose_basename.replace('pose_', 'image_').replace('.txt', '')
        image_dir = os.path.join(pose_dir, image_dir_name)
        
        # Infer pattern from pose_basename
        # e.g., "pose_lcam_custom0_equirect.txt" -> "{index:06d}_lcam_image_custom0_equirect.png"
        # The pattern is: {index:06d}_{camera}_image_{rest}.png
        pose_body = pose_basename.replace('.txt', '')  # "pose_lcam_custom0_equirect"
        parts = pose_body.split('_')  # ["pose", "lcam", "custom0", "equirect"]
        
        if len(parts) >= 3:
            # Insert "image" after camera name (parts[1])
            # pose_lcam_custom0_equirect -> {index:06d}_lcam_image_custom0_equirect.png
            camera = parts[1]
            rest = '_'.join(parts[2:])  # "custom0_equirect"
            pattern = f"{{index:06d}}_{camera}_image_{rest}.png"
        else:
            pattern = "{index:06d}.png"
    else:
        # Default fallback
        image_dir = pose_dir
        pattern = "{index:06d}.png"
    
    return image_dir, pattern

def test_keyframe_selection(
    pose_file: str,
    strategy: str = "uniform_spatial",
    num_keyframes: int = 32,
    chunk_size: int = 4,
    stride: Optional[int] = None,
    save_path: Optional[str] = None,
    json_path: Optional[str] = None,
):
    """
    Test keyframe selection on a TartanAir sequence.
    
    Args:
        pose_file: path to pose file (.txt)
        strategy: "uniform_spatial" or "translation_distance"
        num_keyframes: for uniform_spatial strategy
        chunk_size: frames per chunk
        stride: chunk stride (None = non-overlapping)
        save_path: optional path to save visualization
        json_path: optional path to save chunk assignment JSON
    
    Returns:
        Dictionary containing poses, keyframe_indices, chunks, fig, and statistics
    """
    # Infer image directory and pattern from pose file
    image_dir, image_pattern = _infer_image_info_from_pose_file(pose_file)
    
    # Infer depth directory and pattern from image directory
    # Replace 'image' with 'depth' and '_image_' with '_depth_' in pattern
    depth_dir = image_dir.replace('image_lcam_custom0', 'depth_lcam_custom0')
    depth_pattern = image_pattern.replace('_image_', '_depth_')
    
    # Load poses
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            if len(values) == 7:
                # DROID-SLAM permutation: [1, 2, 0, 4, 5, 3, 6]
                # tx, ty, tz, qx, qy, qz, qw -> ty, tz, tx, qy, qz, qx, qw
                ty, tz, tx, qy, qz, qx, qw = values
                R = np.array([
                    [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                    [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                    [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
                ])
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = [tx, ty, tz]
                poses.append(pose)
            elif len(values) == 12:
                # Standard 3x4 matrix
                pose = np.eye(4)
                pose[:3, :] = np.array(values).reshape(3, 4)
                poses.append(pose)
    
    poses = torch.from_numpy(np.array(poses)).float()
    print(f"Loaded {len(poses)} poses from {pose_file}")
    
    # Select keyframes
    if strategy == "uniform_spatial":
        num_keyframes = min(len(poses)//4, int(len(poses)*0.1))  # Limit to 20% of frames or 1/4 of total, whichever is smaller
        keyframe_poses, keyframe_indices = uniform_spatial_sampling(poses, num_keyframes)
        print(f"Selected {len(keyframe_indices)} keyframes using uniform spatial sampling")
    elif strategy == "translation_distance":
        keyframe_poses, keyframe_indices = translation_distance_sampling(poses, min_translation=4.0)
        print(f"Selected {len(keyframe_indices)} keyframes using translation distance")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Create chunks
    chunks = create_chunks_from_keyframes(keyframe_indices, chunk_size, stride)
    print(f"Created {len(chunks)} chunks (size={chunk_size}, stride={stride or chunk_size})")
    print(f"First 3 chunks: {chunks[:3]}")
    print(f"Last 3 chunks: {chunks[-3:]}")
    
    # Compute and print statistics
    print("\n" + "=" * 70)
    print("Statistics")
    print("=" * 70)
    
    # All camera translations range
    all_positions = poses[:, :3, 3].numpy()  # [N, 3]
    print("\n1. All Camera Translations Range:")
    print(f"   X: [{all_positions[:, 0].min():.4f}, {all_positions[:, 0].max():.4f}] (range: {all_positions[:, 0].max() - all_positions[:, 0].min():.4f})")
    print(f"   Y: [{all_positions[:, 1].min():.4f}, {all_positions[:, 1].max():.4f}] (range: {all_positions[:, 1].max() - all_positions[:, 1].min():.4f})")
    print(f"   Z: [{all_positions[:, 2].min():.4f}, {all_positions[:, 2].max():.4f}] (range: {all_positions[:, 2].max() - all_positions[:, 2].min():.4f})")
    
    # Overall bounding box and trajectory statistics
    bbox_min = all_positions.min(axis=0)
    bbox_max = all_positions.max(axis=0)
    bbox_range = bbox_max - bbox_min
    
    # Global trajectory statistics
    frame_distances = np.linalg.norm(np.diff(all_positions, axis=0), axis=1)
    total_trajectory_length = np.sum(frame_distances)
    avg_frame_distance = np.mean(frame_distances)
    
    print(f"   Bounding Box: {bbox_range}")
    print(f"   Total trajectory length: {total_trajectory_length:.4f}")
    print(f"   Average frame-to-frame distance: {avg_frame_distance:.4f}")
    
    # Each chunk's camera translation range and trajectory length
    print(f"\n2. Per-Chunk Camera Translation Range Statistics:")
    chunk_ranges_x = []
    chunk_ranges_y = []
    chunk_ranges_z = []
    chunk_diagonal_ranges = []
    
    for i, chunk in enumerate(chunks):
        chunk_positions = all_positions[chunk]  # [chunk_size, 3]
        chunk_min = chunk_positions.min(axis=0)
        chunk_max = chunk_positions.max(axis=0)
        chunk_range = chunk_max - chunk_min
        chunk_diagonal = np.linalg.norm(chunk_range)
        
        chunk_ranges_x.append(chunk_range[0])
        chunk_ranges_y.append(chunk_range[1])
        chunk_ranges_z.append(chunk_range[2])
        chunk_diagonal_ranges.append(chunk_diagonal)
    
    chunk_ranges_x = np.array(chunk_ranges_x)
    chunk_ranges_y = np.array(chunk_ranges_y)
    chunk_ranges_z = np.array(chunk_ranges_z)
    chunk_diagonal_ranges = np.array(chunk_diagonal_ranges)
    
    print(f"   X range - Mean: {chunk_ranges_x.mean():.4f}, Std: {chunk_ranges_x.std():.4f}, Min: {chunk_ranges_x.min():.4f}, Max: {chunk_ranges_x.max():.4f}")
    print(f"   Y range - Mean: {chunk_ranges_y.mean():.4f}, Std: {chunk_ranges_y.std():.4f}, Min: {chunk_ranges_y.min():.4f}, Max: {chunk_ranges_y.max():.4f}")
    print(f"   Z range - Mean: {chunk_ranges_z.mean():.4f}, Std: {chunk_ranges_z.std():.4f}, Min: {chunk_ranges_z.min():.4f}, Max: {chunk_ranges_z.max():.4f}")
    print(f"   Diagonal range (3D bounding box) - Mean: {chunk_diagonal_ranges.mean():.4f}, Std: {chunk_diagonal_ranges.std():.4f}, Min: {chunk_diagonal_ranges.min():.4f}, Max: {chunk_diagonal_ranges.max():.4f}")
    
    # Chunk trajectory length statistics
    print(f"\n3. Chunk Trajectory Statistics:")
    chunk_trajectory_lengths = []
    
    for chunk in chunks:
        chunk_positions = all_positions[chunk]  # [chunk_size, 3]
        chunk_distances = np.linalg.norm(np.diff(chunk_positions, axis=0), axis=1)
        chunk_trajectory_lengths.append(np.sum(chunk_distances))
    
    chunk_trajectory_lengths = np.array(chunk_trajectory_lengths)
    print(f"   Per-chunk trajectory length - Mean: {chunk_trajectory_lengths.mean():.4f}, Std: {chunk_trajectory_lengths.std():.4f}")
    print(f"   Min: {chunk_trajectory_lengths.min():.4f}, Max: {chunk_trajectory_lengths.max():.4f}")
    print(f"   Ratio to average frame distance: {chunk_trajectory_lengths.mean() / avg_frame_distance:.2f}x")
    
    print("=" * 70)
    
    # Build statistics dictionary
    statistics = {
        "all_positions_range": {
            "x": [float(all_positions[:, 0].min()), float(all_positions[:, 0].max())],
            "y": [float(all_positions[:, 1].min()), float(all_positions[:, 1].max())],
            "z": [float(all_positions[:, 2].min()), float(all_positions[:, 2].max())],
        },
        "bounding_box": bbox_range.tolist(),
        "total_trajectory_length": float(total_trajectory_length),
        "avg_frame_distance": float(avg_frame_distance),
        "avg_chunk_trajectory_length": float(chunk_trajectory_lengths.mean()),
        "chunk_trajectory_length_stats": {
            "mean": float(chunk_trajectory_lengths.mean()),
            "std": float(chunk_trajectory_lengths.std()),
            "min": float(chunk_trajectory_lengths.min()),
            "max": float(chunk_trajectory_lengths.max()),
        }
    }
    
    # Build chunk info for JSON
    chunks_info = []
    for i, chunk in enumerate(chunks):
        chunk_positions = all_positions[chunk]
        chunk_min = chunk_positions.min(axis=0)
        chunk_max = chunk_positions.max(axis=0)
        
        frame_info = []
        for idx in chunk:
            filename = image_pattern.format(index=idx)
            # Get pose for this frame (4x4 transformation matrix)
            pose_matrix = poses[idx].numpy()  # Convert from torch.Tensor to numpy
            frame_info.append({
                "index": int(idx),
                "filename": filename,
                "pose": pose_matrix.tolist()  # 4x4 transformation matrix
            })
        
        chunks_info.append({
            "chunk_id": i,
            "frame_indices": [int(x) for x in chunk],
            "frame_info": frame_info,
            "pose_range": {
                "x": [float(chunk_min[0]), float(chunk_max[0])],
                "y": [float(chunk_min[1]), float(chunk_max[1])],
                "z": [float(chunk_min[2]), float(chunk_max[2])],
            },
            "trajectory_length": float(chunk_trajectory_lengths[i])
        })
    
    # Prepare JSON data
    json_data = {
        "metadata": {
            "pose_file": pose_file,
            "image_dir": image_dir,
            "image_pattern": image_pattern,
            "depth_dir": depth_dir,
            "depth_pattern": depth_pattern,
            "strategy": strategy,
            "num_keyframes": num_keyframes,
            "chunk_size": chunk_size,
            "stride": stride if stride is not None else chunk_size,
            "total_frames": len(poses),
            "total_keyframes": len(keyframe_indices),
            "total_chunks": len(chunks),
        },
        "keyframes": {
            "indices": [int(x) for x in keyframe_indices],
        },
        "chunks": chunks_info,
        "statistics": statistics
    }
    
    # Save JSON if path provided
    if json_path:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        save_chunk_assignment_json(json_data, json_path)
    
    # Visualize
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = visualize_keyframe_selection(
        poses, keyframe_indices, chunks,
        title=f"{strategy} | {len(keyframe_indices)} keyframes | {len(chunks)} chunks",
        save_path=save_path
    )
    
    return {
        'poses': poses,
        'keyframe_indices': keyframe_indices,
        'chunks': chunks,
        'fig': fig,
        'statistics': statistics,
        'json_data': json_data,
    }


# =============================================================================
# JSON I/O Functions
# =============================================================================

def save_chunk_assignment_json(data: Dict[str, Any], json_path: str) -> None:
    """
    Save chunk assignment data to JSON file.
    
    Args:
        data: Dictionary containing chunk assignment information
        json_path: Path to save JSON file
    """
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    data = convert_to_native(data)
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved chunk assignment to {json_path}")


def load_chunk_assignment_json(json_path: str) -> Dict[str, Any]:
    """
    Load chunk assignment data from JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Dictionary containing chunk assignment information
        
    Example:
        >>> data = load_chunk_assignment_json('chunks.json')
        >>> print(f"Total chunks: {data['metadata']['total_chunks']}")
        >>> for chunk in data['chunks']:
        ...     print(f"Chunk {chunk['chunk_id']}: frames {chunk['frame_indices']}")
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def print_chunk_summary(json_path: str) -> None:
    """
    Print a concise summary of chunk assignment from JSON file.
    
    Args:
        json_path: Path to JSON file
    """
    data = load_chunk_assignment_json(json_path)
    meta = data['metadata']
    stats = data['statistics']
    
    print("\n" + "=" * 70)
    print("Chunk Assignment Summary")
    print("=" * 70)
    print(f"Pose file: {meta['pose_file']}")
    print(f"Strategy: {meta['strategy']}")
    print(f"Configuration: {meta['chunk_size']} frames/chunk, stride={meta['stride']}")
    print(f"Total: {meta['total_frames']} frames -> {meta['total_keyframes']} keyframes -> {meta['total_chunks']} chunks")
    print("\nStatistics:")
    print(f"  Total trajectory length: {stats['total_trajectory_length']:.2f}")
    print(f"  Average frame distance: {stats['avg_frame_distance']:.4f}")
    print(f"  Average chunk trajectory: {stats['avg_chunk_trajectory_length']:.4f}")
    print("\nFirst 3 chunks:")
    for chunk in data['chunks'][:3]:
        print(f"  Chunk {chunk['chunk_id']}: indices={chunk['frame_indices']}, traj_len={chunk['trajectory_length']:.4f}")
    print("=" * 70)


def load_tartanair_poses(pose_file: str) -> torch.Tensor:
    """
    Load TartanAir poses from pose_lcam_front.txt and convert to 4x4 transformation matrices.
    
    The pose file format is: tx ty tz qx qy qz qw (7 floats per line)
    DROID-SLAM coordinate permutation is applied: [1,2,0,4,5,3,6] -> [y,z,x,qy,qz,qx,qw]
    
    Args:
        pose_file: Path to pose_lcam_front.txt file
        
    Returns:
        poses: [N, 4, 4] tensor of camera-to-world transformation matrices
        
    Example:
        >>> poses = load_tartanair_poses('/path/to/pose_lcam_front.txt')
        >>> print(poses.shape)  # [N, 4, 4]
    """
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            if len(values) != 7:
                raise ValueError(f"Expected 7 values per pose line (tx ty tz qx qy qz qw), got {len(values)}")
            
            # DROID-SLAM permutation: [1, 2, 0, 4, 5, 3, 6]
            # tx, ty, tz, qx, qy, qz, qw -> ty, tz, tx, qy, qz, qx, qw
            ty, tz, tx, qy, qz, qx, qw = values
            
            # Convert quaternion to rotation matrix
            R = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
            ])
            
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = [tx, ty, tz]
            poses.append(pose)
    
    return torch.from_numpy(np.array(poses)).float()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_file', type=str, required=True,
                       help='Path to pose file')
    parser.add_argument('--strategy', type=str, default='uniform_spatial',
                       choices=['uniform_spatial', 'translation_distance'])
    parser.add_argument('--num_keyframes', type=int, default=32)
    parser.add_argument('--chunk_size', type=int, default=4)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--save_path', type=str, default='./')
    args = parser.parse_args()
    
    result = test_keyframe_selection(
        args.pose_file,
        args.strategy,
        args.num_keyframes,
        args.chunk_size,
        args.stride,
        args.save_path,
    )
