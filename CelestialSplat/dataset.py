"""
Dataset implementations for CelestialSplat training.

Supports:
1. Sequence chunking - splits sequence into non-overlapping chunks
2. Random chunk sampling - randomly select chunks per iteration
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple, Optional, Dict
import random
from pathlib import Path
import json


class SequenceChunkDataset(Dataset):
    """
    Dataset that splits a sequence into non-overlapping chunks.
    
    Each chunk contains `chunk_size` consecutive frames.
    Training randomly samples chunks; validation iterates through all chunks.
    """
    
    def __init__(
        self,
        images: torch.Tensor,           # [N, 3, H, W]
        poses: torch.Tensor,            # [N, 4, 4] world-to-camera
        intrinsics: Optional[torch.Tensor] = None,
        chunk_size: int = 4,            # Number of frames per chunk
        mode: str = 'train',            # 'train' or 'val'
        cache_chunks: bool = True,
    ):
        """
        Args:
            images: Sequence of images [N, 3, H, W]
            poses: Camera poses [N, 4, 4]
            intrinsics: Camera intrinsics (not used for ERP)
            chunk_size: Number of frames in each chunk
            mode: 'train' or 'val'
            cache_chunks: Whether to pre-compute and cache chunk indices
        """
        super().__init__()
        self.images = images
        self.poses = poses
        self.intrinsics = intrinsics
        self.chunk_size = chunk_size
        self.mode = mode
        self.num_frames = len(images)
        
        # Create chunks (non-overlapping)
        self.chunks = self._create_chunks()
        
        print(f"SequenceChunkDataset: {self.num_frames} frames -> {len(self.chunks)} chunks (size={chunk_size})")
    
    def _create_chunks(self) -> List[List[int]]:
        """Create non-overlapping chunks of frame indices."""
        chunks = []
        for start_idx in range(0, self.num_frames, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, self.num_frames)
            if end_idx - start_idx == self.chunk_size:
                # Only use full chunks
                chunk = list(range(start_idx, end_idx))
                chunks.append(chunk)
        return chunks
    
    def __len__(self) -> int:
        """Number of chunks."""
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a chunk of frames.
        
        Returns:
            Dict with:
                - images: [chunk_size, 3, H, W]
                - poses: [chunk_size, 4, 4]
                - indices: [chunk_size] original frame indices
        """
        chunk_indices = self.chunks[idx]
        
        return {
            'images': self.images[chunk_indices],
            'poses': self.poses[chunk_indices],
            'indices': torch.tensor(chunk_indices, dtype=torch.long),
        }


class TartanAir360Dataset(Dataset):
    """
    Dataset for TartanAir 360° sequences (equirectangular projection).
    
    Loads a sequence from disk and splits into chunks for training.
    Expected directory structure:
        root_dir/
            image_lcam_custom0_equirect/
                000000.png
                000001.png
                ...
            pose_lcam_custom0_equirect.txt
    
    Pose file format: each line is 12 floats (3x4 transformation matrix)
    """
    
    def __init__(
        self,
        root_dir: str,
        chunk_size: int = 4,
        image_size: Tuple[int, int] = (512, 1024),
        mode: str = 'train',
        max_frames: Optional[int] = None,
        camera_name: str = 'lcam_custom0',
    ):
        """
        Args:
            root_dir: Path to sequence directory (e.g., .../P000/)
            chunk_size: Number of frames per chunk
            image_size: (H, W) to resize images
            mode: 'train' or 'val'
            max_frames: Maximum number of frames to load (for debugging)
            camera_name: Camera name for finding image/pose files (default: 'lcam_custom0')
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.mode = mode
        self.camera_name = camera_name
        
        # Load data
        self.images, self.poses = self._load_sequence(max_frames)
        
        # Create chunks
        self.chunks = self._create_chunks()
        
        print(f"TartanAir360Dataset: {len(self.images)} frames -> {len(self.chunks)} chunks")
    
    def _load_sequence(self, max_frames: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load images and poses from disk."""
        from PIL import Image
        
        # Try new equirect structure first
        image_dir = self.root_dir / f'image_{self.camera_name}_equirect'
        pose_file = self.root_dir / f'pose_{self.camera_name}_equirect.txt'
        
        # Fallback to old structure if not found
        if not image_dir.exists():
            image_dir = self.root_dir / 'image_left'
            pose_file = self.root_dir / 'pose_left.txt'
        
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not pose_file.exists():
            raise FileNotFoundError(f"Pose file not found: {pose_file}")
        
        # Get sorted image files
        image_files = sorted(image_dir.glob('*.png'))
        if not image_files:
            image_files = sorted(image_dir.glob('*.jpg'))
        
        if max_frames is not None:
            image_files = image_files[:max_frames]
        
        print(f"Loading {len(image_files)} images from {image_dir}...")
        
        # Load images
        images = []
        for img_file in image_files:
            img = Image.open(img_file).convert('RGB')
            img = img.resize((self.image_size[1], self.image_size[0]))  # PIL uses (W, H)
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
            images.append(img)
        
        images = torch.stack(images)  # [N, 3, H, W]
        
        # Load poses
        # Try to detect format: 12 values (3x4 matrix) or 7 values (translation + quaternion)
        poses = []
        with open(pose_file, 'r') as f:
            lines = f.readlines()[:len(image_files)]
            for line in lines:
                values = list(map(float, line.strip().split()))
                
                if len(values) == 12:
                    # Standard TartanAir format: 3x4 matrix
                    pose_3x4 = np.array(values).reshape(3, 4)
                    pose_4x4 = np.eye(4)
                    pose_4x4[:3, :] = pose_3x4
                elif len(values) == 7:
                    # Equirect format: tx, ty, tz, qx, qy, qz, qw
                    tx, ty, tz, qx, qy, qz, qw = values
                    
                    # Convert quaternion to rotation matrix
                    # q = qw + qx*i + qy*j + qz*k
                    R = np.array([
                        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
                    ])
                    
                    pose_4x4 = np.eye(4)
                    pose_4x4[:3, :3] = R
                    pose_4x4[:3, 3] = [tx, ty, tz]
                else:
                    raise ValueError(f"Expected 7 or 12 values per pose line, got {len(values)}")
                
                poses.append(torch.from_numpy(pose_4x4).float())
        
        poses = torch.stack(poses)  # [N, 4, 4]
        
        print(f"Loaded {len(poses)} poses from {pose_file}")
        
        return images, poses
    
    def _create_chunks(self) -> List[List[int]]:
        """Create non-overlapping chunks of frame indices."""
        num_frames = len(self.images)
        chunks = []
        for start_idx in range(0, num_frames, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, num_frames)
            if end_idx - start_idx == self.chunk_size:
                chunk = list(range(start_idx, end_idx))
                chunks.append(chunk)
        return chunks
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk_indices = self.chunks[idx]
        
        return {
            'images': self.images[chunk_indices],
            'poses': self.poses[chunk_indices],
            'indices': torch.tensor(chunk_indices, dtype=torch.long),
        }


class MultiSequenceDataset(Dataset):
    """
    Dataset that aggregates multiple sequences.
    """
    
    def __init__(
        self,
        sequence_datasets: List[Dataset],
    ):
        self.datasets = sequence_datasets
        
        # Compute cumulative sizes
        self.cumulative_sizes = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)
    
    def __len__(self) -> int:
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cumsum in enumerate(self.cumulative_sizes):
            if idx < cumsum:
                dataset_idx = i
                break
        
        # Compute local index
        local_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
        
        return self.datasets[dataset_idx][local_idx]


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Create train and validation dataloaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        batch_size: Batch size per GPU
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory
    
    Returns:
        train_loader, val_loader (or None)
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    
    return train_loader, val_loader


# Example usage
if __name__ == '__main__':
    # Test with dummy data
    print("Testing SequenceChunkDataset...")
    
    # Create dummy sequence
    num_frames = 20
    images = torch.randn(num_frames, 3, 512, 1024)
    
    # Create dummy poses (cameras on a circle)
    poses = []
    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        pose = torch.eye(4)
        pose[0, 3] = 2.0 * np.cos(angle)
        pose[2, 3] = 2.0 * np.sin(angle)
        # Add rotation to face center
        pose[:3, :3] = torch.tensor([
            [np.cos(angle + np.pi), 0, np.sin(angle + np.pi)],
            [0, 1, 0],
            [-np.sin(angle + np.pi), 0, np.cos(angle + np.pi)],
        ], dtype=torch.float32)
        poses.append(pose)
    poses = torch.stack(poses)
    
    # Create dataset
    dataset = SequenceChunkDataset(
        images=images,
        poses=poses,
        chunk_size=4,
        mode='train',
    )
    
    print(f"\nDataset length: {len(dataset)}")
    print(f"Chunks: {dataset.chunks}")
    
    # Sample a few examples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nChunk {i}:")
        print(f"  Images shape: {sample['images'].shape}")
        print(f"  Poses shape: {sample['poses'].shape}")
        print(f"  Indices: {sample['indices'].tolist()}")
    
    # Test dataloader
    print("\nTesting DataLoader...")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )
    
    for batch in loader:
        print(f"\nBatch:")
        print(f"  Images: {batch['images'].shape}")
        print(f"  Poses: {batch['poses'].shape}")
        print(f"  Indices: {batch['indices']}")
        break
    
    print("\n✓ All tests passed!")
