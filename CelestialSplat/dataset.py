"""
Dataset implementations for CelestialSplat training.

Supports:
1. Sequence chunking - splits sequence into non-overlapping chunks
2. Random chunk sampling - randomly select chunks per iteration
3. Keyframe-based chunking - select keyframes then create chunks
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
import random
from pathlib import Path
import json

from CelestialSplat.utils.keyframe import (
    uniform_spatial_sampling,
    translation_distance_sampling,
    create_chunks_from_keyframes,
    load_chunk_assignment_json,
    save_chunk_assignment_json,
    load_tartanair_poses,
)


class SequenceChunkDataset(Dataset):
    """
    Dataset that splits a sequence into chunks.
    
    Each chunk contains `chunk_size` frames.
    Supports consecutive or precomputed chunks.
    Training randomly samples chunks; validation iterates through all chunks.
    """
    
    def __init__(
        self,
        images: torch.Tensor,           # [N, 3, H, W]
        poses: torch.Tensor,            # [N, 4, 4] world-to-camera
        intrinsics: Optional[torch.Tensor] = None,
        chunk_size: int = 4,            # Number of frames per chunk
        chunk_stride: Optional[int] = None,  # None = non-overlapping
        mode: str = 'train',            # 'train' or 'val'
        cache_chunks: bool = True,
        chunks: Optional[List[List[int]]] = None,  # Precomputed chunks
    ):
        """
        Args:
            images: Sequence of images [N, 3, H, W]
            poses: Camera poses [N, 4, 4]
            intrinsics: Camera intrinsics (not used for ERP)
            chunk_size: Number of frames in each chunk
            chunk_stride: Stride between chunks (None = non-overlapping)
            mode: 'train' or 'val'
            cache_chunks: Whether to pre-compute and cache chunk indices
            chunks: Optional precomputed chunk indices (overrides chunk_size/stride)
        """
        super().__init__()
        self.images = images
        self.poses = poses
        self.intrinsics = intrinsics
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride
        self.mode = mode
        self.num_frames = len(images)
        
        # Use precomputed chunks or create new ones
        if chunks is not None:
            self.chunks = chunks
        else:
            self.chunks = self._create_chunks()
        
        stride_str = f"stride={chunk_stride or chunk_size}"
        print(f"SequenceChunkDataset: {self.num_frames} frames -> {len(self.chunks)} chunks (size={chunk_size}, {stride_str})")
    
    def _create_chunks(self) -> List[List[int]]:
        """Create chunks with optional overlap."""
        stride = self.chunk_stride or self.chunk_size
        chunks = []
        for start_idx in range(0, self.num_frames - self.chunk_size + 1, stride):
            chunk = list(range(start_idx, start_idx + self.chunk_size))
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


def load_chunks_from_json(json_path: str) -> Tuple[List[List[int]], Dict]:
    """
    Load chunks from a JSON file.
    
    Args:
        json_path: Path to chunks.json file
    
    Returns:
        chunks: List of chunk index lists
        metadata: Dictionary with metadata
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    chunks = [chunk['frame_indices'] for chunk in data['chunks']]
    metadata = {
        'strategy': data.get('metadata', {}).get('strategy', 'unknown'),
        'chunk_size': data.get('metadata', {}).get('chunk_size', 4),
        'stride': data.get('metadata', {}).get('stride', 4),
        'total_chunks': len(chunks),
    }
    return chunks, metadata


class TartanAir360Dataset(Dataset):
    """
    Dataset for TartanAir 360° sequences (equirectangular projection).
    
    Loads sequences from the TartanAir360 dataset structure:
        root_dir/
            {sequence_name}/
                Data_easy/
                    P000/
                        image_lcam_custom0_equirect/
                        pose_lcam_front.txt
                        chunks.json (optional)
                Data_hard/
                    P000/
                        ...
    
    Training treats easy and hard as separate sequences.
    
    If chunks.json exists, uses the precomputed chunks.
    If chunks.json doesn't exist, uses uniform_spatial strategy to extract keyframes
    and create chunks (same as test_keyframe_selection in keyframe.py).
    """
    
    def __init__(
        self,
        root_dir: str,
        image_size: Tuple[int, int] = (512, 1024),
        mode: str = 'train',
        num_sequences: Optional[int] = None,
        chunk_size: int = 4,
        chunk_stride: int = 2,
    ):
        """
        Args:
            root_dir: Root directory containing all sequences 
                     (e.g., '/homes/shaun/main/dataset/tartanair360/')
            image_size: (H, W) to resize images
            mode: 'train' or 'val'
            num_sequences: Limit to first N sequences (for debugging)
            chunk_size: Number of frames per chunk
            chunk_stride: Stride between chunks
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.mode = mode
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride
        
        # Scan for all sequences
        self.sequences = self._scan_sequences(num_sequences)
        
        # Load all sequences
        self.chunk_list = []  # List of chunk data
        for seq_name, seq_path in self.sequences.items():
            self._load_sequence(seq_name, seq_path)
        
        print(f"TartanAir360Dataset: {len(self.sequences)} sequences, {len(self.chunk_list)} total chunks")
    
    def _scan_sequences(self, num_sequences: Optional[int] = None) -> Dict[str, str]:
        """Scan root_dir for all sequences."""
        sequences = {}
        
        for seq_dir in sorted(self.root_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            
            seq_name = seq_dir.name
            
            # Check for Data_easy and Data_hard
            for difficulty in ['Data_easy', 'Data_hard']:
                diff_dir = seq_dir / difficulty
                if not diff_dir.exists():
                    continue
                
                # Find P* subdirectories
                for subdir in sorted(diff_dir.iterdir()):
                    if subdir.is_dir() and subdir.name.startswith('P'):
                        key = f"{seq_name}_{difficulty}"
                        sequences[key] = str(subdir)
                        break
        
        if num_sequences is not None:
            sequences = dict(list(sequences.items())[:num_sequences])
        
        print(f"Found {len(sequences)} sequences in {self.root_dir}")
        for key, path in list(sequences.items())[:5]:
            print(f"  {key}: {path}")
        if len(sequences) > 5:
            print(f"  ... and {len(sequences) - 5} more")
        
        return sequences
    
    def _load_sequence(self, seq_name: str, seq_path: str):
        """Load a single sequence (from chunks.json or create using uniform_spatial)."""
        seq_path = Path(seq_path)
        json_path = seq_path / 'chunks.json'
        
        if json_path.exists():
            # Load from precomputed chunks.json
            data = load_chunk_assignment_json(str(json_path))
            metadata = data['metadata']
            
            image_dir = Path(metadata['image_dir'])
            image_pattern = metadata['image_pattern']
            
            for chunk in data['chunks']:
                self.chunk_list.append({
                    'seq_name': seq_name,
                    'image_dir': image_dir,
                    'image_pattern': image_pattern,
                    'frame_indices': chunk['frame_indices'],
                    'poses': torch.tensor([f['pose'] for f in chunk['frame_info']], dtype=torch.float32),
                })
        else:
            # Use uniform_spatial strategy to create chunks
            pose_file = seq_path / 'pose_lcam_front.txt'
            image_dir = seq_path / 'image_lcam_custom0_equirect'
            
            if not pose_file.exists():
                raise FileNotFoundError(f"Pose file not found: {pose_file}")
            if not image_dir.exists():
                raise FileNotFoundError(f"Image directory not found: {image_dir}")
            
            # Load poses
            poses = load_tartanair_poses(str(pose_file))
            num_frames = len(poses)
            
            # Check image files
            image_files = sorted(image_dir.glob('*.png'))
            if not image_files:
                image_files = sorted(image_dir.glob('*.jpg'))
            
            if len(image_files) < num_frames:
                num_frames = len(image_files)
                poses = poses[:num_frames]
            
            # Use uniform_spatial sampling (default: ~10% of frames or num_frames//4)
            num_keyframes = min(num_frames // 4, int(num_frames * 0.1))
            num_keyframes = max(num_keyframes, self.chunk_size * 2)  # Ensure enough keyframes
            
            _, keyframe_indices = uniform_spatial_sampling(poses, num_keyframes)
            
            # Create chunks from keyframes
            chunks = create_chunks_from_keyframes(
                keyframe_indices, self.chunk_size, self.chunk_stride
            )
            
            image_pattern = "{index:06d}_lcam_image_custom0_equirect.png"
            
            for frame_indices in chunks:
                self.chunk_list.append({
                    'seq_name': seq_name,
                    'image_dir': image_dir,
                    'image_pattern': image_pattern,
                    'frame_indices': frame_indices,
                    'poses': poses[frame_indices],
                })
    
    def __len__(self) -> int:
        return len(self.chunk_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a chunk of frames on-the-fly."""
        from PIL import Image
        
        chunk = self.chunk_list[idx]
        images = []
        
        for frame_idx in chunk['frame_indices']:
            img_file = chunk['image_dir'] / chunk['image_pattern'].format(index=frame_idx)
            img = Image.open(img_file).convert('RGB')
            img = img.resize((self.image_size[1], self.image_size[0]))  # PIL uses (W, H)
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
            images.append(img)
        
        return {
            'images': torch.stack(images),  # [chunk_size, 3, H, W]
            'poses': chunk['poses'],  # [chunk_size, 4, 4]
            'indices': torch.tensor(chunk['frame_indices'], dtype=torch.long),
            'seq_name': chunk['seq_name'],
        }


def load_tartanair_sequences(
    base_dir: str,
    split: str = 'train',
    num_sequences: Optional[int] = None,
) -> Dict[str, str]:
    """
    Scan TartanAir360 dataset directory and return sequence paths.
    
    Note: This is a standalone helper function. For most use cases, you can
    directly use TartanAir360Dataset(root_dir) which handles scanning internally.
    
    Each sequence's easy and hard are treated as separate sequences.
    
    Args:
        base_dir: Base directory containing sequences (e.g., /path/to/tartanair360/)
        split: 'train' or 'val' or 'all' (currently not used)
        num_sequences: Limit to first N sequences (for debugging)
    
    Returns:
        Dict mapping sequence_name to chunk directory path
        e.g., {'AbandonedCable_Data_easy': '/path/to/AbandonedCable/Data_easy/P000'}
    
    Example:
        >>> sequences = load_tartanair_sequences('/homes/shaun/main/dataset/tartanair360/')
        >>> print(sequences)
        {'AbandonedCable_Data_easy': '/path/to/AbandonedCable/Data_easy/P000', ...}
    """
    base_dir = Path(base_dir)
    sequences = {}
    
    # Find all sequence directories
    for seq_dir in sorted(base_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        
        seq_name = seq_dir.name
        
        # Check for Data_easy and Data_hard
        for difficulty in ['Data_easy', 'Data_hard']:
            diff_dir = seq_dir / difficulty
            if not diff_dir.exists():
                continue
            
            # Find P* subdirectories
            for subdir in sorted(diff_dir.iterdir()):
                if subdir.is_dir() and subdir.name.startswith('P'):
                    key = f"{seq_name}_{difficulty}"
                    sequences[key] = str(subdir)
                    break  # Only use first P* subdirectory
    
    if num_sequences is not None:
        sequences = dict(list(sequences.items())[:num_sequences])
    
    print(f"Found {len(sequences)} sequences in {base_dir}")
    for key, path in list(sequences.items())[:5]:
        print(f"  {key}: {path}")
    if len(sequences) > 5:
        print(f"  ... and {len(sequences) - 5} more")
    
    return sequences


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


def precompute_chunks_for_sequence(
    pose_file: str,
    chunk_size: int = 4,
    chunk_stride: Optional[int] = None,
    strategy: str = 'consecutive',
    num_keyframes: Optional[int] = None,
    min_translation: float = 0.5,
    save_path: Optional[str] = None,
) -> List[List[int]]:
    """
    Precompute chunks for a sequence without loading images.
    
    Args:
        pose_file: Path to pose file
        chunk_size: Frames per chunk
        chunk_stride: Stride between chunks
        strategy: 'consecutive', 'keyframe_uniform', 'keyframe_distance'
        num_keyframes: For keyframe_uniform
        min_translation: For keyframe_distance
        save_path: Optional path to save chunks as JSON
    
    Returns:
        chunks: List of chunk index lists
    """
    # Load poses only (fast)
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            if len(values) == 7:
                tx, ty, tz, qx, qy, qz, qw = values
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
                pose = np.eye(4)
                pose[:3, :] = np.array(values).reshape(3, 4)
                poses.append(pose)
    
    poses = torch.from_numpy(np.array(poses)).float()
    
    # Create chunks based on strategy
    if strategy == 'consecutive':
        stride = chunk_stride or chunk_size
        chunks = []
        for start_idx in range(0, len(poses) - chunk_size + 1, stride):
            chunk = list(range(start_idx, start_idx + chunk_size))
            chunks.append(chunk)
    
    elif strategy == 'keyframe_uniform':
        num_kf = num_keyframes or min(len(poses) // chunk_size, int(len(poses) * 0.1))  # Limit to 10% of frames or 1/4 of total, whichever is smaller
        _, keyframe_indices = uniform_spatial_sampling(poses, num_kf)
        stride = chunk_stride or chunk_size
        chunks = create_chunks_from_keyframes(keyframe_indices, chunk_size, stride)
    
    elif strategy == 'keyframe_distance':
        _, keyframe_indices = translation_distance_sampling(poses, min_translation)
        stride = chunk_stride or chunk_size
        chunks = create_chunks_from_keyframes(keyframe_indices, chunk_size, stride)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"Precomputed {len(chunks)} chunks for {pose_file}")
    print(f"  Strategy: {strategy}, Chunk size: {chunk_size}, Stride: {chunk_stride or chunk_size}")
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump({
                'chunks': chunks,
                'strategy': strategy,
                'chunk_size': chunk_size,
                'chunk_stride': chunk_stride,
                'num_frames': len(poses),
            }, f, indent=2)
        print(f"Saved chunks to {save_path}")
    
    return chunks


# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='dummy', choices=['dummy', 'keyframe'])
    parser.add_argument('--pose_file', type=str, default=None)
    args = parser.parse_args()
    
    if args.test == 'dummy':
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
        
        # Test different strategies
        for strategy in ['consecutive', 'keyframe_uniform']:
            print(f"\n--- Testing {strategy} ---")
            dataset = SequenceChunkDataset(
                images=images,
                poses=poses,
                chunk_size=4,
                chunk_stride=2 if strategy == 'consecutive' else None,
                mode='train',
                chunks=None,
            )
            print(f"Chunks: {dataset.chunks}")
    
    elif args.test == 'keyframe' and args.pose_file:
        # Test keyframe selection with visualization
        from CelestialSplat.utils.keyframe import test_keyframe_selection
        
        result = test_keyframe_selection(
            args.pose_file,
            strategy='uniform_spatial',
            num_keyframes=32,
            chunk_size=4,
            stride=2,  # 50% overlap
        )
