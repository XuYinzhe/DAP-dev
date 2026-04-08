"""
Geometry utilities for ERP (Equirectangular) 360° images.

Provides coordinate conversions between:
- UV (pixel coordinates)
- LonLat (longitude, latitude in radians)
- XYZ (3D Cartesian coordinates)
"""

import torch
import numpy as np
from typing import Tuple, Optional


class OmniGeometry:
    """
    Utility class for 360° omnidirectional image geometry.
    Handles coordinate conversions between UV, LonLat, and XYZ.
    """
    
    def __init__(self, img_w: int = 1024, img_h: int = 512, device: str = 'cuda'):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        
        # Image dimensions
        self.img_w = img_w
        self.img_h = img_h
        
        # Focal lengths for ERP projection
        # lon = (u + 0.5 - cx) / fx
        # lat = (v + 0.5 - cy) / fy
        self.fx = img_w / (2 * np.pi)
        self.fy = -img_h / np.pi  # Negative because v increases downward
        self.cx = img_w / 2
        self.cy = img_h / 2
        
        # Precompute direction grid (xyz for each pixel)
        self._init_direction_grid()
    
    def _init_direction_grid(self):
        """Initialize direction grid for all pixels."""
        u = torch.arange(self.img_w, device=self.device, dtype=torch.float32)
        v = torch.arange(self.img_h, device=self.device, dtype=torch.float32)
        u, v = torch.meshgrid(u, v, indexing='xy')
        
        # UV to XYZ
        x, y, z = self.uv2xyz(u, v)
        self.direction_grid = torch.stack([x, y, z], dim=-1)  # [H, W, 3]
    
    def uv2lonlat(self, u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert pixel coordinates (u, v) to longitude and latitude (radians).
        
        Args:
            u: pixel column index [0, W-1]
            v: pixel row index [0, H-1]
        
        Returns:
            lon: longitude in range [-π, π]
            lat: latitude in range [-π/2, π/2] (0 at equator)
        """
        lon = ((u + 0.5) - self.cx) / self.fx
        lat = ((v + 0.5) - self.cy) / self.fy
        return lon, lat
    
    def lonlat2xyz(self, lon: torch.Tensor, lat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert longitude and latitude to 3D Cartesian coordinates on unit sphere.
        
        Args:
            lon: longitude in radians
            lat: latitude in radians
        
        Returns:
            x, y, z: unit vectors (pointing outward from sphere center)
            
        Convention:
            - x: right
            - y: up  
            - z: forward (pointing to lon=0, lat=0)
        """
        cos_lat = torch.cos(lat)
        x = cos_lat * torch.sin(lon)
        y = torch.sin(-lat)  # Negate because positive lat should be up
        z = cos_lat * torch.cos(lon)
        return x, y, z
    
    def uv2xyz(self, u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert pixel coordinates to 3D Cartesian coordinates on unit sphere."""
        lon, lat = self.uv2lonlat(u, v)
        return self.lonlat2xyz(lon, lat)
    
    def xyz2lonlat(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert 3D Cartesian coordinates to longitude and latitude.
        
        Returns:
            lon: longitude in range [-π, π]
            lat: latitude in range [-π/2, π/2]
        """
        lon = torch.atan2(x, z)
        lat = torch.atan2(-y, torch.sqrt(x**2 + z**2))
        return lon, lat
    
    def lonlat2uv(self, lon: torch.Tensor, lat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert longitude and latitude to pixel coordinates."""
        u = lon * self.fx + self.cx - 0.5
        v = lat * self.fy + self.cy - 0.5
        return u, v
    
    def xyz2uv(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert 3D Cartesian coordinates to pixel coordinates."""
        lon, lat = self.xyz2lonlat(x, y, z)
        return self.lonlat2uv(lon, lat)
    
    def depth_to_3d(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Convert ERP depth map to 3D points in camera space.
        
        Args:
            depth: [..., H, W] depth values in metric units
        
        Returns:
            xyz: [..., 3, H, W] 3D coordinates
        """
        # Get direction grid with proper batching
        shape = depth.shape
        h, w = shape[-2:]
        
        # direction_grid is [H, W, 3], expand to match depth
        directions = self.direction_grid[:h, :w]  # [H, W, 3]
        
        # Expand dimensions to match depth
        for _ in range(len(shape) - 2):
            directions = directions.unsqueeze(0)
        
        # depth[..., None, H, W] * directions[..., H, W, 3] -> xyz[..., H, W, 3]
        xyz = depth[..., None] * directions  # [..., H, W, 3]
        
        # Convert to [..., 3, H, W]
        xyz = xyz.permute(*range(len(shape) - 2), -1, *range(len(shape) - 2, len(shape)))
        
        return xyz
    
    def depth_to_3d_downsampled(self, depth: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        Convert depth to 3D points at downsampled resolution.
        
        Args:
            depth: [..., H, W] depth at original resolution
            target_h, target_w: target resolution
        
        Returns:
            xyz: [..., 3, target_h, target_w]
        """
        # Downsample depth
        depth_ds = torch.nn.functional.interpolate(
            depth.flatten(0, -3) if depth.dim() > 2 else depth.unsqueeze(0),
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
        if depth.dim() > 2:
            shape = depth.shape
            depth_ds = depth_ds.view(*shape[:-2], target_h, target_w)
        else:
            depth_ds = depth_ds.squeeze(0)
        
        # Create geometry for downsampled resolution
        temp_geo = OmniGeometry(target_w, target_h, str(self.device))
        return temp_geo.depth_to_3d(depth_ds)


def batch_depth_to_3d(depth: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
    """
    Convert batched ERP depth maps to 3D points.
    
    Args:
        depth: [B, N, H, W] or [B*N, H, W] depth maps
        img_h, img_w: image dimensions
    
    Returns:
        xyz: [B, N, 3, H, W] or [B*N, 3, H, W] 3D coordinates
    """
    geo = OmniGeometry(img_w, img_h, str(depth.device))
    return geo.depth_to_3d(depth)


def transform_points(points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """
    Transform 3D points using pose matrix.
    
    Args:
        points: [..., 3] or [..., 3, H, W] 3D points
        pose: [4, 4] or [B, 4, 4] transformation matrix (world-to-camera or camera-to-world)
    
    Returns:
        transformed_points: same shape as input
    """
    original_shape = points.shape
    
    # Handle different input shapes
    if points.dim() == 3 and points.shape[-2] == 3:  # [..., 3, H, W]
        b = points.shape[0] if points.dim() > 3 else 1
        h, w = points.shape[-2:]
        points_flat = points.reshape(-1, 3, h * w).permute(0, 2, 1)  # [B, H*W, 3]
    elif points.dim() >= 2 and points.shape[-1] == 3:  # [..., 3]
        points_flat = points.reshape(-1, 3)
    else:
        raise ValueError(f"Unexpected points shape: {points.shape}")
    
    # Handle pose
    if pose.dim() == 2:  # [4, 4]
        R = pose[:3, :3]
        t = pose[:3, 3]
    elif pose.dim() == 3:  # [B, 4, 4]
        R = pose[:, :3, :3]
        t = pose[:, :3, 3]
    else:
        raise ValueError(f"Unexpected pose shape: {pose.shape}")
    
    # Transform: X' = R @ X + t
    if R.dim() == 2:
        transformed = torch.matmul(points_flat, R.T) + t
    else:
        # Batch matmul
        transformed = torch.matmul(points_flat, R.transpose(-2, -1)) + t.unsqueeze(-2)
    
    # Reshape back
    if len(original_shape) >= 3 and original_shape[-2] == 3:
        # [..., 3, H, W]
        transformed = transformed.permute(0, 2, 1).reshape(original_shape)
    else:
        transformed = transformed.reshape(original_shape)
    
    return transformed
