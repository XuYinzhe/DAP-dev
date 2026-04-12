"""
CelestialSplat Model Architecture

Main model file containing:
- DAPFeatureAdapter: Aggregates DINOv3 4-layer features
- CrossViewTransformer: Geometry-guided cross-view attention
- GSDecoder: Gaussian parameter decoder
- CelestialSplat: Main model integrating all components
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


def _init_weights(m):
    """Initialize weights for linear and conv layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


@dataclass
@dataclass
class FusionConfig:
    """Configuration for Gaussian fusion strategy."""
    strategy: str = 'simple'  # 'simple', 'voxel', or custom
    # Simple strategy params
    conf_thresh: float = 0.1
    opacity_thresh: float = 0.01
    # Voxel strategy params
    voxel_size: float = 0.1
    max_gs_per_voxel: int = 2
    conf_weight: float = 0.7
    opacity_weight: float = 0.3
    # Future: additional strategy-specific params


@dataclass
class CelestialSplatConfig:
    """Configuration for CelestialSplat model."""
    # DAP backbone config
    encoder: str = 'vitl'  # 'vits', 'vitb', 'vitl', 'vitg'
    features: int = 256
    out_channels: Tuple[int, ...] = (256, 512, 1024, 1024)
    use_bn: bool = False
    use_clstoken: bool = False
    max_depth: float = 10.0
    
    # Feature adapter config
    dino_embed_dim: int = 1024  # 768 for vitb, 1024 for vitl
    adapter_out_dim: int = 256
    
    # Cross-view transformer config
    transformer_dim: int = 256
    num_transformer_layers: int = 4
    num_heads: int = 8
    K_neighbors: int = 3  # Number of neighbor views for cross-attention
    
    # GS Decoder config
    decoder_hidden_dim: int = 128
    sh_degree: int = 2  # Spherical harmonics degree (2 -> 27 channels)
    
    # Gaussian Fusion config
    fusion = FusionConfig()
    
    # ERP image config
    image_height: int = 512
    image_width: int = 1024


class DAPFeatureAdapter(nn.Module):
    """
    Adapter for DAP (DINOv3) features to CelestialSplat.
    
    Aggregates 4-layer DINOv3 features into a unified representation.
    
    Input: List of 4 tensors from DINOv3 intermediate layers
           Each: [B, embed_dim, H/16, W/16]
    Output: [B, out_dim, H/16, W/16]
    """
    
    def __init__(self, in_dim: int = 1024, out_dim: int = 256, num_layers: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = out_dim // num_layers  # 64 if out_dim=256
        
        # Project each layer feature to hidden_dim
        self.patch_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, self.hidden_dim, 1, bias=False),
                nn.BatchNorm2d(self.hidden_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_layers)
        ])
        
        # Optional: cls token projection for global modulation
        self.cls_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True)
        )
        
        self.apply(_init_weights)
    
    def forward(self, features: List[torch.Tensor], cls_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: List of 4 tensors [B, in_dim, H/16, W/16]
            cls_tokens: Optional [B, num_layers, in_dim]
        Returns:
            [B, out_dim, H/16, W/16]
        """
        # Project and concatenate patch features
        outs = [proj(f) for proj, f in zip(self.patch_projs, features)]
        local_feat = torch.cat(outs, dim=1)  # [B, out_dim, H/16, W/16]
        
        # Global feature modulation
        if cls_tokens is not None:
            global_feat = self.cls_proj(cls_tokens.mean(dim=1))  # [B, out_dim]
            local_feat = local_feat + global_feat[:, :, None, None]
        
        return local_feat


class GeometryGuidedCrossAttention(nn.Module):
    """
    Geometry-guided cross-view attention module.
    
    Simplified implementation: For each query view, aggregate from neighbor views
    using standard multi-head attention with geometric weighting.
    """
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Geometry confidence MLP
        self.geo_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.apply(_init_weights)
    
    def forward(
        self, 
        query: torch.Tensor,  # [B, 1, L, C]
        key: torch.Tensor,    # [B, K, L, C]
        value: torch.Tensor,  # [B, K, L, C]
        pts3d_q: torch.Tensor, # [B, 1, L, 3]
        pts3d_k: torch.Tensor, # [B, K, L, 3]
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [B, 1, L, C] - single query view
            key: [B, K, L, C] - K neighbor views
            value: [B, K, L, C]
            pts3d_q: [B, 1, L, 3] - 3D points for query view
            pts3d_k: [B, K, L, 3] - 3D points for key views
            mask: Optional attention mask
        Returns:
            [B, L, C]
        """
        B, _, L, C = query.shape
        K = key.shape[1]
        
        # # Compute per-point geometric confidence based on 3D distance
        # # Average distance between query points and key points
        # geo_dist = torch.cdist(
        #     pts3d_q.squeeze(1),  # [B, L, 3]
        #     pts3d_k.transpose(1, 2).reshape(B, L * K, 3)  # [B, K*L, 3]
        # )  # [B, L, K*L]
        # geo_dist = geo_dist.mean(dim=-1, keepdim=True)  # [B, L, 1]
        # geo_weight = self.geo_mlp(geo_dist.expand(-1, -1, 3))  # [B, L, 1]
        
        # Project queries, keys, values
        # Query: [B, 1, L, C] -> [B, L, C]
        q = self.q_proj(query.squeeze(1))  # [B, L, C]
        
        # Key/Value: [B, K, L, C] -> [B, K*L, C]
        k = self.k_proj(key.reshape(B, K * L, C))  # [B, K*L, C]
        v = self.v_proj(value.reshape(B, K * L, C))  # [B, K*L, C]
        
        # Multi-head attention
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, L, head_dim]
        k = k.reshape(B, K * L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, K*L, head_dim]
        v = v.reshape(B, K * L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, K*L, head_dim]
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, heads, L, K*L]
        
        # # Apply geometric weighting
        # geo_weight_expanded = geo_weight.unsqueeze(1).expand(-1, self.num_heads, -1, K * L)  # [B, heads, L, K*L]
        # attn = attn * geo_weight_expanded
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, heads, L, head_dim]
        
        # Reshape back
        out = out.transpose(1, 2).reshape(B, L, C)
        out = self.out_proj(out)
        
        return out


class CrossViewTransformerLayer(nn.Module):
    """Single layer of cross-view transformer."""
    
    def __init__(self, dim: int, num_heads: int = 8, K_neighbors: int = 4):
        super().__init__()
        self.K = K_neighbors
        
        # Self-attention (within each view)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        
        # Cross-view attention
        self.cross_attn = GeometryGuidedCrossAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm3 = nn.LayerNorm(dim)
        
        self.apply(_init_weights)
    
    def forward(
        self, 
        tokens: torch.Tensor,      # [B, N, L, C]
        pts3d_cam: torch.Tensor,   # [B, N, L, 3]
        poses: torch.Tensor        # [B, N, 4, 4]
    ) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, L, C] where L = H*W
            pts3d_cam: [B, N, L, 3] - 3D points in camera coordinates
            poses: [B, N, 4, 4] - camera poses (world-to-camera)
        Returns:
            [B, N, L, C]
        """
        B, N, L, C = tokens.shape
        
        # 1. Self-Attention (within each view)
        tokens_flat = tokens.reshape(B * N, L, C)
        attn_out, _ = self.self_attn(tokens_flat, tokens_flat, tokens_flat)
        tokens_flat = self.norm1(tokens_flat + attn_out)
        tokens = tokens_flat.reshape(B, N, L, C)
        
        # 2. Cross-View Attention
        # Transform points to world coordinates
        pts3d_world = self._cam_to_world(pts3d_cam, poses)  # [B, N, L, 3]
        
        # For each view, aggregate from other views
        cross_out = []
        for i in range(N):
            # Sample features from neighboring views
            neighbor_features = []
            neighbor_pts3d = []
            
            for j in range(N):
                if i == j:
                    continue
                # Project world points of view i to view j
                pts_in_j = self._world_to_cam(pts3d_world[:, i], poses[:, j])
                neighbor_features.append(tokens[:, j])
                neighbor_pts3d.append(pts_in_j)
            
            if neighbor_features:
                # Stack neighbor features and points
                kv_features = torch.stack(neighbor_features, dim=1)  # [B, K, L, C]
                kv_pts3d = torch.stack(neighbor_pts3d, dim=1)        # [B, K, L, 3]
                
                q = tokens[:, i:i+1]           # [B, 1, L, C]
                pts_q = pts3d_world[:, i:i+1]  # [B, 1, L, 3]
                
                # Cross-attention
                attn_out = self.cross_attn(q, kv_features, kv_features, pts_q, kv_pts3d)
                cross_out.append(self.norm2(tokens[:, i] + attn_out.squeeze(1)))
            else:
                cross_out.append(tokens[:, i])
        
        tokens = torch.stack(cross_out, dim=1)  # [B, N, L, C]
        
        # 3. FFN
        tokens = self.norm3(tokens + self.ffn(tokens))
        
        return tokens
    
    def _cam_to_world(self, pts3d_cam: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
        """Transform points from camera to world coordinates."""
        # poses: [B, N, 4, 4] (world-to-camera)
        # We need cam-to-world: inv(poses)
        B, N, L, _ = pts3d_cam.shape
        
        R = poses[:, :, :3, :3]  # [B, N, 3, 3]
        t = poses[:, :, :3, 3]   # [B, N, 3]
        
        # X_world = R^T @ (X_cam - t) = R^T @ X_cam - R^T @ t
        # But we store world-to-cam: X_cam = R @ X_world + t
        # So X_world = R^T @ (X_cam - t)
        pts_world = torch.einsum('bnij,bnlj->bnli', R.transpose(-2, -1), pts3d_cam - t.unsqueeze(2))
        return pts_world
    
    def _world_to_cam(self, pts3d_world: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """Transform points from world to camera coordinates.
        
        Args:
            pts3d_world: [B, L, 3] or [B, 1, L, 3]
            pose: [B, 4, 4] single camera pose
        Returns:
            [B, L, 3]
        """
        if pts3d_world.dim() == 4:
            pts3d_world = pts3d_world.squeeze(1)  # [B, L, 3]
        
        R = pose[:, :3, :3]  # [B, 3, 3]
        t = pose[:, :3, 3]   # [B, 3]
        
        # X_cam = R @ X_world + t
        pts_cam = torch.einsum('bij,bkj->bki', R, pts3d_world) + t.unsqueeze(1)
        return pts_cam


class CrossViewTransformer(nn.Module):
    """
    Cross-view transformer for multi-view feature fusion.
    
    Uses geometry-guided attention to aggregate features across views.
    """
    
    def __init__(self, dim: int = 256, num_layers: int = 6, num_heads: int = 8, K_neighbors: int = 4):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            CrossViewTransformerLayer(dim, num_heads, K_neighbors)
            for _ in range(num_layers)
        ])
        
        self.apply(_init_weights)
    
    def forward(
        self,
        features: torch.Tensor,   # [B, N, C, H, W]
        depths: torch.Tensor,     # [B, N, 1, H_full, W_full]
        poses: torch.Tensor,      # [B, N, 4, 4]
        intrinsics: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            features: [B, N, C, H, W] at 1/16 resolution
            depths: [B, N, 1, H_full, W_full] at full resolution
            poses: [B, N, 4, 4] camera poses
            intrinsics: Optional camera intrinsics (not used for ERP)
        Returns:
            [B, N, C, H, W]
        """
        B, N, C, H, W = features.shape
        
        # Downsample depths to feature resolution
        depths_low = F.interpolate(
            depths.flatten(0, 1),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).view(B, N, 1, H, W)
        
        # Convert depth to 3D points (ERP projection)
        pts3d_cam = self._depth_to_3d_erp(depths_low)  # [B, N, 3, H, W]
        pts3d_cam = pts3d_cam.permute(0, 1, 3, 4, 2).reshape(B, N, H * W, 3)  # [B, N, L, 3]
        
        # Flatten features to tokens
        tokens = features.flatten(3).permute(0, 1, 3, 2)  # [B, N, L, C]
        
        # Process through transformer layers
        for layer in self.layers:
            tokens = layer(tokens, pts3d_cam, poses)
        
        # Unflatten back to spatial
        fused = tokens.permute(0, 1, 3, 2).reshape(B, N, C, H, W)
        return fused
    
    def _depth_to_3d_erp(self, depth: torch.Tensor, img_h: int = None, img_w: int = None) -> torch.Tensor:
        """
        Convert ERP depth map to 3D points in Cartesian coordinates.
        
        Uses standard ERP projection:
        - lon (longitude) = ((u + 0.5) - cx) / fx, where cx = W/2, fx = W/(2π)
        - lat (latitude) = ((v + 0.5) - cy) / fy, where cy = H/2, fy = -H/π
        - x = cos(lat) * sin(lon)
        - y = sin(-lat)  # negative because v increases downward
        - z = cos(lat) * cos(lon)
        
        Args:
            depth: [B, N, 1, H, W] or [..., H, W]
            img_h, img_w: Original image dimensions (if different from depth shape)
        Returns:
            [B, N, 3, H, W] or [..., 3, H, W]
        """
        input_shape = depth.shape
        if depth.dim() == 5:
            B, N, _, H, W = depth.shape
        else:
            H, W = depth.shape[-2:]
            B, N = 1, 1
            depth = depth.reshape(1, 1, 1, H, W)
        
        device = depth.device
        
        # Use provided image dimensions or infer from depth
        img_h = img_h if img_h is not None else H * (512 // H)  # assume 16x downsampling
        img_w = img_w if img_w is not None else W * (1024 // W)
        
        # ERP focal lengths
        fx = img_w / (2 * math.pi)
        fy = -img_h / math.pi  # negative because v increases downward
        cx = img_w / 2
        cy = img_h / 2
        
        # Create pixel coordinate grid at feature resolution
        u = torch.arange(W, device=device, dtype=torch.float32)
        v = torch.arange(H, device=device, dtype=torch.float32)
        u, v = torch.meshgrid(u, v, indexing='xy')
        
        # UV to LonLat
        lon = ((u + 0.5) - cx) / fx  # range: [-π, π]
        lat = ((v + 0.5) - cy) / fy  # range: [-π/2, π/2]
        
        # LonLat to XYZ (unit sphere)
        cos_lat = torch.cos(lat)
        x = cos_lat * torch.sin(lon)
        y = torch.sin(-lat)  # flip y because positive lat should be up
        z = cos_lat * torch.cos(lon)  # Forward is -Z (TartanAir convention)
        
        # Stack directions [H, W, 3]
        directions = torch.stack([x, y, z], dim=-1)
        
        # Scale by depth and reshape
        # depth: [B, N, 1, H, W], directions: [H, W, 3]
        pts3d = depth.squeeze(2).unsqueeze(-1) * directions  # [B, N, H, W, 3]
        pts3d = pts3d.permute(0, 1, 4, 2, 3)  # [B, N, 3, H, W]
        
        # Restore original shape if needed
        if len(input_shape) != 5:
            pts3d = pts3d.reshape(*input_shape[:-2], 3, H, W)
        
        return pts3d


class GSDecoder(nn.Module):
    """
    Gaussian Splatting parameter decoder.
    
    Predicts Gaussian parameters from fused features:
    - Depth residual (Δr)
    - Covariance (σ_x, σ_y, σ_z)
    - Rotation (quaternion)
    - Opacity
    - SH color coefficients
    - Confidence
    """
    
    def __init__(self, in_dim: int = 256, hidden_dim: int = 128, sh_degree: int = 2):
        super().__init__()
        self.sh_degree = sh_degree
        self.sh_channels = (sh_degree + 1) ** 2 * 3  # 27 for degree 2 (9 coeffs * 3 RGB)
        
        # Upsampling blocks
        self.up1 = self._make_up_block(in_dim, hidden_dim)           # 32 -> 64
        self.up2 = self._make_up_block(hidden_dim, hidden_dim // 2)   # 64 -> 128
        self.up3 = self._make_up_block(hidden_dim // 2, hidden_dim // 4)  # 128 -> 256
        self.up4 = self._make_up_block(hidden_dim // 4, hidden_dim // 4)  # 256 -> 512
        
        # Prediction heads
        self.head_depth = nn.Conv2d(hidden_dim // 4, 1, 1)      # Δr
        self.head_cov = nn.Conv2d(hidden_dim // 4, 3, 1)        # (σ_x, σ_y, σ_z)
        self.head_rot = nn.Conv2d(hidden_dim // 4, 4, 1)        # quaternion
        self.head_opacity = nn.Conv2d(hidden_dim // 4, 1, 1)    # α
        self.head_sh = nn.Conv2d(hidden_dim // 4, self.sh_channels, 1)  # SH coeffs
        self.head_conf = nn.Conv2d(hidden_dim // 4, 1, 1)       # confidence
        
        self.apply(_init_weights)
    
    def _make_up_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        fused_feat: torch.Tensor,   # [B, N, C, H, W]
        depth_prior: torch.Tensor,  # [B, N, H_full, W_full]
        mask: torch.Tensor          # [B, N, H_full, W_full]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused_feat: [B, N, C, H, W] at 1/16 resolution
            depth_prior: [B, N, H_full, W_full] from DAP
            mask: [B, N, H_full, W_full] from DAP
        Returns:
            Dict with Gaussian parameters
        """
        B, N = fused_feat.shape[:2]
        x = fused_feat.flatten(0, 1)  # [B*N, C, H, W]
        
        # Upsampling
        x = self.up1(x)   # [B*N, 128, 64, 128]
        x = self.up2(x)   # [B*N, 64, 128, 256]
        x = self.up3(x)   # [B*N, 32, 256, 512]
        x = self.up4(x)   # [B*N, 32, 512, 1024]
        
        # Predictions
        delta_d = self.head_depth(x)    # [B*N, 1, H, W]
        cov = self.head_cov(x)          # [B*N, 3, H, W]
        rot = self.head_rot(x)          # [B*N, 4, H, W]
        opacity = torch.sigmoid(self.head_opacity(x))  # [B*N, 1, H, W]
        sh = self.head_sh(x)            # [B*N, 27, H, W]
        conf = torch.sigmoid(self.head_conf(x))        # [B*N, 1, H, W]
        
        # Apply mask (sky regions should have zero depth)
        mask_expanded = mask.flatten(0, 1)[:, None, :, :]  # [B*N, 1, H, W]
        
        # Depth residual (DAP prior + predicted delta)
        depth = depth_prior.flatten(0, 1)[:, None, :, :] + delta_d
        depth = depth * (1 - mask_expanded)  # Zero depth in sky regions
        
        # Normalize rotation quaternion
        rot = F.normalize(rot, dim=1)
        
        # Restore batch and view dimensions
        def restore(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, N, *t.shape[1:])
        
        return {
            'depth': restore(depth).squeeze(2),         # [B, N, H, W]
            'covariance': restore(cov),                  # [B, N, 3, H, W]
            'rotation': restore(rot),                    # [B, N, 4, H, W]
            'opacity': restore(opacity).squeeze(2),     # [B, N, H, W]
            'sh_color': restore(sh),                     # [B, N, 27, H, W]
            'confidence': restore(conf).squeeze(2),     # [B, N, H, W]
            'delta_depth': restore(delta_d).squeeze(2)  # For monitoring
        }


class GaussianFusionStrategy:
    """
    Base class for Gaussian fusion strategies.
    
    This allows easy swapping and experimentation with different fusion methods:
    - SimpleFusion: Original flatten + threshold
    - VoxelDeduplication: Voxel-based deduplication to reduce redundancy
    - Future: Learned compression, clustering-based, etc.
    """
    
    def __call__(
        self,
        means: torch.Tensor,          # [B, P, 3]
        scales: torch.Tensor,         # [B, P, 3]
        rotations: torch.Tensor,      # [B, P, 4]
        opacities: torch.Tensor,      # [B, P, 1]
        shs: torch.Tensor,            # [B, P, K, 3]
        confidences: torch.Tensor,    # [B, P, 1]
        masks: torch.Tensor,          # [B, P]
        metadata: Dict,               # Extra info (num_views, etc.)
    ) -> Dict[str, torch.Tensor]:
        """Apply fusion strategy."""
        raise NotImplementedError


class SimpleFusion(GaussianFusionStrategy):
    """
    Simple fusion: just flatten and threshold.
    This is the original behavior.
    """
    
    def __call__(self, means, scales, rotations, opacities, shs, confidences, masks, metadata):
        return {
            'means': means,
            'scales': scales,
            'rotations': rotations,
            'opacities': opacities,
            'shs': shs,
            'confidences': confidences,
            'masks': masks,
            'num_per_view': metadata.get('num_per_view', means.shape[1]),
            'num_views': metadata.get('num_views', 1),
        }


class VoxelDeduplicationFusion(GaussianFusionStrategy):
    """
    Voxel-based deduplication fusion.
    
    Divides space into voxels and keeps only top-k Gaussians per voxel
    based on confidence scores. This reduces redundant Gaussians in
    overlapping view regions.
    
    Args:
        voxel_size: Size of each voxel in world coordinates
        max_gs_per_voxel: Maximum number of Gaussians to keep per voxel
        conf_weight: Weight for confidence in scoring (0-1)
        opacity_weight: Weight for opacity in scoring (0-1)
    """
    
    def __init__(
        self,
        voxel_size: float = 0.1,
        max_gs_per_voxel: int = 2,
        conf_weight: float = 0.7,
        opacity_weight: float = 0.3,
    ):
        self.voxel_size = voxel_size
        self.max_gs_per_voxel = max_gs_per_voxel
        self.conf_weight = conf_weight
        self.opacity_weight = opacity_weight
    
    def __call__(self, means, scales, rotations, opacities, shs, confidences, masks, metadata):
        B, P, _ = means.shape
        device = means.device
        
        # Compute scores for each Gaussian
        scores = (
            self.conf_weight * confidences +
            self.opacity_weight * opacities
        ).squeeze(-1)  # [B, P]
        
        # Assign each Gaussian to a voxel
        voxel_coords = torch.floor(means / self.voxel_size).long()  # [B, P, 3]
        
        # Create unique voxel IDs for each batch
        # Hash: (x, y, z) -> single int, with batch offset
        batch_size = 100000  # Large enough offset
        voxel_ids = (
            voxel_coords[..., 0] +
            voxel_coords[..., 1] * batch_size +
            voxel_coords[..., 2] * batch_size * batch_size +
            torch.arange(B, device=device).view(B, 1) * batch_size**3
        )  # [B, P]
        
        # Process each batch separately
        all_outputs = []
        for b in range(B):
            valid_mask = masks[b]  # [P]
            valid_indices = torch.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                # No valid Gaussians, return empty
                empty = torch.zeros(0, 3, device=device)
                all_outputs.append({
                    'means': empty,
                    'scales': empty,
                    'rotations': torch.zeros(0, 4, device=device),
                    'opacities': torch.zeros(0, 1, device=device),
                    'shs': torch.zeros(0, *shs.shape[2:], device=device),
                    'confidences': torch.zeros(0, 1, device=device),
                    'masks': torch.zeros(0, dtype=torch.bool, device=device),
                })
                continue
            
            # Get valid Gaussians
            v_ids = voxel_ids[b, valid_indices]  # [N_valid]
            v_scores = scores[b, valid_indices]  # [N_valid]
            
            # For each voxel, keep top-k Gaussians
            unique_voxels = torch.unique(v_ids)
            selected_indices = []
            
            for v_id in unique_voxels:
                in_voxel = (v_ids == v_id)
                voxel_indices = valid_indices[in_voxel]
                voxel_scores = v_scores[in_voxel]
                
                # Sort by score and take top-k
                k = min(self.max_gs_per_voxel, len(voxel_scores))
                top_k = torch.topk(voxel_scores, k).indices
                selected_indices.append(voxel_indices[top_k])
            
            selected_indices = torch.cat(selected_indices)
            
            # Gather selected Gaussians
            all_outputs.append({
                'means': means[b, selected_indices],
                'scales': scales[b, selected_indices],
                'rotations': rotations[b, selected_indices],
                'opacities': opacities[b, selected_indices],
                'shs': shs[b, selected_indices],
                'confidences': confidences[b, selected_indices],
                'masks': torch.ones(len(selected_indices), dtype=torch.bool, device=device),
            })
        
        # Batch the outputs (note: different batch items may have different sizes)
        # For now, return list format; caller should handle variable sizes
        return all_outputs


class GaussianFusion(nn.Module):
    """
    Fuse per-view Gaussians into unified world-coordinate Gaussians.
    
    Supports multiple fusion strategies via pluggable strategy classes.
    
    Usage:
        # Simple fusion (default)
        fusion = GaussianFusion(strategy='simple')
        
        # Voxel deduplication
        fusion = GaussianFusion(
            strategy='voxel',
            voxel_size=0.1,
            max_gs_per_voxel=2
        )
    """
    
    def __init__(
        self,
        strategy: str = 'simple',
        conf_thresh: float = 0.1,
        opacity_thresh: float = 0.01,
        **strategy_kwargs
    ):
        """
        Args:
            strategy: Fusion strategy - 'simple', 'voxel', or custom strategy instance
            conf_thresh: Minimum confidence for valid Gaussians
            opacity_thresh: Minimum opacity for valid Gaussians
            **strategy_kwargs: Additional arguments for the strategy (e.g., voxel_size)
        """
        super().__init__()
        self.conf_thresh = conf_thresh
        self.opacity_thresh = opacity_thresh
        
        # Set up strategy
        if strategy == 'simple':
            self.strategy = SimpleFusion()
        elif strategy == 'voxel':
            self.strategy = VoxelDeduplicationFusion(**strategy_kwargs)
        elif isinstance(strategy, GaussianFusionStrategy):
            self.strategy = strategy
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        self.strategy_name = strategy if isinstance(strategy, str) else strategy.__class__.__name__
    
    def forward(
        self,
        per_view_gaussians: Dict[str, torch.Tensor],
        poses: torch.Tensor,
        img_h: int,
        img_w: int,
        return_raw: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            per_view_gaussians: Dict with per-view params
                - depth: [B, N, H, W]
                - covariance: [B, N, 3, H, W]
                - rotation: [B, N, 4, H, W]
                - opacity: [B, N, H, W]
                - sh_color: [B, N, C, H, W]
                - confidence: [B, N, H, W]
            poses: [B, N, 4, 4] camera poses (world-to-camera)
            img_h, img_w: Original image dimensions for ERP projection
            return_raw: If True, also return raw (unfused) Gaussians for debugging
        
        Returns:
            Unified Gaussians in world coordinates (format depends on strategy):
                - means: [B, P, 3] or List[[P_i, 3]]
                - scales: [B, P, 3] or List[[P_i, 3]]
                - rotations: [B, P, 4] or List[[P_i, 4]]
                - opacities: [B, P, 1] or List[[P_i, 1]]
                - shs: [B, P, K, 3] or List[[P_i, K, 3]]
                - confidences: [B, P, 1] or List[[P_i, 1]]
                - masks: [B, P] or List[[P_i]]
        """
        B, N, H, W = per_view_gaussians['depth'].shape
        device = per_view_gaussians['depth'].device
        
        # Step 1: Convert to world coordinates (common to all strategies)
        means_world, all_params = self._convert_to_world(
            per_view_gaussians, poses, img_h, img_w
        )
        
        # Step 2: Create validity mask
        confidences = all_params['confidences']
        opacities = all_params['opacities']
        conf_mask = confidences.squeeze(-1) > self.conf_thresh
        opacity_mask = opacities.squeeze(-1) > self.opacity_thresh
        masks = conf_mask & opacity_mask
        
        # Step 3: Apply fusion strategy
        metadata = {
            'num_per_view': H * W,
            'num_views': N,
            'img_h': img_h,
            'img_w': img_w,
        }
        
        result = self.strategy(
            means=means_world,
            scales=all_params['scales'],
            rotations=all_params['rotations'],
            opacities=opacities,
            shs=all_params['shs'],
            confidences=confidences,
            masks=masks,
            metadata=metadata,
        )
        
        # Handle variable-size outputs (for strategies like voxel)
        if isinstance(result, list):
            # Variable-size batch (each item has different number of Gaussians)
            # Return as list for now; caller needs to handle appropriately
            output = {
                'gaussians': result,  # List of dicts
                'is_variable_size': True,
                'num_gaussians_per_item': [len(r['means']) for r in result],
            }
        else:
            output = result
            output['is_variable_size'] = False
        
        output['strategy'] = self.strategy_name
        
        if return_raw:
            output['raw'] = {
                'means': means_world,
                **all_params,
                'masks': masks,
            }
        
        return output
    
    def _convert_to_world(
        self,
        per_view_gaussians: Dict[str, torch.Tensor],
        poses: torch.Tensor,
        img_h: int,
        img_w: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Convert per-view Gaussians to world coordinates."""
        B, N, H, W = per_view_gaussians['depth'].shape
        device = per_view_gaussians['depth'].device
        
        # ERP focal lengths
        fx = img_w / (2 * math.pi)
        fy = -img_h / math.pi
        cx = img_w / 2
        cy = img_h / 2
        
        # Create pixel coordinate grid
        u = torch.arange(W, device=device, dtype=torch.float32)
        v = torch.arange(H, device=device, dtype=torch.float32)
        u, v = torch.meshgrid(u, v, indexing='xy')
        
        # UV to LonLat
        lon = ((u + 0.5) - cx) / fx
        lat = ((v + 0.5) - cy) / fy
        
        # LonLat to XYZ directions
        cos_lat = torch.cos(lat)
        x_dir = cos_lat * torch.sin(lon)
        y_dir = torch.sin(-lat)
        z_dir = cos_lat * torch.cos(lon)
        directions = torch.stack([x_dir, y_dir, z_dir], dim=-1)  # [H, W, 3]
        
        # Convert depth to 3D points for each view
        all_means_cam = []
        for i in range(N):
            depth = per_view_gaussians['depth'][:, i]  # [B, H, W]
            pts_cam = depth.unsqueeze(-1) * directions  # [B, H, W, 3]
            all_means_cam.append(pts_cam)
        
        all_means_cam = torch.stack(all_means_cam, dim=1)  # [B, N, H, W, 3]
        
        # Transform to world coordinates
        # TartanAir pose is camera-to-world (c2w): X_world = R @ X_cam + t
        all_means_world = []
        for i in range(N):
            pts_cam = all_means_cam[:, i]  # [B, H, W, 3]
            R = poses[:, i, :3, :3]  # [B, 3, 3] - c2w rotation
            t = poses[:, i, :3, 3]   # [B, 3] - c2w translation
            
            # c2w transformation: X_world = R @ X_cam + t
            pts_world = torch.einsum(
                'bij,bhwj->bhwi',
                R,
                pts_cam
            ) + t.unsqueeze(1).unsqueeze(1)
            all_means_world.append(pts_world)
        
        all_means_world = torch.stack(all_means_world, dim=1)  # [B, N, H, W, 3]
        P = N * H * W
        means = all_means_world.reshape(B, P, 3)
        
        # Process other parameters
        scales = per_view_gaussians['covariance'].permute(0, 1, 3, 4, 2).reshape(B, P, 3)
        rotations = per_view_gaussians['rotation'].permute(0, 1, 3, 4, 2).reshape(B, P, 4)
        opacities = per_view_gaussians['opacity'].reshape(B, P, 1)
        
        sh_color = per_view_gaussians['sh_color']  # [B, N, C, H, W]
        C = sh_color.shape[2]
        K = C // 3
        shs = sh_color.reshape(B, N, K, 3, H, W).permute(0, 1, 4, 5, 2, 3).reshape(B, P, K, 3)
        
        confidences = per_view_gaussians['confidence'].reshape(B, P, 1)
        
        return means, {
            'scales': scales,
            'rotations': rotations,
            'opacities': opacities,
            'shs': shs,
            'confidences': confidences,
        }


class CelestialSplat(nn.Module):
    """
    CelestialSplat: Feed-forward 360° Gaussian Splatting with Cross-View Attention.
    
    Main model integrating:
    1. DAP backbone (DINOv3 + DPT heads) for depth, mask, and features
    2. Feature adapter to aggregate multi-layer features
    3. Cross-view transformer for geometry-guided feature fusion
    4. GS decoder for Gaussian parameter prediction
    5. Gaussian fusion for unified world-coordinate representation
    """
    
    def __init__(self, config: Optional[CelestialSplatConfig] = None, dap_model=None):
        super().__init__()
        self.config = config or CelestialSplatConfig()
        
        # Store DAP model (will be set externally or passed in)
        self.dap = dap_model
        
        # Feature adapter
        self.feature_adapter = DAPFeatureAdapter(
            in_dim=self.config.dino_embed_dim,
            out_dim=self.config.adapter_out_dim
        )
        
        # Cross-view transformer
        self.transformer = CrossViewTransformer(
            dim=self.config.transformer_dim,
            num_layers=self.config.num_transformer_layers,
            num_heads=self.config.num_heads,
            K_neighbors=self.config.K_neighbors
        )
        
        # GS decoder
        self.gs_decoder = GSDecoder(
            in_dim=self.config.transformer_dim,
            hidden_dim=self.config.decoder_hidden_dim,
            sh_degree=self.config.sh_degree
        )
        
        # Gaussian fusion for unified representation
        fusion_config = self.config.fusion
        if fusion_config.strategy == 'voxel':
            self.gaussian_fusion = GaussianFusion(
                strategy='voxel',
                conf_thresh=fusion_config.conf_thresh,
                opacity_thresh=fusion_config.opacity_thresh,
                voxel_size=fusion_config.voxel_size,
                max_gs_per_voxel=fusion_config.max_gs_per_voxel,
                conf_weight=fusion_config.conf_weight,
                opacity_weight=fusion_config.opacity_weight,
            )
        else:
            self.gaussian_fusion = GaussianFusion(
                strategy='simple',
                conf_thresh=fusion_config.conf_thresh,
                opacity_thresh=fusion_config.opacity_thresh,
            )
    
    def set_dap_model(self, dap_model):
        """Set the DAP model (for loading pretrained weights)."""
        self.dap = dap_model
    
    def forward(
        self,
        images: torch.Tensor,      # [B, N, 3, H, W]
        poses: torch.Tensor,       # [B, N, 4, 4]
        intrinsics: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        return_per_view: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of CelestialSplat.
        
        Args:
            images: Input ERP images [B, N, 3, H, W]
            poses: Camera poses [B, N, 4, 4] (world-to-camera)
            intrinsics: Optional camera intrinsics
            return_intermediates: Whether to return intermediate features
            return_per_view: Whether to return per-view Gaussians (for debugging)
        
        Returns:
            Dict with:
                - gaussians: Unified Gaussian parameters in world coordinates
                    - means: [B, P, 3]
                    - scales: [B, P, 3]
                    - rotations: [B, P, 4]
                    - opacities: [B, P, 1]
                    - shs: [B, P, K, 3]
                    - masks: [B, P] (valid Gaussians)
                - dap_depth: DAP depth predictions
                - dap_mask: DAP mask predictions
                - features: (optional) intermediate features
        """
        B, N = images.shape[:2]
        H, W = images.shape[-2:]
        
        # Flatten for DAP processing
        images_flat = images.view(B * N, *images.shape[2:])  # [B*N, 3, H, W]
        
        # 1. DAP forward - get depth, mask, and intermediate features
        with torch.no_grad():
            dap_out = self._forward_dap(images_flat)
        
        depth = dap_out['depth']      # [B*N, H, W]
        mask = dap_out['mask']        # [B*N, H, W]
        features = dap_out['features']  # List of 4 tensors [B*N, 1024, H/16, W/16]
        cls_tokens = dap_out.get('cls_tokens')  # [B*N, 4, 1024]
        
        # 2. Adapt features
        adapted_features = self.feature_adapter(features, cls_tokens)  # [B*N, 256, H/16, W/16]
        adapted_features = adapted_features.view(B, N, *adapted_features.shape[1:])  # [B, N, 256, H/16, W/16]
        
        # Reshape depth and mask
        depth = depth.view(B, N, H, W)
        mask = mask.view(B, N, H, W)
        
        # 3. Cross-view transformer fusion
        fused_features = self.transformer(
            adapted_features,
            depth.unsqueeze(2),  # [B, N, 1, H, W]
            poses,
            intrinsics
        )  # [B, N, 256, H/16, W/16]
        
        # 4. GS decoder - outputs per-view Gaussians
        per_view_gaussians = self.gs_decoder(fused_features, depth, mask)
        
        # 5. Fuse into unified world-coordinate Gaussians
        unified_gaussians = self.gaussian_fusion(
            per_view_gaussians, 
            poses, 
            img_h=H, 
            img_w=W
        )
        
        outputs = {
            'gaussians': unified_gaussians,
            'dap_depth': depth,
            'dap_mask': mask,
        }
        
        if return_intermediates:
            outputs['features'] = adapted_features
            outputs['fused_features'] = fused_features
        
        if return_per_view:
            outputs['per_view_gaussians'] = per_view_gaussians
        
        return outputs
    
    def _forward_dap(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward through DAP model.
        
        Returns dict with:
            - depth: [B, H, W]
            - mask: [B, H, W]
            - features: List of 4 tensors [B, 1024, H/16, W/16]
            - cls_tokens: [B, 4, 1024]
        """
        assert self.dap is not None, "DAP model not set. Call set_dap_model() first."
        
        # Handle networks.dap.DAP wrapper (has .core attribute)
        if hasattr(self.dap, 'core'):
            dap_core = self.dap.core  # This is DepthAnythingV2
        else:
            dap_core = self.dap
        
        # Get patch size and dimensions
        patch_size = getattr(dap_core.pretrained, "patch_size", 16)
        patch_h, patch_w = images.shape[-2] // patch_size, images.shape[-1] // patch_size
        
        # Get intermediate features from DINOv3
        layer_indices = dap_core.intermediate_layer_idx[dap_core.encoder]
        raw_features = dap_core.pretrained.get_intermediate_layers(
            images, layer_indices, return_class_token=True
        )
        
        # Extract patch maps and cls tokens
        # raw_features: list of (patch_map, cls_token) tuples
        patch_maps = []
        cls_tokens = []
        for feat in raw_features:
            pm, ct = feat  # pm: [B, 1024, H/16, W/16] or [B, L, C], ct: [B, 1024]
            if pm.dim() == 3:
                pm = pm.permute(0, 2, 1).reshape(pm.shape[0], pm.shape[-1], patch_h, patch_w)
            patch_maps.append(pm)
            cls_tokens.append(ct)
        
        cls_tokens = torch.stack(cls_tokens, dim=1)  # [B, 4, 1024]
        
        # Forward through DPT heads
        depth = dap_core.depth_head(raw_features, patch_h, patch_w, patch_size) * dap_core.max_depth
        mask = dap_core.mask_head(raw_features, patch_h, patch_w, patch_size)
        
        return {
            'depth': depth.squeeze(1),  # [B, H, W]
            'mask': mask.squeeze(1),    # [B, H, W]
            'features': patch_maps,      # List of 4 tensors
            'cls_tokens': cls_tokens     # [B, 4, 1024]
        }
    
    def get_trainable_params(self, freeze_dap: bool = True) -> List[nn.Parameter]:
        """Get trainable parameters, optionally freezing DAP."""
        params = []
        
        # Always trainable
        params.extend(self.feature_adapter.parameters())
        params.extend(self.transformer.parameters())
        params.extend(self.gs_decoder.parameters())
        params.extend(self.gaussian_fusion.parameters())
        
        # DAP parameters (if not frozen)
        if not freeze_dap and self.dap is not None:
            params.extend(self.dap.parameters())
        
        return params


def build_celestial_splat(
    dap_model,
    encoder: str = 'vitl',
    num_transformer_layers: int = 4,
    K_neighbors: int = 4
) -> CelestialSplat:
    """
    Build CelestialSplat model with pretrained DAP.
    
    Args:
        dap_model: Pretrained DAP model
        encoder: Encoder type ('vits', 'vitb', 'vitl', 'vitg')
        num_transformer_layers: Number of transformer layers
        K_neighbors: Number of neighbor views for cross-attention
    
    Returns:
        CelestialSplat model
    """
    config = CelestialSplatConfig(
        encoder=encoder,
        dino_embed_dim=1024 if encoder in ['vitl', 'vitg'] else 768,
        num_transformer_layers=num_transformer_layers,
        K_neighbors=K_neighbors
    )
    
    model = CelestialSplat(config, dap_model=dap_model)
    return model
