import os
import sys
import math
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from diff_gaussian_rasterization_omni import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    CameraModelType,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from networks.models import make

from .model import _init_weights
from CelestialSplat.utils.math import inverse_sigmoid, inverse_softplus

@dataclass
class MonoCelestialSplatConfig:
    # input config
    image_height: int = 512
    image_width: int = 1024
    depth_eps: float = 1e-3

    # Depth bounds for GaussianInitializer and GaussianComposer (normalized space).
    # These apply AFTER median-normalization, so depth_eps=1e-3 means ~5mm when
    # scene median is 5m. depth_max=1e4 is a generous upper bound (~50km).
    depth_max: float = 1e4

    # Delta output limits to prevent delta_head from drifting to extreme values.
    # delta_z_limit=100 with delta_factor_z=0.01 gives ±1.0 change in softplus
    # input space, which translates to ~±2m depth change at median=5m.
    delta_z_limit: float = 100.0

    # Sigmoid input safety bounds to prevent inverse_sigmoid explosion.
    # Color uses a slightly wider range (0.01-0.99) than opacity (1e-6-0.999999).
    sigmoid_input_min: float = 1e-6
    sigmoid_input_max: float = 0.999999
    color_sigmoid_input_min: float = 0.01
    color_sigmoid_input_max: float = 0.99

    # Sky threshold for masking base opacities (meters).
    # Must be in METRIC space, independent of DAP's normalized depth range.
    sky_depth_threshold: float = 100.0

    # DAP backbone configuration
    dap_depth_metric_scale: float = 100.0
    dap_config_path: str = "config/infer.yaml"
    dap_weights_path: str = "weights/model.pth"
    dap_encoder: str = 'vitl'
    dap_features: int = 256
    dap_out_channels: List[int] = (256, 512, 1024, 1024)
    dap_use_bn: bool = False
    dap_use_clstoken: bool = False
    dap_finetune: bool = False

    # Gaussian Initializer config
    num_layers: int = 2
    stride: int = 2
    scale_factor: float = 1.0

    # Decoder config
    hidden_dim: int = 256

    # Gaussian Composer config
    min_scale: float = 0.0
    max_scale: float = 10.0
    delta_factor_xy: float = 0.01    # Increased: delta=1 -> ~2.5px tangential shift at 10m
    delta_factor_z: float = 0.01     # Increased: delta=1 -> ~0.5m depth change
    delta_factor_color: float = 0.1
    delta_factor_opacity: float = 0.5  # Reduced: less sensitive to avoid sky alpha leaks
    delta_factor_scale: float = 1.0
    delta_factor_quaternion: float = 1.0

    # DepthScaler config
    depth_scaler_mode: str = "global_median"  # "none", "global_median", "learned_local"

    # OmniGaussianRender config
    sh_degree: int = 0
    bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class GaussianBaseValues:
    directions: torch.Tensor   # [B, 3, num_layers, H/s, W/s] unit direction vectors on sphere
    lon_ndc: torch.Tensor      # [B, 1, num_layers, H/s, W/s] longitude in [-1, 1]
    lat_ndc: torch.Tensor      # [B, 1, num_layers, H/s, W/s] latitude in [-1, 1]
    depths: torch.Tensor       # [B, 1, num_layers, H/s, W/s] radial distances
    scales: torch.Tensor
    quaternions: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor


@dataclass
class Gaussians3D:
    mean_vectors: torch.Tensor
    singular_values: torch.Tensor
    quaternions: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    base_positions: Optional[torch.Tensor] = None    # for L_delta offset loss
    projected_coords: Optional[torch.Tensor] = None  # [B, P, 2] normalized ERP coords for grid_sample
    depths: Optional[torch.Tensor] = None            # [B, P, 1] per-Gaussian metric depths for L_splat


class DepthScaler(nn.Module):
    """Align predicted depth scale to GT depth.

    Modes:
    - none: identity (no scaling).
    - global_median: compute a single global scale factor via median ratio.
      Suitable for synthetic stage-1 training where a constant bias dominates.
    - learned_local: (reserved) per-pixel scale map predicted by a small network.
    """

    def __init__(self, config: MonoCelestialSplatConfig):
        super().__init__()
        self.mode = config.depth_scaler_mode
        self.depth_eps = config.depth_eps
        
        if self.mode == "learned_local":
            # Reserved: small UNet-like scale map estimator
            self.scale_net = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 3, padding=1),
                nn.Sigmoid(),
            )
            self.apply(_init_weights)

    def forward(
        self,
        pred_depth: torch.Tensor,
        gt_depth: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pred_depth: [B, C, H, W] predicted depth.
            gt_depth:   [B, 1, H, W] ground-truth depth (training only).
            mask:       [B, 1, H, W] binary valid mask.
        Returns:
            aligned_depth: [B, C, H, W] scaled depth.
            scale_map:     [B, 1, H, W] per-pixel scale factor (for loss/debug).
        """
        if self.mode == "none" or gt_depth is None:
            return pred_depth, torch.ones_like(pred_depth[:, :1])

        if self.mode == "global_median":
            # Compute median scale over valid pixels
            if mask is not None:
                # Handle dimension mismatch: mask may be [B,1,H,W] while gt_depth is [B,H,W]
                mask_bool = mask.bool() if mask.dtype != torch.bool else mask
                if mask_bool.dim() == 4:
                    mask_bool = mask_bool.squeeze(1)  # [B, H, W]
                pred_valid = pred_depth[:, :1].squeeze(1)[mask_bool]
                gt_valid = gt_depth.squeeze(1)[mask_bool] if gt_depth.dim() == 4 else gt_depth[mask_bool]
            else:
                pred_valid = pred_depth[:, :1].flatten()
                gt_valid = gt_depth.flatten()

            # Filter out near-zero depths; use >= to include the clamp boundary
            valid = (pred_valid >= self.depth_eps) & (gt_valid >= self.depth_eps)
            if valid.sum() > 0:
                scale = torch.median(gt_valid[valid]) / (torch.median(pred_valid[valid]) + 1e-6)
            else:
                scale = torch.tensor(1.0, device=pred_depth.device)
            aligned = pred_depth * scale
            scale_map = torch.full_like(pred_depth[:, :1], scale.item())
            print(f"DepthScaler global_median scale: {scale.item():.4f}, gt depth median: {torch.median(gt_valid[valid]).item():.4f}, valid pixels: {valid.sum().item()}/{valid.numel()}")

            # print(f"pred_valid: {pred_valid.min().item():.4f} ~ {pred_valid.max().item():.4f}, gt_valid: {gt_valid.min().item():.4f} ~ {gt_valid.max().item():.4f}, aligned: {aligned[:, :1].min().item():.4f} ~ {aligned[:, :1].max().item():.4f}, scale: {scale.item():.4f}")

            return aligned, scale_map

        elif self.mode == "learned_local":
            # Predict per-pixel scale from the first channel of pred_depth
            scale_map = self.scale_net(pred_depth[:, :1])  # [B, 1, H, W]
            aligned = pred_depth * scale_map
            return aligned, scale_map

        else:
            raise ValueError(f"Unknown depth_scaler_mode: {self.mode}")


class GaussianInitializer:
    """Initialize base Gaussians from RGB-D. No trainable parameters."""

    def __init__(self, config: MonoCelestialSplatConfig):
        self.num_layers = config.num_layers
        self.stride = config.stride
        self.scale_factor = config.scale_factor
        self.image_height = config.image_height
        self.image_width = config.image_width
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_eps = config.depth_eps
        self.depth_max = config.depth_max
        self.sky_depth_threshold = config.sky_depth_threshold  # Metric sky threshold

        self._precompute_angles()

    def _precompute_angles(self):
        # Pre-compute longitude, latitude, and unit directions for ERP pixels at stride resolution
        u = torch.arange(0.5 * self.stride, self.image_width, self.stride, device=self.device, dtype=torch.float32)
        v = torch.arange(0.5 * self.stride, self.image_height, self.stride, device=self.device, dtype=torch.float32)
        u, v = torch.meshgrid(u, v, indexing="xy")

        fx = self.image_width / (2.0 * math.pi)
        fy = -self.image_height / math.pi
        cx = self.image_width / 2.0
        cy = self.image_height / 2.0

        lon = ((u + 0.5) - cx) / fx
        lat = ((v + 0.5) - cy) / fy
        cos_lat = torch.cos(lat)
        x_dir = cos_lat * torch.sin(lon)
        y_dir = torch.sin(-lat)
        z_dir = cos_lat * torch.cos(lon)
        self.directions = torch.stack([x_dir, y_dir, z_dir], dim=0)  # [3, H/s, W/s]

        # Spherical NDC: map angles to [-1, 1]
        self.lon_ndc = (lon / math.pi).unsqueeze(0)             # [1, H/s, W/s]
        self.lat_ndc = (lat / (math.pi / 2.0)).unsqueeze(0)     # [1, H/s, W/s]

    def _prepare_feature_input(self, image: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        # if self.feature_input_stop_grad:
        image = image.detach()
        depth = depth.detach()

        normalized_disparity = 1.0 / depth.clamp(min=self.depth_eps)
        features_in = torch.cat([image, normalized_disparity], dim=1)
        # features_in = torch.cat([image, depth], dim=1)
        features_in = 2.0 * features_in - 1.0
        return features_in

    def __call__(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        nonsky_mask: torch.Tensor,
    ) -> Tuple[GaussianBaseValues, torch.Tensor, torch.Tensor]:
        """
        Args:
            image: [B, 3, H, W]
            depth: [B, C, H, W] where C == num_layers (e.g. 2 for duplicate_depth)
            nonsky_mask: [B, 1, H, W] binary (1=valid, 0=sky)
        Returns:
            base_values: GaussianBaseValues with shapes [B, C, num_layers, H/stride, W/stride]
            feature_input: [B, 3+C, H, W] ready for GaussianHead's image encoder
        """
        image = image.contiguous()
        depth = depth.contiguous()
        device = depth.device
        batch_size = depth.shape[0]
        _, _, image_height, image_width = depth.shape
        base_h = image_height // self.stride
        base_w = image_width // self.stride

        # Compute sky mask from ORIGINAL depth before normalization.
        # TartanAir sky depth is ~4093m, not 0, so depth > 0 does NOT catch sky.
        # We use the same logic as the dataloader: valid = 0 < depth < sky_depth_threshold.
        # NOTE: self.max_depth is DAP's normalized depth range (~1.0), NOT metric.
        # We must use sky_depth_threshold (metric, default 100m) for GT depth masking.
        with torch.no_grad():
            valid_depth_mask = ((depth > 0.0) & (depth < self.sky_depth_threshold))
            valid_depth_mask = valid_depth_mask.any(dim=1, keepdim=True).float()  # [B, 1, H, W]

        # Normalize depth to stable range for numerically stable training.
        # Use median (not min) to compute depth_factor because min is sensitive
        # to outliers: if depth_head predicts a single near-zero pixel, min-based
        # factor explodes and clamp(max=1e2) truncates all real depths.
        # Example: min=0.001 -> factor=1000 -> depth=50m becomes 50000 -> clamp=100
        # -> metric depth = 100/1000 = 0.1m (off by 500x). Median avoids this.
        with torch.no_grad():
            current_depth_median = depth.flatten(1).median(dim=-1).values
            depth_factor = 1.0 / (current_depth_median + 1e-6)
            normalized_depth = (depth * depth_factor[..., None, None, None])
            global_scale = 1.0 / depth_factor
        depth = normalized_depth.clamp(min=self.depth_eps, max=self.depth_max)  # [B]
        # print(f"GaussianInitializer global_scale: {global_scale}")

        # Surface depth: min-pooling to get nearest surface (consistent with max-pooling disparity)
        surface_depth = -F.max_pool2d(-depth, self.stride, self.stride)
        # Expand to num_layers
        if surface_depth.shape[1] == self.num_layers:
            depths = surface_depth.unsqueeze(1)  # [B, 1, num_layers, H/s, W/s]
        else:
            depths = surface_depth[:, :1, ...].repeat(1, self.num_layers, 1, 1).unsqueeze(1)

        # ERP spherical directions for each pixel at stride resolution
        # fx = image_width / (2.0 * math.pi)
        # fy = -image_height / math.pi
        # cx = image_width / 2.0
        # cy = image_height / 2.0

        # u = torch.arange(0.5 * self.stride, image_width, self.stride, device=device, dtype=torch.float32)
        # v = torch.arange(0.5 * self.stride, image_height, self.stride, device=device, dtype=torch.float32)
        # u, v = torch.meshgrid(u, v, indexing="xy")

        # lon = ((u + 0.5) - cx) / fx
        # lat = ((v + 0.5) - cy) / fy
        # cos_lat = torch.cos(lat)
        # x_dir = cos_lat * torch.sin(lon)
        # y_dir = torch.sin(-lat)
        # z_dir = cos_lat * torch.cos(lon)
        # directions = torch.stack([x_dir, y_dir, z_dir], dim=0)  # [3, H/s, W/s]
        directions = self.directions[None, :, None, :, :].repeat(batch_size, 1, self.num_layers, 1, 1)
        lon_ndc = self.lon_ndc[None, :, None, :, :].repeat(batch_size, 1, self.num_layers, 1, 1)
        lat_ndc = self.lat_ndc[None, :, None, :, :].repeat(batch_size, 1, self.num_layers, 1, 1)

        # Base scales derived from angular resolution * depth
        dx_angle = 2.0 * math.pi * self.stride / float(self.image_width)
        dy_angle = math.pi * self.stride / float(self.image_height)
        dz_angle = min(dx_angle, dy_angle)
        scale_x = depths * dx_angle
        scale_y = depths * dy_angle
        scale_z = depths * dz_angle
        base_scales = torch.cat([scale_x, scale_y, scale_z], dim=1) * self.scale_factor

        # Base quaternions (identity)
        base_quaternions = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        base_quaternions = base_quaternions[None, :, None, None, None].expand(
            batch_size, 4, self.num_layers, base_h, base_w
        )

        # Base colors
        base_colors = torch.full((batch_size, 3, self.num_layers, base_h, base_w), 0.5, device=device)
        temp = F.avg_pool2d(image, self.stride, self.stride)
        base_colors = temp[:, :, None, :, :].repeat(1, 1, self.num_layers, 1, 1)

        # Base opacities
        base_opacities = torch.full((batch_size, 1, self.num_layers, base_h, base_w), min(1.0 / self.num_layers, 0.5), device=device)

        # Apply combined validity mask: original depth in (0, sky_depth_threshold) AND non-sky.
        # Using the original-depth mask ensures sky regions (depth ~4093m in TartanAir)
        # always get zero base opacity, preventing DAP mask false-positives from creating
        # unwanted high-opacity bands at the ERP poles.
        if nonsky_mask is not None:
            valid_depth_mask = valid_depth_mask * nonsky_mask.float()
        mask_pooled = F.avg_pool2d(valid_depth_mask, self.stride, self.stride)
        mask_binary = (mask_pooled > 0.5).float()
        base_opacities = base_opacities * mask_binary[:, :, None, :, :]

        base_values = GaussianBaseValues(
            directions=directions,
            lon_ndc=lon_ndc,
            lat_ndc=lat_ndc,
            depths=depths,
            scales=base_scales,
            quaternions=base_quaternions,
            colors=base_colors,
            opacities=base_opacities,
        )

        feature_input = self._prepare_feature_input(image, depth)
        return base_values, feature_input, global_scale


class GaussianDecoder(nn.Module):
    """Decoder that fuses DPT multi-scale features and upsamples to target resolution.

    Design rationale:
    - DPT decoder outputs a natural multi-scale feature pyramid:
        feats[0]: [B, 256,  H/4,  W/4 ]
        feats[1]: [B, 512,  H/8,  W/8 ]
        feats[2]: [B, 1024, H/16, W/16]
        feats[3]: [B, 1024, H/32, W/32]
    - This is exactly the format that ml-sharp's MultiresConvDecoder was designed
      for. We therefore adopt its proven architecture:
      1) Project each pyramid level to the same hidden_dim.
      2) Fuse progressively from the lowest resolution (deep) to the highest
         resolution (shallow) using FeatureFusionBlock2d, which performs
         residual conv -> upsample x2 -> 1x1 projection at each step.
    - The output is already at H/4 resolution for the highest branch, and we
      only need one additional x2 upsample to reach H/stride (stride=2).

    Data-flow shape example (input 512x1024, hidden_dim=256, stride=2):
        feats[0]: [B, 256,  128, 256]
        feats[1]: [B, 512,  64,  128]
        feats[2]: [B, 1024, 32,  64]
        feats[3]: [B, 1024, 16,  32]
        proj0..3: [B, 256,  128, 256], [B, 256, 64, 128], [B, 256, 32, 64], [B, 256, 16, 32]
        fuse3:    [B, 256,  32,  64]   (upsampled from 16->32)
        fuse2:    [B, 256,  64,  128]  (upsampled from 32->64)
        fuse1:    [B, 256,  128, 256]  (upsampled from 64->128)
        fuse0:    [B, 256,  128, 256]  (no upsample, highest res)
        upsample: [B, 256,  256, 512]  <- final output at H/stride x W/stride
    """

    def __init__(self, dims_encoder: List[int], hidden_dim: int):
        super().__init__()
        self.dims_encoder = dims_encoder
        num_levels = len(dims_encoder)

        # Projections: level 0 uses 1x1 if mismatch, others use 3x3 (same as ml-sharp)
        convs = []
        for i, dim_in in enumerate(dims_encoder):
            if i == 0:
                conv = nn.Conv2d(dim_in, hidden_dim, kernel_size=1, bias=False) if dim_in != hidden_dim else nn.Identity()
            else:
                conv = nn.Conv2d(dim_in, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False)
            convs.append(conv)
        self.convs = nn.ModuleList(convs)

        # Fusion blocks: equivalent to ml-sharp FeatureFusionBlock2d
        # Each block does: resnet1(skip) + x -> resnet2 -> deconv(x2) -> out_conv
        fusions = []
        for i in range(num_levels):
            upsample = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2, padding=0, bias=False) if i != 0 else nn.Identity()
            fusions.append(nn.ModuleDict({
                'resnet1': nn.Sequential(
                    nn.ReLU(False),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ReLU(False),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True),
                ),
                'resnet2': nn.Sequential(
                    nn.ReLU(False),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ReLU(False),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True),
                ),
                'deconv': upsample,
                'out_conv': nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            }))
        self.fusions = nn.ModuleList(fusions)

        # One final upsample from H/4 to H/stride (stride=2 -> x2)
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )

        self.apply(_init_weights)

    def forward(self, encodings: List[torch.Tensor]) -> torch.Tensor:
        # Start from deepest level
        features = self.convs[-1](encodings[-1])
        features = self.fusions[-1]['resnet2'](features)
        features = self.fusions[-1]['deconv'](features)
        features = self.fusions[-1]['out_conv'](features)

        for i in range(len(encodings) - 2, -1, -1):
            features_i = self.convs[i](encodings[i])
            res = self.fusions[i]['resnet1'](features_i)
            features = features + res
            features = self.fusions[i]['resnet2'](features)
            features = self.fusions[i]['deconv'](features)
            features = self.fusions[i]['out_conv'](features)

        features = self.final_upsample(features)
        return features


class GaussianHead(nn.Module):
    """Head that fuses decoder features with image+depth skip and predicts deltas.

    Data-flow shape example (input 512x1024, stride=2, hidden_dim=256, num_layers=2):
        decoded_features:  [B, 256, 256, 512]
        feature_input:     [B, 5,   512, 1024]  (RGB + 2-channel disparity)
        image_encoder:     [B, 256, 256, 512]   (stride-2 conv)
        fusion_block:      [B, 256, 256, 512]   (residual fusion)
        geometry_head:     [B, 6,   256, 512]   -> unflatten -> [B, 3, 2, 256, 512]
        texture_head:      [B, 22,  256, 512]   -> unflatten -> [B, 11, 2, 256, 512]
        delta (concat):    [B, 14,  2, 256, 512]
    """

    def __init__(self, dim_in: int, hidden_dim: int, stride: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers

        # Image encoder: lift image+disparity to hidden_dim and downsample to stride resolution
        kernel_size = 3 if stride != 1 else 1
        padding = (kernel_size - 1) // 2
        self.image_encoder = nn.Conv2d(
            dim_in, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

        # Fusion block: residual-style fusion of decoder features and skip features
        self.skip_resnet = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.fusion_out_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        # Prediction heads: zero-initialized so initial delta = 0
        self.geometry_head = nn.Conv2d(hidden_dim, 3 * num_layers, kernel_size=1)
        self.texture_head = nn.Conv2d(hidden_dim, 11 * num_layers, kernel_size=1)
        
        self.apply(_init_weights)

        nn.init.zeros_(self.geometry_head.weight)
        nn.init.zeros_(self.geometry_head.bias)
        nn.init.zeros_(self.texture_head.weight)
        nn.init.zeros_(self.texture_head.bias)

    def forward(self, decoded_features: torch.Tensor, feature_input: torch.Tensor) -> torch.Tensor:
        skip_features = self.image_encoder(feature_input)
        res = self.skip_resnet(skip_features)
        fused = decoded_features + res
        fused = self.fusion_out_conv(fused)

        geom = self.geometry_head(fused)
        tex = self.texture_head(fused)
        geom = geom.unflatten(1, (3, self.num_layers))
        tex = tex.unflatten(1, (11, self.num_layers))
        delta = torch.cat([geom, tex], dim=1)
        return delta


class GaussianComposer(nn.Module):
    """Compose base Gaussians and predicted deltas into final 3D Gaussians.

    NOTE: post_process is reserved for sky-region handling strategies
    (e.g. hard-coding far-depth sky Gaussians or sky-sphere texture training).
    Currently it simply applies the sky mask by zeroing opacities.
    """

    def __init__(self, config: MonoCelestialSplatConfig):
        super().__init__()
        self.min_scale = config.min_scale
        self.max_scale = config.max_scale
        self.delta_factor_xy = config.delta_factor_xy
        self.delta_factor_z = config.delta_factor_z
        self.delta_factor_color = config.delta_factor_color
        self.delta_factor_opacity = config.delta_factor_opacity
        self.delta_factor_scale = config.delta_factor_scale
        self.delta_factor_quaternion = config.delta_factor_quaternion
        self.depth_eps = config.depth_eps
        self.depth_max = config.depth_max
        self.delta_z_limit = config.delta_z_limit
        self.sigmoid_input_min = config.sigmoid_input_min
        self.sigmoid_input_max = config.sigmoid_input_max
        self.color_sigmoid_input_min = config.color_sigmoid_input_min
        self.color_sigmoid_input_max = config.color_sigmoid_input_max

        if self.max_scale != 1.0:
            constant_a = (self.max_scale - self.min_scale) / (1.0 - self.min_scale) / (self.max_scale - 1.0)
        else:
            constant_a = 1.0
        constant_b = inverse_sigmoid(torch.tensor((1.0 - self.min_scale) / (self.max_scale - self.min_scale))).item()
        self.register_buffer('scale_constant_a', torch.tensor(constant_a))
        self.register_buffer('scale_constant_b', torch.tensor(constant_b))

    def forward(
        self,
        delta: torch.Tensor,
        base_values: GaussianBaseValues,
        global_scale: Optional[torch.Tensor] = None
    ) -> Gaussians3D:
        # Compute base mean vectors BEFORE delta for L_delta offset loss
        base_mean_vectors = base_values.directions * base_values.depths  # [B, 3, N, H, W]

        mean_vectors, projected_coords = self._forward_mean(base_values, delta)
        base_scales = base_values.scales
        singular_values = self._scale_activation(base_scales, delta[:, 3:6])
        quaternions = self._quaternion_activation(base_values.quaternions, delta[:, 6:10])
        colors = self._color_activation(base_values.colors, delta[:, 10:13])
        opacities = self._opacity_activation(base_values.opacities, delta[:, 13].unsqueeze(1))

        # Flatten spatial dimensions: [B, C, N, H, W] -> [B, N*H*W, C]
        mean_vectors = mean_vectors.permute(0, 2, 3, 4, 1).flatten(1, 3)
        projected_coords = projected_coords.permute(0, 2, 3, 4, 1).flatten(1, 3)
        base_mean_vectors = base_mean_vectors.permute(0, 2, 3, 4, 1).flatten(1, 3)
        singular_values = singular_values.permute(0, 2, 3, 4, 1).flatten(1, 3)
        quaternions = quaternions.permute(0, 2, 3, 4, 1).flatten(1, 3)
        colors = colors.permute(0, 2, 3, 4, 1).flatten(1, 3)
        opacities = opacities.permute(0, 2, 3, 4, 1).flatten(1, 3).squeeze(-1)

        if global_scale is not None:
            mean_vectors = global_scale[:, None, None] * mean_vectors
            base_mean_vectors = global_scale[:, None, None] * base_mean_vectors
            singular_values = global_scale[:, None, None] * singular_values

        # Pre-compute per-Gaussian metric depths for L_splat (avoids recomputing norm in loss)
        depths = torch.norm(mean_vectors, dim=-1, keepdim=True).clamp(min=self.depth_eps)  # [B, P, 1]

        gaussians = Gaussians3D(
            mean_vectors=mean_vectors,
            singular_values=singular_values,
            quaternions=quaternions,
            colors=colors,
            opacities=opacities,
            base_positions=base_mean_vectors,
            projected_coords=projected_coords,
            depths=depths,
        )
        return gaussians

    def _forward_mean(self, base_values: GaussianBaseValues, delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Spherical NDC mean residual for 360 ERP.

        We treat lon and lat as normalized to [-1, 1], analogous to perspective NDC.
        delta[:, 0] -> longitude NDC offset
        delta[:, 1] -> latitude NDC offset
        delta[:, 2] -> radial depth residual in softplus space

        Returns:
            mean_vectors:     [B, 3, N, H, W] final 3D positions
            projected_coords: [B, 2, N, H, W] (u_norm, v_norm) in [-1, 1] for grid_sample
        """
        delta_factor = torch.tensor(
            [self.delta_factor_xy, self.delta_factor_xy, self.delta_factor_z],
            device=delta.device,
        )[None, :, None, None, None]

        # Apply NDC offsets
        # Lon: wrap-around to preserve 360° continuity (ERP left-right boundary).
        # Lat: hard-clamp because poles are singular and |lat| > 90° flips direction.
        new_lon_ndc_raw = base_values.lon_ndc + delta_factor[:, 0:1] * delta[:, 0:1]
        new_lon_ndc = torch.fmod(new_lon_ndc_raw + 3.0, 2.0) - 1.0
        new_lat_ndc = (base_values.lat_ndc + delta_factor[:, 1:2] * delta[:, 1:2]).clamp(min=-1.0, max=1.0)

        # grid_sample coordinates: u_norm = lon/π, v_norm = -lat/(π/2)
        # new_lon_ndc is already lon/π, new_lat_ndc is already lat/(π/2)
        projected_coords = torch.cat([new_lon_ndc, -new_lat_ndc], dim=1)  # [B, 2, N, H, W]

        # Denormalize back to radians
        new_lon = new_lon_ndc * math.pi
        new_lat = new_lat_ndc * (math.pi / 2.0)

        # Reconstruct unit direction from updated angles
        cos_lat = torch.cos(new_lat)
        new_x = cos_lat * torch.sin(new_lon)
        new_y = -torch.sin(new_lat)
        new_z = cos_lat * torch.cos(new_lon)
        new_directions = torch.cat([new_x, new_y, new_z], dim=1)

        # Softplus residual for inverse depth (1/z), matching ml-sharp paper §A.1.
        # In NDC space, 1/z is the natural coordinate for depth. Negative delta
        # pushes 1/z -> 0 (i.e. z -> +inf, safe), positive delta pushes 1/z up
        # (i.e. z -> 0, bounded by OffsetLoss radial_threshold).
        inv_base = 1.0 / base_values.depths.clamp(min=self.depth_eps)
        delta_z = delta[:, 2:3].clamp(min=-self.delta_z_limit, max=self.delta_z_limit)
        new_inv_depths = F.softplus(
            inverse_softplus(inv_base) + delta_factor[:, 2:3] * delta_z
        )
        new_depths = (1.0 / new_inv_depths).clamp(min=self.depth_eps, max=self.depth_max)

        print(f"delta_z stats: min {delta_z.min().item():.4f}, max {delta_z.max().item():.4f}, mean {delta_z.mean().item():.4f}")

        mean_vectors = new_directions * new_depths
        return mean_vectors, projected_coords

    def _scale_activation(self, base: torch.Tensor, learned_delta: torch.Tensor) -> torch.Tensor:
        scale_factor = (self.max_scale - self.min_scale) * torch.sigmoid(
            self.scale_constant_a * self.delta_factor_scale * learned_delta + self.scale_constant_b
        ) + self.min_scale
        return base * scale_factor

    def _quaternion_activation(self, base: torch.Tensor, learned_delta: torch.Tensor) -> torch.Tensor:
        return base + self.delta_factor_quaternion * learned_delta

    def _color_activation(self, base: torch.Tensor, learned_delta: torch.Tensor) -> torch.Tensor:
        base = torch.clamp(base, min=self.color_sigmoid_input_min, max=self.color_sigmoid_input_max)
        inv_base = inverse_sigmoid(base)
        colors = torch.sigmoid(inv_base + self.delta_factor_color * learned_delta)
        return colors

    def _opacity_activation(self, base: torch.Tensor, learned_delta: torch.Tensor) -> torch.Tensor:
        # Preserve exact zeros (sky regions) so delta cannot resurrect them
        zero_mask = (base <= 0).float()
        base = torch.clamp(base, min=self.sigmoid_input_min, max=self.sigmoid_input_max)
        inv_base = inverse_sigmoid(base)
        result = torch.sigmoid(inv_base + self.delta_factor_opacity * learned_delta)
        return result * (1.0 - zero_mask)


class MonoCelestialSplat(nn.Module):
    def __init__(self, config: MonoCelestialSplatConfig):
        super().__init__()
        self.config = config
        
        # DAP backbone for depth and mask prediction
        self._create_dap_model()

        # Depth scaler: align pred depth to GT depth during training
        self.depth_scaler = DepthScaler(config)

        # Base Gaussian initializer (no trainable params)
        self.initializer = GaussianInitializer(config)

        # Decoder: fuse DPT multi-scale features and upsample
        self.gaussian_decoder = GaussianDecoder(
            dims_encoder=list(config.dap_out_channels),
            hidden_dim=config.hidden_dim,
        )

        # Head: fuse decoder output with image+depth skip and predict deltas
        # dim_in=5 because depth_head outputs 2 channels (duplicate_depth=True)
        self.gaussian_head = GaussianHead(
            dim_in=5,
            hidden_dim=config.hidden_dim,
            stride=config.stride,
            num_layers=config.num_layers,
        )

        # Composer: base + delta -> final Gaussians
        self.composer = GaussianComposer(config)

        self.to('cuda')

    def _create_dap_model(self):
        """Load DAP model with pretrained weights."""
        with open(self.config.dap_config_path, 'r') as f:
            dap_config = yaml.load(f, Loader=yaml.FullLoader)
        
        dap_model = make(dap_config['model'])
        dap_state = torch.load(self.config.dap_weights_path, map_location='cuda')
        
        dap_model = dap_model.to('cuda')
        m_state = dap_model.state_dict()

        m_state_dict = {}
        for k, v in dap_state.items():
            if k.startswith('module'):
                k = k.replace('module.', '')  # Remove 'module.'

            if k not in m_state:
                continue

            if not dap_model.duplicate_depth:
                m_state_dict[k] = v
            else:
                if k == 'core.depth_head.scratch.output_conv2.2.weight':
                    # Duplicate the single-channel depth weights to the second channel
                    m_state_dict[k] = torch.cat([v, v.clone()], dim=0)
                elif k == 'core.depth_head.scratch.output_conv2.2.bias':
                    # Duplicate the bias as well
                    m_state_dict[k] = torch.cat([v, v.clone()], dim=0)
                else:
                    m_state_dict[k] = v
        dap_model.load_state_dict(m_state_dict, strict=False)
        
        if hasattr(dap_model, 'core'):
            self.dap_core = dap_model.core
        else:
            raise ValueError("DAP model does not have 'core' attribute. Check the model structure.")

        if self.config.dap_finetune:
            self.dap_core.train()
        else:
            self.dap_core.eval()

        # self.config.depth_eps = max(self.config.depth_eps, dap_model.min_depth)  # Ensure a safe minimum depth for numerical stability
        # self.config.max_depth = min(self.config.max_depth, dap_model.max_depth)  # Ensure max depth does not exceed DAP's capability
        # print(f"DAP model loaded. Depth range: [{self.config.depth_eps}, {self.config.max_depth}]")

    def _forward_dap(self, x: torch.Tensor):
        # DAv2 patch
        patch_size = getattr(self.dap_core.pretrained, "patch_size", 16)
        patch_h, patch_w = x.shape[-2] // patch_size, x.shape[-1] // patch_size

        # DINOv3 intermediate features, e.g., [1, 512, 1024] -> [1, 1024, 32, 64]
        with torch.no_grad():
            raw_features = self.dap_core.pretrained.get_intermediate_layers(x, self.dap_core.intermediate_layer_idx[self.dap_core.encoder], return_class_token=True)
        
        patch_maps = []
        cls_tokens = []
        for feat in raw_features:
            pm, ct = feat  # pm: [B, C, H/16, W/16] already reshaped by DINOv3Adapter, ct: [B, C]
            patch_maps.append(pm)
            cls_tokens.append(ct)
        cls_tokens = torch.stack(cls_tokens, dim=1) if cls_tokens else None  # [B, 4, embed_dim] or None

        pred_depth, depth_feats = self.dap_core.depth_head(patch_maps, patch_h, patch_w, patch_size, return_features=True)  # [B, H, W]
        with torch.no_grad():
            pred_mask = self.dap_core.mask_head(patch_maps, patch_h, patch_w, patch_size)  # [B, H, W]
        nonsky_mask = (1 - pred_mask) > 0.5  # Binary mask for non-sky regions

        # Clamp pred_depth to prevent near-zero values from causing 1/depth explosion
        # in loss functions. 1e-3 (1 mm) is a safe physical lower bound.
        if self.config.dap_depth_metric_scale > 1.0:
            pred_depth = pred_depth * self.config.dap_depth_metric_scale

        pred_depth = pred_depth.clamp(min=self.config.depth_eps)

        return {
            "pred_depth": pred_depth,   # [B, 2, H, W]
            "pred_mask": pred_mask,     # [B, 1, H, W] - probability map
            "nonsky_mask": nonsky_mask, # [B, 1, H, W] - binary confidence (1=valid, 0=sky)
            "features": depth_feats,    # List of 4 DPT multi-scale features: [1,256,128,256], [1,512,64,128], [1,1024,32,64], [1,1024,16,32]
            "cls_tokens": cls_tokens    # [B, 4, 1024]
        }

    def forward(
        self,
        image: torch.Tensor,
        gt_depth: Optional[torch.Tensor] = None,
        return_extras: bool = False,
    ):
        # 1. DAP encoder: depth + DINOv3 patch features
        dap_out = self._forward_dap(image)
        pred_depth = dap_out["pred_depth"]      # [B, 2, H, W]
        pred_mask = dap_out["pred_mask"]        # [B, 1, H, W] probability map
        nonsky_mask = dap_out["nonsky_mask"]    # [B, 1, H, W] binary
        features = dap_out["features"]          # DPT multi-scale features

        # 2. (Optional) align predicted depth to GT depth for stage-1 synthetic training
        aligned_depth, scale_map = self.depth_scaler(pred_depth, gt_depth, mask=nonsky_mask)

        # 3. Base Gaussians from depth
        base_values, feature_input, global_scale = self.initializer(image, aligned_depth, nonsky_mask)

        # 4. Delta Gaussians from DINOv3 features
        decoded_features = self.gaussian_decoder(features)      # [B, hidden_dim, H/stride, W/stride]
        delta = self.gaussian_head(decoded_features, feature_input)  # [B, 14, num_layers, H/stride, W/stride]

        # 5. Compose base + delta into final Gaussians
        gaussians = self.composer(
            delta=delta,
            base_values=base_values,
            global_scale=global_scale,
        )

        if return_extras:
            return gaussians, {
                "pred_depth": pred_depth,
                "aligned_depth": aligned_depth,
                "pred_mask": pred_mask,
                "nonsky_mask": nonsky_mask,
                "scale_map": scale_map,
            }
        return gaussians


class OmniGaussianRender(nn.Module):
    """Differentiable ERP Gaussian rasterizer wrapper."""

    def __init__(self, config: MonoCelestialSplatConfig):
        super().__init__()
        self.image_height = config.image_height
        self.image_width = config.image_width
        self.bg_color = torch.tensor(config.bg_color, dtype=torch.float32, device='cuda')

        # For LONLAT (ERP) camera, tanfovx/tanfovy are not heavily used in projection,
        # but we provide sensible defaults (90° half-FOV -> tan(90°) is undefined,
        # so we use a large proxy value of 1.0 which is common in practice).
        raster_settings = GaussianRasterizationSettings(
            image_height=self.image_height,
            image_width=self.image_width,
            tanfovx=1.0,
            tanfovy=1.0,
            bg=self.bg_color,
            scale_modifier=1.0,
            viewmatrix=torch.eye(4, dtype=torch.float32, device='cuda'),
            projmatrix=torch.eye(4, dtype=torch.float32, device='cuda'),
            RwcT=torch.zeros(3, dtype=torch.float32, device='cuda'),
            sh_degree=config.sh_degree,
            campos=torch.zeros(3, dtype=torch.float32, device='cuda'),
            prefiltered=False,
            camera_type=CameraModelType.LONLAT,
            render_depth=True,
            render_opacity=True,
        )
        self.rasterizer = GaussianRasterizer(raster_settings)

    def forward(self, gaussians: Gaussians3D) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            gaussians: Gaussians3D with per-batch flattened parameters.
        Returns:
            rendered_images: [B, 3, H, W]
            rendered_depths: [B, 1, H, W]
            rendered_opacities: [B, 1, H, W]
        """
        B = gaussians.mean_vectors.shape[0]
        rendered_images = []
        rendered_depths = []
        rendered_opacities = []
        for b in range(B):
            color, depth, opacity, _ = self.rasterizer(
                means3D=gaussians.mean_vectors[b],          # [P, 3]
                means2D=torch.zeros_like(gaussians.mean_vectors[b]),  # [P, 3]
                opacities=gaussians.opacities[b].unsqueeze(-1),       # [P, 1]
                colors_precomp=gaussians.colors[b],                   # [P, 3]
                scales=gaussians.singular_values[b],                  # [P, 3]
                rotations=gaussians.quaternions[b],                   # [P, 4]
            )
            rendered_images.append(color)
            rendered_depths.append(depth)
            rendered_opacities.append(opacity)
        return (
            torch.stack(rendered_images, dim=0),      # [B, 3, H, W]
            torch.stack(rendered_depths, dim=0),      # [B, 1, H, W]
            torch.stack(rendered_opacities, dim=0),   # [B, 1, H, W]
        )