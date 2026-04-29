"""SHARP Loss Functions for CelestialSplat.

This module implements the composite loss function described in the SHARP paper
(arXiv:2512.10685v2).

Loss Components (matching paper §3.4):
    1. L_color      — L1 pixel reconstruction (Eq. 3.3)
    2. L_percep    — VGG/ResNet feature + Gram matrix (Eq. 3.4)
    3. L_alpha     — BCE on rendered alpha (Eq. 3.5)
    4. L_depth     — L1 on inverse depth / disparity (Eq. 3.6)
    5. L_tv        — Total variation on inverse depth (Eq. 3.7)
    6. L_grad      — Per-Gaussian floater suppression via disparity gradient (Eq. 3.8)
    7. L_delta     — Per-Gaussian position offset hinge loss (Eq. 3.9)
    8. L_splat     — Per-Gaussian projected 2D variance hinge loss (Eq. 3.10)

360 / ERP specific notes:
    - L_grad projects each Gaussian center to ERP coordinates to sample the
      disparity gradient at its exact projected location.
    - L_splat uses an angular-to-pixel approximation for ERP projection.
    - L_delta operates on 3D Euclidean offsets; the threshold should be set
      according to the metric scale of the scene (default 1.0 m).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# ------------------------------------------------------------------------------
# Loss Configuration
# ------------------------------------------------------------------------------
@dataclass
class LossConfig:
    """Centralized configuration for all loss terms.

    All weights are multiplied against their respective raw loss values.
    A weight of 0.0 disables the corresponding loss term.
    """
    image_height: int = 512
    image_width: int = 1024
    depth_eps: float = 1e-3  # Minimum depth to prevent 1/depth explosion in losses

    # --- Reconstruction Losses ---
    l1_color_weight: float = 2.0 # 1.0
    perceptual_weight: float = 0.1 # 3.0          # Feature + Gram combined
    perceptual_on_novel_only: bool = True   # Paper: perceptual loss only on novel views
    perceptual_layers: List[str] = field(default_factory=lambda: ["layer1", "layer2", "layer3", "layer4"])
    perceptual_feat_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    perceptual_gram_weights: List[float] = field(default_factory=lambda: [10.0, 10.0, 10.0, 10.0])

    # --- Geometric Regularization ---
    disparity_weight: float = 0.2           # Inverse depth L1 (L_depth)
    alpha_bce_weight: float = 1.0           # Foreground alpha BCE (L_alpha)
    tv_depth_weight: float = 1.0            # Background depth smoothness (L_tv)

    # --- Floater Suppression (L_grad, Eq. 3.8) ---
    floater_weight: float = 0.5
    floater_sigma: float = 1e-2             # paper default: 10^{-2}
    floater_epsilon: float = 1e-2           # paper default: 10^{-2}

    # --- Gaussian Offset Constraint (L_delta, Eq. 3.9) ---
    delta_weight: float = 0.5
    delta_threshold: float = 0.1            # Tangential offset threshold (meters).
                                            # Paper says δ=400.0 but that is in their
                                            # internal units. For 256x512, 0.1m ≈ 0.8px.
    delta_radial_threshold: float = 1.0     # Radial (depth) offset threshold (meters).
                                            # Kept separate from tangential to allow
                                            # reasonable depth-error compensation while
                                            # preventing unbounded radial drift.

    # --- Projected Gaussian Variance (L_splat, Eq. 3.10) ---
    splat_weight: float = 0.5
    splat_sigma_min: float = 1e-1           # paper default: 10^{-1}
    splat_sigma_max: float = 1e2            # paper default: 10^{2}

    # --- Scale Regularization (Depth Adjustment Map) ---
    # Only meaningful when DepthScaler is in learned_local mode.
    enable_scale_reg: bool = False          # set True for learned_local scaler

    # --- Depth Adjustment Regularization ---
    scale_reg_weight_l1: float = 0.1
    scale_reg_weight_tv: float = 0.1
    scale_reg_num_scales: int = 6

    # Distortion-aware weighting for ERP non-uniform pixel density
    use_distortion_map: bool = False  # DAP paper: weight by cos(latitude)

    # other
    enable_print: bool = True


# ------------------------------------------------------------------------------
# 1. L1 Color Loss (Pixel Reconstruction)  —  L_color (Eq. 3.3)
# ------------------------------------------------------------------------------
class L1ColorLoss(nn.Module):
    """Pixel-level L1 reconstruction loss."""

    def __init__(self, config: LossConfig):
        super().__init__()
        self.weight = config.l1_color_weight

    def forward(
        self,
        pred_img: torch.Tensor,
        target_img: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        distortion_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diff = torch.abs(pred_img - target_img)
        if distortion_map is not None:
            diff = diff * distortion_map.view(1, 1, self.image_height, 1)
        if mask is not None:
            diff = diff * mask
            denom = (mask.sum() * 3 + 1e-8) if distortion_map is None else (diff.sum() / (torch.abs(pred_img - target_img) * mask).clamp(min=1e-8)).sum() + 1e-8
            # Simpler: if distortion_map is used, we still normalize by mask pixels but weighted
            if distortion_map is not None:
                weight_sum = (mask * distortion_map.view(1, 1, self.image_height, 1)).sum() * 3
                loss = diff.sum() / (weight_sum + 1e-8)
            else:
                loss = diff.sum() / (mask.sum() * 3 + 1e-8)
        else:
            if distortion_map is not None:
                loss = diff.sum() / (diff.numel() / 3 * distortion_map.mean() + 1e-8)
            else:
                loss = diff.mean()
        return loss * self.weight


# ------------------------------------------------------------------------------
# 2. Perceptual Loss (VGG / ResNet Feature + Gram Matrix)  —  L_percep (Eq. 3.4)
# ------------------------------------------------------------------------------
class PerceptualLoss(nn.Module):
    """Perceptual loss in feature space with Gram matrix sharpness term.

    Uses a frozen pretrained ResNet-50 to extract multi-scale features.
    Two terms are combined:
      1. Feature loss: L2 distance between intermediate feature maps.
      2. Gram matrix loss: L2 distance between normalized Gram matrices.

    **Precision warning**: Compute this loss in FP32. BF16 can cause numerical
    instability in Gram matrix computation (singularity in covariance).

    Note on DINOv3 vs ResNet-50:
    - DINOv3 features are semantically rich but spatially coarse (all at H/16).
      Using them for perceptual loss would require heavy resizing and lacks the
      natural multi-scale pyramid that CNNs provide. ResNet-50 remains the
      standard choice for perceptual loss (Johnson et al. 2016).
    - If you want to experiment with DINOv3 perceptual loss, add a
      DinoPerceptualLoss variant and swap it in via config.
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self.weight = config.perceptual_weight
        self.layers = config.perceptual_layers
        self.feat_weights = config.perceptual_feat_weights
        self.gram_weights = config.perceptual_gram_weights

        # Load pretrained ResNet-50 and extract named blocks
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.ModuleDict({
            "layer1": nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
            ),
            "layer2": resnet.layer2,
            "layer3": resnet.layer3,
            "layer4": resnet.layer4,
        })
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization constants
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    @staticmethod
    def gram_matrix(features: torch.Tensor) -> torch.Tensor:
        """Compute normalized Gram matrix.

        Args:
            features: [B, C, H, W]
        Returns:
            gram: [B, C, C]
        """
        b, c, h, w = features.size()
        f = features.view(b, c, h * w)
        gram = torch.bmm(f, f.transpose(1, 2))
        return gram / (h * w)  # Normalize

    def forward(self, pred_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        # Normalize to ImageNet statistics
        pred = (pred_img - self.mean) / self.std
        tgt = (target_img - self.mean) / self.std

        total_loss = 0.0
        for layer_name, fw, gw in zip(self.layers, self.feat_weights, self.gram_weights):
            pred = self.features[layer_name](pred)
            tgt = self.features[layer_name](tgt)

            if fw > 0.0:
                feat_loss = fw * F.mse_loss(pred, tgt)
            if gw > 0.0:
                pred_gram = self.gram_matrix(pred)
                tgt_gram = self.gram_matrix(tgt)
                gram_loss = gw * F.mse_loss(pred_gram, tgt_gram)
            
            total_loss += feat_loss + gram_loss

            # print(f"feat shape: {pred.shape}, feat_loss: {feat_loss.item():.4f}, gram shape: {pred_gram.shape}, gram_loss: {gram_loss.item():.4f}")

        return total_loss * self.weight


# ------------------------------------------------------------------------------
# 3. Inverse Depth (Disparity) Loss  —  L_depth (Eq. 3.6)
# ------------------------------------------------------------------------------
class InverseDepthLoss(nn.Module):
    """L1 loss on inverse depth (disparity) for metric depth alignment.

    The paper applies this ONLY to the first depth layer (Layer-1) and ONLY on
    the input view.
    
    NOTE: min_depth is set to 1e-3 (1 mm) to prevent 1/depth explosion when
    depth_head predicts near-zero values. This is a safe physical lower bound
    for most scenes.
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self.weight = config.disparity_weight
        self.eps = config.depth_eps  # Minimum depth to prevent 1/depth explosion in losses

    def forward(
        self,
        pred_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        distortion_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inv_pred = 1.0 / (pred_depth.clamp(min=self.eps))
        inv_gt = 1.0 / (gt_depth.clamp(min=self.eps))
        diff = torch.abs(inv_pred - inv_gt)
        if distortion_map is not None:
            diff = diff * distortion_map.view(1, 1, pred_depth.shape[-2], 1)
        if mask is not None:
            diff = diff * mask
            if distortion_map is not None:
                weight_sum = (mask * distortion_map.view(1, 1, pred_depth.shape[-2], 1)).sum()
                loss = diff.sum() / (weight_sum + 1e-8)
            else:
                loss = diff.sum() / (mask.sum() + 1e-8)
        else:
            if distortion_map is not None:
                loss = diff.sum() / (diff.numel() * distortion_map.mean() + 1e-8)
            else:
                loss = diff.mean()
        return loss * self.weight


# ------------------------------------------------------------------------------
# 4. Alpha BCE Loss (Foreground Opacity)  —  L_alpha (Eq. 3.5)
# ------------------------------------------------------------------------------
class AlphaBCELoss(nn.Module):
    """Binary cross-entropy loss forcing rendered alpha toward 1.0.

    The paper applies BCE on the rendered alpha map for both input and novel
    views. A mask is optional; the paper formulation does not explicitly mask
    it, but if sky regions should be excluded you can pass ``mask``.
    
    NOTE: pred_alpha is clamped before BCE to prevent the
    PyTorch BCE(0, 1) = 100 explosion for uncovered pixels.
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self.weight = config.alpha_bce_weight
        self.bce = nn.BCELoss(reduction="mean")
        self.eps = 1e-4

    def forward(
        self,
        pred_alpha: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        distortion_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = torch.ones_like(pred_alpha)
        # Clamp to avoid BCE(0, 1) = 100 explosion for uncovered pixels
        pred_alpha = pred_alpha.clamp(min=self.eps, max=1.0)
        if mask is not None:
            pred_alpha = pred_alpha * mask
            target = target * mask
        if distortion_map is not None:
            pred_alpha = pred_alpha * distortion_map.view(1, 1, pred_alpha.shape[-2], 1)
            target = target * distortion_map.view(1, 1, pred_alpha.shape[-2], 1)
        if mask is not None or distortion_map is not None:
            # BCE computed only over valid pixels; avoid div-by-zero
            denom = (mask.sum() + 1e-8) if distortion_map is None else (target.sum() + 1e-8)
            loss = F.binary_cross_entropy(pred_alpha, target, reduction="sum") / denom
        else:
            loss = self.bce(pred_alpha, target)
        return loss * self.weight


# ------------------------------------------------------------------------------
# 5. Total Variation Loss (Background Smoothness)  —  L_tv (Eq. 3.7)
# ------------------------------------------------------------------------------
class TotalVariationLoss(nn.Module):
    """Total variation on inverse depth for background smoothness.

    The paper applies TV to the **second** depth layer (Layer-2).

    NOTE on 360 / ERP images:
    - Standard forward differences are used here, same as the paper.
    - Ideally the horizontal (x) gradient should wrap around at the left-right
      boundary because ERP is 360° continuous. However, the paper does not
      mention circular padding and most implementations use replicate/border
      padding. If wrap-around artifacts appear, switch to circular padding
      for the horizontal dimension.
    
    NOTE: min_depth is set to 1e-3 (1 mm) to prevent 1/depth explosion when
    depth_head predicts near-zero values.
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self.weight = config.tv_depth_weight
        self.eps = config.depth_eps  # Minimum depth to prevent 1/depth explosion in losses

    def forward(
        self,
        depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        distortion_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inv_depth = 1.0 / (depth.clamp(min=self.eps))

        grad_x = torch.abs(inv_depth[:, :, :, :-1] - inv_depth[:, :, :, 1:])
        grad_y = torch.abs(inv_depth[:, :, :-1, :] - inv_depth[:, :, 1:, :])
        
        if distortion_map is not None:
            M = distortion_map.view(1, 1, depth.shape[-2], 1)
            grad_x = grad_x * M[:, :, :, :-1]
            grad_y = grad_y * M[:, :, :-1, :]
        
        if mask is not None:
            mask_x = mask[:, :, :, :-1]
            mask_y = mask[:, :, :-1, :]
            grad_x = grad_x * mask_x
            grad_y = grad_y * mask_y
            denom = (mask_x.sum() + mask_y.sum() + 1e-8)
            if distortion_map is not None:
                M = distortion_map.view(1, 1, depth.shape[-2], 1)
                denom = (mask_x * M[:, :, :, :-1]).sum() + (mask_y * M[:, :, :-1, :]).sum() + 1e-8
            loss = (grad_x.sum() + grad_y.sum()) / denom
        else:
            if distortion_map is not None:
                denom = grad_x.numel() * distortion_map.mean() + grad_y.numel() * distortion_map.mean()
                loss = (grad_x.sum() + grad_y.sum()) / denom
            else:
                loss = grad_x.mean() + grad_y.mean()

        return loss * self.weight


# ------------------------------------------------------------------------------
# 6. Floater Suppression Loss  —  L_grad (Eq. 3.8)
# ------------------------------------------------------------------------------
class FloaterSuppressionLoss(nn.Module):
    """Per-Gaussian penalty that suppresses floaters near depth discontinuities.

    Paper equation (3.8):
        L_grad = E_i[ G_alpha(i) * (1 - exp(-1/σ * max{0, |∇D̄^{-1}(π(G_0(i)))| - ε})) ]

    Where:
        - G_alpha(i)  : opacity of Gaussian i
        - π(·)        : projection of Gaussian center to 2D image plane
        - ∇D̄^{-1}     : gradient magnitude of the rendered inverse-depth map
        - σ = ε = 10^{-2}

    NOTE: This loss now receives pre-computed ``projected_coords`` from the
    GaussianComposer (the ERP normalized coordinates for grid_sample), avoiding
    redundant atan2/asin/norm computation and producing a shorter, more stable
    backward graph.
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self.weight = config.floater_weight
        self.sigma = config.floater_sigma
        self.epsilon = config.floater_epsilon
        self.depth_eps = config.depth_eps  # Minimum depth to prevent 1/depth explosion in losses

    @staticmethod
    def _gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude with replicate padding."""
        grad_x = x[:, :, :, :-1] - x[:, :, :, 1:]
        grad_y = x[:, :, :-1, :] - x[:, :, 1:, :]
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode="replicate")
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode="replicate")
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    def forward(
        self,
        projected_coords: torch.Tensor,
        opacities: torch.Tensor,
        rendered_depth: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            projected_coords: [B, P, 2] pre-computed ERP coords (u_norm, v_norm) in [-1, 1].
            opacities:        [B, P]    final Gaussian opacities.
            rendered_depth:   [B, 1, H, W] rendered depth map.
        """
        inv_depth = 1.0 / (rendered_depth.clamp(min=self.depth_eps))
        grad_mag = self._gradient_magnitude(inv_depth)  # [B, 1, H, W]

        # Use pre-computed projected coordinates from GaussianComposer
        grid = projected_coords.unsqueeze(2)  # [B, P, 1, 2] for grid_sample

        # Bilinear sample; border padding clamps to edges (approximation for 360 wrap-around)
        sampled_grad = F.grid_sample(
            grad_mag, grid,
            align_corners=False,
            padding_mode="border",
        )  # [B, 1, P, 1]
        sampled_grad = sampled_grad.squeeze(1).squeeze(-1)  # [B, P]

        excess = torch.clamp(sampled_grad - self.epsilon, min=0.0)
        penalty = 1.0 - torch.exp(-excess / self.sigma)

        # Weight by per-Gaussian opacity
        loss = (opacities * penalty).mean()
        return loss * self.weight


# ------------------------------------------------------------------------------
# 7. Gaussian Offset Constraint  —  L_delta (Eq. 3.9)
# ------------------------------------------------------------------------------
class OffsetLoss(nn.Module):
    """Hinge loss constraining Gaussian position offsets from base positions.

    Paper equation (3.9):
        L_delta = E_i[ max{|ΔG_x(i)| - δ, 0} + max{|ΔG_y(i)| - δ, 0} ]

    The paper constrains only the x and y offsets (tangential directions), not
    the radial (depth) offset. We follow the paper for the tangential component
    but add an independent radial constraint because:
    1. Our frozen depth_head has local errors that need compensation.
    2. Without a radial bound, delta_z can drift arbitrarily, and once depth
       becomes very large even tiny angular deviations produce huge tangential
       offsets (||tangential|| ≈ depth * sin(delta_angle)), causing divergence.
    3. ml-sharp uses clamp_with_pushback on delta values themselves, providing
       an implicit bound we do not have.

    For 360 / ERP:
    - The base position is on a sphere: p_base = direction * depth.
    - The offset is Δp = p_final - p_base.
    - Tangential: Δp_tangential = Δp - (Δp · dir) * dir.
    - Radial:     Δp_radial     = (Δp · dir).
    - We penalize both with independent hinge thresholds.
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self.weight = config.delta_weight
        self.tangential_threshold = config.delta_threshold
        self.radial_threshold = config.delta_radial_threshold

    def forward(
        self,
        positions: torch.Tensor,
        base_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            positions:      [B, P, 3] final Gaussian centers.
            base_positions: [B, P, 3] initial centers from depth unprojection.
        """
        delta_p = positions - base_positions  # [B, P, 3]
        base_dirs = F.normalize(base_positions, dim=-1)  # [B, P, 3]
        radial = (delta_p * base_dirs).sum(dim=-1)  # [B, P]
        tangential = delta_p - radial.unsqueeze(-1) * base_dirs  # [B, P, 3]

        tangential_norm = torch.norm(tangential, dim=-1)  # [B, P]
        tangential_loss = torch.clamp(
            tangential_norm - self.tangential_threshold, min=0.0
        ).mean()

        radial_loss = torch.clamp(
            torch.abs(radial) - self.radial_threshold, min=0.0
        ).mean()

        return (tangential_loss + radial_loss) * self.weight


# ------------------------------------------------------------------------------
# 8. Projected Gaussian Variance Regularizer  —  L_splat (Eq. 3.10)
# ------------------------------------------------------------------------------
class ProjectedScaleLoss(nn.Module):
    """Hinge loss regularizing the projected 2D variance of each Gaussian.

    Paper equation (3.10):
        L_splat = E_i[ max{σ(G(i)) - σ_max, 0} + max{σ_min - σ(G(i)), 0} ]

    where σ(·) computes the projected Gaussian variance, σ_min = 10^{-1},
    σ_max = 10^{2}.

    For 360 / ERP projection, an exact 2D covariance projection requires the
    full Jacobian of the ERP mapping. We use a practical approximation:

        angular_scale  = scale_3d / depth          [radians]
        pixel_scale_x  = angular_scale_x * W / (2π)
        pixel_scale_y  = angular_scale_y * H / π
        σ(G(i))        ≈ max(pixel_scale_x, pixel_scale_y)

    This gives the approximate maximum screen-space radius of the Gaussian
    in pixels, which is the quantity we want to bound.
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self.weight = config.splat_weight
        self.sigma_min = config.splat_sigma_min
        self.sigma_max = config.splat_sigma_max

    def forward(
        self,
        depths: torch.Tensor,
        scales: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> torch.Tensor:
        """
        Args:
            depths:    [B, P, 1] per-Gaussian metric depths (pre-computed by GaussianComposer).
            scales:    [B, P, 3] final Gaussian scales (3D, metric).
        """
        # Angular extent in radians
        angular_scale = scales / depths  # [B, P, 3]

        # Convert to pixels via ERP angular-to-pixel mapping
        pixel_scale_x = angular_scale[..., 0:1] * image_width / (2.0 * math.pi)
        pixel_scale_y = angular_scale[..., 1:2] * image_height / math.pi
        pixel_scale = torch.cat([pixel_scale_x, pixel_scale_y], dim=-1)  # [B, P, 2]

        # Approximate projected variance by max screen-space extent
        sigma = pixel_scale.max(dim=-1, keepdim=True).values  # [B, P, 1]

        # Hinge losses
        upper_violation = torch.clamp(sigma - self.sigma_max, min=0.0)
        lower_violation = torch.clamp(self.sigma_min - sigma, min=0.0)

        loss = (upper_violation + lower_violation).mean()
        return loss * self.weight


# ------------------------------------------------------------------------------
# 9. Scale Regularization Loss (Depth Adjustment Map)
# ------------------------------------------------------------------------------
class ScaleRegularizationLoss(nn.Module):
    """Regularize the depth-adjustment scale map toward identity and smoothness."""

    def __init__(self, config: LossConfig):
        super().__init__()
        self.weight_l1 = config.scale_reg_weight_l1
        self.weight_tv = config.scale_reg_weight_tv
        self.num_scales = config.scale_reg_num_scales

    def forward(self, scale_map: torch.Tensor) -> torch.Tensor:
        # L1 deviation from identity
        l1_loss = torch.abs(scale_map - 1.0).mean()

        # Multiscale total variation
        tv_loss = 0.0
        current = scale_map
        for k in range(self.num_scales):
            if k > 0:
                current = F.avg_pool2d(current, kernel_size=2, stride=2)
            grad_x = torch.abs(current[:, :, :, :-1] - current[:, :, :, 1:])
            grad_y = torch.abs(current[:, :, :-1, :] - current[:, :, 1:, :])
            tv_loss += grad_x.mean() + grad_y.mean()
        tv_loss /= self.num_scales

        return self.weight_l1 * l1_loss + self.weight_tv * tv_loss


# ------------------------------------------------------------------------------
# Composite Loss Wrapper
# ------------------------------------------------------------------------------
class CelestialSplatLoss(nn.Module):
    """Composite loss that combines all SHARP loss terms.

    This is the top-level loss module used during training. It receives a
    dictionary of model outputs and a dictionary of ground-truth targets,
    computes each loss term individually, and returns both the total scalar
    loss and a dictionary of per-term values for logging.

    Expected ``outputs`` dict (produced by the model / renderer):
        - ``image``          : [B, 3, H, W] rendered RGB image.
        - ``alpha``          : [B, 1, H, W] accumulated alpha from rasterization.
        - ``depth``          : [B, 1, H, W] rendered depth map.
        - ``positions``      : [B, P, 3] final Gaussian 3D centers.
        - ``base_positions`` : [B, P, 3] initial centers from depth (for L_delta).
        - ``projected_coords``: [B, P, 2] pre-computed ERP coords for grid_sample (for L_grad).
        - ``opacities``      : [B, P] per-Gaussian opacities (for L_grad).
        - ``depths``         : [B, P, 1] per-Gaussian metric depths (for L_splat).
        - ``scales``         : [B, P, 3] final Gaussian scales.
        - ``scale_map``      : [B, 1, H, W] depth-adjustment scale map (optional).
        - ``mask``           : [B, 1, H, W] binary valid mask (optional).

    Expected ``targets`` dict (from data loader):
        - ``image``    : [B, 3, H, W] ground-truth RGB.
        - ``depth``    : [B, 1, H, W] ground-truth metric depth.

    Args:
        config: LossConfig instance.
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self.image_height = config.image_height
        self.image_width = config.image_width

        self.l1_color = L1ColorLoss(config)
        self.perceptual = PerceptualLoss(config)
        self.disparity = InverseDepthLoss(config)
        self.alpha_bce = AlphaBCELoss(config)
        self.tv = TotalVariationLoss(config)
        self.floater = FloaterSuppressionLoss(config)
        self.offset = OffsetLoss(config)
        self.splat = ProjectedScaleLoss(config)
        self.scale_reg = ScaleRegularizationLoss(config) if config.enable_scale_reg else None

        self.perceptual_on_novel_only = config.perceptual_on_novel_only
        self.enable_print = config.enable_print

        # Precompute ERP distortion-aware weight map (DAP paper)
        # Equatorial regions have higher pixel density → higher weight
        if config.use_distortion_map:
            v = torch.arange(config.image_height, dtype=torch.float32)
            phi = (v / max(config.image_height - 1, 1) - 0.5) * math.pi  # [-pi/2, pi/2]
            M = torch.cos(phi).view(config.image_height, 1)  # [H, 1]
            M = M / M.mean()  # Normalize to mean=1
            self.register_buffer("distortion_map", M)  # [H, 1]
        else:
            self.distortion_map = None

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        pred_depth: Optional[torch.Tensor] = None,
        is_novel_view: bool = False,
        nonsky_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute composite loss.

        Args:
            outputs: Model output dictionary (see class docstring).
            targets: Ground-truth dictionary with ``image`` and ``depth``.
            pred_depth: [B, C, H, W] DPT-predicted multi-layer depth (C=2 for
                duplicate_depth). If provided, disparity loss is applied to
                Layer-1 and TV loss to Layer-2, matching ml-sharp paper §3.4.
                If None, falls back to rendered depth (legacy behavior).
            is_novel_view: If True, this is a novel-view rendering. Used to
                optionally disable perceptual loss on input views per paper.
            nonsky_mask: [B, 1, H, W] DAP-predicted non-sky mask. Used for
                alpha_bce to tell the model which pixels should have Gaussians.
                If None, falls back to the GT depth mask.

        Returns:
            total_loss: Scalar tensor for backpropagation.
            loss_dict : Dictionary of individual loss values for logging.
        """
        mask = outputs.get("mask")
        loss_dict: Dict[str, torch.Tensor] = {}

        # --- Reconstruction ---
        loss_dict["l1_color"] = self.l1_color(
            outputs["image"], targets["image"], mask=mask,
            distortion_map=self.distortion_map,
        )

        # Paper: perceptual loss only on novel views to encourage plausible inpainting.
        if not (self.perceptual_on_novel_only and not is_novel_view):
            loss_dict["perceptual"] = self.perceptual(
                outputs["image"], targets["image"]
            )
        else:
            # Perceptual loss disabled on input views; return 0-tensor for logging
            loss_dict["perceptual"] = outputs["image"].sum() * 0.0

        # --- Geometric ---
        if is_novel_view:
            # Novel view: no DAP pred_depth available; do not compute depth supervision.
            # Using rendered_depth for disparity produces bogus gradients because
            # uncovered pixels have depth=0 → 1/clamp=1000 explosion.
            loss_dict["disparity"] = outputs["image"].sum() * 0.0
            loss_dict["tv_depth"] = outputs["image"].sum() * 0.0
        elif pred_depth is not None:
            # ml-sharp: L_depth only on Layer-1, L_tv only on Layer-2, input view only.
            loss_dict["disparity"] = self.disparity(
                pred_depth[:, 0:1], targets["depth"], mask=mask,
                distortion_map=self.distortion_map,
            )
            loss_dict["tv_depth"] = self.tv(
                pred_depth[:, 1:2], mask=mask,
                distortion_map=self.distortion_map,
            )
        else:
            # Fallback: use rendered depth (both layers mixed)
            loss_dict["disparity"] = self.disparity(
                outputs["depth"], targets["depth"], mask=mask,
                distortion_map=self.distortion_map,
            )
            loss_dict["tv_depth"] = self.tv(
                outputs["depth"], mask=mask,
                distortion_map=self.distortion_map,
            )

        # Alpha BCE: use DAP's nonsky_mask if available, otherwise GT mask.
        # nonsky_mask tells us which pixels should have Gaussians (1=valid, 0=sky).
        alpha_mask = nonsky_mask if nonsky_mask is not None else mask
        loss_dict["alpha_bce"] = self.alpha_bce(
            outputs["alpha"], mask=alpha_mask,
            distortion_map=self.distortion_map,
        )

        loss_dict["floater"] = self.floater(
            projected_coords=outputs["projected_coords"],
            opacities=outputs["opacities"],
            rendered_depth=outputs["depth"],
        )

        # --- Gaussian Constraints ---
        if "base_positions" in outputs:
            loss_dict["offset"] = self.offset(
                outputs["positions"], outputs["base_positions"]
            )
        else:
            loss_dict["offset"] = outputs["image"].sum() * 0.0

        loss_dict["splat"] = self.splat(
            depths=outputs["depths"],
            scales=outputs["scales"],
            image_height=self.image_height,
            image_width=self.image_width,
        )

        # --- Scale Regularization (only when learned_local scaler is active) ---
        if self.scale_reg is not None and "scale_map" in outputs:
            loss_dict["scale_reg"] = self.scale_reg(outputs["scale_map"])
        else:
            loss_dict["scale_reg"] = outputs["image"].sum() * 0.0

        total_loss = sum(loss_dict.values())
        loss_dict["total"] = total_loss

        if self.enable_print:
            print(f"Loss breakdown:", end=" ")
            for key, value in loss_dict.items():
                print(f" {key}: {value.item():.4f}", end="; ")
            print()  # New line at the end

        return total_loss, loss_dict
