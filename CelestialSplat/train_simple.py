"""
Simple training script for CelestialSplat.

Trains on a single TartanAir sequence using:
- L1 + SSIM loss (no GT depth)
- Differentiable Gaussian Rasterizer (ERP / omnidirectional)
"""

import sys
import os
sys.path.insert(0, '/homes/shaun/main/DAP')
os.chdir('/homes/shaun/main/DAP')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, Tuple, List
import math

# Import diff-gaussian-rasterization-omni
try:
    from diff_gaussian_rasterization_omni import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
        CameraModelType,
    )
    HAS_RASTERIZER = True
    print("✓ diff_gaussian_rasterization_omni imported successfully")
except ImportError as e:
    print(f"Warning: Could not import diff_gaussian_rasterization_omni: {e}")
    print("Training will not work without the rasterizer.")
    HAS_RASTERIZER = False

from CelestialSplat import CelestialSplat, CelestialSplatConfig
from CelestialSplat.dataset import TartanAir360Dataset, create_dataloaders


def build_rasterizer_settings(H: int, W: int, pose: torch.Tensor, sh_degree: int = 0):
    """
    Build rasterization settings for ERP camera.
    
    Args:
        H, W: Image dimensions
        pose: [4, 4] camera pose (world-to-camera)
        sh_degree: Spherical harmonics degree
    
    Returns:
        GaussianRasterizationSettings
    """
    # Background color (black)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')
    
    # Camera pose to view matrix and projection
    # viewmatrix is world-to-camera transformation
    viewmatrix = pose.clone()
    
    # For ERP, we use a simple projection matrix (identity-like)
    # The actual projection is handled by the LONLAT camera model
    projmatrix = torch.eye(4, dtype=torch.float32, device='cuda')
    
    # RwcT: rotation matrix from world to camera, transposed
    RwcT = pose[:3, :3].T
    
    # Camera position in world coordinates
    campos = -pose[:3, :3].T @ pose[:3, 3]
    
    # For ERP, tanfovx and tanfovy are not used but need to be provided
    tanfovx = 1.0
    tanfovy = 1.0
    
    return GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        RwcT=RwcT,
        sh_degree=sh_degree,
        campos=campos,
        prefiltered=False,
        camera_type=CameraModelType.LONLAT,
        render_depth=False,
    )


def render_gaussians(
    gaussians: Dict[str, torch.Tensor],
    pose: torch.Tensor,
    H: int,
    W: int,
    sh_degree: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Render Gaussians to image.
    
    Args:
        gaussians: Dict with unified Gaussian parameters
            - means: [P, 3]
            - scales: [P, 3]
            - rotations: [P, 4]
            - opacities: [P, 1]
            - shs: [P, K, 3]
        pose: [4, 4] camera pose
        H, W: Image dimensions
        sh_degree: Spherical harmonics degree
    
    Returns:
        rendered_image: [3, H, W]
        rendered_depth: [1, H, W]
    """
    if not HAS_RASTERIZER:
        raise RuntimeError("diff_gaussian_rasterization_omni not available")
    
    # Build rasterization settings
    raster_settings = build_rasterizer_settings(H, W, pose, sh_degree)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Prepare Gaussian parameters
    means3D = gaussians['means']  # [P, 3]
    scales = gaussians['scales']  # [P, 3]
    rotations = gaussians['rotations']  # [P, 4]
    opacities = gaussians['opacities']  # [P, 1]
    shs = gaussians['shs']  # [P, K, 3]
    
    # Placeholder for means2D (for gradient computation)
    means2D = torch.zeros_like(means3D)
    
    # Render
    rendered_image, rendered_depth, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacities,
        shs=shs,
        scales=scales,
        rotations=rotations,
    )
    
    return rendered_image, rendered_depth


def ssim_loss(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Compute SSIM loss between two images.
    
    Args:
        img1, img2: [3, H, W] images
        window_size: Size of Gaussian window
    
    Returns:
        SSIM loss (1 - SSIM)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Create Gaussian window
    sigma = 1.5
    gauss = torch.tensor([
        math.exp(-(x - window_size//2)**2 / (2 * sigma**2))
        for x in range(window_size)
    ], dtype=torch.float32, device=img1.device)
    gauss = gauss / gauss.sum()
    
    # 2D Gaussian window
    window_1d = gauss.unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(3, 1, window_size, window_size).contiguous()
    
    # Compute means
    mu1 = F.conv2d(img1.unsqueeze(0), window, padding=window_size//2, groups=3)
    mu2 = F.conv2d(img2.unsqueeze(0), window, padding=window_size//2, groups=3)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = F.conv2d(img1.unsqueeze(0) * img1.unsqueeze(0), window, padding=window_size//2, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(img2.unsqueeze(0) * img2.unsqueeze(0), window, padding=window_size//2, groups=3) - mu2_sq
    sigma12 = F.conv2d(img1.unsqueeze(0) * img2.unsqueeze(0), window, padding=window_size//2, groups=3) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return 1 - ssim_map.mean()


def compute_loss(
    rendered_images: List[torch.Tensor],
    target_images: torch.Tensor,
    lambda_l1: float = 1.0,
    lambda_ssim: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute training loss.
    
    Args:
        rendered_images: List of [3, H, W] rendered images (one per view)
        target_images: [N, 3, H, W] target images
        lambda_l1: Weight for L1 loss
        lambda_ssim: Weight for SSIM loss
    
    Returns:
        total_loss: Total loss
        loss_dict: Dictionary of individual losses
    """
    total_loss = 0
    loss_dict = {}
    
    l1_loss_total = 0
    ssim_loss_total = 0
    
    for i, rendered in enumerate(rendered_images):
        target = target_images[i]  # [3, H, W]
        
        # L1 loss
        l1 = F.l1_loss(rendered, target)
        l1_loss_total += l1
        
        # SSIM loss
        ssim = ssim_loss(rendered, target)
        ssim_loss_total += ssim
    
    # Average over views
    num_views = len(rendered_images)
    l1_loss_total /= num_views
    ssim_loss_total /= num_views
    
    # Total loss
    total_loss = lambda_l1 * l1_loss_total + lambda_ssim * ssim_loss_total
    
    loss_dict = {
        'total': total_loss.item(),
        'l1': l1_loss_total.item(),
        'ssim': ssim_loss_total.item(),
    }
    
    return total_loss, loss_dict


def load_dap_model(config_path: str = 'config/infer.yaml', device: str = 'cuda'):
    """Load DAP model via networks."""
    import yaml
    from argparse import Namespace
    from networks.models import make
    
    print(f"Loading DAP model from config: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model using make
    dap_model = make(config['model'])
    dap_model = dap_model.to(device)
    
    # Load weights if available
    weights_path = os.path.join(config['load_weights_dir'], 'model.pth')
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        
        # Handle DataParallel wrapper
        if any(k.startswith("module") for k in state_dict.keys()):
            dap_model = nn.DataParallel(dap_model)
            dap_model.load_state_dict(state_dict, strict=False)
            dap_model = dap_model.module  # Unwrap
        else:
            dap_model.load_state_dict(state_dict, strict=False)
        print("✓ Weights loaded")
    else:
        print(f"Warning: No weights found at {weights_path}")
    
    dap_model.eval()
    for param in dap_model.parameters():
        param.requires_grad = False
    
    return dap_model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: str = 'cuda',
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_l1 = 0
    total_ssim = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['images'].to(device)  # [B, N, 3, H, W]
        poses = batch['poses'].to(device)    # [B, N, 4, 4]
        
        B, N = images.shape[:2]
        H, W = images.shape[-2:]
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images, poses)
        
        # Get unified Gaussians
        gaussians = outputs['gaussians']
        
        # Render each view
        rendered_images = []
        for b in range(B):
            for n in range(N):
                # Extract Gaussians for this batch item
                gaussians_b = {
                    'means': gaussians['means'][b],
                    'scales': gaussians['scales'][b],
                    'rotations': gaussians['rotations'][b],
                    'opacities': gaussians['opacities'][b],
                    'shs': gaussians['shs'][b],
                }
                
                # Render
                rendered, _ = render_gaussians(
                    gaussians_b,
                    poses[b, n],
                    H, W,
                    sh_degree=model.config.sh_degree,
                )
                rendered_images.append(rendered)
        
        # Compute loss
        target_images = images.view(B * N, 3, H, W)
        loss, loss_dict = compute_loss(rendered_images, target_images)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss_dict['total']
        total_l1 += loss_dict['l1']
        total_ssim += loss_dict['ssim']
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"  Batch [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss_dict['total']:.4f} "
                  f"(L1: {loss_dict['l1']:.4f}, SSIM: {loss_dict['ssim']:.4f})")
    
    return {
        'loss': total_loss / num_batches,
        'l1': total_l1 / num_batches,
        'ssim': total_ssim / num_batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default='/homes/shaun/main/dataset/tartanair360/',
                        help='Root directory containing all TartanAir sequences '
                             '(e.g., /path/to/tartanair360/)')
    parser.add_argument('--config', type=str, default='config/infer.yaml',
                        help='Path to DAP config file')
    parser.add_argument('--chunk_size', type=int, default=4)
    parser.add_argument('--chunk_stride', type=int, default=2)
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 1024])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_sequences', type=int, default=None,
                        help='Limit to first N sequences (for debugging)')
    parser.add_argument('--sh_degree', type=int, default=0, choices=[0, 1, 2, 3],
                        help='Spherical harmonics degree (0=RGB only, faster)')
    args = parser.parse_args()
    
    if not HAS_RASTERIZER:
        print("Error: diff_gaussian_rasterization_omni is required for training")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load DAP model
    dap_model = load_dap_model(config_path=args.config, device=device)
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = TartanAir360Dataset(
        root_dir=args.data_dir,
        image_size=tuple(args.image_size),
        mode='train',
        num_sequences=args.num_sequences,
        chunk_size=args.chunk_size,
        chunk_stride=args.chunk_stride,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create model
    print("\nCreating model...")
    
    # Get encoder from DAP config
    import yaml
    with open(args.config, 'r') as f:
        dap_config = yaml.safe_load(f)
    encoder = dap_config['model']['args']['midas_model_type']
    
    config = CelestialSplatConfig(
        encoder=encoder,
        num_transformer_layers=4,
        K_neighbors=min(4, args.chunk_size - 1),
        image_height=args.image_size[0],
        image_width=args.image_size[1],
        sh_degree=args.sh_degree,
    )
    
    model = CelestialSplat(config, dap_model=dap_model).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.get_trainable_params(freeze_dap=True))
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.get_trainable_params(freeze_dap=True),
        lr=args.lr,
    )
    
    # Training loop
    print(f"\nTraining for {args.num_epochs} epochs...")
    print("=" * 70)
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 70)
        
        metrics = train_one_epoch(model, dataloader, optimizer, epoch, device)
        
        print(f"Epoch {epoch + 1} summary: "
              f"Loss: {metrics['loss']:.4f} "
              f"(L1: {metrics['l1']:.4f}, SSIM: {metrics['ssim']:.4f})")
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()
