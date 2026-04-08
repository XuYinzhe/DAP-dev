"""
Integration script for loading pretrained DAP into CelestialSplat.

This shows how to:
1. Load pretrained DAP model
2. Integrate it with CelestialSplat
3. Fine-tune with different freezing strategies
"""

import torch
import torch.nn as nn
from typing import Optional

from depth_anything_v2_metric.depth_anything_v2.dpt import DepthAnythingV2
from .model import CelestialSplat, CelestialSplatConfig


def load_dap_model(
    encoder: str = 'vitl',
    max_depth: float = 10.0,
    weights_path: Optional[str] = None,
    device: str = 'cuda'
) -> DepthAnythingV2:
    """
    Load pretrained DAP model.
    
    Args:
        encoder: Encoder type ('vits', 'vitb', 'vitl', 'vitg')
        max_depth: Maximum depth value
        weights_path: Path to pretrained weights (optional)
        device: Device to load model on
    
    Returns:
        DepthAnythingV2 model
    """
    model = DepthAnythingV2(
        encoder=encoder,
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        max_depth=max_depth
    )
    
    if weights_path is not None:
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint.get('model', checkpoint), strict=False)
        print(f"Loaded DAP weights from {weights_path}")
    
    model = model.to(device)
    return model


def build_celestial_splat_with_dap(
    encoder: str = 'vitl',
    max_depth: float = 10.0,
    dap_weights_path: Optional[str] = None,
    num_transformer_layers: int = 6,
    K_neighbors: int = 4,
    device: str = 'cuda'
) -> CelestialSplat:
    """
    Build CelestialSplat model with pretrained DAP backbone.
    
    Args:
        encoder: DAP encoder type
        max_depth: Maximum depth value
        dap_weights_path: Path to DAP pretrained weights
        num_transformer_layers: Number of transformer layers
        K_neighbors: Number of neighbor views for cross-attention
        device: Device to load model on
    
    Returns:
        CelestialSplat model with integrated DAP
    """
    # Load DAP model
    print("Loading DAP model...")
    dap_model = load_dap_model(
        encoder=encoder,
        max_depth=max_depth,
        weights_path=dap_weights_path,
        device=device
    )
    
    # Determine embed dim based on encoder
    embed_dim = 1024 if encoder in ['vitl', 'vitg'] else 768
    
    # Create config
    config = CelestialSplatConfig(
        encoder=encoder,
        dino_embed_dim=embed_dim,
        max_depth=max_depth,
        num_transformer_layers=num_transformer_layers,
        K_neighbors=K_neighbors
    )
    
    # Build CelestialSplat
    print("Building CelestialSplat model...")
    model = CelestialSplat(config, dap_model=dap_model)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    dap_params = sum(p.numel() for p in dap_model.parameters())
    celestial_params = total_params - dap_params
    
    print(f"\nModel Statistics:")
    print(f"  DAP parameters:      {dap_params:,}")
    print(f"  CelestialSplat new:  {celestial_params:,}")
    print(f"  Total parameters:    {total_params:,}")
    
    return model


def get_training_strategy(model: CelestialSplat, strategy: str = 'warmup'):
    """
    Get parameter groups for different training strategies.
    
    Args:
        model: CelestialSplat model
        strategy: One of ['warmup', 'single_view', 'multi_view', 'finetune']
    
    Returns:
        List of parameter groups with different learning rates
    """
    if strategy == 'warmup':
        # Freeze DINOv3, train DPT heads and new modules
        for param in model.dap.pretrained.parameters():
            param.requires_grad = False
        
        param_groups = [
            {'params': model.dap.depth_head.parameters(), 'lr': 1e-4, 'name': 'depth_head'},
            {'params': model.dap.mask_head.parameters(), 'lr': 1e-4, 'name': 'mask_head'},
            {'params': model.feature_adapter.parameters(), 'lr': 1e-4, 'name': 'feature_adapter'},
            {'params': model.transformer.parameters(), 'lr': 1e-4, 'name': 'transformer'},
            {'params': model.gs_decoder.parameters(), 'lr': 1e-4, 'name': 'gs_decoder'},
        ]
        print("Training strategy: Warmup (DINOv3 frozen)")
    
    elif strategy == 'single_view':
        # Same as warmup but focus on single-view GS
        for param in model.dap.pretrained.parameters():
            param.requires_grad = False
        
        param_groups = [
            {'params': model.dap.depth_head.parameters(), 'lr': 1e-4},
            {'params': model.dap.mask_head.parameters(), 'lr': 1e-4},
            {'params': model.feature_adapter.parameters(), 'lr': 1e-4},
            {'params': model.gs_decoder.parameters(), 'lr': 1e-4},
        ]
        print("Training strategy: Single-View (Transformer frozen)")
    
    elif strategy == 'multi_view':
        # Unfreeze last 6 layers of DINOv3
        total_layers = len(model.dap.pretrained.blocks) if hasattr(model.dap.pretrained, 'blocks') else 24
        
        for i, param in enumerate(model.dap.pretrained.parameters()):
            # Heuristic: unfreeze last 20% of layers
            param.requires_grad = False
        
        # Try to unfreeze specific blocks if architecture allows
        if hasattr(model.dap.pretrained, 'blocks'):
            for block in model.dap.pretrained.blocks[-6:]:
                for param in block.parameters():
                    param.requires_grad = True
        
        param_groups = [
            {'params': model.dap.pretrained.parameters(), 'lr': 1e-5},  # Lower LR for DINOv3
            {'params': model.dap.depth_head.parameters(), 'lr': 1e-4},
            {'params': model.dap.mask_head.parameters(), 'lr': 1e-4},
            {'params': model.feature_adapter.parameters(), 'lr': 1e-4},
            {'params': model.transformer.parameters(), 'lr': 5e-5},
            {'params': model.gs_decoder.parameters(), 'lr': 1e-4},
        ]
        print("Training strategy: Multi-View (DINOv3 last layers unfrozen)")
    
    elif strategy == 'finetune':
        # Full fine-tuning
        for param in model.parameters():
            param.requires_grad = True
        
        param_groups = [
            {'params': model.dap.pretrained.parameters(), 'lr': 1e-6},  # Very low LR
            {'params': model.dap.depth_head.parameters(), 'lr': 5e-5},
            {'params': model.dap.mask_head.parameters(), 'lr': 5e-5},
            {'params': model.feature_adapter.parameters(), 'lr': 5e-5},
            {'params': model.transformer.parameters(), 'lr': 5e-5},
            {'params': model.gs_decoder.parameters(), 'lr': 5e-5},
        ]
        print("Training strategy: Full Fine-tune")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Filter out groups with no trainable parameters
    param_groups = [
        g for g in param_groups 
        if any(p.requires_grad for p in g['params'])
    ]
    
    return param_groups


# Example usage
if __name__ == '__main__':
    # Build model with pretrained DAP
    model = build_celestial_splat_with_dap(
        encoder='vitl',
        max_depth=10.0,
        dap_weights_path=None,  # Set to path if you have weights
        num_transformer_layers=6,
        K_neighbors=4,
        device='cuda'
    )
    
    # Test forward pass
    B, N = 2, 4
    H, W = 512, 1024
    
    images = torch.randn(B, N, 3, H, W).cuda()
    poses = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1).cuda()
    
    print("\nTesting forward pass...")
    with torch.no_grad():
        outputs = model(images, poses)
    
    print("Output keys:", list(outputs.keys()))
    print("Gaussian parameters:", list(outputs['gaussians'].keys()))
    
    # Test different training strategies
    print("\n" + "=" * 60)
    for strategy in ['warmup', 'single_view', 'multi_view', 'finetune']:
        param_groups = get_training_strategy(model, strategy)
        total_trainable = sum(
            sum(p.numel() for p in g['params'] if p.requires_grad)
            for g in param_groups
        )
        print(f"{strategy:15s}: {total_trainable:,} trainable parameters")
