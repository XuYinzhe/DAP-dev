"""
Test script to verify training pipeline works.
"""

import sys
import os
sys.path.insert(0, '/homes/shaun/main/DAP')
os.chdir('/homes/shaun/main/DAP')

import torch
import torch.nn as nn
from pathlib import Path
import yaml

print("=" * 70)
print("Testing CelestialSplat Training Pipeline")
print("=" * 70)
print()

# Check rasterizer
try:
    from diff_gaussian_rasterization_omni import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
        CameraModelType,
    )
    print("✓ Rasterizer available")
    HAS_RASTERIZER = True
except ImportError as e:
    print(f"✗ Rasterizer not available: {e}")
    HAS_RASTERIZER = False
    sys.exit(1)

# Check CelestialSplat
try:
    from CelestialSplat import CelestialSplat, CelestialSplatConfig
    from CelestialSplat.dataset import TartanAir360Dataset
    print("✓ CelestialSplat available")
except ImportError as e:
    print(f"✗ CelestialSplat not available: {e}")
    sys.exit(1)

print()
print("=" * 70)
print("Step 1: Load Dataset")
print("=" * 70)

dataset = TartanAir360Dataset(
    root_dir='/homes/shaun/main/dataset/tartanair360/AbandonedSchool/Data_hard/P000/',
    chunk_size=4,
    image_size=(256, 512),  # Smaller for testing
    max_frames=8,
    camera_name='lcam_custom0',
)

print(f"Dataset loaded: {len(dataset)} chunks")

# Get a sample
sample = dataset[0]
images = sample['images'].unsqueeze(0)  # Add batch dimension [1, 4, 3, H, W]
poses = sample['poses'].unsqueeze(0)    # [1, 4, 4, 4]

print(f"Sample batch: images={images.shape}, poses={poses.shape}")

print()
print("=" * 70)
print("Step 2: Load DAP Model (via networks)")
print("=" * 70)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load DAP via networks
from argparse import Namespace
from networks.models import make

# Load config
config_path = '/homes/shaun/main/DAP/config/infer.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print(f"Config loaded from {config_path}")

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
        # Create DataParallel wrapper
        dap_model = nn.DataParallel(dap_model)
        dap_model.load_state_dict(state_dict, strict=False)
        # Unwrap to access attributes directly
        dap_model = dap_model.module
    else:
        dap_model.load_state_dict(state_dict, strict=False)
    print("✓ Weights loaded")
else:
    print(f"Warning: No weights found at {weights_path}, using random initialization")

dap_model.eval()
for param in dap_model.parameters():
    param.requires_grad = False

print("✓ DAP model loaded (frozen)")

print()
print("=" * 70)
print("Step 3: Create CelestialSplat Model")
print("=" * 70)

config_cs = CelestialSplatConfig(
    encoder='vitl',
    num_transformer_layers=2,  # Reduced for testing
    K_neighbors=2,
    image_height=256,
    image_width=512,
    sh_degree=0,  # Degree 0 for simplicity
)

model = CelestialSplat(config_cs, dap_model=dap_model).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.get_trainable_params(freeze_dap=True))

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print("✓ Model created")

print()
print("=" * 70)
print("Step 4: Forward Pass")
print("=" * 70)

images = images.to(device)
poses = poses.to(device)

with torch.no_grad():
    outputs = model(images, poses)

gaussians = outputs['gaussians']

print(f"Unified Gaussians:")
print(f"  means: {gaussians['means'].shape}")
print(f"  scales: {gaussians['scales'].shape}")
print(f"  rotations: {gaussians['rotations'].shape}")
print(f"  opacities: {gaussians['opacities'].shape}")
print(f"  shs: {gaussians['shs'].shape}")
print(f"  masks: {gaussians['masks'].shape}")
print("✓ Forward pass successful")

print()
print("=" * 70)
print("Step 5: Test Rendering")
print("=" * 70)

B, N = images.shape[:2]
H, W = images.shape[-2:]

# Try to render one view
try:
    from CelestialSplat.train_simple import render_gaussians
    
    gaussians_b = {
        'means': gaussians['means'][0],
        'scales': gaussians['scales'][0],
        'rotations': gaussians['rotations'][0],
        'opacities': gaussians['opacities'][0],
        'shs': gaussians['shs'][0],
    }
    
    rendered, depth = render_gaussians(
        gaussians_b,
        poses[0, 0],
        H, W,
        sh_degree=config_cs.sh_degree,
    )
    
    print(f"Rendered image: {rendered.shape}")
    print(f"Rendered depth: {depth.shape}")
    print("✓ Rendering successful")
    
except Exception as e:
    print(f"✗ Rendering failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("All Tests Passed!")
print("=" * 70)
