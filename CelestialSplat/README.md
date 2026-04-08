# CelestialSplat

**Feed-forward 360° Gaussian Splatting with Cross-View Attention**

CelestialSplat is a transformer-based sparse 360° panorama 3D Gaussian Splatting reconstruction system. It takes sparse 360° image sequences as input and leverages cross-view attention mechanisms to fuse geometric features from neighboring frames.

## Architecture Overview

```
Input: N ERP Images [B, N, 3, H, W]
         │
         ▼
┌─────────────────────────────────────────┐
│  Siamese DINOv3 Encoders (DAP Backbone) │
│  • Depth prediction                     │
│  • Mask prediction (sky segmentation)   │
│  • 4-layer feature extraction           │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  DAPFeatureAdapter                      │
│  Aggregates 4-layer features [B,N,256,H/16,W/16]
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  CrossViewTransformer                   │
│  • Geometry-guided cross-attention      │
│  • K-neighbor view fusion (K=2~4)       │
│  • 6 transformer layers                 │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  GSDecoder                              │
│  Predicts per-pixel Gaussian params:    │
│  • Depth residual (Δr)                  │
│  • Covariance (σ_x, σ_y, σ_z)           │
│  • Rotation quaternion                  │
│  • Opacity (α)                          │
│  • SH color (27 channels for deg 2)     │
│  • Confidence                           │
└─────────────────────────────────────────┘
         │
         ▼
    3D Gaussians
```

## Files

```
CelestialSplat/
├── __init__.py          # Package initialization
├── model.py             # Main model architecture
│   ├── CelestialSplatConfig
│   ├── DAPFeatureAdapter
│   ├── CrossViewTransformer
│   ├── GSDecoder
│   └── CelestialSplat (main model)
├── integrate_dap.py     # DAP integration utilities
└── example_train.py     # Training example
```

## Quick Start

### 1. Basic Usage with Dummy DAP

```python
from CelestialSplat import CelestialSplat, CelestialSplatConfig

# Create config
config = CelestialSplatConfig(
    encoder='vitl',
    num_transformer_layers=6,
    K_neighbors=4
)

# Build model (with your DAP model)
model = CelestialSplat(config, dap_model=your_dap_model)

# Forward pass
images = torch.randn(B, N, 3, H, W)  # ERP images
poses = torch.randn(B, N, 4, 4)      # Camera poses

outputs = model(images, poses)
gaussians = outputs['gaussians']
```

### 2. Integration with Real DAP

```python
from CelestialSplat import build_celestial_splat_with_dap

# Build with pretrained DAP
model = build_celestial_splat_with_dap(
    encoder='vitl',
    max_depth=10.0,
    dap_weights_path='path/to/dap/weights.pth',
    num_transformer_layers=6,
    K_neighbors=4,
    device='cuda'
)
```

### 3. Training Strategy

```python
from CelestialSplat import get_training_strategy

# Phase 1: Warmup (freeze DINOv3)
param_groups = get_training_strategy(model, 'warmup')
optimizer = torch.optim.Adam(param_groups)

# Phase 2: Single-view training
param_groups = get_training_strategy(model, 'single_view')

# Phase 3: Multi-view with partial DINOv3 unfrozen
param_groups = get_training_strategy(model, 'multi_view')

# Phase 4: Full fine-tuning
param_groups = get_training_strategy(model, 'finetune')
```

## Model Components

### DAPFeatureAdapter

Aggregates 4-layer DINOv3 features into a unified representation:
- Input: List of 4 tensors `[B, 1024, H/16, W/16]`
- Output: `[B, 256, H/16, W/16]`

### CrossViewTransformer

Geometry-guided cross-view attention:
- Each layer: Self-Attention → Cross-View Attention → FFN
- Uses 3D point cloud from depth for geometric guidance
- K-neighbor view aggregation (K=2~4)

### GSDecoder

Predicts Gaussian parameters per pixel:
- Depth residual: DAP prior + predicted delta
- Covariance: 3D scale values
- Rotation: Quaternion (normalized)
- Opacity: Sigmoid activation
- SH Color: 27 channels (degree 2)
- Confidence: For quality weighting

## Configuration

```python
@dataclass
class CelestialSplatConfig:
    # DAP backbone
    encoder: str = 'vitl'           # 'vits', 'vitb', 'vitl', 'vitg'
    max_depth: float = 10.0
    
    # Feature adapter
    dino_embed_dim: int = 1024      # 768 for vitb, 1024 for vitl
    adapter_out_dim: int = 256
    
    # Transformer
    transformer_dim: int = 256
    num_transformer_layers: int = 6
    num_heads: int = 8
    K_neighbors: int = 4
    
    # GS Decoder
    decoder_hidden_dim: int = 128
    sh_degree: int = 2
```

## Training Pipeline

### Phase 1: Warmup (Epoch 1-10)
- Freeze DINOv3 encoder
- Train DPT heads, FeatureAdapter, Transformer, GS Decoder
- Learning rate: 1e-4

### Phase 2: Single-View (Epoch 11-30)
- Continue with frozen DINOv3
- Focus on per-view Gaussian quality
- Learning rate: 1e-4

### Phase 3: Multi-View (Epoch 31-70)
- Unfreeze last 6 layers of DINOv3
- Enable cross-view attention training
- Learning rate: 5e-5 for new modules, 1e-5 for DINOv3

### Phase 4: Fine-tuning (Epoch 71-100)
- Full network fine-tuning
- Very low learning rate for DINOv3: 1e-6
- Learning rate: 5e-5 for other modules

## Testing

```bash
# Test model shapes and forward pass
python test_celestial_splat.py

# Test training loop
python CelestialSplat/example_train.py
```

## Output Format

The model outputs a dictionary with Gaussian parameters:

```python
{
    'gaussians': {
        'depth': [B, N, H, W],          # Final depth (DAP + residual)
        'covariance': [B, N, 3, H, W],   # Scale values
        'rotation': [B, N, 4, H, W],     # Quaternion
        'opacity': [B, N, H, W],         # Alpha values
        'sh_color': [B, N, 27, H, W],    # SH coefficients
        'confidence': [B, N, H, W],      # Quality weights
    },
    'dap_depth': [B, N, H, W],          # DAP depth prior
    'dap_mask': [B, N, H, W],           # Sky mask
}
```

## Notes

- Input should be ERP (equirectangular) format 360° images
- Camera poses should be world-to-camera transformation matrices
- DAP model should be loaded with pretrained weights for best results
- Consider gradient checkpointing for large number of views (N>8)

## Citation

```bibtex
@article{celestialsplat2025,
  title={CelestialSplat: Feed-forward 360° Gaussian Splatting with Cross-View Attention},
  year={2025}
}
```
