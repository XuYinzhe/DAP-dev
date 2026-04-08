# CelestialSplat 模型结构图与参数量分析

## 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CelestialSplat Architecture                            │
│              360° Gaussian Splatting with Cross-View Attention                   │
│                   Unified World-Coordinate Gaussian Representation               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  INPUT: ERP Images [B, N, 3, H, W]  (e.g., B=2, N=4, H=512, W=1024)             │
│                           │                                                     │
│                           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐       │
│  │ DAP BACKBONE (Pretrained, ~360M params, ~1.4 GB)                   │       │
│  │ ┌─────────────────────────────────────────────────────────────┐    │       │
│  │ │ DINOv3 Encoder (ViT-L)                                       │    │       │
│  │ │ • 24 Transformer Blocks                                       │    │       │
│  │ │ • Patch Size: 16×16                                           │    │       │
│  │ │ • Embed Dim: 1024                                             │    │       │
│  │ │ • Output: 4 intermediate features [B*N, 1024, 32, 64]         │    │       │
│  │ │                                                               │    │       │
│  │ │ Params: ~300M  │  Memory: ~1.2 GB                            │    │       │
│  │ └─────────────────────────────────────────────────────────────┘    │       │
│  │                              │                                     │       │
│  │                              ▼                                     │       │
│  │              ┌─────────────────────────────┐                       │       │
│  │              │ DPT Depth Head              │                       │       │
│  │              │ • Projects (4 convs)        │                       │       │
│  │              │ • RefineNets (4 layers)     │                       │       │
│  │              │ • Output: [B*N, 1, H, W]    │                       │       │
│  │              │                             │                       │       │
│  │              │ Params: ~30M  │  Memory: ~120 MB                   │       │
│  │              └─────────────────────────────┘                       │       │
│  │                              │                                     │       │
│  │                              ▼                                     │       │
│  │              ┌─────────────────────────────┐                       │       │
│  │              │ DPT Mask Head               │                       │       │
│  │              │ (Same architecture)         │                       │       │
│  │              │                             │                       │       │
│  │              │ Params: ~30M  │  Memory: ~120 MB                   │       │
│  │              └─────────────────────────────┘                       │       │
│  │                                                                    │       │
│  │ DAP Output:                                                        │       │
│  │   - depth: [B, N, H, W]           (depth prior)                    │       │
│  │   - mask:  [B, N, H, W]           (sky segmentation)               │       │
│  │   - features: 4×[B, 1024, H/16, W/16] (intermediate features)      │       │
│  └────────────────────────────────────────────────────────────────────┘       │
│                           │                                                     │
│                           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐       │
│  │ 1. DAP FEATURE ADAPTER  (525K params, 2.0 MB)                      │       │
│  │                                                                    │       │
│  │   Input:  4×[B, 1024, H/16, W/16] (DINOv3 intermediate features)   │       │
│  │          │                                                         │       │
│  │          ├─> patch_projs[0] ──> [B, 64, H/16, W/16]                │       │
│  │          ├─> patch_projs[1] ──> [B, 64, H/16, W/16]                │       │
│  │          ├─> patch_projs[2] ──> [B, 64, H/16, W/16]                │       │
│  │          └─> patch_projs[3] ──> [B, 64, H/16, W/16]                │       │
│  │                              │                                     │       │
│  │                              ▼                                     │       │
│  │                    concat ──> [B, 256, H/16, W/16]                 │       │
│  │                              +                                     │       │
│  │                    cls_proj ──> [B, 256] ──> broadcast            │       │
│  │                                                                    │       │
│  │   Output: [B, N, 256, H/16, W/16]  (aggregated features)          │       │
│  └────────────────────────────────────────────────────────────────────┘       │
│                           │                                                     │
│                           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐       │
│  │ 2. CROSS-VIEW TRANSFORMER                                          │       │
│  │                                                                    │       │
│  │   Input: [B, N, 256, H/16, W/16]  +  depth [B, N, H, W]            │       │
│  │                                +  poses [B, N, 4, 4]               │       │
│  │                                                                    │       │
│  │   ERP Projection (NEW - Standard lon/lat):                         │       │
│  │     lon = ((u + 0.5) - cx) / fx   where fx = W/(2π)                │       │
│  │     lat = ((v + 0.5) - cy) / fy   where fy = -H/π                  │       │
│  │     x = cos(lat) * sin(lon)                                        │       │
│  │     y = sin(-lat)  # v increases downward                          │       │
│  │     z = cos(lat) * cos(lon)                                        │       │
│  │                                                                    │       │
│  │   For each of L layers:                                            │       │
│  │   ├─ Self-Attention (per view)                                     │       │
│  │   ├─ Geometry-Guided Cross-Attention (between views)               │       │
│  │   └─ FFN + Residual                                                │       │
│  │                                                                    │       │
│  │   Output: [B, N, 256, H/16, W/16]  (fused features)                │       │
│  └────────────────────────────────────────────────────────────────────┘       │
│                           │                                                     │
│                           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐       │
│  │ 3. GS DECODER  (~700K params)                                       │       │
│  │                                                                    │       │
│  │   Input:  fused_feat [B, N, 256, H/16, W/16]                       │       │
│  │           depth_prior [B, N, H, W]                                 │       │
│  │           mask [B, N, H, W]                                        │       │
│  │                                                                    │       │
│  │   Upsampling Pathway: 32→64→128→256→512                            │       │
│  │   ┌────────────────────────────────────────────────────────────┐   │       │
│  │   │ Prediction Heads (1×1 convs):                                │   │       │
│  │   │ head_depth:    32→1    (Δr - depth residual)                 │   │       │
│  │   │ head_cov:      32→3    (σ_x, σ_y, σ_z)                        │   │       │
│  │   │ head_rot:      32→4    (quaternion)                          │   │       │
│  │   │ head_opacity:  32→1    (α)                                   │   │       │
│  │   │ head_sh:       32→C    (C = K×3, K=(sh_degree+1)²)          │   │       │
│  │   │   Example: sh_degree=2 → K=9 → C=27                         │   │       │
│  │   │ head_conf:     32→1    (confidence)                          │   │       │
│  │   └────────────────────────────────────────────────────────────┘   │       │
│  │                                                                    │       │
│  │   Output: Per-view Gaussians (will be fused to world coords)      │       │
│  │   ├─ depth:       [B, N, H, W]    (DAP + Δr)                      │       │
│  │   ├─ covariance:  [B, N, 3, H, W] (scales)                        │       │
│  │   ├─ rotation:    [B, N, 4, H, W] (quaternions)                   │       │
│  │   ├─ opacity:     [B, N, H, W]                                     │       │
│  │   ├─ sh_color:    [B, N, C, H, W] (SH coeffs, C=K×3)             │       │
│  │   └─ confidence:  [B, N, H, W]                                     │       │
│  └────────────────────────────────────────────────────────────────────┘       │
│                           │                                                     │
│                           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐       │
│  │ 4. GAUSSIAN FUSION  (NEW MODULE)                                    │       │
│  │                                                                    │       │
│  │   Converts per-view pixel-wise Gaussians to unified world coords   │       │
│  │                                                                    │       │
│  │   Input: per_view_gaussians + poses [B, N, 4, 4]                   │       │
│  │                                                                    │       │
│  │   For each view:                                                   │       │
│  │   ├─ Depth → 3D points (camera coords) using ERP projection        │       │
│  │   ├─ Transform to world coords: X_world = R^T @ (X_cam - t)        │       │
│  │   └─ Flatten spatial dims: [N, H, W] → [P] where P = N×H×W         │       │
│  │                                                                    │       │
│  │   Filtering (optional):                                            │       │
│  │   ├─ conf_mask = confidence > conf_thresh (default 0.1)            │       │
│  │   └─ opacity_mask = opacity > opacity_thresh (default 0.01)        │       │
│  │                                                                    │       │
│  │   Output: Unified World-Coordinate Gaussians                       │       │
│  │   ├─ means:       [B, P, 3]     (3D positions)                    │       │
│  │   ├─ scales:      [B, P, 3]     (covariance scales)               │       │
│  │   ├─ rotations:   [B, P, 4]     (quaternions)                     │       │
│  │   ├─ opacities:   [B, P, 1]                                        │       │
│  │   ├─ shs:         [B, P, K, 3]  (SH coefficients, RGB)            │       │
│  │   ├─ confidences: [B, P, 1]                                        │       │
│  │   └─ masks:       [B, P]        (valid Gaussians)                 │       │
│  │                                                                    │       │
│  │   where P = N × H × W (total Gaussians from all views)            │       │
│  └────────────────────────────────────────────────────────────────────┘       │
│                           │                                                     │
│                           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐       │
│  │ OUTPUT: Unified Gaussians for Differentiable Rendering             │       │
│  │                                                                    │       │
│  │ For rendering each target view:                                    │       │
│  │ ├─ Use all unified Gaussians [means, scales, rotations, shs]      │       │
│  │ ├─ Apply view transform (world-to-camera)                          │       │
│  │ └─ Rasterize with diff-gaussian-rasterization-omni                 │       │
│  │                                                                    │       │
│  │ Training Loss: L1 + SSIM between rendered and target images        │       │
│  └─────────────────────────────────────────────────────────────────────┘       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

1. 目前训练方式：以seq为单位，每个4相邻帧分为一个chunk输入，【B, N, C, H, W】中的N=4
1.1. 相邻帧的选择可以knn预处理gt pose，N可以固定为4或随机2-4
2. 目前在omnigs rasterizer渲染时oom，原因是估计的gs数目太多
2.1. 目前思路是GS DECODER每一帧估计delta r修正dap先验
2.2. 对于每个chunk输出的gs用GS FUSION整合成这个chunk可以共用的gs，数量会大大减少，但是目前还是太多了





















---

## 关键变更说明 (v2)

### 1. ERP 投影标准化
**之前**: 使用基本的球面坐标转换
```python
x = r * sin(φ) * cos(θ)
y = r * cos(φ)
z = r * sin(φ) * sin(θ)
```

**现在**: 使用标准的等距圆柱投影 (Equirectangular)
```python
lon = ((u + 0.5) - cx) / fx    # fx = W/(2π)
lat = ((v + 0.5) - cy) / fy    # fy = -H/π
x = cos(lat) * sin(lon)
y = sin(-lat)                  # flip because v increases downward
z = cos(lat) * cos(lon)
```

### 2. GaussianFusion 模块 (新增)
将每视图的像素级 Gaussians 融合为统一的世界坐标表示：
- **输入**: Per-view Gaussians `[B, N, ...]` + Camera poses `[B, N, 4, 4]`
- **处理**: 将深度转换为 3D 点 → 变换到世界坐标 → 展平
- **输出**: Unified Gaussians `[B, P, ...]` 其中 `P = N×H×W`

### 3. SH 系数计算修复
**之前**: `sh_channels = (sh_degree + 1) ** 2` (少了 ×3)

**现在**: `sh_channels = (sh_degree + 1) ** 2 * 3`
- Degree 0: 3 channels (RGB, no SH)
- Degree 1: 12 channels (4 SH coeffs × 3 RGB)
- Degree 2: 27 channels (9 SH coeffs × 3 RGB)
- Degree 3: 48 channels (16 SH coeffs × 3 RGB)

### 4. DAP 集成适配
支持通过 `networks.models.make` 创建的 DAP wrapper：
```python
if hasattr(self.dap, 'core'):
    dap_core = self.dap.core  # Access DepthAnythingV2
```

---

## 参数量汇总表

| 模块 | 参数量 | 内存 (fp32) | 内存 (fp16) | 备注 |
|------|--------|-------------|-------------|------|
| **DAP Backbone** | | | | **Pretrained** |
| ├─ DINOv3 Encoder (ViT-L) | ~300M | ~1.2 GB | ~600 MB | 可冻结 |
| ├─ DPT Depth Head | ~30M | ~120 MB | ~60 MB | 可训练 |
| └─ DPT Mask Head | ~30M | ~120 MB | ~60 MB | 可训练 |
| **DAP Subtotal** | **~360M** | **~1.4 GB** | **~720 MB** | |
| | | | | |
| **CelestialSplat New** | | | | **需训练** |
| ├─ DAPFeatureAdapter | 525K | 2.0 MB | 1.0 MB | 特征聚合 |
| ├─ CrossViewTransformer | 6.3M | 24.1 MB | 12.1 MB | 跨视图注意力 |
| ├─ GSDecoder | ~700K | 2.7 MB | 1.4 MB | 高斯参数解码 |
| └─ **GaussianFusion** | ~1K | 可忽略 | 可忽略 | 坐标变换+融合 |
| **CelestialSplat Subtotal** | **~7.6M** | **~28.8 MB** | **~14.5 MB** | |
| | | | | |
| **TOTAL** | **~367.6M** | **~1.43 GB** | **~735 MB** | |

---

## 显存占用详细分析

### 训练阶段 (B=1, N=4, H=512, W=1024)

| 组件 | fp32 | fp16 | 备注 |
|------|------|------|------|
| **模型参数** | | | |
| CelestialSplat部分 | 28.8 MB | 14.4 MB | 可训练 |
| DAP (if trainable) | 1.4 GB | 720 MB | 通常冻结 |
| **激活值** | | | |
| DINOv3 features | 256 MB | 128 MB | 4层中间特征 |
| FeatureAdapter输出 | 16 MB | 8 MB | `[B,N,256,H/16,W/16]` |
| **Transformer激活** | **~7.3 GB** | **~3.6 GB** | **最大开销** |
| GS Decoder | 64 MB | 32 MB | 上采样中间结果 |
| **Unified Gaussians** | **~750 MB** | **~375 MB** | **P = 2M 个 Gaussians** |
| ├─ means/scales/rots | 72 MB | 36 MB | 基本参数 |
| ├─ SH coefficients | 648 MB | 324 MB | Degree 2 (K=9, C=27) |
| └─ Other | 30 MB | 15 MB | opacity/conf/mask |
| **梯度** | ~1.1 GB | ~550 MB | 反向传播 |
| **Optimizer (Adam)** | ~60 MB | - | momentum + variance |

### 不同配置下的总显存

| 配置 | 估计显存 | 适合 GPU |
|------|----------|----------|
| 256×512, SH deg 0, B=1 | ~5 GB | 任何 24GB GPU |
| 256×512, SH deg 2, B=1 | ~6 GB | 任何 24GB GPU |
| 512×1024, SH deg 0, B=1 | ~8 GB | 任何 24GB GPU |
| 512×1024, SH deg 2, B=1 | ~12 GB | 任何 24GB GPU |
| 512×1024, SH deg 2, B=2 | ~20 GB | 24GB GPU |
| 1024×2048, SH deg 0, B=1 | ~35 GB | **需要多 GPU** |

---

## 6×24GB GPU 训练策略

| 阶段 | 配置 | 单GPU显存 | Batch/GPU | 总Batch |
|------|------|-----------|-----------|---------|
| **Warmup** | freeze DAP, fp16 | ~8 GB | 2-4 | 12-24 |
| **Single-View** | freeze DAP, fp16 | ~8 GB | 2-4 | 12-24 |
| **Multi-View** | unfreeze DAP last 6 layers | ~11 GB | 1-2 | 6-12 |
| **Fine-tune** | full fine-tune, fp16 | ~11 GB | 1-2 | 6-12 |

---

## 输出格式对比

### 之前 (Per-View Gaussians)
```python
{
    'gaussians': {
        'depth':       [B, N, H, W],
        'covariance':  [B, N, 3, H, W],
        'rotation':    [B, N, 4, H, W],
        'opacity':     [B, N, H, W],
        'sh_color':    [B, N, 27, H, W],
        'confidence':  [B, N, H, W],
    }
}
```

### 现在 (Unified World-Coordinate Gaussians)
```python
{
    'gaussians': {
        'means':       [B, P, 3],      # P = N×H×W, world coordinates
        'scales':      [B, P, 3],
        'rotations':   [B, P, 4],
        'opacities':   [B, P, 1],
        'shs':         [B, P, K, 3],   # K = (sh_degree+1)²
        'confidences': [B, P, 1],
        'masks':       [B, P],         # valid Gaussians
        'num_per_view': H * W,
        'num_views':   N,
    }
}
```

---

## 推荐配置

### 快速实验 (推荐)
```python
config = CelestialSplatConfig(
    encoder='vitl',
    num_transformer_layers=4,
    K_neighbors=2,
    sh_degree=0,          # RGB only, fastest
    image_height=256,
    image_width=512,
)
# 显存: ~5 GB, 训练速度: 快
```

### 平衡质量与速度
```python
config = CelestialSplatConfig(
    encoder='vitl',
    num_transformer_layers=6,
    K_neighbors=4,
    sh_degree=2,          # Good quality
    image_height=512,
    image_width=1024,
)
# 显存: ~12 GB, 训练速度: 中等
```

### 高质量 (需要更多显存)
```python
config = CelestialSplatConfig(
    encoder='vitl',
    num_transformer_layers=6,
    K_neighbors=4,
    sh_degree=3,          # Best quality
    image_height=512,
    image_width=1024,
)
# 显存: ~15 GB, 训练速度: 较慢
```
