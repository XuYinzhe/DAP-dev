# CelestialSplat: DAP深度嵌入方案 (方案3) - 完整技术规格

## 1. 项目定位与核心创新

**目标**: 构建首个将DAP (Depth Anything with DINOv3) 作为共享Backbone深度嵌入的360° Gaussian Splatting系统。

**核心创新点**:
- **Unified DINOv3 Backbone**: 用DINOv3 Encoder替代ConvNeXt-Tiny，同时驱动深度估计、掩码预测和特征提取
- **Multi-Scale Feature Fusion**: 利用DAP的DPTHead内部FPN特征(path_1)作为GS Decoder的高质量输入
- **Geometry-Aware Cross-View Attention**: 在球面坐标系下，基于DAP深度先验构建跨视图Transformer
- **End-to-End Trainable**: DAP骨干网络与CelestialSplat联合训练，深度先验随任务自适应优化

---

## 2. 架构设计与Shape规格

### 2.1 整体数据流 (以输入 512×1024 ERP图像为例)

```
输入: ERP Image I ∈ R^(B×3×H×W)  e.g. [1, 3, 512, 1024]
                │
                ▼
    ┌─────────────────────────────┐
    │    DINOv3 Encoder (Frozen→  │  ← DAP.core.pretrained
    │     Progressive Unfreeze)   │     Output: 4层 intermediate features
    └─────────────────────────────┘
                │
    ┌───────────┼───────────┐
    │           │           │
    ▼           ▼           ▼
┌───────┐  ┌───────┐  ┌────────────────┐
│Depth  │  │ Mask  │  │ 4-layer Features│
│ Head  │  │ Head  │  │ (DPT processing)│
└───┬───┘  └───┬───┘  └───────┬────────┘
    │          │              │
    ▼          ▼              ▼
D_prior    M_mask      F_dap (aggregated)
[1,512,   [1,512,      [1,4096,32,64] 
 1024]     1024]       (4×1024 channels)
    │          │              │
    │          │              ├──► FeatureAdapter ──► [1,256,32,64]
    │          │              │                         │
    │          │              │    ┌────────────────────┘
    │          │              │    ▼
    │          │         ┌────┴───────────────────┐
    │          │         │ Spherical Cross-View   │
    │          │         │ Transformer            │
    │          │         │ (at 1/16 resolution)   │
    │          │         └────┬───────────────────┘
    │          │              │ F_fused [1,256,32,64]
    │          │              ▼
    │          │    ┌─────────────────┐
    │          │    │ DPT FPN path_1  │ ◄── 额外注入 (from DPTHead)
    │          │    │ [1,256,256,512] │     at 1/2 resolution
    │          │    └─────────────────┘
    │          │              │
    │          │              ▼
    │          │    ┌─────────────────┐
    │          │    │  GS Decoder     │
    │          │    │  (w/ Skip Conn) │
    │          │    └────────┬────────┘
    │          │              │
    ▼          ▼              ▼
┌───────────────────────────────────────┐
│         Gaussian Parameters           │
│  • Position: (r+Δr, θ, φ)             │
│  • Covariance: (σ_r, σ_θ, σ_φ)        │
│  • Color: SH coefficients             │
│  • Opacity: α                         │
│  • Confidence: w                      │
└───────────────────────────────────────┘
                │
                ▼
        Render + Skybox
        (M_mask weighted)
```

### 2.2 详细Shape变换表

#### Stage 1: DINOv3 Encoder (DAP Backbone)

| 模块 | 输入Shape | 输出Shape | 备注 |
|------|----------|----------|------|
| Input | `[B, 3, H, W]` | - | e.g. `[1, 3, 512, 1024]` |
| PatchEmbed | `[B, 3, 512, 1024]` | `[B, 2048, 1024]` | 2048 = 32×64 patches |
| Transformer Block 4 | - | `[B, 2049, 1024]` | +1 cls_token |
| Transformer Block 11 | - | `[B, 2049, 1024]` | intermediate layer |
| Transformer Block 17 | - | `[B, 2049, 1024]` | intermediate layer |
| Transformer Block 23 | - | `[B, 2049, 1024]` | intermediate layer |
| LayerNorm + Reshape | 4×`[B, 2048, 1024]` | 4×`[B, 1024, 32, 64]` | patch_maps |
| cls_tokens | - | 4×`[B, 1024]` | class tokens |

**关键维度**:
- `patch_size = 16`
- `patch_h = H // 16 = 32`
- `patch_w = W // 16 = 64`
- `embed_dim = 1024` (for vitl)

#### Stage 2: DPTHead (Depth & Mask Prediction)

**输入**: 4层特征 `[(B,1024,32,64), ...]`

| 操作 | Layer 1 | Layer 2 | Layer 3 | Layer 4 |
|------|---------|---------|---------|---------|
| Project (1×1 Conv) | `[B,256,32,64]` | `[B,512,32,64]` | `[B,1024,32,64]` | `[B,1024,32,64]` |
| Resize | 4×Up: `[B,256,128,256]` | 2×Up: `[B,512,64,128]` | Identity: `[B,1024,32,64]` | 2×Down: `[B,1024,16,32]` |

**FPN融合路径**:
```
layer_4_rn [B,1024,16,32] ──► refinenet4 ──► path_4 [B,256,32,64]
                                          │
layer_3_rn [B,1024,32,64] ──► refinenet3 ◄──┘ path_3 [B,256,64,128]
                                          │
layer_2_rn [B,512,64,128] ──► refinenet2 ◄──┘ path_2 [B,256,128,256]
                                          │
layer_1_rn [B,256,128,256] ──► refinenet1 ◄──┘ path_1 [B,256,256,512]
```

**输出头**:
| 操作 | Shape |
|------|-------|
| output_conv1 | `[B, 128, 256, 512]` |
| Interpolate to (H,W) | `[B, 128, 512, 1024]` |
| output_conv2 (final) | `[B, 1, 512, 1024]` |

**输出**: 
- `depth`: `[B, H, W]` (squeeze后), 值域 `[0, max_depth]`
- `mask`: `[B, H, W]` (squeeze后), 值域 `[0, 1]` (sigmoid)

#### Stage 3: FeatureAdapter (新增模块)

```python
class DAPFeatureAdapter(nn.Module):
    """
    将DINOv3的4层特征聚合为统一表示
    输入: 4层特征 [B,1024,32,64] × 4
    输出: [B, out_dim, 32, 64]  (保持1/16分辨率)
    """
    def __init__(self, in_dims=[1024,1024,1024,1024], out_dim=256):
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, out_dim//4, 1),  # 每层压缩到64
                nn.BatchNorm2d(out_dim//4),
                nn.ReLU(inplace=True)
            ) for d in in_dims
        ])
    
    def forward(self, features):
        # features: list of 4 tensors, each [B,1024,32,64]
        outs = [proj(f) for proj, f in zip(self.projs, features)]
        # each [B,64,32,64]
        return torch.cat(outs, dim=1)  # [B,256,32,64]
```

#### Stage 4: Spherical Cross-View Transformer

**输入**:
- Query: `F_dap` from FeatureAdapter `[B, N, 256, 32, 64]` (N views)
- Key/Value: Sampled neighbor features at K depth hypotheses

**几何投影** (Spherical Coordinates):
```python
# For each pixel p(θ, φ) at view i with depth r:
# 1. Spherical → Cartesian:
#    x = r * sin(φ) * cos(θ)
#    y = r * cos(φ)
#    z = r * sin(φ) * sin(θ)
#    Shape: [B, N, 3, 32, 64]
#
# 2. Transform to view j: X' = T_j^(-1) @ T_i @ X
#    Shape maintained: [B, N, 3, 32, 64]
#
# 3. Cartesian → Spherical:
#    r' = sqrt(x²+y²+z²)
#    θ' = atan2(z, x)
#    φ' = arccos(y/r')
#
# 4. Project to image coordinates:
#    u' = θ' * W / (2π)
#    v' = φ' * H / π
#    Sample F_j at (u', v') using grid_sample
```

**Attention机制**:
```
Input: Q [B,N,256,32,64], K [B,N,K,256,32,64], V [B,N,K,256,32,64]
       (K=8 depth hypotheses)

Reshape: Q -> [B*N*32*64, 1, 256]
         K -> [B*N*32*64, K, 256]
         V -> [B*N*32*64, K, 256]

Multi-Head Attention (8 heads):
    Output: [B*N*32*64, 1, 256] -> [B,N,256,32,64]

MLP + Residual:
    F_fused = F_dap + MLP(Attention(Q,K,V))
    Output: [B, N, 256, 32, 64]
```

#### Stage 5: GS Decoder with DPT FPN Injection

**输入**:
- Fused features: `[B, N, 256, 32, 64]` (from Transformer)
- DPT path_1: `[B, N, 256, 256, 512]` (from DPTHead内部, 1/2分辨率)
- Depth prior: `[B, N, 1, 512, 1024]` (from DAP)
- Mask: `[B, N, 1, 512, 1024]` (from DAP)

**Decoder结构**:
```python
class GSDecoder(nn.Module):
    def __init__(self, in_dim=256, path1_dim=256):
        # Upsample pathway
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 128, 4, 2, 1),  # 32->64
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),      # 64->128
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),       # 128->256
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # At 256x512, concat with DPT path_1 (256 channels)
        self.skip_fusion = nn.Conv2d(32+256, 128, 3, 1, 1)
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),      # 256->512
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Prediction heads
        self.depth_residual = nn.Conv2d(64, 1, 1)     # Δr
        self.covariance = nn.Conv2d(64, 3, 1)         # (σ_r, σ_θ, σ_φ)
        self.rotation = nn.Conv2d(64, 4, 1)           # quaternion
        self.opacity = nn.Conv2d(64, 1, 1)            # α
        self.sh_color = nn.Conv2d(64, 27, 1)          # SH degree 2
        self.confidence = nn.Conv2d(64, 1, 1)         # w

    def forward(self, fused_feat, path1, depth_prior, mask):
        # fused_feat: [B,256,32,64]
        x = self.up1(fused_feat)    # [B,128,64,128]
        x = self.up2(x)             # [B,64,128,256]
        x = self.up3(x)             # [B,32,256,512]
        
        # Inject DPT FPN path_1
        x = torch.cat([x, path1], dim=1)  # [B,288,256,512]
        x = self.skip_fusion(x)           # [B,128,256,512]
        
        x = self.up4(x)             # [B,64,512,1024]
        
        # Apply mask to predictions in sky regions
        valid_mask = (mask < 0.5).float()
        
        delta_r = self.depth_residual(x) * valid_mask
        final_depth = depth_prior + delta_r  # Residual connection
        
        return {
            'depth': final_depth,           # [B,1,512,1024]
            'covariance': self.covariance(x),  # [B,3,512,1024]
            'rotation': self.rotation(x),   # [B,4,512,1024]
            'opacity': self.opacity(x),     # [B,1,512,1024]
            'sh_color': self.sh_color(x),   # [B,27,512,1024]
            'confidence': self.confidence(x) * valid_mask  # [B,1,512,1024]
        }
```

---

## 3. 训练策略与阶段规划

### 3.1 三阶段训练路线图

#### Phase 1: Warmup (Epoch 1-10)
**目标**: 让DAP适应360°数据域，稳定深度和mask预测

| 模块 | 状态 | LR |
|------|------|-----|
| DINOv3 Encoder | ❄️ Frozen | 0 |
| DPT Heads (depth/mask) | 🔥 Train | 1e-4 |
| FeatureAdapter | 🔥 Train | 1e-4 |
| Transformer | 🔥 Train | 1e-4 |
| GS Decoder | 🔥 Train | 1e-4 |

**损失函数**:
```python
L = λ1*L1(depth_pred, depth_gt) + λ2*L_mask_bce
  + λ3*L_photometric(rendered, target)  # 仅depth/mask稳定后
```

**验证指标**:
- Depth AbsRel < 0.1 (在Matterport3D验证集上)
- Mask IoU > 0.8 (天空区域)

#### Phase 2: Joint Training (Epoch 11-50)
**目标**: 端到端联合优化，DINOv3特征更服务于GS任务

| 模块 | 状态 | LR |
|------|------|-----|
| DINOv3 (最后6层) | 🔥 Unfrozen | 1e-5 |
| DINOv3 (前18层) | ❄️ Frozen | 0 |
| 其他所有模块 | 🔥 Train | 1e-4 |

**学习率调度**: CosineAnnealing with warmup

#### Phase 3: Full Fine-tuning (Epoch 51-100)
**目标**: 全网络微调，追求最佳质量

| 模块 | 状态 | LR |
|------|------|-----|
| DINOv3 (全部) | 🔥 Unfrozen | 1e-6 |
| 其他所有模块 | 🔥 Train | 5e-5 |

**梯度裁剪**: max_norm=1.0 (防止Transformer不稳定)

### 3.2 损失函数完整定义

```python
class PanoSplatLoss(nn.Module):
    def __init__(self):
        self.lambda_photo = 1.0
        self.lambda_ssim = 0.5
        self.lambda_lpips = 0.1
        self.lambda_depth = 0.01  # 仅在GT depth可用时
        self.lambda_mask = 0.1
        self.lambda_feature = 0.05  # DINOv2 feature matching
        
    def forward(self, outputs, targets, masks):
        # 1. Photometric Loss (仅非天空区域)
        valid_mask = (masks['pred_mask'] < 0.5).float()
        
        l1_loss = F.l1_loss(
            outputs['rendered'] * valid_mask,
            targets['image'] * valid_mask
        )
        
        ssim_loss = 1 - ssim(
            outputs['rendered'] * valid_mask,
            targets['image'] * valid_mask
        )
        
        # 2. LPIPS perceptual loss
        lpips_loss = self.lpips(
            outputs['rendered'],
            targets['image']
        )
        
        # 3. Depth consistency (with DAP prior)
        if 'depth_gt' in targets:
            depth_loss = F.l1_loss(
                outputs['depth'] * valid_mask,
                targets['depth_gt'] * valid_mask
            )
        else:
            depth_loss = 0
            
        # 4. Feature matching (DINOv2 frozen features)
        with torch.no_grad():
            target_feat = self.dinov2(targets['image'])
        pred_feat = self.dinov2(outputs['rendered'])
        feat_loss = F.mse_loss(pred_feat, target_feat)
        
        # 5. Mask regularization
        mask_loss = F.binary_cross_entropy(
            masks['pred_mask'],
            targets['sky_mask']  # pseudo label or hand-crafted
        )
        
        # 6. Gaussian regularization
        opacity_reg = outputs['opacity'].mean()
        scale_reg = outputs['covariance'].abs().mean()
        
        total_loss = (
            self.lambda_photo * l1_loss +
            self.lambda_ssim * ssim_loss +
            self.lambda_lpips * lpips_loss +
            self.lambda_depth * depth_loss +
            self.lambda_feature * feat_loss +
            self.lambda_mask * mask_loss +
            0.01 * opacity_reg +
            0.001 * scale_reg
        )
        
        return total_loss, {
            'l1': l1_loss, 'ssim': ssim_loss, 'lpips': lpips_loss,
            'depth': depth_loss, 'feature': feat_loss, 'mask': mask_loss
        }
```

---

## 4. 实施路线图 (12周)

### Week 1-2: Infrastructure & DAP Integration
**Week 1**:
- [ ] 修改 `dpt.py` 暴露4层intermediate features和path_1
- [ ] 实现 `FeatureAdapter` 模块
- [ ] 编写Shape测试脚本，验证全链路维度一致性

**Week 2**:
- [ ] 实现 `DAPWrapper` 类，统一管理DINOv3 + DPT Heads
- [ ] 集成到CelestialSplat数据Pipeline
- [ ] **验证点**: 输入 `[1,3,512,1024]` → 输出depth `[1,512,1024]` + features `[4,1024,32,64]` + path_1 `[1,256,256,512]`

### Week 3-4: Single-View Baseline
**Week 3**:
- [ ] 冻结DINOv3，训练DPT Heads + FeatureAdapter
- [ ] 单视图Depth/Mask预测优化

**Week 4**:
- [ ] 实现GS Decoder (不含Cross-View)
- [ ] 单视图GS训练
- [ ] **验证点**: Matterport3D单视图PSNR > 22dB

### Week 5-6: Cross-View Transformer
**Week 5**:
- [ ] 实现Spherical几何投影模块
- [ ] 实现K-NN Neighbor Graph构建 (基于pose距离)
- [ ] 实现基础Cross-Attention

**Week 6**:
- [ ] Multi-Head Attention集成
- [ ] DINOv2 Feature Matching Loss
- [ ] **验证点**: N=2 vs 单视图×2，PSNR提升 > 1.5dB

### Week 7-8: Multi-View Scale-up
**Week 7**:
- [ ] 扩展至N=4, 8
- [ ] 实现Confidence-Based Fusion
- [ ] 显存优化 (Gradient Checkpointing)

**Week 8**:
- [ ] N=16-30训练 (完整house-scale)
- [ ] 混合数据集训练
- [ ] **验证点**: 自监督训练，无GT depth数据集上验证几何精度

### Week 9-10: Skybox & Refinement
**Week 9**:
- [ ] 实现Skybox Branch (利用DAP mask)
- [ ] Alpha Blending渲染管线
- [ ] DINOv3后6层解冻

**Week 10**:
- [ ] 全量实验对比 (CylinderSplat, MVSplat, 3DGS)
- [ ] Ablation Studies

### Week 11-12: Evaluation & Writing
**Week 11**:
- [ ] Real-world demo (Insta360数据)
- [ ] 论文图表生成

**Week 12**:
- [ ] 论文撰写
- [ ] Code开源准备

---

## 5. 关键代码模块清单

### 5.1 修改DAP核心 (depth_anything_v2_metric/)

```python
# dpt.py - DepthAnythingV2.forward 修改
def forward(self, x, return_features=False, return_path1=False):
    patch_size = self.patch_size  # 16
    patch_h, patch_w = x.shape[-2] // patch_size, x.shape[-1] // patch_size
    
    features = self.pretrained.get_intermediate_layers(
        x, self.intermediate_layer_idx[self.encoder], 
        return_class_token=True
    )
    # features: [(patch_map, cls_token), ...] × 4
    # patch_map: [B,1024,32,64], cls_token: [B,1024]
    
    # 同时获取DPT内部的path_1 (需要在DPTHead中暴露)
    depth, path_1 = self.depth_head(
        features, patch_h, patch_w, patch_size, 
        return_path1=return_path1
    )
    mask = self.mask_head(features, patch_h, patch_w, patch_size)
    
    outputs = {
        'depth': depth.squeeze(1),  # [B,H,W]
        'mask': mask.squeeze(1),    # [B,H,W]
    }
    
    if return_features:
        # 提取patch_maps
        patch_maps = [f[0] for f in features]  # list of [B,1024,32,64]
        outputs['features'] = patch_maps
        
    if return_path1:
        outputs['path_1'] = path_1  # [B,256,256,512]
        
    return outputs
```

### 5.2 FeatureAdapter (networks/dap_adapter.py)

```python
class DAPFeatureAdapter(nn.Module):
    """
    Adapter for DAP (DINOv3) features to CelestialSplat Transformer
    
    Input: List of 4 tensors from DINOv3 intermediate layers
           Each: [B, 1024, H/16, W/16] (e.g. [1,1024,32,64] for 512x1024 input)
    
    Output: Aggregated feature [B, out_dim, H/16, W/16] (e.g. [1,256,32,64])
    """
    def __init__(self, in_dim=1024, out_dim=256, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        hidden_dim = out_dim // num_layers  # 64
        
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_layers)
        ])
        
    def forward(self, features):
        """
        Args:
            features: List of 4 tensors [B,1024,32,64]
        Returns:
            [B,256,32,64]
        """
        outs = [proj(f) for proj, f in zip(self.projs, features)]
        return torch.cat(outs, dim=1)  # [B,256,32,64]
```

### 5.3 DAPWrapper (networks/dap_wrapper.py)

```python
class DAPWrapper(nn.Module):
    """
    High-level wrapper for DAP in CelestialSplat
    Manages DINOv3 backbone, DPT heads, and feature extraction
    """
    def __init__(self, model_type='vitl', max_depth=10.0):
        super().__init__()
        self.dap = make_model(
            midas_model_type=model_type, 
            max_depth=max_depth
        )
        self.feature_adapter = DAPFeatureAdapter(
            in_dim=1024, 
            out_dim=256
        )
        
    def forward(self, images, return_all=True):
        """
        Args:
            images: [B, N, 3, H, W] or [B*N, 3, H, W]
        Returns:
            dict with:
                - depth: [B,N,H,W]
                - mask: [B,N,H,W]
                - features: [B,N,256,H/16,W/16] (aggregated)
                - path_1: [B,N,256,H/2,W/2] (FPN feature for GS decoder)
        """
        B, N = images.shape[:2]
        images_flat = images.view(B*N, *images.shape[2:])
        
        outputs = self.dap.core(
            images_flat,
            return_features=True,
            return_path1=True
        )
        
        # Adapt features
        adapted = self.feature_adapter(outputs['features'])
        
        # Reshape to multi-view
        results = {
            'depth': outputs['depth'].view(B, N, *outputs['depth'].shape[1:]),
            'mask': outputs['mask'].view(B, N, *outputs['mask'].shape[1:]),
            'features': adapted.view(B, N, *adapted.shape[1:]),
            'path_1': outputs['path_1'].view(B, N, *outputs['path_1'].shape[1:])
        }
        
        return results
```

---

## 6. 风险提示与备选方案

### 高风险点

1. **DINOv3显存占用高**
   - **风险**: vitl模型在N=16时可能OOM (24GB显存)
   - **缓解**: 
     - 使用gradient checkpointing
     - 降低Transformer维度 (256→128)
     - 使用vitb替代vitl (embed_dim 768 vs 1024)

2. **DPTHead输出的path_1分辨率过高**
   - **风险**: `[B,256,256,512]`在N=16时显存占用大
   - **缓解**: 下采样到 `[B,256,128,256]` 后再输入GS Decoder

3. **Mask预测不准确**
   - **风险**: 天空/建筑边界模糊导致GS训练不稳定
   - **缓解**: 
     - 先用传统方法生成伪标签预训练Mask Head
     - 使用CRF后处理mask

4. **Cross-View Attention收敛慢**
   - **风险**: 多视图几何对应错误导致注意力失效
   - **缓解**: 
     - 先用单视图训练稳定，再逐步增加视图数
     - 添加Gating机制控制残差连接

### Shape调试检查清单

在训练开始前，验证以下shape一致性：

```python
def verify_shapes(model, batch_size=2, num_views=4, h=512, w=1024):
    dummy_input = torch.randn(batch_size, num_views, 3, h, w).cuda()
    
    with torch.no_grad():
        outputs = model(dummy_input)
        
    expected = {
        'depth': (batch_size, num_views, h, w),
        'mask': (batch_size, num_views, h, w),
        'features': (batch_size, num_views, 256, h//16, w//16),
        'path_1': (batch_size, num_views, 256, h//2, w//2),
        'gaussians': {
            'depth': (batch_size, num_views, 1, h, w),
            'covariance': (batch_size, num_views, 3, h, w),
            'rotation': (batch_size, num_views, 4, h, w),
            'opacity': (batch_size, num_views, 1, h, w),
            'sh_color': (batch_size, num_views, 27, h, w),
        }
    }
    
    for key, shape in expected.items():
        if isinstance(shape, dict):
            for k, s in shape.items():
                assert outputs[key][k].shape == s, f"{key}.{k} shape mismatch"
        else:
            assert outputs[key].shape == shape, f"{key} shape mismatch"
    
    print("✅ All shapes verified!")
```

---

## 7. 期望输出

**代码结构**:
```
CelestialSplat/
├── networks/
│   ├── dap_wrapper.py      # DAPWrapper类
│   ├── dap_adapter.py      # FeatureAdapter
│   ├── transformer.py      # Spherical Cross-View Transformer
│   ├── gs_decoder.py       # GS Decoder with path_1 injection
│   └── skybox.py           # Skybox Branch
├── depth_anything_v2_metric/  # DAP源码 (修改版)
│   └── depth_anything_v2/
│       └── dpt.py          # 暴露features和path_1
├── models/
│   └── panosplat.py        # 主模型整合
└── train.py                # 三阶段训练脚本
```

**关键交付物**:
1. 端到端可训练模型 (DAP + Transformer + GS)
2. 预训练权重 (在Matterport3D + 360Loc混合数据上)
3. 单视图NVS PSNR > 24dB, 多视图(N=4) PSNR > 28dB
4. 支持实时推理: 512×1024输入, 单视图<100ms

---

**Note**: 本文档基于DAP实测Shape参数编写，所有维度以512×1024 ERP输入为基准。实际部署时需根据目标分辨率 (256×512, 1024×2048等) 按比例调整。
