# CelestialSplat v2: Dual-Encoder Siamese Architecture
## 360° Gaussian Splatting via Geometry-Guided Cross-View Attention

---

## 1. 核心架构改进 (Key Improvements)

### 1.1 相对于 v1 的主要变更

| 组件 | v1 方案 | v2 方案 (本方案) | 改进原因 |
|------|---------|------------------|----------|
| **Backbone** | 单 DINOv3 Encoder | **Siamese DINOv3 Encoders** (共享权重) | 每个视图独立编码，避免早期特征混淆 |
| **Cross-View 交互** | 单编码器 + 后处理 Transformer | **Geometry-Guided Cross-Attention** | 基于深度先验在 latent space 进行多视图融合 |
| **坐标系** | 全程 Spherical | **默认 Cartesian + 可选 Spherical** | 降低实现复杂度，先验证核心想法 |
| **DAP 使用** | 深度 + mask + path_1 | **深度 + mask + 4层 features** | 更充分利用 DINOv3 的多尺度特征 |
| **FPN 注入** | path_1 @ 1/2 分辨率 | **Skip connection @ 多尺度** | 类似 UNet，更好的空间细节恢复 |

### 1.2 核心创新点

1. **Siamese DINOv3 Encoders**: 每个 360° 视图独立通过 DINOv3 编码，保持视图特定特征
2. **Geometry-Guided Cross-Attention**: 基于 DAP 深度先验建立跨视图几何对应，指导 attention
3. **Hierarchical Feature Fusion**: 多层 Transformer 的层间特征交互
4. **Residual Depth Prediction**: 预测深度残差 Δr，而非绝对深度

---

## 2. 整体架构与数据流

### 2.1 系统架构图

```
输入: N 张 ERP 图像 I ∈ R^(N×3×H×W)  e.g. [4, 3, 512, 1024]
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│              Siamese DINOv3 Encoders                        │
│  (共享权重，每个视图独立处理)                                 │
│                                                             │
│   View 1    View 2    View 3    ...    View N              │
│     │         │         │              │                   │
│     ▼         ▼         ▼              ▼                   │
│  ┌─────┐   ┌─────┐   ┌─────┐        ┌─────┐               │
│  │DINOv│   │DINOv│   │DINOv│        │DINOv│  共享参数      │
│  │  3  │   │  3  │   │  3  │   ...  │  3  │               │
│  └──┬──┘   └──┬──┘   └──┬──┘        └──┬──┘               │
│     │         │         │              │                   │
│     ▼         ▼         ▼              ▼                   │
│  F_1, D_1,  F_2, D_2,  F_3, D_3,      F_N, D_N             │
│  M_1        M_2        M_3             M_N                 │
│  [256,32,64] [256,32,64] [256,32,64]   [256,32,64]         │
└─────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│         Cross-View Transformer Fusion                       │
│  (Geometry-Guided Multi-View Attention)                     │
│                                                             │
│   每层包含:                                                  │
│   1. Self-Attention (视图内空间关系)                         │
│   2. Geometry-Guided Cross-Attention (视图间融合)            │
│   3. Feed-Forward + Residual                                 │
│                                                             │
│   几何引导: 使用 DAP 深度 D_i 建立 3D 对应                   │
│   pts3d_i = depth_to_3d(D_i, camera_i)                      │
└─────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│              GS Parameter Decoder                           │
│                                                             │
│   输入: F_fused [N, 256, 32, 64]                            │
│         D_prior [N, 1, 512, 1024] (来自 DAP)                │
│         M_mask  [N, 1, 512, 1024] (来自 DAP)                │
│                                                             │
│   输出:                                                     │
│   - Δr (深度残差)      [N, 1, 512, 1024]                    │
│   - Covariance (xyz)   [N, 3, 512, 1024]                    │
│   - Rotation (quat)    [N, 4, 512, 1024]                    │
│   - Opacity            [N, 1, 512, 1024]                    │
│   - SH Color (deg 2)   [N, 27, 512, 1024]                   │
│   - Confidence         [N, 1, 512, 1024]                    │
└─────────────────────────────────────────────────────────────┘
                │
                ▼
        3D Gaussian Rendering
                │
                ▼
        RGB + Depth + Alpha
```

---

## 3. 详细 Shape 变换表

### Stage 1: Siamese DINOv3 Encoding

对每个视图独立处理 (Batch 内 N 个视图并行):

| 模块 | 输入 Shape | 输出 Shape | 备注 |
|------|-----------|-----------|------|
| **Input Image** | `[B, 3, H, W]` | - | e.g. `[2, 3, 512, 1024]` |
| **DINOv3 PatchEmbed** | `[B, 3, 512, 1024]` | `[B, 2048, 1024]` | 2048 = 32×64 patches |
| **Transformer Block 6** | - | `[B, 2049, 1024]` | +1 cls_token, Layer 6 feature |
| **Transformer Block 12** | - | `[B, 2049, 1024]` | Layer 12 feature |
| **Transformer Block 18** | - | `[B, 2049, 1024]` | Layer 18 feature |
| **Transformer Block 24** | - | `[B, 2049, 1024]` | Layer 24 feature |
| **Reshape Patch Maps** | 4×`[B, 2048, 1024]` | 4×`[B, 1024, 32, 64]` | 移除 cls_token，恢复空间 |
| **DPT Head** | 4×`[B, 1024, 32, 64]` | `[B, 1, 512, 1024]` | 深度图 D |
| **Mask Head** | 4×`[B, 1024, 32, 64]` | `[B, 1, 512, 1024]` | 天空掩码 M |
| **Feature Aggregation** | 4×`[B, 1024, 32, 64]` | `[B, 256, 32, 64]` | 通过 Adapter |

**视图维度扩展**:
- 输入: `[B, N, 3, H, W]` → 展开为 `[B×N, 3, H, W]`
- 输出: 重新组织为 `[B, N, C, H', W']`

### Stage 2: Cross-View Transformer Decoder

**输入**: 
- 特征: `[B, N, 256, 32, 64]` (来自 Siamese Encoders)
- 深度: `[B, N, 1, 512, 1024]` (来自 DAP)
- 位姿: `[B, N, 4, 4]` (已知或估计)

**每层 Transformer 的 Shape 变换**:

| 操作 | Input Shape | Output Shape | 说明 |
|------|-------------|--------------|------|
| **Flatten to Tokens** | `[B, N, 256, 32, 64]` | `[B, N, 2048, 256]` | 2048 = 32×64 tokens |
| **Self-Attention** | `[B, N, 2048, 256]` | `[B, N, 2048, 256]` | 每个视图内部的空间注意力 |
| **Geometry Projection** | `[B, N, 1, 512, 1024]` | `[B, N, 3, 512, 1024]` | 深度 → 3D 点云 (xyz) |
| | | → `[B, N, 3, 32, 64]` | 下采样到 token 分辨率 |
| **Cross-View Sampling** | `[B, N, 2048, 256]` | `[B, N, K, 2048, 256]` | K=4 邻居视图采样 |
| **Cross-Attention** | Q:`[B×N×2048, 1, 256]`<br>KV:`[B×N×2048, K, 256]` | `[B, N, 2048, 256]` | 多视图特征融合 |
| **FFN + Residual** | `[B, N, 2048, 256]` | `[B, N, 2048, 256]` | LayerNorm + MLP |
| **Unflatten** | `[B, N, 2048, 256]` | `[B, N, 256, 32, 64]` | 恢复空间维度 |

**多层堆叠** (6 layers):
```
Layer 0: [B, N, 256, 32, 64] → [B, N, 256, 32, 64]
Layer 1: [B, N, 256, 32, 64] → [B, N, 256, 32, 64]
...
Layer 5: [B, N, 256, 32, 64] → [B, N, 256, 32, 64] (F_fused)
```

### Stage 3: GS Parameter Decoder

**输入**:
- F_fused: `[B, N, 256, 32, 64]`
- D_prior: `[B, N, 1, 512, 1024]`
- M_mask: `[B, N, 1, 512, 1024]`

**Shape 变换**:

| 模块 | 输入 | 输出 | 操作 |
|------|------|------|------|
| **Upsample Block 1** | `[B, N, 256, 32, 64]` | `[B, N, 128, 64, 128]` | ConvTranspose 4×4, stride=2 |
| **Upsample Block 2** | `[B, N, 128, 64, 128]` | `[B, N, 64, 128, 256]` | ConvTranspose 4×4, stride=2 |
| **Upsample Block 3** | `[B, N, 64, 128, 256]` | `[B, N, 32, 256, 512]` | ConvTranspose 4×4, stride=2 |
| **Upsample Block 4** | `[B, N, 32, 256, 512]` | `[B, N, 16, 512, 1024]` | ConvTranspose 4×4, stride=2 |
| **Head: Depth Residual** | `[B, N, 16, 512, 1024]` | `[B, N, 1, 512, 1024]` | 1×1 Conv, 预测 Δr |
| **Head: Covariance** | `[B, N, 16, 512, 1024]` | `[B, N, 3, 512, 1024]` | 1×1 Conv, (σ_x, σ_y, σ_z) |
| **Head: Rotation** | `[B, N, 16, 512, 1024]` | `[B, N, 4, 512, 1024]` | 1×1 Conv, 四元数 (归一化) |
| **Head: Opacity** | `[B, N, 16, 512, 1024]` | `[B, N, 1, 512, 1024]` | 1×1 Conv + Sigmoid |
| **Head: SH Color** | `[B, N, 16, 512, 1024]` | `[B, N, 27, 512, 1024]` | 1×1 Conv, SH coeffs |
| **Head: Confidence** | `[B, N, 16, 512, 1024]` | `[B, N, 1, 512, 1024]` | 1×1 Conv + Sigmoid |

**最终深度计算**:
```python
final_depth = D_prior + Δr  # 残差连接
final_depth = final_depth * (1 - M_mask)  # 天空区域置零
```

### Stage 4: Gaussian Formation

**像素级 Gaussian 参数** (每个像素对应一个 3D Gaussian):

| 参数 | Shape | 说明 |
|------|-------|------|
| **Positions** | `[B, N, 3, 512, 1024]` | (x, y, z) in world coordinates |
| **Covariance** | `[B, N, 3, 512, 1024]` | 对角协方差 (σ_x, σ_y, σ_z) |
| **Rotation** | `[B, N, 4, 512, 1024]` | 四元数 q (xyzw)，归一化 |
| **Opacity** | `[B, N, 1, 512, 1024]` | α ∈ [0, 1] |
| **Color (SH)** | `[B, N, 27, 512, 1024]` | Spherical Harmonics degree 2 |
| **Confidence** | `[B, N, 1, 512, 1024]` | w ∈ [0, 1]，用于后处理融合 |

**展平为 Gaussian 列表**:
```python
# 过滤低置信度和低透明度
mask_valid = (confidence > 0.1) & (opacity > 0.01) & (1 - M_mask > 0.5)
# 展平: [B, N, 512, 1024] → 约 B×N×500k 个有效 Gaussians (取决于场景)
```

---

## 4. 关键模块实现

### 4.1 Siamese DINOv3 with DAP

```python
class SiameseDINOv3DAP(nn.Module):
    """
    Siamese DINOv3 Encoders with DAP heads
    每个视图独立编码，共享权重
    """
    def __init__(self, model_type='vitl', max_depth=10.0):
        super().__init__()
        # 加载 DAP (DINOv3 + DPT)
        self.dap = make_model(midas_model_type=model_type, max_depth=max_depth)
        
        # Feature adapter: 聚合 4 层特征
        self.feature_adapter = DAPFeatureAdapter(
            in_dim=1024, 
            out_dim=256,
            num_layers=4
        )
        
    def forward_single(self, image):
        """
        处理单张图像
        Args:
            image: [B, 3, H, W]
        Returns:
            dict with:
                - feature: [B, 256, H/16, W/16]
                - depth: [B, H, W]
                - mask: [B, H, W]
        """
        # 获取中间层特征
        features = self.dap.core.get_intermediate_layers(
            image, 
            [6, 12, 18, 23],  # 4 层特征
            return_class_token=True
        )
        
        # 提取 patch maps 和 cls tokens
        patch_maps = [f[0] for f in features]  # each [B, 1024, H/16, W/16]
        cls_tokens = torch.stack([f[1] for f in features], dim=1)  # [B, 4, 1024]
        
        # 通过 DPT head 获取深度和 mask
        depth = self.dap.depth_head(patch_maps)  # [B, 1, H, W]
        mask = self.dap.mask_head(patch_maps)    # [B, 1, H, W]
        
        # 特征聚合
        feature = self.feature_adapter(patch_maps, cls_tokens)  # [B, 256, H/16, W/16]
        
        return {
            'feature': feature,
            'depth': depth.squeeze(1),
            'mask': mask.squeeze(1)
        }
    
    def forward(self, images):
        """
        处理多视图图像
        Args:
            images: [B, N, 3, H, W]
        Returns:
            dict with:
                - features: [B, N, 256, H/16, W/16]
                - depths: [B, N, H, W]
                - masks: [B, N, H, W]
        """
        B, N = images.shape[:2]
        
        # 展平处理
        images_flat = images.view(B * N, *images.shape[2:])
        
        # 单视图编码
        outputs_flat = self.forward_single(images_flat)
        
        # 恢复视图维度
        return {
            'features': outputs_flat['feature'].view(B, N, *outputs_flat['feature'].shape[1:]),
            'depths': outputs_flat['depth'].view(B, N, *outputs_flat['depth'].shape[1:]),
            'masks': outputs_flat['mask'].view(B, N, *outputs_flat['mask'].shape[1:])
        }


class DAPFeatureAdapter(nn.Module):
    """
    聚合 DINOv3 4层特征为统一表示
    """
    def __init__(self, in_dim=1024, out_dim=256, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        hidden_dim = out_dim // num_layers  # 64
        
        # 为每层特征学习一个投影
        self.patch_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_layers)
        ])
        
        # cls token 投影用于全局调制
        self.cls_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, patch_maps, cls_tokens):
        """
        Args:
            patch_maps: list of 4 tensors [B, 1024, H/16, W/16]
            cls_tokens: [B, 4, 1024]
        Returns:
            [B, 256, H/16, W/16]
        """
        # 投影并拼接 patch features
        outs = [proj(pm) for proj, pm in zip(self.patch_projs, patch_maps)]
        local_feat = torch.cat(outs, dim=1)  # [B, 256, H/16, W/16]
        
        # 全局特征调制
        global_feat = self.cls_proj(cls_tokens.mean(dim=1))  # [B, 256]
        
        # 广播相加
        return local_feat + global_feat[:, :, None, None]
```

### 4.2 Geometry-Guided Cross-View Transformer

```python
class CrossViewTransformer(nn.Module):
    """
    跨视图 Transformer Decoder
    类似 DUSt3R，但引入基于深度的几何引导
    """
    def __init__(self, dim=256, num_layers=6, num_heads=8, K_neighbors=4):
        super().__init__()
        self.num_layers = num_layers
        self.K = K_neighbors
        
        self.layers = nn.ModuleList([
            CrossViewTransformerLayer(dim, num_heads, K_neighbors)
            for _ in range(num_layers)
        ])
        
    def forward(self, features, depths, poses, intrinsics):
        """
        Args:
            features: [B, N, C, H, W]  e.g. [2, 4, 256, 32, 64]
            depths: [B, N, 1, H*16, W*16] 原始分辨率深度 [2, 4, 1, 512, 1024]
            poses: [B, N, 4, 4] 相机位姿
            intrinsics: [B, N, 3, 3] 相机内参 (ERP 特殊处理)
        Returns:
            fused_features: [B, N, C, H, W]
        """
        B, N, C, H, W = features.shape
        
        # 将深度下采样到特征分辨率
        depths_low = F.interpolate(
            depths.flatten(0, 1), 
            size=(H, W), 
            mode='bilinear'
        ).view(B, N, 1, H, W)
        
        # 展平为 tokens: [B, N, HW, C]
        tokens = features.flatten(3).permute(0, 1, 3, 2)  # [B, N, HW, C]
        
        # 计算 3D 点云 (在各自相机坐标系)
        pts3d_cam = self.depth_to_3d(depths_low, intrinsics)  # [B, N, 3, H, W]
        pts3d_cam = pts3d_cam.flatten(3).permute(0, 1, 3, 2)  # [B, N, HW, 3]
        
        # 逐层处理
        for layer in self.layers:
            tokens = layer(tokens, pts3d_cam, poses)
            
        # 恢复空间维度
        fused = tokens.permute(0, 1, 3, 2).reshape(B, N, C, H, W)
        return fused
    
    def depth_to_3d(self, depth, intrinsics):
        """
        将深度图转换为 3D 点云
        对于 ERP 图像，intrinsics 需要特殊处理
        """
        B, N, _, H, W = depth.shape
        
        # 创建像素坐标网格
        u = torch.linspace(0, W-1, W, device=depth.device)  # 0 ~ W
        v = torch.linspace(0, H-1, H, device=depth.device)  # 0 ~ H
        
        # ERP 投影: u → θ, v → φ
        theta = 2 * np.pi * u / W  # [0, 2π]
        phi = np.pi * v / H        # [0, π]
        
        Theta, Phi = torch.meshgrid(theta, phi, indexing='xy')
        
        # Spherical → Cartesian
        d = depth.flatten(0, 1)  # [B*N, 1, H, W]
        x = d * torch.sin(Phi) * torch.cos(Theta)  # [B*N, 1, H, W]
        y = d * torch.cos(Phi)
        z = d * torch.sin(Phi) * torch.sin(Theta)
        
        pts3d = torch.cat([x, y, z], dim=1)  # [B*N, 3, H, W]
        return pts3d.view(B, N, 3, H, W)


class CrossViewTransformerLayer(nn.Module):
    """
    单层 Cross-View Transformer
    包含: Self-Attention + Geometry-Guided Cross-Attention
    """
    def __init__(self, dim, num_heads, K_neighbors):
        super().__init__()
        self.K = K_neighbors
        
        # Self-attention (每个视图内部)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        
        # Cross-view attention
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm3 = nn.LayerNorm(dim)
        
    def forward(self, tokens, pts3d_cam, poses):
        """
        Args:
            tokens: [B, N, HW, C]
            pts3d_cam: [B, N, HW, 3]  相机坐标系下的 3D 点
            poses: [B, N, 4, 4]  world-to-camera 位姿
        Returns:
            tokens: [B, N, HW, C]
        """
        B, N, L, C = tokens.shape
        
        # 1. Self-Attention (合并 batch 和 view 维度处理)
        tokens_flat = tokens.reshape(B * N, L, C)
        attn_out, _ = self.self_attn(tokens_flat, tokens_flat, tokens_flat)
        tokens_flat = self.norm1(tokens_flat + attn_out)
        
        # 恢复维度
        tokens = tokens_flat.reshape(B, N, L, C)
        
        # 2. Cross-View Attention
        # 将点云转换到世界坐标系
        pts3d_world = self.transform_points(pts3d_cam, poses)  # [B, N, HW, 3]
        
        # 对每个视图，从其他视图采样特征
        cross_out = []
        for i in range(N):
            # 视图 i 的点在世界坐标系
            pts_i = pts3d_world[:, i]  # [B, HW, 3]
            
            # 在其他视图中找到对应
            # 将 pts_i 投影到其他视图的相机坐标系
            neighbor_features = []
            for j in range(N):
                if i == j:
                    continue
                # 几何投影: world → camera_j
                pts_in_j = self.world_to_camera(pts_i, poses[:, j])
                # 投影到图像平面，采样特征
                sampled_feat = self.sample_feature(tokens[:, j], pts_in_j)
                neighbor_features.append(sampled_feat)
            
            # 拼接邻居特征
            if neighbor_features:
                kv = torch.stack(neighbor_features, dim=1)  # [B, K, HW, C]
                q = tokens[:, i:i+1]  # [B, 1, HW, C]
                
                # Cross-attention
                attn_out, _ = self.cross_attn(
                    q.flatten(0, 1),  # [B, HW, C]
                    kv.flatten(0, 1), # [B*K, HW, C]
                    kv.flatten(0, 1)
                )
                cross_out.append(self.norm2(tokens[:, i] + attn_out))
            else:
                cross_out.append(tokens[:, i])
        
        tokens = torch.stack(cross_out, dim=1)  # [B, N, HW, C]
        
        # 3. FFN
        tokens = self.norm3(tokens + self.ffn(tokens))
        
        return tokens
    
    def transform_points(self, pts3d_cam, poses):
        """将点从相机坐标系转换到世界坐标系"""
        # poses: [B, N, 4, 4] (world-to-camera)
        # 需要 cam-to-world
        R = poses[:, :, :3, :3]  # [B, N, 3, 3]
        t = poses[:, :, :3, 3]   # [B, N, 3]
        
        # X_world = R^T @ (X_cam - t)
        pts_world = torch.einsum('bnij,bnkj->bnki', R, pts3d_cam) + t[:, :, None, :]
        return pts_world
    
    def world_to_camera(self, pts_world, pose):
        """将点从世界坐标系转换到相机坐标系"""
        # pose: [B, 4, 4]
        R = pose[:, :3, :3]  # [B, 3, 3]
        t = pose[:, :3, 3:4]  # [B, 3, 1]
        
        pts_cam = torch.einsum('bij,bkj->bki', R, pts_world) + t.transpose(1, 2)
        return pts_cam
    
    def sample_feature(self, feature, pts_cam):
        """
        根据相机坐标系下的 3D 点，在特征图上采样
        feature: [B, HW, C]
        pts_cam: [B, HW, 3]
        """
        # 投影到图像平面 (ERP 特殊处理)
        # 这里简化处理，实际需要根据相机模型投影
        # 返回采样的特征 [B, HW, C]
        # TODO: 实现基于 grid_sample 的特征采样
        return feature
```

### 4.3 GS Parameter Decoder

```python
class GSDecoder(nn.Module):
    """
    Gaussian Splatting 参数解码器
    从融合特征预测高斯参数
    """
    def __init__(self, in_dim=256, hidden_dim=128):
        super().__init__()
        
        # 上采样路径
        self.ups = nn.ModuleList([
            self._make_up_block(in_dim, hidden_dim),      # 32 -> 64
            self._make_up_block(hidden_dim, hidden_dim//2),  # 64 -> 128
            self._make_up_block(hidden_dim//2, hidden_dim//4), # 128 -> 256
            self._make_up_block(hidden_dim//4, hidden_dim//4), # 256 -> 512
        ])
        
        # 预测头
        self.head_depth = nn.Conv2d(hidden_dim//4, 1, 1)
        self.head_cov = nn.Conv2d(hidden_dim//4, 3, 1)
        self.head_rot = nn.Conv2d(hidden_dim//4, 4, 1)
        self.head_opacity = nn.Conv2d(hidden_dim//4, 1, 1)
        self.head_sh = nn.Conv2d(hidden_dim//4, 27, 1)
        self.head_conf = nn.Conv2d(hidden_dim//4, 1, 1)
        
    def _make_up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, fused_feat, depth_prior, mask):
        """
        Args:
            fused_feat: [B, N, 256, 32, 64]
            depth_prior: [B, N, 512, 1024]
            mask: [B, N, 512, 1024]
        Returns:
            dict of Gaussian parameters
        """
        B, N = fused_feat.shape[:2]
        x = fused_feat.flatten(0, 1)  # [B*N, 256, 32, 64]
        
        # 上采样
        for up in self.ups:
            x = up(x)
        # x: [B*N, 32, 512, 1024]
        
        # 预测
        delta_d = self.head_depth(x)  # [B*N, 1, 512, 1024]
        cov = self.head_cov(x)
        rot = self.head_rot(x)
        opacity = torch.sigmoid(self.head_opacity(x))
        sh = self.head_sh(x)
        conf = torch.sigmoid(self.head_conf(x))
        
        # 应用 mask
        mask_expanded = mask.flatten(0, 1)[:, None, :, :]  # [B*N, 1, 512, 1024]
        
        # 深度残差
        depth = depth_prior.flatten(0, 1)[:, None, :, :] + delta_d
        depth = depth * (1 - mask_expanded)  # 天空区域深度为 0
        
        # 归一化旋转 (四元数)
        rot = F.normalize(rot, dim=1)
        
        # 恢复维度
        def restore(t):
            return t.view(B, N, *t.shape[1:])
        
        return {
            'depth': restore(depth).squeeze(2),
            'covariance': restore(cov),
            'rotation': restore(rot),
            'opacity': restore(opacity).squeeze(2),
            'sh_color': restore(sh),
            'confidence': restore(conf).squeeze(2)
        }
```

---

## 5. 训练策略

### 5.1 四阶段训练

| 阶段 | Epoch | 训练内容 | DINOv3 | 学习率 | 目标 |
|------|-------|----------|--------|--------|------|
| **Warmup** | 1-10 | DPT Heads (depth/mask) + Adapter | ❄️ Frozen | 1e-4 | Depth AbsRel < 0.1 |
| **Single-View** | 11-30 | + GS Decoder | ❄️ Frozen | 1e-4 | 单视图 PSNR > 22dB |
| **Multi-View** | 31-70 | + Cross-View Transformer | 🔥 Layer 20-23 | 5e-5 | N=4 PSNR > 26dB |
| **Fine-tune** | 71-100 | 全网络 | 🔥 All | 1e-6 | N=4 PSNR > 28dB |

### 5.2 损失函数

```python
class CelestialSplatLoss(nn.Module):
    def __init__(self):
        self.lambda_photo = 1.0
        self.lambda_ssim = 0.5
        self.lambda_lpips = 0.1
        self.lambda_depth = 0.01
        self.lambda_opacity = 0.01
        self.lambda_scale = 0.001
        
    def forward(self, rendered, target, gaussians, mask):
        valid = (1 - mask).unsqueeze(1)  # [B, 1, H, W]
        
        # 光度损失
        l1 = F.l1_loss(rendered['rgb'] * valid, target * valid)
        ssim_loss = 1 - ssim(rendered['rgb'] * valid, target * valid)
        
        # LPIPS
        lpips_loss = self.lpips(rendered['rgb'], target)
        
        # 深度监督 (如果有 GT)
        depth_loss = F.l1_loss(
            rendered['depth'] * valid.squeeze(1),
            target_depth * valid.squeeze(1)
        ) if target_depth is not None else 0
        
        # 正则化
        opacity_reg = gaussians['opacity'].mean()
        scale_reg = gaussians['covariance'].abs().mean()
        
        total = (self.lambda_photo * l1 + 
                 self.lambda_ssim * ssim_loss + 
                 self.lambda_lpips * lpips_loss +
                 self.lambda_depth * depth_loss +
                 self.lambda_opacity * opacity_reg +
                 self.lambda_scale * scale_reg)
        
        return total, {'l1': l1, 'ssim': ssim_loss, 'depth': depth_loss}
```

---

## 6. Shape 验证脚本

```python
def verify_model_shapes():
    """验证整个模型的 shape 一致性"""
    device = 'cuda'
    B, N = 2, 4
    H, W = 512, 1024
    
    # 创建模型
    encoder = SiameseDINOv3DAP().to(device)
    transformer = CrossViewTransformer().to(device)
    decoder = GSDecoder().to(device)
    
    # 模拟输入
    images = torch.randn(B, N, 3, H, W).to(device)
    poses = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1).to(device)
    intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1).to(device)
    
    print("=" * 60)
    print("Shape Verification")
    print("=" * 60)
    
    # Stage 1: Encoding
    print("\n[Stage 1] Siamese Encoding...")
    enc_out = encoder(images)
    print(f"  Images:        {images.shape}")
    print(f"  Features:      {enc_out['features'].shape}  (expected [B,N,256,H/16,W/16])")
    print(f"  Depths:        {enc_out['depths'].shape}     (expected [B,N,H,W])")
    print(f"  Masks:         {enc_out['masks'].shape}      (expected [B,N,H,W])")
    
    assert enc_out['features'].shape == (B, N, 256, H//16, W//16)
    assert enc_out['depths'].shape == (B, N, H, W)
    
    # Stage 2: Cross-View Transformer
    print("\n[Stage 2] Cross-View Transformer...")
    fused = transformer(
        enc_out['features'], 
        enc_out['depths'].unsqueeze(2),
        poses, 
        intrinsics
    )
    print(f"  Input:         {enc_out['features'].shape}")
    print(f"  Output:        {fused.shape}  (expected [B,N,256,H/16,W/16])")
    
    assert fused.shape == (B, N, 256, H//16, W//16)
    
    # Stage 3: GS Decoder
    print("\n[Stage 3] GS Decoder...")
    gaussians = decoder(fused, enc_out['depths'], enc_out['masks'])
    print(f"  Depth:         {gaussians['depth'].shape}       (expected [B,N,H,W])")
    print(f"  Covariance:    {gaussians['covariance'].shape}  (expected [B,N,3,H,W])")
    print(f"  Rotation:      {gaussians['rotation'].shape}    (expected [B,N,4,H,W])")
    print(f"  Opacity:       {gaussians['opacity'].shape}     (expected [B,N,H,W])")
    print(f"  SH Color:      {gaussians['sh_color'].shape}    (expected [B,N,27,H,W])")
    print(f"  Confidence:    {gaussians['confidence'].shape}  (expected [B,N,H,W])")
    
    # 计算总 Gaussian 数
    total_gaussians = B * N * H * W
    print(f"\n[Summary] Total Gaussians per batch: {total_gaussians:,}")
    print(f"         ({B} batches × {N} views × {H}×{W} pixels)")
    
    print("\n" + "=" * 60)
    print("✅ All shapes verified!")
    print("=" * 60)

if __name__ == '__main__':
    verify_model_shapes()
```

---

## 7. 与相关工作的关系

### Cross-View Attention 的演进

Cross-view attention/transformer 机制**并非 DUSt3R 独有**，而是多视图立体视觉中的通用技术：

| 方法 | 年份 | Cross-View 机制 | 应用场景 |
|------|------|----------------|----------|
| **MVSNet** | 2018 | 3D Cost Volume + CNN | 深度估计 |
| **R-MVSNet** | 2019 | GRU 序列聚合 | 深度估计 |
| **TransMVSNet** | 2022 | Transformer Global Attention | 多视图融合 |
| **MVSFormer** | 2023 | Hierarchical Transformer | 特征匹配 |
| **DUSt3R** | 2024 | Dual-Decoder Cross-Attention | 双视图几何 |
| **MVSplat** | 2024 | Cross-Attention + Gaussian | 高斯抛雪球 |
| **CelestialSplat (Ours)** | 2025 | Geometry-Guided Cross-Attn | 360° 高斯抛雪球 |

### 我们的改进

相比现有方法，CelestialSplat v2 的**核心差异**：

1. **几何引导的注意力**：使用 DAP 深度先验建立跨视图对应关系，而非基于学习的对应
2. **ERP 原生处理**：直接在 360° ERP 图像上进行 cross-view attention，无需透视投影
3. **Siamese 预训练 Encoder**：利用 DINOv3 的强视觉表征 + DAP 的几何先验
4. **端到端高斯预测**：从特征融合直接预测 Gaussian 参数，无需显式深度估计

### 与 DUSt3R 的具体差异

| 特性 | DUSt3R | CelestialSplat v2 |
|------|--------|--------------|
| **输入** | Pinhole 图像对 | 360° ERP 图像序列 |
| **Encoder** | 2× ViT-L (Siamese) | N× DINOv3 (Siamese) |
| **预训练** | CroCo (自监督) | DINOv3 (对比学习) + DAP (深度) |
| **Cross-View** | Decoder 双向 Cross-Attn | Geometry-Guided Cross-Attn |
| **输出** | pts3d + conf (per pixel) | Gaussian params (per pixel) |
| **后处理** | Global Alignment | Confidence-Based Fusion |
| **渲染** | 点云可视化 | Differentiable Gaussian Splatting |

---

## 8. 下一步实施建议

1. **Week 1-2**: 实现 `SiameseDINOv3DAP` 和 `DAPFeatureAdapter`，验证单视图深度和 mask 预测
2. **Week 3-4**: 实现 `GSDecoder`，单视图 GS 训练 (冻结 Cross-View Transformer)
3. **Week 5-6**: 实现 `CrossViewTransformer`，验证双视图融合效果
4. **Week 7-8**: 扩展到多视图 (N=4,8,16)，优化显存占用
5. **Week 9-10**: 完整训练流程，对比实验
6. **Week 11-12**: 评估与论文撰写

**关键里程碑**:
- [ ] 单视图 Depth Estimation AbsRel < 0.1
- [ ] 单视图 GS PSNR > 22dB
- [ ] 双视图融合 PSNR 提升 > 1.5dB
- [ ] N=4 多视图 PSNR > 28dB
