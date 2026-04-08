# SHARP 模型架构与数据流详解

本文档详细描述了 SHARP (Sharp Monocular View Synthesis in Less Than a Second) 方法的模型结构、数据流以及对应的代码文件。

## 概述

SHARP 是一个从单张图像生成 3D 高斯表示 (3D Gaussian Splatting) 的神经网络方法。它通过单次前向传播，在不到一秒的时间内从单张图片预测出可用于实时渲染的 3D 高斯参数。

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SHARP 整体架构                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐                   │
│  │  Input      │───▶│  Monodepth      │───▶│  Initializer    │                   │
│  │  Image      │    │  Network (DPT)  │    │  (Base Values)  │                   │
│  └─────────────┘    └─────────────────┘    └─────────────────┘                   │
│                              │                        │                          │
│                              ▼                        ▼                          │
│                     ┌─────────────────┐    ┌─────────────────┐                   │
│                     │  Feature        │───▶│  Gaussian       │                   │
│                     │  Encodings      │    │  Decoder (DPT)  │                   │
│                     └─────────────────┘    └─────────────────┘                   │
│                                                     │                            │
│                                                     ▼                            │
│  ┌──────────────────────────────────────────────────────────────────────┐       │
│  │                       Prediction Head                                 │       │
│  │  ┌───────────────┐    ┌───────────────┐    ┌──────────────────────┐  │       │
│  │  │ Texture Head  │    │ Geometry Head │───▶│ Delta Values         │  │       │
│  │  └───────────────┘    └───────────────┘    └──────────────────────┘  │       │
│  └──────────────────────────────────────────────────────────────────────┘       │
│                                                     │                            │
│                                                     ▼                            │
│  ┌──────────────────────────────────────────────────────────────────────┐       │
│  │                    Gaussian Composer                                  │       │
│  │  Base Values + Delta Values ──▶ Final 3D Gaussians                  │       │
│  └──────────────────────────────────────────────────────────────────────┘       │
│                                                     │                            │
│                                                     ▼                            │
│  ┌──────────────────────────────────────────────────────────────────────┐       │
│  │                    Unproject to World Space                           │       │
│  └──────────────────────────────────────────────────────────────────────┘       │
│                                                     │                            │
│                                                     ▼                            │
│                                          ┌─────────────────┐                     │
│                                          │  Output PLY     │                     │
│                                          │  (3D Gaussians) │                     │
│                                          └─────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 核心组件详解

### 1. Monodepth Network (单目深度网络)

**功能**: 从输入图像估计深度图 (视差图)，并提取多尺度特征。

**对应文件**: 
- `src/sharp/models/monodepth.py`
- `src/sharp/models/encoders/spn_encoder.py`
- `src/sharp/models/decoders/multires_conv_decoder.py`

**架构细节**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MonodepthDensePredictionTransformer               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Sliding Pyramid Network (SPN)             │    │
│  │                                                              │    │
│  │   Input Image (1536x1536)                                    │    │
│  │        │                                                     │    │
│  │        ▼                                                     │    │
│  │   Image Pyramid:                                             │    │
│  │   ├─ x0: 1536x1536 ──▶ 5x5 patches @ 384x384 (overlap=0.25) │    │
│  │   ├─ x1: 768x768   ──▶ 3x3 patches @ 384x384 (overlap=0.5)  │    │
│  │   └─ x2: 384x384   ──▶ 1x1 patch  @ 384x384                 │    │
│  │        │                                                     │    │
│  │        ▼                                                     │    │
│  │   Patch Encoder (DINOv2-L/16)                                │    │
│  │   ├─ Patch Embed: [35, 3, 384, 384] ──▶ [35, 576, 1024]     │    │
│  │   ├─ Position Embed                                         │    │
│  │   ├─ 24 Transformer Blocks                                  │    │
│  │   │   └─ Intermediate features from blocks 5, 11, 17, 23    │    │
│  │   └─ Output: [35, 1024, 24, 24]                             │    │
│  │        │                                                     │    │
│  │        ▼                                                     │    │
│  │   Merge Patches                                              │    │
│  │   ├─ x0_features: [1, 1024, 96, 96]                         │    │
│  │   ├─ x1_features: [1, 1024, 48, 48]                         │    │
│  │   └─ x2_features: [1, 1024, 24, 24]                         │    │
│  │        │                                                     │    │
│  │        ▼                                                     │    │
│  │   Upsample & Fuse                                            │    │
│  │   ├─ x_latent0: [1, 256, 768, 768]  (upsample_latent0)      │    │
│  │   ├─ x_latent1: [1, 256, 384, 384]  (upsample_latent1)      │    │
│  │   ├─ x0: [1, 512, 192, 192]         (upsample0)             │    │
│  │   ├─ x1: [1, 1024, 96, 96]          (upsample1)             │    │
│  │   └─ x_lowres: [1, 1024, 48, 48]    (fuse_lowres)           │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │               MultiresConvDecoder (DPT Decoder)              │    │
│  │                                                              │    │
│  │   Input: 5-level features [256, 256, 512, 1024, 1024]        │    │
│  │        │                                                     │    │
│  │        ▼                                                     │    │
│  │   Feature Fusion Blocks (bottom-up)                          │    │
│  │   ├─ Conv projection at each level                           │    │
│  │   ├─ Fusion with skip connections                            │    │
│  │   └─ Upsampling (2x) between levels                          │    │
│  │        │                                                     │    │
│  │        ▼                                                     │    │
│  │   Output: [1, 256, 768, 768]                                 │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      Prediction Head                         │    │
│  │                                                              │    │
│  │   Conv3x3 ──▶ ConvTranspose2d ──▶ Conv3x3 ──▶ ReLU ──▶ Conv1x1│    │
│  │                                                              │    │
│  │   Output: Disparity [1, 2, 1536, 1536] (2 layers)           │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

**输出**:
- `disparity`: 视差图 (深度的倒数)
- `encoder_features`: 多尺度编码器特征
- `decoder_features`: 解码器特征
- `output_features`: 用于高斯预测器的特征列表
- `intermediate_features`: 用于蒸馏的中间特征

### 2. Initializer (初始化器)

**功能**: 从深度图和图像生成高斯基值 (Base Values)，为后续预测提供初始值。

**对应文件**: `src/sharp/models/initializer.py`

**架构细节**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                       MultiLayerInitializer                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Inputs:                                                              │
│  ├─ image: [B, 3, H, W]                                              │
│  └─ depth: [B, 1, H, W] (from monodepth)                             │
│                                                                       │
│  Processing:                                                          │
│  ├─ Normalize depth (optional)                                       │
│  │   └─ depth_factor = depth_min / min(depth)                        │
│  │   └─ depth = depth * depth_factor                                 │
│  │                                                                  │
│  ├─ Create disparity layers (num_layers=2)                          │
│  │   ├─ first_layer: surface_min (max pooling of disparity)         │
│  │   └─ rest_layer: surface_min                                     │
│  │                                                                  │
│  ├─ Generate base values:                                           │
│  │   ├─ mean_x_ndc: [B, 1, N, H//stride, W//stride]                 │
│  │   ├─ mean_y_ndc: [B, 1, N, H//stride, W//stride]                 │
│  │   ├─ mean_inverse_z_ndc: disparity                               │
│  │   ├─ scales: base_scale * 1/disparity                            │
│  │   ├─ quaternions: [1, 0, 0, 0] (identity)                        │
│  │   ├─ colors: from image (avg pooling)                            │
│  │   └─ opacities: 1/num_layers                                     │
│  │                                                                  │
│  ├─ Prepare feature input:                                          │
│  │   └─ cat([image, normalized_disparity])                          │
│  │                                                                  │
│  Output: InitializerOutput                                           │
│  ├─ gaussian_base_values: GaussianBaseValues                        │
│  ├─ feature_input: [B, 4, H, W] (RGB + disparity)                   │
│  └─ global_scale: for unnormalization                               │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 3. Gaussian Decoder (高斯解码器)

**功能**: 使用 Dense Prediction Transformer (DPT) 架构，结合图像特征和深度特征，预测高斯参数的变化量 (delta values)。

**对应文件**: `src/sharp/models/gaussian_decoder.py`

**架构细节**:

```
┌─────────────────────────────────────────────────────────────────────┐
│              GaussianDensePredictionTransformer (DPT)                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Inputs:                                                              │
│  ├─ input_features: [B, 4, H, W] (RGBD from initializer)            │
│  └─ encodings: list of features from monodepth encoder              │
│                                                                       │
│  Processing:                                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 1. Decode multi-resolution features                          │    │
│  │    decoder(encodings) ──▶ [B, 256, 768, 768]                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 2. Image Encoder (SkipConvBackbone)                          │    │
│  │    Conv2d(kernel=3, stride=2) on input_features              │    │
│  │    Output: [B, 256, 384, 384]                                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 3. Feature Fusion                                            │    │
│  │    FeatureFusionBlock2d(decoder_features, skip_features)     │    │
│  │    Output: [B, 256, 768, 768]                                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 4. Prediction Heads                                          │    │
│  │                                                              │    │
│  │   Texture Head:                                              │    │
│  │   ├─ ResidualBlock(256 ──▶ 256)                              │    │
│  │   ├─ ResidualBlock(256 ──▶ 256)                              │    │
│  │   ├─ ReLU ──▶ Conv1x1 ──▶ ReLU                               │    │
│  │   └─ Output: texture_features [B, 32, 768, 768]              │    │
│  │                                                              │    │
│  │   Geometry Head:                                             │    │
│  │   ├─ ResidualBlock(256 ──▶ 256)                              │    │
│  │   ├─ ResidualBlock(256 ──▶ 256)                              │    │
│  │   ├─ ReLU ──▶ Conv1x1 ──▶ ReLU                               │    │
│  │   └─ Output: geometry_features [B, 32, 768, 768]             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  Output: ImageFeatures                                               │
│  ├─ texture_features: [B, 32, 768, 768]                             │
│  └─ geometry_features: [B, 32, 768, 768]                            │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 4. Prediction Head (预测头)

**功能**: 将解码器的特征转换为高斯参数的 delta 值。

**对应文件**: `src/sharp/models/heads.py`

**架构细节**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DirectPredictionHead                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Inputs:                                                              │
│  ├─ geometry_features: [B, 32, H, W]                                │
│  └─ texture_features: [B, 32, H, W]                                 │
│                                                                       │
│  Processing:                                                          │
│  ├─ geometry_prediction_head: Conv2d(32, 3*N, 1)                    │
│  │   └─ Output: delta for means [B, 3, N, H, W]                     │
│  │                                                                  │
│  └─ texture_prediction_head: Conv2d(32, 11*N, 1)                    │
│      ├─ scales: [B, 3, N, H, W]                                     │
│      ├─ quaternions: [B, 4, N, H, W]                                │
│      ├─ colors: [B, 3, N, H, W]                                     │
│      └─ opacities: [B, 1, N, H, W]                                  │
│                                                                       │
│  Output: delta_values [B, 14, N, H, W]                              │
│  (14 = 3 means + 3 scales + 4 quaternions + 3 colors + 1 opacity)   │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 5. Gaussian Composer (高斯组合器)

**功能**: 将基值 (base values) 和 delta 值组合，应用激活函数，生成最终的 3D 高斯参数。

**对应文件**: `src/sharp/models/composer.py`

**处理流程**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GaussianComposer                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Inputs:                                                              │
│  ├─ delta: [B, 14, N, H, W] (from prediction head)                  │
│  ├─ base_values: GaussianBaseValues (from initializer)              │
│  └─ global_scale: [B] (for metric space conversion)                 │
│                                                                       │
│  Processing (per attribute):                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Mean Activation:                                             │    │
│  │   xx = base_x + delta_x                                       │    │
│  │   yy = base_y + delta_y                                       │    │
│  │   inverse_zz = softplus(inverse_softplus(base_z) + delta_z)  │    │
│  │   mean = [zz*xx, zz*yy, zz]                                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Scale Activation:                                            │    │
│  │   scale_factor = (max_scale - min_scale) *                   │    │
│  │                  sigmoid(a * delta + b) + min_scale          │    │
│  │   scales = base_scales * scale_factor                        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Quaternion Activation:                                       │    │
│  │   quaternions = base_quaternions + delta_quaternions         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Color Activation:                                            │    │
│  │   colors = activation(inverse_activation(base) + delta)      │    │
│  │   (supports: sigmoid, exp, softplus)                         │    │
│  │   Convert to linearRGB if needed                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Opacity Activation:                                          │    │
│  │   opacities = activation(inverse_activation(base) + delta)   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  Post-processing:                                                     │
│  ├─ Flatten spatial dimensions: [B, N, H, W, C] ──▶ [B, N*H*W, C]   │
│  └─ Apply global_scale to means and scales                          │
│                                                                       │
│  Output: Gaussians3D                                                  │
│  ├─ mean_vectors: [B, N*H*W, 3]                                     │
│  ├─ singular_values: [B, N*H*W, 3]                                  │
│  ├─ quaternions: [B, N*H*W, 4]                                      │
│  ├─ colors: [B, N*H*W, 3]                                           │
│  └─ opacities: [B, N*H*W]                                           │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## 完整数据流

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           完整数据流图                                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  Stage 1: 输入预处理                                                               │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  Input Image (H, W, 3)                                                           │
│       │                                                                          │
│       ▼                                                                          │
│  Resize to 1536x1536 ──▶ Normalize to [0,1] ──▶ [1, 3, 1536, 1536]              │
│       │                                                                          │
│       ▼                                                                          │
│  disparity_factor = f_px / width                                                 │
│                                                                                  │
│  Stage 2: 深度估计与特征提取 (Monodepth)                                          │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  [1, 3, 1536, 1536]                                                              │
│       │                                                                          │
│       ▼                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐       │
│  │ Sliding Pyramid Network                                              │       │
│  │ ├─ Create image pyramid (1536, 768, 384)                             │       │
│  │ ├─ Extract patches with sliding window                               │       │
│  │ ├─ ViT encoding (DINOv2-L/16)                                        │       │
│  │ └─ Multi-scale feature fusion                                        │       │
│  │     Output: [256@768x768, 256@384x384, 512@192x192,                  │       │
│  │             1024@96x96, 1024@48x48]                                  │       │
│  └──────────────────────────────────────────────────────────────────────┘       │
│       │                                                                          │
│       ▼                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐       │
│  │ MultiresConvDecoder                                                  │       │
│  │ └─ Progressive upsampling and fusion                                 │       │
│  │     Output: [1, 256, 768, 768]                                       │       │
│  └──────────────────────────────────────────────────────────────────────┘       │
│       │                                                                          │
│       ▼                                                                          │
│  Prediction Head ──▶ Disparity [1, 2, 1536, 1536]                               │
│       │                                                                          │
│       ▼                                                                          │
│  Depth = disparity_factor / disparity                                           │
│                                                                                  │
│  Stage 3: 初始化 (Initializer)                                                    │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  Inputs: Image [1, 3, 1536, 1536], Depth [1, 2, 1536, 1536]                     │
│       │                                                                          │
│       ▼                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐       │
│  │ MultiLayerInitializer                                                │       │
│  │ ├─ Normalize depth                                                   │       │
│  │ ├─ Create 2 disparity layers (surface + background)                  │       │
│  │ ├─ Generate base means in NDC                                        │       │
│  │ ├─ Compute base scales from disparity                                │       │
│  │ ├─ Initialize colors from image                                      │       │
│  │ └─ Set base quaternions (identity) and opacities                     │       │
│  │                                                                      │       │
│  │ Output:                                                              │       │
│  │ ├─ gaussian_base_values: [B, C, 2, 768, 768]                        │       │
│  │ ├─ feature_input: [B, 4, 1536, 1536]                                │       │
│  │ └─ global_scale: [B]                                                │       │
│  └──────────────────────────────────────────────────────────────────────┘       │
│                                                                                  │
│  Stage 4: 高斯参数预测 (Gaussian Decoder)                                         │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  feature_input [1, 4, 1536, 1536]                                                │
│       │                                                                          │
│       ▼                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐       │
│  │ GaussianDensePredictionTransformer                                   │       │
│  │ ├─ Decode monodepth features (reuses SPN features)                   │       │
│  │ ├─ Image encoding with skip connection                               │       │
│  │ ├─ Feature fusion                                                    │       │
│  │ └─ Separate texture and geometry heads                               │       │
│  │     Output: texture_features, geometry_features [1, 32, 768, 768]    │       │
│  └──────────────────────────────────────────────────────────────────────┘       │
│       │                                                                          │
│       ▼                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐       │
│  │ DirectPredictionHead                                                 │       │
│  │ ├─ geometry_head: Conv1x1 ──▶ [1, 6, 768, 768] (3*2 layers)          │       │
│  │ └─ texture_head:  Conv1x1 ──▶ [1, 22, 768, 768] (11*2 layers)        │       │
│  │     Output: delta_values [1, 14, 2, 768, 768]                        │       │
│  └──────────────────────────────────────────────────────────────────────┘       │
│                                                                                  │
│  Stage 5: 高斯组合 (Gaussian Composer)                                            │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  Inputs: delta [1, 14, 2, 768, 768], base_values, global_scale                  │
│       │                                                                          │
│       ▼                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐       │
│  │ GaussianComposer                                                     │       │
│  │ ├─ Mean activation (softplus-based)                                  │       │
│  │ ├─ Scale activation (sigmoid-based, clamped)                         │       │
│  │ ├─ Quaternion addition                                               │       │
│  │ ├─ Color activation                                                  │       │
│  │ ├─ Opacity activation                                                │       │
│  │ ├─ Flatten spatial dimensions                                        │       │
│  │ └─ Apply global scale                                                │       │
│  │                                                                      │       │
│  │ Output: Gaussians3D                                                  │       │
│  │ ├─ means: [1, 2*768*768, 3] = [1, 1179648, 3]                       │       │
│  │ ├─ scales: [1, 1179648, 3]                                          │       │
│  │ ├─ quaternions: [1, 1179648, 4]                                     │       │
│  │ ├─ colors: [1, 1179648, 3]                                          │       │
│  │ └─ opacities: [1, 1179648]                                          │       │
│  └──────────────────────────────────────────────────────────────────────┘       │
│                                                                                  │
│  Stage 6: 后处理与输出                                                            │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  Gaussians3D (NDC space)                                                         │
│       │                                                                          │
│       ▼                                                                          │
│  Unproject to world space (using camera intrinsics)                              │
│       │                                                                          │
│       ▼                                                                          │
│  Save to PLY file                                                                │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 关键参数配置

**对应文件**: `src/sharp/models/params.py`

| 参数类别 | 关键参数 | 默认值 | 说明 |
|---------|---------|--------|------|
| **MonodepthParams** | patch_encoder_preset | "dinov2l16_384" | Patch编码器使用DINOv2-L/16 |
| | image_encoder_preset | "dinov2l16_384" | 图像编码器使用DINOv2-L/16 |
| | dims_decoder | (256, 256, 256, 256, 256) | 解码器各层维度 |
| | use_patch_overlap | True | 使用重叠的patches |
| **GaussianDecoderParams** | dim_in | 5 | 输入维度 (RGB + depth) |
| | dim_out | 32 | 输出特征维度 |
| | stride | 2 | 输出步长 |
| | dims_decoder | (128, 128, 128, 128, 128) | DPT解码器维度 |
| **InitializerParams** | num_layers | 2 | 高斯层数 |
| | stride | 2 | 空间下采样率 |
| | base_depth | 10.0 | 基础深度值 |
| | color_option | "all_layers" | 颜色初始化选项 |
| **DeltaFactor** | xy | 0.001 | XY位置delta系数 |
| | z | 0.001 | Z深度delta系数 |
| | color | 0.1 | 颜色delta系数 |
| | scale | 1.0 | 尺度delta系数 |
| | quaternion | 1.0 | 旋转delta系数 |
| | opacity | 1.0 | 透明度delta系数 |

## 文件对应关系总结

| 功能模块 | 主要文件 | 辅助文件 |
|---------|---------|---------|
| **主预测器** | `predictor.py` | `__init__.py` (工厂函数) |
| **单目深度网络** | `monodepth.py` | `encoders/spn_encoder.py`, `decoders/multires_conv_decoder.py` |
| **ViT编码器** | `encoders/vit_encoder.py` | `presets/vit.py` |
| **高斯解码器** | `gaussian_decoder.py` | `decoders/base_decoder.py` |
| **初始化器** | `initializer.py` | - |
| **预测头** | `heads.py` | - |
| **高斯组合器** | `composer.py` | `params.py` (DeltaFactor) |
| **基础模块** | `blocks.py` | - |
| **高斯工具** | `utils/gaussians.py` | `utils/gsplat.py` (渲染) |
| **推理入口** | `cli/predict.py` | - |

## 技术特点

1. **Sliding Pyramid Network (SPN)**: 使用图像金字塔和滑动窗口策略，结合ViT编码器，在保持高分辨率的同时有效处理大图像。

2. **Dense Prediction Transformer (DPT)**: 借鉴DPT架构，将ViT特征解码为密集预测，支持多尺度特征融合。

3. **分层高斯表示**: 使用2层高斯层，第一层表示表面，第二层表示背景/遮挡区域。

4. **残差学习**: 不直接预测高斯参数，而是预测相对于启发式基值的delta值，降低学习难度。

5. **NDC空间预测**: 在归一化设备坐标(NDC)空间中预测，然后通过global_scale转换为度量空间。

6. **分离的纹理和几何头**: 分别处理颜色和几何属性，提高预测质量。
