import os
import sys
import cv2
import torch
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from networks.dap import make_model

IMG_PATH = "/homes/shaun/main/360Splat/dataset/360Roam_512x1024/bar/images/0_0000.jpg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"Device: {DEVICE}")
    print(f"Image path: {IMG_PATH}")
    
    # 1. 创建模型
    print("\n" + "="*70)
    print("Initializing DAP model (vitl, max_depth=1.0)")
    print("="*70)
    model = make_model(midas_model_type='vitl', max_depth=1.0)
    model = model.to(DEVICE)
    model.eval()
    
    # 2. 读取 360 图像
    img_bgr = cv2.imread(IMG_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {IMG_PATH}")
    print(f"\nOriginal image size (H,W,C): {img_bgr.shape}")
    
    # 3. 测试路径 A: 通过 DAP.infer_image (包含预处理)
    print("\n" + "="*70)
    print("TEST A: DAP.infer_image() path (with Resize/Normalize pre-processing)")
    print("="*70)
    with torch.no_grad():
        depth = model.infer_image(img_bgr, input_size=518)
    print(f"\n[DAP.infer_image] final depth output shape: {depth.shape}")
    
    # 4. 测试路径 B: 直接 tensor 输入（无预处理，直接进 core）
    print("\n" + "="*70)
    print("TEST B: Direct tensor input to model.forward()")
    print("="*70)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(DEVICE)
    print(f"Direct tensor shape: {img_tensor.shape}")
    
    with torch.no_grad():
        outputs = model(img_tensor)
    
    print("\n[DAP] output dict:")
    for k, v in outputs.items():
        print(f"  {k}: {v.shape if hasattr(v, 'shape') else type(v)}")
    
    # 5. 测试路径 C: 使用经过预处理后的尺寸（模拟 infer_image 的预处理结果）
    print("\n" + "="*70)
    print("TEST C: Pre-processed tensor (same as infer_image pipeline)")
    print("="*70)
    image_proc, (orig_h, orig_w) = model.image2tensor(img_bgr, input_size=518)
    print(f"Pre-processed tensor shape: {image_proc.shape}, original size: ({orig_h}, {orig_w})")
    
    with torch.no_grad():
        outputs_proc = model(image_proc)
    
    print("\n[DAP] output dict:")
    for k, v in outputs_proc.items():
        print(f"  {k}: {v.shape if hasattr(v, 'shape') else type(v)}")
    
    print("\n" + "="*70)
    print("Shape tracing complete!")
    print("="*70)

if __name__ == "__main__":
    main()
