#!/usr/bin/env python
"""
预处理脚本：对360度RGB图片进行深度估计
- 输入：包含512x1024 RGB图片的目录
- 输出：深度估计结果（.npy格式）
- 可选：深度可视化（.jpg格式）
- 支持多GPU并行处理
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import cv2
import torch
import yaml
import argparse
import numpy as np
import torch.nn as nn
import torch.multiprocessing as mp
from tqdm import tqdm
from glob import glob

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from networks.models import make
import matplotlib


def colorize_depth_fixed(depth_u8: np.ndarray, cmap: str = "Spectral") -> np.ndarray:
    """
    depth_u8: uint8, 0~255
    return: RGB uint8
    """
    disp = depth_u8.astype(np.float32) / 255.0
    colored = matplotlib.colormaps[cmap](disp)[..., :3]
    colored = (colored * 255).astype(np.uint8)
    return np.ascontiguousarray(colored)


def ensure_dir_for_file(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_model(config, device):
    """加载模型到指定设备"""
    model_path = os.path.join(config["load_weights_dir"], "model.pth")
    
    state = torch.load(model_path, map_location=device)
    
    m = make(config["model"])
    if any(k.startswith("module") for k in state.keys()):
        m = nn.DataParallel(m)
    
    m = m.to(device)
    m_state = m.state_dict()
    m.load_state_dict({k: v for k, v in state.items() if k in m_state}, strict=False)
    m.eval()
    
    return m


def infer_raw(model, img_rgb_u8: np.ndarray) -> np.ndarray:
    """
    img_rgb_u8: HWC uint8 RGB
    return: pred float32 (H,W)
    """
    img = img_rgb_u8.astype(np.float32) / 255.0
    
    with torch.inference_mode():
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(model.device)
        outputs = model(tensor)
        
        if isinstance(outputs, dict) and "pred_depth" in outputs:
            if "pred_mask" in outputs:
                mask = 1 - outputs["pred_mask"]
                mask = mask > 0.5
                outputs["pred_depth"][~mask] = 1
            pred = outputs["pred_depth"][0].detach().cpu().squeeze().numpy()
        else:
            pred = outputs[0].detach().cpu().squeeze().numpy()
    
    return pred.astype(np.float32)


def pred_to_vis(pred: np.ndarray, vis_range: str = "100m", cmap: str = "Spectral"):
    """
    return:
      depth_gray_u8: (H,W) uint8
      depth_color_rgb: (H,W,3) uint8 RGB
    """
    if vis_range == "100m":
        pred_clip = np.clip(pred, 0.0, 1.0)
        depth_gray = (pred_clip * 255).astype(np.uint8)
    elif vis_range == "10m":
        pred_clip = np.clip(pred, 0.0, 0.1)
        depth_gray = (pred_clip * 10.0 * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown vis_range: {vis_range} (use '100m' or '10m')")
    
    depth_color = colorize_depth_fixed(depth_gray, cmap=cmap)
    return depth_gray, depth_color


def process_image(args_tuple):
    """
    处理单张图片的函数（用于多进程）
    args_tuple: (img_path, input_dir, output_dir, viz_dir, config, gpu_id, vis_range, cmap)
    """
    img_path, input_dir, output_dir, viz_dir, config, gpu_id, vis_range, cmap = args_tuple
    
    # 设置GPU设备
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    model = load_model(config, device)
    model.device = device  # 用于infer_raw函数
    
    # 读取图片
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"⚠️ 无法读取图片: {img_path}")
        return False
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 检查图片尺寸是否为512x1024
    h, w = img_rgb.shape[:2]
    if h != 512 or w != 1024:
        print(f"⚠️ 图片尺寸不匹配 {img_path}: 期望512x1024, 实际{h}x{w}")
        # return False
        img_rgb = cv2.resize(img_rgb, (1024,512))
    
    # 推理
    pred = infer_raw(model, img_rgb)
    
    # 计算输出路径（保持相对目录结构）
    rel_path = os.path.relpath(img_path, input_dir)
    base_name = os.path.splitext(rel_path)[0]
    
    # 保存npy文件
    npy_path = os.path.join(output_dir, base_name + ".npy")
    ensure_dir_for_file(npy_path)
    np.save(npy_path, pred)
    
    # 保存可视化图片（如果指定了viz_dir）
    if viz_dir is not None:
        depth_gray, depth_color_rgb = pred_to_vis(pred, vis_range=vis_range, cmap=cmap)
        viz_path = os.path.join(viz_dir, base_name + ".jpg")
        ensure_dir_for_file(viz_path)
        cv2.imwrite(viz_path, cv2.cvtColor(depth_color_rgb, cv2.COLOR_RGB2BGR))
    
    return True


def get_all_images(input_dir):
    """获取输入目录下所有图片文件"""
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    img_list = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                img_list.append(os.path.join(root, f))
    return sorted(img_list)


def main():
    parser = argparse.ArgumentParser(description="预处理360度RGB图片进行深度估计")
    parser.add_argument("--inputdir", "-i", required=True, help="输入图片目录（包含512x1024的RGB图片）")
    parser.add_argument("--outputdir", "-o", required=True, help="输出目录（保存.npy深度文件）")
    parser.add_argument("--vizdir", "-v", default=None, help="可视化输出目录（可选，保存.jpg可视化深度）")
    parser.add_argument("--config", "-c", default="config/infer.yaml", help="配置文件路径")
    parser.add_argument("--gpus", "-g", default="0", help="使用的GPU编号，多个用逗号分隔，如'0,1,2,3'")
    parser.add_argument("--vis", default="100m", choices=["100m", "10m"], help="可视化范围（只影响jpg，不影响npy）")
    parser.add_argument("--cmap", default="Spectral", help="matplotlib colormap名称，如Spectral, Turbo, Viridis")
    
    args = parser.parse_args()
    
    # 解析GPU列表
    gpu_list = [int(x.strip()) for x in args.gpus.split(",")]
    num_gpus = len(gpu_list)
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        print(f"🔹 可用GPU数量: {available_gpus}")
        for g in gpu_list:
            if g >= available_gpus:
                print(f"❌ GPU {g} 不存在，可用GPU为 0~{available_gpus-1}")
                return
    else:
        print("⚠️ CUDA不可用，使用CPU")
        gpu_list = [0]
        num_gpus = 1
    
    # 加载配置
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(f"✅ 配置加载完成: {args.config}")
    
    # 获取所有图片
    img_list = get_all_images(args.inputdir)
    if not img_list:
        print(f"❌ 在 {args.inputdir} 中未找到图片文件")
        return
    
    print(f"🔹 找到 {len(img_list)} 张图片")
    print(f"🔹 使用GPU: {gpu_list}")
    print(f"🔹 输出目录: {args.outputdir}")
    if args.vizdir:
        print(f"🔹 可视化目录: {args.vizdir}")
        print(f"🔹 可视化范围: {args.vis}, colormap: {args.cmap}")
    print()
    
    # 创建输出目录
    os.makedirs(args.outputdir, exist_ok=True)
    if args.vizdir:
        os.makedirs(args.vizdir, exist_ok=True)
    
    # 准备任务列表
    tasks = []
    for idx, img_path in enumerate(img_list):
        gpu_id = gpu_list[idx % num_gpus]
        tasks.append((img_path, args.inputdir, args.outputdir, args.vizdir, 
                     config, gpu_id, args.vis, args.cmap))
    
    # 使用多进程并行处理
    num_workers = min(num_gpus, len(img_list))
    print(f"🔹 启动 {num_workers} 个进程进行并行处理...\n")
    
    # 使用spawn方法启动多进程（避免CUDA fork问题）
    mp.set_start_method('spawn', force=True)
    
    success_count = 0
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_image, tasks),
            total=len(tasks),
            desc="深度估计处理中"
        ))
        success_count = sum(results)
    
    print(f"\n✅ 处理完成！成功 {success_count}/{len(img_list)}")
    print(f"   深度npy文件: {args.outputdir}")
    if args.vizdir:
        print(f"   可视化jpg文件: {args.vizdir}")


if __name__ == "__main__":
    main()
