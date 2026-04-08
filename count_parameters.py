import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from networks.dap import make_model

def count_parameters(model):
    """Count the total and trainable parameters of a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_number(num):
    """Format large numbers with commas and convert to millions if applicable."""
    if num >= 1e6:
        return f"{num:,.0f} ({num/1e6:.2f}M)"
    elif num >= 1e3:
        return f"{num:,.0f} ({num/1e3:.2f}K)"
    else:
        return f"{num:,.0f}"

def main():
    print("="*70)
    print("DAP Model Parameter Count")
    print("="*70)
    
    # Create model with default settings (vitl)
    model = make_model(midas_model_type='vitl', max_depth=1.0)
    model.eval()
    
    # Count total parameters
    total_params, trainable_params = count_parameters(model)
    
    print(f"\nTotal parameters:      {format_number(total_params)}")
    print(f"Trainable parameters:  {format_number(trainable_params)}")
    
    # Count parameters by component
    print("\n" + "="*70)
    print("Parameters by Component")
    print("="*70)
    
    # Encoder (pretrained - DINOv3Adapter)
    encoder_params = sum(p.numel() for p in model.core.pretrained.parameters())
    print(f"\n1. Encoder (DINOv3Adapter):")
    print(f"   Parameters: {format_number(encoder_params)}")
    
    # Depth head
    depth_head_params = sum(p.numel() for p in model.core.depth_head.parameters())
    print(f"\n2. Depth Head (DPTHead):")
    print(f"   Parameters: {format_number(depth_head_params)}")
    
    # Mask head
    mask_head_params = sum(p.numel() for p in model.core.mask_head.parameters())
    print(f"\n3. Mask Head (DPTHead):")
    print(f"   Parameters: {format_number(mask_head_params)}")
    
    # Verify total
    calculated_total = encoder_params + depth_head_params + mask_head_params
    print(f"\n" + "-"*70)
    print(f"Sum of components:     {format_number(calculated_total)}")
    print(f"Actual total:          {format_number(total_params)}")
    print(f"Match: {'✓' if calculated_total == total_params else '✗'}")
    
    # Detailed breakdown of Depth Head
    print("\n" + "="*70)
    print("Depth Head Detailed Breakdown")
    print("="*70)
    
    depth_head = model.core.depth_head
    
    # Projects (4 conv layers)
    projects_params = sum(p.numel() for p in depth_head.projects.parameters())
    print(f"\nProjects (4 Conv2d layers):       {format_number(projects_params)}")
    
    # Resize layers
    resize_params = sum(p.numel() for p in depth_head.resize_layers.parameters())
    print(f"Resize layers:                    {format_number(resize_params)}")
    
    # Scratch module
    scratch_params = sum(p.numel() for p in depth_head.scratch.parameters())
    print(f"Scratch module:                   {format_number(scratch_params)}")
    
    # Detailed breakdown of Scratch
    print("\n  Scratch sub-components:")
    layer1_rn_params = sum(p.numel() for p in depth_head.scratch.layer1_rn.parameters())
    layer2_rn_params = sum(p.numel() for p in depth_head.scratch.layer2_rn.parameters())
    layer3_rn_params = sum(p.numel() for p in depth_head.scratch.layer3_rn.parameters())
    layer4_rn_params = sum(p.numel() for p in depth_head.scratch.layer4_rn.parameters())
    print(f"    layer1_rn: {format_number(layer1_rn_params)}")
    print(f"    layer2_rn: {format_number(layer2_rn_params)}")
    print(f"    layer3_rn: {format_number(layer3_rn_params)}")
    print(f"    layer4_rn: {format_number(layer4_rn_params)}")
    
    refinenet1_params = sum(p.numel() for p in depth_head.scratch.refinenet1.parameters())
    refinenet2_params = sum(p.numel() for p in depth_head.scratch.refinenet2.parameters())
    refinenet3_params = sum(p.numel() for p in depth_head.scratch.refinenet3.parameters())
    refinenet4_params = sum(p.numel() for p in depth_head.scratch.refinenet4.parameters())
    print(f"    refinenet1: {format_number(refinenet1_params)}")
    print(f"    refinenet2: {format_number(refinenet2_params)}")
    print(f"    refinenet3: {format_number(refinenet3_params)}")
    print(f"    refinenet4: {format_number(refinenet4_params)}")
    
    output_conv1_params = sum(p.numel() for p in depth_head.scratch.output_conv1.parameters())
    output_conv2_params = sum(p.numel() for p in depth_head.scratch.output_conv2.parameters())
    print(f"    output_conv1: {format_number(output_conv1_params)}")
    print(f"    output_conv2: {format_number(output_conv2_params)}")
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"\nDAP (with vitl encoder) has {format_number(total_params)} parameters")
    print(f"  - Encoder:  {format_number(encoder_params)} ({encoder_params/total_params*100:.1f}%)")
    print(f"  - Depth Head:  {format_number(depth_head_params)} ({depth_head_params/total_params*100:.1f}%)")
    print(f"  - Mask Head:   {format_number(mask_head_params)} ({mask_head_params/total_params*100:.1f}%)")

if __name__ == "__main__":
    main()
