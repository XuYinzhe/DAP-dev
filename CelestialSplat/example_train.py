"""
Example training script for CelestialSplat.

This demonstrates how to:
1. Load CelestialSplat with pretrained DAP
2. Set up different training phases
3. Run a simple training loop
"""

import sys
sys.path.insert(0, '/homes/shaun/main/DAP')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from CelestialSplat import CelestialSplat, CelestialSplatConfig


class DummyDataset(Dataset):
    """Dummy dataset for testing."""
    
    def __init__(self, num_samples=100, num_views=4, h=512, w=1024):
        self.num_samples = num_samples
        self.num_views = num_views
        self.h = h
        self.w = w
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random ERP images
        images = torch.randn(self.num_views, 3, self.h, self.w)
        
        # Random camera poses (world-to-camera)
        poses = torch.eye(4).unsqueeze(0).repeat(self.num_views, 1, 1)
        # Add some random rotation/translation
        for i in range(self.num_views):
            angle = 2 * 3.14159 * i / self.num_views
            poses[i, 0, 3] = 2.0 * torch.cos(torch.tensor(angle))
            poses[i, 2, 3] = 2.0 * torch.sin(torch.tensor(angle))
        
        # Target depth (for supervision)
        target_depth = torch.rand(self.num_views, self.h, self.w) * 10.0
        
        # Target image (for rendering loss)
        target_image = torch.randn(self.num_views, 3, self.h, self.w)
        
        return {
            'images': images,
            'poses': poses,
            'target_depth': target_depth,
            'target_image': target_image
        }


class SimpleLoss(nn.Module):
    """Simple loss for CelestialSplat training."""
    
    def __init__(self):
        super().__init__()
        self.lambda_depth = 0.1
        self.lambda_mask = 0.01
        self.lambda_opacity = 0.001
        self.lambda_scale = 0.0001
    
    def forward(self, outputs, targets):
        gaussians = outputs['gaussians']
        dap_depth = outputs['dap_depth']
        dap_mask = outputs['dap_mask']
        
        # Depth loss
        depth_loss = F.l1_loss(
            dap_depth * (1 - dap_mask),
            targets['target_depth'] * (1 - dap_mask)
        )
        
        # Mask regularization (encourage valid mask)
        mask_loss = dap_mask.mean()
        
        # Gaussian regularization
        opacity_reg = gaussians['opacity'].mean()
        scale_reg = gaussians['covariance'].abs().mean()
        
        total_loss = (
            self.lambda_depth * depth_loss +
            self.lambda_mask * mask_loss +
            self.lambda_opacity * opacity_reg +
            self.lambda_scale * scale_reg
        )
        
        return {
            'total': total_loss,
            'depth': depth_loss,
            'mask': mask_loss,
            'opacity_reg': opacity_reg,
            'scale_reg': scale_reg
        }


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    num_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        # Move all batch data to device
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device)
        
        images = batch['images']
        poses = batch['poses']
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images, poses)
        
        # Compute loss
        losses = criterion(outputs, batch)
        loss = losses['total']
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"  Batch [{batch_idx}/{num_batches}] Loss: {loss.item():.4f} "
                  f"(depth: {losses['depth'].item():.4f}, "
                  f"mask: {losses['mask'].item():.4f})")
    
    return total_loss / num_batches


def main():
    """Main training function."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dummy DAP model for testing
    # In practice, replace this with real DAP loading
    from test_celestial_splat import DummyDAPModel
    dap_model = DummyDAPModel(encoder='vitl', max_depth=10.0).to(device)
    
    # Build CelestialSplat
    config = CelestialSplatConfig(
        encoder='vitl',
        num_transformer_layers=4,  # Reduced for faster training
        K_neighbors=2
    )
    model = CelestialSplat(config, dap_model=dap_model).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Create dataset and dataloader
    dataset = DummyDataset(num_samples=50, num_views=4, h=256, w=512)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    # Loss and optimizer
    criterion = SimpleLoss()
    optimizer = torch.optim.Adam(
        model.get_trainable_params(freeze_dap=True),
        lr=1e-4
    )
    
    # Training loop
    num_epochs = 3
    print(f"\nTraining for {num_epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        avg_loss = train_one_epoch(
            model, dataloader, optimizer, criterion, device, epoch
        )
        
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    # Test inference
    print("\nTesting inference...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        images = batch['images'].to(device)
        poses = batch['poses'].to(device)
        
        outputs = model(images, poses)
        gaussians = outputs['gaussians']
        
        print("\nInference output shapes:")
        for key, value in gaussians.items():
            print(f"  {key}: {list(value.shape)}")


if __name__ == '__main__':
    main()
