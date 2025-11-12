"""
Training for Image Inpainting with Diffusion
=============================================
Trains the model to fill in missing/masked regions of images.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from timm.utils import ModelEmaV3

from .inpainting_unet import InpaintingUNET
from .ddpm_scheduler import DDPM_Scheduler
from .utils import set_seed, setup_cuda_device


class GeneralInpaintingDataset(torch.utils.data.Dataset):
    """
    General-purpose inpainting dataset that wraps any image dataset
    and creates random masks for training.
    """
    
    def __init__(self, base_dataset, max_images=None):
        self.base_dataset = base_dataset
        self.max_images = max_images or len(base_dataset)
        self.length = min(self.max_images, len(base_dataset))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Get image from base dataset
        if isinstance(self.base_dataset[idx], tuple):
            image, _ = self.base_dataset[idx]  # Ignore label for CIFAR-10
        else:
            image = self.base_dataset[idx]
        
        # Create random mask
        mask = self._create_random_mask(image.shape[-2:])  # H, W
        
        # Apply mask to image
        masked_image = image * mask
        
        return {
            'image': image,           # Original clean image  
            'masked_image': masked_image,  # Image with holes
            'mask': mask             # Binary mask (1=keep, 0=inpaint)
        }
    
    def _create_random_mask(self, image_size):
        """Create random mask with holes to fill."""
        h, w = image_size
        mask = torch.ones((1, h, w))
        
        # Create random rectangular holes
        num_holes = random.randint(1, 4)
        for _ in range(num_holes):
            hole_h = random.randint(h//8, h//3)
            hole_w = random.randint(w//8, w//3)
            y = random.randint(0, h - hole_h)
            x = random.randint(0, w - hole_w)
            mask[:, y:y+hole_h, x:x+hole_w] = 0
        
        # Sometimes add circular holes
        if random.random() < 0.3:
            center_y = random.randint(h//4, 3*h//4)
            center_x = random.randint(w//4, 3*w//4)
            radius = random.randint(min(h,w)//10, min(h,w)//4)
            
            y_coords, x_coords = torch.meshgrid(
                torch.arange(h), torch.arange(w), indexing='ij'
            )
            distances = torch.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
            mask[0, distances < radius] = 0
        
        return mask


def train_inpainting(
      batch_size: int=32,
      num_time_steps: int=1000,
      num_epochs: int=2000,
      seed: int=-1,
      ema_decay: float=0.9999,  
      lr=1e-4,
      checkpoint_path: str=None,
      max_dataset_size: int=None,
      save_every_n_epochs: int=100,
      image_size: int=128):
    """
    Train image inpainting model.
    
    The model learns to:
    1. Take an image with missing regions (masked)
    2. Fill in those regions realistically
    3. Match the surrounding context
    """
    
    # Set seed
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)

    # Use CIFAR-10 dataset for general-purpose inpainting (more diverse than faces)
    print("📥 Loading CIFAR-10 dataset for general image inpainting...")
    
    # CIFAR-10 has diverse images: animals, vehicles, objects, etc.
    cifar_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    cifar_dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=cifar_transform
    )
    
    # Create inpainting dataset wrapper
    print("📂 Creating masks for inpainting training...")
    train_dataset = GeneralInpaintingDataset(
        base_dataset=cifar_dataset,
        max_images=max_dataset_size
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=0
    )

    # Setup
    device = setup_cuda_device(preferred_gpu=0)
    
    # Initialize inpainting model
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    model = InpaintingUNET().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    
    # Load checkpoint if exists
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"📥 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("🆕 Starting training from scratch")
        start_epoch = 0
    
    scheduler = scheduler.to(device)
    criterion = nn.MSELoss(reduction='mean')

    # Training loop
    print(f"\n🎓 Training for {num_epochs} epochs...")
    for i in range(start_epoch, num_epochs):
        total_loss = 0
        
        for bidx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")):
            # Get clean images and masks
            clean_images = batch_data['image'].to(device)
            masks = batch_data['mask'].to(device)
            
            # Diffusion training
            t = torch.randint(0, num_time_steps, (batch_size,), device=device)
            e = torch.randn_like(clean_images, requires_grad=False)
            a = scheduler.alpha[t].view(batch_size, 1, 1, 1)
            
            # Add noise to clean image
            noisy_image = (torch.sqrt(a) * clean_images) + (torch.sqrt(1 - a) * e)
            
            # Apply mask (keep known regions, mask unknown)
            masked_noisy = noisy_image * masks
            
            # Train model to predict noise, conditioned on mask
            predicted_noise = model(masked_noisy, t, masks)
            
            # Only compute loss in masked regions (regions to inpaint)
            # This focuses learning on filling holes, not copying known regions
            loss = criterion(predicted_noise * (1 - masks), e * (1 - masks))
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)
            
            total_loss += loss.item()
        
        # Epoch stats
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {i+1}/{num_epochs} | Loss: {avg_loss:.5f}')
        
        # Save checkpoint
        if (i + 1) % save_every_n_epochs == 0 or (i + 1) == num_epochs:
            checkpoint = {
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'ema': ema.state_dict(),
                'epoch': i + 1,
                'loss': avg_loss
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")

    print(f"\n✅ Training complete! Model saved to {checkpoint_path}")
