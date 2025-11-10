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
from tqdm import tqdm
from timm.utils import ModelEmaV3

from .inpainting_dataset import CelebAInpaintingDataset
from .inpainting_unet import InpaintingUNET
from .ddpm_scheduler import DDPM_Scheduler
from .utils import set_seed, setup_cuda_device


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

    # Download CelebA dataset
    print("📥 Downloading CelebA dataset...")
    import kagglehub
    dataset_path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
    print(f"Dataset downloaded to: {dataset_path}")
    
    # Create dataset
    print("📂 Loading images and creating masks...")
    train_dataset = CelebAInpaintingDataset(
        dataset_path=os.path.join(dataset_path, "img_align_celeba"),
        image_size=image_size,
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
