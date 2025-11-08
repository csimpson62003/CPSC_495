"""
Training Function for Face-Swap Diffusion Model
===============================================
Trains the U-Net to predict noise added to images at various timesteps.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import subprocess
from torch.utils.data import DataLoader
from tqdm import tqdm
from timm.utils import ModelEmaV3

from .face_swap_dataset import FaceSwapDataset
from .unet import UNET
from .ddpm_scheduler import DDPM_Scheduler
from .utils import set_seed, setup_cuda_device


def train(batch_size: int=64,
      num_time_steps: int=1000,
      num_epochs: int=15,
      seed: int=-1,
      ema_decay: float=0.9999,  
      lr=2e-5,
      checkpoint_path: str=None,
      max_dataset_size: int=None,
      save_every_n_epochs: int=10,
      push_to_github: bool=False):
    
    # Set random seed for reproducibility
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)

    # Download face-swap dataset from Kaggle
    import kagglehub
    dataset_path = kagglehub.dataset_download("rdjarbeng/face-swap-images")
    print(f"Dataset downloaded to: {dataset_path}")
    
    # Create face-swap dataset and dataloader
    if max_dataset_size is not None:
        print(f"🎯 Training on {max_dataset_size} image pairs (out of ~7000 available)")
    else:
        print("🎯 Training on ALL available image pairs (~7000)")
    
    train_dataset = FaceSwapDataset(
        dataset_path, 
        split="train", 
        image_size=64,
        max_pairs=max_dataset_size
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    # Set up device
    device = setup_cuda_device(preferred_gpu=0)
    
    # Initialize model components
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    model = UNET().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    
    # Load existing checkpoint if it exists
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("Starting training from scratch")
    
    scheduler = scheduler.to(device)
    criterion = nn.MSELoss(reduction='mean')

    # Main training loop
    for i in range(num_epochs):
        total_loss = 0
        
        for bidx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")):
            # Extract face images from batch
            original_faces = batch_data['original'].to(device)
            altered_faces = batch_data['altered'].to(device)
            
            # Train on altered (face-swapped) images
            x = altered_faces
            
            # Diffusion training step
            t = torch.randint(0, num_time_steps, (batch_size,), device=device)
            e = torch.randn_like(x, requires_grad=False)
            a = scheduler.alpha[t].view(batch_size, 1, 1, 1)
            
            # Add noise: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
            x = (torch.sqrt(a) * x) + (torch.sqrt(1 - a) * e)
            
            # Train model to predict the noise
            output = model(x, t)
            
            # Backpropagation
            optimizer.zero_grad()
            loss = criterion(output, e)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)
        
        # Print epoch statistics
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {i+1} | Loss {avg_loss:.5f} | Processed {len(train_dataset)} face-swap pairs')
        
        # Save checkpoint periodically and push to GitHub
        if (i + 1) % save_every_n_epochs == 0 or (i + 1) == num_epochs:
            checkpoint = {
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'ema': ema.state_dict(),
                'epoch': i + 1,
                'loss': avg_loss
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"📁 Checkpoint saved to {checkpoint_path}")
            
            if push_to_github:
                print("📤 Pushing checkpoint to GitHub...")
                try:
                    # Add and commit checkpoint
                    subprocess.run(['git', 'add', checkpoint_path], check=True)
                    subprocess.run(['git', 'commit', '-m', f'Training checkpoint: epoch {i+1}, loss {avg_loss:.5f}'], check=True)
                    subprocess.run(['git', 'push'], check=True)
                    print("✅ Checkpoint pushed to GitHub successfully!")
                except subprocess.CalledProcessError as e:
                    print(f"⚠️  Failed to push to GitHub: {e}")
                    print("   Continuing training...")

    # Save final model
    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(), 
        'ema': ema.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
