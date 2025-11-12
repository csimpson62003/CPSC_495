"""
Training loop for the image inpainting diffusion model.
Trains the model to predict noise in masked regions conditioned on surrounding context.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from timm.utils import ModelEmaV3

from .inpainting_unet import InpaintingUNET
from .ddpm_scheduler import DDPM_Scheduler
from .utils import set_seed, setup_cuda_device
from .inpainting_dataset import InpaintingDataset


def train_inpainting(
      batch_size: int=32,
      num_time_steps: int=1000,
      num_epochs: int=2000,
      seed: int=-1,
      ema_decay: float=0.9999,  
      lr=1e-4,
      checkpoint_path: str=None,
      dataset_path: str='data/',
      max_dataset_size: int=None,
      save_every_n_epochs: int=100,
      image_size: int=128):
    
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)

    train_dataset = InpaintingDataset(
        dataset_path=dataset_path,
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

    device = setup_cuda_device(preferred_gpu=0)
    
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    model = InpaintingUNET().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)
    else:
        start_epoch = 0
    
    scheduler = scheduler.to(device)
    criterion = nn.MSELoss(reduction='mean')

    for i in range(start_epoch, num_epochs):
        total_loss = 0
        
        for bidx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")):
            clean_images = batch_data['image'].to(device)
            masks = batch_data['mask'].to(device)
            
            t = torch.randint(0, num_time_steps, (batch_size,), device=device)
            e = torch.randn_like(clean_images, requires_grad=False)
            a = scheduler.alpha[t].view(batch_size, 1, 1, 1)
            
            noisy_image = (torch.sqrt(a) * clean_images) + (torch.sqrt(1 - a) * e)
            masked_noisy = noisy_image * masks
            
            predicted_noise = model(masked_noisy, t, masks)
            loss = criterion(predicted_noise * (1 - masks), e * (1 - masks))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        if (i + 1) % save_every_n_epochs == 0 or (i + 1) == num_epochs:
            checkpoint = {
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'ema': ema.state_dict(),
                'epoch': i + 1,
                'loss': avg_loss
            }
            torch.save(checkpoint, checkpoint_path)
