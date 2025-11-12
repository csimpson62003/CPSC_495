"""
Google Colab Training Script
============================
Simple script to train the general image inpainting model after cloning the repo.

This will:
- Download the CIFAR-10 dataset
- Train the diffusion inpainting model
- Save the trained model to checkpoints/
"""

import os
import torch
from main.train_inpainting import train_inpainting


def main():
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training configuration
    checkpoint_path = 'checkpoints/inpainting_checkpoint'
    
    print("=" * 60)
    print("🚀 Starting General Image Inpainting Model Training")
    print("=" * 60)
    print("   - Dataset: CIFAR-10 (diverse images)")
    print("   - Task: Fill holes/masks in any image")
    print("   - Output: General-purpose inpainting model")
    
    # Training parameters - adjust based on your needs
    config = {
        'checkpoint_path': checkpoint_path,
        'batch_size': 32,              # Increase since CIFAR-10 images are smaller
        'num_epochs': 100,             # Fewer epochs needed for CIFAR-10
        'lr': 1e-4,                   # Learning rate
        'num_time_steps': 1000,       # Diffusion timesteps
        'max_dataset_size': 500,     # Use full CIFAR-10 dataset
        'save_every_n_epochs': 10,    # Save checkpoint every 10 epochs
        'image_size': 128             # Training image size
    }
    
    print("\n📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print("\n🎓 Starting training process...")
    print("   This may take several hours depending on your hardware.")
    print("\n💡 Checkpoints will be saved every 100 epochs and pushed to GitHub")
    print("   This protects your progress if Colab disconnects!")
    
    # Start training
    train_inpainting(**config)
    
    print("\n" + "=" * 60)
    print("Training Complete!")



if __name__ == "__main__":
    main()
