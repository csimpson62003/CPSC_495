"""
Training Script for Image Inpainting
=====================================
Train the inpainting model on your dataset.

This will:
- Load images from the data/ directory
- Generate random masks for training
- Train the diffusion inpainting model
- Save checkpoints to checkpoints/
"""

import os
import torch
from main.train_inpainting import train_inpainting


def main():
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Training configuration
    checkpoint_path = 'checkpoints/inpainting_checkpoint'
    dataset_path = 'data/'  # Put your training images here
    
    print("=" * 60)
    print("🎨 Starting Image Inpainting Model Training")
    print("=" * 60)
    print(f"   - Dataset: {dataset_path}")
    print("   - Task: Fill in masked regions of images")
    print("   - Output: Inpainting model")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
        print("\n⚠️  WARNING: No training data found!")
        print(f"   Please add images to: {dataset_path}")
        print("   Or download a dataset (e.g., CelebA, ImageNet, etc.)")
        return
    
    # Training parameters
    config = {
        'checkpoint_path': checkpoint_path,
        'dataset_path': dataset_path,
        'batch_size': 32,
        'num_epochs': 2000,
        'lr': 1e-4,
        'num_time_steps': 1000,
        'max_dataset_size': None,  # Use all images
        'save_every_n_epochs': 100,
        'image_size': 128
    }
    
    print("\n📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print("\n🎓 Starting training process...")
    print("   This may take several hours depending on your hardware.")
    print("\n💡 Checkpoints will be saved every 100 epochs")
    print("   Monitor the loss - it should decrease steadily!")
    
    # Start training
    train_inpainting(**config)
    
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print("💾 Model saved to: checkpoints/inpainting_checkpoint")
    print("🧪 Test it with: python inpaint.py")



if __name__ == "__main__":
    main()
