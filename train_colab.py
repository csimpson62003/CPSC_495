"""
Google Colab Training Script
============================
Simple script to train the face-swapping model after cloning the repo.

Usage in Google Colab:
1. Clone this repository
2. Install dependencies: !pip install -r requirements.txt
3. Run this script: !python train_colab.py

This will:
- Download the face-swap dataset from Kaggle
- Train the diffusion model
- Save the trained model to checkpoints/
"""

import os
import torch
from main.train import train


def main():
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training configuration
    checkpoint_path = 'checkpoints/ddpm_faceswap_checkpoint'
    
    print("=" * 60)
    print("🚀 Starting Face-Swap Model Training")
    print("=" * 60)
    
    # Training parameters - adjust based on your needs
    config = {
        'checkpoint_path': checkpoint_path,
        'batch_size': 8,              # Reduce if you get OOM errors
        'num_epochs': 1500,            # Number of training epochs
        'lr': 1e-4,                   # Learning rate
        'num_time_steps': 1000,       # Diffusion timesteps
        'max_dataset_size': 200,     # Set to a number (e.g., 1000) to limit dataset size for testing
        'save_every_n_epochs': 10,    # Save checkpoint every N epochs
        'push_to_github': True        # Push checkpoints to GitHub (requires git configured)
    }
    
    print("\n📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print("\n🎓 Starting training process...")
    print("   This may take several hours depending on your hardware.")
    print("\n💡 Checkpoints will be saved every 10 epochs and pushed to GitHub")
    print("   This protects your progress if Colab disconnects!")
    
    # Start training
    train(**config)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print(f"📁 Model saved to: {checkpoint_path}")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("   1. Use main.py or use_my_images.py to perform face swaps")
    print("   2. Put your images in my_photos/ folder")
    print("   3. Run the face swap script")


if __name__ == "__main__":
    main()
