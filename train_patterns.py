#!/usr/bin/env python3
"""
Train Pattern Inpainting Model
==============================
Train a diffusion model specifically for pixel pattern inpainting.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'main'))

from main.train_inpainting import train_inpainting

def train_pattern_model():
    """Train model on pixel patterns."""
    print("🎨 Starting Pattern Inpainting Training...")
    print("=" * 50)
    
    # Train with pixel patterns
    train_inpainting(
        batch_size=16,           # Smaller batch size for stability
        num_time_steps=1000,     # Standard diffusion timesteps
        num_epochs=500,          # Fewer epochs since patterns are simpler
        seed=42,                 # Fixed seed for reproducibility
        ema_decay=0.9999,        # Exponential moving average
        lr=2e-4,                 # Learning rate
        checkpoint_path="checkpoints/pattern_inpainting_checkpoint",
        max_dataset_size=10000,  # 10k patterns per epoch
        save_every_n_epochs=50,  # Save checkpoints more frequently
        image_size=128           # 128x128 images
    )
    
    print("✅ Pattern inpainting training completed!")

if __name__ == "__main__":
    train_pattern_model()