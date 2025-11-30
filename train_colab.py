"""
Training Script for Image Inpainting
=====================================
Train the inpainting model on your dataset.

This will:
- Auto-download dataset if not present (COCO val2017)
- Load images from the data/ directory
- Generate random masks for training
- Train the diffusion inpainting model
- Save checkpoints to checkpoints/
"""

import os
import torch
import glob
from main.train_inpainting import train_inpainting


def auto_download_dataset():
    """Auto-generate pattern dataset if no images found."""
    from generate_patterns import generate_pattern_dataset
    
    print("\n🎨 Auto-generating pattern dataset...")
    print("   Dataset: Synthetic geometric patterns")
    print("   Images: 5000")
    print("   Content: Checkers, stripes, dots, grids, waves, shapes")
    print()
    
    try:
        # Generate patterns
        print("🎨 Generating patterns... (this will take 1-2 minutes)")
        generate_pattern_dataset(
            output_dir='data',
            num_images=3000,
            size=32
        )
        print("\n✅ Pattern generation complete!")
        return True
        
    except Exception as e:
        print(f"\n❌ Pattern generation failed: {e}")
        print("\nPlease manually run: python generate_patterns.py")
        return False


def main():
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Training configuration
    checkpoint_path = 'checkpoints/inpainting_checkpoint'
    dataset_path = 'data/'
    
    print("=" * 60)
    print("🎨 Starting Image Inpainting Model Training")
    print("=" * 60)
    print(f"   - Dataset: {dataset_path}")
    print("   - Task: Fill in masked regions of images")
    print("   - Output: Inpainting model")
    
    # Check if dataset exists
    image_files = glob.glob(os.path.join(dataset_path, '**/*.jpg'), recursive=True)
    image_files += glob.glob(os.path.join(dataset_path, '**/*.png'), recursive=True)
    image_files += glob.glob(os.path.join(dataset_path, '**/*.jpeg'), recursive=True)
    
    if len(image_files) == 0:
        print("\n⚠️  No training data found!")
        print("   Attempting to auto-download COCO dataset...")
        
        if not auto_download_dataset():
            print("\n❌ Cannot proceed without training data.")
            return
        
        # Re-check for images
        image_files = glob.glob(os.path.join(dataset_path, '**/*.jpg'), recursive=True)
        image_files += glob.glob(os.path.join(dataset_path, '**/*.png'), recursive=True)
        
        if len(image_files) == 0:
            print("❌ Still no images found. Exiting.")
            return
    
    print(f"\n✅ Found {len(image_files)} training images")
    
    # Training parameters
    config = {
        'checkpoint_path': checkpoint_path,
        'dataset_path': dataset_path,
        'batch_size': 32,
        'num_epochs': 200,
        'lr': 1e-4,
        'num_time_steps': 1000,
        'max_dataset_size': None,  # Use all images
        'save_every_n_epochs': 20,
        'image_size': 128
    }
    
    print("\n📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print("\n🎓 Starting training process...")
    print("   This will run for 100 epochs to test if it's working.")
    print("\n💡 Checkpoints will be saved every 10 epochs")
    print("   Monitor the loss - it should decrease steadily!")
    
    # Start training
    train_inpainting(**config)
    
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print("💾 Model saved to: checkpoints/inpainting_checkpoint")
    print("🧪 Test it with: python inpaint.py")



if __name__ == "__main__":
    main()
