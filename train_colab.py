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
    """Auto-download COCO dataset if no images found."""
    import urllib.request
    import zipfile
    import shutil
    from pathlib import Path
    
    print("\n📦 Auto-downloading COCO dataset...")
    print("   Dataset: COCO val2017 (5000 diverse images)")
    print("   Size: ~1GB")
    print("   Content: Natural scenes, objects, people, animals")
    print()
    
    url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = "val2017.zip"
    extract_dir = "val2017"
    
    try:
        # Download
        print("⬇️  Downloading... (this may take 5-10 minutes)")
        
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            mb_down = downloaded / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            print(f"\r   Progress: {percent:.1f}% ({mb_down:.1f} MB / {mb_total:.1f} MB)", end='')
        
        urllib.request.urlretrieve(url, zip_path, progress_hook)
        print("\n✅ Download complete!")
        
        # Extract
        print("📂 Extracting images...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        print("✅ Extraction complete!")
        
        # Move to data/
        print("📁 Organizing images...")
        data_dir = Path('data')
        source_dir = Path(extract_dir)
        
        images = list(source_dir.glob('*.jpg'))
        for i, img_path in enumerate(images[:5000]):
            shutil.copy2(img_path, data_dir / img_path.name)
            if (i + 1) % 500 == 0:
                print(f"   Copied {i + 1}/{len(images)} images...")
        
        print(f"✅ {len(list(data_dir.glob('*.jpg')))} images ready!")
        
        # Cleanup
        print("🧹 Cleaning up...")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        
        print("✅ Dataset download complete!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nPlease manually add images to data/ folder")
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
