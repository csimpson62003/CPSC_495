"""
Download Dataset for Inpainting Training
=========================================
Downloads a diverse, multipurpose dataset suitable for general inpainting.

Using: COCO dataset (Common Objects in Context)
- Natural images with diverse content
- Objects, scenes, people, animals, textures
- Great for general-purpose inpainting
"""

import os
import urllib.request
import zipfile
from pathlib import Path
import shutil


def download_coco_samples():
    """Download COCO validation set (smaller, but diverse)."""
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("📦 Downloading COCO Dataset for Inpainting Training")
    print("=" * 60)
    print("Dataset: COCO val2017 (5000 diverse images)")
    print("Size: ~1GB")
    print("Content: Natural scenes, objects, people, animals")
    print()
    
    # COCO validation set (smaller but still diverse)
    url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = "val2017.zip"
    extract_dir = "val2017"
    
    # Download
    if not os.path.exists(zip_path):
        print("⬇️  Downloading COCO validation set...")
        print(f"   URL: {url}")
        print("   This may take 5-10 minutes...")
        
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            print(f"\r   Progress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)", end='')
        
        urllib.request.urlretrieve(url, zip_path, progress_hook)
        print("\n✅ Download complete!")
    else:
        print("✓ Dataset already downloaded")
    
    # Extract
    if not os.path.exists(extract_dir):
        print("\n📂 Extracting images...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        print("✅ Extraction complete!")
    else:
        print("✓ Dataset already extracted")
    
    # Move images to data/ directory
    print("\n📁 Organizing images...")
    source_dir = Path(extract_dir)
    
    if source_dir.exists():
        # Copy first 5000 images (or all if fewer)
        images = list(source_dir.glob('*.jpg'))
        print(f"   Found {len(images)} images")
        
        for i, img_path in enumerate(images[:5000]):
            dest_path = data_dir / img_path.name
            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)
            
            if (i + 1) % 100 == 0:
                print(f"   Copied {i + 1}/{min(len(images), 5000)} images...")
        
        print(f"✅ {len(list(data_dir.glob('*.jpg')))} images ready in data/")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print("   Removed zip file")
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
        print("   Removed temporary directory")
    
    print("\n" + "=" * 60)
    print("✅ DATASET READY!")
    print("=" * 60)
    print(f"📊 Images: {len(list(data_dir.glob('*.jpg')))}")
    print(f"📁 Location: {data_dir.absolute()}")
    print("\nYou can now run: python train_colab.py")
    print("=" * 60)


def download_alternative_dataset():
    """Alternative: Download Places365 sample (diverse scenes)."""
    print("Alternative option: Places365-Standard (scene-centric)")
    print("Contains: Indoor/outdoor scenes, landscapes, buildings")
    print("\nTo use Places365 instead:")
    print("1. Visit: http://places2.csail.mit.edu/download.html")
    print("2. Download: Small images (256x256) - 24GB")
    print("3. Extract to data/ directory")


if __name__ == "__main__":
    import sys
    
    print("\n🎨 Dataset Options for General-Purpose Inpainting:\n")
    print("1. COCO (Recommended) - Diverse objects & scenes")
    print("   - 5000 images, ~1GB")
    print("   - Natural images with varied content")
    print("   - Best for general multipurpose use")
    print()
    print("2. Places365 - Scene-focused dataset")
    print("   - Many more images, larger download")
    print("   - Focused on scenes and environments")
    print()
    
    choice = input("Enter choice (1 or 2) [default: 1]: ").strip()
    
    if choice == "2":
        download_alternative_dataset()
    else:
        try:
            download_coco_samples()
        except KeyboardInterrupt:
            print("\n\n⚠️  Download cancelled by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            print("\nIf download fails, you can:")
            print("1. Try again (sometimes servers are busy)")
            print("2. Manually download from: http://cocodataset.org/#download")
            print("3. Use any folder of images you have")
            sys.exit(1)
