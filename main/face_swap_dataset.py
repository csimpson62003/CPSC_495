"""
Face-Swap Dataset Loader
========================
Loads face-swap image pairs for training.
"""

import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class FaceSwapDataset(Dataset):
    def __init__(self, dataset_path: str, split: str = "train", image_size: int = 256, max_pairs: int = None):
        """
        Args:
            dataset_path: Path to the face-swap dataset
            split: "train", "val", or "test"
            image_size: Target image size (will resize to image_size x image_size)
            max_pairs: Maximum number of image pairs to load (None = load all)
        """
        self.image_size = image_size
        self.max_pairs = max_pairs
        
        # Dataset path structure
        self.split_path = os.path.join(dataset_path, "kaggle", "working", "dataset", "Faceswap_images", split)
        self.original_dir = os.path.join(self.split_path, "original")
        self.altered_dir = os.path.join(self.split_path, "altered")
        
        print(f"Looking for data in: {self.split_path}")
        
        # Find matching pairs
        self.image_pairs = self._find_matching_pairs()
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print(f"Loaded {len(self.image_pairs)} face-swap pairs from {split} set")
    
    def _find_matching_pairs(self):
        """Find matching original and altered image pairs."""
        pairs = []
        
        original_files = glob.glob(os.path.join(self.original_dir, "*.png"))
        
        # Apply max_pairs limit if specified
        if self.max_pairs is not None:
            print(f"Limiting dataset to {self.max_pairs} pairs")
            estimated_files_needed = min(len(original_files), self.max_pairs * 2)
            original_files = original_files[:estimated_files_needed]
        
        print(f"Processing {len(original_files)} original files...")
        
        for orig_path in original_files:
            if self.max_pairs is not None and len(pairs) >= self.max_pairs:
                break
                
            orig_filename = os.path.basename(orig_path)
            parts = orig_filename.replace('.png', '').split('_')
            if len(parts) >= 2:
                orig_id = parts[0]
                frame_num = parts[1]
                
                # Look for corresponding altered images
                altered_pattern = os.path.join(self.altered_dir, f"*_{orig_id}_{frame_num}.png")
                altered_matches = glob.glob(altered_pattern)
                
                for altered_path in altered_matches:
                    if self.max_pairs is not None and len(pairs) >= self.max_pairs:
                        break
                        
                    altered_filename = os.path.basename(altered_path)
                    altered_parts = altered_filename.replace('.png', '').split('_')
                    if len(altered_parts) >= 3:
                        source_id = altered_parts[0]
                        target_id = altered_parts[1]
                        frame = altered_parts[2]
                        
                        pairs.append({
                            'original': orig_path,
                            'altered': altered_path,
                            'source_id': source_id,
                            'target_id': target_id,
                            'frame': frame
                        })
        
        print(f"Found {len(pairs)} matching face-swap pairs")
        return pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        pair = self.image_pairs[idx]
        
        # Load images
        original_img = Image.open(pair['original']).convert('RGB')
        altered_img = Image.open(pair['altered']).convert('RGB')
        
        # Apply transformations
        original_tensor = self.transform(original_img)
        altered_tensor = self.transform(altered_img)
        
        return {
            'original': original_tensor,
            'altered': altered_tensor,
            'source_id': pair['source_id'],
            'target_id': pair['target_id'],
            'frame': pair['frame']
        }
