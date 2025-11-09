"""
Image Inpainting Dataset
=========================
Loads images and creates random masks for training image inpainting with diffusion.
"""

import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import random


class InpaintingDataset(Dataset):
    """
    Dataset for image inpainting.
    Loads images and creates random masks to simulate missing regions.
    """
    
    def __init__(self, dataset_path: str, image_size: int = 128, max_images: int = None):
        """
        Args:
            dataset_path: Path to image dataset
            image_size: Target image size
            max_images: Maximum number of images (None = all)
        """
        self.image_size = image_size
        self.max_images = max_images
        
        # Find all images
        self.image_paths = self._find_images(dataset_path)
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print(f"✅ Loaded {len(self.image_paths)} images for inpainting")
    
    def _find_images(self, dataset_path):
        """Find all image files."""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']
        images = []
        
        for ext in extensions:
            pattern = os.path.join(dataset_path, '**', ext)
            images.extend(glob.glob(pattern, recursive=True))
        
        if self.max_images is not None:
            images = images[:self.max_images]
        
        return images
    
    def create_random_mask(self, img_size):
        """
        Create random mask with various shapes and sizes.
        Returns mask where 1 = keep, 0 = inpaint
        """
        mask = np.ones((img_size, img_size), dtype=np.float32)
        
        # Random mask type
        mask_type = random.choice(['rectangle', 'circle', 'random_strokes'])
        
        if mask_type == 'rectangle':
            # Random rectangle
            h = random.randint(img_size // 4, img_size // 2)
            w = random.randint(img_size // 4, img_size // 2)
            y = random.randint(0, img_size - h)
            x = random.randint(0, img_size - w)
            mask[y:y+h, x:x+w] = 0
            
        elif mask_type == 'circle':
            # Random circle
            cy = random.randint(img_size // 4, 3 * img_size // 4)
            cx = random.randint(img_size // 4, 3 * img_size // 4)
            radius = random.randint(img_size // 6, img_size // 3)
            
            y, x = np.ogrid[:img_size, :img_size]
            circle_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
            mask[circle_mask] = 0
            
        else:  # random_strokes
            # Random brush strokes
            num_strokes = random.randint(3, 8)
            for _ in range(num_strokes):
                x1 = random.randint(0, img_size)
                y1 = random.randint(0, img_size)
                x2 = random.randint(0, img_size)
                y2 = random.randint(0, img_size)
                thickness = random.randint(3, 8)
                
                # Simple line drawing
                steps = max(abs(x2 - x1), abs(y2 - y1))
                if steps > 0:
                    for i in range(steps):
                        t = i / steps
                        x = int(x1 + t * (x2 - x1))
                        y = int(y1 + t * (y2 - y1))
                        
                        y_min = max(0, y - thickness // 2)
                        y_max = min(img_size, y + thickness // 2)
                        x_min = max(0, x - thickness // 2)
                        x_max = min(img_size, x + thickness // 2)
                        
                        mask[y_min:y_max, x_min:x_max] = 0
        
        return torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Returns:
            - image: Clean image
            - masked_image: Image with regions masked out
            - mask: Binary mask (1 = keep, 0 = inpaint)
        """
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)
        
        # Create random mask
        mask = self.create_random_mask(self.image_size)
        
        # Apply mask to image (set masked regions to gray)
        masked_image = image_tensor * mask + torch.zeros_like(image_tensor) * (1 - mask)
        
        return {
            'image': image_tensor,           # Clean image (target)
            'masked_image': masked_image,    # Image with holes
            'mask': mask,                    # Mask (1 = keep, 0 = fill)
            'path': img_path
        }


class CelebAInpaintingDataset(InpaintingDataset):
    """CelebA-specific dataset loader."""
    
    def _find_images(self, dataset_path):
        """CelebA has specific structure."""
        possible_paths = [
            os.path.join(dataset_path, 'img_align_celeba', '*.jpg'),
            os.path.join(dataset_path, 'img_align_celeba', '*.png'),
            os.path.join(dataset_path, '*.jpg'),
        ]
        
        images = []
        for pattern in possible_paths:
            found = glob.glob(pattern)
            if found:
                images.extend(found)
                break
        
        if self.max_images is not None:
            images = images[:self.max_images]
        
        print(f"Found {len(images)} CelebA images")
        return images
