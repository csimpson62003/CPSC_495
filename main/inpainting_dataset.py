"""
Dataset for loading images and generating random masks for inpainting training.
Creates various mask types (rectangles, circles, strokes) to simulate missing regions.
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
    def __init__(self, dataset_path: str, image_size: int = 128, max_images: int = None):
        self.image_size = image_size
        self.max_images = max_images
        
        self.image_paths = self._find_images(dataset_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _find_images(self, dataset_path):
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']
        images = []
        
        for ext in extensions:
            pattern = os.path.join(dataset_path, '**', ext)
            images.extend(glob.glob(pattern, recursive=True))
        
        if self.max_images is not None:
            images = images[:self.max_images]
        
        return images
    
    def create_random_mask(self, img_size):
        mask = np.ones((img_size, img_size), dtype=np.float32)
        
        mask_type = random.choice(['rectangle', 'circle', 'random_strokes'])
        
        if mask_type == 'rectangle':
            h = random.randint(img_size // 4, img_size // 2)
            w = random.randint(img_size // 4, img_size // 2)
            y = random.randint(0, img_size - h)
            x = random.randint(0, img_size - w)
            mask[y:y+h, x:x+w] = 0
            
        elif mask_type == 'circle':
            cy = random.randint(img_size // 4, 3 * img_size // 4)
            cx = random.randint(img_size // 4, 3 * img_size // 4)
            radius = random.randint(img_size // 6, img_size // 3)
            
            y, x = np.ogrid[:img_size, :img_size]
            circle_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
            mask[circle_mask] = 0
            
        else:
            num_strokes = random.randint(3, 8)
            for _ in range(num_strokes):
                x1 = random.randint(0, img_size)
                y1 = random.randint(0, img_size)
                x2 = random.randint(0, img_size)
                y2 = random.randint(0, img_size)
                thickness = random.randint(3, 8)
                
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
        
        return torch.from_numpy(mask).unsqueeze(0)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)
        
        mask = self.create_random_mask(self.image_size)
        # Use gray (0.0) instead of black (-1.0) for masked regions to distinguish from black pattern areas
        masked_image = image_tensor * mask + torch.zeros_like(image_tensor) * (1 - mask)
        
        return {
            'image': image_tensor,
            'masked_image': masked_image,
            'mask': mask,
            'path': img_path
        }
