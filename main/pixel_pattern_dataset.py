"""
Pixel Pattern Dataset for Inpainting
====================================
Generates synthetic pixel patterns like solid colors, rainbows, checkers, etc.
Perfect for training inpainting models on geometric and predictable patterns.
"""

import torch
import numpy as np
import random
from torch.utils.data import Dataset
import math


class PixelPatternDataset(Dataset):
    """
    Dataset that generates synthetic pixel patterns for inpainting training.
    Creates patterns like:
    - Solid colored squares/rectangles
    - Rainbow gradients (horizontal, vertical, radial)
    - Checker patterns
    - Stripes
    - Concentric circles
    - Diagonal patterns
    """
    
    def __init__(self, 
                 size: int = 10000,
                 image_size: int = 128,
                 pattern_types: list = None):
        """
        Args:
            size: Number of synthetic patterns to generate per epoch
            image_size: Size of generated images (image_size x image_size)
            pattern_types: List of pattern types to generate
        """
        self.size = size
        self.image_size = image_size
        
        if pattern_types is None:
            self.pattern_types = [
                'solid_color',
                'horizontal_rainbow', 
                'vertical_rainbow',
                'radial_rainbow',
                'checkerboard',
                'horizontal_stripes',
                'vertical_stripes',
                'diagonal_stripes',
                'concentric_circles',
                'gradient'
            ]
        else:
            self.pattern_types = pattern_types
            
        print(f"✅ Created PixelPatternDataset with {self.size} patterns")
        print(f"   Pattern types: {self.pattern_types}")
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        """Generate a random pixel pattern and mask."""
        # Choose random pattern type
        pattern_type = random.choice(self.pattern_types)
        
        # Generate the pattern
        image = self._generate_pattern(pattern_type)
        
        # Create random mask
        mask = self._create_random_mask()
        
        # Apply mask to image
        masked_image = image * mask
        
        return {
            'image': image,           # Original pattern (target)
            'masked_image': masked_image,  # Pattern with holes
            'mask': mask,             # Binary mask (1=keep, 0=inpaint)
            'pattern_type': pattern_type
        }
    
    def _generate_pattern(self, pattern_type: str):
        """Generate a specific type of pixel pattern."""
        h, w = self.image_size, self.image_size
        
        if pattern_type == 'solid_color':
            return self._solid_color(h, w)
        elif pattern_type == 'horizontal_rainbow':
            return self._horizontal_rainbow(h, w)
        elif pattern_type == 'vertical_rainbow':
            return self._vertical_rainbow(h, w)
        elif pattern_type == 'radial_rainbow':
            return self._radial_rainbow(h, w)
        elif pattern_type == 'checkerboard':
            return self._checkerboard(h, w)
        elif pattern_type == 'horizontal_stripes':
            return self._horizontal_stripes(h, w)
        elif pattern_type == 'vertical_stripes':
            return self._vertical_stripes(h, w)
        elif pattern_type == 'diagonal_stripes':
            return self._diagonal_stripes(h, w)
        elif pattern_type == 'concentric_circles':
            return self._concentric_circles(h, w)
        elif pattern_type == 'gradient':
            return self._gradient(h, w)
        else:
            return self._solid_color(h, w)  # fallback
    
    def _solid_color(self, h: int, w: int):
        """Generate solid colored rectangle."""
        # Random bright color
        color = torch.rand(3) * 0.8 + 0.2  # Avoid very dark colors
        image = color.view(3, 1, 1).expand(3, h, w)
        return self._normalize_image(image)
    
    def _horizontal_rainbow(self, h: int, w: int):
        """Generate horizontal rainbow gradient."""
        image = torch.zeros(3, h, w)
        
        for i in range(w):
            hue = i / w  # 0 to 1 across width
            rgb = self._hsv_to_rgb(hue, 1.0, 1.0)
            image[:, :, i] = rgb.view(3, 1)
        
        return self._normalize_image(image)
    
    def _vertical_rainbow(self, h: int, w: int):
        """Generate vertical rainbow gradient."""
        image = torch.zeros(3, h, w)
        
        for i in range(h):
            hue = i / h  # 0 to 1 across height
            rgb = self._hsv_to_rgb(hue, 1.0, 1.0)
            image[:, i, :] = rgb.view(3, 1)
        
        return self._normalize_image(image)
    
    def _radial_rainbow(self, h: int, w: int):
        """Generate radial rainbow from center."""
        image = torch.zeros(3, h, w)
        center_y, center_x = h // 2, w // 2
        max_distance = math.sqrt(center_y**2 + center_x**2)
        
        for y in range(h):
            for x in range(w):
                distance = math.sqrt((y - center_y)**2 + (x - center_x)**2)
                hue = (distance / max_distance) % 1.0
                rgb = self._hsv_to_rgb(hue, 1.0, 1.0)
                image[:, y, x] = rgb
        
        return self._normalize_image(image)
    
    def _checkerboard(self, h: int, w: int):
        """Generate checkerboard pattern."""
        # Random square size
        square_size = random.randint(4, 16)
        
        # Two random colors
        color1 = torch.rand(3) * 0.8 + 0.2
        color2 = torch.rand(3) * 0.8 + 0.2
        
        image = torch.zeros(3, h, w)
        
        for y in range(h):
            for x in range(w):
                square_y = y // square_size
                square_x = x // square_size
                
                if (square_y + square_x) % 2 == 0:
                    image[:, y, x] = color1
                else:
                    image[:, y, x] = color2
        
        return self._normalize_image(image)
    
    def _horizontal_stripes(self, h: int, w: int):
        """Generate horizontal stripes."""
        stripe_height = random.randint(3, 12)
        num_colors = random.randint(2, 5)
        colors = [torch.rand(3) * 0.8 + 0.2 for _ in range(num_colors)]
        
        image = torch.zeros(3, h, w)
        
        for y in range(h):
            stripe_idx = (y // stripe_height) % num_colors
            image[:, y, :] = colors[stripe_idx].view(3, 1)
        
        return self._normalize_image(image)
    
    def _vertical_stripes(self, h: int, w: int):
        """Generate vertical stripes."""
        stripe_width = random.randint(3, 12)
        num_colors = random.randint(2, 5)
        colors = [torch.rand(3) * 0.8 + 0.2 for _ in range(num_colors)]
        
        image = torch.zeros(3, h, w)
        
        for x in range(w):
            stripe_idx = (x // stripe_width) % num_colors
            image[:, :, x] = colors[stripe_idx].view(3, 1)
        
        return self._normalize_image(image)
    
    def _diagonal_stripes(self, h: int, w: int):
        """Generate diagonal stripes."""
        stripe_width = random.randint(5, 15)
        num_colors = random.randint(2, 4)
        colors = [torch.rand(3) * 0.8 + 0.2 for _ in range(num_colors)]
        
        image = torch.zeros(3, h, w)
        
        for y in range(h):
            for x in range(w):
                stripe_idx = ((x + y) // stripe_width) % num_colors
                image[:, y, x] = colors[stripe_idx]
        
        return self._normalize_image(image)
    
    def _concentric_circles(self, h: int, w: int):
        """Generate concentric circles."""
        center_y, center_x = h // 2, w // 2
        ring_width = random.randint(5, 15)
        num_colors = random.randint(2, 5)
        colors = [torch.rand(3) * 0.8 + 0.2 for _ in range(num_colors)]
        
        image = torch.zeros(3, h, w)
        
        for y in range(h):
            for x in range(w):
                distance = math.sqrt((y - center_y)**2 + (x - center_x)**2)
                ring_idx = int(distance // ring_width) % num_colors
                image[:, y, x] = colors[ring_idx]
        
        return self._normalize_image(image)
    
    def _gradient(self, h: int, w: int):
        """Generate color gradient."""
        # Random start and end colors
        start_color = torch.rand(3) * 0.8 + 0.2
        end_color = torch.rand(3) * 0.8 + 0.2
        
        # Random direction
        direction = random.choice(['horizontal', 'vertical', 'diagonal'])
        
        image = torch.zeros(3, h, w)
        
        if direction == 'horizontal':
            for x in range(w):
                t = x / (w - 1)
                color = start_color * (1 - t) + end_color * t
                image[:, :, x] = color.view(3, 1)
        elif direction == 'vertical':
            for y in range(h):
                t = y / (h - 1)
                color = start_color * (1 - t) + end_color * t
                image[:, y, :] = color.view(3, 1)
        else:  # diagonal
            for y in range(h):
                for x in range(w):
                    t = (x + y) / (w + h - 2)
                    color = start_color * (1 - t) + end_color * t
                    image[:, y, x] = color
        
        return self._normalize_image(image)
    
    def _hsv_to_rgb(self, h: float, s: float, v: float):
        """Convert HSV to RGB."""
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c
        
        if h < 1/6:
            rgb = torch.tensor([c, x, 0])
        elif h < 2/6:
            rgb = torch.tensor([x, c, 0])
        elif h < 3/6:
            rgb = torch.tensor([0, c, x])
        elif h < 4/6:
            rgb = torch.tensor([0, x, c])
        elif h < 5/6:
            rgb = torch.tensor([x, 0, c])
        else:
            rgb = torch.tensor([c, 0, x])
        
        return rgb + m
    
    def _normalize_image(self, image):
        """Normalize image to [-1, 1] range for diffusion models."""
        return image * 2.0 - 1.0
    
    def _create_random_mask(self):
        """Create random mask with holes to fill."""
        h, w = self.image_size, self.image_size
        mask = torch.ones((1, h, w))
        
        # Create random holes
        num_holes = random.randint(1, 3)
        
        for _ in range(num_holes):
            hole_type = random.choice(['rectangle', 'circle', 'irregular'])
            
            if hole_type == 'rectangle':
                hole_h = random.randint(h//6, h//3)
                hole_w = random.randint(w//6, w//3)
                y = random.randint(0, h - hole_h)
                x = random.randint(0, w - hole_w)
                mask[:, y:y+hole_h, x:x+hole_w] = 0
                
            elif hole_type == 'circle':
                center_y = random.randint(h//4, 3*h//4)
                center_x = random.randint(w//4, 3*w//4)
                radius = random.randint(min(h,w)//8, min(h,w)//4)
                
                y_coords, x_coords = torch.meshgrid(
                    torch.arange(h), torch.arange(w), indexing='ij'
                )
                distances = torch.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
                mask[0, distances < radius] = 0
                
            else:  # irregular
                # Random brush strokes
                num_strokes = random.randint(2, 5)
                for _ in range(num_strokes):
                    y1, x1 = random.randint(0, h-1), random.randint(0, w-1)
                    y2, x2 = random.randint(0, h-1), random.randint(0, w-1)
                    thickness = random.randint(3, 8)
                    
                    # Draw line
                    steps = max(abs(x2 - x1), abs(y2 - y1))
                    if steps > 0:
                        for i in range(steps + 1):
                            t = i / steps
                            y = int(y1 + t * (y2 - y1))
                            x = int(x1 + t * (x2 - x1))
                            
                            y_min = max(0, y - thickness // 2)
                            y_max = min(h, y + thickness // 2 + 1)
                            x_min = max(0, x - thickness // 2)
                            x_max = min(w, x + thickness // 2 + 1)
                            
                            mask[:, y_min:y_max, x_min:x_max] = 0
        
        return mask


def visualize_patterns(dataset, num_samples=9):
    """Visualize sample patterns from the dataset."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_samples):
        sample = dataset[i]
        
        # Convert from [-1,1] to [0,1] for visualization
        image = (sample['image'] + 1) / 2
        image = torch.clamp(image, 0, 1)
        
        # Convert to numpy and transpose for matplotlib
        image_np = image.permute(1, 2, 0).numpy()
        
        axes[i].imshow(image_np)
        axes[i].set_title(f"Pattern: {sample['pattern_type']}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    # Test the dataset
    dataset = PixelPatternDataset(size=100, image_size=64)
    
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Pattern type: {sample['pattern_type']}")
    
    # Uncomment to visualize patterns
    # visualize_patterns(dataset)