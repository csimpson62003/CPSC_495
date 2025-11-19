"""
Generate Synthetic Pattern Dataset
===================================
Creates a diverse dataset of geometric patterns for inpainting training.

Patterns use RED and BLUE colors for clear distinction.

Patterns include:
- Checkerboards (various sizes)
- Stripes (horizontal, vertical, diagonal, varying widths)
- Dots/circles (grids and random)
- Grids and lattices
- Waves and zigzags
- Concentric shapes
- Mixed patterns
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import random
from pathlib import Path
from tqdm import tqdm


# Color definitions: RED and BLUE
RED = (255, 0, 0)
BLUE = (0, 0, 255)


def create_checkerboard(size=128, square_size=None):
    """Create a checkerboard pattern with red and blue."""
    if square_size is None:
        square_size = random.choice([4, 8, 16, 32])
    
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(0, size, square_size):
        for j in range(0, size, square_size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                img[i:i+square_size, j:j+square_size] = RED
            else:
                img[i:i+square_size, j:j+square_size] = BLUE
    
    return img


def create_stripes(size=128, orientation='horizontal', stripe_width=None):
    """Create striped pattern with red and blue."""
    if stripe_width is None:
        stripe_width = random.choice([2, 4, 8, 16, 32])
    
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    if orientation == 'horizontal':
        for i in range(0, size, stripe_width * 2):
            img[i:i+stripe_width, :] = RED
            if i + stripe_width < size:
                img[i+stripe_width:i+stripe_width*2, :] = BLUE
    elif orientation == 'vertical':
        for j in range(0, size, stripe_width * 2):
            img[:, j:j+stripe_width] = RED
            if j + stripe_width < size:
                img[:, j+stripe_width:j+stripe_width*2] = BLUE
    elif orientation == 'diagonal':
        # Create diagonal stripes
        img[:, :] = BLUE
        for i in range(-size, size * 2, stripe_width * 2):
            for x in range(size):
                y = x + i
                if 0 <= y < size:
                    for w in range(stripe_width):
                        if 0 <= y + w < size:
                            img[y + w, x] = RED
    
    return img


def create_dots(size=128, dot_size=None, spacing=None):
    """Create dot/circle pattern with red dots on blue background."""
    if dot_size is None:
        dot_size = random.choice([2, 4, 6, 8])
    if spacing is None:
        spacing = random.choice([12, 16, 20, 24, 32])
    
    img = Image.new('RGB', (size, size), BLUE)
    draw = ImageDraw.Draw(img)
    
    for i in range(0, size, spacing):
        for j in range(0, size, spacing):
            x0 = j - dot_size // 2
            y0 = i - dot_size // 2
            x1 = j + dot_size // 2
            y1 = i + dot_size // 2
            draw.ellipse([x0, y0, x1, y1], fill=RED)
    
    return np.array(img)


def create_grid(size=128, line_width=None, spacing=None):
    """Create grid pattern with red lines on blue background."""
    if line_width is None:
        line_width = random.choice([1, 2, 3, 4])
    if spacing is None:
        spacing = random.choice([16, 20, 24, 32])
    
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = BLUE
    
    # Horizontal lines
    for i in range(0, size, spacing):
        img[i:i+line_width, :] = RED
    
    # Vertical lines
    for j in range(0, size, spacing):
        img[:, j:j+line_width] = RED
    
    return img


def create_waves(size=128, orientation='horizontal'):
    """Create wave pattern with red wave on blue background."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = BLUE
    
    amplitude = random.randint(10, 30)
    frequency = random.uniform(0.1, 0.3)
    thickness = random.choice([2, 4, 6, 8])
    
    if orientation == 'horizontal':
        for x in range(size):
            y_center = int(size // 2 + amplitude * np.sin(frequency * x))
            for t in range(-thickness//2, thickness//2 + 1):
                y = y_center + t
                if 0 <= y < size:
                    img[y, x] = RED
    else:  # vertical
        for y in range(size):
            x_center = int(size // 2 + amplitude * np.sin(frequency * y))
            for t in range(-thickness//2, thickness//2 + 1):
                x = x_center + t
                if 0 <= x < size:
                    img[y, x] = RED
    
    return img


def create_concentric_circles(size=128):
    """Create concentric circles with red on blue background."""
    img = Image.new('RGB', (size, size), BLUE)
    draw = ImageDraw.Draw(img)
    
    center = size // 2
    num_circles = random.randint(4, 8)
    max_radius = size // 2 - 5
    
    for i in range(num_circles):
        radius = int(max_radius * (i + 1) / num_circles)
        draw.ellipse([center - radius, center - radius, 
                     center + radius, center + radius], 
                     outline=RED, width=random.choice([1, 2, 3]))
    
    return np.array(img)


def create_concentric_squares(size=128):
    """Create concentric squares with red on blue background."""
    img = Image.new('RGB', (size, size), BLUE)
    draw = ImageDraw.Draw(img)
    
    num_squares = random.randint(4, 8)
    line_width = random.choice([1, 2, 3])
    
    for i in range(num_squares):
        offset = int((size // 2) * (i + 1) / num_squares)
        # Ensure we have valid coordinates
        if offset + line_width < size - offset:
            draw.rectangle([offset, offset, size - offset - 1, size - offset - 1], 
                          outline=RED, width=line_width)
    
    return np.array(img)


def create_zigzag(size=128, orientation='horizontal'):
    """Create zigzag pattern with red on blue background."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = BLUE
    
    amplitude = random.randint(8, 20)
    period = random.randint(16, 32)
    thickness = random.choice([2, 4, 6])
    
    if orientation == 'horizontal':
        for x in range(size):
            # Sawtooth wave
            y_center = int(size // 2 + amplitude * (2 * ((x % period) / period) - 1))
            for t in range(-thickness//2, thickness//2 + 1):
                y = y_center + t
                if 0 <= y < size:
                    img[y, x] = RED
    else:
        for y in range(size):
            x_center = int(size // 2 + amplitude * (2 * ((y % period) / period) - 1))
            for t in range(-thickness//2, thickness//2 + 1):
                x = x_center + t
                if 0 <= x < size:
                    img[y, x] = RED
    
    return img


def create_random_shapes(size=128):
    """Create random geometric shapes with red on blue background."""
    img = Image.new('RGB', (size, size), BLUE)
    draw = ImageDraw.Draw(img)
    
    num_shapes = random.randint(3, 8)
    
    for _ in range(num_shapes):
        shape_type = random.choice(['circle', 'rectangle', 'triangle'])
        x = random.randint(10, size - 10)
        y = random.randint(10, size - 10)
        s = random.randint(10, 30)
        
        if shape_type == 'circle':
            draw.ellipse([x - s, y - s, x + s, y + s], fill=RED)
        elif shape_type == 'rectangle':
            draw.rectangle([x - s, y - s, x + s, y + s], fill=RED)
        else:  # triangle
            points = [(x, y - s), (x - s, y + s), (x + s, y + s)]
            draw.polygon(points, fill=RED)
    
    return np.array(img)


def create_mixed_pattern(size=128):
    """Combine multiple patterns."""
    # Start with a base pattern
    base_type = random.choice(['checkerboard', 'stripes', 'dots', 'grid'])
    
    if base_type == 'checkerboard':
        img = create_checkerboard(size)
    elif base_type == 'stripes':
        img = create_stripes(size, random.choice(['horizontal', 'vertical', 'diagonal']))
    elif base_type == 'dots':
        img = create_dots(size)
    else:
        img = create_grid(size)
    
    return img


def generate_pattern_dataset(output_dir='data', num_images=5000, size=128):
    """Generate complete pattern dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("🎨 Generating Synthetic Pattern Dataset")
    print("=" * 60)
    print(f"Output: {output_path.absolute()}")
    print(f"Images: {num_images}")
    print(f"Size: {size}x{size}")
    print(f"Colors: RED and BLUE")
    print()
    
    patterns = [
        ('checkerboard', create_checkerboard),
        ('horizontal_stripes', lambda s: create_stripes(s, 'horizontal')),
        ('vertical_stripes', lambda s: create_stripes(s, 'vertical')),
        ('diagonal_stripes', lambda s: create_stripes(s, 'diagonal')),
        ('dots', create_dots),
        ('grid', create_grid),
        ('horizontal_waves', lambda s: create_waves(s, 'horizontal')),
        ('vertical_waves', lambda s: create_waves(s, 'vertical')),
        ('concentric_circles', create_concentric_circles),
        ('concentric_squares', create_concentric_squares),
        ('horizontal_zigzag', lambda s: create_zigzag(s, 'horizontal')),
        ('vertical_zigzag', lambda s: create_zigzag(s, 'vertical')),
        ('random_shapes', create_random_shapes),
        ('mixed', create_mixed_pattern),
    ]
    
    print("Pattern types:")
    for name, _ in patterns:
        print(f"  • {name}")
    print()
    
    print("🎨 Generating patterns...")
    for i in tqdm(range(num_images)):
        # Randomly select pattern type
        pattern_name, pattern_func = random.choice(patterns)
        
        # Generate pattern
        img_array = pattern_func(size)
        
        # Save
        img = Image.fromarray(img_array, mode='RGB')
        img.save(output_path / f'pattern_{i:05d}_{pattern_name}.png')
    
    print("\n" + "=" * 60)
    print("✅ DATASET READY!")
    print("=" * 60)
    print(f"📊 Images: {num_images}")
    print(f"📁 Location: {output_path.absolute()}")
    print(f"💾 Total size: ~{num_images * size * size * 3 / (1024*1024):.1f} MB")
    print("\nYou can now run: python train_colab.py")
    print("=" * 60)


if __name__ == "__main__":
    generate_pattern_dataset(
        output_dir='data',
        num_images=5000,
        size=128
    )
