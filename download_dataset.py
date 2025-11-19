"""
Generate Pattern Dataset for Inpainting Training
=================================================
Creates synthetic geometric patterns perfect for pattern inpainting.

Patterns: Checkers, stripes, dots, grids, waves, shapes, and more!
"""

import os
from generate_patterns import generate_pattern_dataset


def main():
    print("=" * 60)
    print("🎨 Pattern Dataset Generator")
    print("=" * 60)
    print("This will create 5000 synthetic pattern images")
    print("Including: checkers, stripes, dots, grids, waves, shapes")
    print()
    
    response = input("Generate patterns? (y/n): ").lower()
    
    if response == 'y':
        generate_pattern_dataset(
            output_dir='data',
            num_images=5000,
            size=128
        )
    else:
        print("Cancelled.")


if __name__ == "__main__":
    main()
