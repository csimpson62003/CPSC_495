# Image Inpainting with Diffusion Models

A PyTorch implementation of an image inpainting model using denoising diffusion probabilistic models (DDPM). The model learns to fill in missing or masked regions of images by training on the CelebA face dataset.

## Quick Start for Google Colab

### 1. Clone the Repository
```bash
!git clone https://github.com/csimpson62003/CPSC_495.git
%cd CPSC_495
```

### 2. Install Dependencies
```bash
!pip install -r requirements.txt
```

### 3. Train the Model
```bash
!python train_inpainting_colab.py
```

This will:
- Download the CelebA dataset from Kaggle (~200k face images)
- Train the mask-conditioned diffusion model
- Save checkpoints to `checkpoints/` every 100 epochs
- Train at 128x128 resolution (upscaled to 512x512 during inference)

Training typically takes several hours depending on your hardware.

### 4. Use the Trained Model
```bash
!python inpaint.py
```

This demo script will:
- Load an image from `my_photos/`
- Create a random mask (or use a provided mask)
- Fill in the masked region using the trained model
- Display before/after comparison
- Upscale result to 512x512

## How It Works

The model uses a **mask-conditioned U-Net** that takes:
- **Masked Image**: Original image with missing regions (black pixels)
- **Binary Mask**: White = keep original, Black = fill in
- **Timestep**: Current diffusion timestep

During training:
- Random masks are generated (rectangles, circles, brush strokes)
- The model learns to denoise only the masked regions
- Loss is computed only on masked pixels
- EMA (Exponential Moving Average) weights for stability

During inference:
- The model iteratively denoises the masked region
- Known pixels are preserved at each step
- Result is upscaled from 128x128 to 512x512

## Training Configuration

Adjust parameters in `train_inpainting_colab.py`:
- `batch_size`: Default 32 (reduce if out-of-memory)
- `num_epochs`: Default 2000
- `lr`: Learning rate (default: 1e-4)
- `image_size`: Training resolution (default: 128)
- `save_every_n_epochs`: Save checkpoint interval (default: 100)

## Usage Examples

### Basic Inpainting
```python
from main.inpainting_inference import inpaint_image

# Load image and mask
image_path = "my_photos/face.jpg"
mask_path = "my_photos/mask.png"  # White=keep, Black=fill

# Inpaint
result = inpaint_image(
    image_path=image_path,
    mask_path=mask_path,
    checkpoint_path="checkpoints/inpainting_epoch_2000.pth",
    output_size=512
)
```

### Auto-Generate Mask
```python
# The demo script can auto-create masks:
!python inpaint.py  # Creates random mask automatically
```

Mask types:
- `center`: Rectangle in center
- `random`: Random rectangles
- `strokes`: Random brush strokes

## Project Structure

```
CPSC_495/
├── train_inpainting_colab.py  # Training script for Colab
├── inpaint.py                  # Demo inference script
├── requirements.txt            # Python dependencies
├── my_photos/                  # Put your test images here
├── checkpoints/                # Trained models saved here
└── main/                       # Core model components
    ├── inpainting_dataset.py   # Dataset with mask generation
    ├── inpainting_unet.py      # Mask-conditioned U-Net
    ├── train_inpainting.py     # Training logic
    ├── inpainting_inference.py # Inference logic
    ├── unet.py                 # Base U-Net architecture
    ├── ddpm_scheduler.py       # DDPM noise scheduler
    ├── attention.py            # Attention mechanisms
    ├── res_block.py            # Residual blocks
    └── ...                     # Other components
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- einops
- timm
- tqdm
- matplotlib
- numpy
- kagglehub

## Technical Details

### Architecture
- **Base**: U-Net with attention mechanisms and residual blocks
- **Conditioning**: Concatenates masked image + binary mask in parallel branches
- **Resolution**: Trains at 128x128, upscales to 512x512 with bicubic interpolation
- **Timesteps**: 1000 diffusion steps

### Training
- **Dataset**: CelebA (~200k face images)
- **Optimizer**: Adam with learning rate 1e-4
- **Loss**: MSE computed only on masked regions
- **Stability**: EMA weights with decay 0.995

### Mask Generation
The dataset automatically creates diverse masks:
- **Rectangles**: Random position and size (20-80% of image)
- **Circles**: Random center and radius
- **Brush Strokes**: Multiple curved strokes with random width

## Notes

- GPU strongly recommended (CPU training will be extremely slow)
- The model preserves known pixel values during inference
- Larger masks are more challenging to fill convincingly
- Training for 2000+ epochs recommended for best quality
