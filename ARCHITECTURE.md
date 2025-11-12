# Image Inpainting Diffusion Model - Architecture Guide

## Overview

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** for image inpainting. The model learns to fill in missing or masked regions of images by iteratively denoising random noise conditioned on the surrounding context and a binary mask.

---

## Training Process Flow

### High-Level Training Steps

1. **Data Loading**: Load images and generate random masks (rectangles, circles, strokes)
2. **Forward Diffusion**: Add noise to clean images according to a noise schedule
3. **Masking**: Apply mask to noisy images (keep known regions, mask unknown)
4. **Noise Prediction**: Model predicts the noise in masked regions only
5. **Loss Calculation**: Compare predicted noise to actual noise in masked regions
6. **Backpropagation**: Update model weights to improve predictions
7. **Checkpoint Saving**: Periodically save model state for inference

### Detailed Training Loop

```
For each epoch:
    For each batch of images:
        1. Load clean image and generate random mask
        2. Sample random timestep t (0 to 1000)
        3. Generate random noise ε ~ N(0, 1)
        4. Add noise to image: x_t = √(α_t) * x_0 + √(1-α_t) * ε
        5. Apply mask: x_masked = x_t * mask
        6. Predict noise: ε_pred = model(x_masked, t, mask)
        7. Compute loss only in masked regions: loss = MSE(ε_pred * (1-mask), ε * (1-mask))
        8. Backpropagate and update weights
        9. Update EMA (Exponential Moving Average) for stable inference
```

---

## File-by-File Architecture

### Core Model Components

#### 1. `inpainting_unet.py` - **Main Model Architecture**
**Purpose**: The primary neural network that performs inpainting

**Key Features**:
- Takes 3 inputs: noisy masked image, timestep, and binary mask
- Processes image and mask through separate conv layers
- Combines both signals to condition generation on mask information
- Encoder-decoder structure with skip connections
- Outputs predicted noise in masked regions

**Flow**:
```
Input: (noisy_image, timestep, mask)
  ↓
[Image Conv] + [Mask Conv] → Combined Features
  ↓
Encoder (3 layers) → Progressively downsample
  ↓
Bottleneck → Compressed representation
  ↓
Decoder (3 layers) → Progressively upsample + skip connections
  ↓
Output: Predicted noise
```

#### 2. `unet.py` - **Base U-Net Architecture**
**Purpose**: Generic U-Net implementation (can be used for other diffusion tasks)

**Key Features**:
- Encoder-decoder with symmetric architecture
- Skip connections preserve spatial information
- Configurable channels, attention layers, and upscale layers
- Time embeddings injected at each layer

**When Used**: Can be used as an alternative to InpaintingUNET for tasks without mask conditioning

#### 3. `unet_layer.py` - **Individual U-Net Layer**
**Purpose**: Building block for U-Net, handles one encoding or decoding step

**Components**:
- 2 residual blocks for feature processing
- Optional self-attention for long-range dependencies
- Conv/TransposeConv for spatial dimension changes

**Flow**:
```
Input features
  ↓
ResBlock 1 (+ time embedding)
  ↓
[Optional] Self-Attention
  ↓
ResBlock 2 (+ time embedding)
  ↓
Conv/TransposeConv (change spatial dims)
  ↓
Output: (processed features, residual for skip connection)
```

#### 4. `res_block.py` - **Residual Block**
**Purpose**: Stable feature processing with skip connections

**Why Important**: 
- Prevents vanishing gradients in deep networks
- Allows training of very deep U-Nets
- Incorporates time embeddings into feature processing

**Flow**:
```
Input + Time Embedding
  ↓
GroupNorm → ReLU → Conv → Dropout
  ↓
GroupNorm → ReLU → Conv
  ↓
Add to original input (skip connection)
```

#### 5. `attention.py` - **Multi-Head Self-Attention**
**Purpose**: Capture long-range spatial dependencies in images

**Why Important**:
- Allows model to relate distant pixels
- Helps maintain global coherence in generated content
- Particularly useful for structural patterns

**How It Works**:
- Splits features into multiple heads
- Each head learns different relationships
- Combines all heads for rich representations

#### 6. `sinusoidal_embeddings.py` - **Time Embeddings**
**Purpose**: Encode diffusion timestep information

**Why Important**:
- Model needs to know "how noisy" the input is
- Different timesteps require different denoising strategies
- Sinusoidal encoding provides smooth, learnable representations

**Output**: High-dimensional vector representing timestep (e.g., t=500 → 512-dim vector)

---

### Training Components

#### 7. `train_inpainting.py` - **Training Loop**
**Purpose**: Main training orchestration

**What It Does**:
1. Loads InpaintingDataset from specified path
2. Initializes InpaintingUNET model
3. Sets up optimizer (Adam) and scheduler (DDPM)
4. For each epoch and batch:
   - Samples random timesteps
   - Adds noise to clean images
   - Applies masks
   - Computes loss in masked regions only
   - Updates model weights
5. Saves checkpoints periodically

**Key Parameters**:
- `batch_size`: Number of images per training step
- `num_epochs`: Total training iterations
- `lr`: Learning rate (default: 1e-4)
- `checkpoint_path`: Where to save trained model
- `dataset_path`: Path to training images

#### 8. `inpainting_dataset.py` - **Data Loading**
**Purpose**: Load images and generate training masks

**What It Does**:
- Recursively finds all images in dataset folder
- Resizes images to training size (128x128)
- Normalizes to [-1, 1] range
- Generates random masks (rectangles, circles, or strokes)
- Returns: clean image, masked image, and mask

**Mask Types**:
- **Rectangle**: Random rectangular holes
- **Circle**: Random circular holes  
- **Random Strokes**: Brush-like stroke patterns

**Output Format**:
```python
{
    'image': clean_image_tensor,        # Ground truth
    'masked_image': masked_image_tensor, # Image with holes
    'mask': binary_mask_tensor,         # 1=keep, 0=fill
    'path': image_file_path
}
```

#### 9. `ddpm_scheduler.py` - **Noise Schedule**
**Purpose**: Manages noise addition/removal schedule

**What It Does**:
- Defines β (beta) schedule: how much noise to add at each step
- Computes α (alpha) schedule: how much signal remains
- Provides noise parameters for any timestep t

**Key Concepts**:
- **Beta Schedule**: Linear from 0.0001 to 0.02 over 1000 steps
- **Alpha**: α_t = 1 - β_t
- **Cumulative Alpha**: ᾱ_t = ∏(α_i) for i=0 to t

**Why Linear Schedule**: Works well for images, provides gradual noise addition

---

### Inference Components

#### 10. `inpainting_inference.py` - **Image Generation**
**Purpose**: Fill in masked regions using trained model

**Inference Process**:
1. Load trained model checkpoint
2. Load input image and mask
3. Apply mask to image
4. Initialize masked regions with random noise
5. Iteratively denoise from t=1000 to t=0:
   - Predict noise at current timestep
   - Remove predicted noise
   - Keep known regions unchanged
   - Add smaller noise for next step
6. Return final inpainted image

**Key Parameters**:
- `num_denoising_steps`: Quality vs speed tradeoff (50 is good balance)
- Higher steps = better quality but slower
- Lower steps = faster but lower quality

**Denoising Algorithm (DDIM)**:
```
Start with: x_t = image * mask + noise * (1-mask)

For t = 1000 down to 0:
    ε_pred = model(x_t * mask, t, mask)
    x_0_pred = (x_t - √(1-α_t) * ε_pred) / √(α_t)
    x_0_pred = image * mask + x_0_pred * (1-mask)  # Keep known regions
    x_{t-1} = √(α_{t-1}) * x_0_pred + √(1-α_{t-1}) * ε_pred
    x_{t-1} = image * mask + x_{t-1} * (1-mask)    # Enforce known regions

Return: x_0
```

---

### Utility Components

#### 11. `utils.py` - **Helper Functions**
**Purpose**: Common utilities for training and inference

**Functions**:
- `set_seed()`: Ensures reproducible results
- `setup_cuda_device()`: Selects best available GPU
- `display_reverse()`: Visualizes denoising process

---

### Package Interface

#### 12. `__init__.py` - **Package Initialization**
**Purpose**: Defines public API and imports

**Exports**:
- Model classes: `InpaintingUNET`, `UNET`
- Training: `train_inpainting`
- Inference: `inpaint_image`
- Components: `DDPM_Scheduler`, `ResBlock`, `Attention`, etc.
- Utils: `set_seed`, `setup_cuda_device`, `display_reverse`

---

## Complete Training → Inference Flow

### Training Phase
```
1. Dataset Preparation
   ├─ Place images in data/ folder
   └─ Images automatically loaded by InpaintingDataset

2. Training Script (train_colab.py or custom script)
   ├─ Import: from main import train_inpainting
   ├─ Call: train_inpainting(
   │         dataset_path='data/',
   │         checkpoint_path='checkpoints/model',
   │         num_epochs=2000,
   │         batch_size=32
   │      )
   └─ Model trains and saves checkpoints

3. Training Loop (train_inpainting.py)
   ├─ InpaintingDataset loads images + generates masks
   ├─ InpaintingUNET initialized
   ├─ DDPM_Scheduler manages noise
   └─ For each batch:
      ├─ Add noise via scheduler
      ├─ Mask the noisy image
      ├─ Model predicts noise
      ├─ Compute loss in masked regions
      └─ Update weights

4. Checkpoint Saved
   └─ Contains: model weights, optimizer state, EMA weights
```

### Inference Phase
```
1. Load Image and Mask
   ├─ image.png (RGB image with regions to fill)
   └─ mask.png (white=keep, black=fill)

2. Inference Script (inpaint.py or custom script)
   ├─ Import: from main import inpaint_image
   └─ Call: inpaint_image(
            image_path='image.png',
            mask_path='mask.png',
            checkpoint_path='checkpoints/model',
            num_denoising_steps=50,
            save_result='output.png'
         )

3. Inference Process (inpainting_inference.py)
   ├─ Load trained InpaintingUNET
   ├─ Load and preprocess image + mask
   ├─ Apply mask to image
   ├─ Initialize masked regions with noise
   └─ Iterative denoising:
      ├─ For t = 1000 → 0:
      │  ├─ Predict noise using model
      │  ├─ Remove predicted noise
      │  └─ Keep known regions unchanged
      └─ Return final result

4. Output
   ├─ output.png (inpainted result)
   └─ output_comparison.png (side-by-side visualization)
```

---

## Key Architectural Decisions

### Why Mask Conditioning?
- Model needs to know which regions to fill vs preserve
- Prevents model from trying to "improve" known regions
- Focuses learning on filling masked areas coherently

### Why Loss Only in Masked Regions?
```python
loss = MSE(predicted_noise * (1-mask), actual_noise * (1-mask))
```
- Only train on regions we want to inpaint
- Known regions don't contribute to loss
- Model learns to fill holes, not reconstruct entire image

### Why EMA (Exponential Moving Average)?
- Averages model weights over training
- Provides more stable inference
- Reduces variance in generated results

### Why Skip Connections?
- Preserve fine details lost during downsampling
- Allow gradients to flow easily during training
- Critical for high-quality image generation

---

## Model Size and Performance

### Architecture Dimensions
- **Input**: 128×128×3 (RGB image) + 128×128×1 (mask)
- **Channels**: [64, 128, 256, 512, 512, 384]
- **Attention Layers**: At 128 and 512 channel layers
- **Total Parameters**: ~40-50M parameters

### Training Requirements
- **GPU Memory**: ~8GB VRAM minimum
- **Training Time**: ~12-24 hours for 2000 epochs (depends on dataset size)
- **Dataset Size**: Flexible (100s to 100,000s of images)

### Inference Performance
- **Single Image**: ~5-10 seconds on GPU
- **Quality**: Depends on training data similarity
- **Resolution**: 128×128 (can be upscaled to 512×512)

---

## Usage Examples

### Training
```python
from main import train_inpainting

train_inpainting(
    dataset_path='data/my_images/',
    checkpoint_path='checkpoints/my_model',
    num_epochs=2000,
    batch_size=32,
    lr=1e-4,
    image_size=128,
    save_every_n_epochs=100
)
```

### Inference
```python
from main import inpaint_image

result = inpaint_image(
    image_path='test_image.png',
    mask_path='test_mask.png',
    checkpoint_path='checkpoints/my_model',
    num_denoising_steps=50,
    save_result='result.png'
)
```

---

## Limitations and Considerations

### Current Limitations
1. **Training Data Dependency**: Model only works well on similar patterns to training data
2. **Resolution**: Fixed 128×128 training size (upscaled for display)
3. **Pattern-based**: Current checkpoint trained on geometric patterns, not natural images

### For Natural Image Inpainting
To use this model for natural images (faces, landscapes, etc.):
1. Collect dataset of natural images (1000s of images)
2. Train new model: `train_inpainting(dataset_path='natural_images/')`
3. Train for sufficient epochs (~2000+)
4. Model will learn natural image statistics

### Future Improvements
- Higher resolution training (256×256, 512×512)
- Conditional generation (text prompts, style control)
- Faster sampling (DDIM with fewer steps)
- Multi-scale architecture for better detail

---

## Summary

This diffusion-based inpainting model learns to fill missing regions by:
1. **Training**: Learning to predict and remove noise in masked areas
2. **Inference**: Iteratively denoising random noise conditioned on surrounding context

The modular architecture allows easy experimentation with different components while maintaining a clean separation between training, inference, and model architecture.
