# How to Get Sharp Face Swap Results

## Problem
Your current model produces blurry results because it's trained on 64x64 images then upscaled to 512x512.

## Solution: Train at Higher Resolution

### Changes Made:
1. ✅ Fixed `train_colab.py` - corrected `num_time_steps` from 1500 to 1000
2. ✅ Updated `train.py` - changed training resolution from 64x64 to **128x128**
3. ✅ Updated `face_swapper.py` - changed inference resolution to **128x128**

### What This Means:
- **Before**: 64x64 → upscale 8x → 512x512 (very blurry)
- **After**: 128x128 → upscale 4x → 512x512 (2x sharper!)

## Training on Google Colab

### Step 1: Delete Old Checkpoint
```python
# In Colab, run:
!rm checkpoints/ddpm_faceswap_checkpoint
```

### Step 2: Start New Training
```python
!python train_colab.py
```

### Training Settings:
- **Resolution**: 128x128 (2x improvement)
- **Epochs**: 2500 (should give good results)
- **Batch Size**: 8 (reduce to 4 if you get OOM errors)
- **Dataset**: 400 image pairs
- **Time**: Approximately 3-6 hours on Colab GPU

### Important Notes:
- ⚠️ **You MUST retrain** - the old 64x64 checkpoint won't work with 128x128
- 📊 Training at 128x128 uses ~2x more memory
- ⏱️ Training takes ~1.5x longer than 64x64
- 💾 Checkpoint file will be ~2x larger

## Expected Results:
- **Much sharper** facial features
- **Better defined** edges and details
- **Less blur** in the final output
- Still some blur due to diffusion smoothing, but significantly improved

## For Even Sharper Results (Advanced):

### Option 1: Train at 256x256
Change `image_size=128` to `image_size=256` in `train.py`
- **Pros**: 4x sharper than current
- **Cons**: Requires A100 GPU, takes 6-8 hours, needs more epochs

### Option 2: Increase Training Epochs
Change `num_epochs: 2500` to `num_epochs: 5000` in `train_colab.py`
- **Pros**: Better quality with same resolution
- **Cons**: Takes 2x longer

### Option 3: Add Post-Processing (No Retraining!)
The current code already includes:
- Unsharp masking for edge enhancement
- Bicubic upscaling with antialiasing
- Smart blending to preserve source features

To increase sharpening strength, edit `face_swapper.py` line ~213:
```python
sharpened = face_swapped_result + 0.5 * (face_swapped_result - blurred)  # Changed from 0.3 to 0.5
```

## Quick Start:

### On Google Colab:
```bash
# 1. Clone repo (if not already done)
!git clone https://github.com/csimpson62003/CPSC_495.git
%cd CPSC_495

# 2. Install requirements
!pip install -r requirements.txt

# 3. Delete old checkpoint
!rm -f checkpoints/ddpm_faceswap_checkpoint

# 4. Start training at 128x128
!python train_colab.py
```

### After Training:
```python
# Use your trained model
!python use_my_images.py
```

## Troubleshooting:

### "CUDA out of memory"
Reduce batch size in `train_colab.py`:
```python
'batch_size': 4,  # Reduced from 8
```

### "Index out of bounds"
Make sure `num_time_steps: 1000` (not 1500) in `train_colab.py`

### Still blurry?
1. Train longer (increase `num_epochs`)
2. Increase sharpening (change 0.3 to 0.5)
3. Train at 256x256 (requires more powerful GPU)

## Summary:
**You need to retrain at 128x128 for sharper results!** The changes are already made in your code - just delete the old checkpoint and run `train_colab.py` again.
