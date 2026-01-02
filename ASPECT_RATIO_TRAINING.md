# Aspect Ratio Preservation Training Guide

## What Changed

The `kaggle-classify-videos.py` script has been modified to use **`crop_to_aspect_ratio=True`** instead of the default `False`.

### Before (Original):
```python
# Images were stretched/squashed to 320×240 regardless of aspect ratio
crop_to_aspect_ratio=False  # Default behavior
```

**Result**: 16:9 images (1280×720) were horizontally squeezed by 33% to fit 4:3 (320×240)

### After (Modified):
```python
# Images are center-cropped to preserve aspect ratio
crop_to_aspect_ratio=True
```

**Result**: Images are cropped to 4:3 aspect ratio BEFORE resizing to 320×240, preserving natural proportions

## How `crop_to_aspect_ratio=True` Works

When enabled, TensorFlow's `smart_resize` is used:

```
Input: 1280×720 (16:9)
↓
1. Calculate target aspect ratio: 320/240 = 1.33 (4:3)
2. Crop to largest centered 4:3 window: 960×720
3. Resize cropped region to: 320×240
↓
Output: 320×240 with NO distortion
```

**Visual Example:**
```
Original 1280×720 (16:9)
┌─────────────────────────┐
│     │   Hands    │      │  ← Crop sides, keep center
│     └────────────┘      │
└─────────────────────────┘
         ↓
      960×720 (4:3)
      ┌────────────┐
      │   Hands    │  ← Now 4:3 aspect ratio
      └────────────┘
         ↓
      320×240 (4:3)
      ┌──────┐
      │Hands │  ← Resized without distortion
      └──────┘
```

## Impact on Different Input Resolutions

| Input Resolution | Aspect Ratio | Action with `crop_to_aspect_ratio=True` |
|------------------|--------------|----------------------------------------|
| 640×480 | 4:3 (1.33) | ✓ No crop needed, direct resize |
| 720×480 | 3:2 (1.50) | Crop to 640×480 → resize to 320×240 |
| 1280×720 | 16:9 (1.78) | Crop to 960×720 → resize to 320×240 |
| 1920×1080 | 16:9 (1.78) | Crop to 1440×1080 → resize to 320×240 |

## Training With This Configuration

### Prerequisites
1. Download Kaggle dataset:
   ```bash
   cd dataset-kaggle
   bash get-and-preprocess-dataset.sh
   cd ..
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training Command
```bash
# Set environment variables (optional)
export HANDWASH_NN="MobileNetV2"
export HANDWASH_NUM_EPOCHS=40
export HANDWASH_NUM_FRAMES=5

# Run training
python kaggle-classify-videos.py
```

### Expected Behavior

**Training logs will show:**
- Model input shape: `(None, 5, 240, 320, 3)` (unchanged)
- Images are center-cropped during loading
- Training time: similar to before
- Final accuracy: potentially **higher** due to undistorted images

### Model Output

The trained model will be saved in the current directory with name pattern:
- `kaggle-videos_<architecture>_<timestamp>.keras`

## Updating Production System

After retraining with aspect ratio preservation:

### Step 1: Convert to ONNX
```bash
python convert-model-to-onnx.py \
  --model_path kaggle-videos_MobileNetV2_final.keras \
  --output_path kaggle-videos-final-model.onnx
```

### Step 2: Update C# Application

**IMPORTANT**: The C# inference code needs to match the training preprocessing.

**Option A: Add Center Cropping to C# (Recommended)**
Modify `OpenCvCameraService.cs` to center-crop frames before resizing:
```csharp
// Before resizing to 320×240, crop to 4:3 aspect ratio
private Mat CenterCropTo4x3(Mat input)
{
    double targetAspect = 320.0 / 240.0;  // 4:3
    double currentAspect = (double)input.Width / input.Height;

    if (Math.Abs(currentAspect - targetAspect) < 0.01)
        return input.Clone();  // Already 4:3

    int cropWidth, cropHeight, x, y;

    if (currentAspect > targetAspect)  // Too wide, crop width
    {
        cropHeight = input.Height;
        cropWidth = (int)(cropHeight * targetAspect);
        x = (input.Width - cropWidth) / 2;
        y = 0;
    }
    else  // Too tall, crop height
    {
        cropWidth = input.Width;
        cropHeight = (int)(cropWidth / targetAspect);
        x = 0;
        y = (input.Height - cropHeight) / 2;
    }

    var roi = new Rect(x, y, cropWidth, cropHeight);
    return new Mat(input, roi).Clone();
}
```

**Option B: Capture at 640×480 (Simpler)**
Change camera capture to native 4:3 resolution:
```csharp
await _cv!.StartAsync(cameraIndex: index, width: 640, height: 480, fps: 30);
```

### Step 3: Record Training Videos at 4:3

For new training data, record at 4:3 aspect ratios:
- 640×480 (VGA)
- 960×720 (HD 4:3)
- 1280×960 (HD+ 4:3)

Avoid 16:9 resolutions (720p, 1080p) to minimize cropping loss.

## Benefits of Aspect Ratio Preservation

✓ **No distortion**: Hands appear natural, not stretched/squeezed
✓ **Better accuracy**: Model learns true hand shapes and movements
✓ **Consistent with human perception**: Matches how humans see movements
✓ **Robust to different cameras**: Works well with any aspect ratio input
✓ **Easier debugging**: Visual inspection shows undistorted images

## Comparison: Before vs After

### Training on Original Dataset (720×480)

| Configuration | Processing | Distortion |
|--------------|-----------|------------|
| **Before** (`crop_to_aspect_ratio=False`) | 720×480 → stretched to 320×240 | ~12% horizontal squeeze |
| **After** (`crop_to_aspect_ratio=True`) | 720×480 → crop to 640×480 → resize to 320×240 | ✓ No distortion |

### Training on HD Video (1280×720)

| Configuration | Processing | Distortion |
|--------------|-----------|------------|
| **Before** (`crop_to_aspect_ratio=False`) | 1280×720 → stretched to 320×240 | ~33% horizontal squeeze |
| **After** (`crop_to_aspect_ratio=True`) | 1280×720 → crop to 960×720 → resize to 320×240 | ✓ No distortion |

## Troubleshooting

### Issue: "Model accuracy decreased after retraining"

**Possible causes:**
1. Training data contains mostly non-4:3 aspect ratios → significant cropping removes important hand movement areas
2. Need more epochs for convergence
3. Learning rate needs adjustment

**Solutions:**
- Use 4:3 source videos (640×480, 960×720)
- Increase `HANDWASH_NUM_EPOCHS=60`
- Check that cropping preserves hand regions (visualize training data)

### Issue: "Out of memory during training"

**Solution:**
```bash
export HANDWASH_BATCH_SIZE=16  # Reduce from default 32
```

### Issue: "C# inference results don't match Python"

**Cause:** Preprocessing mismatch

**Solution:** Ensure C# applies same center-cropping before resize, or use 640×480 camera capture.

## Reverting Changes

To revert to original distortion-based training:

```bash
cd mga-heart-handwash
git checkout kaggle-classify-videos.py
```

Or manually change all three instances to:
```python
crop_to_aspect_ratio=False
```

## Additional Notes

- This modification only affects the **training pipeline**
- Inference scripts (`inference_video.py`) should use same `crop_to_aspect_ratio=True`
- For best results, retrain from scratch rather than fine-tuning
- Consider retraining all three architectures (frames, videos, merged-network) for comparison

---

**Created**: 2026-01-02
**Modified for**: Axioma.HandWash project
**Reference**: TensorFlow `smart_resize` documentation
