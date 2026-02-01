import numpy as np
from pathlib import Path

def count_gt_pixels(label_path, shape):
    try:
        # Data type 1 is byte (uint8)
        data = np.fromfile(label_path, dtype=np.uint8)
        # Reshape to ensure it matches expected dimensions
        data = data.reshape(shape)
        # Count non-zero pixels
        return np.count_nonzero(data)
    except Exception as e:
        print(f"Error reading {label_path}: {e}")
        return 0

base_dir = Path('/Users/shibuyayuunin/dev/CV-MsAtViT_original/Datasets/Baltrum/dataset/Pol-InSAR-Island_updated/label')

# FP1 Dimensions: 3616 lines, 2502 samples
fp1_shape = (3616, 2502)
fp1_path = base_dir / 'FP1'
fp1_train_count = count_gt_pixels(fp1_path / 'label_train.bin', fp1_shape)
fp1_test_count = count_gt_pixels(fp1_path / 'label_test.bin', fp1_shape)
fp1_total = fp1_train_count + fp1_test_count

# FP2 Dimensions: 3616 lines, 2540 samples
fp2_shape = (3616, 2540)
fp2_path = base_dir / 'FP2'
fp2_train_count = count_gt_pixels(fp2_path / 'label_train.bin', fp2_shape)
fp2_test_count = count_gt_pixels(fp2_path / 'label_test.bin', fp2_shape)
fp2_total = fp2_train_count + fp2_test_count

print(f"FP1 (2502x3616):")
print(f"  Train GT pixels: {fp1_train_count}")
print(f"  Test GT pixels:  {fp1_test_count}")
print(f"  Total GT pixels: {fp1_total}")

print(f"\nFP2 (2540x3616):")
print(f"  Train GT pixels: {fp2_train_count}")
print(f"  Test GT pixels:  {fp2_test_count}")
print(f"  Total GT pixels: {fp2_total}")

print(f"\nGrand Total (FP1 + FP2): {fp1_total + fp2_total}")
