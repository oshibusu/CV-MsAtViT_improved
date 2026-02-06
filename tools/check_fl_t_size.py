import sys
import os
import types
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock cvnn to avoid dependency error
m = types.ModuleType("cvnn")
m.layers = types.ModuleType("layers")
sys.modules["cvnn"] = m
sys.modules["cvnn.layers"] = m.layers

from Load_Data import load_data
import numpy as np
from SAR_utils import get_gt_coords

dataset = "FL_T"
print(f"Loading {dataset}...")
data, gt = load_data(dataset)
print(f"Data shape: {data.shape}")
print(f"GT shape: {gt.shape}")

print("Getting valid coordinates...")
coords = get_gt_coords(gt, removeZeroLabels=True)
print(f"Number of valid pixels (GT > 0): {len(coords)}")

# Check memory for patches
window_size = 15
channels = data.shape[2]
# complex64 = 8 bytes per element
patch_bytes = window_size * window_size * channels * 8
total_memory = len(coords) * patch_bytes / (1024**3)

print(f"Estimated memory for X_train (all patches): {total_memory:.2f} GB")
