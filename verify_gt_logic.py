import sys
from unittest.mock import MagicMock
import numpy as np

# Mock tensorflow and keras to avoid import errors/crashes
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.keras'] = MagicMock()
sys.modules['keras'] = MagicMock()

# Now import SAR_utils
# modifying sys.path might be needed if SAR_utils is not in current dir
import SAR_utils

def test_createImageCubes_returns_coords():
    print("Testing createImageCubes...")
    # Create dummy data: 10x10 image, 1 channel
    # GT with some labels
    data = np.zeros((10, 10, 1), dtype=np.complex64)
    gt = np.zeros((10, 10), dtype=np.uint8)
    
    # Set some GT pixels
    gt[2, 3] = 1 # Class 1
    gt[5, 5] = 2 # Class 2
    
    # Expected coords: (2, 3) and (5, 5) if window_size is small
    # Note: createImageCubes uses padWithZeros. 
    # The coords returned should be (r, c) in ORIGINAL image space.
    
    X_coh, y, coords = SAR_utils.createImageCubes(data, gt, windowSize=3, removeZeroLabels=True)
    
    print(f"Returned {len(coords)} samples.")
    print(f"Coords shape: {coords.shape}")
    print(f"Coords: \n{coords}")
    
    expected = np.array([[2, 3], [5, 5]])
    # Sort to compare if needed, but createImageCubes processes typically in order
    
    if np.array_equal(coords, expected):
        print("SUCCESS: Coords match expected GT locations.")
    else:
        print("FAILURE: Coords do not match.")

if __name__ == "__main__":
    test_createImageCubes_returns_coords()
