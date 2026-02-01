import numpy as np
from pathlib import Path

def check_overlap(label_dir, shape):
    train_path = label_dir / 'label_train.bin'
    test_path = label_dir / 'label_test.bin'
    
    try:
        train_gt = np.fromfile(train_path, dtype=np.uint8).reshape(shape)
        test_gt = np.fromfile(test_path, dtype=np.uint8).reshape(shape)
        
        # Check if any pixel has non-zero value in BOTH train and test
        overlap = (train_gt > 0) & (test_gt > 0)
        overlap_count = np.count_nonzero(overlap)
        
        print(f"Checking {label_dir}...")
        if overlap_count == 0:
            print("  Result: Train and Test GT are completely disjoint (no overlap).")
        else:
            print(f"  Result: Found {overlap_count} overlapping pixels!")
            
    except Exception as e:
        print(f"Error: {e}")

base_dir = Path('/Users/shibuyayuunin/dev/CV-MsAtViT_original/Datasets/Baltrum/dataset/Pol-InSAR-Island_updated/label')

# FP1
check_overlap(base_dir / 'FP1', (3616, 2502))

# FP2
check_overlap(base_dir / 'FP2', (3616, 2540))
