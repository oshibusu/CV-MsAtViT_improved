import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import sys

# Add root directory to path to import Load_Data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Load_Data import load_data

def save_mask(mask, filename, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

def inspect_dataset(dataset_name):
    print(f"Inspecting {dataset_name}...")
    try:
        if dataset_name == 'SF':
             # Use direct load for SF to avoid any processing in load_data if any
             # But load_data is safe. Let's use load_data to match pipeline.
             _, gt = load_data('SF')
             # SF gt is usually named 'SanFrancisco_gt.mat'
        else:
             _, gt = load_data(dataset_name)
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return

    unique_vals = np.unique(gt)
    print(f"Unique values: {unique_vals}")

    out_dir = f"results/inspect_{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)

    for val in unique_vals:
        if val == 0: continue # Skip background
        
        mask = (gt == val)
        count = np.sum(mask)
        print(f"Value {val}: {count} pixels")
        
        save_mask(mask, f"{out_dir}/mask_{val}.png", f"Value {val} ({count} px)")

def main():
    inspect_dataset('FL_T')
    inspect_dataset('SF')

if __name__ == "__main__":
    main()
