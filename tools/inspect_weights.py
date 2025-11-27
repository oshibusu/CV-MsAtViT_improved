import h5py
import numpy as np
import argparse

def inspect_weights(weights_path):
    print(f"Inspecting weights from: {weights_path}")
    try:
        with h5py.File(weights_path, 'r') as f:
            def print_attrs(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"\nDataset: {name}")
                    print(f"  Shape: {obj.shape}")
                    print(f"  Dtype: {obj.dtype}")
                    data = obj[:]
                    data = obj[:]
                    if np.iscomplexobj(data):
                        print(f"  Complex: Yes")
                        print(f"  Max Real: {np.max(np.real(data))}")
                        print(f"  Max Imag: {np.max(np.imag(data))}")
                    else:
                        print(f"  Complex: No")
                        print(f"  Min: {np.min(data)}")
                        print(f"  Max: {np.max(data)}")
                        print(f"  Mean: {np.mean(data)}")
                        if "kernel_i" in name or "bias_i" in name:
                            print(f"  Non-zero count: {np.count_nonzero(data)}")

            f.visititems(print_attrs)
    except Exception as e:
        print(f"Error reading H5 file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_path", help="Path to .h5 weights file")
    args = parser.parse_args()
    inspect_weights(args.weights_path)
