import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from spectral.io import envi
from pathlib import Path

# User requested solution using Rasterio, but in this environment we have 'spectral' installed 
# and verified working with these specific ENVI files (Load_Data.py uses it).
# This script uses 'spectral' for robustness but follows the logic requested.

def load_envi_band(bin_path):
    """
    Load a single band from an ENVI binary file using spectral.io.envi.
    Assumes .hdr file exists at bin_path + '.hdr' or similar.
    """
    bin_path = Path(bin_path)
    # Spectral expects the header file path usually, or finds it auto
    hdr_path = bin_path.with_suffix('.bin.hdr')
    if not hdr_path.exists():
        hdr_path = bin_path.with_name(bin_path.name + '.hdr')
    
    if not hdr_path.exists():
        print(f"Error: Header not found for {bin_path}")
        return None

    # Load
    # envi.open returns a SpyFile object. read_band(0) reads the first band.
    img = envi.open(hdr_path, bin_path)
    band = img.read_band(0)
    return band

def create_pauli_rgb(t11, t22, t33):
    """
    Create Pauli RGB image from T11, T22, T33.
    Formulation:
    T11 = <|HH+VV|^2> / 2  (Surface) -> Blue
    T22 = <|HH-VV|^2> / 2  (Double Bounce) -> Red
    T33 = <|2HV|^2> / 2 = 2<|HV|^2> (Volume) -> Green
    
    Note: T elements are usually 1/2 scaled in standard definition, but relative power matters.
    User Spec:
    Red = |HH-VV| approx sqrt(T22) * sqrt(2)?
    Green = 4|HV| approx sqrt(T33) * sqrt(2)?
    Blue = |HH+VV| approx sqrt(T11) * sqrt(2)?
    
    Actually, color balance is key.
    We will use the amplitudes (sqrt of T elements) for visualization as it handles dynamic range better,
    or use Log scale. Linear power is often too dark.
    Let's use Square Root (Amplitude) which is standard for Pauli "Image".
    
    Red = sqrt(T22)
    Green = sqrt(T33)
    Blue = sqrt(T11)
    """
    
    # Amplitudes
    r = np.sqrt(np.abs(t22))
    g = np.sqrt(np.abs(t33))
    b = np.sqrt(np.abs(t11))
    
    img = np.stack([r, g, b], axis=-1)
    
    return img

def normalize_percentile(img, percentile=98):
    """
    Clip outliers and normalize to 0-1 for each channel.
    """
    norm_img = np.zeros_like(img)
    for i in range(3):
        band = img[:, :, i]
        p_val = np.percentile(band, percentile)
        print(f"  Channel {i} clipping at {p_val:.4f}")
        band = np.clip(band, 0, p_val)
        band = band / p_val
        norm_img[:, :, i] = band
    return norm_img

def main():
    # Base path
    base_dir = Path("Datasets/Baltrum/dataset/Pol-InSAR-Island_updated/data/FP1/S/T6")
    
    print(f"Reading data from {base_dir}...")
    
    # Load diagonal elements of T6 matrix
    # real parts are enough for diagonal T11, T22, T33 as they are real by definition
    # But usually provided as .bin. (Load_Data checks T11.bin)
    
    t11 = load_envi_band(base_dir / "T11.bin")
    t22 = load_envi_band(base_dir / "T22.bin")
    t33 = load_envi_band(base_dir / "T33.bin")
    
    if t11 is None or t22 is None or t33 is None:
        print("Failed to load bands.")
        return

    print("Generating Pauli RGB...")
    pauli_img = create_pauli_rgb(t11, t22, t33)
    
    print("Normalizing...")
    pauli_vis = normalize_percentile(pauli_img)
    
    # Save
    out_path = "results/FP1_S_Pauli_RGB.png"
    plt.imsave(out_path, pauli_vis)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
