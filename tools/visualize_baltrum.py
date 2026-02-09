import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

# User provided colors for Baltrum 12 classes
# Class 0 is background (Black)
# Classes 1-12 as provided
COLORS_RGB = [
    (0, 0, 0),          # 0: Background
    (0, 205, 220),      # 1: tidal flat
    (0, 16, 169),       # 2: water
    (194, 116, 0),      # 3: coastal shrub
    (0, 86, 0),         # 4: dense, high vegetation
    (208, 185, 0),      # 5: white dune
    (195, 0, 160),      # 6: peat bog
    (135, 133, 131),    # 7: grey dunes
    (175, 192, 132),    # 8: couch grass
    (255, 125, 0),      # 9: upper salt marsh
    (247, 164, 233),    # 10: lower salt marsh
    (239, 217, 0),      # 11: sand
    (200, 0, 0)         # 12: settlement
]

CLASS_NAMES = [
    "Background",
    "Tidal flat",
    "Water",
    "Coastal shrub",
    "Dense, high vegetation",
    "White dune",
    "Peat bog",
    "Grey dunes",
    "Couch grass",
    "Upper salt marsh",
    "Lower salt marsh",
    "Sand",
    "Settlement"
]

def create_baltrum_cmap():
    # Convert 0-255 RGB to 0-1
    colors_norm = [tuple(c / 255.0 for c in rgb) for rgb in COLORS_RGB]
    return ListedColormap(colors_norm)

# Add root directory to path to import Load_Data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from Load_Data import load_data
except ImportError:
    print("Warning: Could not import Load_Data. GT visualization will be skipped.")
    load_data = None

def visualize_array(pred_map, out_path, cmap, title):
    h, w = pred_map.shape
    print(f"  Map shape: {h} x {w}")
    
    # Calculate appropriate figsize keeping aspect ratio
    # Base width 12 inches
    # Aspect ratio = w / h
    if h == 0:
        print("Error: Map has 0 height")
        return
        
    aspect_ratio = w / h
    fig_w = 12
    fig_h = fig_w / aspect_ratio
    
    # Cap height/width to reasonable limits to avoid excessive size
    if fig_h > 20: 
        fig_h = 20
        fig_w = fig_h * aspect_ratio
    
    # Add room for legend and title
    fig_h += 1 # extra inch
    
    # Plotting
    plt.figure(figsize=(fig_w, fig_h))
    
    # We use BoundaryNorm to ensure exact mapping of integer labels to colors
    # Classes are 0 to 12. Bounds should be 0, 1, ..., 13 
    bounds = np.arange(len(COLORS_RGB) + 1) - 0.5
    norm = BoundaryNorm(bounds, len(COLORS_RGB))

    im = plt.imshow(pred_map, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
    plt.title(title)
    plt.axis('off')

    # Create legend
    patches = [mpatches.Patch(color=cmap(i), label=name) 
               for i, name in enumerate(CLASS_NAMES) if i > 0] # Skip background
    # Move legend outside
    plt.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., 
               handlelength=1.0, handleheight=1.0, frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {out_path} (Size: {fig_w:.1f}x{fig_h:.1f})")

def visualize_mat(mat_path, out_path, cmap, title):
    if not os.path.exists(mat_path):
        print(f"File not found: {mat_path}")
        return

    print(f"Processing {mat_path}...")
    try:
        data = sio.loadmat(mat_path)
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return

    # Find key (ignoring __*)
    key = None
    for k in data.keys():
        if not k.startswith("__"):
            key = k
            break
    
    if key is None:
        print("No valid data found in .mat file")
        return

    pred_map = data[key]
    visualize_array(pred_map, out_path, cmap, title)

def main():
    # Set font to Times New Roman
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    
    cmap = create_baltrum_cmap()
    
    # Visualize GT if possible
    if load_data:
        print("Loading Ground Truth for Baltrum_S_FP1...")
        try:
            _, gt = load_data('Baltrum_S_FP1')
            visualize_array(gt, "results/Baltrum_GT_vis.png", cmap, "Ground Truth (Baltrum_S_FP1)")
        except Exception as e:
            print(f"Error loading GT: {e}")
            import traceback
            traceback.print_exc()

    files = [
        ("results/CV_MsAtViT_default.mat", "results/CV_MsAtViT_default_vis.png", "CV-MsAtViT Default"),
        ("results/CV_MsAtViT_propose_Baltrum_S_FP1.mat", "results/CV_MsAtViT_proposed_vis.png", "CV-MsAtViT Proposed (Baltrum_S_FP1)"),
         # Also visualizing the Full map for default if exists
        ("results/CV_MsAtViT_default_Full.mat", "results/CV_MsAtViT_default_Full_vis.png", "CV-MsAtViT Default (Full Map)")
    ]

    for mp, op, title in files:
        visualize_mat(mp, op, cmap, title)

if __name__ == "__main__":
    main()
