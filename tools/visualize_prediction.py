import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import argparse

# Add root directory to path to import Load_Data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from Load_Data import load_data
except ImportError:
    print("Warning: Could not import Load_Data. GT visualization will be skipped.")
    load_data = None

# --- Color Definitions ---

# Baltrum 12 classes (from visualize_baltrum.py)
COLORS_BALTRUM = [
    (0, 0, 0), (0, 205, 220), (0, 16, 169), (194, 116, 0), (0, 86, 0),
    (208, 185, 0), (195, 0, 160), (135, 133, 131), (175, 192, 132),
    (255, 125, 0), (247, 164, 233), (239, 217, 0), (200, 0, 0)
]

# Flevoland 15 classes (Colors reordered per user request to match Display_GT)
# 0: Background (Black)
COLORS_FL_T = [
    (0, 0, 0),              # 0: Background
    (0, 0, 255),            # 1: Water (Blue)
    (0, 100, 0),            # 2: Forest (Dark Green)
    (0, 255, 255),          # 3: Lucerne (Cyan)
    (0, 255, 0),            # 4: Grass (Bright Green)
    (255, 165, 0),          # 5: Rapeseed (Orange)
    (255, 0, 255),          # 6: Beet (Magenta)
    (255, 255, 0),          # 7: Potatoes (Yellow)
    (128, 0, 128),          # 8: Peas (Purple)
    (255, 0, 0),            # 9: Stem Beans (Red)
    (139, 69, 19),          # 10: Bare Soil (Brown)
    (255, 192, 203),        # 11: Wheat (Pink)
    (221, 160, 221),        # 12: Wheat 2 (Light Purple)
    (144, 238, 144),        # 13: Wheat 3 (Light Green)
    (139, 0, 0),            # 14: Barley (Dark Red)
    (245, 245, 220)         # 15: Buildings (Beige)
]

# San Francisco 5 classes (Updated per user corrected instruction and colors)
# 0: Background (Black)
# 1: Bare Soil (Cyan)
# 2: Mountain (Yellow)
# 3: Water (Blue)
# 4: Urban (Red)
# 5: Vegetation (Green)
COLORS_SF = [
    (0, 0, 0),              # 0: Background
    (0, 255, 255),          # 1: Bare Soil (Cyan)
    (255, 255, 0),          # 2: Mountain (Yellow)
    (0, 0, 255),            # 3: Water (Blue)
    (255, 0, 0),            # 4: Urban (Red)
    (0, 128, 0)             # 5: Vegetation (Green)
]

# --- Class Names ---

CLASS_NAMES_BALTRUM = [
    "Background", "Tidal flat", "Water", "Coastal shrub", "Dense, high vegetation",
    "White dune", "Peat bog", "Grey dunes", "Couch grass", "Upper salt marsh",
    "Lower salt marsh", "Sand", "Settlement"
]

CLASS_NAMES_FL_T = [
    "Background", "Water", "Forest", "Lucerne", "Grass", "Rapeseed", "Beet", "Potatoes",
    "Peas", "Stem Beans", "Bare Soil", "Wheat", "Wheat 2", "Wheat 3", "Barley", "Buildings"
]

CLASS_NAMES_SF = [
    "Background", "Bare Soil", "Mountain", "Water", "Urban", "Vegetation"
]

def get_dataset_info(dataset_name):
    if "Baltrum" in dataset_name:
        return COLORS_BALTRUM, CLASS_NAMES_BALTRUM
    elif "FL_T" in dataset_name:
        return COLORS_FL_T, CLASS_NAMES_FL_T
    elif "SF" in dataset_name:
        return COLORS_SF, CLASS_NAMES_SF
    else:
        # Fallback to tab20 for unknown
        cmap = plt.get_cmap('tab20')
        colors = [(0, 0, 0)] + [cmap(i)[:3] for i in range(19)]
        names = ["Background"] + [f"Class {i}" for i in range(1, 20)]
        # Scale back to 0-255 inputs if using tuple logic, but here we process mostly normalized
        # To keep consistency, let's assume 0-255 RGB tuples for custom lists and normalize later
        # But tab20 returns 0-1.
        return colors, names

def create_cmap(colors_rgb):
    # Check if first element is float or int
    if isinstance(colors_rgb[0][0], float) and colors_rgb[0][0] <= 1.0:
        # Already normalized (e.g. from fallback)
        colors_norm = colors_rgb
    else:
        # Normalize 0-255 to 0-1
        colors_norm = [tuple(c / 255.0 for c in rgb) for rgb in colors_rgb]
    return ListedColormap(colors_norm)

def visualize_array(pred_map, out_path, cmap, class_names, title):
    h, w = pred_map.shape
    print(f"  Map shape: {h} x {w}")
    
    if h == 0 or w == 0:
        print("Error: Map has 0 dimension")
        return
        
    aspect_ratio = w / h
    fig_w = 12
    fig_h = fig_w / aspect_ratio
    
    if fig_h > 20: 
        fig_h = 20
        fig_w = fig_h * aspect_ratio
    
    fig_h += 1 
    
    plt.figure(figsize=(fig_w, fig_h))
    
    # Ensure correct mapping
    num_classes = len(class_names)
    bounds = np.arange(num_classes + 1) - 0.5
    norm = BoundaryNorm(bounds, num_classes)

    im = plt.imshow(pred_map, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
    plt.title(title)
    plt.axis('off')

    # Legend
    patches = [mpatches.Patch(color=cmap(i), label=name) 
               for i, name in enumerate(class_names) if i > 0] # Skip background
    
    plt.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., 
               handlelength=1.0, handleheight=1.0, frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {out_path}")

def visualize_mat(mat_path, out_path, cmap, class_names, title):
    if not os.path.exists(mat_path):
        print(f"File not found: {mat_path}")
        return

    print(f"Processing {mat_path}...")
    try:
        data = sio.loadmat(mat_path)
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return

    key = None
    for k in data.keys():
        if not k.startswith("__"):
            key = k
            break
    
    if key is None:
        print("No valid data found in .mat file")
        return

    pred_map = data[key]
    visualize_array(pred_map, out_path, cmap, class_names, title)

def main():
    plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = ["Times New Roman"]

    parser = argparse.ArgumentParser(description="Visualize prediction .mat files.")
    parser.add_argument("--dataset", required=True, help="Dataset name (FL_T, SF, Baltrum_S_FP1)")
    parser.add_argument("files", nargs="+", help="List of .mat files to visualize")
    args = parser.parse_args()

    dataset_name = args.dataset
    colors, class_names = get_dataset_info(dataset_name)
    cmap = create_cmap(colors)

    # Visualize GT
    if load_data:
        print(f"Loading Ground Truth for {dataset_name}...")
        try:
            # load_data returns T, labels. We only need labels.
            _, gt = load_data(dataset_name)
            gt_out_path = f"results/{dataset_name}_GT_vis.png"
            visualize_array(gt, gt_out_path, cmap, class_names, f"Ground Truth ({dataset_name})")
        except Exception as e:
            print(f"Error loading GT for {dataset_name}: {e}")
            # import traceback
            # traceback.print_exc()

    for f in args.files:
        basename = os.path.splitext(os.path.basename(f))[0]
        out_path = os.path.join(os.path.dirname(f), f"{basename}_vis.png")
        title = f"{basename} ({dataset_name})"
        
        visualize_mat(f, out_path, cmap, class_names, title)

if __name__ == "__main__":
    main()
