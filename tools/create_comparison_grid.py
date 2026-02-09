import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io as sio
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

# Add generic tools path
sys.path.append(os.path.dirname(__file__))
# Import dataset info from visualizer
from visualize_prediction import get_dataset_info, create_cmap

# Add root path for Load_Data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Load_Data import load_data

def load_mat_prediction(mat_path):
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"{mat_path} not found")
    data = sio.loadmat(mat_path)
    for k in data.keys():
        if not k.startswith("__"):
            return data[k]
    raise ValueError("No valid key found in mat file")

def main():
    parser = argparse.ArgumentParser(description="Generate 2x2 comparison grid (Pauli, GT, Default, Proposed)")
    parser.add_argument("--dataset", required=True, help="Dataset name (FL_T, SF, Baltrum_S_FP1)")
    parser.add_argument("--pauli", required=True, help="Path to Pauli RGB image")
    parser.add_argument("--default_mat", required=True, help="Path to Default prediction .mat")
    parser.add_argument("--proposed_mat", required=True, help="Path to Proposed prediction .mat")
    parser.add_argument("--output", required=True, help="Output path for the grid image (auto generates _1x3.png as well)")
    
    args = parser.parse_args()
    
    dataset_name = args.dataset
    if dataset_name == "Baltrum": # Handle alias if needed
         dataset_name = "Baltrum_S_FP1"

    # Get colors and names
    colors, class_names = get_dataset_info(dataset_name)
    cmap = create_cmap(colors)
    
    # Load Data
    print(f"Loading Pauli: {args.pauli}")
    pauli_img = mpimg.imread(args.pauli)
    
    print(f"Loading GT for {dataset_name}...")
    # Map dataset name for load_data if needed
    load_data_name = dataset_name
    if "Baltrum" in dataset_name: 
        load_data_name = "Baltrum_S_FP1" # Load_Data requires Baltrum_Band_FP format
    elif "SF" in dataset_name:
        load_data_name = "SF"
        
    _, gt_map = load_data(load_data_name)
    
    print(f"Loading Default Mat: {args.default_mat}")
    def_map = load_mat_prediction(args.default_mat)
    
    print(f"Loading Proposed Mat: {args.proposed_mat}")
    prop_map = load_mat_prediction(args.proposed_mat)
    
    # Dataset specific modifications
    if "Baltrum" in dataset_name:
        crop_top = 300
        crop_bottom = 400
        print(f"Applying cropping for Baltrum: Removing top {crop_top}, bottom {crop_bottom} pixels")
        
        # Helper to crop
        def crop_img(img, top_px, bot_px):
            if img.shape[0] > (top_px + bot_px):
                return img[top_px:-bot_px, ...]
            return img

        pauli_img = crop_img(pauli_img, crop_top, crop_bottom)
        gt_map = crop_img(gt_map, crop_top, crop_bottom)
        def_map = crop_img(def_map, crop_top, crop_bottom)
        prop_map = crop_img(prop_map, crop_top, crop_bottom)
        
        print(f"New shapes -> Pauli: {pauli_img.shape}, GT: {gt_map.shape}")
    
    # --- Plotting 2x2 Grid ---
    
    # 2x2 Layout: Pauli, GT, Previous, Proposed
    layout_config = {
        'FL_T': {'figsize': (12, 10), 'bottom': 0.12, 'wspace': 0.05, 'hspace': 0.02, 'legend_y': 0.01},
        'SF':   {'figsize': (11, 10), 'bottom': 0.10, 'wspace': 0.05, 'hspace': 0.02, 'legend_y': 0.0},
        'Baltrum_S_FP1': {'figsize': (10, 16), 'bottom': 0.06, 'wspace': 0.1, 'hspace': 0.02, 'legend_y': 0.0, 'legend_ncols': 4},
        'default': {'figsize': (12, 12), 'bottom': 0.10, 'wspace': 0.05, 'hspace': 0.02, 'legend_y': 0.0}
    }
    
    config = layout_config.get(dataset_name, layout_config['default'])
    
    fig, axes = plt.subplots(2, 2, figsize=config['figsize'], constrained_layout=False)
    plt.subplots_adjust(bottom=config['bottom'], wspace=config['wspace'], hspace=config['hspace'])
    
    def clean_ax(ax, title):
        ax.set_title(title, fontsize=24) # Increased font size
        ax.axis('off')
    
    # TL: Pauli
    axes[0, 0].imshow(pauli_img, aspect='equal')
    clean_ax(axes[0, 0], "Pauli RGB")
    
    # Prepare Normalization
    num_classes = len(class_names)
    bounds = np.arange(num_classes + 1) - 0.5
    norm = BoundaryNorm(bounds, num_classes)
    
    # TR: GT
    axes[0, 1].imshow(gt_map, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
    clean_ax(axes[0, 1], "Ground Truth")
    
    # BL: Previous (was Default)
    axes[1, 0].imshow(def_map, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
    clean_ax(axes[1, 0], "Previous Prediction")
    
    # BR: Proposed
    axes[1, 1].imshow(prop_map, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
    clean_ax(axes[1, 1], "Proposed Prediction")
    
    # Legend
    patches = [mpatches.Patch(color=cmap(i), label=name) 
               for i, name in enumerate(class_names) if i > 0]
    
    calculated_ncols = 4 if len(patches) < 8 else 6
    if len(patches) > 12: calculated_ncols = 5
    
    ncols = config.get('legend_ncols', calculated_ncols)
    
    fig.legend(handles=patches, loc='lower center', 
               bbox_to_anchor=(0.5, config['legend_y']), ncol=ncols, 
               fontsize=18, frameon=False) # Increased font size
    
    output_2x2 = args.output
    print(f"Saving 2x2 grid to {output_2x2}")
    plt.savefig(output_2x2, dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plotting 1x3 Grid ---
    
    # 1x3 Layout: GT, Previous, Proposed (No Pauli)
    
    # Adjust config for 1x3
    if "Baltrum" in dataset_name:
         figsize_1x3 = (20, 10)
         bottom_1x3 = 0.15
         wspace_1x3 = 0.05
    else:
         figsize_1x3 = (18, 6)
         bottom_1x3 = 0.15
         wspace_1x3 = 0.05 # Increased wspace for others to match style if needed, but 0.05 is fine

    fig, axes = plt.subplots(1, 3, figsize=figsize_1x3)
    plt.subplots_adjust(bottom=bottom_1x3, wspace=wspace_1x3)

    # 1. GT
    axes[0].imshow(gt_map, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
    clean_ax(axes[0], "Ground Truth")

    # 2. Previous
    axes[1].imshow(def_map, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
    clean_ax(axes[1], "Previous Prediction")

    # 3. Proposed
    axes[2].imshow(prop_map, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
    clean_ax(axes[2], "Proposed Prediction")

    # Legend (Reuse patches)
    calculated_ncols_1x3 = 6
    if len(patches) > 12: calculated_ncols_1x3 = 5
    
    # Place legend slightly higher up if needed, or same logic
    fig.legend(handles=patches, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=calculated_ncols_1x3, 
               fontsize=18, frameon=False)

    # Output filename for 1x3
    base, ext = os.path.splitext(args.output)
    output_1x3 = f"{base}_1x3{ext}"
    
    print(f"Saving 1x3 grid to {output_1x3}")
    plt.savefig(output_1x3, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
