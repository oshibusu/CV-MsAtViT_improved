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
    parser.add_argument("--output", required=True, help="Output path for the grid image")
    
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
    
    # Plotting
    # 2x2 Grid using constrained_layout for better spacing
    # Use subplot_mosaic for semantic layout? Or just standard subplots.
    
    # Dataset-specific layout configuration
    # Increased bottom margin to separate legend from images
    layout_config = {
        'FL_T': {'figsize': (12, 10), 'bottom': 0.12, 'wspace': 0.05, 'hspace': 0.02, 'legend_y': 0.01},
        'SF':   {'figsize': (11, 10), 'bottom': 0.10, 'wspace': 0.05, 'hspace': 0.02, 'legend_y': 0.0},
        'Baltrum_S_FP1': {'figsize': (10, 16), 'bottom': 0.06, 'wspace': 0.1, 'hspace': 0.02, 'legend_y': 0.0, 'legend_ncols': 4},
        'default': {'figsize': (12, 12), 'bottom': 0.10, 'wspace': 0.05, 'hspace': 0.02, 'legend_y': 0.0}
    }
    
    config = layout_config.get(dataset_name, layout_config['default'])
    # Fallback for "Baltrum" alias if exact match fails but key exists via other means, or just rely on 'default' if unknown.
    # Since we normalize dataset_name earlier, accurate keys are important.
    
    fig, axes = plt.subplots(2, 2, figsize=config['figsize'], constrained_layout=False)
    # Apply specific spacing
    plt.subplots_adjust(bottom=config['bottom'], wspace=config['wspace'], hspace=config['hspace'])
    
    # Helper to clean axis
    def clean_ax(ax, title):
        ax.set_title(title, fontsize=14)
        ax.axis('off')
    
    # TL: Pauli
    axes[0, 0].imshow(pauli_img, aspect='equal')
    clean_ax(axes[0, 0], "Pauli RGB")
    
    # Prepare Normalization for Class Maps
    num_classes = len(class_names)
    bounds = np.arange(num_classes + 1) - 0.5
    norm = BoundaryNorm(bounds, num_classes)
    
    # TR: GT
    axes[0, 1].imshow(gt_map, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
    clean_ax(axes[0, 1], "Ground Truth")
    
    # BL: Default
    axes[1, 0].imshow(def_map, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
    clean_ax(axes[1, 0], "Default Prediction")
    
    # BR: Proposed
    axes[1, 1].imshow(prop_map, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
    clean_ax(axes[1, 1], "Proposed Prediction")
    
    # Legend
    # Create patches
    patches = [mpatches.Patch(color=cmap(i), label=name) 
               for i, name in enumerate(class_names) if i > 0] # Skip background 0
    
    # Place Legend at the bottom
    # We use fig.legend method
    # Calculate calculated_ncols as default
    calculated_ncols = 4 if len(patches) < 8 else 6
    if len(patches) > 12: calculated_ncols = 5 # FL_T has 15 classes
    
    ncols = config.get('legend_ncols', calculated_ncols)
    
    fig.legend(handles=patches, loc='lower center', 
               bbox_to_anchor=(0.5, config['legend_y']), ncol=ncols, 
               fontsize=14, frameon=False)
    
    print(f"Saving to {args.output}")
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
