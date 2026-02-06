import os
import sys
import numpy as np
import scipy.io as sio
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Add root directory to path to import Load_Data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Load_Data import load_data

def get_mat_key(mat):
    for key in mat.keys():
        if not key.startswith('__'):
            return key
    return None

# Class Names Definitions
CLASS_NAMES_BALTRUM = [
    "Background", "Tidal flat", "Water", "Coastal shrub", "Dense, high vegetation",
    "White dune", "Peat bog", "Grey dunes", "Couch grass", "Upper salt marsh",
    "Lower salt marsh", "Sand", "Settlement"
]

# Flevoland 15 classes (Standard AIRSAR)
CLASS_NAMES_FL_T = [
    "Background",
    "Water",
    "Forest",
    "Lucerne",
    "Grass",
    "Rapeseed",
    "Beet",
    "Potatoes",
    "Peas",
    "Stem Beans",
    "Bare Soil",
    "Wheat",
    "Wheat 2",
    "Wheat 3",
    "Barley",
    "Buildings"
]

# San Francisco 5 classes
# San Francisco 5 classes (Updated per user corrected instruction)
CLASS_NAMES_SF = [
    "Background",
    "Bare Soil",
    "Mountain",
    "Water",
    "Urban",
    "Vegetation"
]

def get_class_names(dataset_name):
    if "Baltrum" in dataset_name:
        return CLASS_NAMES_BALTRUM
    elif "FL_T" in dataset_name:
        return CLASS_NAMES_FL_T
    elif "SF" in dataset_name:
        return CLASS_NAMES_SF
    else:
        return [f"Class {i}" for i in range(20)] # Fallback

CLASS_NAMES = []

def analyze_file(filepath, gt):
    print(f"\n{'='*50}")
    print(f"Analyzing: {os.path.basename(filepath)}")
    print(f"{'='*50}")
    
    if not os.path.exists(filepath):
        print("File not found.")
        return

    mat = sio.loadmat(filepath)
    key = get_mat_key(mat)
    if key is None:
        print("No valid key found in mat file.")
        return
    
    pred_map = mat[key]
    
    # 1. Non-zero count
    nonzero_count = np.count_nonzero(pred_map)
    zero_count = pred_map.size - nonzero_count
    print(f"Non-zero labels count: {nonzero_count}")
    print(f"Zero labels count (Background): {zero_count}")
    
    # Check against GT non-zero count
    gt_nonzero = np.count_nonzero(gt)
    gt_zero = gt.size - gt_nonzero
    print(f"Ground Truth non-zero count: {gt_nonzero}")
    print(f"Ground Truth zero count: {gt_zero}")
    
    if nonzero_count == gt_nonzero:
        print("-> Matches GT count exactly.")
    else:
        print(f"-> DIFFERS from GT count by {nonzero_count - gt_nonzero}")

    # 2. Precision, Recall, F1
    # Flatten and mask
    mask = gt > 0
    y_true = gt[mask]
    y_pred = pred_map[mask]
    
    print("\n--- Classification Report ---")
    
    if len(y_true) == 0:
        print("No valid GT pixels found for evaluation.")
        return

    # Get class names present in the classification report
    # labels are 1-based
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    target_names = [CLASS_NAMES[i] for i in unique_labels if i < len(CLASS_NAMES)]
    
    print(classification_report(y_true, y_pred, digits=4, labels=unique_labels, target_names=target_names))

    # 3. Misclassification Stats
    print("\n--- Confusion Statistics (Normalized by True Labels) ---")
    cm = confusion_matrix(y_true, y_pred)
    # Normalize by row (True)
    with np.errstate(divide='ignore', invalid='ignore'):
         cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
         cm_norm = np.nan_to_num(cm_norm)
    
    classes = sorted(list(set(y_true) | set(y_pred)))
    
    # Display nicely
    # For each class, show top misclassifications
    for i, class_label in enumerate(classes):
        # Diagonal is accuracy for this class
        if i < cm_norm.shape[0] and i < cm_norm.shape[1]:
             acc = cm_norm[i, i]
        else:
             acc = 0.0
        
        c_name = CLASS_NAMES[class_label] if class_label < len(CLASS_NAMES) else f"Class {class_label}"
        print(f"\nClass {class_label} ({c_name}) (Total: {np.sum(cm[i])}): Correct: {acc:.2%}")
        
        # Sort misclassifications
        row = cm_norm[i]
        sorted_indices = np.argsort(row)[::-1]
        
        found_mistake = False
        for idx in sorted_indices:
            if idx == i: continue
            if row[idx] > 0.001: # Show if > 0.1% confusion
                target_cls = classes[idx]
                target_name = CLASS_NAMES[target_cls] if target_cls < len(CLASS_NAMES) else f"Class {target_cls}"
                print(f"  -> Confused as Class {target_cls} ({target_name}): {row[idx]:.2%} ({cm[i, idx]} samples)")
                found_mistake = True
        
        if not found_mistake and acc < 1.0:
             print("  -> Scattered errors (all < 0.1%)")
             
    # 4. Visualize Confusion Matrix
    print("\n--- Visualizing Confusion Matrix ---")
    plt.figure(figsize=(14, 12))
    plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = ["Times New Roman"] # Might not exist on minimal docker
    
    # Use full class names for axis
    tick_labels = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i) for i in classes]
    
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=tick_labels, yticklabels=tick_labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix: {os.path.basename(filepath)}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save fig
    out_name = os.path.splitext(os.path.basename(filepath))[0] + "_cm.png"
    out_path = os.path.join(os.path.dirname(filepath), out_name)
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"Confusion matrix visualization saved to {out_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze prediction mat files against GT.")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. FL_T, SF, Baltrum_S_FP1)")
    parser.add_argument("files", nargs="+", help="List of .mat files to analyze")
    args = parser.parse_args()

    dataset_name = args.dataset
    print(f"Loading Ground Truth for {dataset_name}...")
    try:
        _, gt = load_data(dataset_name)
    except Exception as e:
        print(f"Error loading GT: {e}")
        return

    # Determine class names
    global CLASS_NAMES
    CLASS_NAMES = get_class_names(dataset_name)

    for f in args.files:
        analyze_file(f, gt)

if __name__ == "__main__":
    main()
