#!/usr/bin/env python3
"""Generate per-branch heatmaps for selected filters."""
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from Load_Data import load_data
from SAR_utils import Standardize_data, padWithZeros
from model_factory import build_msatvit, load_saved_msatvit

DEFAULT_BRANCHES: Dict[str, List[int]] = {
    "spatial_conv3d_block1": list(range(8)),
    "polar_conv3d_block1": list(range(8)),
    "joint_conv3d_block1": list(range(8)),
}


def parse_args():
    p = argparse.ArgumentParser(description="Branch heatmap generator")
    p.add_argument("--dataset", default="SF", help="Dataset identifier (e.g., SF)")
    p.add_argument("--window-size", type=int, default=15, help="Patch window size")
    p.add_argument("--weights", default="ckpt/CV_MsAtViT_weights.h5", help="Path to weight file (fallback)")
    p.add_argument("--saved-model", default="ckpt/CV_MsAtViT_saved_model", help="Path to SavedModel directory")
    p.add_argument("--output", default="results/analysis/heatmaps", help="Output directory")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size for inference")
    return p.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    min_val = arr.min()
    max_val = arr.max()
    denom = max_val - min_val
    if denom < 1e-9:
        return np.zeros_like(arr)
    return (arr - min_val) / denom


def extract_patches_with_centers(
    data: np.ndarray,
    gt: np.ndarray,
    window_size: int,
    remove_zero_labels: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    margin = int((window_size - 1) / 2)
    padded = padWithZeros(data, margin=margin)
    H, W, _ = data.shape
    patches = []
    centers = []
    for r in range(margin, margin + H):
        for c in range(margin, margin + W):
            label = gt[r - margin, c - margin]
            if remove_zero_labels and label == 0:
                continue
            patch = padded[r - margin : r + margin + 1, c - margin : c + margin + 1]
            patches.append(patch)
            centers.append((r - margin, c - margin))
    patches = np.asarray(patches, dtype=np.complex64)
    centers = np.asarray(centers, dtype=np.int32)
    patches = np.expand_dims(patches, axis=4)
    return patches, centers


def get_kernel_slice(layer: tf.keras.layers.Layer, filter_idx: int, in_idx: int = 0) -> np.ndarray:
    weights = layer.get_weights()
    if not weights:
        raise RuntimeError(f"Layer {layer.name} has no weights")
    
    # Check if weights are split into real/imag parts (common in cvnn)
    if len(weights) >= 2 and weights[0].shape == weights[1].shape:
        # Assume weights[0] is real and weights[1] is imag
        kernel = weights[0] + 1j * weights[1]
    else:
        kernel = weights[0]

    if not np.iscomplexobj(kernel):
        kernel = kernel.astype(np.complex64)
    print(f"[heatmap kernels] {layer.name} kernel shape: {kernel.shape}")
    
    # kernel shape is (H, W, D, In, Out)
    # We want to return (H, W, D) for the specific filter and input channel
    
    kD, kH, kW, in_ch, out_ch = kernel.shape
    if filter_idx >= out_ch:
        raise ValueError(f"Filter index {filter_idx} out of range for layer {layer.name}")
    if in_idx >= in_ch:
        in_idx = 0

    # Extract the full 3D kernel for this filter
    kernel_3d = kernel[:, :, :, in_idx, filter_idx]

    print(
        "[kernel viz]",
        layer.name,
        "kernel shape:",
        kernel.shape,
        "extracted 3d shape:",
        kernel_3d.shape,
        "max imaginary:",
        float(np.abs(np.imag(kernel_3d)).max()),
    )
    return np.asarray(kernel_3d)


def save_filter_heatmap_combo(
    branch_name: str,
    filter_idx: int,
    kernel_3d: np.ndarray,
    heatmap: np.ndarray,
    out_dir: Path,
):
    # kernel_3d shape: (H, W, D)
    H, W, D = kernel_3d.shape
    
    # Create a directory for the filter to hold depth slices
    filter_dir = ensure_dir(out_dir / branch_name / f"sample{filter_idx:02d}") # Using sampleXX to match previous structure or just filterXX?
    # Previous code put combo_filterXX.png in branch_name/
    # Let's put them in branch_name/filterXX/
    filter_dir = ensure_dir(out_dir / branch_name / f"filter{filter_idx:02d}")

    heatmap_norm = normalize(heatmap)
    
    for d in range(D):
        kernel_slice = kernel_3d[:, :, d]
        
        fig = plt.figure(figsize=(10, 5))
        gs = fig.add_gridspec(2, 3)
        titles = ["Re", "Im", "|z|", "arg(z)"]
        values = [
            np.real(kernel_slice),
            np.imag(kernel_slice),
            np.abs(kernel_slice),
            np.angle(kernel_slice),
        ]
        cmaps = ["gray", "gray", "gray", "twilight"]
        for i in range(4):
            ax = fig.add_subplot(gs[i // 2, i % 2])
            # Use raw values, let imshow auto-scale
            im_sub = ax.imshow(values[i], cmap=cmaps[i], vmin=None if i < 3 else -np.pi, vmax=None if i < 3 else np.pi)
            ax.set_title(titles[i])
            ax.axis("off")
            # Add colorbar for each kernel subplot
            fig.colorbar(im_sub, ax=ax, fraction=0.046, pad=0.04)

        ax_heat = fig.add_subplot(gs[:, 2])
        im = ax_heat.imshow(heatmap_norm, cmap="hot", vmin=0.0, vmax=1.0)
        ax_heat.set_title("Response heatmap")
        ax_heat.axis("off")
        fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        
        # Add depth info to title
        fig.suptitle(f"{branch_name} filter {filter_idx} depth {d}")
        
        # Save as combo_depthXX.png
        combo_file = filter_dir / f"depth{d:02d}.png"
        fig.tight_layout()
        fig.savefig(combo_file, dpi=200)
        plt.close(fig)


def generate_branch_heatmaps(
    model: tf.keras.Model,
    x_all: np.ndarray,
    centers: np.ndarray,
    image_shape: Tuple[int, int],
    branch_name: str,
    filter_list: List[int],
    out_dir: Path,
    batch_size: int,
):
    branch_layer = model.get_layer(branch_name)
    branch_model = tf.keras.Model(inputs=model.input, outputs=branch_layer.output)
    heatmaps = {f: np.zeros(image_shape, dtype=np.float32) for f in filter_list}
    counts = np.zeros(image_shape, dtype=np.float32)

    for start in range(0, x_all.shape[0], batch_size):
        batch = x_all[start : start + batch_size]
        feat = branch_model(batch, training=False)
        amp = tf.abs(feat).numpy()
        reduce_axes = tuple(range(1, amp.ndim - 1))
        s_batch = amp.max(axis=reduce_axes)
        for bi in range(batch.shape[0]):
            idx = start + bi
            if idx >= centers.shape[0]:
                break
            y, x = centers[idx]
            counts[y, x] += 1
            for f in filter_list:
                if f >= s_batch.shape[-1]:
                    continue
                heatmaps[f][y, x] += s_batch[bi, f]

    counts[counts == 0] = 1.0
    save_root = ensure_dir(out_dir / branch_name)
    branch_layer = model.get_layer(branch_name)
    for f, arr in heatmaps.items():
        norm = arr / counts
        norm = np.where(norm > 0, norm, 0)
        if norm.max() > norm.min():
            norm = (norm - norm.min()) / (norm.max() - norm.min())
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(norm, cmap="hot", vmin=0.0, vmax=1.0)
        ax.set_title(f"{branch_name} filter {f}")
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        outfile = save_root / f"filter{f:02d}.png"
        fig.savefig(outfile, dpi=200)
        plt.close(fig)
        kernel_slice = get_kernel_slice(branch_layer, f)
        save_filter_heatmap_combo(branch_name, f, kernel_slice, norm, out_dir)


def main():
    args = parse_args()
    print("Loading dataset", args.dataset)
    data, gt = load_data(args.dataset)
    data = Standardize_data(data)
    x_all, centers = extract_patches_with_centers(data, gt, args.window_size)
    print(f"Total patches: {x_all.shape[0]}")

    if args.saved_model and os.path.isdir(args.saved_model):
        print(f"Loading SavedModel from {args.saved_model} ...")
        model = load_saved_msatvit(args.saved_model)
    else:
        print("SavedModel not found. Rebuilding and loading weights...")
        model = build_msatvit(
            input_shape=x_all.shape[1:],
            dataset=args.dataset,
            window_size=args.window_size,
        )
        model.load_weights(args.weights)

    out_dir = ensure_dir(Path(args.output))
    for branch_name, filters in DEFAULT_BRANCHES.items():
        print(f"Generating heatmaps for {branch_name} -> filters {filters}")
        generate_branch_heatmaps(
            model,
            x_all,
            centers,
            gt.shape,
            branch_name,
            filters,
            out_dir,
            args.batch_size,
        )

    print(f"Heatmaps saved under {out_dir}")


if __name__ == "__main__":
    main()
