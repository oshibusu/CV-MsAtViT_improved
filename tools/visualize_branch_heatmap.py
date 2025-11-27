#!/usr/bin/env python3
"""Generate per-branch heatmaps for selected filters."""
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from Load_Data import load_data
from SAR_utils import Standardize_data, padWithZeros
from model_factory import build_msatvit

DEFAULT_BRANCHES: Dict[str, List[int]] = {
    "spatial_conv3d_block1": list(range(8)),
    "polar_conv3d_block1": list(range(8)),
    "joint_conv3d_block1": list(range(8)),
}


def parse_args():
    p = argparse.ArgumentParser(description="Branch heatmap generator")
    p.add_argument("--dataset", default="SF", help="Dataset identifier (e.g., SF)")
    p.add_argument("--window-size", type=int, default=15, help="Patch window size")
    p.add_argument("--weights", default="ckpt/CV_MsAtViT_weights.h5", help="Path to weight file")
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


def get_kernel_slice(layer: tf.keras.layers.Layer, filter_idx: int, depth_idx: int = 0, in_idx: int = 0) -> np.ndarray:
    weights = layer.get_weights()
    if not weights:
        raise RuntimeError(f"Layer {layer.name} has no weights")
    kernel = weights[0]
    if not np.iscomplexobj(kernel):
        kernel = kernel.astype(np.complex64)
    print(f"[heatmap kernels] {layer.name} kernel shape: {kernel.shape}")
    kD, kH, kW, in_ch, out_ch = kernel.shape

    def branch_type(name: str) -> str:
        if "spatial_conv3d" in name:
            return "spatial"
        if "polar_conv3d" in name:
            return "polar"
        if "joint_conv3d" in name:
            return "joint"
        return "other"

    btype = branch_type(layer.name)
    if filter_idx >= out_ch:
        raise ValueError(f"Filter index {filter_idx} out of range for layer {layer.name}")
    if in_idx >= in_ch:
        in_idx = 0

    if btype == "spatial":
        slice_2d = kernel[:, :, 0, in_idx, filter_idx]
    elif btype == "polar":
        slice_2d = kernel[0, 0, :, in_idx, filter_idx].reshape(1, -1)
    elif btype == "joint":
        depth_idx = depth_idx % max(1, kD)
        slice_2d = kernel[depth_idx, :, :, in_idx, filter_idx]
    else:
        depth_idx = depth_idx % max(1, kD)
        slice_2d = kernel[depth_idx, :, :, in_idx, filter_idx]

    print(
        "[kernel viz]",
        layer.name,
        "kernel shape:",
        kernel.shape,
        "slice shape:",
        slice_2d.shape,
        "max imaginary:",
        float(np.abs(np.imag(slice_2d)).max()),
    )
    return np.asarray(slice_2d)


def save_filter_heatmap_combo(
    branch_name: str,
    filter_idx: int,
    kernel_slice: np.ndarray,
    heatmap: np.ndarray,
    out_dir: Path,
):
    heatmap_dir = ensure_dir(out_dir / branch_name)
    heatmap_norm = normalize(heatmap)
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 3)
    titles = ["Re", "Im", "|z|", "arg(z)"]
    values = [
        normalize(np.real(kernel_slice)),
        normalize(np.imag(kernel_slice)),
        normalize(np.abs(kernel_slice)),
        np.angle(kernel_slice),
    ]
    cmaps = ["gray", "gray", "gray", "twilight"]
    for i in range(4):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.imshow(values[i], cmap=cmaps[i], vmin=None if i < 3 else -np.pi, vmax=None if i < 3 else np.pi)
        ax.set_title(titles[i])
        ax.axis("off")
    ax_heat = fig.add_subplot(gs[:, 2])
    im = ax_heat.imshow(heatmap_norm, cmap="hot", vmin=0.0, vmax=1.0)
    ax_heat.set_title("Response heatmap")
    ax_heat.axis("off")
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    fig.suptitle(f"{branch_name} filter {filter_idx}")
    combo_file = heatmap_dir / f"combo_filter{filter_idx:02d}.png"
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
