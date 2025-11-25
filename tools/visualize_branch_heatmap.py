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
    "spatial_conv3d_block1": [0, 2, 5],
    "polar_conv3d_block1": [1, 4, 6],
    "joint_conv3d_block1": [0, 3, 7],
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
