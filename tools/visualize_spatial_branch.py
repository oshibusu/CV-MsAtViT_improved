#!/usr/bin/env python3
"""Filter selection and kernel/feature visualization for CV-MsAtViT branches."""
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from Load_Data import load_data
from SAR_utils import Standardize_data, createImageCubes, splitTrainTestSet
from model_factory import build_msatvit

BRANCH_LAYERS = [
    "spatial_conv3d_block1",
    "polar_conv3d_block1",
    "joint_conv3d_block1",
]


def parse_args():
    p = argparse.ArgumentParser(description="Branch visualization toolkit")
    p.add_argument("--dataset", default="FL_T", help="Dataset identifier (e.g., FL_T, SF)")
    p.add_argument("--window-size", type=int, default=15, help="Patch window size")
    p.add_argument("--test-ratio", type=float, default=0.99, help="Train/test split ratio")
    p.add_argument("--weights", default="ckpt/CV_MsAtViT_weights.h5", help="Path to weight file")
    p.add_argument("--output", default="results/analysis/spatial_branch", help="Base directory for figures")
    p.add_argument("--samples", type=int, default=4, help="#patches for feature-map visualization")
    p.add_argument("--top-k", type=int, default=3, help="#filters to select per branch")
    p.add_argument("--selection-batch-size", type=int, default=128, help="Batch size for filter scoring")
    p.add_argument(
        "--feature-layers",
        default="spatial_conv3d_block1",
        help="Comma-separated layer names for feature maps",
    )
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


def save_complex_panel(matrix: np.ndarray, title: str, out_path: Path):
    real = normalize(np.real(matrix))
    imag = normalize(np.imag(matrix))
    magnitude = normalize(np.abs(matrix))
    phase = np.angle(matrix)

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    axes = axes.ravel()
    axes[0].imshow(real, cmap="gray")
    axes[0].set_title("Re")
    axes[1].imshow(imag, cmap="gray")
    axes[1].set_title("Im")
    axes[2].imshow(magnitude, cmap="gray")
    axes[2].set_title("|z|")
    axes[3].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[3].set_title("arg(z)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_training_split(args):
    data, gt = load_data(args.dataset)
    data = Standardize_data(data)
    X_coh, y = createImageCubes(data, gt, args.window_size)
    X_coh = np.expand_dims(X_coh, axis=4)
    X_train, _, y_train, _ = splitTrainTestSet(X_coh, y, args.test_ratio)
    if X_train.shape[0] == 0:
        raise RuntimeError("split produced zero training samples")
    return X_train, y_train


def select_top_filters(
    model: tf.keras.Model,
    layer_name: str,
    data: np.ndarray,
    batch_size: int,
    top_k: int,
) -> np.ndarray:
    layer = model.get_layer(layer_name)
    sub_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
    scores_accum: List[np.ndarray] = []
    for start in range(0, data.shape[0], batch_size):
        batch = data[start : start + batch_size]
        feat = sub_model(batch, training=False)
        amp = tf.abs(feat)
        reduce_axes = tuple(range(1, len(amp.shape) - 1))
        s_batch = tf.reduce_max(amp, axis=reduce_axes)  # (B, C_out)
        scores_accum.append(s_batch.numpy())
    scores = np.concatenate(scores_accum, axis=0).mean(axis=0)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices


def visualize_kernels_for_branch(
    layer: tf.keras.layers.Layer,
    filter_indices: List[int],
    save_dir: Path,
):
    weights = layer.get_weights()
    if not weights:
        return
    kernel = weights[0]
    if not np.iscomplexobj(kernel):
        kernel = kernel.astype(np.complex64)
    print(f"[kernels] {layer.name} kernel shape: {kernel.shape}")
    kD, kH, kW, in_ch, out_ch = kernel.shape
    spatial_axes = [i for i, size in enumerate((kD, kH, kW)) if size == 3]
    if len(spatial_axes) < 2:
        raise ValueError(
            f"Layer {layer.name} does not have two spatial axes of size 3; shape={kernel.shape}"
        )
    spatial_axes = spatial_axes[:2]
    depth_axis = [i for i in range(3) if i not in spatial_axes][0]
    perm = spatial_axes + [depth_axis, 3, 4]
    kernel_perm = np.transpose(kernel, axes=perm)
    depth_len = kernel_perm.shape[2]
    for oc in filter_indices:
        if oc >= kernel_perm.shape[-1]:
            continue
        for ic in range(kernel_perm.shape[-2]):
            for depth in range(depth_len):
                slice_2d = kernel_perm[:, :, depth, ic, oc]
                if slice_2d.shape != (3, 3):
                    slice_2d = np.reshape(slice_2d, (3, 3))
                title = f"{layer.name} | filter {oc} | in {ic} | depth {depth}"
                outfile = (
                    save_dir
                    / layer.name
                    / f"filter{oc:02d}"
                    / f"in{ic:02d}_depth{depth:02d}.png"
                )
                save_complex_panel(slice_2d, title, outfile)


def visualize_feature_maps(
    model: tf.keras.Model,
    samples: np.ndarray,
    labels: np.ndarray,
    layer_name: str,
    filter_indices: Optional[List[int]],
    out_dir: Path,
):
    extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    fm = extractor(samples, training=False).numpy()
    batch = fm.shape[0]
    depth_len = fm.shape[-2]
    out_ch = fm.shape[-1]
    filters = filter_indices or list(range(out_ch))
    for sample_idx in range(batch):
        label = int(labels[sample_idx]) if labels is not None and len(labels) > sample_idx else -1
        for depth in range(depth_len):
            for oc in filters:
                if oc >= out_ch:
                    continue
                slice_2d = fm[sample_idx, ..., depth, oc]
                title = (
                    f"{layer_name} | sample {sample_idx} (label {label}) | "
                    f"depth {depth} filter {oc}"
                )
                outfile = (
                    out_dir
                    / "feature_maps"
                    / layer_name
                    / f"sample{sample_idx:02d}"
                    / f"depth{depth:02d}_filter{oc:02d}.png"
                )
                save_complex_panel(slice_2d, title, outfile)


def main():
    args = parse_args()
    print("Loading training split...")
    X_train, y_train = load_training_split(args)
    sample_count = min(args.samples, X_train.shape[0])
    samples = X_train[:sample_count]
    sample_labels = y_train[:sample_count]

    print("Rebuilding model and loading weights...")
    model = build_msatvit(
        input_shape=X_train.shape[1:],
        dataset=args.dataset,
        window_size=args.window_size,
    )
    model.load_weights(args.weights)

    out_dir = ensure_dir(Path(args.output))

    print("Selecting top filters per branch...")
    branch_filters: Dict[str, np.ndarray] = {}
    for layer_name in BRANCH_LAYERS:
        top_idx = select_top_filters(
            model,
            layer_name,
            X_train,
            batch_size=args.selection_batch_size,
            top_k=args.top_k,
        )
        branch_filters[layer_name] = top_idx
        print(f"Top filters for {layer_name}: {top_idx.tolist()}")
        visualize_kernels_for_branch(
            model.get_layer(layer_name),
            top_idx,
            out_dir / "kernels",
        )

    feature_layers = [name.strip() for name in args.feature_layers.split(",") if name.strip()]
    if sample_count > 0 and feature_layers:
        print("Generating feature maps for selected layers...")
        for layer_name in feature_layers:
            filters = branch_filters.get(layer_name)
            if filters is None:
                print(f"[warn] layer {layer_name} not in branch filter list; skipping")
                continue
            visualize_feature_maps(
                model,
                samples,
                sample_labels,
                layer_name,
                filters.tolist(),
                out_dir,
            )

    print(f"Saved visualizations to {out_dir}")


if __name__ == "__main__":
    main()
