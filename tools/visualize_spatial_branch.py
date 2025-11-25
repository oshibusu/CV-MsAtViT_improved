#!/usr/bin/env python3
"""Visualize kernels and feature maps of the spatial branch of CV-MsAtViT."""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from Load_Data import load_data
from SAR_utils import Standardize_data, createImageCubes, splitTrainTestSet, cart_gelu
from cvnn.activations import cart_relu, cart_sigmoid, softmax_real_with_abs

SPATIAL_LAYER_NAMES = [
    "spatial_conv3d_block1",
    "spatial_conv3d_block2",
]


def parse_args():
    p = argparse.ArgumentParser(description="Visualize CV-MsAtViT spatial branch")
    p.add_argument("--dataset", default="FL_T", help="Dataset key used during training")
    p.add_argument("--window-size", type=int, default=15, help="Sliding window size")
    p.add_argument("--test-ratio", type=float, default=0.99, help="Train/test split ratio")
    p.add_argument("--ckpt", default="ckpt/CV_MsAtViT_saved", help="Path to SavedModel directory")
    p.add_argument("--output", default="results/analysis/spatial_branch", help="Base directory for figures")
    p.add_argument("--samples", type=int, default=4, help="Number of training patches to visualize")
    return p.parse_args()


def ensure_dir(path: Path):
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

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(real, cmap="gray")
    axes[0].set_title("Re")
    axes[1].imshow(imag, cmap="gray")
    axes[1].set_title("Im")
    axes[2].imshow(magnitude, cmap="gray")
    axes[2].set_title("|z|")
    im = axes[3].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[3].set_title("arg(z)")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def visualize_kernels(model: tf.keras.Model, out_dir: Path):
    for layer_name in SPATIAL_LAYER_NAMES:
        layer = model.get_layer(layer_name)
        weights = layer.get_weights()
        if not weights:
            continue
        kernel = weights[0]
        if not np.iscomplexobj(kernel):
            kernel = kernel.astype(np.complex64)
        kd, kh, kw, in_ch, out_ch = kernel.shape
        for oc in range(out_ch):
            for ic in range(in_ch):
                for depth in range(kd):
                    slice_2d = kernel[depth, :, :, ic, oc]
                    title = f"{layer_name} | out {oc} in {ic} depth {depth}"
                    outfile = out_dir / "kernels" / layer_name / f"filter{oc:02d}" / f"in{ic:02d}_depth{depth:02d}.png"
                    save_complex_panel(slice_2d, title, outfile)


def build_feature_extractor(model: tf.keras.Model) -> tf.keras.Model:
    outputs = [model.get_layer(name).output for name in SPATIAL_LAYER_NAMES]
    return tf.keras.Model(inputs=model.input, outputs=outputs)


def load_samples(args) -> np.ndarray:
    data, gt = load_data(args.dataset)
    data = Standardize_data(data)
    X_coh, y = createImageCubes(data, gt, args.window_size)
    X_coh = np.expand_dims(X_coh, axis=4)
    X_train, _, y_train, _ = splitTrainTestSet(X_coh, y, args.test_ratio)
    count = min(args.samples, X_train.shape[0])
    if count == 0:
        raise RuntimeError("No training samples available after split")
    return X_train[:count], y_train[:count]


def visualize_feature_maps(model: tf.keras.Model, samples: np.ndarray, labels: np.ndarray, out_dir: Path):
    extractor = build_feature_extractor(model)
    feature_maps = extractor(samples, training=False)
    if not isinstance(feature_maps, (list, tuple)):
        feature_maps = [feature_maps]
    for layer_idx, layer_name in enumerate(SPATIAL_LAYER_NAMES):
        fm = feature_maps[layer_idx].numpy()
        batch, dim1, dim2, dim3, out_ch = fm.shape
        for sample_idx in range(batch):
            label = int(labels[sample_idx]) if labels is not None else -1
            for depth in range(dim3):
                for oc in range(out_ch):
                    map2d = fm[sample_idx, :, :, depth, oc]
                    title = f"{layer_name} | sample {sample_idx} (label {label}) | depth {depth} filter {oc}"
                    outfile = out_dir / "feature_maps" / layer_name / f"sample{sample_idx:02d}" / f"depth{depth:02d}_filter{oc:02d}.png"
                    save_complex_panel(map2d, title, outfile)


def main():
    args = parse_args()
    custom_objects = {"cart_gelu": cart_gelu,
                       "cart_relu": cart_relu,
                       "cart_sigmoid": cart_sigmoid,
                       "softmax_real_with_abs": softmax_real_with_abs}
    model = tf.keras.models.load_model(args.ckpt, compile=False, custom_objects=custom_objects)
    out_dir = ensure_dir(Path(args.output))
    visualize_kernels(model, out_dir)
    samples, labels = load_samples(args)
    visualize_feature_maps(model, samples, labels, out_dir)
    print(f"Saved visualizations to {out_dir}")


if __name__ == "__main__":
    main()
