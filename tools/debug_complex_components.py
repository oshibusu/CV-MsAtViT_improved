#!/usr/bin/env python3
"""Debug complex-valued components: kernels and feature maps."""
import argparse
import os
import numpy as np
import tensorflow as tf

from Load_Data import load_data
from SAR_utils import Standardize_data, createImageCubes, splitTrainTestSet
from model_factory import build_msatvit, load_saved_msatvit

BRANCH_LAYERS = [
    "spatial_conv3d_block1",
    "polar_conv3d_block1",
    "joint_conv3d_block1",
]


def summarize_tensor(name: str, tensor: tf.Tensor):
    real = tf.math.real(tensor)
    imag = tf.math.imag(tensor)
    print(
        f"{name}: dtype={tensor.dtype}, "
        f"real max={float(tf.reduce_max(tf.abs(real)))}, "
        f"imag max={float(tf.reduce_max(tf.abs(imag)))}"
    )


def _dataset_tag(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def main():
    parser = argparse.ArgumentParser("Inspect complex kernels and feature maps")
    parser.add_argument("--dataset", default="FL_T")
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--test-ratio", type=float, default=0.99)
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to weight file (fallback). Default derives from dataset name.",
    )
    parser.add_argument(
        "--saved-model",
        default=None,
        help="Path to SavedModel directory. Default derives from dataset name.",
    )
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()
    dataset_tag = _dataset_tag(args.dataset)
    default_weights = os.path.join("ckpt", f"CV_MsAtViT_{dataset_tag}_weights.h5")
    default_saved_model = os.path.join("ckpt", f"CV_MsAtViT_{dataset_tag}_saved_model")
    weights_path = args.weights or default_weights
    saved_model_dir = args.saved_model or default_saved_model

    print("[1] Loading data ...")
    data, gt = load_data(args.dataset)
    data = Standardize_data(data)
    X_coh, y = createImageCubes(data, gt, args.window_size)
    X_coh = np.expand_dims(X_coh, axis=4)
    X_train, _, y_train, _ = splitTrainTestSet(X_coh, y, args.test_ratio)
    print("  X_train shape:", X_train.shape, X_train.dtype)
    print("  sample real/imag stats (channel 3):",
          np.max(np.abs(np.real(X_train[..., 3, :])), axis=None),
          np.max(np.abs(np.imag(X_train[..., 3, :])), axis=None))

    print("[2] Loading model ...")
    if saved_model_dir and os.path.isdir(saved_model_dir):
        model = load_saved_msatvit(saved_model_dir)
    else:
        model = build_msatvit(
            input_shape=X_train.shape[1:],
            dataset=args.dataset,
            window_size=args.window_size,
        )
        model.load_weights(weights_path)

    print("[3] Kernel stats per branch")
    for layer_name in BRANCH_LAYERS:
        layer = model.get_layer(layer_name)
        weights = layer.get_weights()
        if not weights:
            print(layer_name, "has no weights")
            continue
        kernel = weights[0]
        if not np.iscomplexobj(kernel):
            kernel = kernel.astype(np.complex64)
        summarize_tensor(f"kernel[{layer_name}]", tf.convert_to_tensor(kernel))

    print("[4] Feature map stats for small batch")
    sample_count = min(args.samples, X_train.shape[0])
    if sample_count == 0:
        raise RuntimeError("No samples available for feature map inspection")
    batch = X_train[:sample_count]
    for layer_name in BRANCH_LAYERS:
        extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        feat = extractor(batch)
        summarize_tensor(f"feat[{layer_name}]", feat)


if __name__ == "__main__":
    main()
