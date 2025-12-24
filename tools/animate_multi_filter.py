#!/usr/bin/env python3
"""Animate multi-filter kernel slices across batches/epochs."""
import argparse
import glob
import io
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Load_Data import load_data
from SAR_utils import Standardize_data, createImageCubes
from model_factory import build_msatvit

BRANCH_LAYERS = [
    "spatial_conv3d_block1",
    "polar_conv3d_block1",
    "joint_conv3d_block1",
]


def parse_args():
    p = argparse.ArgumentParser(description="Animate multi-filter kernel evolution")
    p.add_argument("--dataset", default="SF")
    p.add_argument("--window-size", type=int, default=15)
    p.add_argument("--mode", choices=["epoch", "batch"], default="batch")
    p.add_argument("--weights-pattern", default=None)
    p.add_argument("--batch-trace-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--frame-duration", type=int, default=400)
    p.add_argument("--max-epochs", type=int, default=1)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--branches", default="")
    p.add_argument("--filters", default="")
    p.add_argument("--in-index", type=int, default=0)
    p.add_argument("--depths", default="")
    p.add_argument("--frame-filters", type=int, default=24)
    p.add_argument("--combine-branches", action="store_true", help="Combine all branches into a single GIF")
    return p.parse_args()


def dataset_tag(name):
    return name.replace("/", "_").replace("\\", "_")


def combine_complex_kernel(weights):
    if not weights:
        raise ValueError("Layer has no weights")
    kernel = weights[0]
    if len(weights) >= 2 and weights[0].shape == weights[1].shape and not np.iscomplexobj(weights[0]):
        kernel = weights[0] + 1j * weights[1]
    if not np.iscomplexobj(kernel):
        kernel = kernel.astype(np.complex64)
    return kernel


def parse_index_spec(spec: str, max_len: int):
    if not spec:
        return list(range(max_len))
    indices = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            for i in range(start, end + 1):
                if 0 <= i < max_len:
                    indices.add(i)
        else:
            try:
                val = int(part)
            except ValueError:
                continue
            if 0 <= val < max_len:
                indices.add(val)
    return sorted(indices)


def run_epoch_mode(args, tag):
    pattern = args.weights_pattern or os.path.join("ckpt", f"CV_MsAtViT_{tag}_epoch*.weights.h5")
    weight_files = sorted(glob.glob(pattern))
    if args.max_epochs:
        weight_files = weight_files[: args.max_epochs]
    if len(weight_files) < 1:
        raise SystemExit("No weights found for epoch mode")

    branches = [b.strip() for b in args.branches.split(",") if b.strip()]
    if branches:
        invalid = [b for b in branches if b not in BRANCH_LAYERS]
        if invalid:
            raise SystemExit(f"Unknown branches: {invalid}")
    else:
        branches = BRANCH_LAYERS

    data, gt = load_data(args.dataset)
    data = Standardize_data(data)
    patches, _ = createImageCubes(data, gt, args.window_size)
    del patches
    model = build_msatvit(
        input_shape=(args.window_size, args.window_size, data.shape[-1], 1),
        dataset=args.dataset,
        window_size=args.window_size,
    )

    out_root = Path(args.output_dir) if args.output_dir else Path("results/analysis") / tag / "epoch_multi"
    out_root.mkdir(parents=True, exist_ok=True)

    for branch in branches:
        frames = []
        labels = []
        for wf in weight_files:
            model.load_weights(wf)
            layer = model.get_layer(branch)
            kernel = combine_complex_kernel(layer.get_weights())
            filter_indices = parse_index_spec(args.filters, kernel.shape[-1])
            depth_indices = parse_index_spec(args.depths, kernel.shape[2])
            frame = render_multi_filter_frame(
                kernel,
                filter_indices,
                args.in_index,
                depth_indices,
                title=os.path.basename(wf),
            )
            frames.append(frame)
            labels.append(os.path.basename(wf))
        if frames:
            out_path = out_root / f"{branch}_multi.gif"
            save_gif(frames, out_path, args.frame_duration)
            print("Saved", out_path)


def collect_batch_snapshots(trace_root: Path, max_epochs: int, max_batches: Optional[int]):
    batch_dirs = sorted(
        [d for d in trace_root.glob("epoch*_batch*") if d.is_dir()],
        key=batch_dir_sort_key,
    )
    if not batch_dirs:
        raise SystemExit(f"No batch trace folders under {trace_root}")
    snapshots = []
    epoch_counts = {}
    for d in batch_dirs:
        match = re.search(r"epoch(\d+)_batch", d.name)
        epoch_idx = int(match.group(1)) if match else 1
        if epoch_idx > max_epochs:
            continue
        epoch_counts.setdefault(epoch_idx, 0)
        if max_batches and epoch_counts[epoch_idx] >= max_batches:
            continue
        weights_path = d / "weights.h5"
        if not weights_path.exists():
            continue
        metrics_path = d / "metrics.json"
        metrics = {}
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
            except Exception:
                metrics = {}
        def _to_float(val):
            if val in (None, ""):
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
        snapshots.append(
            {
                "weights": weights_path,
                "label": d.name,
                "metrics": {"loss": _to_float(metrics.get("loss")), "accuracy": _to_float(metrics.get("accuracy"))},
                "epoch": epoch_idx,
                "branch_dir": d,
            }
        )
        epoch_counts[epoch_idx] += 1
    return snapshots


def run_batch_mode(args, tag):
    trace_root = Path(args.batch_trace_dir) if args.batch_trace_dir else Path("ckpt") / "batch_traces" / tag
    snapshots = collect_batch_snapshots(trace_root, args.max_epochs, args.max_batches)
    if not snapshots:
        raise SystemExit("No usable batch weights found")

    branches = [b.strip() for b in args.branches.split(",") if b.strip()]
    if branches:
        invalid = [b for b in branches if b not in BRANCH_LAYERS]
        if invalid:
            raise SystemExit(f"Unknown branches: {invalid}")
    else:
        branches = BRANCH_LAYERS

    data, gt = load_data(args.dataset)
    data = Standardize_data(data)
    patches, _ = createImageCubes(data, gt, args.window_size)
    del patches
    model = build_msatvit(
        input_shape=(args.window_size, args.window_size, data.shape[-1], 1),
        dataset=args.dataset,
        window_size=args.window_size,
    )

    out_root = Path(args.output_dir) if args.output_dir else Path("results/analysis") / tag / "batch_multi"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.combine_branches:
        frames = []
        for snap in snapshots:
            weights_path = snap["weights"]
            label = snap["label"]
            metrics = snap.get("metrics") or {}
            subtitle = []
            if metrics.get("loss") is not None:
                subtitle.append(f"loss={metrics['loss']:.4f}")
            if metrics.get("accuracy") is not None:
                subtitle.append(f"acc={metrics['accuracy']:.4f}")
            subtitle_text = ", ".join(subtitle) if subtitle else None
            model.load_weights(weights_path)
            combined_slices = []
            for branch in branches:
                layer = model.get_layer(branch)
                kernel = combine_complex_kernel(layer.get_weights())
                filter_indices = parse_index_spec(args.filters, kernel.shape[-1])
                depth_indices = parse_index_spec(args.depths, kernel.shape[2])
                for filt_idx in filter_indices:
                    if filt_idx >= kernel.shape[-1]:
                        continue
                    vol = kernel[:, :, :, args.in_index, filt_idx]
                    depth_len = vol.shape[2]
                    for depth in depth_indices:
                        d = depth % depth_len
                        matrix = vol[:, :, d]
                        combined_slices.append((f"{branch}:f{filt_idx} d{depth}", matrix))
            frame = render_multi_filter_frame_from_slices(
                combined_slices,
                title=label,
                subtitle=subtitle_text,
                filters_per_frame=args.frame_filters,
            )
            frames.append(frame)
        if frames:
            out_path = out_root / "combined_multi.gif"
            save_gif(frames, out_path, args.frame_duration)
            print("Saved", out_path)
    else:
        for branch in branches:
            frames = []
            for snap in snapshots:
                weights_path = snap["weights"]
                label = snap["label"]
                metrics = snap.get("metrics") or {}
                subtitle = []
                if metrics.get("loss") is not None:
                    subtitle.append(f"loss={metrics['loss']:.4f}")
                if metrics.get("accuracy") is not None:
                    subtitle.append(f"acc={metrics['accuracy']:.4f}")
                subtitle_text = ", ".join(subtitle) if subtitle else None
                model.load_weights(weights_path)
                layer = model.get_layer(branch)
                kernel = combine_complex_kernel(layer.get_weights())
                filter_indices = parse_index_spec(args.filters, kernel.shape[-1])
                depth_indices = parse_index_spec(args.depths, kernel.shape[2])
                frame = render_multi_filter_frame(
                    kernel,
                    filter_indices,
                    args.in_index,
                    depth_indices,
                    title=label,
                    subtitle=subtitle_text,
                    filters_per_frame=args.frame_filters,
                )
                frames.append(frame)
            if frames:
                out_path = out_root / f"{branch}_multi.gif"
                save_gif(frames, out_path, args.frame_duration)
                print("Saved", out_path)
@@
 def render_multi_filter_frame(
@@
-    n_rows = len(chunk)
-    fig = plt.figure(figsize=(10, max(2, n_rows * 0.6)))
-    gs = GridSpec(n_rows, 4, figure=fig, wspace=0.2, hspace=0.4)
-    for row, (label, matrix) in enumerate(chunk):
-        panels = [np.real(matrix), np.imag(matrix), np.abs(matrix), np.angle(matrix)]
-        cmaps = ["gray", "gray", "gray", "twilight"]
-        norms = [norm, norm, norm, Normalize(vmin=-math.pi, vmax=math.pi)]
-        titles = ["Re", "Im", "|z|", "arg(z)"]
-        for col in range(4):
-            ax = fig.add_subplot(gs[row, col])
-            ax.imshow(panels[col], cmap=cmaps[col], norm=norms[col])
-            if row == 0:
-                ax.set_title(titles[col])
-            ax.set_xticks([])
-            ax.set_yticks([])
-            if col == 0:
-                ax.set_ylabel(label)
+    n_rows = len(chunk)
+    fig = plt.figure(figsize=(12, max(2, n_rows * 0.6)))
+    gs = GridSpec(n_rows, 4, figure=fig, wspace=0.2, hspace=0.4)
+    re_list, im_list, abs_list = [], [], []
+    for _, matrix in chunk:
+        re_list.append(np.real(matrix))
+        im_list.append(np.imag(matrix))
+        abs_list.append(np.abs(matrix))
+    re_stack = np.concatenate([m.ravel() for m in re_list]) if re_list else np.array([0])
+    im_stack = np.concatenate([m.ravel() for m in im_list]) if im_list else np.array([0])
+    abs_stack = np.concatenate([m.ravel() for m in abs_list]) if abs_list else np.array([0])
+    re_norm = Normalize(vmin=float(np.nanmin(re_stack)), vmax=float(np.nanmax(re_stack)))
+    im_norm = Normalize(vmin=float(np.nanmin(im_stack)), vmax=float(np.nanmax(im_stack)))
+    abs_norm = Normalize(vmin=float(np.nanmin(abs_stack)), vmax=float(np.nanmax(abs_stack)))
+
+    for row, (label, matrix) in enumerate(chunk):
+        panels = [np.real(matrix), np.imag(matrix), np.abs(matrix), np.angle(matrix)]
+        cmaps = ["gray", "gray", "gray", "twilight"]
+        norms = [re_norm, im_norm, abs_norm, Normalize(vmin=-math.pi, vmax=math.pi)]
+        titles = ["Re", "Im", "|z|", "arg(z)"]
+        for col in range(4):
+            ax = fig.add_subplot(gs[row, col])
+            im = ax.imshow(panels[col], cmap=cmaps[col], norm=norms[col])
+            if row == 0:
+                ax.set_title(titles[col])
+            ax.set_xticks([])
+            ax.set_yticks([])
+            if col == 0:
+                ax.set_ylabel(label, fontsize=7)
+            if row == n_rows - 1:
+                cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
@@
 def render_multi_filter_frame_from_slices(
*** End Patch
