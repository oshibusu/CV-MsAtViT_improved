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
                filters_per_frame=args.frame_filters,
                branch_name=branch,
            )
            frames.append(frame)
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
                    branch_name=branch,
                )
                frames.append(frame)
            if frames:
                out_path = out_root / f"{branch}_multi.gif"
                save_gif(frames, out_path, args.frame_duration)
                print("Saved", out_path)
 
def render_multi_filter_frame(
    kernel: np.ndarray,
    filter_indices: List[int],
    in_idx: int,
    depth_indices: List[int],
    title: str,
    subtitle: Optional[str] = None,
    filters_per_frame: int = 24,
    branch_name: Optional[str] = None,
):
    slices: List[Tuple[str, np.ndarray]] = []
    for filt_idx in filter_indices:
        if filt_idx >= kernel.shape[-1]:
            continue
        vol = kernel[:, :, :, in_idx, filt_idx]
        depth_len = vol.shape[2]
        for depth in depth_indices:
            d = depth % depth_len
            prefix = f"{branch_name} " if branch_name else ""
            label = f"{prefix}f{filt_idx} d{depth}"
            slices.append((label.strip(), vol[:, :, d]))
    return _render_slices_grid(
        slices,
        title=title,
        subtitle=subtitle,
        filters_per_frame=filters_per_frame,
    )


def render_multi_filter_frame_from_slices(
    slices: List[Tuple[str, np.ndarray]],
    title: str,
    subtitle: Optional[str] = None,
    filters_per_frame: int = 24,
):
    return _render_slices_grid(
        slices,
        title=title,
        subtitle=subtitle,
        filters_per_frame=filters_per_frame,
    )


def _render_slices_grid(
    slices: List[Tuple[str, np.ndarray]],
    title: str,
    subtitle: Optional[str],
    filters_per_frame: int,
) -> Image.Image:
    if not slices:
        raise ValueError("No slices to render")
    max_rows = len(slices)
    if filters_per_frame and filters_per_frame > 0:
        max_rows = min(max_rows, filters_per_frame)
    chunk = slices[:max_rows]
    n_rows = len(chunk)
    fig = plt.figure(figsize=(12, max(2, n_rows * 0.65)))
    gs = GridSpec(n_rows, 4, figure=fig, wspace=0.3, hspace=0.45)

    # Re/Im は共通でゼロ対称の正規化、|z| は独立したパーセンタイルベースの正規化を行う。
    re_vals = np.concatenate([np.real(m).ravel() for _, m in chunk])
    im_vals = np.concatenate([np.imag(m).ravel() for _, m in chunk])
    abs_vals = np.concatenate([np.abs(m).ravel() for _, m in chunk])
    max_abs = max(
        float(np.nanmax(np.abs(re_vals))),
        float(np.nanmax(np.abs(im_vals))),
        1e-9,
    )
    reim_norm = Normalize(vmin=-max_abs, vmax=max_abs)

    valid_abs = abs_vals[np.isfinite(abs_vals)]
    if valid_abs.size:
        abs_lo, abs_hi = np.percentile(valid_abs, [2, 98])
    else:
        abs_lo, abs_hi = 0.0, 1.0
    if abs_hi - abs_lo < 1e-9:
        eps = abs_hi * 1e-3 if abs_hi != 0 else 1e-6
        abs_lo -= eps
        abs_hi += eps
    abs_norm = Normalize(vmin=float(abs_lo), vmax=float(abs_hi))
    arg_norm = Normalize(vmin=-math.pi, vmax=math.pi)

    column_specs = [
        ("Re", np.real, "gray", reim_norm),
        ("Im", np.imag, "gray", reim_norm),
        ("|z|", np.abs, "gray", abs_norm),
        ("arg(z)", np.angle, "twilight", arg_norm),
    ]

    for row, (label, matrix) in enumerate(chunk):
        for col, (col_title, extractor, cmap, norm) in enumerate(column_specs):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(extractor(matrix), cmap=cmap, norm=norm)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=7)
            if row == 0:
                ax.set_title(col_title)
                cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
                cbar.ax.tick_params(labelsize=6)
    caption = title if not subtitle else f"{title}\n{subtitle}"
    fig.suptitle(caption, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_gif(frames: List[Image.Image], output_path: Path, duration: int):
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )


def batch_dir_sort_key(path: Path):
    name = path.name
    match_epoch = re.search(r"epoch(\d+)", name)
    epoch = int(match_epoch.group(1)) if match_epoch else 0
    match_batch = re.search(r"batch(\d+)", name)
    batch_idx = int(match_batch.group(1)) if match_batch else 0
    pre_flag = 0 if name.endswith("pre") else 1
    return (epoch, batch_idx, pre_flag, name)


def main():
    args = parse_args()
    tag = dataset_tag(args.dataset)
    if args.mode == "batch":
        run_batch_mode(args, tag)
    else:
        run_epoch_mode(args, tag)


if __name__ == "__main__":
    main()
