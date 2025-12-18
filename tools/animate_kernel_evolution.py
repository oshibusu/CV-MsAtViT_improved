#!/usr/bin/env python3
"""Generate GIF animations for all complex kernels across epochs."""
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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

from Load_Data import load_data
from SAR_utils import Standardize_data, createImageCubes, padWithZeros
from model_factory import build_msatvit

BRANCH_LAYERS = [
    "spatial_conv3d_block1",
    "polar_conv3d_block1",
    "joint_conv3d_block1",
]

PAULI_PATHS = {
    "FL_T": Path("Datasets/Flevoland/T3/PauliRGB.bmp"),
    "SF": Path("Datasets/san_francisco/T3/PauliRGB.bmp"),
    "ober": Path("Datasets/Oberpfaffenhofen/ESAR_Oberpfaffenhofen_T6/PauliRGB_T1.bmp"),
}


def dataset_tag(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def resolve_pauli_path(dataset: str, override: Optional[str]) -> Optional[Path]:
    if override:
        candidate = Path(override)
        if candidate.exists():
            return candidate
    default = PAULI_PATHS.get(dataset)
    if default is None:
        return None
    if default.exists():
        return default
    if default.is_dir():
        bmp = sorted(default.glob("PauliRGB*.bmp"))
        return bmp[0] if bmp else None
    return None


def load_pauli_array(dataset: str, override: Optional[str], target_shape: Tuple[int, int]) -> np.ndarray:
    path = resolve_pauli_path(dataset, override)
    if path and path.exists():
        img = Image.open(path).convert("RGB")
    else:
        img = Image.new("RGB", (target_shape[1], target_shape[0]), color=(0, 0, 0))
    if img.size != (target_shape[1], target_shape[0]):
        img = ImageOps.fit(img, (target_shape[1], target_shape[0]), method=Image.BILINEAR)
    return np.asarray(img)


def parse_args():
    p = argparse.ArgumentParser(description="Animate all kernel slices across epochs")
    p.add_argument("--dataset", default="SF", help="Dataset identifier used for training (e.g., SF, FL_T)")
    p.add_argument("--window-size", type=int, default=15, help="Patch window size used during training")
    p.add_argument(
        "--weights-pattern",
        default=None,
        help="Glob pattern for epoch weights (default: ckpt/CV_MsAtViT_<dataset>_epoch*.weights.h5)",
    )
    p.add_argument("--output-dir", default=None, help="Root directory for GIF outputs")
    p.add_argument("--frame-duration", type=int, default=400, help="GIF frame duration in ms")
    p.add_argument("--max-epochs", type=int, default=None, help="Upper bound on checkpoint count (optional)")
    p.add_argument(
        "--branches",
        default="",
        help="Comma-separated subset of branches to animate (default: all)",
    )
    p.add_argument("--heatmap-batch-size", type=int, default=128, help="Batch size when computing feature responses")
    p.add_argument(
        "--heatmap-sample-limit",
        type=int,
        default=0,
        help="Number of patches to use when estimating heatmaps (<=0 for all)",
    )
    p.add_argument("--topk", type=int, default=15, help="Number of strongest response locations to annotate")
    p.add_argument(
        "--pauli-path",
        default=None,
        help="Optional override for Pauli RGB image path",
    )
    p.add_argument(
        "--filters",
        default="",
        help="Filter indices to visualize (e.g., '0-3,5'). Empty for all",
    )
    p.add_argument(
        "--in-indices",
        default="",
        help="Input channel indices to visualize (e.g., '0,2'). Empty for all",
    )
    p.add_argument(
        "--depth-indices",
        default="",
        help="Depth indices/slices to visualize (e.g., '0-2'). Empty for all",
    )
    p.add_argument(
        "--mode",
        choices=["epoch", "batch"],
        default="epoch",
        help="Visualization mode: epoch checkpoints or recorded batch traces",
    )
    p.add_argument(
        "--batch-trace-dir",
        default=None,
        help="Directory containing batch trace outputs (default: ckpt/batch_traces/<dataset>)",
    )
    return p.parse_args()


def collect_weight_files(pattern: str, limit: Optional[int]) -> List[Tuple[str, str]]:
    def epoch_key(path: str) -> int:
        match = re.search(r"epoch(\d+)", os.path.basename(path))
        return int(match.group(1)) if match else -1

    files = sorted(glob.glob(pattern), key=lambda p: (epoch_key(p), p))
    if limit:
        files = files[:limit]
    labeled = []
    for pth in files:
        match = re.search(r"epoch(\d+)", os.path.basename(pth))
        label = match.group(1) if match else os.path.basename(pth)
        labeled.append((pth, label))
    return labeled


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


def combine_complex_kernel(weights: List[np.ndarray]) -> np.ndarray:
    if not weights:
        raise ValueError("Layer has no weights")
    kernel = weights[0]
    if len(weights) >= 2 and weights[0].shape == weights[1].shape and not np.iscomplexobj(weights[0]):
        kernel = weights[0] + 1j * weights[1]
    if not np.iscomplexobj(kernel):
        kernel = kernel.astype(np.complex64)
    return kernel


def prepare_heatmap_inputs(args):
    data, gt = load_data(args.dataset)
    data = Standardize_data(data)
    patches, centers = extract_patches_with_centers(data, gt, args.window_size)
    if args.heatmap_sample_limit is not None and args.heatmap_sample_limit > 0:
        limit = min(args.heatmap_sample_limit, patches.shape[0])
        patches = patches[:limit]
        centers = centers[:limit]
    input_shape = patches.shape[1:]
    image_shape = gt.shape
    pauli_array = load_pauli_array(args.dataset, args.pauli_path, image_shape)
    return patches, centers, input_shape, image_shape, pauli_array


def normalize_heatmap(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.size == 0:
        return arr
    vmin = float(arr.min())
    vmax = float(arr.max())
    if np.isclose(vmin, vmax):
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def find_top_coords(heatmap: np.ndarray, k: int) -> List[Tuple[int, int]]:
    if k <= 0:
        return []
    flat = heatmap.reshape(-1)
    count = min(k, flat.size)
    if count == 0:
        return []
    idx = np.argpartition(flat, -count)[-count:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    H, W = heatmap.shape
    coords = [(int(i // W), int(i % W)) for i in idx]
    return coords


def parse_index_spec(spec: str, max_len: int) -> List[int]:
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


def render_frame_with_heatmap(
    matrix: np.ndarray,
    heatmap: np.ndarray,
    pauli_array: np.ndarray,
    top_coords: List[Tuple[int, int]],
    title: str,
    subtitle: Optional[str] = None,
) -> Image.Image:
    real = np.real(matrix)
    imag = np.imag(matrix)
    magnitude = np.abs(matrix)
    phase = np.angle(matrix)
    heatmap_norm = normalize_heatmap(heatmap)

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1.2, 1.2])
    kernel_axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    plots = [
        (real, "Re", "gray", None, None),
        (imag, "Im", "gray", None, None),
        (magnitude, "|z|", "gray", None, None),
        (phase, "arg(z)", "twilight", -np.pi, np.pi),
    ]
    for ax, (data, subtitle, cmap, vmin, vmax) in zip(kernel_axes, plots):
        img = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(subtitle)
        ax.axis("off")
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.02)

    ax_heat = fig.add_subplot(gs[0, 2:])
    heat_img = ax_heat.imshow(heatmap_norm, cmap="hot", vmin=0.0, vmax=1.0)
    ax_heat.set_title("Response heatmap")
    ax_heat.axis("off")
    fig.colorbar(heat_img, ax=ax_heat, fraction=0.046, pad=0.02)

    ax_pauli = fig.add_subplot(gs[1, 2:])
    ax_pauli.imshow(pauli_array)
    ax_pauli.set_title("Pauli RGB")
    ax_pauli.axis("off")

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(top_coords))))
    for idx, (y, x) in enumerate(top_coords):
        color = colors[idx % len(colors)] if len(top_coords) > 0 else "cyan"
        circle_heat = Circle((x, y), radius=3, fill=False, edgecolor=color, linewidth=2)
        ax_heat.add_patch(circle_heat)

        circle_pauli = Circle((x, y), radius=6, fill=False, edgecolor=color, linewidth=2)
        ax_pauli.add_patch(circle_pauli)

    if subtitle:
        fig.suptitle(f"{title}\n{subtitle}")
    else:
        fig.suptitle(title)
    fig.tight_layout()
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


def compute_branch_heatmaps(
    branch_model: tf.keras.Model,
    patches: np.ndarray,
    centers: np.ndarray,
    image_shape: Tuple[int, int],
    batch_size: int,
) -> np.ndarray:
    if patches.size == 0:
        filter_count = int(branch_model.output_shape[-1])
        return np.zeros((filter_count,) + image_shape, dtype=np.float32)
    filter_count = int(branch_model.output_shape[-1])
    heatmaps = np.zeros((filter_count,) + image_shape, dtype=np.float32)
    counts = np.zeros(image_shape, dtype=np.float32)
    for start in range(0, patches.shape[0], batch_size):
        batch = patches[start : start + batch_size]
        feat = branch_model(batch, training=False)
        amp = tf.abs(feat).numpy()
        reduce_axes = tuple(range(1, amp.ndim - 1))
        summary = amp.max(axis=reduce_axes)
        for bi in range(summary.shape[0]):
            idx = start + bi
            if idx >= centers.shape[0]:
                break
            y, x = centers[idx]
            counts[y, x] += 1
            heatmaps[:, y, x] += summary[bi]
    counts[counts == 0] = 1.0
    heatmaps = heatmaps / counts
    return heatmaps


def animate_branch(
    branch_name: str,
    kernels: List[np.ndarray],
    heatmaps: List[np.ndarray],
    epoch_labels: List[str],
    pauli_array: np.ndarray,
    topk: int,
    filter_spec: str,
    in_spec: str,
    depth_spec: str,
    frame_duration: int,
    out_root: Path,
    metrics_list: Optional[List[Optional[Dict[str, float]]]] = None,
):
    if not kernels:
        return
    kD, kH, kW, in_ch, out_ch = kernels[0].shape
    filter_indices = parse_index_spec(filter_spec, out_ch)
    if not filter_indices:
        return
    in_indices = parse_index_spec(in_spec, in_ch)
    depth_indices = parse_index_spec(depth_spec, kW)
    for out_idx in filter_indices:
        for in_idx in in_indices:
            for depth_idx in depth_indices:
                frames = []
                for idx_frame, (kernel, label, heatmap_stack) in enumerate(
                    zip(kernels, epoch_labels, heatmaps)
                ):
                    vol = kernel[:, :, :, in_idx, out_idx]
                    depth = depth_idx % vol.shape[2]
                    matrix = vol[:, :, depth]
                    filter_heatmap = heatmap_stack[out_idx]
                    top_coords = find_top_coords(filter_heatmap, topk)
                    title = (
                        f"{branch_name} | filter {out_idx} | in {in_idx} | depth {depth} | {label}"
                    )
                    subtitle = None
                    if metrics_list and idx_frame < len(metrics_list):
                        metrics = metrics_list[idx_frame] or {}
                        loss = metrics.get("loss")
                        acc = metrics.get("accuracy")
                        parts = []
                        if loss is not None and not math.isnan(loss):
                            parts.append(f"loss={loss:.4f}")
                        if acc is not None and not math.isnan(acc):
                            parts.append(f"acc={acc:.4f}")
                        if parts:
                            subtitle = ", ".join(parts)
                    frames.append(
                        render_frame_with_heatmap(
                            matrix,
                            filter_heatmap,
                            pauli_array,
                            top_coords,
                            title,
                            subtitle=subtitle,
                        )
                    )
                out_path = (
                    out_root
                    / branch_name
                    / f"filter{out_idx:02d}"
                    / f"in{in_idx:02d}_depth{depth_idx:02d}.gif"
                )
                save_gif(frames, out_path, frame_duration)
                print("Saved", out_path)


def main():
    args = parse_args()
    tag = dataset_tag(args.dataset)
    if args.mode == "batch":
        run_batch_mode(args, tag)
        return

    pattern = args.weights_pattern or os.path.join("ckpt", f"CV_MsAtViT_{tag}_epoch*.weights.h5")
    weights = collect_weight_files(pattern, args.max_epochs)
    if len(weights) < 2:
        raise SystemExit(
            f"Found {len(weights)} weight file(s) for pattern {pattern}. Need at least 2 checkpoints."
        )

    branches = [b.strip() for b in args.branches.split(",") if b.strip()]
    if branches:
        invalid = [b for b in branches if b not in BRANCH_LAYERS]
        if invalid:
            raise SystemExit(f"Unknown branches: {invalid}. Valid: {BRANCH_LAYERS}")
    else:
        branches = BRANCH_LAYERS

    data, gt = load_data(args.dataset)
    data = Standardize_data(data)
    patches, centers = extract_patches_with_centers(data, gt, args.window_size)
    if args.heatmap_sample_limit is not None and args.heatmap_sample_limit > 0:
        limit = min(args.heatmap_sample_limit, patches.shape[0])
        patches = patches[:limit]
        centers = centers[:limit]
    input_shape = patches.shape[1:]
    image_shape = gt.shape
    pauli_array = load_pauli_array(args.dataset, args.pauli_path, image_shape)

    model = build_msatvit(input_shape=input_shape, dataset=args.dataset, window_size=args.window_size)
    branch_models = {
        b: tf.keras.Model(inputs=model.input, outputs=model.get_layer(b).output)
        for b in branches
    }

    kernel_history: Dict[str, List[np.ndarray]] = {b: [] for b in branches}
    heatmap_history: Dict[str, List[np.ndarray]] = {b: [] for b in branches}

    for wf, label in weights:
        print(f"Loading weights {wf}")
        model.load_weights(wf)
        for branch in branches:
            layer = model.get_layer(branch)
            kernel_history[branch].append(combine_complex_kernel(layer.get_weights()))
            heatmaps = compute_branch_heatmaps(
                branch_models[branch],
                patches,
                centers,
                image_shape,
                args.heatmap_batch_size,
            )
            heatmap_history[branch].append(heatmaps)

    out_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path("results/analysis") / tag / "animations"
    )

    epoch_labels = [label for _, label in weights]
    for branch in branches:
        animate_branch(
            branch,
            kernel_history[branch],
            heatmap_history[branch],
            epoch_labels,
            pauli_array,
            args.topk,
            args.filters,
            args.in_indices,
            args.depth_indices,
            args.frame_duration,
            out_root,
            metrics_list=None,
        )


def run_batch_mode(args, tag):
    trace_root = (
        Path(args.batch_trace_dir)
        if args.batch_trace_dir
        else Path("ckpt") / "batch_traces" / tag
    )
    if not trace_root.exists():
        raise SystemExit(f"Batch trace directory not found: {trace_root}")
    snapshot_dirs = sorted(
        [d for d in trace_root.glob("epoch*_batch*") if d.is_dir()],
        key=batch_dir_sort_key,
    )
    if not snapshot_dirs:
        raise SystemExit(f"No batch trace folders under {trace_root}")

    snapshots = collect_batch_snapshots(snapshot_dirs, args.max_epochs)
    if not snapshots:
        raise SystemExit("No usable batch weight files found")

    branches = [b.strip() for b in args.branches.split(",") if b.strip()]
    if branches:
        invalid = [b for b in branches if b not in BRANCH_LAYERS]
        if invalid:
            raise SystemExit(f"Unknown branches: {invalid}. Valid: {BRANCH_LAYERS}")
    else:
        branches = BRANCH_LAYERS

    patches, centers, input_shape, image_shape, pauli_array = prepare_heatmap_inputs(args)
    model = build_msatvit(input_shape=input_shape, dataset=args.dataset, window_size=args.window_size)
    branch_models = {
        b: tf.keras.Model(inputs=model.input, outputs=model.get_layer(b).output)
        for b in branches
    }

    kernel_history: Dict[str, List[np.ndarray]] = {b: [] for b in branches}
    heatmap_history: Dict[str, List[np.ndarray]] = {b: [] for b in branches}
    labels: List[str] = []
    metrics_records: List[Optional[Dict[str, float]]] = []

    for snapshot in snapshots:
        weights_path = snapshot["weights"]
        label = snapshot["label"]
        metrics = snapshot.get("metrics") if isinstance(snapshot, dict) else None
        print(f"Loading batch weights {weights_path}")
        model.load_weights(weights_path)
        for branch in branches:
            layer = model.get_layer(branch)
            kernel_history[branch].append(combine_complex_kernel(layer.get_weights()))
            heatmaps = compute_branch_heatmaps(
                branch_models[branch],
                patches,
                centers,
                image_shape,
                args.heatmap_batch_size,
            )
            heatmap_history[branch].append(heatmaps)
        labels.append(label)
        metrics_records.append(metrics if isinstance(metrics, dict) else None)

    out_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path("results/analysis") / tag / "batch_animations"
    )

    for branch in branches:
        animate_branch(
            branch,
            kernel_history[branch],
            heatmap_history[branch],
            labels,
            pauli_array,
            args.topk,
            args.filters,
            args.in_indices,
            args.depth_indices,
            args.frame_duration,
            out_root,
            metrics_records,
        )


def batch_dir_sort_key(path: Path):
    name = path.name
    match_epoch = re.search(r"epoch(\d+)", name)
    epoch = int(match_epoch.group(1)) if match_epoch else 0
    match_batch = re.search(r"batch(\d+)", name)
    batch_idx = int(match_batch.group(1)) if match_batch else 0
    pre_flag = 0 if name.endswith("pre") else 1
    return (epoch, batch_idx, pre_flag, name)


def parse_batch_progress(batch_dir: Path) -> Optional[Dict[str, int]]:
    log_file = batch_dir / "progress.txt"
    if not log_file.exists():
        return None
    try:
        text = log_file.read_text().strip()
        match = re.search(r"start=(\d+),end=(\d+)", text)
        if match:
            return {"start": int(match.group(1)), "end": int(match.group(2))}
    except Exception:
        return None
    return None

def collect_batch_snapshots(
    batch_dirs: List[Path], limit: Optional[int]
) -> List[Dict[str, object]]:
    snapshots: List[Dict[str, object]] = []
    for batch_dir in batch_dirs:
        weights_path = batch_dir / "weights.h5"
        if not weights_path.exists():
            continue
        progress = parse_batch_progress(batch_dir)
        label = batch_dir.name
        if progress:
            label = f"{label} (patch {progress['start']}-{progress['end']})"
        metrics_path = batch_dir / "metrics.json"
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
        loss_val = _to_float(metrics.get("loss")) if isinstance(metrics, dict) else None
        acc_val = _to_float(metrics.get("accuracy")) if isinstance(metrics, dict) else None
        snapshots.append(
            {
                "weights": weights_path,
                "label": label,
                "metrics": {"loss": loss_val, "accuracy": acc_val},
            }
        )
        if limit and len(snapshots) >= limit:
            break
    return snapshots
if __name__ == "__main__":
    main()
