#!/usr/bin/env python3
"""Generate GIF animations for all complex kernels across epochs."""
import argparse
import glob
import io
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
import numpy as np
from PIL import Image

from Load_Data import load_data
from SAR_utils import Standardize_data, createImageCubes
from model_factory import build_msatvit

BRANCH_LAYERS = [
    "spatial_conv3d_block1",
    "polar_conv3d_block1",
    "joint_conv3d_block1",
]


def dataset_tag(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


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


def ensure_input_shape(dataset: str, window_size: int):
    data, gt = load_data(dataset)
    data = Standardize_data(data)
    X_coh, _ = createImageCubes(data, gt, window_size)
    X_coh = np.expand_dims(X_coh, axis=4)
    shape = X_coh.shape[1:]
    del X_coh
    return shape


def combine_complex_kernel(weights: List[np.ndarray]) -> np.ndarray:
    if not weights:
        raise ValueError("Layer has no weights")
    kernel = weights[0]
    if len(weights) >= 2 and weights[0].shape == weights[1].shape and not np.iscomplexobj(weights[0]):
        kernel = weights[0] + 1j * weights[1]
    if not np.iscomplexobj(kernel):
        kernel = kernel.astype(np.complex64)
    return kernel


def render_frame(matrix: np.ndarray, title: str) -> Image.Image:
    real = np.real(matrix)
    imag = np.imag(matrix)
    magnitude = np.abs(matrix)
    phase = np.angle(matrix)

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    axes = axes.ravel()
    plots = [
        (real, "Re", "gray", None, None),
        (imag, "Im", "gray", None, None),
        (magnitude, "|z|", "gray", None, None),
        (phase, "arg(z)", "twilight", -np.pi, np.pi),
    ]
    for ax, (data, subtitle, cmap, vmin, vmax) in zip(axes, plots):
        img = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(subtitle)
        ax.axis("off")
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
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


def animate_branch(
    branch_name: str,
    kernels: List[np.ndarray],
    epoch_labels: List[str],
    frame_duration: int,
    out_root: Path,
):
    if not kernels:
        return
    kD, kH, kW, in_ch, out_ch = kernels[0].shape
    for out_idx in range(out_ch):
        for in_idx in range(in_ch):
            for depth_idx in range(kW):
                frames = []
                for kernel, label in zip(kernels, epoch_labels):
                    vol = kernel[:, :, :, in_idx, out_idx]
                    depth = depth_idx % vol.shape[2]
                    matrix = vol[:, :, depth]
                    title = (
                        f"{branch_name} | filter {out_idx} | in {in_idx} | depth {depth} | epoch {label}"
                    )
                    frames.append(render_frame(matrix, title))
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

    input_shape = ensure_input_shape(args.dataset, args.window_size)
    model = build_msatvit(input_shape=input_shape, dataset=args.dataset, window_size=args.window_size)

    kernel_history: Dict[str, List[np.ndarray]] = {b: [] for b in branches}

    for wf, label in weights:
        print(f"Loading weights {wf}")
        model.load_weights(wf)
        for branch in branches:
            layer = model.get_layer(branch)
            kernel_history[branch].append(combine_complex_kernel(layer.get_weights()))

    out_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path("results/analysis") / tag / "animations"
    )

    for branch in branches:
        animate_branch(
            branch,
            kernel_history[branch],
            [label for _, label in weights],
            args.frame_duration,
            out_root,
        )


if __name__ == "__main__":
    main()
