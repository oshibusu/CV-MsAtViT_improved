#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import loadmat
from PIL import Image


def load_pred(path: Path) -> np.ndarray:
    m = loadmat(path)
    for k in ('CV_MsAtViT', 'CV_MsAtViT_Full'):
        if k in m:
            return np.asarray(m[k])
    for k in m.keys():
        if not k.startswith('__'):
            return np.asarray(m[k])
    raise RuntimeError('no variable in pred mat')


def load_gt(path: Path) -> np.ndarray:
    m = loadmat(path)
    for k in m.keys():
        if not k.startswith('__'):
            return np.asarray(m[k])
    raise RuntimeError('no variable in gt mat')


def resize_label_nearest(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    th, tw = shape
    img = Image.fromarray(arr.astype(np.int32))
    img = img.resize((tw, th), resample=Image.NEAREST)
    return np.array(img).astype(arr.dtype)


def collapse_ober_pred(pred: np.ndarray) -> np.ndarray:
    # map 4->3, 5->3
    out = pred.copy()
    out[pred == 4] = 3
    out[pred == 5] = 3
    return out


def roll2d(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(a, dy, axis=0), dx, axis=1)


def score_oa(gt: np.ndarray, pred: np.ndarray) -> float:
    m = gt > 0
    if not np.any(m):
        return 0.0
    g = gt[m].astype(np.int64) - 1
    p = pred[m].astype(np.int64) - 1
    return float((g == p).mean())


def coarse_to_fine(gt: np.ndarray, pred: np.ndarray, max_shift: int = 80) -> Tuple[int, int, float]:
    best = (0, 0, score_oa(gt, pred))
    # coarse
    for step, rng in [(8, range(-max_shift, max_shift+1, 8)), (2, range(-16, 17, 2)), (1, range(-4, 5, 1))]:
        base_dy, base_dx, base_s = best
        for dy in rng:
            for dx in rng:
                ddy = base_dy + dy
                ddx = base_dx + dx
                s = score_oa(roll2d(gt, ddy, ddx), pred)
                if s > best[2]:
                    best = (ddy, ddx, s)
    return best


def main():
    ap = argparse.ArgumentParser(description='Estimate integer-pixel offset between GT and prediction')
    ap.add_argument('--pred', required=True, help='pred mat path')
    ap.add_argument('--gt', required=True, help='gt mat path')
    ap.add_argument('--collapse-ober', action='store_true', help='collapse ober pred classes 4,5 -> 3')
    ap.add_argument('--max-shift', type=int, default=80)
    args = ap.parse_args()

    pred = load_pred(Path(args.pred))
    gt = load_gt(Path(args.gt))
    if gt.shape != pred.shape:
        gt = resize_label_nearest(gt, pred.shape)
    if args.collapse_ober:
        pred = collapse_ober_pred(pred)

    dy, dx, s = coarse_to_fine(gt, pred, max_shift=args.max_shift)
    print({'dy': dy, 'dx': dx, 'OA_aligned': s})


if __name__ == '__main__':
    main()

