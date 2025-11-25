#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image, ImageCms


# Dataset metadata (class namesはSAR_utils.targetに準拠)
CLASS_INFO: Dict[str, Dict] = {
    "FL_T": {
        "n": 15,
        "names": [
            "Unassigned", "Water", "Forest", "Lucerne", "Grass", "Rapeseed",
            "Beet", "Potatoes", "Peas", "Stem Beans", "Bare Soil",
            "Wheat", "Wheat 2", "Wheat 3", "Barley", "Buildings",
        ],
        "gt": Path("Datasets/Flevoland/Flevoland_gt.mat"),
    },
    "FL_C": {  # 参照用（実体はFL_Tと同じGT・クラス）
        "n": 15,
        "names": [
            "Unassigned", "Water", "Forest", "Lucerne", "Grass", "Rapeseed",
            "Beet", "Potatoes", "Peas", "Stem Beans", "Bare Soil",
            "Wheat", "Wheat 2", "Wheat 3", "Barley", "Buildings",
        ],
        "gt": Path("Datasets/Flevoland/Flevoland_gt.mat"),
    },
    "SF": {
        "n": 5,
        "names": [
            "Unassigned", "Bare Soil", "Mountain", "Water", "Urban", "Vegetation",
        ],
        "gt": Path("Datasets/san_francisco/SanFrancisco_gt.mat"),
    },
    "ober": {
        "n": 3,
        "names": ["Unassigned", "Class1", "Class2", "Class3"],
        "gt": Path("Datasets/Oberpfaffenhofen/Oberpfaffenhofen_gt.mat"),
    },
}


def guess_dataset_from_filename(p: Path) -> Optional[str]:
    name = p.name
    if "SF" in name:
        return "SF"
    if "FL_T" in name or "FL_C" in name:
        # 優先はFL_T、なければFL_C
        return "FL_T" if "FL_T" in name else "FL_C"
    if "ober" in name.lower():
        return "ober"
    return None


def load_pred_mat(path: Path) -> Tuple[np.ndarray, str]:
    m = loadmat(path)
    keys = [k for k in m.keys() if not k.startswith("__")]
    if not keys:
        raise ValueError(f"No array variables found in {path}")
    # 既定の変数名（CV_MsAtViT または CV_MsAtViT_Full）が想定
    preferred = [k for k in keys if k in ("CV_MsAtViT", "CV_MsAtViT_Full")]
    var = preferred[0] if preferred else keys[0]
    arr = np.asarray(m[var])
    if arr.ndim != 2:
        # 2次元に限定
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Unsupported array shape {arr.shape} in {path}")
    return arr, var


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict:
    valid = gt > 0
    if not np.any(valid):
        return {}
    p = (pred[valid].astype(np.int64) - 1).ravel()
    g = (gt[valid].astype(np.int64) - 1).ravel()
    cm = confusion_matrix(g, p)
    oa = float(accuracy_score(g, p))
    # 各クラス精度
    with np.errstate(divide='ignore', invalid='ignore'):
        each = np.diag(cm) / cm.sum(axis=1)
    each = np.nan_to_num(each)
    aa = float(each.mean())
    kap = float(cohen_kappa_score(g, p))
    return {
        "OA": oa,
        "AA": aa,
        "Kappa": kap,
        "confusion": cm.tolist(),
        "each_class_acc": each.tolist(),
    }


def make_palette(n_classes_plus_bg: int) -> ListedColormap:
    # 0は背景色（黒）。以降はタブカラーを循環
    tab = plt.get_cmap('tab20').colors
    colors = [(0, 0, 0, 1.0)]
    for i in range(1, n_classes_plus_bg):
        colors.append(tab[(i - 1) % len(tab)])
    return ListedColormap(colors)


def save_label_png(arr: np.ndarray, out_png: Path, n_classes: int, title: Optional[str] = None):
    cmap = make_palette(n_classes + 1)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=n_classes)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Class ID')
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    # Enforce: 8-bit sRGB, RGB full color, no alpha, non-interlaced
    try:
        im = Image.open(out_png)
        im = im.convert('RGB')  # drop alpha/palette
        try:
            srgb = ImageCms.createProfile('sRGB')
            im.save(out_png, format='PNG', icc_profile=srgb.tobytes(), optimize=True, interlace=0)
        except Exception:
            im.save(out_png, format='PNG', optimize=True, interlace=0)
    except Exception as e:
        print(f"[WARN] PNG postprocess failed for {out_png}: {e}")


def resize_label_nearest(label: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """NearestでGTなどのラベル配列をリサイズする。
    target_shape: (H, W)
    """
    th, tw = target_shape
    img = Image.fromarray(label.astype(np.int32))
    img = img.resize((tw, th), resample=Image.NEAREST)
    out = np.array(img)
    return out.astype(label.dtype)


def shift_label_pad(label: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """ゼロ埋めで平行移動（ラベル0=未割当）。wrapしない。
    dy>0で下へ、dx>0で右へ移動。
    """
    h, w = label.shape
    out = np.zeros_like(label)
    ys = max(0, dy)
    xs = max(0, dx)
    ye = h + min(0, dy)
    xe = w + min(0, dx)
    out[ys:ye, xs:xe] = label[ys-dy:ye-dy, xs-dx:xe-dx]
    return out


def collapse_ober_pred(pred: np.ndarray) -> np.ndarray:
    out = pred.copy()
    out[pred == 4] = 3
    out[pred == 5] = 3
    return out


def main():
    ap = argparse.ArgumentParser(description="Analyze .mat preds and visualize label maps")
    ap.add_argument("inputs", nargs="+", help=".mat files or directories containing them")
    ap.add_argument("--outdir", default="results/vis", help="output directory for images")
    ap.add_argument("--metricsdir", default="results/metrics", help="output directory for metrics")
    ap.add_argument("--align-gt", choices=["none", "crop", "nearest"], default="nearest",
                    help="GTと予測の形状が異なる場合の整合方法")
    ap.add_argument("--gt-offset", default=None, help="'dy,dx' 形式でGTを平行移動（ゼロ埋め）")
    ap.add_argument("--ober-collapse", action="store_true", help="oberの予測クラス(4,5)を3に統合")
    args = ap.parse_args()

    # 収集
    mats = []
    for s in args.inputs:
        p = Path(s)
        if p.is_dir():
            mats.extend(sorted(p.glob("*.mat")))
        elif p.suffix.lower() == ".mat":
            mats.append(p)
    mats = sorted(set(mats))
    if not mats:
        print("No .mat files found.")
        return

    outdir = Path(args.outdir)
    mdir = Path(args.metricsdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)

    summary = []
    for mp in mats:
        try:
            arr, var = load_pred_mat(mp)
        except Exception as e:
            print(f"[WARN] {mp}: {e}")
            continue

        ds = guess_dataset_from_filename(mp) or "unknown"
        info = CLASS_INFO.get(ds)
        n_classes = info["n"] if info else int(arr.max())  # フォールバック

        # oberのクラス統合（評価整合）
        if args.ober_collapse and ds == 'ober':
            arr = collapse_ober_pred(arr)

        # 保存（ファイル名由来の短いラベル）
        tag = mp.stem.replace("CV_MsAtViT_", "")
        title = f"{tag} ({var})"
        out_png = outdir / f"{tag}.png"
        save_label_png(arr, out_png, n_classes=n_classes, title=title)

        # GTがあれば指標
        metrics = {}
        if info and info.get("gt") and info["gt"].exists():
            try:
                gm = loadmat(info["gt"])
                gkeys = [k for k in gm.keys() if not k.startswith("__")]
                gvar = gkeys[0]
                gt = np.asarray(gm[gvar])
                # 形状が合わない場合はスキップ
                if gt.shape != arr.shape:
                    if args.align_gt == "nearest":
                        gt = resize_label_nearest(gt, arr.shape)
                        print(f"[INFO] Resampled GT to {arr.shape} for {mp.name} (nearest)")
                    elif args.align_gt == "crop":
                        h = min(gt.shape[0], arr.shape[0])
                        w = min(gt.shape[1], arr.shape[1])
                        gt = gt[:h, :w]
                        arr = arr[:h, :w]
                        print(f"[INFO] Cropped GT/Pred to {gt.shape} for {mp.name}")
                    else:
                        print(f"[WARN] GT shape mismatch for {mp.name}: pred {arr.shape} vs gt {gt.shape}")
                # 任意オフセット
                if args.gt_offset:
                    dy, dx = map(int, args.gt_offset.split(','))
                    gt = shift_label_pad(gt, dy, dx)
                    print(f"[INFO] Shifted GT by (dy,dx)=({dy},{dx}) for {mp.name}")
                if gt.shape == arr.shape:
                    metrics = compute_metrics(arr, gt)
            except Exception as e:
                print(f"[WARN] failed to load GT for {ds}: {e}")

        # 基本統計
        vals, cnt = np.unique(arr, return_counts=True)
        stats = {
            "file": str(mp),
            "variable": var,
            "dataset": ds,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "unique": {int(v): int(c) for v, c in zip(vals, cnt)},
            "image": str(out_png),
            "metrics": metrics,
        }
        summary.append(stats)

        # 個別メトリクスの保存
        base = mdir / f"{tag}.json"
        with base.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"[OK] {mp.name}: saved {out_png} and {base}")

    # 総合サマリ
    with (mdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[DONE] Wrote {len(summary)} entries to {mdir/'summary.json'}")


if __name__ == "__main__":
    main()
