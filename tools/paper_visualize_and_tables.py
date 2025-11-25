#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image, ImageCms
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score


# --- 定義: データセットごとのクラス名と色（論文図に合わせて近似色を指定） ---
# 0はUnassigned（黒）で固定。以降はクラスID=1..n。

Color = Tuple[float, float, float]

def hex_to_rgb01(h: str) -> Color:
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))  # type: ignore


DATASETS: Dict[str, Dict] = {
    # Flevoland: 15 classes
    "FL_T": {
        "classes": [
            "Unassigned", "Water", "Forest", "Lucerne", "Grass", "Rapeseed",
            "Beet", "Potatoes", "Peas", "Stem Beans", "Bare Soil",
            "Wheat", "Wheat 2", "Wheat 3", "Barley", "Buildings",
        ],
        "classes_ja": [
            "未割当", "水域", "森林", "ルーサン", "草地", "菜種",
            "てん菜", "ジャガイモ", "エンドウ", "いんげん(茎)", "裸地",
            "小麦", "小麦2", "小麦3", "大麦", "建物",
        ],
        # 論文図の色に近い配色（必要に応じ調整可）
        "colors": [
            "#000000",  # 0 Unassigned
            "#e41a1c",  # 1 Water (paper figure uses red surround region)
            "#ff7f00",  # 2 Forest (orange)
            "#ffff33",  # 3 Lucerne (yellow)
            "#a6d854",  # 4 Grass (yellow-green)
            "#00c853",  # 5 Rapeseed (vivid green)
            "#1b9e77",  # 6 Beet (greenish)
            "#66a61e",  # 7 Potatoes (green)
            "#1f78b4",  # 8 Peas (blue)
            "#33a02c",  # 9 Stem Beans (green-blue)
            "#377eb8",  # 10 Bare Soil (blue, per figure palette)
            "#984ea3",  # 11 Wheat (purple)
            "#8da0cb",  # 12 Wheat 2 (light purple)
            "#b07aa1",  # 13 Wheat 3 (violet)
            "#a65628",  # 14 Barley (brown)
            "#ff1493",  # 15 Buildings (magenta/pink)
        ],
        "gt": Path("Datasets/Flevoland/Flevoland_gt.mat"),
        "pauli": Path("Datasets/Flevoland/T3/PauliRGB.bmp"),
    },
    # San Francisco: 5 classes
    "SF": {
        "classes": [
            "Unassigned", "Bare Soil", "Mountain", "Water", "Urban", "Vegetation",
        ],
        "classes_ja": [
            "未割当", "裸地", "山地", "水域", "都市", "植生",
        ],
        # 図に合わせた近似色（海=青、都市=紫、植生=緑、裸地=赤、山=黄）
        "colors": [
            "#000000",  # 0 Unassigned
            "#e41a1c",  # 1 Bare Soil (red)
            "#ffff33",  # 2 Mountain (yellow)
            "#377eb8",  # 3 Water (blue)
            "#984ea3",  # 4 Urban (purple)
            "#4daf4a",  # 5 Vegetation (green)
        ],
        "gt": Path("Datasets/san_francisco/SanFrancisco_gt.mat"),
        "pauli": Path("Datasets/san_francisco/T3/PauliRGB.bmp"),
    },
    # Oberpfaffenhofen: 3 classes
    "ober": {
        "classes": ["Unassigned", "Build-Up Areas", "Wood Land", "Open Areas"],
        "classes_ja": ["未割当", "市街地", "森林", "開放地"],
        "colors": [
            "#000000",  # 0 Unassigned
            "#e41a1c",  # 1 Build-Up Areas (red)
            "#4daf4a",  # 2 Wood Land (green)
            "#ffff33",  # 3 Open Areas (yellow)
        ],
        "gt": Path("Datasets/Oberpfaffenhofen/Oberpfaffenhofen_gt.mat"),
        # Oberはフォルダ内の PauliRGB_*.bmp のいずれかを利用（T1優先）
        "pauli": Path("Datasets/Oberpfaffenhofen/ESAR_Oberpfaffenhofen_T6"),
    },
}

# 表示用データセット名（英語）
DISPLAY_NAME_EN: Dict[str, str] = {
    'FL_T': 'Flevoland',
    'SF': 'San Francisco',
    'ober': 'Oberpfaffenhofen',
}


def make_cmap(hex_list: List[str]) -> ListedColormap:
    return ListedColormap([hex_to_rgb01(h) for h in hex_list])


def detect_dataset_from_name(name: str) -> Optional[str]:
    if "SF" in name:
        return "SF"
    if "FL_T" in name or "FL_C" in name:
        return "FL_T"  # 可視化では同一クラス構成
    if "ober" in name.lower():
        return "ober"
    return None


def load_pred(path: Path) -> Tuple[np.ndarray, str]:
    m = loadmat(path)
    keys = [k for k in m.keys() if not k.startswith("__")]
    if not keys:
        raise RuntimeError(f"No variables in {path}")
    prefer = [k for k in keys if k in ("CV_MsAtViT", "CV_MsAtViT_Full")]
    var = prefer[0] if prefer else keys[0]
    arr = np.asarray(m[var])
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise RuntimeError(f"Unsupported shape {arr.shape} in {path}")
    return arr, var


def load_gt(gt_path: Path) -> Tuple[np.ndarray, str]:
    m = loadmat(gt_path)
    keys = [k for k in m.keys() if not k.startswith("__")]
    var = keys[0]
    arr = np.asarray(m[var])
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    return arr, var


def align_arrays(gt: np.ndarray, pred: np.ndarray, how: str) -> Tuple[np.ndarray, np.ndarray]:
    """GTと予測の形状を整合。how: none|crop|nearest"""
    if gt.shape == pred.shape or how == 'none':
        return gt, pred
    if how == 'crop':
        h = min(gt.shape[0], pred.shape[0])
        w = min(gt.shape[1], pred.shape[1])
        return gt[:h, :w], pred[:h, :w]
    # nearest: GTを予測サイズへ最近傍リサイズ
    from PIL import Image
    th, tw = pred.shape
    img = Image.fromarray(gt.astype(np.int32))
    img = img.resize((tw, th), resample=Image.NEAREST)
    return np.array(img).astype(gt.dtype), pred


def shift_label_pad(label: np.ndarray, dy: int, dx: int) -> np.ndarray:
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


def find_pauli_image(pauli_entry: Path) -> Optional[Path]:
    """pauli_entryがファイルならそのまま、フォルダならPauliRGB*.bmpを探索。
    Oberのように T1/T2 がある場合は T1 を優先。
    """
    if pauli_entry.is_file():
        return pauli_entry if pauli_entry.exists() else None
    if pauli_entry.is_dir():
        # T1優先、なければT2、そのほかPauliRGB*.bmp
        t1 = sorted(pauli_entry.glob('PauliRGB*T1*.bmp'))
        t2 = sorted(pauli_entry.glob('PauliRGB*T2*.bmp'))
        anyp = sorted(pauli_entry.glob('PauliRGB*.bmp'))
        for lst in (t1, t2, anyp):
            if lst:
                return lst[0]
    return None


def compute_class_acc(pred: np.ndarray, gt: np.ndarray, n_cls: int) -> Tuple[np.ndarray, float, float, float]:
    valid = gt > 0
    p = (pred[valid] - 1).astype(np.int64).ravel()
    g = (gt[valid] - 1).astype(np.int64).ravel()
    cm = confusion_matrix(g, p, labels=list(range(n_cls)))
    with np.errstate(divide='ignore', invalid='ignore'):
        per_cls = np.diag(cm) / cm.sum(axis=1)
    per_cls = np.nan_to_num(per_cls)
    oa = accuracy_score(g, p)
    aa = float(per_cls.mean())
    kap = cohen_kappa_score(g, p)
    return per_cls, oa, aa, kap


def draw_triptych(pauli_img: Optional[Path], gt: np.ndarray, pred: np.ndarray, cmap: ListedColormap, n_cls: int,
                  titles: Tuple[str, str, str], out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    # Pauli
    if pauli_img and pauli_img.exists():
        img = Image.open(pauli_img)
        axes[0].imshow(img)
    else:
        axes[0].imshow(np.zeros((*gt.shape, 3), dtype=np.uint8))
        axes[0].text(0.5, 0.5, 'Pauli RGB not found', color='w', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_facecolor('k')
    axes[0].set_title(titles[0])
    axes[0].set_axis_off()
    # GT
    im1 = axes[1].imshow(gt, cmap=cmap, vmin=0, vmax=n_cls)
    axes[1].set_title(titles[1])
    axes[1].set_axis_off()
    # Pred
    im2 = axes[2].imshow(pred, cmap=cmap, vmin=0, vmax=n_cls)
    axes[2].set_title(titles[2])
    axes[2].set_axis_off()
    # カラーバー
    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label('Class ID')
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    # Enforce PNG properties (8-bit sRGB RGB, no alpha, non-interlaced)
    try:
        im = Image.open(out_path)
        im = im.convert('RGB')
        try:
            srgb = ImageCms.createProfile('sRGB')
            im.save(out_path, format='PNG', icc_profile=srgb.tobytes(), optimize=True, interlace=0)
        except Exception:
            im.save(out_path, format='PNG', optimize=True, interlace=0)
    except Exception as e:
        print(f"[WARN] PNG postprocess failed for {out_path}: {e}")


def finalize_png(path: Path):
    try:
        im = Image.open(path)
        im = im.convert('RGB')
        try:
            srgb = ImageCms.createProfile('sRGB')
            im.save(path, format='PNG', icc_profile=srgb.tobytes(), optimize=True, interlace=0)
        except Exception:
            im.save(path, format='PNG', optimize=True, interlace=0)
    except Exception as e:
        print(f"[WARN] PNG postprocess failed for {path}: {e}")


def draw_pauli_gt_with_legend(pauli_img: Optional[Path], gt: np.ndarray, cmap: ListedColormap, class_names: List[str],
                              colors_hex: List[str], out_path: Path, lang: str = 'en', dataset_display: str = ''):
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])
    # Pauli
    if pauli_img and pauli_img.exists():
        img = Image.open(pauli_img)
        ax0.imshow(img)
    else:
        ax0.imshow(np.zeros((*gt.shape, 3), dtype=np.uint8))
        ax0.text(0.5, 0.5, 'Pauli RGB not found', color='w', ha='center', va='center', transform=ax0.transAxes)
        ax0.set_facecolor('k')
    title_ds = dataset_display if dataset_display else 'Dataset'
    ax0.set_title(f'{title_ds} PauliRGB')
    ax0.set_axis_off()
    # GT
    n_cls = len(class_names) - 1
    ax1.imshow(gt, cmap=cmap, vmin=0, vmax=n_cls)
    ax1.set_title(f'{title_ds} GT')
    ax1.set_axis_off()
    # Legend row（黒=Unassigned(0) を明記）
    _draw_legend_row(ax2, class_names, colors_hex)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    finalize_png(out_path)


def _draw_legend_row(ax, class_names: List[str], colors_hex: List[str], include_bg: bool = True):
    ax.set_axis_off()
    names = class_names[1:]
    cols = [hex_to_rgb01(c) for c in colors_hex[1:]]
    # レイアウト: 水平方向に並べ、はみ出す場合は改行
    x, y = 0.02, 0.65
    step_x = 0.16
    box_w = 0.03
    box_h = 0.15
    if include_bg:
        # 黒=Unassigned(0) を先頭に配置
        ax.add_patch(plt.Rectangle((x, y), box_w, box_h, color=(0,0,0), transform=ax.transAxes))
        ax.text(x + box_w + 0.01, y + 0.01, 'Unassigned (0)', va='bottom', fontsize=10, transform=ax.transAxes)
        x += step_x
    for nm, col in zip(names, cols):
        ax.add_patch(plt.Rectangle((x, y), box_w, box_h, color=col, transform=ax.transAxes))
        ax.text(x + box_w + 0.01, y + 0.01, nm, va='bottom', fontsize=10, transform=ax.transAxes)
        x += step_x
        if x + step_x > 0.98:
            x = 0.02
            y -= 0.22


def draw_four_panel(pauli_img: Optional[Path], gt: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                    cmap: ListedColormap, class_names: List[str], colors_hex: List[str],
                    labels: Tuple[str, str, str, str], out_path: Path, legend_bottom: bool = False):
    n_cls = len(class_names) - 1
    if legend_bottom:
        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.35])
        ax00 = fig.add_subplot(gs[0, 0])
        ax01 = fig.add_subplot(gs[0, 1])
        ax10 = fig.add_subplot(gs[1, 0])
        ax11 = fig.add_subplot(gs[1, 1])
        axL = fig.add_subplot(gs[2, :])
    else:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        ax00, ax01 = axs[0,0], axs[0,1]
        ax10, ax11 = axs[1,0], axs[1,1]
        axL = None

    # (a) Pauli
    if pauli_img and pauli_img.exists():
        ax00.imshow(Image.open(pauli_img))
    else:
        ax00.imshow(np.zeros((*gt.shape, 3), dtype=np.uint8))
        ax00.text(0.5, 0.5, 'Pauli RGB not found', color='w', ha='center', va='center', transform=ax00.transAxes)
        ax00.set_facecolor('k')
    ax00.set_title(labels[0]); ax00.set_axis_off()
    # (b) GT
    ax01.imshow(gt, cmap=cmap, vmin=0, vmax=n_cls)
    ax01.set_title(labels[1]); ax01.set_axis_off()
    # (c) Pred A
    ax10.imshow(pred_a, cmap=cmap, vmin=0, vmax=n_cls)
    ax10.set_title(labels[2]); ax10.set_axis_off()
    # (d) Pred B
    ax11.imshow(pred_b, cmap=cmap, vmin=0, vmax=n_cls)
    ax11.set_title(labels[3]); ax11.set_axis_off()

    if legend_bottom and axL is not None:
        _draw_legend_row(axL, class_names, colors_hex)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    finalize_png(out_path)


def draw_legend(dataset_key: str, class_names: List[str], colors: List[str], out_path: Path, lang: str = 'en', include_bg_note: bool = True):
    # 0: Unassigned は除いて1..nを表示
    names = class_names[1:]
    cols = [hex_to_rgb01(c) for c in colors[1:]]
    n = len(names) + (1 if include_bg_note else 0)
    h = max(320, 22 * n)
    fig, ax = plt.subplots(figsize=(6, h/100))
    ax.set_axis_off()
    title = f"凡例 - {dataset_key}" if lang == 'ja' else f"Legend - {dataset_key}"
    ax.set_title(title)
    row = 0
    if include_bg_note:
        y = n - 1 - row
        ax.add_patch(plt.Rectangle((0.0, y), 1.0, 0.8, color=(0,0,0), transform=ax.transData))
        ax.text(1.2, y+0.1, 'Unassigned (0)', va='bottom', fontsize=9, transform=ax.transData)
        row += 1
    for i, (nm, col) in enumerate(zip(names, cols)):
        y = n - 1 - (i + row)
        ax.add_patch(plt.Rectangle((0.0, y), 1.0, 0.8, color=col, transform=ax.transData))
        ax.text(1.2, y+0.1, nm, va='bottom', fontsize=9, transform=ax.transData)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n+1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    # Enforce PNG properties
    try:
        im = Image.open(out_path)
        im = im.convert('RGB')
        try:
            srgb = ImageCms.createProfile('sRGB')
            im.save(out_path, format='PNG', icc_profile=srgb.tobytes(), optimize=True, interlace=0)
        except Exception:
            im.save(out_path, format='PNG', optimize=True, interlace=0)
    except Exception as e:
        print(f"[WARN] PNG postprocess failed for {out_path}: {e}")

def save_table_md(dataset_key: str, class_names: List[str], per_cls: np.ndarray, oa: float, aa: float, kap: float,
                  out_md: Path, lang: str = 'en'):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open('w', encoding='utf-8') as f:
        header = "クラス別精度" if lang == 'ja' else "Per-class accuracy"
        f.write(f"# {header} ({dataset_key})\n\n")
        f.write(("| クラス | 精度(%) |\n|---|---:|\n") if lang == 'ja' else ("| Class | Accuracy (%) |\n|---|---:|\n"))
        for cid in range(1, len(class_names)):
            name = class_names[cid]
            acc = 100.0 * float(per_cls[cid-1])
            f.write(f"| {name} | {acc:.2f} |\n")
        f.write("\n")
        if lang == 'ja':
            f.write(f"OA(%) = {100.0*oa:.2f}  ")
            f.write(f"AA(%) = {100.0*aa:.2f}  ")
            f.write(f"カッパ×100 = {100.0*kap:.2f}\n")
        else:
            f.write(f"OA (%) = {100.0*oa:.2f}  ")
            f.write(f"AA (%) = {100.0*aa:.2f}  ")
            f.write(f"Kappa x100 = {100.0*kap:.2f}\n")


def main():
    ap = argparse.ArgumentParser(description="Paper-style visualization and tables from .mat predictions")
    ap.add_argument('--results', default='results', help='directory containing prediction .mat files')
    ap.add_argument('--outdir', default='results/paper', help='output directory for figures and tables')
    ap.add_argument('--align', choices=['none', 'crop', 'nearest'], default='nearest', help='align GT and prediction shapes')
    ap.add_argument('--lang', choices=['en', 'ja'], default='en', help='legend/table language')
    ap.add_argument('--gt-offset', default=None, help="apply integer offset to GT: 'dy,dx'")
    ap.add_argument('--ober-collapse', action='store_true', help='collapse ober pred classes (4,5->3)')
    ap.add_argument('--make', default='triptych,legend,tables,fig6-8,fig9-11',
                    help='comma-separated: triptych,legend,tables,fig6-8,fig9-11')
    ap.add_argument('--datasets', default='all', help='comma-separated list (FL_T,SF,ober) or all')
    args = ap.parse_args()

    res_dir = Path(args.results)
    out_dir = Path(args.outdir)
    out_img = out_dir / 'figs'
    out_tbl = out_dir / 'tables'
    out_img.mkdir(parents=True, exist_ok=True)
    out_tbl.mkdir(parents=True, exist_ok=True)

    # 期待ファイル名
    candidates = sorted(res_dir.glob('*.mat'))
    if not candidates:
        print('No .mat found in', res_dir)
        return

    make = set([s.strip() for s in args.make.split(',') if s.strip()])
    sel = set(DATASETS.keys()) if args.datasets == 'all' else set([s.strip() for s in args.datasets.split(',') if s.strip()])

    # 事前にデータセットごとにnonfull/fullのパスを収集
    by_ds: Dict[str, Dict[str, Optional[Path]]] = {k: {'nonfull': None, 'full': None} for k in DATASETS.keys() if k in sel}
    for mp in candidates:
        ds = detect_dataset_from_name(mp.name)
        if not ds or ds not in sel:
            continue
        if 'Full_' in mp.stem or mp.stem.startswith('CV_MsAtViT_Full_'):
            by_ds[ds]['full'] = mp
        else:
            by_ds[ds]['nonfull'] = mp

    # 1) 個別処理（triptych/legend/tables）
    for mp in candidates:
        ds = detect_dataset_from_name(mp.name)
        if ds is None or ds not in sel:
            print('[SKIP] cannot detect dataset for', mp.name)
            continue
        meta = DATASETS[ds]
        classes: List[str] = meta['classes_ja'] if args.lang == 'ja' and 'classes_ja' in meta else meta['classes']
        colors: List[str] = meta['colors']
        gt_path: Path = meta['gt']
        pauli_entry: Path = meta['pauli']
        pauli = find_pauli_image(pauli_entry)
        cmap = make_cmap(colors)
        n_cls = len(classes) - 1

        pred, var = load_pred(mp)
        if args.ober_collapse and ds == 'ober':
            pred = collapse_ober_pred(pred)
        # GT
        have_gt = gt_path.exists()
        if have_gt:
            gt, gvar = load_gt(gt_path)
            if gt.shape != pred.shape:
                print(f"[INFO] Aligning GT ({gt.shape}) to Pred ({pred.shape}) by {args.align}")
                gt, pred = align_arrays(gt, pred, args.align)
            if args.gt_offset:
                dy, dx = map(int, args.gt_offset.split(','))
                gt = shift_label_pad(gt, dy, dx)
                print(f"[INFO] Shifted GT by (dy,dx)=({dy},{dx})")
        else:
            gt = np.zeros_like(pred)

        # 予測はFullが1..n、マスク適用は0を含む可能性あり → そのまま可視化
        tag = mp.stem.replace('CV_MsAtViT_', '')
        if 'triptych' in make:
            trip = out_img / f"triptych_{tag}.png"
            draw_triptych(pauli if pauli and pauli.exists() else None, gt, pred, cmap, n_cls,
                          (f"{DISPLAY_NAME_EN.get(ds, ds)} PauliRGB", f"{DISPLAY_NAME_EN.get(ds, ds)} GT", f"{DISPLAY_NAME_EN.get(ds, ds)} CV-MsAtViT"), trip)
            print('[IMG]', trip)

        # 凡例
        if 'legend' in make:
            legend_png = out_img / f"legend_{ds}.png"
            draw_legend(ds, classes, colors, legend_png, lang=args.lang, include_bg_note=True)
            print('[LEGEND]', legend_png)

        # 指標（GTがあれば）
        if have_gt and 'tables' in make:
            per_cls, oa, aa, kap = compute_class_acc(pred, gt, n_cls)
            md = out_tbl / f"table_{tag}.md"
            save_table_md(tag, classes, per_cls, oa, aa, kap, md, lang=args.lang)
            # JSONも保存
            with (out_tbl / f"table_{tag}.json").open('w', encoding='utf-8') as f:
                json.dump({
                    'dataset': tag,
                    'classes': classes,
                    'per_class_acc': (per_cls*100.0).tolist(),
                    'OA_pct': 100.0*oa,
                    'AA_pct': 100.0*aa,
                    'Kappa_x100': 100.0*kap,
                }, f, indent=2)
            print('[TBL]', md)
        else:
            print(f"[NOTE] GT not found for {ds}. Skipped table generation.")

    # 2) fig6-8: Pauli + GT + Legend
    if 'fig6-8' in make:
        for ds, pair in by_ds.items():
            # 参照サイズ用にfull優先
            ref_mp = pair['full'] or pair['nonfull']
            if not ref_mp:
                continue
            meta = DATASETS[ds]
            classes: List[str] = meta['classes_ja'] if args.lang == 'ja' and 'classes_ja' in meta else meta['classes']
            colors: List[str] = meta['colors']
            cmap = make_cmap(colors)
            gt_path = meta['gt']
            pauli = find_pauli_image(meta['pauli'])
            # 形状合わせ
            pred_ref, _ = load_pred(ref_mp)
            gt, _ = load_gt(gt_path)
            gt, _ = align_arrays(gt, pred_ref, args.align)
            if args.gt_offset:
                dy, dx = map(int, args.gt_offset.split(','))
                gt = shift_label_pad(gt, dy, dx)
            out_name = {'FL_T':'fig6_FL_T.png', 'SF':'fig7_SF.png', 'ober':'fig8_ober.png'}[ds]
            draw_pauli_gt_with_legend(pauli if pauli and pauli.exists() else None, gt, cmap, classes, colors,
                                       out_img / out_name, lang=args.lang, dataset_display=DISPLAY_NAME_EN.get(ds, ds))
            print('[FIG6-8]', out_img / out_name)

    # 3) fig9-11: Pauli, GT, CV-MsAtViT(nonfull), CV-MsAtViT(Full)
    if 'fig9-11' in make:
        for ds, pair in by_ds.items():
            if not (pair['nonfull'] and pair['full']):
                continue
            meta = DATASETS[ds]
            classes: List[str] = meta['classes_ja'] if args.lang == 'ja' and 'classes_ja' in meta else meta['classes']
            colors: List[str] = meta['colors']
            cmap = make_cmap(colors)
            gt_path = meta['gt']
            pauli = find_pauli_image(meta['pauli'])
            pred_nf, _ = load_pred(pair['nonfull'])
            pred_f, _ = load_pred(pair['full'])
            if args.ober_collapse and ds == 'ober':
                pred_nf = collapse_ober_pred(pred_nf)
                pred_f = collapse_ober_pred(pred_f)
            gt, _ = load_gt(gt_path)
            # align to full
            gt, _ = align_arrays(gt, pred_f, args.align)
            if args.gt_offset:
                dy, dx = map(int, args.gt_offset.split(','))
                gt = shift_label_pad(gt, dy, dx)
            name_map = {'FL_T':'fig9_FL_T.png', 'SF':'fig10_SF.png', 'ober':'fig11_ober.png'}
            disp = DISPLAY_NAME_EN.get(ds, ds)
            labels = (f'{disp} PauliRGB', f'{disp} GT', f'{disp} CV-MsAtViT', f'{disp} CV-MsAtViT (Full)')
            place_legend = ds in ['FL_T', 'SF']  # 要望: fig9とfig10にのみ凡例を付与
            draw_four_panel(pauli if pauli and pauli.exists() else None, gt, pred_nf, pred_f,
                            cmap, classes, colors, labels, out_img / name_map[ds], legend_bottom=place_legend)
            print('[FIG9-11]', out_img / name_map[ds])


if __name__ == '__main__':
    main()
