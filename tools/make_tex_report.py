#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


DS_TITLES_JA = {
    'FL_T': 'Flevoland',
    'SF': 'San Francisco',
    'ober': 'Oberpfaffenhofen',
}


def load_json(p: Path) -> Dict:
    if not p.exists():
        return {}
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description='Generate LaTeX report summarizing metrics and figures')
    ap.add_argument('--metricsdir', default='results/metrics')
    ap.add_argument('--paperdir', default='results/paper')
    ap.add_argument('--outdir', default='results/report')
    ap.add_argument('--datasets', default='FL_T,SF,ober')
    ap.add_argument('--use-full', action='store_true', help='use Full_* metrics and figures')
    ap.add_argument('--lang', choices=['en', 'ja'], default='ja')
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [s.strip() for s in args.datasets.split(',') if s.strip()]
    rows_summary: List[Tuple[str, float, float, float]] = []

    sections: List[str] = []
    for ds in datasets:
        tag = f"Full_{ds}" if args.use_full else ds
        mpath = Path(args.metricsdir) / f"{tag}.json"
        tpath = Path(args.paperdir) / 'tables' / f"table_{ds}.json"
        fimg = Path(args.paperdir) / 'figs' / f"triptych_{tag}.png"

        m = load_json(mpath)
        t = load_json(tpath)

        # 指標（OA/AA/Kappa）
        if m.get('metrics'):
            met = m['metrics']
            oa = float(met.get('OA', 0.0))*100.0
            aa = float(met.get('AA', 0.0))*100.0
            kap = float(met.get('Kappa', 0.0))*100.0
        else:
            oa = float(t.get('OA_pct', 0.0))
            aa = float(t.get('AA_pct', 0.0))
            kap = float(t.get('Kappa_x100', 0.0))

        rows_summary.append((ds, oa, aa, kap))

        # セクション本文
        title = DS_TITLES_JA.get(ds, ds) if args.lang == 'ja' else ds
        sections.append(
            '\n'.join([
                f"\\section{{{title}}}",
                "\\begin{figure}[h]",
                "  \\centering",
                f"  \\includegraphics[width=0.95\\linewidth]{{{fimg.as_posix()}}}",
                f"  \\caption{{{title}: PauliRGB/GT/予測}}" if args.lang=='ja' else f"  \\caption{{{title}: PauliRGB/GT/Prediction}}",
                "\\end{figure}",
                "",
                "\\begin{table}[h]",
                "  \\centering",
                "  \\begin{tabular}{lrrr}",
                "    \\toprule",
                ("    指標 & OA(%) & AA(%) & カッパ×100 \\" if args.lang=='ja' else "    Metric & OA(%) & AA(%) & Kappa \\"),
                "    \\midrule",
                (f"    値 & {oa:.2f} & {aa:.2f} & {kap:.2f} \\" if args.lang=='ja' else f"    Value & {oa:.2f} & {aa:.2f} & {kap:.2f} \\"),
                "    \\bottomrule",
                "  \\end{tabular}",
                "\\end{table}",
            ])
        )

    # サマリ表
    header = (
        "\\begin{table}[h]\n  \\centering\n  \\begin{tabular}{lrrr}\n    \\toprule\n    データセット & OA(%) & AA(%) & カッパ×100 \\ \n    \\midrule" if args.lang=='ja'
        else "\\begin{table}[h]\n  \\centering\n  \\begin{tabular}{lrrr}\n    \\toprule\n    Dataset & OA(%) & AA(%) & Kappa \\ \n    \\midrule"
    )
    lines = [header]
    for ds, oa, aa, kap in rows_summary:
        title = DS_TITLES_JA.get(ds, ds) if args.lang == 'ja' else ds
        lines.append(f"    {title} & {oa:.2f} & {aa:.2f} & {kap:.2f} \\")
    lines.append("    \\bottomrule\n  \\end{tabular}\n\\end{table}")
    summary_table = '\n'.join(lines)

    # LaTeX本文
    title_text = 'CV-MsAtViT 3地点レポート' if args.lang=='ja' else 'CV-MsAtViT Report (3 Sites)'
    doc = '\n'.join([
        "\\documentclass[11pt]{article}",
        "\\usepackage{graphicx}",
        "\\usepackage{booktabs}",
        "\\usepackage[margin=22mm]{geometry}",
        f"\\title{{{title_text}}}",
        "\\date{}",
        "\\begin{document}",
        "\\maketitle",
        summary_table,
        *sections,
        "\\end{document}",
    ])

    out_tex = out_dir / 'report.tex'
    with out_tex.open('w', encoding='utf-8') as f:
        f.write(doc)
    print('[TEX]', out_tex)


if __name__ == '__main__':
    main()
