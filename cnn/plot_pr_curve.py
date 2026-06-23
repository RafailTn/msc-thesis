#!/usr/bin/env python3
"""
Precision–recall (AUPRC) curve plot for miRNA–MRE CNN error dumps.

Reads one or more *_errors.tsv files (the dumps written by
cnn_branches_mirbind.py --error-dump, which carry `label` and `prob`
columns) and overlays their precision–recall curves on a single axis, each
annotated with its average precision (AP = area under the PR curve) and a
dashed chance baseline at that set's positive prevalence.

AP is the metric used for model selection here, so this is the visual
companion to section [8] of error_analysis.py.

Usage:
    python plot_pr_curve.py                       # default: manakov test + leftout
    python plot_pr_curve.py --files A.tsv B.tsv --labels "Test" "Leftout"
    python plot_pr_curve.py --out results/pr_curve.png --per-binding-type
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve


def _label_from_path(path: Path) -> str:
    """Human-readable legend label from a dump filename."""
    stem = path.stem
    for junk in ("_errors_v7_restructure", "_errors", "_v7_restructure"):
        stem = stem.replace(junk, "")
    return stem.replace("_", " ").strip() or path.stem


def _load_yp(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load (y, p, df) from an error dump; drop rows with missing label/prob."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if "label" not in df.columns or "prob" not in df.columns:
        raise ValueError(f"{path} lacks 'label'/'prob' columns "
                         f"(has {list(df.columns)[:10]}...)")
    y = pd.to_numeric(df["label"], errors="coerce")
    p = pd.to_numeric(df["prob"], errors="coerce")
    ok = y.notna() & p.notna()
    return y[ok].astype(int).to_numpy(), p[ok].to_numpy(), df.loc[ok].reset_index(drop=True)


def plot_overall(files: list[Path], labels: list[str], out: Path,
                 title: str) -> None:
    """One PR curve per file, overlaid, with AP and chance baseline each."""
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (path, lab) in enumerate(zip(files, labels)):
        y, p, _ = _load_yp(path)
        if len(np.unique(y)) < 2:
            print(f"WARNING: {path} has a single class — skipping.", file=sys.stderr)
            continue
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        prev = float(y.mean())
        color = colors[i % len(colors)]
        # step='post' matches how AP integrates the PR curve (no optimistic interp).
        ax.step(rec, prec, where="post", color=color, lw=2,
                label=f"{lab}  (AP={ap:.3f}, n={len(y):,})")
        ax.axhline(prev, color=color, ls=":", lw=1, alpha=0.7)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}", file=sys.stderr)


def plot_per_binding_type(path: Path, out: Path, min_n: int = 50,
                          classifier=None) -> None:
    """PR curve per binding type for a single dump (where it can be classified)."""
    y, p, df = _load_yp(path)
    if "binding_type" not in df.columns:
        if classifier is None:
            print("No 'binding_type' column and no classifier available — "
                  "skipping per-binding-type plot.", file=sys.stderr)
            return
        gene_col = next((c for c in ("gene", "mre_sequence", "target_seq")
                         if c in df.columns), None)
        mirna_col = next((c for c in ("noncodingRNA", "mirna_sequence", "query_seq")
                          if c in df.columns), None)
        if not (gene_col and mirna_col):
            print("No sequence columns to classify binding type — skipping.",
                  file=sys.stderr)
            return
        df["binding_type"] = [classifier(str(m), str(t))
                              for m, t in zip(df[mirna_col], df[gene_col])]

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    sub = pd.DataFrame({"y": y, "p": p, "bt": df["binding_type"].to_numpy()})
    rows = [(len(g), cat, g) for cat, g in sub.groupby("bt")
            if len(g) >= min_n and g["y"].nunique() == 2]
    for n, cat, g in sorted(rows, key=lambda r: -r[0]):
        prec, rec, _ = precision_recall_curve(g["y"], g["p"])
        ap = average_precision_score(g["y"], g["p"])
        ax.step(rec, prec, where="post", lw=1.6,
                label=f"{cat}  (AP={ap:.3f}, n={n:,})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"PR by binding type — {_label_from_path(path)}")
    ax.legend(loc="lower left", frameon=True, fontsize=7.5)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot PR/AUPRC curves from CNN error dumps.")
    ap.add_argument(
        "--files", nargs="+",
        default=[
            "results/manakov_test_errors_v7_restructure.tsv",
            "results/manakov_leftout_errors_v7_restructure.tsv",
        ],
        help="Error TSV dumps to plot (default: manakov test + leftout).",
    )
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Legend labels (default: derived from filenames).")
    ap.add_argument("--out", default="results/pr_curve.png",
                    help="Output image path (default: results/pr_curve.png).")
    ap.add_argument("--title", default="Precision–recall (AUPRC)",
                    help="Plot title.")
    ap.add_argument("--per-binding-type", action="store_true",
                    help="Also write one per-binding-type PR plot per input file "
                         "(<out>_<label>_bt.png).")
    args = ap.parse_args()

    files = [Path(f) for f in args.files]
    missing = [f for f in files if not f.exists()]
    if missing:
        for f in missing:
            print(f"WARNING: {f} not found — skipping.", file=sys.stderr)
        files = [f for f in files if f.exists()]
    if not files:
        print("ERROR: no input files exist.", file=sys.stderr)
        return 1

    labels = args.labels or [_label_from_path(f) for f in files]
    if len(labels) != len(files):
        print("ERROR: --labels count must match --files count.", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_overall(files, labels, out, args.title)

    if args.per_binding_type:
        try:
            from binding_types import classify_binding_type
        except ImportError:
            from cnn.binding_types import classify_binding_type
        for f, lab in zip(files, labels):
            bt_out = out.with_name(f"{out.stem}_{lab.replace(' ', '_')}_bt{out.suffix}")
            plot_per_binding_type(f, bt_out, classifier=classify_binding_type)

    return 0


if __name__ == "__main__":
    sys.exit(main())
