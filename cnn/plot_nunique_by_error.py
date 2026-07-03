#!/usr/bin/env python3
"""
Nunique (read-support) distribution by error type for CNN error dumps.

Reads *_errors_v7_restructure.tsv dumps (must carry `Nunique` and
`error_type` columns) and, for each file, plots the distribution of
Nunique split by error type (TP/FN and TN/FP), with a Mann-Whitney U
test comparing FN vs TP (the question of interest: do low-read-support
positives get missed more often than confidently-supported ones).

Also plots the model's predicted probability against Nunique (true
positives only) to check whether low-read-support interactions are
scored systematically lower — the likely mechanism behind the FN/TP gap.

Usage:
    python plot_nunique_by_error.py
    python plot_nunique_by_error.py --files A.tsv B.tsv --outdir results
    python plot_nunique_by_error.py --per-binding-type --min-n 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt


def _label_from_path(path: Path) -> str:
    stem = path.stem
    for junk in ("_errors_v7_restructure", "_errors", "_v7_restructure"):
        stem = stem.replace(junk, "")
    return stem.replace("_", " ").strip() or path.stem


def _prob_col(df: pd.DataFrame) -> str | None:
    """Pick the predicted-probability column (`prob` and `interaction_probability`
    are identical in these dumps; prefer the shorter name)."""
    for col in ("prob", "interaction_probability"):
        if col in df.columns:
            return col
    return None


def _effect_stats(fn: pd.Series, tp: pd.Series) -> dict:
    """FN-vs-TP Mann-Whitney test + effect sizes for one Nunique sample pair."""
    u, p = stats.mannwhitneyu(fn, tp, alternative="less")
    n1, n2 = len(fn), len(tp)
    # NOTE: scipy's U for mannwhitneyu(x, y) satisfies U/(n1*n2) == P(x > y)
    # (Vargha-Delaney A treating x as the "larger" sample), NOT P(x < y) —
    # verified empirically. So P(FN < TP) is the complement of u/(n1*n2).
    cles = 1 - u / (n1 * n2)  # P(FN value < TP value); 0.5 = no effect
    rank_biserial = 2 * cles - 1  # signed Cliff's delta: >0 means FN skews lower
    pooled_std = np.sqrt(
        ((n1 - 1) * fn.var(ddof=1) + (n2 - 1) * tp.var(ddof=1)) / (n1 + n2 - 2)
    )
    cohens_d = (fn.mean() - tp.mean()) / pooled_std
    return {
        "n_fn": n1, "n_tp": n2,
        "mean_fn": fn.mean(), "mean_tp": tp.mean(),
        "median_fn": fn.median(), "median_tp": tp.median(),
        "p": p, "cles": cles, "rank_biserial": rank_biserial, "cohens_d": cohens_d,
    }


def plot_file(path: Path, out: Path) -> None:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    for col in ("Nunique", "error_type"):
        if col not in df.columns:
            print(f"WARNING: {path} lacks '{col}' — skipping.", file=sys.stderr)
            return

    fn = df.loc[df["error_type"] == "FN", "Nunique"].dropna()
    tp = df.loc[df["error_type"] == "TP", "Nunique"].dropna()

    s = _effect_stats(fn, tp)
    p, cles, rank_biserial, cohens_d = s["p"], s["cles"], s["rank_biserial"], s["cohens_d"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    label = _label_from_path(path)

    # Left: grouped bars of P(Nunique=k) for FN vs TP, log-scaled — the two
    # groups overlap almost completely as raw density curves (both peak at
    # Nunique=1), so this shows the actual gap: FN is disproportionately
    # concentrated at Nunique=1, TP has more mass at 2+.
    ax = axes[0]
    max_val = max(fn.max(), tp.max())
    cap = int(min(max_val, 10))
    edges = list(range(1, cap)) + [cap]
    fn_binned = fn.clip(upper=cap)
    tp_binned = tp.clip(upper=cap)
    fn_props = fn_binned.value_counts(normalize=True).reindex(edges, fill_value=0)
    tp_props = tp_binned.value_counts(normalize=True).reindex(edges, fill_value=0)
    x = np.arange(len(edges))
    w = 0.38
    ax.bar(x - w / 2, fn_props.values, width=w, label=f"FN (n={len(fn):,})", color="tab:red")
    ax.bar(x + w / 2, tp_props.values, width=w, label=f"TP (n={len(tp):,})", color="tab:green")
    ax.set_yscale("log")
    xt = [str(e) for e in edges]
    xt[-1] = f"{cap}+"
    ax.set_xticks(x, xt)
    ax.set_xlabel("Nunique (read support)")
    ax.set_ylabel("Proportion within group (log scale)")
    ax.set_title(
        f"Positives: FN vs TP\n"
        f"mean FN={fn.mean():.2f}, mean TP={tp.mean():.2f}, MWU p={p:.2e}\n"
        f"rank-biserial={rank_biserial:.3f}, Cohen's d={cohens_d:.3f}"
    )
    ax.legend(fontsize=9)

    # Right: ECDF of FN vs TP (medians tie at 1, so a boxplot hides the
    # shift; TN/FP are omitted here since Nunique==0 for all label==0 rows).
    ax = axes[1]
    for data, color, name in ((fn, "tab:red", "FN"), (tp, "tab:green", "TP")):
        xs = np.sort(data.to_numpy())
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post", color=color, lw=2, label=f"{name} (n={len(data):,})")
    ax.set_xlabel("Nunique (read support)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(f"ECDF: FN vs TP\nP(FN < TP) [CLES] = {cles:.3f} (0.5 = no effect)")
    ax.set_xlim(0, min(max_val, 20))
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)

    fig.suptitle(f"{label}  (n={len(df):,})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}", file=sys.stderr)
    print(
        f"{label}: FN mean={fn.mean():.3f} median={fn.median():.1f} n={len(fn)} | "
        f"TP mean={tp.mean():.3f} median={tp.median():.1f} n={len(tp)} | "
        f"MannWhitney(FN<TP) p={p:.3e} | "
        f"CLES(P[FN<TP])={cles:.4f} | rank-biserial={rank_biserial:.4f} | "
        f"Cohen's d={cohens_d:.4f}"
    )


def plot_prob_vs_nunique(path: Path, out: Path) -> None:
    """Does the model's predicted probability track read support?

    Restricted to true positives (label==1: FN+TP together) since Nunique is
    structurally 0 for all negatives. If probability rises with Nunique, that
    would explain the FN/TP gap found elsewhere in this script: low-read-
    support interactions get systematically lower scores, some of which fall
    below the 0.5 decision threshold and become FNs.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    prob_col = _prob_col(df)
    if prob_col is None or "Nunique" not in df.columns or "label" not in df.columns:
        print(f"WARNING: {path} lacks Nunique/label/probability columns — skipping.",
              file=sys.stderr)
        return

    pos = df.loc[df["label"] == 1, ["Nunique", prob_col]].dropna()
    rho, rho_p = stats.spearmanr(pos["Nunique"], pos[prob_col])

    label = _label_from_path(path)
    cap = int(min(pos["Nunique"].max(), 10))
    capped = pos["Nunique"].clip(upper=cap)
    grouped = pos.groupby(capped)[prob_col].agg(["mean", "std", "count"])
    sem = grouped["std"] / np.sqrt(grouped["count"])
    ci95 = 1.96 * sem

    fig, ax = plt.subplots(figsize=(7, 5))
    x = grouped.index.to_numpy()
    ax.errorbar(x, grouped["mean"], yerr=ci95, fmt="o-", color="tab:purple",
               capsize=3, lw=1.5, label="mean predicted probability (±95% CI)")
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="decision threshold (0.5)")
    xt = [str(v) for v in x]
    xt[-1] = f"{cap}+"
    ax.set_xticks(x, xt)
    ax.set_xlabel("Nunique (read support)")
    ax.set_ylabel("Predicted probability")
    ax.set_title(
        f"Predicted probability vs Nunique (true positives) — {label}\n"
        f"Spearman rho={rho:.3f}, p={rho_p:.2e}, n={len(pos):,}"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}", file=sys.stderr)
    print(
        f"{label}: prob-vs-Nunique Spearman rho={rho:.4f} p={rho_p:.3e} n={len(pos):,} | "
        + " | ".join(
            f"Nunique={i if i < cap else f'{cap}+'}: mean_prob={row['mean']:.3f} n={int(row['count'])}"
            for i, row in grouped.iterrows()
        )
    )


def _classify_fn_tp(path: Path, classifier) -> pd.DataFrame | None:
    """Load a dump and return its FN/TP rows tagged with a binding_type column."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    for col in ("Nunique", "error_type", "gene", "noncodingRNA"):
        if col not in df.columns:
            print(f"WARNING: {path} lacks '{col}' — skipping binding-type breakdown.",
                  file=sys.stderr)
            return None

    sub = df[df["error_type"].isin(["FN", "TP"])].copy()
    sub["binding_type"] = [
        classifier(str(m), str(g)) for m, g in zip(sub["noncodingRNA"], sub["gene"])
    ]
    return sub


def write_counts_table(sub: pd.DataFrame, out: Path, label: str) -> None:
    """Write absolute FN/TP counts (+ mean predicted probability) per
    (binding_type, Nunique) to a TSV.

    Unlike the forest plot, this is not filtered by --min-n — it's the raw
    cross-tab so small categories/values are still visible in the file.
    """
    grp = sub.groupby(["binding_type", "Nunique", "error_type"])
    counts = grp.size().unstack("error_type", fill_value=0)
    for col in ("FN", "TP"):
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["FN", "TP"]].rename(columns={"FN": "n_FN", "TP": "n_TP"})

    prob_col = _prob_col(sub)
    if prob_col is not None:
        mean_prob = grp[prob_col].mean().unstack("error_type")
        for col in ("FN", "TP"):
            if col not in mean_prob.columns:
                mean_prob[col] = np.nan
        mean_prob = mean_prob[["FN", "TP"]].rename(
            columns={"FN": "mean_prob_FN", "TP": "mean_prob_TP"}
        )
        counts = counts.join(mean_prob)

    counts = counts.reset_index().sort_values(["binding_type", "Nunique"])
    counts.insert(0, "dataset", label)
    counts.to_csv(out, sep="\t", index=False)
    print(f"Wrote {out}", file=sys.stderr)


def plot_by_binding_type(sub: pd.DataFrame, out: Path, label: str, min_n: int = 50) -> None:
    """Forest-plot the FN-vs-TP Nunique effect size, stratified by binding type.

    Answers whether the overall FN < TP shift is uniform across binding types
    or concentrated in a subset (e.g. non-canonical/seedless sites) — the
    overall two-panel plot from plot_file() pools all binding types together
    and can't distinguish those cases.
    """
    rows = []
    for cat, g in sub.groupby("binding_type"):
        fn = g.loc[g["error_type"] == "FN", "Nunique"].dropna()
        tp = g.loc[g["error_type"] == "TP", "Nunique"].dropna()
        if len(fn) < min_n or len(tp) < min_n:
            continue
        s = _effect_stats(fn, tp)
        s["binding_type"] = cat
        rows.append(s)

    if not rows:
        print(f"No binding type has >= {min_n} FN and >= {min_n} TP rows in "
              f"{label} — skipping breakdown.", file=sys.stderr)
        return

    rows.sort(key=lambda r: r["n_fn"] + r["n_tp"], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.5))
    y = np.arange(len(rows))
    rb = [r["rank_biserial"] for r in rows]
    colors = ["tab:red" if v > 0 else "tab:blue" for v in rb]
    ax.barh(y, rb, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    names = [
        f"{r['binding_type']}  (n_FN={r['n_fn']:,}, n_TP={r['n_tp']:,}, p={r['p']:.1e})"
        for r in rows
    ]
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Rank-biserial correlation (FN vs TP Nunique)\n"
                  "positive = FN skews lower (harder-to-detect, low read support)")
    ax.set_title(f"Nunique effect size by binding type — {label}\n"
                 f"(only types with >= {min_n} FN and >= {min_n} TP shown)",
                 fontsize=10)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}", file=sys.stderr)
    for r in rows:
        print(
            f"  {label} / {r['binding_type']}: n_FN={r['n_fn']} n_TP={r['n_tp']} | "
            f"mean_FN={r['mean_fn']:.3f} mean_TP={r['mean_tp']:.3f} | "
            f"p={r['p']:.3e} | rank-biserial={r['rank_biserial']:.4f} | "
            f"Cohen's d={r['cohens_d']:.4f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot Nunique distributions by error type.")
    ap.add_argument(
        "--files", nargs="+",
        default=[
            "results/manakov_test_errors_v7_restructure.tsv",
            "results/manakov_leftout_errors_v7_restructure.tsv",
        ],
        help="Error TSV dumps to plot (default: manakov test + leftout).",
    )
    ap.add_argument("--outdir", default="results",
                    help="Directory to write plots into (default: results).")
    ap.add_argument("--per-binding-type", action="store_true",
                    help="Also write a per-binding-type effect-size forest plot "
                         "per input file (nunique_by_error_<label>_bt.png).")
    ap.add_argument("--min-n", type=int, default=50,
                    help="Minimum FN and TP rows required to plot a binding "
                         "type in the breakdown (default: 50).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = [Path(f) for f in args.files]
    missing = [f for f in files if not f.exists()]
    for f in missing:
        print(f"WARNING: {f} not found — skipping.", file=sys.stderr)
    files = [f for f in files if f.exists()]
    if not files:
        print("ERROR: no input files exist.", file=sys.stderr)
        return 1

    for f in files:
        label = _label_from_path(f)
        slug = label.replace(" ", "_")
        plot_file(f, outdir / f"nunique_by_error_{slug}.png")
        plot_prob_vs_nunique(f, outdir / f"prob_vs_nunique_{slug}.png")

    if args.per_binding_type:
        try:
            from binding_types import classify_binding_type
        except ImportError:
            from cnn.binding_types import classify_binding_type
        for f in files:
            label = _label_from_path(f)
            sub = _classify_fn_tp(f, classify_binding_type)
            if sub is None:
                continue
            slug = label.replace(" ", "_")
            write_counts_table(sub, outdir / f"nunique_by_error_{slug}_bt_counts.tsv", label)
            plot_by_binding_type(sub, outdir / f"nunique_by_error_{slug}_bt.png",
                                 label, min_n=args.min_n)

    return 0


if __name__ == "__main__":
    sys.exit(main())
