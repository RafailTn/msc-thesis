#!/usr/bin/env python3
"""
eqtl_pip_separation.py

Downstream analysis of the two delta_pred output TSVs produced by
eqtl_analysis_cnn.py — one for high-PIP (likely causal, pip > 0.9) variants and
one for low-PIP (pip < 0.01) variants.

Pipeline:
  1. Filter each file: for mutations on the same position of the same transcript
     (group by gene + snp_pos), keep only the single tissue row with the largest
     |beta_marginal|. (Matches the dedup applied to the earlier eQTL files.)
  2. Drop overlaps: any (gene, snp_pos) present in BOTH the high- and low-PIP
     filtered sets is removed from the LOW-PIP set (high PIP wins).
  3. Separation test: Mann-Whitney U on delta_pred (high vs low) + a histogram of
     delta_pred for the two categories.
  4. Correlation: Pearson + Spearman of delta_pred vs beta_marginal.

Allele convention (from eqtl_analysis_cnn.py): delta_pred = P(alt) - P(ref).

Usage:
    python cnn/eqtl_pip_separation.py \\
        --high results/eqtl_delta_pred_pip_gt_0_9.tsv \\
        --low  results/eqtl_delta_pred_pip_lt_0_01.tsv \\
        --out-dir results/eqtl_pip_separation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GROUP_COLS = ["gene", "snp_pos"]   # "same position of the same transcript"
VALUE_COL  = "delta_pred"
BETA_COL   = "beta_marginal"


def filter_max_abs_beta(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Keep, per (gene, snp_pos), the single row with the largest |beta_marginal|.

    Rows lacking delta_pred or beta_marginal are dropped first."""
    before = len(df)
    df = df.dropna(subset=[VALUE_COL, BETA_COL]).copy()
    df["_abs_beta"] = df[BETA_COL].abs()
    idx = df.groupby(GROUP_COLS)["_abs_beta"].idxmax()
    out = df.loc[idx].drop(columns="_abs_beta").sort_index()
    print(f"  [{label}] {before:,} rows → {len(out):,} after max|beta| dedup "
          f"({df[GROUP_COLS].drop_duplicates().shape[0]:,} unique gene×pos)")
    return out


def mann_whitney(high: np.ndarray, low: np.ndarray, name: str) -> None:
    u, p = scipy_stats.mannwhitneyu(high, low, alternative="two-sided")
    n1, n2 = len(high), len(low)
    # Rank-biserial effect size = 1 - 2U/(n1*n2)  (a.k.a. common-language effect).
    rbc = 1.0 - (2.0 * u) / (n1 * n2)
    print(f"\n  Mann-Whitney U ({name}):  U={u:,.0f}  p={p:.3e}")
    print(f"    high: n={n1:,}  median={np.median(high):+.4f}  mean={high.mean():+.4f}")
    print(f"    low : n={n2:,}  median={np.median(low):+.4f}  mean={low.mean():+.4f}")
    print(f"    rank-biserial effect size = {rbc:+.4f}")


def correlate(df: pd.DataFrame, label: str) -> None:
    valid = df.dropna(subset=[VALUE_COL, BETA_COL])
    if len(valid) < 3:
        print(f"  [{label}] n={len(valid)} — too few for correlation.")
        return
    sp_r, sp_p = scipy_stats.spearmanr(valid[VALUE_COL], valid[BETA_COL])
    pe_r, pe_p = scipy_stats.pearsonr(valid[VALUE_COL], valid[BETA_COL])
    print(f"  [{label}] n={len(valid):,}  "
          f"Pearson r={pe_r:+.4f} (p={pe_p:.3e})  "
          f"Spearman r={sp_r:+.4f} (p={sp_p:.3e})")


def plot_histogram(high: np.ndarray, low: np.ndarray, path: Path) -> None:
    lo = min(high.min(), low.min())
    hi = max(high.max(), low.max())
    bins = np.linspace(lo, hi, 41)
    plt.figure(figsize=(8, 5))
    plt.hist(low,  bins=bins, alpha=0.55, density=True,
             label=f"pip < 0.01 (n={len(low):,})", color="#4C72B0")
    plt.hist(high, bins=bins, alpha=0.55, density=True,
             label=f"pip > 0.9 (n={len(high):,})", color="#C44E52")
    plt.axvline(0.0, color="grey", lw=0.8, ls="--")
    plt.xlabel("delta_pred  =  P(alt) - P(ref)")
    plt.ylabel("density")
    plt.title("delta_pred by PIP category")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Histogram written to: {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--high", required=True, help="delta_pred TSV for pip > 0.9")
    ap.add_argument("--low",  required=True, help="delta_pred TSV for pip < 0.01")
    ap.add_argument("--out-dir", default="results/eqtl_pip_separation",
                    help="Directory for filtered TSVs + histogram.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading delta_pred tables...")
    high = pd.read_csv(args.high, sep="\t")
    low  = pd.read_csv(args.low,  sep="\t")
    print(f"  high (pip>0.9): {len(high):,} rows   low (pip<0.01): {len(low):,} rows")

    print("\nFiltering (max |beta_marginal| per gene×position)...")
    high_f = filter_max_abs_beta(high, "high")
    low_f  = filter_max_abs_beta(low,  "low")

    # Drop (gene, snp_pos) shared with the high-PIP set from the low-PIP set. --
    high_keys = set(map(tuple, high_f[GROUP_COLS].itertuples(index=False, name=None)))
    low_mask  = ~low_f[GROUP_COLS].apply(tuple, axis=1).isin(high_keys)
    n_dropped = int((~low_mask).sum())
    low_f = low_f[low_mask]
    print(f"\n  Dropped {n_dropped:,} low-PIP rows overlapping high-PIP "
          f"(gene×position); low set now {len(low_f):,} rows.")

    high_f.to_csv(out_dir / "high_pip_filtered.tsv", sep="\t", index=False,
                  float_format="%.6f")
    low_f.to_csv(out_dir / "low_pip_filtered.tsv", sep="\t", index=False,
                 float_format="%.6f")

    # -- 3. Separation: Mann-Whitney + histogram on delta_pred ----------------
    hv = high_f[VALUE_COL].to_numpy()
    lv = low_f[VALUE_COL].to_numpy()
    print(f"\n{'='*60}\nSEPARATION  (delta_pred: pip>0.9 vs pip<0.01)\n{'='*60}")
    mann_whitney(hv, lv, "signed delta_pred")
    mann_whitney(np.abs(hv), np.abs(lv), "|delta_pred|")
    plot_histogram(hv, lv, out_dir / "delta_pred_hist.png")

    # -- 4. Correlation of delta_pred with beta_marginal ----------------------
    print(f"\n{'='*60}\nCORRELATION  (delta_pred vs beta_marginal)\n{'='*60}")
    combined = pd.concat([high_f, low_f], ignore_index=True)
    correlate(combined, "combined")
    correlate(high_f,   "high pip>0.9")
    correlate(low_f,    "low  pip<0.01")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
