#!/usr/bin/env python3
"""
Downsample positives within each miRNA family so that no family has more
positives than negatives.  Families that already satisfy pos <= neg are
left untouched.

Usage
-----
    python balance_by_family.py \\
        --input  data/3utr_interactions.csv \\
        --output data/3utr_balanced.csv \\
        --family-col mirna_family \\
        --label-col  label \\
        --seed 42
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd


def balance(
    df: pd.DataFrame,
    family_col: str,
    label_col: str,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    keep_idx: list[np.ndarray] = []
    removed_pos = 0
    removed_neg = 0

    families = df[family_col].fillna("unknown").unique()
    for fam in sorted(families):
        mask = df[family_col].fillna("unknown") == fam
        sub  = df[mask]

        pos_idx = sub.index[sub[label_col] == 1].to_numpy()
        neg_idx = sub.index[sub[label_col] == 0].to_numpy()

        n_pos, n_neg = len(pos_idx), len(neg_idx)

        if n_pos > n_neg:
            chosen_pos = rng.choice(pos_idx, size=n_neg, replace=False)
            removed_pos += n_pos - n_neg
            print(f"  {fam:<40s}  pos {n_pos:>6d} → {n_neg:>6d}  "
                  f"neg {n_neg:>6d}  removed_pos {n_pos - n_neg:>6d}")
            keep_idx.append(chosen_pos)
            keep_idx.append(neg_idx)
        elif n_neg > n_pos:
            chosen_neg = rng.choice(neg_idx, size=n_pos, replace=False)
            removed_neg += n_neg - n_pos
            print(f"  {fam:<40s}  pos {n_pos:>6d}  "
                  f"neg {n_neg:>6d} → {n_pos:>6d}  removed_neg {n_neg - n_pos:>6d}")
            keep_idx.append(pos_idx)
            keep_idx.append(chosen_neg)
        else:
            keep_idx.append(pos_idx)
            keep_idx.append(neg_idx)

    kept = np.concatenate(keep_idx)
    kept.sort()
    result = df.loc[kept].reset_index(drop=True)

    orig_pos  = int((df[label_col] == 1).sum())
    orig_neg  = int((df[label_col] == 0).sum())
    final_pos = int((result[label_col] == 1).sum())
    final_neg = int((result[label_col] == 0).sum())

    print(f"\nSummary")
    print(f"  before : {len(df):>8d} rows  pos={orig_pos}  neg={orig_neg}  "
          f"ratio={orig_pos/max(orig_neg,1):.3f}")
    print(f"  after  : {len(result):>8d} rows  pos={final_pos}  neg={final_neg}  "
          f"ratio={final_pos/max(final_neg,1):.3f}")
    print(f"  removed: {removed_pos} positives, {removed_neg} negatives")

    return result


def main() -> int:
    p = argparse.ArgumentParser(
        description="Per-family positive downsampling to match negative count.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      required=True,  help="Input CSV/TSV")
    p.add_argument("--output",     required=True,  help="Output CSV")
    p.add_argument("--family-col", default="mirna_family", dest="family_col",
                   help="Column with miRNA family labels.")
    p.add_argument("--label-col",  default="label", dest="label_col",
                   help="Binary label column (0=negative, 1=positive).")
    p.add_argument("--seed",       type=int, default=42,
                   help="Random seed for reproducibility.")
    args = p.parse_args()

    sep = "\t" if args.input.endswith(".tsv") else ","
    df  = pd.read_csv(args.input, sep=sep)
    print(f"Read {len(df)} rows from {args.input}")

    for col in (args.family_col, args.label_col):
        if col not in df.columns:
            sys.exit(f"ERROR: column '{col}' not found. "
                     f"Available: {list(df.columns)}")

    print(f"\nPer-family downsampling (family_col='{args.family_col}', "
          f"only families with pos > neg shown):\n")
    result = balance(df, args.family_col, args.label_col, args.seed)

    result.to_csv(args.output, index=False)
    print(f"\nWrote {len(result)} rows → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
