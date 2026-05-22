#!/usr/bin/env python3
"""
Compute AutoGluon permutation feature importance on labeled data.

Usage
-----
python3 permutation_importance.py \
    --model  path/to/autogluon_model \
    --data   path/to/labeled_data.csv \
    --output results/permutation_importance.tsv \
    [--label      label]   # column name of the target (default: label)
    [--shuffles   10]      # shuffle sets per feature (default: 10)
    [--subsample  2000]    # max rows sampled (default: all)
    [--sep        ,]       # input file delimiter (default: auto-detect)
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from autogluon.tabular import TabularPredictor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AutoGluon permutation feature importance on labeled data."
    )
    parser.add_argument("--model",    required=True,
                        help="Path to saved AutoGluon TabularPredictor directory")
    parser.add_argument("--data",     required=True,
                        help="Labeled CSV/TSV file (must contain the label column)")
    parser.add_argument("--output",   default="permutation_importance.tsv",
                        help="Output TSV path (default: permutation_importance.tsv)")
    parser.add_argument("--label",    default="label",
                        help="Name of the target column (default: label)")
    parser.add_argument("--shuffles", type=int, default=10,
                        help="Shuffle sets per feature (default: 10)")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample this many rows for speed (default: use all)")
    parser.add_argument("--sep",      default=None,
                        help="Column delimiter — auto-detected from extension if omitted")
    args = parser.parse_args()

    # -- Infer separator from file extension if not given ----------------------
    data_path = Path(args.data)
    if args.sep is not None:
        sep = args.sep
    elif data_path.suffix in (".tsv", ".tab"):
        sep = "\t"
    else:
        sep = ","

    # -- Load data -------------------------------------------------------------
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, sep=sep)

    if args.label not in df.columns:
        print(f"Error: label column '{args.label}' not found in {data_path}.",
              file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        return 1

    n_total = len(df)
    if args.subsample and args.subsample < n_total:
        df = df.sample(n=args.subsample, random_state=42)
        print(f"Subsampled {args.subsample} / {n_total} rows.")
    else:
        print(f"Using all {n_total} rows.")

    # -- Load model ------------------------------------------------------------
    print(f"Loading predictor from: {args.model}")
    predictor = TabularPredictor.load(args.model)

    # -- Permutation importance ------------------------------------------------
    print(f"Computing permutation importance ({args.shuffles} shuffle sets) ...")
    np.random.seed(42)  # reset numpy random state to avoid numpy version incompatibilities
    fi = predictor.feature_importance(
        data=df,
        num_shuffle_sets=args.shuffles,
    )

    # AutoGluon returns a DataFrame indexed by feature name
    fi = fi.rename_axis("feature").reset_index()
    fi = fi.rename(columns={"importance": "permutation_importance",
                             "stddev":     "permutation_stddev"})

    keep = ["feature", "permutation_importance", "permutation_stddev"]
    if "p_value" in fi.columns:
        keep.append("p_value")
    fi = fi[keep].sort_values("permutation_importance", ascending=False).reset_index(drop=True)

    # -- Write output ----------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fi.to_csv(out_path, sep="\t", index=False, float_format="%.6f")
    print(f"\nPermutation importance written to: {out_path}")

    print("\nTop 10 features:")
    for _, row in fi.head(10).iterrows():
        p = f"  p={row['p_value']:.3f}" if "p_value" in fi.columns else ""
        print(f"  {row['feature']:45s}  {row['permutation_importance']:+.4f} "
              f"± {row['permutation_stddev']:.4f}{p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
