#!/usr/bin/env python3
"""
Evaluate an AutoGluon TabularPredictor on a labeled dataset.

Metrics reported:
  - ROC-AUC
  - Average Precision (PR-AUC)
  - Weighted ROC-AUC   (mir_fam inverse-frequency weights, matching training)
  - Weighted Average Precision

Usage:
    python3 evaluate_model.py \
        --model  path/to/predictor_dir \
        --data   path/to/labeled.csv \
        [--label     label]             # target column (default: label)
        [--sep       ,]                 # column separator (default: ,)
        [--pos       1]                 # positive class value (default: 1)
        [--fam-col   mir_fam]           # miRNA family column for weighting (default: mir_fam)
        [--dedup-col chimeric_sequence] # deduplicate on this column, keeping first occurrence
                                        # (default: chimeric_sequence)
        [--no-dedup]                    # skip deduplication entirely
"""

import argparse
import sys

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


def family_weights(mir_fam: pd.Series) -> np.ndarray:
    """
    Per-sample weights mirroring training: total / (n_families * family_count).
    The training clip(lower=100) is intentionally omitted here — it was a
    gradient-stability measure and would distort weights on the smaller eval set.
    """
    counts = mir_fam.value_counts()
    n_families = len(counts)
    total = len(mir_fam)
    return mir_fam.map(lambda x: total / (n_families * counts[x])).values.astype(float)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate an AutoGluon model: ROC-AUC and Average Precision."
    )
    parser.add_argument("--model", "-m", required=True,
                        help="Path to the saved AutoGluon TabularPredictor directory")
    parser.add_argument("--data", "-d", required=True,
                        help="Labeled CSV/TSV file (must contain the label column)")
    parser.add_argument("--label", default="label",
                        help="Name of the label column (default: label)")
    parser.add_argument("--sep", default=",",
                        help="Column separator (default: ,  use $'\\t' for TSV)")
    parser.add_argument("--pos", default=None,
                        help="Positive class value (default: auto-detect as 1)")
    parser.add_argument("--fam-col", default="mir_fam", dest="fam_col",
                        help="miRNA family column used for weighting (default: mir_fam)")
    parser.add_argument("--dedup-col", default="chimeric_sequence", dest="dedup_col",
                        help="Deduplicate on this column, keeping first occurrence "
                             "(default: chimeric_sequence)")
    parser.add_argument("--no-dedup", action="store_true", dest="no_dedup",
                        help="Skip deduplication entirely")
    args = parser.parse_args()

    # -- Load data ----------------------------------------------------------------
    print(f"Loading data: {args.data}")
    df = pd.read_csv(args.data, sep=args.sep)
    print(f"  Rows: {len(df)}  Columns: {df.shape[1]}")

    # -- Deduplication (matches training preprocessing) ---------------------------
    if not args.no_dedup:
        if args.dedup_col in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=[args.dedup_col], keep='first')
            dropped = before - len(df)
            print(f"  Dedup on '{args.dedup_col}': dropped {dropped} rows -> {len(df)} remaining")
        else:
            print(f"  Warning: dedup column '{args.dedup_col}' not found, skipping deduplication.")

    # -- Validate required columns ------------------------------------------------
    if args.label not in df.columns:
        print(f"Error: label column '{args.label}' not found.", file=sys.stderr)
        print(f"Available columns: {df.columns.tolist()}", file=sys.stderr)
        return 1

    if args.fam_col not in df.columns:
        print(f"Error: family column '{args.fam_col}' not found.", file=sys.stderr)
        print(f"Available columns: {df.columns.tolist()}", file=sys.stderr)
        return 1

    df[args.label] = df[args.label].astype(int)
    y_true  = df[args.label].values
    weights = family_weights(df[args.fam_col])
    X       = df.drop(columns=[args.label])

    # -- Load model ---------------------------------------------------------------
    print(f"Loading predictor: {args.model}")
    predictor = TabularPredictor.load(args.model)

    # -- Predict ------------------------------------------------------------------
    proba = predictor.predict_proba(X)

    if args.pos is not None:
        pos_col = int(args.pos) if args.pos.lstrip('-').isdigit() else args.pos
    else:
        pos_col = 1 if 1 in proba.columns else True

    if pos_col not in proba.columns:
        print(f"Error: positive class '{pos_col}' not in proba columns: {proba.columns.tolist()}",
              file=sys.stderr)
        return 1

    y_score = proba[pos_col].values
    y_pred  = predictor.predict(X).values

    # -- Metrics ------------------------------------------------------------------
    roc   = roc_auc_score(y_true, y_score)
    ap    = average_precision_score(y_true, y_score)
    roc_w = roc_auc_score(y_true, y_score, sample_weight=weights)
    ap_w  = average_precision_score(y_true, y_score, sample_weight=weights)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    pos_rate = (y_true == pos_col).mean()
    n_fams   = df[args.fam_col].nunique()

    print(f"\n{'='*52}")
    print("EVALUATION RESULTS")
    print(f"{'='*52}")
    print(f"Samples:                {len(y_true)}")
    print(f"Positive rate:          {pos_rate:.4f}  ({int(pos_rate * len(y_true))} / {len(y_true)})")
    print(f"miRNA families:         {n_fams}  (weighting column: {args.fam_col})")
    print(f"{'='*52}")
    print(f"ROC-AUC:                {roc:.4f}")
    print(f"Average Precision:      {ap:.4f}")
    print(f"Weighted ROC-AUC:       {roc_w:.4f}")
    print(f"Weighted Avg Precision: {ap_w:.4f}")
    print(f"{'='*52}")
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"                Pred 0    Pred 1")
    print(f"  Actual 0    {tn:>8}  {fp:>8}")
    print(f"  Actual 1    {fn:>8}  {tp:>8}")
    print(f"{'='*52}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
