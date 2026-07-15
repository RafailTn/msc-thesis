#!/usr/bin/env python3
"""
Featurewiz feature selection over the full feature superset.

RUNS IN ITS OWN ENVIRONMENT, not the pixi env: featurewiz pins dependencies that
conflict with the rest of the pipeline. Hence this script only ever reads and writes
CSV/JSON - it imports nothing from src/, so it needs neither IntaRNA nor ViennaRNA.
Its dependencies are just featurewiz, polars, pandas, numpy and scikit-learn.

Input:  the `--all-features` output of src/feature_extraction.py (train/test/leftout).
Output: the features selected in *every* fold, written to --output as a JSON list
        that `feature_extraction.py --features-file` reads back directly. Paste the
        same list into SELECTED_FEATURES to make it the default.

    python feature_selection_featurewiz/feature_selection.py \
        --train   data/manakov_train_all.csv \
        --test    data/manakov_test_all.csv \
        --leftout data/manakov_leftout_all.csv \
        --output  data/selected_features.json

NaN handling. The duplex stacking energies are NaN wherever the statistic is
genuinely undefined (no 3' supplementary pairing, std of a single stack, ...), which
is a large fraction of `mirna_3p_*`. Featurewiz's XGBoost stage routes NaN natively,
but the LogisticRegression probe used to score each fold does not, so the probe runs
behind a median imputer. The imputation exists only to score the selection; it never
touches the features written to --output, and AutoGluon handles the NaN itself.
"""

import json
import argparse

import numpy as np
import pandas as pd
import polars as pl
from featurewiz import FeatureWiz
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score

# Identifiers, sequences and v7 passthrough columns: carried in the feature CSV for
# traceability and error analysis, never fed to the model.
NON_FEATURE_COLS = [
    'target_id', 'query_id', 'hybrid_dp', 'subseq_dp',
    'mre_sequence', 'mirna_sequence', 'chimeric_sequence', 'energy_source',
    'gene', 'noncodingRNA', 'noncodingRNA_name', 'noncodingRNA_fam', 'feature',
    'chr', 'start', 'end', 'strand', 'gene_cluster_ID',
    'gene_phyloP', 'gene_phastCons', 'label_right',
]


def probe():
    """Linear probe used to score a fold's selection. Imputes only for scoring."""
    return make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )


def load(path: str) -> pd.DataFrame:
    df = pl.read_csv(path, infer_schema_length=10000)
    df = df.drop([c for c in NON_FEATURE_COLS if c in df.columns])
    return df.to_dummies('binding_type').to_pandas()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--leftout', required=True)
    p.add_argument('--output', required=True, help='JSON list of selected features')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    train = load(args.train)
    test = load(args.test)
    leftout = load(args.leftout)

    # Align the binding_type dummies: a category absent from test/leftout becomes 0.
    test = test.reindex(columns=train.columns, fill_value=0)
    leftout = leftout.reindex(columns=train.columns, fill_value=0)

    X = train.drop(columns=['label', 'mir_fam']).astype(np.float32)
    X_test = test.drop(columns=['label', 'mir_fam']).astype(np.float32)
    X_leftout = leftout.drop(columns=['label', 'mir_fam']).astype(np.float32)

    y = train['label'].values
    y_test = test['label'].values
    y_leftout = leftout['label'].values
    groups = train['mir_fam'].values

    print(f"Train {X.shape}, test {X_test.shape}, leftout {X_leftout.shape}")

    nan_frac = X.isna().mean().sort_values(ascending=False)
    nan_cols = nan_frac[nan_frac > 0]
    print(f"\n{len(nan_cols)} feature(s) carry NaN (undefined statistics, expected):")
    for name, frac in nan_cols.head(15).items():
        print(f"  {frac:6.1%}  {name}")
    if len(nan_cols) > 15:
        print(f"  ... and {len(nan_cols) - 15} more")

    # A feature that is NaN everywhere carries nothing, and would break the probe's
    # imputer (median of an empty column).
    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        print(f"\nWARNING: dropping {len(all_nan)} all-NaN feature(s): {all_nan}")
        X, X_test, X_leftout = (d.drop(columns=all_nan) for d in (X, X_test, X_leftout))

    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    per_fold, fwiz_last = [], None

    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups), 1):
        fwiz = FeatureWiz(feature_engg='', nrows=None, transform_target=True, scalers="std",
                          category_encoders="auto", add_missing=False, verbose=0,
                          imbalanced=False, ae_options={})
        X_tr_sel, y_tr_sel = fwiz.fit_transform(X.iloc[tr], pd.Series(y[tr], name='label'))
        X_va_sel = fwiz.transform(X.iloc[va])

        model = probe().fit(X_tr_sel, y_tr_sel)
        ap = average_precision_score(y[va], model.predict_proba(X_va_sel)[:, 1])
        print(f"Fold {fold}: {len(fwiz.features):3d} features, validation APS = {ap:.4f}")

        per_fold.append(fwiz.features)
        fwiz_last = fwiz

    common = sorted(set(per_fold[0]).intersection(*per_fold[1:]))
    print(f"\nStable across all {args.folds} folds: {len(common)} features")

    X_t = fwiz_last.transform(X)
    X_te = fwiz_last.transform(X_test)
    X_lo = fwiz_last.transform(X_leftout)

    aps_test, aps_leftout = [], []
    for feats in per_fold:
        m = probe().fit(X_t[feats], y)
        aps_test.append(average_precision_score(y_test, m.predict_proba(X_te[feats])[:, 1]))
        aps_leftout.append(average_precision_score(
            y_leftout, m.predict_proba(X_lo[feats])[:, 1]))

    m = probe().fit(X_t[common], y)
    ap_test = average_precision_score(y_test, m.predict_proba(X_te[common])[:, 1])
    ap_leftout = average_precision_score(y_leftout, m.predict_proba(X_lo[common])[:, 1])

    print(f"\nAPS test    : per-fold mean {np.mean(aps_test):.4f} | common {ap_test:.4f}")
    print(f"APS leftout : per-fold mean {np.mean(aps_leftout):.4f} | common {ap_leftout:.4f}")

    with open(args.output, 'w') as f:
        json.dump(common, f, indent=2)
    print(f"\nWrote {len(common)} selected features to {args.output}")
    print("Paste this list into SELECTED_FEATURES in src/feature_extraction.py to "
          "make it the default.")


if __name__ == '__main__':
    main()
