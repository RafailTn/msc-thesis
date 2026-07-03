#!/usr/bin/env python3
"""
FN-vs-TP comparison of full-dataset-derived features (cov, Nreads, base
composition) recovered by match_full_dataset_errors.py.

For each numeric feature, runs a Mann-Whitney U test (FN < TP) plus effect
sizes (CLES / rank-biserial correlation / Cohen's d), matching the
methodology used for Nunique in plot_nunique_by_error.py.

Also writes the same numbers to results/manakov_feature_effect_sizes.tsv (overall,
per dataset) and results/manakov_mfe_by_nunique.tsv (mfe stratified by Nunique bin,
since mfe showed by far the largest effect and was checked for a Nunique confound).

Usage:
    python analyze_matched_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FILES = {
    "test": "results/manakov_test_errors_full_matched_keepfirst.tsv",
    "leftout": "results/manakov_leftout_errors_full_matched_keepfirst.tsv",
}
FEATURES = ["cov", "Nreads", "A_perc", "G_perc", "T_perc", "C_perc", "Ndups", "dups_perc", "mfe"]
NUNIQUE_CAP = 6


def effect_stats(fn: pd.Series, tp: pd.Series) -> dict:
    u, p = stats.mannwhitneyu(fn, tp, alternative="two-sided")
    n1, n2 = len(fn), len(tp)
    # scipy's U for mannwhitneyu(x, y) satisfies U/(n1*n2) == P(x > y) + 0.5*P(tie)
    cles = 1 - u / (n1 * n2)  # P(FN value < TP value); 0.5 = no effect
    rank_biserial = 2 * cles - 1
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


def main() -> int:
    overall_rows = []
    mfe_rows = []

    for label, path in FILES.items():
        p = Path(path)
        if not p.exists():
            print(f"WARNING: {p} not found — skipping.", file=sys.stderr)
            continue
        df = pd.read_csv(p, sep="\t", low_memory=False)
        print(f"\n=== {label} (n_FN={len(df[df.error_type=='FN']):,}, "
              f"n_TP={len(df[df.error_type=='TP']):,}) ===")
        for feat in FEATURES:
            fn = df.loc[df["error_type"] == "FN", feat].dropna()
            tp = df.loc[df["error_type"] == "TP", feat].dropna()
            s = effect_stats(fn, tp)
            print(
                f"{feat:8s} mean(FN)={s['mean_fn']:.4f} mean(TP)={s['mean_tp']:.4f} "
                f"median(FN)={s['median_fn']:.4f} median(TP)={s['median_tp']:.4f} | "
                f"MWU p={s['p']:.3e} CLES(P[FN<TP])={s['cles']:.4f} "
                f"rank_biserial={s['rank_biserial']:+.4f} cohens_d={s['cohens_d']:+.4f}"
            )
            overall_rows.append({"dataset": label, "feature": feat, **s})

        if "mfe" in df.columns and "Nunique" in df.columns:
            nu_bin = df["Nunique"].clip(upper=NUNIQUE_CAP)
            for nu, g in df.groupby(nu_bin):
                fn = g.loc[g["error_type"] == "FN", "mfe"].dropna()
                tp = g.loc[g["error_type"] == "TP", "mfe"].dropna()
                if len(fn) < 20 or len(tp) < 20:
                    continue
                s = effect_stats(fn, tp)
                nu_label = f"{NUNIQUE_CAP}+" if nu == NUNIQUE_CAP else str(nu)
                mfe_rows.append({"dataset": label, "nunique_bin": nu_label, **s})

    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)

    overall_out = outdir / "manakov_feature_effect_sizes.tsv"
    pd.DataFrame(overall_rows).to_csv(overall_out, sep="\t", index=False)
    print(f"\nWrote {overall_out}", file=sys.stderr)

    mfe_out = outdir / "manakov_mfe_by_nunique.tsv"
    pd.DataFrame(mfe_rows).to_csv(mfe_out, sep="\t", index=False)
    print(f"Wrote {mfe_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
