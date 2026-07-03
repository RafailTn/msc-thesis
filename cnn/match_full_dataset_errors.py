#!/usr/bin/env python3
"""
Match FN/TP samples from the CNN error dumps back to their originating rows
in the Manakov full dataset, via the concatenated (MRE seq + miRNA seq) key.

The error dumps (`gene` + `noncodingRNA`) carry the same sequences as the
full dataset's `seq.g` + `noncodingRNA_real_seq`, but not the raw assay
columns (Nunique, cov, Nreads, mfe, base composition) that were dropped
during restructuring. This script recovers those columns by key-matching.

The concatenated key is NOT unique in the full dataset: ~43k sequence pairs
recur at different genomic loci with different Nunique/cov/Nreads/mfe values.
Those ambiguous groups cannot be resolved by sequence alone, so (per user
decision) they are dropped entirely -- only error-file rows whose key is
unique in the full dataset are matched and kept.

Usage:
    python match_full_dataset_errors.py
    python match_full_dataset_errors.py --out results/manakov_errors_full_matched.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

FULL_DATASET = "data/AGO2_eCLIP_Manakov2022_full_dataset_v7.tsv"
ERROR_FILES = {
    "test": "results/manakov_test_errors_v7_restructure.tsv",
    "leftout": "results/manakov_leftout_errors_v7_restructure.tsv",
}
RETAIN_COLS = ["Nunique", "cov", "Nreads", "mfe", "A_perc", "G_perc", "T_perc", "C_perc",
               "Ndups", "dups_perc"]


def load_full_dataset(path: Path, dup_strategy: str = "drop") -> pd.DataFrame:
    usecols = ["seq.g", "noncodingRNA_real_seq"] + RETAIN_COLS
    full = pd.read_csv(path, sep="\t", low_memory=False, usecols=usecols)
    full["key"] = full["seq.g"].astype(str) + full["noncodingRNA_real_seq"].astype(str)

    n_total = len(full)
    dup_mask = full["key"].duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    n_dup_keys = int(full.loc[dup_mask, "key"].nunique())

    if dup_strategy == "drop":
        print(
            f"Full dataset: {n_total:,} rows, {full['key'].nunique():,} unique keys. "
            f"{n_dup_keys:,} keys are ambiguous ({n_dup_rows:,} rows) -- dropping them.",
            file=sys.stderr,
        )
        unique_full = full.loc[~dup_mask, ["key"] + RETAIN_COLS].set_index("key")
    else:  # "first"
        print(
            f"Full dataset: {n_total:,} rows, {full['key'].nunique():,} unique keys. "
            f"{n_dup_keys:,} keys are ambiguous ({n_dup_rows:,} rows) -- keeping first "
            f"occurrence (file order) of each.",
            file=sys.stderr,
        )
        unique_full = (
            full.drop_duplicates(subset="key", keep="first")[["key"] + RETAIN_COLS]
            .set_index("key")
        )
    return unique_full


def match_error_file(path: Path, dataset_label: str, unique_full: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    sub = df[df["error_type"].isin(["TP", "FN"])].copy()
    sub["key"] = sub["gene"].astype(str) + sub["noncodingRNA"].astype(str)
    sub = sub.drop(columns=[c for c in RETAIN_COLS if c in sub.columns])

    merged = sub.join(unique_full, on="key", how="inner")
    n_before = len(sub)
    n_after = len(merged)
    print(
        f"{dataset_label}: {n_before:,} FN/TP rows -> {n_after:,} matched "
        f"({n_before - n_after:,} dropped: ambiguous or unmatched key).",
        file=sys.stderr,
    )

    out = merged[["gene", "noncodingRNA", "error_type"] + RETAIN_COLS].copy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full-dataset", default=FULL_DATASET)
    ap.add_argument("--test-errors", default=ERROR_FILES["test"])
    ap.add_argument("--leftout-errors", default=ERROR_FILES["leftout"])
    ap.add_argument("--test-out", default="results/manakov_test_errors_full_matched.tsv")
    ap.add_argument("--leftout-out", default="results/manakov_leftout_errors_full_matched.tsv")
    ap.add_argument("--dup-strategy", choices=["drop", "first"], default="drop",
                    help="How to resolve sequence-key duplicates in the full dataset: "
                         "drop ambiguous groups entirely, or keep the first occurrence "
                         "(file order) of each (default: drop).")
    args = ap.parse_args()

    unique_full = load_full_dataset(Path(args.full_dataset), dup_strategy=args.dup_strategy)

    for path, label, out_arg in (
        (args.test_errors, "test", args.test_out),
        (args.leftout_errors, "leftout", args.leftout_out),
    ):
        result = match_error_file(Path(path), label, unique_full)
        out_path = Path(out_arg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, sep="\t", index=False)
        print(f"Wrote {out_path} ({len(result):,} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
