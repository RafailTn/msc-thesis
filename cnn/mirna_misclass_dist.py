#!/usr/bin/env python3
"""
miRNA-name distribution across misclassified weak-seed samples.

For each error dump, classifies binding type from the raw sequences, restricts
to the weak-seed categories (seedless, 3prime.compensatory), and reports — per
category and per error type (FN, FP) — which miRNAs the misclassifications fall
on, each miRNA's misclassification *rate* within that category (to separate
"abundant" from "intrinsically hard"), and how concentrated the errors are.

Usage:
    python mirna_misclass_dist.py
    python mirna_misclass_dist.py --files a.tsv b.tsv --out results/dist.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from binding_types import classify_binding_type
except ImportError:
    from cnn.binding_types import classify_binding_type

CATS = ["seedless", "3prime.compensatory"]
_SEP = "=" * 78


def analyse(path: Path, out, top_n: int = 15) -> None:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df["binding_type"] = [classify_binding_type(str(m), str(t))
                          for m, t in zip(df["noncodingRNA"], df["gene"])]

    print(f"\n{_SEP}", file=out)
    print(f"FILE: {path.name}", file=out)
    print(_SEP, file=out)

    for cat in CATS:
        c = df[df["binding_type"] == cat]
        mis = c[c["error_type"].isin(["FN", "FP"])]
        if not len(c):
            print(f"\n{cat}: none in this file.", file=out)
            continue
        print(f"\n{cat}:  total={len(c):,}  "
              f"misclassified(FN+FP)={len(mis):,} ({100*len(mis)/len(c):.1f}%)  "
              f"unique miRNAs(mis)={mis['noncodingRNA_name'].nunique()}", file=out)

        for et in ["FN", "FP"]:
            sub = mis[mis["error_type"] == et]
            if not len(sub):
                continue
            print(f"\n  --- {et}  (n={len(sub):,}, "
                  f"{sub['noncodingRNA_name'].nunique()} unique names) ---", file=out)
            vc = sub["noncodingRNA_name"].value_counts().head(top_n)
            for name, n in vc.items():
                in_cat = int((c["noncodingRNA_name"] == name).sum())
                rate = 100 * n / in_cat if in_cat else 0.0
                print(f"    {name:<55s} {n:>5,}  "
                      f"({rate:>5.1f}% of its {in_cat:,} {cat})", file=out)

        vc_all = mis["noncodingRNA_name"].value_counts()
        for k in (5, 10, 20):
            cover = 100 * vc_all.head(k).sum() / len(mis) if len(mis) else 0.0
            print(f"  top-{k} miRNA names cover {cover:.1f}% of misclassified",
                  file=out)
    print(f"\n{_SEP}", file=out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="miRNA-name distribution across misclassified weak-seed samples.")
    ap.add_argument(
        "--files", nargs="+",
        default=[
            "results/manakov_test_errors_v7_restructure.tsv",
            "results/manakov_leftout_errors_v7_restructure.tsv",
        ],
        help="Error TSV dumps (default: manakov test + leftout).",
    )
    ap.add_argument("--out", default="results/mirna_misclass_dist.txt",
                    help="Write report here in addition to stdout.")
    ap.add_argument("--top-n", type=int, default=15,
                    help="Top miRNA names to list per error type (default: 15).")
    args = ap.parse_args()

    outputs = [sys.stdout]
    fh = open(args.out, "w") if args.out else None
    if fh:
        outputs.append(fh)

    class Tee:
        def write(self, m):
            for o in outputs:
                o.write(m)
        def flush(self):
            for o in outputs:
                o.flush()
    tee = Tee()

    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"WARNING: {p} not found — skipping.", file=sys.stderr)
            continue
        print(f"Loading {p} ...", file=sys.stderr)
        analyse(p, tee, top_n=args.top_n)

    if fh:
        fh.close()
        print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
