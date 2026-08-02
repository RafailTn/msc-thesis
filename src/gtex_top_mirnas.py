"""Rank miRNAs by expression within each GTEx tissue / sub-tissue.

The per-tissue matrices in data/GTEx_v11_counts/{by_tissue,by_subtissue}/<T>/miRNA_counts.tsv
hold fractional read counts (multi-mapping reads are split across loci), 811 miRNAs x N
samples. Library sizes span ~90x within a single tissue, so counts are converted to CPM
per sample before any cross-sample summary.

Ranking is on the median CPM across the samples of a tissue (robust to the handful of
very deep / very shallow libraries), tie-broken on mean CPM.

Outputs, in results/gtex_mirna_expression/:
  top{N}_by_tissue.tsv / top{N}_by_subtissue.tsv   long format, one row per tissue x rank
  median_cpm_by_tissue.tsv / median_cpm_by_subtissue.tsv   full 811 x tissue matrix
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
COUNTS_DIR = REPO / "data" / "GTEx_v11_counts"
OUT_DIR = REPO / "results" / "gtex_mirna_expression"


def summarise_tissue(path):
    """Per-miRNA expression summaries for one tissue's count matrix."""
    counts = pd.read_csv(path, sep="\t", index_col=0)
    cpm = counts / counts.sum(axis=0) * 1e6

    log_cpm = np.log2(cpm + 1)
    summary = pd.DataFrame(
        {
            "median_cpm": cpm.median(axis=1),
            "mean_cpm": cpm.mean(axis=1),
            "mean_log2_cpm": log_cpm.mean(axis=1),
            "pct_library": counts.sum(axis=1) / counts.values.sum() * 100,
            "pct_samples_detected": (counts > 0).mean(axis=1) * 100,
        }
    )
    summary.index.name = "miRNA"
    return summary, counts.shape[1]


def run_level(level_dir, level_name, top_n):
    tissues = sorted(d for d in level_dir.iterdir() if d.is_dir())

    top_rows, median_cols = [], {}
    for tissue_dir in tissues:
        matrix = tissue_dir / "miRNA_counts.tsv"
        if not matrix.exists():
            print(f"  skipping {tissue_dir.name}: no miRNA_counts.tsv")
            continue

        summary, n_samples = summarise_tissue(matrix)
        median_cols[tissue_dir.name] = summary["median_cpm"]

        ranked = summary.sort_values(
            ["median_cpm", "mean_cpm"], ascending=False
        ).head(top_n)
        ranked = ranked.reset_index()
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        ranked.insert(0, "n_samples", n_samples)
        ranked.insert(0, "tissue", tissue_dir.name)
        top_rows.append(ranked)

        print(f"  {tissue_dir.name:<45} n={n_samples:<5} top1={ranked.miRNA[0]}")

    top = pd.concat(top_rows, ignore_index=True)
    medians = pd.DataFrame(median_cols)
    medians.index.name = "miRNA"

    top_path = OUT_DIR / f"top{top_n}_by_{level_name}.tsv"
    med_path = OUT_DIR / f"median_cpm_by_{level_name}.tsv"
    top.to_csv(top_path, sep="\t", index=False, float_format="%.4f")
    medians.to_csv(med_path, sep="\t", float_format="%.4f")
    print(f"  -> {top_path.relative_to(REPO)}  ({len(top)} rows)")
    print(f"  -> {med_path.relative_to(REPO)}  ({medians.shape[0]} x {medians.shape[1]})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument(
        "--levels",
        nargs="+",
        default=["tissue", "subtissue"],
        choices=["tissue", "subtissue"],
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for level in args.levels:
        print(f"[{level}]")
        run_level(COUNTS_DIR / f"by_{level}", level, args.top_n)


if __name__ == "__main__":
    main()
