#!/usr/bin/env python3
"""
eqtl_analysis_cnn.py

Sequence-only CNN scoring of GTEx eQTL variants against their overlapping
miRBench MRE fragments. Operates on the *merged* GTEx-eQTL × miRBench-fragment
TSVs (one self-contained file per PIP group), e.g.

    data/GTEx_v8_eQTLs_pip_gt_0_9_..._merged_with_miRBench_fragments.tsv
    data/GTEx_v8_eQTLs_pip_lt_0_01_..._merged_with_miRBench_fragments.tsv

For each unique (variant, MRE-fragment) pair:
  1. Read the 50-nt MRE sequence (gene_b) + miRNA sequence (noncodingRNA).
  2. Verify the SNP falls inside the MRE window (start_b..end_b) and apply the
     ref→alt substitution to build the ALT sequence.
  3. Score REF and ALT sequences with a sequence-only cnn_branches_mirbind.py
     checkpoint → P(interaction).
  4. Compute delta_pred = P(alt) - P(ref).

delta_pred is purely sequence-driven, so it is identical across tissues for the
same (variant, fragment). The score is therefore computed once per unique pair
and broadcast back onto *every* input row, preserving the per-tissue
pip / beta_marginal / tissue / gene columns needed by the downstream
separation/correlation analysis (see eqtl_pip_separation.py).

Allele convention: allele1 = REF, allele2 = ALT (so delta_pred = P(alt)-P(ref);
positive delta_pred ⇒ the alt allele increases predicted miRNA binding).

Usage:
    python cnn/eqtl_analysis_cnn.py \\
        --input data/GTEx_v8_eQTLs_pip_gt_0_9_..._merged_with_miRBench_fragments.tsv \\
        --checkpoint checkpoints/cnn_seqonly.pt \\
        -o results/eqtl_delta_pred_pip_gt_0_9.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from predict_cnn import chimeric_key, dedup_chimeric, score_dataframe  # noqa: E402

_COMPLEMENT = str.maketrans("ACGT", "TGCA")

# Column names in the merged GTEx × miRBench fragment file.
COL_SNP_POS  = "start"            # 1-based; == end == variant position for SNPs
COL_REF      = "allele1"          # reference allele
COL_ALT      = "allele2"          # alternative (effect) allele
COL_KEY      = "unique_key"       # miRBench fragment id
COL_MRE_SEQ  = "gene_b"           # 50-nt MRE sequence
COL_MIRNA    = "noncodingRNA"     # miRNA sequence
COL_MRE_START = "start_b"         # MRE window, 1-based inclusive
COL_MRE_END   = "end_b"
COL_STRAND   = "strand"
COL_CHR      = "chr"              # without 'chr' prefix
COL_FAM      = "noncodingRNA_fam"

# Pair-identity columns (carried into pairs + used to broadcast scores back).
_PAIR_KEY = ["chr", "snp_pos", "ref_allele", "alt_allele", "mirbench_key"]


def _complement(base: str) -> str:
    return base.upper().translate(_COMPLEMENT)


def _pair_id(chrom, snp_pos, ref, alt, key) -> str:
    return f"{chrom}|{snp_pos}|{ref}|{alt}|{key}"


# =============================================================================
# PAIR CONSTRUCTION
# =============================================================================

def build_pairs(df: pd.DataFrame) -> list[dict]:
    """
    Return one dict per unique (variant, MRE-fragment) pair where the SNP
    overlaps the stored 50-nt MRE sequence and the reference base validates.

    Strand handling:
      +  strand  →  offset = snp_pos - mb_start  (0-based into stored seq)
      −  strand  →  stored sequence is rev-comp;
                    offset = mb_end - snp_pos
                    ref/alt bases must be complemented
    """
    key_cols = [COL_CHR, COL_SNP_POS, COL_REF, COL_ALT, COL_KEY,
                COL_MRE_SEQ, COL_MIRNA, COL_MRE_START, COL_MRE_END,
                COL_STRAND, COL_FAM]
    unique_pairs = df[key_cols].drop_duplicates()

    pairs: list[dict] = []
    n_no_overlap = n_ref_mismatch = n_indel = 0

    for _, row in unique_pairs.iterrows():
        snp_pos  = int(float(row[COL_SNP_POS]))   # 1-based genomic
        ref_base = str(row[COL_REF]).upper()
        alt_base = str(row[COL_ALT]).upper()

        if len(ref_base) != 1 or len(alt_base) != 1:
            n_indel += 1
            continue

        key       = row[COL_KEY]
        mre_seq   = str(row[COL_MRE_SEQ]).upper().replace("T", "U")
        mirna_seq = str(row[COL_MIRNA]).upper().replace("T", "U")
        chrom     = str(row[COL_CHR])
        mb_start  = int(float(row[COL_MRE_START]))  # 1-based inclusive
        mb_end    = int(float(row[COL_MRE_END]))    # 1-based inclusive
        strand    = str(row[COL_STRAND])
        mirna_fam = str(row.get(COL_FAM, ""))

        if not (mb_start <= snp_pos <= mb_end):
            n_no_overlap += 1
            continue

        if strand == "+":
            offset     = snp_pos - mb_start
            ref_in_seq = ref_base
            alt_in_seq = alt_base
        else:
            offset     = mb_end - snp_pos
            ref_in_seq = _complement(ref_base)
            alt_in_seq = _complement(alt_base)

        mre_dna = mre_seq.replace("U", "T")

        if offset >= len(mre_dna):
            n_ref_mismatch += 1
            continue

        if mre_dna[offset].upper() != ref_in_seq.upper():
            n_ref_mismatch += 1
            continue

        alt_mre_seq = (
            mre_dna[:offset] + alt_in_seq + mre_dna[offset + 1:]
        ).upper().replace("T", "U")

        pairs.append({
            "chr"         : chrom,
            "snp_pos"     : snp_pos,
            "ref_allele"  : ref_base,
            "alt_allele"  : alt_base,
            "mirbench_key": key,
            "strand"      : strand,
            "mirna_fam"   : mirna_fam,
            "ref_mre_seq" : mre_seq,
            "alt_mre_seq" : alt_mre_seq,
            "mirna_seq"   : mirna_seq,
            "pair_id"     : _pair_id(chrom, snp_pos, ref_base, alt_base, key),
        })

    print(f"  Unique (variant, fragment) pairs : {len(unique_pairs):,}")
    print(f"  Valid pairs                      : {len(pairs):,}")
    print(f"  Skipped — indel (not SNP)        : {n_indel:,}")
    print(f"  Skipped — SNP outside MRE        : {n_no_overlap:,}")
    print(f"  Skipped — ref base mismatch      : {n_ref_mismatch:,}")
    return pairs


# =============================================================================
# CNN SEQUENCE-ONLY SCORING (per REF / ALT batch)
# =============================================================================

def _run_batch(pairs: list[dict], mre_key: str, label: str,
               args: argparse.Namespace) -> np.ndarray:
    """Sequence-only CNN scoring for one batch (REF or ALT).

    De-duplicates on the chimeric (miRNA+MRE) sequence keeping the first
    occurrence, scores the unique pairs, then maps probabilities back to every
    pair by chimeric key. Returns a float array (len == len(pairs)).
    """
    df = pd.DataFrame({
        "mirna_sequence": [p["mirna_seq"] for p in pairs],
        "mre_sequence"  : [p[mre_key]     for p in pairs],
    })
    deduped, n_removed = dedup_chimeric(df, "mre_sequence", "mirna_sequence")
    if n_removed:
        print(f"    [{label}] dedup: {n_removed} duplicate chimeric sequences "
              f"removed ({len(df)} → {len(deduped)} unique)")

    scored, _ = score_dataframe(
        args.checkpoint, deduped, device=args.device,
        mre_col="mre_sequence", mirna_col="mirna_sequence",
        batch_size=args.batch_size, num_workers=args.num_workers)

    prob_by_chimeric = {
        chimeric_key(r["mirna_sequence"], r["mre_sequence"]): float(r["interaction_probability"])
        for _, r in scored.iterrows()
    }
    return np.array([
        prob_by_chimeric.get(chimeric_key(p["mirna_seq"], p[mre_key]), np.nan)
        for p in pairs
    ])


def summarize(delta_pred: np.ndarray, n_rows: int, n_scored_rows: int) -> None:
    print(f"\n{'='*60}")
    print(f"  Input rows                       : {n_rows:,}")
    print(f"  Rows with a delta_pred           : {n_scored_rows:,}")
    dp = delta_pred[~np.isnan(delta_pred)]
    if len(dp):
        print(f"\n  delta_pred (unique pairs, n={len(dp):,})"
              f"  mean={dp.mean():+.4f}  std={dp.std():.4f}"
              f"  range=[{dp.min():+.4f}, {dp.max():+.4f}]")
        print(f"  |delta_pred| > 0.10 : {(np.abs(dp) > 0.10).sum():,}")
        print(f"  |delta_pred| > 0.05 : {(np.abs(dp) > 0.05).sum():,}")
    print(f"{'='*60}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sequence-only CNN REF-vs-ALT scoring of merged GTEx eQTL × "
                    "miRBench fragment files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="Merged GTEx eQTL × miRBench fragment TSV "
                             "(one PIP group per file).")
    parser.add_argument("--checkpoint", required=True,
                        help="Sequence-only cnn_branches_mirbind.py checkpoint (.pt)")
    parser.add_argument("-o", default="eqtl_results_cnn.tsv",
                        help="Output TSV (default: eqtl_results_cnn.tsv)")
    parser.add_argument("--device", default=None,
                        help="Torch device for CNN scoring (default: cuda if available).")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4, dest="num_workers")
    args = parser.parse_args()

    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    print("Loading merged eQTL × miRBench data...")
    df = pd.read_csv(args.input, sep="\t")
    print(f"  {len(df):,} rows, "
          f"{df['variant'].nunique():,} unique variants, "
          f"{df[COL_KEY].nunique():,} unique fragments, "
          f"{df['tissue'].nunique():,} tissues")

    print("Building REF/ALT pairs...")
    pairs = build_pairs(df)
    if not pairs:
        print("No valid pairs found. Exiting.", file=sys.stderr)
        return 1

    # -------------------------------------------------------------------------
    print(f"\n--- REF scoring ({len(pairs):,} pairs) ---")
    ref_proba = _run_batch(pairs, "ref_mre_seq", "ref", args)

    print(f"\n--- ALT scoring ({len(pairs):,} pairs) ---")
    alt_proba = _run_batch(pairs, "alt_mre_seq", "alt", args)

    delta_pred = alt_proba - ref_proba

    # Broadcast per-pair scores back onto every input row by pair_id. ---------
    score_by_pid = {
        p["pair_id"]: (ref_proba[i], alt_proba[i], delta_pred[i])
        for i, p in enumerate(pairs)
    }
    row_pid = (
        df[COL_CHR].astype(str) + "|"
        + df[COL_SNP_POS].astype(float).astype("Int64").astype(str) + "|"
        + df[COL_REF].astype(str).str.upper() + "|"
        + df[COL_ALT].astype(str).str.upper() + "|"
        + df[COL_KEY].astype(str)
    )
    nan3 = (np.nan, np.nan, np.nan)
    scores = [score_by_pid.get(pid, nan3) for pid in row_pid]
    df["p_ref"]      = [s[0] for s in scores]
    df["p_alt"]      = [s[1] for s in scores]
    df["delta_pred"] = [s[2] for s in scores]

    n_scored_rows = int(df["delta_pred"].notna().sum())
    summarize(delta_pred, len(df), n_scored_rows)

    out_path = Path(args.o)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False, float_format="%.6f")
    print(f"\nResults written to: {out_path}  ({len(df):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
