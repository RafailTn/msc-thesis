#!/usr/bin/env python3
"""
eqtl_analysis.py

For each unique (variant, miRBench-pair) entry in an eQTL TSV:
  1. Look up the 50-nt MRE + miRNA sequences from miRBench.
  2. Verify the SNP falls inside the MRE window and apply the substitution
     to build the ALT sequence.
  3. Run the full IntaRNA → feature-extraction pipeline for both REF and ALT.
  4. Score both with the AutoGluon model to get P(interaction).
  5. Compute delta_pred = P(alt) - P(ref).
  6. Aggregate the GTEx normalized effect size (NES) across tissues per pair
     and compute Spearman / Pearson correlation with delta_pred.
  7. Write a per-pair results TSV.

Usage:
    python eqtl_analysis.py \\
        --eqtl   data/V4-hg38.Gene-Links.eQTLs_with_miRBench_keys_3utr.tsv \\
        --mirbench data/V4-miRBench_datasets_combined.tsv \\
        --model  models/autogluon_final_model_nomirnaacc312 \\
        --bigwig data/hg38.phastCons470way.bw \\
        -o       results/eqtl_delta_pred_3utr.tsv \\
        --threads 8
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats as scipy_stats
from autogluon.tabular import TabularPredictor

_HERE = Path(__file__).parent

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _complement(base: str) -> str:
    return base.upper().translate(_COMPLEMENT)


def _run(cmd: list, step: str, timeout: int | None = None) -> None:
    print(f"[{step}] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"Step '{step}' failed with exit code {result.returncode}.\n"
            f"Command: {' '.join(str(c) for c in cmd)}"
        )


# =============================================================================
# PAIR CONSTRUCTION
# =============================================================================

def build_pairs(eqtl_df: pd.DataFrame, mirbench_df: pd.DataFrame) -> list[dict]:
    """
    Return one dict per unique (ENCODE_ID, miRBench_key) combination where the
    SNP overlaps the stored 50-nt MRE sequence and the reference base validates.

    Strand handling:
      +  strand  →  offset = snp_pos - mb_start  (0-based into stored seq)
      −  strand  →  stored sequence is rev-comp;
                    offset = mb_end - snp_pos
                    ref/alt bases must be complemented
    """
    mb_idx = (mirbench_df
              .drop_duplicates(subset="unique_key")
              .set_index("unique_key"))

    key_cols = ["ENCODE_ID", "miRBench_keys", "chr", "coordinates", "reference", "allele"]
    unique_pairs = eqtl_df[key_cols].drop_duplicates()

    pairs: list[dict] = []
    n_no_key = n_no_overlap = n_ref_mismatch = 0

    for _, row in unique_pairs.iterrows():
        key = int(row["miRBench_keys"])

        if key not in mb_idx.index:
            n_no_key += 1
            continue

        mb        = mb_idx.loc[key]
        mre_seq   = str(mb["gene"]).upper().replace("T", "U")
        mirna_seq = str(mb["noncodingRNA"]).upper().replace("T", "U")
        chrom     = str(mb["chr"])          # stored WITHOUT 'chr' prefix
        mb_start  = int(float(mb["start"])) # 1-based inclusive
        mb_end    = int(float(mb["end"]))   # 1-based inclusive
        strand    = str(mb["strand"])
        mirna_fam = str(mb.get("noncodingRNA_fam", ""))

        snp_pos  = int(row["coordinates"])  # 1-based genomic
        ref_base = str(row["reference"]).upper()
        alt_base = str(row["allele"]).upper()

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
            "encode_id"   : str(row["ENCODE_ID"]),
            "mirbench_key": key,
            "chr"         : chrom,
            "start"       : mb_start,
            "end"         : mb_end,
            "strand"      : strand,
            "mirna_fam"   : mirna_fam,
            "ref_mre_seq" : mre_seq,
            "alt_mre_seq" : alt_mre_seq,
            "mirna_seq"   : mirna_seq,
            "snp_pos"     : snp_pos,
            "ref_allele"  : ref_base,
            "alt_allele"  : alt_base,
        })

    print(f"  Valid pairs : {len(pairs):,}")
    print(f"  Skipped — key not in miRBench : {n_no_key:,}")
    print(f"  Skipped — SNP outside MRE     : {n_no_overlap:,}")
    print(f"  Skipped — ref base mismatch   : {n_ref_mismatch:,}")
    return pairs


# =============================================================================
# FILE WRITERS
# =============================================================================

def _write_fasta(pairs: list[dict], mre_key: str,
                 mre_path: Path, mirna_path: Path) -> None:
    with open(mre_path, "w") as fm, open(mirna_path, "w") as fq:
        for i, p in enumerate(pairs, 1):
            pid = f"pair_{i:06d}"
            fm.write(f">{pid}\n{p[mre_key]}\n")
            fq.write(f">{pid}\n{p['mirna_seq']}\n")


def _write_conservation_tsv(pairs: list[dict], path: Path) -> None:
    """
    Write minimal genomic-coordinate TSV for pyBigWig lookup.
    parse_conservation_tsv() in feature_extraction.py prepends 'chr' to the
    chr column, so store it WITHOUT that prefix here.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["chr", "start", "end", "strand", "noncodingRNA_fam", "label"])
        for p in pairs:
            w.writerow([p["chr"], p["start"], p["end"],
                        p["strand"], p["mirna_fam"], 0])


# =============================================================================
# PIPELINE RUNNER
# =============================================================================

def _load_mismatch_flags(mismatch_path: Path, n_pairs: int) -> np.ndarray:
    """Return bool array (shape n_pairs) marking pairs recovered via coord-mismatch fallback."""
    flags = np.zeros(n_pairs, dtype=bool)
    if mismatch_path.exists():
        df = pd.read_csv(str(mismatch_path), sep="\t")
        if "target_id" in df.columns:
            for tid in df["target_id"]:
                try:
                    idx = int(str(tid).split("_")[1]) - 1
                    if 0 <= idx < n_pairs:
                        flags[idx] = True
                except (IndexError, ValueError):
                    pass
    return flags


def _run_batch(pairs: list[dict], mre_key: str, label: str,
               tmp_dir: Path, threads: int, bigwig: str,
               fallback_mismatch: bool = False,
               timeout: int | None = None) -> tuple[Path, Path]:
    """
    Run IntaRNA (MFE + ensemble) → merge → best → feature-extraction
    for one batch (ref or alt).
    Returns (features_path, mismatch_log_path).
    mismatch_log_path exists only when fallback_mismatch=True and mismatches occurred.
    """
    d = tmp_dir / label
    d.mkdir(exist_ok=True)

    fasta_mre    = d / "mre.fasta"
    fasta_mir    = d / "mirna.fasta"
    cons_tsv     = d / "conservation.tsv"
    intarna      = d / "intarna.tsv"
    intarna_e    = d / "intarna_ens.tsv"
    merged       = d / "intarna_merged.tsv"
    best         = d / "intarna_best.tsv"
    features     = d / "features.csv"
    mismatch_log = d / "coord_mismatch.tsv"

    _write_fasta(pairs, mre_key, fasta_mre, fasta_mir)
    _write_conservation_tsv(pairs, cons_tsv)

    _run(["python3", str(_HERE / "intarna_parallel.py"),
          str(fasta_mre), str(fasta_mir),
          "-o", str(intarna), "--threads", str(threads)],
         step=f"{label}-intarna", timeout=timeout)

    _run(["python3", str(_HERE / "intarna_parallel.py"),
          str(fasta_mre), str(fasta_mir),
          "-o", str(intarna_e), "--threads", str(threads), "--ensemble"],
         step=f"{label}-intarna-ensemble", timeout=timeout)

    merge_cmd = ["python3", str(_HERE / "merge_intarna.py"),
                 "-m", str(intarna), "-e", str(intarna_e), "-o", str(merged)]
    if fallback_mismatch:
        merge_cmd += ["--fallback-on-mismatch", "--mismatch-log", str(mismatch_log)]
    _run(merge_cmd, step=f"{label}-merge", timeout=timeout)

    _run(["python3", str(_HERE / "best_intarna.py"),
          "--intarna", str(merged),
          "--mre-fasta", str(fasta_mre), "--mirna-fasta", str(fasta_mir),
          "--output", str(best)],
         step=f"{label}-best", timeout=timeout)

    _run(["python3", str(_HERE / "feature_extraction.py"),
          "--intarna", str(best),
          "--mre-fasta", str(fasta_mre), "--mirna-fasta", str(fasta_mir),
          "--conservation", str(cons_tsv), "--bigwig", str(bigwig),
          "--output", str(features)],
         step=f"{label}-features", timeout=timeout)

    return features, mismatch_log


# =============================================================================
# PREDICTION
# =============================================================================

def _get_probabilities(predictor: TabularPredictor,
                       features_path: Path,
                       n_pairs: int) -> np.ndarray:
    """
    Return a float array of shape (n_pairs,) with positive-class probabilities.
    Pairs for which IntaRNA found no interaction get NaN.

    target_id in the features CSV encodes the 1-based pair index as 'pair_XXXXXX'.
    """
    df = pl.read_csv(str(features_path)).to_pandas()

    pair_ids = df["target_id"].tolist() if "target_id" in df.columns else []
    drop_cols = ["label", "target_id", "query_id",
                 "binding_type", "mre_sequence", "mirna_sequence", "mir_fam"]
    df_pred = df.drop(columns=[c for c in drop_cols if c in df.columns])

    proba    = predictor.predict_proba(df_pred)
    pos_col  = 1 if 1 in proba.columns else True
    proba_vals = proba[pos_col].values

    out = np.full(n_pairs, np.nan)
    for feat_i, pid in enumerate(pair_ids):
        try:
            # 'pair_000001' → index 0
            idx = int(pid.split("_")[1]) - 1
            if 0 <= idx < n_pairs:
                out[idx] = proba_vals[feat_i]
        except (IndexError, ValueError):
            pass

    return out


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correlate model prediction change (REF vs ALT) with eQTL effect size."
    )
    parser.add_argument("--eqtl",     required=True,
                        help="eQTL TSV file (with miRBench_keys column)")
    parser.add_argument("--mirbench", required=True,
                        help="miRBench combined TSV (V4-miRBench_datasets_combined.tsv)")
    parser.add_argument("--model",    required=True,
                        help="AutoGluon TabularPredictor directory")
    parser.add_argument("--bigwig",   required=True,
                        help="phastCons BigWig file (hg38.phastCons470way.bw)")
    parser.add_argument("-o",         default="eqtl_results.tsv",
                        help="Output TSV (default: eqtl_results.tsv)")
    parser.add_argument("--threads",  type=int, default=4,
                        help="Threads for IntaRNA (default: 4)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Timeout in seconds for each pipeline step (default: none)")
    parser.add_argument("--fallback-mismatch", action="store_true",
                        help=(
                            "When MFE and ensemble never agree on any interaction "
                            "coordinates for a pair, fall back to an ID-only merge "
                            "using each mode's lowest-E row instead of dropping the pair. "
                            "Adds coord_mismatch_ref / coord_mismatch_alt columns to output."
                        ))
    parser.add_argument("--keep-files", action="store_true",
                        help="Keep intermediate temp files after completion")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    print("Loading eQTL data...")
    eqtl_df = pd.read_csv(args.eqtl, sep="\t")
    print(f"  {len(eqtl_df):,} rows, "
          f"{eqtl_df['ENCODE_ID'].nunique():,} unique variants, "
          f"{eqtl_df['miRBench_keys'].nunique():,} unique miRBench keys")

    print("Loading miRBench data...")
    mirbench_df = pd.read_csv(args.mirbench, sep="\t")
    print(f"  {len(mirbench_df):,} rows")

    print("Building REF/ALT pairs...")
    pairs = build_pairs(eqtl_df, mirbench_df)
    if not pairs:
        print("No valid pairs found. Exiting.", file=sys.stderr)
        return 1

    # -------------------------------------------------------------------------
    tmp_dir = Path(tempfile.mkdtemp(prefix="eqtl_analysis_"))
    print(f"Temp directory: {tmp_dir}\n")

    try:
        print(f"--- REF pipeline ({len(pairs):,} pairs) ---")
        ref_features, ref_mismatch_log = _run_batch(
            pairs, "ref_mre_seq", "ref", tmp_dir, args.threads, args.bigwig,
            fallback_mismatch=args.fallback_mismatch, timeout=args.timeout)

        print(f"\n--- ALT pipeline ({len(pairs):,} pairs) ---")
        alt_features, alt_mismatch_log = _run_batch(
            pairs, "alt_mre_seq", "alt", tmp_dir, args.threads, args.bigwig,
            fallback_mismatch=args.fallback_mismatch, timeout=args.timeout)

        # -------------------------------------------------------------------------
        print("\nLoading AutoGluon model...")
        predictor = TabularPredictor.load(args.model)

        print("Predicting REF probabilities...")
        ref_proba = _get_probabilities(predictor, ref_features, len(pairs))

        print("Predicting ALT probabilities...")
        alt_proba = _get_probabilities(predictor, alt_features, len(pairs))

        delta_pred = alt_proba - ref_proba  # NaN where IntaRNA returned no interaction

        ref_mismatch_flags = _load_mismatch_flags(ref_mismatch_log, len(pairs))
        alt_mismatch_flags = _load_mismatch_flags(alt_mismatch_log, len(pairs))

        # -------------------------------------------------------------------------
        # Assemble per-pair result table
        results = pd.DataFrame({
            "encode_id"          : [p["encode_id"]    for p in pairs],
            "mirbench_key"       : [p["mirbench_key"] for p in pairs],
            "chr"                : [p["chr"]          for p in pairs],
            "snp_pos"            : [p["snp_pos"]      for p in pairs],
            "ref_allele"         : [p["ref_allele"]   for p in pairs],
            "alt_allele"         : [p["alt_allele"]   for p in pairs],
            "p_ref"              : ref_proba,
            "p_alt"              : alt_proba,
            "delta_pred"         : delta_pred,
            "coord_mismatch_ref" : ref_mismatch_flags,
            "coord_mismatch_alt" : alt_mismatch_flags,
        })

        # Aggregate GTEx NES across tissues per (encode_id, mirbench_key)
        nes = (
            eqtl_df
            .groupby(["ENCODE_ID", "miRBench_keys"])["normalized_effect_size_GTEx_v8"]
            .agg(nes_median="median", nes_mean="mean", n_tissues="count")
            .reset_index()
            .rename(columns={"ENCODE_ID": "encode_id", "miRBench_keys": "mirbench_key"})
        )
        results = results.merge(nes, on=["encode_id", "mirbench_key"], how="left")

        # -------------------------------------------------------------------------
        # Correlation
        valid = results.dropna(subset=["delta_pred", "nes_median"])
        n_valid = len(valid)

        print(f"\n{'='*60}")
        print(f"  Total pairs  : {len(results):,}")
        print(f"  Valid (both predictions + NES) : {n_valid:,}")
        print(f"  No IntaRNA interaction (REF)   : {np.isnan(ref_proba).sum():,}")
        print(f"  No IntaRNA interaction (ALT)   : {np.isnan(alt_proba).sum():,}")
        if args.fallback_mismatch:
            print(f"  Coord-mismatch fallback (REF)  : {ref_mismatch_flags.sum():,}")
            print(f"  Coord-mismatch fallback (ALT)  : {alt_mismatch_flags.sum():,}")

        if n_valid >= 3:
            sp_r, sp_p = scipy_stats.spearmanr(valid["delta_pred"], valid["nes_median"])
            pe_r, pe_p = scipy_stats.pearsonr(valid["delta_pred"],  valid["nes_median"])
            print(f"\n  Spearman r = {sp_r:+.4f}  (p = {sp_p:.3e})")
            print(f"  Pearson  r = {pe_r:+.4f}  (p = {pe_p:.3e})")
        else:
            print(f"\n  Not enough valid pairs ({n_valid}) for correlation.")

        dp = delta_pred[~np.isnan(delta_pred)]
        if len(dp):
            print(f"\n  delta_pred  mean={dp.mean():+.4f}  std={dp.std():.4f}"
                  f"  range=[{dp.min():+.4f}, {dp.max():+.4f}]")
            print(f"  |delta_pred| > 0.10 : {(np.abs(dp) > 0.10).sum():,}")
            print(f"  |delta_pred| > 0.05 : {(np.abs(dp) > 0.05).sum():,}")
        print(f"{'='*60}")

        results.to_csv(args.o, sep="\t", index=False, float_format="%.6f")
        print(f"\nResults written to: {args.o}  ({len(results):,} rows)")

    finally:
        if args.keep_files:
            print(f"\nIntermediate files kept: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"\nTemp directory removed: {tmp_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
