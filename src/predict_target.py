import argparse
import subprocess
import os
import sys
import shutil
import tempfile
import requests
import polars as pl
from pathlib import Path
from autogluon.tabular import TabularPredictor

# The official UCSC URL for hg38 470-way conservation
_BW_URL       = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons470way/hg38.phastCons470way.bw"
_BW_LOCAL     = "hg38.phastCons470way.bw"

# Script directory — all pipeline helpers are expected to sit next to this file
_HERE = Path(__file__).parent


def _run(cmd: list[str], step: str, timeout: int = 600) -> None:
    """
    Run a subprocess command (as a list — no shell=True needed).
    Raises RuntimeError with a clear message if the step fails.
    """
    print(f"  [{step}] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"Step '{step}' failed with exit code {result.returncode}.\n"
            f"Command: {' '.join(cmd)}"
        )


def _download_bigwig(dest: Path) -> None:
    """Stream-download the phastCons470way BigWig file."""
    print(f"Downloading BigWig from UCSC → {dest} …")
    r = requests.get(_BW_URL, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            fh.write(chunk)
    print("  Download complete.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the miRNA target prediction pipeline end-to-end."
    )
    parser.add_argument("-target_fasta",      required=True,
                        help="FASTA file containing target/mRNA sequences")
    parser.add_argument("-query_fasta",       required=True,
                        help="FASTA file containing query/miRNA sequences")
    parser.add_argument("-conservation_tsv",
                        help="TSV file with conservation vectors (--conservation arg). "
                             "Mutually exclusive with -bigwig.")
    parser.add_argument("-bigwig",
                        help="phastCons BigWig file (--bigwig arg). "
                             "If neither -conservation_tsv nor -bigwig is given, "
                             "the hg38 470-way BigWig is downloaded automatically.")
    parser.add_argument("-model",             required=True,
                        help="Path to the saved AutoGluon TabularPredictor directory")
    parser.add_argument("-keep_files",        action="store_true",
                        help="Keep intermediate files after completion")
    parser.add_argument("-o",                 default="./results.tsv",
                        help="Output TSV file (default: ./results.tsv)")
    parser.add_argument("-threshold",         type=float, default=0.5,
                        help="Probability threshold above which a prediction is "
                             "considered a positive interaction (default: 0.5)")
    parser.add_argument("-threads",           type=int,   default=4,
                        help="Threads to pass to IntaRNA (default: 4)")
    args = parser.parse_args()

    # ── Validate arguments ────────────────────────────────────────────────────
    if args.threads < 1:
        print("Error: --threads must be ≥ 1.", file=sys.stderr)
        return 1

    cpu_count = os.cpu_count() or 4
    if args.threads > cpu_count * 2:
        print(
            f"Warning: {args.threads} threads requested on a {cpu_count}-CPU system. "
            f"Consider {cpu_count}–{cpu_count * 2} for optimal performance.",
            file=sys.stderr,
        )

    # ── Resolve BigWig path (download if necessary) ───────────────────────────
    bigwig_path: Path | None = None
    if args.bigwig:
        bigwig_path = Path(args.bigwig)
    elif not args.conservation_tsv:
        local_bw = Path(_BW_LOCAL)
        if not local_bw.exists():
            _download_bigwig(local_bw)
        else:
            print(f"Using existing BigWig: {local_bw}")
        bigwig_path = local_bw

    # ── Temp directory: unique, cleaned up automatically ─────────────────────
    # tempfile.mkdtemp() creates a directory with a unique name under /tmp
    # (or the system's temp root), avoiding collisions between parallel runs.
    # A try/finally block guarantees cleanup regardless of how the pipeline exits.
    tmp_dir = Path(tempfile.mkdtemp(prefix="mirna_pipeline_"))
    print(f"Temp directory: {tmp_dir}")

    try:
        # ── Step 1: IntaRNA standard run ──────────────────────────────────────
        intarna_out      = tmp_dir / "intarna_results.tsv"
        intarna_ens_out  = tmp_dir / "intarna_results_ensemble.tsv"
        _run(
            ["python3", str(_HERE / "intarna_parallel.py"),
             args.target_fasta, args.query_fasta,
             "-o", str(intarna_out),
             "--threads", str(args.threads)],
            step="intarna",
        )

        # ── Step 2: IntaRNA ensemble run ──────────────────────────────────────
        _run(
            ["python3", str(_HERE / "intarna_parallel.py"),
             args.target_fasta, args.query_fasta,
             "-o", str(intarna_ens_out),
             "--threads", str(args.threads),
             "--ensemble"],
            step="intarna-ensemble",
        )

        # ── Step 3: Merge standard + ensemble results ─────────────────────────
        merged_out = tmp_dir / "intarna_merged.tsv"
        _run(
            ["python3", str(_HERE / "merge_intarna.py"),
             "-m", str(intarna_out),
             "-e", str(intarna_ens_out),
             "-o", str(merged_out)],
            step="merge",
        )

        # ── Step 4: Select best-scoring structure per pair ───────────────────
        best_out = tmp_dir / "intarna_best.tsv"
        _run(
            ["python3", str(_HERE / "best_intarna.py"),
             "--intarna",     str(merged_out),
             "--mre-fasta",   args.target_fasta,
             "--mirna-fasta", args.query_fasta,
             "--output",      str(best_out)],
            step="best-intarna",
        )

        # ── Step 5: Feature extraction ────────────────────────────────────────
        features_out = tmp_dir / "samples_4_pred.csv"
        feat_cmd = [
            "python3", str(_HERE / "feature_extraction.py"),
            "--intarna",     str(best_out),
            "--mre-fasta",   args.target_fasta,
            "--mirna-fasta", args.query_fasta,
            "--output",      str(features_out),
        ]
        if args.conservation_tsv:
            feat_cmd += ["--conservation", args.conservation_tsv]
        else:
            # Feature extractor still needs a conservation TSV for metadata;
            # pass a placeholder flag so it can fall back to BigWig-only mode.
            feat_cmd += ["--conservation", str(best_out)]   
        if bigwig_path:
            feat_cmd += ["--bigwig", str(bigwig_path)]
        _run(feat_cmd, step="feature-extraction")

        # ── Step 6: Load model and predict ───────────────────────────────────
        print(f"\nLoading predictor from: {args.model}")
        predictor = TabularPredictor.load(args.model)

        input4pred = pl.read_csv(features_out).to_pandas()

        # predict_proba returns a DataFrame with one column per class;
        # column "1" (or True) holds the positive-class probability.
        proba = predictor.predict_proba(input4pred)
        pos_col = 1 if 1 in proba.columns else True
        pos_proba = proba[pos_col]

        # Apply threshold to get binary labels
        binary_labels = (pos_proba >= args.threshold).astype(int)

        # ── Step 7: Build output ──────────────────────────────────────────────
        out_cols = ["mre_seq", "mirna_seq"]
        # Fall back gracefully if column names differ
        available = set(input4pred.columns)
        seq_cols  = [c for c in out_cols if c in available]

        output = (
            pl.from_pandas(input4pred[seq_cols])
            .with_columns([
                pl.Series("interaction_probability", pos_proba.tolist()),
                pl.Series("prediction",              binary_labels.tolist()),
            ])
        )
        output.write_csv(args.o, separator="\t")
        print(f"\nResults written to: {args.o}")
        print(f"  {output.filter(pl.col('prediction') == 1).height} interactions "
              f"above threshold {args.threshold} "
              f"out of {len(output)} pairs.")

    finally:
        # ── Cleanup ───────────────────────────────────────────────────────────
        if args.keep_files:
            print(f"\nIntermediate files kept at: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"\nTemp directory removed: {tmp_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
