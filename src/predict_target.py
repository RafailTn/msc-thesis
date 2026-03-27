import argparse
import subprocess
import os
import sys
import shutil
import tempfile
import warnings
import requests
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from autogluon.tabular import TabularPredictor

# The official UCSC URL for hg38 470-way conservation
_BW_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons470way/hg38.phastCons470way.bw"
_BW_LOCAL = "hg38.phastCons470way.bw"

# Script directory — all pipeline helpers are expected to sit next to this file
_HERE = Path(__file__).parent


# =============================================================================
# SUBPROCESS HELPER
# =============================================================================

def _run(cmd: list[str], step: str, timeout: int = 600) -> None:
    """
    Run a subprocess command (as a list — no shell=True needed).
    Raises RuntimeError with a clear message if the step fails.
    """
    print(f"[{step}] {' '.join(cmd)}")
    result = subprocess.run(cmd, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"Step '{step}' failed with exit code {result.returncode}.\n"
            f"Command: {' '.join(cmd)}"
        )


# =============================================================================
# BIGWIG DOWNLOAD
# =============================================================================

def _download_bigwig(dest: Path) -> None:
    """Stream-download the phastCons470way BigWig file."""
    print(f"Downloading BigWig from UCSC -> {dest} ...")
    r = requests.get(_BW_URL, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            fh.write(chunk)
    print("Download complete.")


# =============================================================================
# EXPLAINABILITY
# =============================================================================

def _drop_non_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only numeric columns - SHAP and permutation importance need floats."""
    return df.select_dtypes(include=[np.number])


def compute_global_importance(
    predictor: TabularPredictor,
    X: pd.DataFrame,
    num_shuffle_sets: int = 5,
) -> pd.DataFrame:
    """
    Global feature importance via AutoGluon's built-in permutation method.

    AutoGluon shuffles each feature `num_shuffle_sets` times and measures the
    mean drop in the model's scoring metric.  Returns a DataFrame sorted by
    importance descending, with columns:
        feature | permutation_importance | permutation_stddev | p_value

    The higher the importance, the more the model relies on that feature.
    """
    print(f"  Computing permutation importance ({num_shuffle_sets} shuffle sets) ...")
    fi = predictor.feature_importance(
        data=X,
        num_shuffle_sets=num_shuffle_sets,
        subsample_size=min(len(X), 2000),
    )
    # AutoGluon returns: index=feature, columns=[importance, stddev, ...]
    # Column names and presence of 'p_value' vary across AG versions.
    # rename_axis ensures the index column is always called 'feature'
    # regardless of whether the original index had a name.
    fi = fi.rename_axis("feature").reset_index()
    rename_map = {"importance": "permutation_importance",
                  "stddev": "permutation_stddev"}
    keep = ["feature", "permutation_importance", "permutation_stddev"]
    fi = fi.rename(columns=rename_map)
    if "p_value" in fi.columns:
        keep.append("p_value")
    else:
        fi["p_value"] = float("nan")   # placeholder so downstream code is stable
        keep.append("p_value")
    fi = fi[keep]
    return fi.sort_values("permutation_importance", ascending=False).reset_index(drop=True)


def compute_shap(
    predictor: TabularPredictor,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    n_background_clusters: int = 25,
    nsamples: int = 200,
) -> tuple[np.ndarray, float]:
    """
    SHAP values via KernelExplainer - fully model-agnostic.

    Returns
    -------
    shap_values : np.ndarray  shape (n_explain_samples, n_features)
    baseline : float expected model output over background (E[f(x)])
    """
    import shap

    pos_col = 1 if 1 in predictor.predict_proba(X_background.head(2)).columns else True

    def _predict_fn(arr: np.ndarray) -> np.ndarray:
        """Wrapper: ndarray -> positive-class probability vector."""
        df = pd.DataFrame(arr, columns=X_background.columns)
        return predictor.predict_proba(df)[pos_col].values

    print(f"Building SHAP background ({n_background_clusters} k-means clusters) ...")
    background = shap.kmeans(X_background, n_background_clusters)

    print(f"Computing SHAP for {len(X_explain)} samples "
          f"(nsamples={nsamples} per call) ...")
    explainer = shap.KernelExplainer(_predict_fn, background)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(X_explain, nsamples=nsamples, silent=True)

    return np.array(shap_values), float(explainer.expected_value)


def build_explanation_outputs(
    X_pred: pd.DataFrame,
    predictor: TabularPredictor,
    pos_proba: pd.Series,
    binary_labels: pd.Series,
    out_stem: Path,
    top_n: int = 3,
    max_shap_samples: int = 200,
    num_shuffle_sets: int = 5,
    nsamples_per_shap: int = 200,
    n_background_clusters: int = 25,
) -> dict[str, list]:
    """
    Run the full explainability stack and write two companion files.

    Outputs
    -------
    1. <out_stem>_global_importance.tsv
       One row per feature, columns:
         feature | permutation_importance | permutation_stddev | p_value
                 | mean_abs_shap | shap_rank | permutation_rank

       Two importance scores are included deliberately:
       - Permutation importance : how much model accuracy drops when the
         feature is shuffled.  Captures the feature's global relevance but can
         be inflated for correlated features.
       - Mean |SHAP| : average magnitude of a feature's contribution
         across all explained samples.  More granular and less sensitive to
         correlation, but estimated only on the positive predictions subset.

       Having both allows you to cross-validate: features that rank high on
       both metrics are the most trustworthy drivers.

    2. <out_stem>_shap_per_sample.tsv
       One row per explained sample (predicted positives), columns:
         row_index | interaction_probability | top1_feature | top1_shap |
         top2_feature | top2_shap | ... | topN_feature | topN_shap | shap_baseline

       The sign of each SHAP value tells you the direction of the feature's
       effect: positive pushes toward interaction, negative pushes away.

    Returns
    -------
    per_sample_cols : dict mapping column names to lists, aligned to the full
                      input length, for appending to the main results TSV.
                      Non-positive rows have empty strings / None values.
    """
    feat_cols = list(_drop_non_numeric(X_pred).columns)
    X_numeric = X_pred[feat_cols]

    # -- Global: permutation importance ----------------------------------------
    perm_fi = compute_global_importance(predictor, X_pred, num_shuffle_sets)

    # -- SHAP: explain predicted positives (capped for runtime) ----------------
    pos_mask = binary_labels == 1
    X_pos = X_numeric[pos_mask.values].reset_index(drop=True)

    if len(X_pos) == 0:
        print("No predicted positives - skipping SHAP.")
        return {}

    if len(X_pos) > max_shap_samples:
        print(f"{len(X_pos)} positives found; explaining a random sample of "
              f"{max_shap_samples} (set -explain-samples to change).")
        X_explain = X_pos.sample(n=max_shap_samples, random_state=42)
    else:
        X_explain = X_pos

    shap_vals, baseline = compute_shap(
        predictor,
        X_background = X_numeric,
        X_explain = X_explain,
        n_background_clusters = n_background_clusters,
        nsamples = nsamples_per_shap,
    )

    shap_df = pd.DataFrame(shap_vals, columns=feat_cols)

    # -- Global SHAP summary ---------------------------------------------------
    mean_abs_shap = shap_df.abs().mean().rename("mean_abs_shap")

    # set_index/join/reset_index: rename_axis guarantees the restored
    # column is always called 'feature', not 'index'.
    global_tbl = (
        perm_fi.set_index("feature")
        .join(mean_abs_shap, how="outer")
        .fillna(0.0)
        .rename_axis("feature")
        .reset_index()
    )
    global_tbl["shap_rank"] = (global_tbl["mean_abs_shap"]
                                       .rank(ascending=False).astype(int))
    global_tbl["permutation_rank"] = (global_tbl["permutation_importance"]
                                       .rank(ascending=False).astype(int))
    global_tbl = global_tbl.sort_values("mean_abs_shap", ascending=False)

    global_path = Path(str(out_stem) + "_global_importance.tsv")
    global_tbl.to_csv(global_path, sep="\t", index=False, float_format="%.6f")
    print(f"\n  Global importance -> {global_path}")
    print("Top 5 features (by mean |SHAP|):")
    for _, row in global_tbl.head(5).iterrows():
        print(f"{row['feature']:45s}  "
              f"SHAP={row['mean_abs_shap']:.4f} (rank {row['shap_rank']:>3})  "
              f"perm={row['permutation_importance']:.4f} (rank {row['permutation_rank']:>3})")

    # -- Per-sample top-N driver columns ---------------------------------------
    per_sample_top: dict[str, list] = {}
    for k in range(1, top_n + 1):
        per_sample_top[f"top{k}_feature"] = []
        per_sample_top[f"top{k}_shap"]    = []

    for i in range(len(shap_df)):
        row_abs = shap_df.iloc[i].abs()
        top_feats = row_abs.nlargest(top_n).index.tolist()
        for k, feat in enumerate(top_feats, start=1):
            per_sample_top[f"top{k}_feature"].append(feat)
            per_sample_top[f"top{k}_shap"].append(
                round(float(shap_df.iloc[i][feat]), 6)
            )
        for k in range(len(top_feats) + 1, top_n + 1):
            per_sample_top[f"top{k}_feature"].append("")
            per_sample_top[f"top{k}_shap"].append(0.0)

    proba_pos = pos_proba[pos_mask.values].reset_index(drop=True)
    if len(proba_pos) > max_shap_samples:
        proba_pos = proba_pos.sample(
            n=max_shap_samples, random_state=42).reset_index(drop=True)

    per_sample_df = pd.DataFrame(
        {"interaction_probability": proba_pos.values,
         **per_sample_top,
         "shap_baseline": [round(baseline, 6)] * len(shap_df)}
    )
    per_sample_path = Path(str(out_stem) + "_shap_per_sample.tsv")
    per_sample_df.to_csv(per_sample_path, sep="\t", index_label="row_index",
                         float_format="%.6f")
    print(f"Per-sample SHAP  -> {per_sample_path}")
    print(f"Baseline E[f(x)] = {baseline:.4f}  "
          f"(model output if all features are at their background mean)")

    # -- Align to full output length (non-positives get empty / None) ----------
    n_total = len(binary_labels)
    full_top: dict[str, list] = {
        f"top{k}_feature": [""] * n_total for k in range(1, top_n + 1)
    }
    full_top.update(
        {f"top{k}_shap": [None] * n_total for k in range(1, top_n + 1)}
    )

    pos_indices = [i for i, v in enumerate(pos_mask.values) if v]
    sampled_pos = pos_indices[:max_shap_samples]

    for shap_i, orig_i in enumerate(sampled_pos):
        if shap_i >= len(shap_df):
            break
        for k in range(1, top_n + 1):
            full_top[f"top{k}_feature"][orig_i] = per_sample_top[f"top{k}_feature"][shap_i]
            full_top[f"top{k}_shap"][orig_i] = per_sample_top[f"top{k}_shap"][shap_i]

    return full_top


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the miRNA target prediction pipeline end-to-end."
    )
    # -- Input / output --------------------------------------------------------
    parser.add_argument("-target_fasta", required=True,
                        help="FASTA file containing target/mRNA sequences")
    parser.add_argument("-query_fasta", required=True,
                        help="FASTA file containing query/miRNA sequences")
    parser.add_argument("-conservation_tsv",
                        help="TSV with conservation vectors (--conservation). "
                             "Mutually exclusive with -bigwig.")
    parser.add_argument("-bigwig",
                        help="phastCons BigWig file. If neither -conservation_tsv "
                             "nor -bigwig is given, hg38 470-way BigWig is "
                             "downloaded automatically.")
    parser.add_argument("-model", required=True,
                        help="Path to the saved AutoGluon TabularPredictor directory")
    parser.add_argument("-o", default="./results.tsv",
                        help="Output TSV file (default: ./results.tsv)")
    # -- Prediction ------------------------------------------------------------
    parser.add_argument("-threshold", type=float, default=0.5,
                        help="Probability threshold for positive interactions "
                             "(default: 0.5)")
    parser.add_argument("-threads", type=int, default=4,
                        help="Threads for IntaRNA (default: 4)")
    parser.add_argument("-keep_files", action="store_true",
                        help="Keep intermediate files after completion")
    # -- Explainability --------------------------------------------------------
    parser.add_argument("-explain", action="store_true",
                        help="Compute feature importance + SHAP explanations. "
                             "Requires: pip install shap. "
                             "Produces two companion files and adds top-N SHAP "
                             "driver columns to the main results TSV.")
    parser.add_argument("-explain-samples", type=int, default=200,
                        dest="explain_samples",
                        help="Max positive predictions to explain with SHAP. "
                             "Higher = more accurate but slower. (default: 200)")
    parser.add_argument("-explain-top-n", type=int, default=3,
                        dest="explain_top_n",
                        help="Top-N SHAP driver columns per sample in main output "
                             "(default: 3)")
    parser.add_argument("-shap-nsamples", type=int, default=200,
                        dest="shap_nsamples",
                        help="SHAP nsamples per explained row (default: 200)")
    parser.add_argument("-shap-clusters", type=int, default=25,
                        dest="shap_clusters",
                        help="k-means clusters for SHAP background (default: 15)")
    parser.add_argument("-perm-shuffles", type=int, default=5,
                        dest="perm_shuffles",
                        help="Shuffle sets for permutation importance (default: 5)")

    args = parser.parse_args()

    # -- Validate --------------------------------------------------------------
    if args.threads < 1:
        print("Error: -threads must be >= 1.", file=sys.stderr)
        return 1

    cpu_count = os.cpu_count() or 4
    if args.threads > cpu_count * 2:
        print(
            f"Warning: {args.threads} threads on a {cpu_count}-CPU system. "
            f"Consider {cpu_count}-{cpu_count * 2}.",
            file=sys.stderr,
        )

    if args.explain:
        try:
            import shap  # noqa: F401
        except ImportError:
            print("Error: -explain requires the `shap` package.\n"
                  "Install with: pip install shap", file=sys.stderr)
            return 1

    # -- Resolve BigWig --------------------------------------------------------
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

    # -- Temp directory --------------------------------------------------------
    tmp_dir = Path(tempfile.mkdtemp(prefix="mirna_pipeline_"))
    print(f"Temp directory: {tmp_dir}\n")

    try:
        # -- Step 1: IntaRNA standard ------------------------------------------
        intarna_out = tmp_dir / "intarna_results.tsv"
        intarna_ens_out = tmp_dir / "intarna_results_ensemble.tsv"
        _run(
            ["python3", str(_HERE / "intarna_parallel.py"),
             args.target_fasta, args.query_fasta,
             "-o", str(intarna_out),
             "--threads", str(args.threads)],
            step="intarna",
        )

        # -- Step 2: IntaRNA ensemble ------------------------------------------
        _run(
            ["python3", str(_HERE / "intarna_parallel.py"),
             args.target_fasta, args.query_fasta,
             "-o", str(intarna_ens_out),
             "--threads", str(args.threads),
             "--ensemble"],
            step="intarna-ensemble",
        )

        # -- Step 3: Merge -----------------------------------------------------
        merged_out = tmp_dir / "intarna_merged.tsv"
        _run(
            ["python3", str(_HERE / "merge_intarna.py"),
             "-m", str(intarna_out),
             "-e", str(intarna_ens_out),
             "-o", str(merged_out)],
            step="merge",
        )

        # -- Step 4: Best structure per pair -----------------------------------
        best_out = tmp_dir / "intarna_best.tsv"
        _run(
            ["python3", str(_HERE / "best_intarna.py"),
             "--intarna", str(merged_out),
             "--mre-fasta", args.target_fasta,
             "--mirna-fasta", args.query_fasta,
             "--output", str(best_out)],
            step="best-intarna",
        )

        # -- Step 5: Feature extraction ----------------------------------------
        features_out = tmp_dir / "samples_4_pred.csv"
        feat_cmd = [
            "python3", str(_HERE / "feature_extraction.py"),
            "--intarna", str(best_out),
            "--mre-fasta", args.target_fasta,
            "--mirna-fasta", args.query_fasta,
            "--output", str(features_out),
        ]
        if args.conservation_tsv:
            feat_cmd += ["--conservation", args.conservation_tsv]
        else:
            feat_cmd += ["--conservation", str(best_out)]
        if bigwig_path:
            feat_cmd += ["--bigwig", str(bigwig_path)]
        _run(feat_cmd, step="feature-extraction")

        # -- Step 6: Load model + predict --------------------------------------
        print(f"\nLoading predictor from: {args.model}")
        predictor  = TabularPredictor.load(args.model)
        input4pred = pl.read_csv(features_out).to_pandas()

        proba = predictor.predict_proba(input4pred)
        pos_col = 1 if 1 in proba.columns else True
        pos_proba = proba[pos_col]
        binary = (pos_proba >= args.threshold).astype(int)

        n_pos = int(binary.sum())
        n_total = len(binary)
        print(f"{n_pos} interactions above threshold {args.threshold} "
              f"out of {n_total} pairs.")

        # -- Step 7: Explainability (optional) ---------------------------------
        shap_cols: dict[str, list] = {}
        if args.explain:
            print("\n--- Explainability ---")
            out_stem  = Path(args.o).with_suffix("")
            shap_cols = build_explanation_outputs(
                X_pred = input4pred,
                predictor = predictor,
                pos_proba = pos_proba,
                binary_labels = binary,
                out_stem = out_stem,
                top_n = args.explain_top_n,
                max_shap_samples = args.explain_samples,
                num_shuffle_sets = args.perm_shuffles,
                nsamples_per_shap = args.shap_nsamples,
                n_background_clusters = args.shap_clusters,
            )

        # -- Step 8: Write main output -----------------------------------------
        seq_cols = [c for c in ("mre_seq", "mirna_seq") if c in input4pred.columns]

        extra_pl_cols = [
            pl.Series("interaction_probability",
                      [round(float(v), 5) for v in pos_proba.tolist()]),
            pl.Series("prediction", binary.tolist()),
        ]
        for col_name, col_vals in shap_cols.items():
            extra_pl_cols.append(pl.Series(col_name, col_vals))

        output = pl.from_pandas(input4pred[seq_cols]).with_columns(extra_pl_cols)
        output.write_csv(args.o, separator="\t")
        print(f"\nResults written to: {args.o}  ({n_total} rows, {output.width} columns)")

    finally:
        if args.keep_files:
            print(f"\nIntermediate files kept at: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"\nTemp directory removed: {tmp_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
