#!/usr/bin/env python3
"""
Error analysis for miRNA–MRE CNN classifier error dumps.

Reads *_errors.tsv files (from cnn_branches_mirbind.py --error-dump),
classifies samples by binding type using the best antiparallel register of
the raw sequences (no IntaRNA structure needed), then reports whether
TP / FN / FP are separable by binding type, miRNA family, genomic feature,
and numeric duplex statistics.

Usage:
    python error_analysis.py
    python error_analysis.py --files results/test_errors.tsv results/leftout_errors.tsv
    python error_analysis.py --files results/test_errors.tsv --out results/analysis.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Watson-Crick / G·U pairing tables (RNA, 5-index so pad index 4 → 0)
# ---------------------------------------------------------------------------

_NUC = {"A": 0, "C": 1, "G": 2, "U": 3}
_WC  = np.zeros((5, 5), dtype=np.float32)
_GU  = np.zeros((5, 5), dtype=np.float32)
for _a, _b in [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")]:
    _WC[_NUC[_a], _NUC[_b]] = 1.0
for _a, _b in [("G", "U"), ("U", "G")]:
    _GU[_NUC[_a], _NUC[_b]] = 1.0

_MAX_MI = 30
_MAX_TI = 50


def _tok(seq: str, length: int) -> np.ndarray:
    out = np.full(length, 4, dtype=np.intp)
    for i, c in enumerate(seq.upper().replace("T", "U")[:length]):
        out[i] = _NUC.get(c, 4)
    return out


# ---------------------------------------------------------------------------
# Antidiagonal binding-type classification
# ---------------------------------------------------------------------------

def _antidiag_features(mirna_dna: str, mre_dna: str) -> dict:
    """
    Find the best antiparallel register (highest paired-nucleotide count) and
    extract the features needed for binding-type classification.
    Mirrors the _duplex_stats logic in cnn_branches_mirbind.py but returns
    richer per-position information for classification.
    """
    mi = _tok(mirna_dna, _MAX_MI)
    ti = _tok(mre_dna, _MAX_TI)

    wc   = _WC[mi[:, None], ti[None, :]]          # (30, 50)
    gu   = _GU[mi[:, None], ti[None, :]]
    pair = wc + gu

    P, Q   = np.indices((_MAX_MI, _MAX_TI))
    d_flat = (P + Q).ravel()
    pair_d = np.bincount(d_flat, weights=pair.ravel(),
                         minlength=_MAX_MI + _MAX_TI - 1)
    best = int(np.argmax(pair_d))

    mi_len = int((mi != 4).sum())
    ti_len = int((ti != 4).sum())

    p_wc = np.zeros(_MAX_MI, dtype=bool)
    p_gu = np.zeros(_MAX_MI, dtype=bool)
    for i in range(min(mi_len, _MAX_MI)):
        j = best - i
        if 0 <= j < ti_len and ti[j] != 4:
            p_wc[i] = bool(_WC[mi[i], ti[j]])
            p_gu[i] = bool(_GU[mi[i], ti[j]])

    any_pair = p_wc | p_gu

    # seed = miRNA positions 2–8, 0-indexed 1–7
    seed     = any_pair[1:8]
    seed_gu  = p_gu[1:8]

    def _max_consec(arr: np.ndarray) -> int:
        best_run = curr = 0
        for v in arr:
            curr = curr + 1 if v else 0
            best_run = max(best_run, curr)
        return best_run

    # start_match: first paired miRNA position (1-indexed)
    start_match = 99
    for i in range(mi_len):
        if any_pair[i]:
            start_match = i + 1
            break

    return {
        "consec_seed":     _max_consec(seed),
        "total_1_9":       int(any_pair[0:9].sum()),
        "start_match":     start_match,
        "gu_in_seed":      int(seed_gu.sum()),
        "eff_3prime":      int(p_wc[8:mi_len].sum()),    # WC-only outside seed
        "consec_centered": _max_consec(any_pair[4:16]),  # positions 5–16
        "pos1_A":          (len(mirna_dna) > 0 and mirna_dna[0].upper() == "A"),
    }


def classify_binding_type(mirna_dna: str, mre_dna: str) -> str:
    """
    Classify a miRNA–MRE pair using the best-antidiagonal register.

    Mirrors the scheme in feature_extraction.py but without IntaRNA
    dot-bracket structures: bulge tags (.target.bulge / .mirna.bulge)
    are not applicable in the strict-register model and are omitted.
    """
    f      = _antidiag_features(mirna_dna, mre_dna)
    consec = f["consec_seed"]
    start  = f["start_match"]
    pos1_A = f["pos1_A"]

    if f["total_1_9"] == 9:
        btype = "9mer"
    elif consec >= 5 and start < 4:
        btype = f"{consec}mer"
    else:
        btype = "seedless"

    if btype == "seedless":
        if f["consec_centered"] >= 8:
            return "centered"
        if start >= 13:
            return "3prime"
        if f["eff_3prime"] >= 6:
            return "3prime.compensatory"
        return "seedless"

    # canonical sub-types (mirroring feature_extraction.py)
    if btype == "8mer" and start == 2 and pos1_A:
        btype = "8mer1A"
    elif btype == "7mer" and start == 2 and pos1_A:
        btype = "8mer1A"
    elif btype == "6mer":
        if start == 3:
            btype = "offset6mer"
        elif start == 2 and pos1_A:
            btype = "7mer1A"

    if "mer" in btype:
        if f["gu_in_seed"] > 0:
            btype += ".GU"
        if f["eff_3prime"] >= 3:
            btype += ".3prime"

    return btype


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEP = "=" * 72


def _pct(n: int, total: int) -> str:
    return f"{n:>7,d}  ({100 * n / total if total else 0:.1f}%)"


def _primary_feature(feat: str) -> str:
    """Collapse composite feature strings like 'exon,intron' to primary type."""
    if not isinstance(feat, str):
        return "unknown"
    return feat.split(",")[0].strip()


def _mw(a: pd.Series, b: pd.Series, label_a: str, label_b: str) -> str:
    """Mann-Whitney U test; returns formatted summary line."""
    a, b = a.dropna(), b.dropna()
    if len(a) < 5 or len(b) < 5:
        return "(too few samples)"
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return (f"median {label_a}={a.median():.3f}  {label_b}={b.median():.3f}  "
            f"MWU p={p:.2e}")


def _chi2_pval(ct: pd.DataFrame) -> float:
    """Chi-square p-value from a contingency table (suppress warnings)."""
    try:
        chi2, p, *_ = stats.chi2_contingency(ct)
        return p
    except Exception:
        return float("nan")


def _enrichment_table(df: pd.DataFrame, col: str,
                      focus: str, ref: str,
                      df_all: pd.DataFrame = None,
                      top_n: int = 15, out=sys.stdout) -> None:
    """
    Print the top_n categories of `col` most enriched in `focus` relative to
    `ref`, using Fisher's exact test on the 2×2 table:

        [category vs rest] × [focus vs ref]

    The odds ratio naturally accounts for the overall prevalence of each
    category in the full dataset (df_all), so rare categories with accidentally
    skewed splits are ranked lower than genuinely enriched common ones.
    Fisher's p-value flags whether the enrichment is statistically reliable.
    """
    sub = df[df["error_type"].isin([focus, ref])]
    if sub.empty:
        return

    grp = (sub.groupby([col, "error_type"])
              .size()
              .unstack(fill_value=0)
              .reindex(columns=[ref, focus], fill_value=0))

    # Total count per category across the whole dataset (background prevalence)
    bg = (df_all if df_all is not None else df).groupby(col).size().rename("n_total")
    grp = grp.join(bg, how="left").fillna(0)

    total_focus = int(grp[focus].sum())
    total_ref   = int(grp[ref].sum())

    odds_ratios, pvals = [], []
    for _, row in grp.iterrows():
        a = int(row[focus])                 # this category, focus
        b = int(row[ref])                   # this category, ref
        c = total_focus - a                 # other categories, focus
        d = total_ref   - b                 # other categories, ref
        _, p = stats.fisher_exact([[a, b], [c, d]])
        or_val = (a * d) / (b * c) if b * c > 0 else (float("inf") if a > 0 else 1.0)
        odds_ratios.append(or_val)
        pvals.append(p)

    grp["odds_ratio"] = odds_ratios
    grp["fisher_p"]   = pvals

    top = grp.sort_values("odds_ratio", ascending=False).head(top_n)
    print(f"\n  Top {top_n} '{col}' enriched in {focus} vs {ref} "
          f"(Fisher's OR, background-aware):", file=out)
    hdr = (f"  {'Category':<35s} {ref:>8s} {focus:>8s}  "
           f"{'n_total':>8s}  {'odds_ratio':>10s}  fisher_p")
    print(hdr, file=out)
    print("  " + "-" * (len(hdr) - 2), file=out)
    for cat, row in top.iterrows():
        or_str = f"{row['odds_ratio']:>10.2f}x" if row['odds_ratio'] != float("inf") else "       inf"
        print(f"  {str(cat):<35s} {int(row[ref]):>8,d} {int(row[focus]):>8,d}  "
              f"{int(row['n_total']):>8,d}  {or_str}  {row['fisher_p']:.2e}", file=out)


# ---------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------

def analyse(df: pd.DataFrame, name: str, out=sys.stdout) -> None:
    print(f"\n{_SEP}", file=out)
    print(f"FILE: {name}", file=out)
    print(_SEP, file=out)

    # ── 0. Guard columns ─────────────────────────────────────────────────────
    if "error_type" not in df.columns:
        print("ERROR: 'error_type' column not found.", file=out)
        return

    present = [t for t in ("TP", "FN", "FP", "TN") if t in df["error_type"].values]
    counts  = df["error_type"].value_counts()
    total   = len(df)

    # ── 1. Binding-type classification ───────────────────────────────────────
    gene_col   = next((c for c in ("gene",   "mre_sequence",  "target_seq")
                       if c in df.columns), None)
    mirna_col  = next((c for c in ("noncodingRNA", "mirna_sequence", "query_seq")
                       if c in df.columns), None)

    if gene_col and mirna_col:
        print("\n[1] Classifying binding types (antidiagonal register) ...", file=out)
        df = df.copy()
        df["binding_type"] = [
            classify_binding_type(str(m), str(t))
            for m, t in zip(df[mirna_col], df[gene_col])
        ]
        print(f"    Done. Unique types: {df['binding_type'].nunique()}", file=out)
    else:
        print("\n[1] No sequence columns found — skipping binding-type classification.",
              file=out)

    # ── 2. Overall breakdown ─────────────────────────────────────────────────
    print(f"\n[2] Outcome breakdown  (total = {total:,})", file=out)
    for t in ("TP", "TN", "FN", "FP"):
        if t in counts:
            print(f"    {t}: {_pct(counts[t], total)}", file=out)

    pos_total = counts.get("TP", 0) + counts.get("FN", 0)
    neg_total = counts.get("TN", 0) + counts.get("FP", 0)
    if pos_total:
        print(f"    Sensitivity (TP rate): "
              f"{100*counts.get('TP',0)/pos_total:.1f}%", file=out)
    if neg_total:
        print(f"    Specificity (TN rate): "
              f"{100*counts.get('TN',0)/neg_total:.1f}%", file=out)

    # ── 3. Binding type separability ─────────────────────────────────────────
    if "binding_type" in df.columns:
        focus_types = [t for t in ("TP", "TN", "FN", "FP") if t in present]
        sub = df[df["error_type"].isin(focus_types)]

        ct = (sub.groupby(["binding_type", "error_type"])
                  .size()
                  .unstack(fill_value=0)
                  .reindex(columns=focus_types, fill_value=0))
        ct["total"] = ct.sum(axis=1)
        ct = ct.sort_values("total", ascending=False)

        p_val = _chi2_pval(ct[focus_types])
        print(f"\n[3] Binding type × outcome  (χ² p = {p_val:.2e})", file=out)
        col_w = max(len(c) for c in ct.index.astype(str)) + 2
        hdr = f"  {'binding_type':<{col_w}}" + "".join(f"{t:>9s}" for t in focus_types) + "  total"
        print(hdr, file=out)
        print("  " + "-" * (len(hdr) - 2), file=out)
        for btype, row in ct.iterrows():
            vals = "".join(f"{int(row[t]):>9,d}" for t in focus_types)
            print(f"  {str(btype):<{col_w}}{vals}  {int(row['total']):>6,d}", file=out)

        # Enrichment ratios (background = full file distribution)
        if "TP" in focus_types:
            _enrichment_table(df, "binding_type", "FN", "TP", df_all=df, top_n=12, out=out)
            if "FP" in focus_types:
                _enrichment_table(df, "binding_type", "FP", "TP", df_all=df, top_n=12, out=out)
            if "TN" in focus_types:
                _enrichment_table(df, "binding_type", "TN", "TP", df_all=df, top_n=12, out=out)
        if "TN" in focus_types:
            if "FP" in focus_types:
                _enrichment_table(df, "binding_type", "FP", "TN", df_all=df, top_n=12, out=out)
            if "FN" in focus_types:
                _enrichment_table(df, "binding_type", "FN", "TN", df_all=df, top_n=12, out=out)

    # ── 4. Genomic feature separability ──────────────────────────────────────
    feat_col = next((c for c in ("feature",) if c in df.columns), None)
    if feat_col:
        df["_primary_feat"] = df[feat_col].apply(_primary_feature)
        focus_types = [t for t in ("TP", "TN", "FN", "FP") if t in present]
        sub = df[df["error_type"].isin(focus_types)]

        ct = (sub.groupby(["_primary_feat", "error_type"])
                  .size()
                  .unstack(fill_value=0)
                  .reindex(columns=focus_types, fill_value=0))
        ct["total"] = ct.sum(axis=1)
        ct = ct.sort_values("total", ascending=False)

        p_val = _chi2_pval(ct[focus_types])
        print(f"\n[4] Genomic feature × outcome  (χ² p = {p_val:.2e})", file=out)
        col_w = max(len(str(c)) for c in ct.index) + 2
        hdr = f"  {'feature':<{col_w}}" + "".join(f"{t:>9s}" for t in focus_types) + "  total"
        print(hdr, file=out)
        print("  " + "-" * (len(hdr) - 2), file=out)
        for feat, row in ct.iterrows():
            vals = "".join(f"{int(row[t]):>9,d}" for t in focus_types)
            print(f"  {str(feat):<{col_w}}{vals}  {int(row['total']):>6,d}", file=out)

    # ── 5. miRNA family separability ─────────────────────────────────────────
    fam_col = next((c for c in ("noncodingRNA_fam", "mirna_family", "mir_fam")
                    if c in df.columns), None)
    if fam_col:
        focus_types = [t for t in ("TP", "TN", "FN", "FP") if t in present]
        sub = df[df["error_type"].isin(focus_types)]

        ct_fam = (sub.groupby([fam_col, "error_type"])
                     .size()
                     .unstack(fill_value=0)
                     .reindex(columns=focus_types, fill_value=0))
        ct_fam["total"] = ct_fam.sum(axis=1)
        p_val = _chi2_pval(ct_fam[focus_types])
        print(f"\n[5] miRNA family × outcome  "
              f"(χ² p = {p_val:.2e}, {ct_fam.shape[0]} families)", file=out)

        if "TP" in focus_types:
            _enrichment_table(df, fam_col, "FN", "TP", df_all=df, top_n=15, out=out)
            if "FP" in focus_types:
                _enrichment_table(df, fam_col, "FP", "TP", df_all=df, top_n=15, out=out)
            if "TN" in focus_types:
                _enrichment_table(df, fam_col, "TN", "TP", df_all=df, top_n=15, out=out)
        if "TN" in focus_types:
            if "FP" in focus_types:
                _enrichment_table(df, fam_col, "FP", "TN", df_all=df, top_n=15, out=out)
            if "FN" in focus_types:
                _enrichment_table(df, fam_col, "FN", "TN", df_all=df, top_n=15, out=out)

    # ── 6. Numeric feature separability ──────────────────────────────────────
    numeric_cols = [c for c in ("n_wc", "n_gu", "n_mm", "max_run", "seed_pairs",
                                 "mirna_len", "mre_len", "prob",
                                 "interaction_probability")
                    if c in df.columns]
    if numeric_cols:
        tp_mask = df["error_type"] == "TP"
        tn_mask = df["error_type"] == "TN"
        fn_mask = df["error_type"] == "FN"
        fp_mask = df["error_type"] == "FP"

        print(f"\n[6] Numeric features — Mann-Whitney U (two-sided)", file=out)

        if tp_mask.any() and fn_mask.any():
            print(f"\n  TP vs FN:", file=out)
            for col in numeric_cols:
                line = _mw(df.loc[tp_mask, col], df.loc[fn_mask, col], "TP", "FN")
                print(f"    {col:<30s} {line}", file=out)

        if tp_mask.any() and fp_mask.any():
            print(f"\n  TP vs FP:", file=out)
            for col in numeric_cols:
                line = _mw(df.loc[tp_mask, col], df.loc[fp_mask, col], "TP", "FP")
                print(f"    {col:<30s} {line}", file=out)

        if tn_mask.any() and fp_mask.any():
            print(f"\n  TN vs FP:", file=out)
            for col in numeric_cols:
                line = _mw(df.loc[tn_mask, col], df.loc[fp_mask, col], "TN", "FP")
                print(f"    {col:<30s} {line}", file=out)

        if tn_mask.any() and fn_mask.any():
            print(f"\n  TN vs FN:", file=out)
            for col in numeric_cols:
                line = _mw(df.loc[tn_mask, col], df.loc[fn_mask, col], "TN", "FN")
                print(f"    {col:<30s} {line}", file=out)

        if fn_mask.any() and fp_mask.any():
            print(f"\n  FN vs FP:", file=out)
            for col in numeric_cols:
                line = _mw(df.loc[fn_mask, col], df.loc[fp_mask, col], "FN", "FP")
                print(f"    {col:<30s} {line}", file=out)

    print(f"\n{_SEP}", file=out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyse CNN error dumps: binding type, family, feature separability."
    )
    parser.add_argument(
        "--files", nargs="+",
        default=[
            "results/test_errors.tsv",
            "results/leftout_errors.tsv",
        ],
        help="Error TSV files to analyse (default: results/test_errors.tsv "
             "results/leftout_errors.tsv)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Write report to this file in addition to stdout.",
    )
    args = parser.parse_args()

    outputs = [sys.stdout]
    fh = None
    if args.out:
        fh = open(args.out, "w")
        outputs.append(fh)

    class Tee:
        def write(self, msg):
            for o in outputs:
                o.write(msg)
        def flush(self):
            for o in outputs:
                o.flush()

    tee = Tee()

    for path in args.files:
        p = Path(path)
        if not p.exists():
            print(f"WARNING: {path} not found — skipping.", file=sys.stderr)
            continue
        print(f"Loading {p} ...", file=sys.stderr)
        df = pd.read_csv(p, sep="\t", low_memory=False)
        analyse(df, p.name, out=tee)

    if fh:
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
