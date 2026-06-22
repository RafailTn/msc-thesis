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

# Binding-type classification lives in a shared, numpy-only module so this script
# and cnn_branches_mirbind.py (negative undersampling) use one definition.
# Imported flat when run as a script from cnn/, package-style when imported as
# cnn.error_analysis.
try:
    from binding_types import (_antidiag_features, classify_binding_type)
except ImportError:
    from cnn.binding_types import (_antidiag_features, classify_binding_type)


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


def _effect_magnitude(d: float) -> str:
    """Verbal label for a |rank-biserial| / |Cliff's delta| effect size.

    Thresholds follow Romano et al. (2006), the conventional cut-offs for
    Cliff's delta: negligible < 0.147, small < 0.33, medium < 0.474, else large.
    """
    d = abs(d)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def _mw(a: pd.Series, b: pd.Series, label_a: str, label_b: str) -> str:
    """Mann-Whitney U test with rank-biserial effect size; formatted summary.

    rb (rank-biserial correlation, == Cliff's delta for two groups) = the
    probability that a random a exceeds a random b, minus the reverse, rescaled
    to [-1, +1].  It is derived directly from the U statistic:
        rb = 2*U / (n_a * n_b) - 1
    Sign is relative to `a`: rb > 0 means values in group `a` tend to be larger
    than in `b`.  Unlike the p-value it does NOT grow with sample size, so it
    separates "real but tiny" from "real and large" — exactly the failure mode
    of the huge manakov split where everything is significant.
    """
    a, b = a.dropna(), b.dropna()
    if len(a) < 5 or len(b) < 5:
        return "(too few samples)"
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    rb = 2 * u / (len(a) * len(b)) - 1
    return (f"median {label_a}={a.median():.3f}  {label_b}={b.median():.3f}  "
            f"MWU p={p:.2e}  rb={rb:+.3f} ({_effect_magnitude(rb)})")


def _load_expression(path: Path, value_col: str = "Mean_RPM") -> dict[str, float]:
    """
    Load the miRNA expression panel (media-2.xlsx) into a {miRNA_name: value}
    map.  The sheet has one row per miRNA with a 'Symbol_miRNA' key (e.g.
    'hsa-miR-16-5p') and several quantification columns; `value_col` selects
    which one to use (default Mean_RPM, library-size normalised).
    """
    expr = pd.read_excel(path)
    name_col = next((c for c in ("Symbol_miRNA", "miRNA", "noncodingRNA_name")
                     if c in expr.columns), None)
    if name_col is None:
        raise ValueError(f"No miRNA-name column found in {path}; "
                         f"columns are {list(expr.columns)}")
    if value_col not in expr.columns:
        raise ValueError(f"Value column {value_col!r} not in {path}; "
                         f"columns are {list(expr.columns)}")
    expr = expr[[name_col, value_col]].dropna()
    return dict(zip(expr[name_col].astype(str), expr[value_col].astype(float)))


def _attach_expression(df: pd.DataFrame, expr_map: dict[str, float]) -> pd.DataFrame:
    """
    Add `mirna_expr` (raw value) and `log10_expr` columns to df, keyed off the
    `noncodingRNA_name` column.  Many names are pipe-delimited ambiguous-mapping
    sets ('hsa-miR-23b-3p|hsa-miR-23c|hsa-miR-23a-3p'); for those we average the
    expression of whichever member names are present in the panel.  miRNAs with
    no panel entry stay NaN (silently ignored by the downstream MWU/median code).
    """
    name_col = next((c for c in ("noncodingRNA_name", "mirna_name")
                     if c in df.columns), None)
    if name_col is None:
        return df

    def _lookup(name: object) -> float:
        if not isinstance(name, str):
            return float("nan")
        vals = [expr_map[t] for t in name.split("|") if t in expr_map]
        return float(np.mean(vals)) if vals else float("nan")

    df = df.copy()
    df["mirna_expr"] = df[name_col].map(_lookup)
    df["log10_expr"] = np.log10(df["mirna_expr"].clip(lower=0) + 1.0)
    cov = 100 * df["mirna_expr"].notna().mean()
    print(f"    Expression attached: {cov:.1f}% of rows matched the panel.",
          file=sys.stderr)
    return df


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

def analyse(df: pd.DataFrame, name: str, out=sys.stdout,
            expr_map: dict[str, float] | None = None) -> None:
    print(f"\n{_SEP}", file=out)
    print(f"FILE: {name}", file=out)
    print(_SEP, file=out)

    # ── 0. Guard columns ─────────────────────────────────────────────────────
    if "error_type" not in df.columns:
        print("ERROR: 'error_type' column not found.", file=out)
        return

    if expr_map is not None:
        df = _attach_expression(df, expr_map)

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
                                 "interaction_probability",
                                 "mirna_expr", "log10_expr")
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

    # ── 7. miRNA expression × binding type × outcome ─────────────────────────
    # Directly tests the hypothesis that highly-expressed miRNAs in specific
    # binding types (e.g. 3prime.compensatory) are over-represented among FN
    # (model misses real, abundant interactions) and under-represented among FP.
    if "mirna_expr" in df.columns and df["mirna_expr"].notna().any():
        print(f"\n[7] miRNA expression (log10 Mean_RPM) by binding type × outcome",
              file=out)
        print(f"    Per binding type: median log10-expr per outcome, plus "
              f"Mann-Whitney U", file=out)
        print(f"    for the two key contrasts (FN vs TP among positives, "
              f"FP vs TN among negatives).", file=out)

        focus_types = [t for t in ("TP", "FN", "TN", "FP") if t in present]
        group_col = "binding_type" if "binding_type" in df.columns else None

        # Overall expression by outcome first.
        print(f"\n  ALL binding types:", file=out)
        for t in focus_types:
            s = df.loc[df["error_type"] == t, "log10_expr"].dropna()
            if len(s):
                print(f"    {t}: median log10-expr={s.median():.3f}  (n={len(s):,})",
                      file=out)
        if "FN" in focus_types and "TP" in focus_types:
            print(f"    FN vs TP: {_mw(df.loc[df.error_type=='TP','log10_expr'], df.loc[df.error_type=='FN','log10_expr'], 'TP', 'FN')}", file=out)
        if "FP" in focus_types and "TN" in focus_types:
            print(f"    FP vs TN: {_mw(df.loc[df.error_type=='TN','log10_expr'], df.loc[df.error_type=='FP','log10_expr'], 'TN', 'FP')}", file=out)

        if group_col:
            # Largest binding types first; skip tiny groups.
            order = df[group_col].value_counts()
            for btype in order.index:
                gsub = df[df[group_col] == btype]
                if len(gsub) < 20:
                    continue
                print(f"\n  binding_type = {btype}  (n={len(gsub):,})", file=out)
                for t in focus_types:
                    s = gsub.loc[gsub["error_type"] == t, "log10_expr"].dropna()
                    if len(s):
                        print(f"    {t}: median log10-expr={s.median():.3f}  "
                              f"(n={len(s):,})", file=out)
                if "FN" in focus_types and "TP" in focus_types:
                    print(f"    FN vs TP: "
                          f"{_mw(gsub.loc[gsub.error_type=='TP','log10_expr'], gsub.loc[gsub.error_type=='FN','log10_expr'], 'TP', 'FN')}",
                          file=out)
                if "FP" in focus_types and "TN" in focus_types:
                    print(f"    FP vs TN: "
                          f"{_mw(gsub.loc[gsub.error_type=='TN','log10_expr'], gsub.loc[gsub.error_type=='FP','log10_expr'], 'TN', 'FP')}",
                          file=out)

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
    parser.add_argument(
        "--expression", default="../media-2.xlsx",
        help="miRNA expression panel (xlsx) to join on noncodingRNA_name. "
             "Adds mirna_expr/log10_expr to the numeric analysis and a "
             "binding-type × expression × outcome section. "
             "Pass '' to disable (default: ../media-2.xlsx).",
    )
    parser.add_argument(
        "--expr-col", default="Mean_RPM",
        help="Column in the expression panel to use (default: Mean_RPM).",
    )
    args = parser.parse_args()

    expr_map = None
    if args.expression:
        p_expr = Path(args.expression)
        if p_expr.exists():
            try:
                expr_map = _load_expression(p_expr, value_col=args.expr_col)
                print(f"Loaded expression panel {p_expr} "
                      f"({len(expr_map):,} miRNAs, col={args.expr_col}).",
                      file=sys.stderr)
            except Exception as e:
                print(f"WARNING: could not load expression panel {p_expr}: {e}",
                      file=sys.stderr)
        else:
            print(f"WARNING: expression panel {p_expr} not found — "
                  f"skipping expression analysis.", file=sys.stderr)

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
        analyse(df, p.name, out=tee, expr_map=expr_map)

    if fh:
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
