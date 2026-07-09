#!/usr/bin/env python
"""Test whether specific RBPs co-localise with false-negative miRNA target sites
more than with true-positive ones.

Rationale: a bound RBP can occlude or remodel an AGO2 target site.  If the CNN
systematically misses (FN) positives that sit inside a particular RBP's binding
footprint, that RBP is a candidate confounder / competing factor.  We compare
the FN vs TP positive sites (both label==1) from an `*_errors_v7_restructure.tsv`
error dump.

Pipeline
--------
1.  Take the FN and TP rows (label==1 positives the model got wrong / right).
2.  Extend each 50-nt target site's genomic coordinates by ``--flank`` nt on each
    side using the SAME genomic extension as ``compute_accessibility.py``'s
    ``ContextBuilder.window`` genomic branch (gws = s-1-flank, gwe = e+flank,
    0-based half-open).  ENCORI peaks are genomic, so we extend in genomic space
    (the MANE-spliced branch can't be overlapped against genomic peaks linearly).
3.  Strand-aware overlap of the extended sites with RBP peaks via PyRanges.  Two
    interchangeable peak sources, exactly one of which must be given:
      --encori   ENCORI narrow peaks (``ENCORI_RBP_targets_*.tsv``), 73 RBPs,
                 1-based inclusive coordinates.
      --bed-dir  iSHAPE per-RBP BED9 files named ``<RBP>_<CELLLINE>.bed``, 171
                 RBPs for HEK293T, 0-based half-open coordinates.
    The two panels share only 27 RBPs, so they are independent peak sets rather
    than a filter of one another; results are written to distinct files (see --tag).
4.  Per RBP, a 2x2 Fisher exact test of (site overlaps RBP) x (FN vs TP), with a
    Benjamini-Hochberg FDR correction.  Positive log2 odds-ratio => enriched in FN.

    python cnn/rbp_enrichment_fn_vs_tp.py \
        --input data/manakov_test_errors_v7_restructure.tsv \
        --encori ENCORI_RBP_targets_HEK293T_hg38.tsv \
        --flank 150 --out results/rbp_enrichment_manakov_test.tsv

    python cnn/rbp_enrichment_fn_vs_tp.py \
        --input data/manakov_test_errors_v7_restructure.tsv \
        --bed-dir data/RBP_from_ishape --cell-line HEK293T \
        --control-density --region UTR3
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import polars as pl
import pyranges as pr
from scipy.stats import fisher_exact

MRE_LEN = 50


def pl_to_pd(df: pl.DataFrame) -> pd.DataFrame:
    """polars -> pandas without pyarrow (not installed in this env)."""
    return pd.DataFrame({c: df[c].to_numpy() for c in df.columns})


def tsv_chrom_to_fa(chrom: str) -> str:
    """Map the v7 TSV chromosome label to an ENCORI/GENCODE contig name (chr...)."""
    c = str(chrom)
    if c in ("MT", "M", "chrMT"):
        return "chrM"
    return c if c.startswith("chr") else "chr" + c


def load_encori(path: str) -> pr.PyRanges:
    """ENCORI RBP peaks -> stranded PyRanges (Chromosome, Start, End, Strand, RBP).

    Some rows carry swapped/garbled narrow coordinates (narrowStart > narrowEnd),
    so we normalise to Start=min, End=max and drop rows with a null strand/coord.
    """
    e = pl.read_csv(path, separator="\t", infer_schema_length=10000,
                    columns=["RBP", "chromosome", "narrowStart", "narrowEnd", "strand"])
    e = e.drop_nulls(["chromosome", "narrowStart", "narrowEnd", "strand"])
    lo = pl.min_horizontal("narrowStart", "narrowEnd")
    hi = pl.max_horizontal("narrowStart", "narrowEnd")
    e = e.with_columns(Start=lo, End=(hi + 1))  # half-open
    df = e.select(
        Chromosome="chromosome", Start="Start", End="End",
        Strand="strand", RBP="RBP",
    )
    df = pl_to_pd(df)
    df = df[df["Strand"].isin(["+", "-"])]
    return pr.PyRanges(df)


def load_bed_dir(path: str, cell_line: str) -> pr.PyRanges:
    """iSHAPE per-RBP BED9 -> stranded PyRanges (Chromosome, Start, End, Strand, RBP).

    Reads ``<path>/<RBP>_<cell_line>.bed``.  BED is already 0-based half-open, so
    (unlike :func:`load_encori`, whose source is 1-based inclusive) the coordinates
    are taken verbatim -- no +1 on End.  The RBP name is taken from the *filename*
    rather than the BED name column, which carries the redundant ``RBP_CELLLINE``.
    """
    files = sorted(glob.glob(os.path.join(path, f"*_{cell_line}.bed")))
    if not files:
        raise SystemExit(f"no *_{cell_line}.bed files under {path}")
    frames = []
    for f in files:
        rbp = os.path.basename(f).rsplit(f"_{cell_line}.bed", 1)[0]
        b = pl.read_csv(f, separator="\t", has_header=False,
                        columns=[0, 1, 2, 5],
                        new_columns=["Chromosome", "Start", "End", "Strand"])
        frames.append(b.with_columns(RBP=pl.lit(rbp)))
    e = pl.concat(frames).drop_nulls()
    df = pl_to_pd(e.select("Chromosome", "Start", "End", "Strand", "RBP"))
    df = df[df["Strand"].isin(["+", "-"])]
    return pr.PyRanges(df)


def sites_pyranges(df: pl.DataFrame, flank: int) -> pr.PyRanges:
    """Extend each site by +/-flank in genomic space and return a stranded
    PyRanges carrying site_id + error_type."""
    start = df["start"].cast(pl.Int64)
    end = df["end"].cast(pl.Int64)
    ext_start = (start - 1 - flank).clip(lower_bound=0)  # 0-based half-open
    ext_end = end + flank
    chrom = df["chr"].map_elements(tsv_chrom_to_fa, return_dtype=pl.Utf8)
    out = pl.DataFrame({
        "Chromosome": chrom,
        "Start": ext_start,
        "End": ext_end,
        "Strand": df["strand"],
        "site_id": np.arange(df.height, dtype=np.int64),
        "error_type": df["error_type"],
    })
    return pr.PyRanges(pl_to_pd(out))


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    return q


def _load_positives(path: str, region: str | None,
                    binding_type: str | None = None) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", infer_schema_length=5000)
    df = df.filter(pl.col("error_type").is_in(["FN", "TP"]))
    if region:
        df = df.filter(pl.col("dominant_region") == region)
    if binding_type:
        try:
            from binding_types import classify_binding_type
        except ModuleNotFoundError:
            from cnn.binding_types import classify_binding_type
        bt = [classify_binding_type(m, t) for m, t in
              zip(df["noncodingRNA"].to_list(), df["gene"].to_list())]
        df = df.filter(pl.Series("_bt", bt) == binding_type)
    return df


def _overlap_pairs(df: pl.DataFrame, encori: pr.PyRanges, flank: int,
                   strand: bool):
    """Return (pairs, meta): distinct (site_id, RBP) overlaps and a per-site
    meta frame (site_id, et, fam, tot=# distinct RBPs overlapping the site)."""
    gr_sites = sites_pyranges(df, flank)
    jdf = gr_sites.join(encori, strandedness=None, suffix="_rbp").df
    if strand:
        jdf = jdf[jdf["Strand"].astype(str) == jdf["Strand_rbp"].astype(str)]
    pairs = jdf[["site_id", "RBP"]].drop_duplicates()
    n = df.height
    tot = (pairs.groupby("site_id").size()
           .reindex(range(n), fill_value=0).to_numpy())
    meta = pd.DataFrame({
        "site_id": np.arange(n, dtype=np.int64),
        "et": df["error_type"].to_numpy(),
        "fam": df["noncodingRNA_fam"].to_numpy().astype(str),
        "tot": tot,
    })
    return pairs, meta


def _cmh(y: np.ndarray, x: np.ndarray, strata: np.ndarray):
    """Cochran-Mantel-Haenszel pooled odds-ratio of exposure x on outcome y,
    stratified by `strata`.  Returns (mh_or, p_value).  y,x are 0/1 arrays."""
    num = den = a_sum = e_sum = v_sum = 0.0
    for s in np.unique(strata):
        m = strata == s
        N = int(m.sum())
        if N < 2:
            continue
        ys, xs = y[m], x[m]
        a = int(((xs == 1) & (ys == 1)).sum())
        b = int(((xs == 1) & (ys == 0)).sum())
        c = int(((xs == 0) & (ys == 1)).sum())
        d = int(((xs == 0) & (ys == 0)).sum())
        num += a * d / N
        den += b * c / N
        r1 = a + b            # exposed
        c1 = a + c            # outcome==1
        a_sum += a
        e_sum += r1 * c1 / N
        v_sum += r1 * (N - r1) * c1 * (N - c1) / (N * N * (N - 1))
    if den <= 0 or v_sum <= 0:
        return np.nan, 1.0
    mh_or = num / den
    from scipy.stats import chi2
    chi = (abs(a_sum - e_sum) - 0.5) ** 2 / v_sum  # continuity-corrected
    return mh_or, float(chi2.sf(chi, 1))


def analyse_density_controlled(path: str, encori: pr.PyRanges, flank: int,
                               strand: bool, min_family: int, min_hits: int,
                               n_bins: int, region: str | None = None,
                               binding_type: str | None = None) -> pd.DataFrame:
    """Per-(miRNA family, RBP) FN-vs-TP enrichment CONTROLLED for total RBP
    occupancy.  Within each family, sites are stratified into `n_bins` bins of
    equal size by their total distinct-RBP count, and each RBP is tested with a
    CMH pooled OR across those bins.  An RBP that stays FN-enriched here predicts
    FN *beyond* the site simply sitting in an RBP-dense region -> a candidate
    miRNA-specific cooperative effect rather than a bulk-accessibility artefact.
    """
    df = _load_positives(path, region, binding_type)
    print(f"\n=== {os.path.basename(path)}  [by family, density-controlled] ===")
    pairs, meta = _overlap_pairs(df, encori, flank, strand)

    fam_tot = meta.groupby("fam")["et"].value_counts().unstack(fill_value=0)
    fam_tot = fam_tot.reindex(columns=["FN", "TP"], fill_value=0)
    keep_fams = fam_tot[(fam_tot["FN"] >= min_family) &
                        (fam_tot["TP"] >= min_family)].index
    print(f"families tested (>= {min_family} FN and TP each): {len(keep_fams)}; "
          f"stratifying on total-RBP count into {n_bins} bins")

    rows = []
    for fam in keep_fams:
        sub = meta[meta["fam"] == fam].reset_index(drop=True)
        sids = sub["site_id"].to_numpy()
        pos = {sid: i for i, sid in enumerate(sids)}
        isfn = (sub["et"].to_numpy() == "FN").astype(int)
        nb = min(n_bins, sub["tot"].nunique())
        if nb < 2:
            continue
        bins = pd.qcut(sub["tot"].rank(method="first"), nb,
                       labels=False).to_numpy()
        fp = pairs[pairs["site_id"].isin(set(sids))]
        for rbp, grp in fp.groupby("RBP"):
            ov = np.zeros(len(sub), dtype=int)
            ov[[pos[s] for s in grp["site_id"]]] = 1
            fn_hit = int(((ov == 1) & (isfn == 1)).sum())
            tp_hit = int(((ov == 1) & (isfn == 0)).sum())
            if fn_hit + tp_hit < min_hits:
                continue
            mh_or, p = _cmh(isfn, ov, bins)
            if not np.isfinite(mh_or):
                continue
            rows.append({
                "fam": fam, "RBP": rbp,
                "n_fn": int(fam_tot.loc[fam, "FN"]),
                "n_tp": int(fam_tot.loc[fam, "TP"]),
                "n_overlap": fn_hit + tp_hit,
                "mh_odds_ratio": mh_or,
                "log2_mh_or": float(np.log2(mh_or)),
                "p_value": p,
                "direction": "FN-enriched" if mh_or > 1 else "TP-enriched",
            })
    res = pd.DataFrame(rows)
    if res.empty:
        print("no (family, RBP) cells passed the count filters")
        return res
    res["q_value"] = bh_fdr(res["p_value"].to_numpy())
    res = res.sort_values(["q_value", "log2_mh_or"],
                          ascending=[True, False]).reset_index(drop=True)
    return res


def analyse(path: str, encori: pr.PyRanges, flank: int, strand: bool,
            min_hits: int, region: str | None = None,
            binding_type: str | None = None) -> pd.DataFrame:
    df = _load_positives(path, region, binding_type)
    n_fn = df.filter(pl.col("error_type") == "FN").height
    n_tp = df.filter(pl.col("error_type") == "TP").height
    print(f"\n=== {os.path.basename(path)} ===")
    print(f"positives: FN={n_fn}  TP={n_tp}")

    gr_sites = sites_pyranges(df, flank)
    et = df["error_type"].to_numpy()  # index == site_id

    # overlapping (site, RBP) pairs. We join strand-agnostically (pyranges'
    # strand-aware join trips over the unused '.' strand category) and enforce
    # same-strand on the result via the RBP strand suffix column.
    joined = gr_sites.join(encori, strandedness=None, suffix="_rbp")
    jdf = joined.df
    if jdf.empty:
        print("no overlaps found")
        return pd.DataFrame()
    if strand:
        jdf = jdf[jdf["Strand"].astype(str) == jdf["Strand_rbp"].astype(str)]

    pairs = jdf[["site_id", "RBP"]].drop_duplicates()
    pairs = pairs.assign(et=et[pairs["site_id"].to_numpy()])
    n_fn_any = pairs.loc[pairs["et"] == "FN", "site_id"].nunique()
    n_tp_any = pairs.loc[pairs["et"] == "TP", "site_id"].nunique()
    print(f"sites overlapping >=1 RBP: FN={n_fn_any} ({n_fn_any/n_fn:.1%})  "
          f"TP={n_tp_any} ({n_tp_any/n_tp:.1%})")
    rows = []
    for rbp, grp in pairs.groupby("RBP"):
        fn_hit = int((grp["et"] == "FN").sum())
        tp_hit = int((grp["et"] == "TP").sum())
        if fn_hit + tp_hit < min_hits:
            continue
        table = [[fn_hit, n_fn - fn_hit], [tp_hit, n_tp - tp_hit]]
        odds, p = fisher_exact(table, alternative="two-sided")
        # Haldane-Anscombe corrected log2 OR for a stable effect size
        a, b, c, d = fn_hit + 0.5, n_fn - fn_hit + 0.5, tp_hit + 0.5, n_tp - tp_hit + 0.5
        log2or = float(np.log2((a / b) / (c / d)))
        rows.append({
            "RBP": rbp,
            "fn_hit": fn_hit, "fn_frac": fn_hit / n_fn,
            "tp_hit": tp_hit, "tp_frac": tp_hit / n_tp,
            "log2_odds_ratio": log2or,
            "p_value": p,
            "direction": "FN-enriched" if log2or > 0 else "TP-enriched",
        })
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["q_value"] = bh_fdr(res["p_value"].to_numpy())
    res = res.sort_values(["q_value", "log2_odds_ratio"],
                          ascending=[True, False]).reset_index(drop=True)
    return res


def analyse_by_family(path: str, encori: pr.PyRanges, flank: int, strand: bool,
                      min_family: int, min_hits: int, region: str | None = None,
                      binding_type: str | None = None) -> pd.DataFrame:
    """Per-(miRNA family, RBP) FN-vs-TP enrichment.

    Tests the cooperative hypothesis directly: an RBP that helps a *specific*
    miRNA bind despite weak complementarity should be enriched in that family's
    FN sites (weak-pairing positives the sequence-only CNN misses) relative to
    its TP sites.  Conditioning on family removes the miRNA-identity / genomic-
    context confound that made the pooled test flip direction between datasets.
    """
    df = _load_positives(path, region, binding_type)
    print(f"\n=== {os.path.basename(path)}  [by miRNA family] ===")

    gr_sites = sites_pyranges(df, flank)
    meta = pd.DataFrame({
        "site_id": np.arange(df.height, dtype=np.int64),
        "et": df["error_type"].to_numpy(),
        "fam": df["noncodingRNA_fam"].to_numpy().astype(str),
    })

    joined = gr_sites.join(encori, strandedness=None, suffix="_rbp")
    jdf = joined.df
    if strand:
        jdf = jdf[jdf["Strand"].astype(str) == jdf["Strand_rbp"].astype(str)]
    pairs = jdf[["site_id", "RBP"]].drop_duplicates().merge(meta, on="site_id")

    # per-family FN/TP totals; keep families with enough of each
    fam_tot = meta.groupby("fam")["et"].value_counts().unstack(fill_value=0)
    fam_tot = fam_tot.reindex(columns=["FN", "TP"], fill_value=0)
    keep_fams = fam_tot[(fam_tot["FN"] >= min_family) &
                        (fam_tot["TP"] >= min_family)].index
    print(f"families tested (>= {min_family} FN and TP each): {len(keep_fams)}")

    hits = (pairs.groupby(["fam", "RBP", "et"]).size().unstack(fill_value=0)
            .reindex(columns=["FN", "TP"], fill_value=0).reset_index())
    rows = []
    for _, r in hits.iterrows():
        fam, rbp = r["fam"], r["RBP"]
        if fam not in keep_fams:
            continue
        fn_hit, tp_hit = int(r["FN"]), int(r["TP"])
        if fn_hit + tp_hit < min_hits:
            continue
        n_fn, n_tp = int(fam_tot.loc[fam, "FN"]), int(fam_tot.loc[fam, "TP"])
        table = [[fn_hit, n_fn - fn_hit], [tp_hit, n_tp - tp_hit]]
        _, p = fisher_exact(table, alternative="two-sided")
        a, b, c, d = fn_hit + .5, n_fn - fn_hit + .5, tp_hit + .5, n_tp - tp_hit + .5
        log2or = float(np.log2((a / b) / (c / d)))
        rows.append({
            "fam": fam, "RBP": rbp, "n_fn": n_fn, "n_tp": n_tp,
            "fn_hit": fn_hit, "fn_frac": fn_hit / n_fn,
            "tp_hit": tp_hit, "tp_frac": tp_hit / n_tp,
            "log2_odds_ratio": log2or, "p_value": p,
            "direction": "FN-enriched" if log2or > 0 else "TP-enriched",
        })
    res = pd.DataFrame(rows)
    if res.empty:
        print("no (family, RBP) cells passed the count filters")
        return res
    res["q_value"] = bh_fdr(res["p_value"].to_numpy())
    res = res.sort_values(["q_value", "log2_odds_ratio"],
                          ascending=[True, False]).reset_index(drop=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True,
                    help="one or more *_errors_v7_restructure.tsv error dumps")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--encori", help="ENCORI RBP targets TSV (1-based inclusive)")
    src.add_argument("--bed-dir", help="directory of iSHAPE <RBP>_<CELLLINE>.bed "
                                       "files (BED9, 0-based half-open)")
    ap.add_argument("--cell-line", default="HEK293T",
                    help="with --bed-dir, which cell line's beds to load (default "
                         "HEK293T)")
    ap.add_argument("--tag", default=None,
                    help="string inserted into the output filename to keep runs "
                         "from different peak sources apart (default: none for "
                         "--encori, 'ishape' for --bed-dir)")
    ap.add_argument("--flank", type=int, default=150,
                    help="nt to extend each side of the 50-nt site (default 150, "
                         "the compute_accessibility.py FLANK)")
    ap.add_argument("--no-strand", action="store_true",
                    help="ignore strand when overlapping (default: same-strand only)")
    ap.add_argument("--min-hits", type=int, default=10,
                    help="skip RBPs with fewer than this many total overlapping "
                         "sites (FN+TP) (default 10)")
    ap.add_argument("--by-family", action="store_true",
                    help="test each (miRNA family, RBP) cell instead of pooling "
                         "all miRNAs (the RBP-helps-specific-miRNA hypothesis)")
    ap.add_argument("--control-density", action="store_true",
                    help="per-(family, RBP) CMH test controlled for each site's "
                         "total RBP occupancy (isolates RBP-specific from bulk "
                         "accessibility signal); implies per-family")
    ap.add_argument("--n-bins", type=int, default=10,
                    help="with --control-density, # of total-RBP-count strata "
                         "per family (default 10)")
    ap.add_argument("--min-family", type=int, default=200,
                    help="with --by-family, require this many FN and TP sites per "
                         "family (default 200)")
    ap.add_argument("--region", default=None,
                    help="restrict to sites whose dominant_region equals this "
                         "(e.g. UTR3, CDS, INTRON, UTR5); default: all sites")
    ap.add_argument("--binding-type", default=None,
                    help="restrict to a canonical binding type from "
                         "binding_types.classify_binding_type (e.g. seedless, "
                         "3prime, 3prime.compensatory, centered); default: all")
    ap.add_argument("--out-dir", default="results",
                    help="directory for per-input <name>.rbp_enrichment.tsv outputs")
    ap.add_argument("--top", type=int, default=20, help="rows to print per input")
    args = ap.parse_args()

    if args.bed_dir:
        encori = load_bed_dir(args.bed_dir, args.cell_line)
        src_name = f"iSHAPE {args.cell_line}"
        tag = args.tag or "ishape"
    else:
        encori = load_encori(args.encori)
        src_name = "ENCORI"
        tag = args.tag
    print(f"{src_name}: {len(encori.df)} peaks, "
          f"{encori.df['RBP'].nunique()} RBPs, flank={args.flank}, "
          f"strand={'same' if not args.no_strand else 'ignored'}")
    os.makedirs(args.out_dir, exist_ok=True)

    for path in args.input:
        if args.control_density:
            res = analyse_density_controlled(
                path, encori, args.flank, not args.no_strand, args.min_family,
                args.min_hits, args.n_bins, args.region, args.binding_type)
            suffix = "rbp_enrichment_density_controlled"
        elif args.by_family:
            res = analyse_by_family(path, encori, args.flank, not args.no_strand,
                                    args.min_family, args.min_hits, args.region,
                                    args.binding_type)
            suffix = "rbp_enrichment_by_family"
        else:
            res = analyse(path, encori, args.flank, not args.no_strand,
                          args.min_hits, args.region, args.binding_type)
            suffix = "rbp_enrichment"
        if args.region:
            suffix += f".{args.region}"
        if args.binding_type:
            suffix += f".{args.binding_type}"
        if tag:
            suffix += f".{tag}"
        if res.empty:
            continue
        base = os.path.basename(path).replace("_errors_v7_restructure.tsv", "")
        out = os.path.join(args.out_dir, f"{base}.{suffix}.tsv")
        res.to_csv(out, sep="\t", index=False)
        sig = res[res["q_value"] < 0.05]
        print(f"significant RBPs (q<0.05): {len(sig)}  ->  {out}")
        with pd.option_context("display.max_rows", None, "display.width", 160,
                               "display.float_format", lambda x: f"{x:.4g}"):
            print(res.head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
