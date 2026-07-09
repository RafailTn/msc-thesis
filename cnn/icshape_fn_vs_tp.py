#!/usr/bin/env python
"""Test whether false-negative miRNA target sites sit in more structurally
accessible (single-stranded) RNA than true-positive ones, using experimental
icSHAPE reactivity.

Rationale: the CNN is sequence-only -- it sees Watson-Crick complementarity and
nothing about whether the target is actually open in vivo.  FN sites (true
positives the model misses) carry weaker direct pairing than TP sites, so
something must compensate.  Local structural accessibility is the leading
candidate: an open MRE lowers the energetic cost of AGO2 loading, letting a
weakly-paired site bind anyway.  icSHAPE reactivity in [0,1] measures exactly
that -- high = unpaired/accessible.  Prediction: FN > TP.

Pipeline
--------
1.  Take FN and TP rows (label==1 positives the model got wrong / right) from an
    ``*_errors_v7_restructure.tsv`` error dump.
2.  Query the strand-matched icSHAPE BigWig (``<CELLLINE>-plus.bw`` /
    ``-minus.bw``) over each site, optionally extended by ``--flank`` nt, and
    reduce to one scalar per site (``--stat``).
3.  Mann-Whitney U of FN vs TP with Cliff's delta as the effect size, pooled and
    per miRNA family (Benjamini-Hochberg across families).

    THE SELECTION BIAS.  icSHAPE only covers well-expressed transcripts (~0.2% of
    the genome), and FN sites are *more likely to be covered at all* than TP sites
    (OR ~1.1, p<1e-9 in manakov_test UTR3).  Restricting to covered sites is
    therefore not a neutral filter -- it enriches for FN.  This script always
    prints that coverage-rate contrast first (``--min-cov`` sweep) so the reader
    can judge how much of any reactivity difference is expression confounding.
    Treat a significant FN>TP as "consistent with, not evidence for" the
    accessibility hypothesis unless it survives within coverage strata.

    python cnn/icshape_fn_vs_tp.py \
        --input data/manakov_test_errors_v7_restructure.tsv \
        --icshape-dir data/icSHAPE --cell-line HEK293T \
        --region UTR3 --min-cov 0.5 --by-family
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import polars as pl
import pyBigWig
from scipy.stats import mannwhitneyu

try:
    from rbp_enrichment_fn_vs_tp import bh_fdr, tsv_chrom_to_fa
except ModuleNotFoundError:  # invoked as cnn.icshape_fn_vs_tp
    from cnn.rbp_enrichment_fn_vs_tp import bh_fdr, tsv_chrom_to_fa


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(delta, p) for group `a` vs `b`.  delta = 2U/(n_a*n_b) - 1, so it is
    +1 when every a > every b, -1 when every a < every b, 0 at no separation.
    Here a=FN, b=TP, so delta>0 means FN sites are the more reactive/accessible.
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return np.nan, 1.0
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return float(2.0 * u / (na * nb) - 1.0), float(p)


def _load_positives(path: str, region: str | None) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", infer_schema_length=5000)
    df = df.filter(pl.col("error_type").is_in(["FN", "TP"]))
    if region:
        df = df.filter(pl.col("dominant_region") == region)
    return df


def score_sites(df: pl.DataFrame, icshape_dir: str, cell_line: str, flank: int,
                stat: str, ss_thresh: float):
    """Per site: (value, covered_frac).  NaN value where nothing is covered.

    Coordinates: the v7 `start` is 1-based, BigWig is 0-based half-open, so the
    site spans [start-1, end) and the flanked window [start-1-flank, end+flank).
    Reactivity is strand-specific, so the site's strand picks the BigWig.
    """
    bws = {}
    for sym, name in (("+", "plus"), ("-", "minus")):
        p = os.path.join(icshape_dir, f"{cell_line}-{name}.bw")
        if not os.path.exists(p):
            raise SystemExit(f"missing icSHAPE BigWig: {p}")
        bws[sym] = pyBigWig.open(p)
    chroms = {s: bw.chroms() for s, bw in bws.items()}

    n = df.height
    vals = np.full(n, np.nan)
    cov = np.zeros(n)
    for i, (c, s, e, st) in enumerate(zip(df["chr"], df["start"], df["end"],
                                          df["strand"])):
        st = str(st)
        if st not in bws:
            continue
        c = tsv_chrom_to_fa(c)
        if c not in chroms[st]:
            continue
        s0 = max(int(s) - 1 - flank, 0)
        e0 = min(int(e) + flank, chroms[st][c])
        if e0 <= s0:
            continue
        try:
            v = bws[st].values(c, s0, e0, numpy=True)
        except (RuntimeError, OverflowError):
            continue
        m = np.isfinite(v)
        cov[i] = m.mean()
        if not m.any():
            continue
        good = v[m]
        if stat == "mean":
            vals[i] = good.mean()
        elif stat == "median":
            vals[i] = np.median(good)
        elif stat == "ssfrac":  # fraction of measured bases that read as open
            vals[i] = (good > ss_thresh).mean()
        else:
            raise SystemExit(f"unknown --stat {stat}")
    for bw in bws.values():
        bw.close()
    return vals, cov


def coverage_bias(cov: np.ndarray, et: np.ndarray) -> None:
    """The load-bearing diagnostic: is *having* icSHAPE data itself FN-biased?"""
    from scipy.stats import fisher_exact
    print("\n--- coverage-rate bias (is the covered subset FN-enriched?) ---")
    print(f"{'min_cov':>8}  {'FN rate':>16}  {'TP rate':>16}  {'OR':>6}  {'p':>10}")
    for thr in (0.0, 0.25, 0.5, 0.75, 0.9):
        keep = cov > thr
        a = int((keep & (et == "FN")).sum()); b = int((~keep & (et == "FN")).sum())
        c = int((keep & (et == "TP")).sum()); d = int((~keep & (et == "TP")).sum())
        if min(a + b, c + d) == 0:
            continue
        orr, p = fisher_exact([[a, b], [c, d]])
        print(f"{thr:8.2f}  {a/(a+b):7.2%} (n={a:6d})  {c/(c+d):7.2%} (n={c:6d})  "
              f"{orr:6.3f}  {p:10.2e}")
    print("  OR>1 => FN sites are preferentially retained by the coverage filter.")


def _row(fam: str, v: np.ndarray, isfn: np.ndarray) -> dict | None:
    fn, tp = v[isfn], v[~isfn]
    if len(fn) < 2 or len(tp) < 2:
        return None
    d, p = cliffs_delta(fn, tp)
    if not np.isfinite(d):
        return None
    return {
        "fam": fam, "n_fn": len(fn), "n_tp": len(tp),
        "mean_fn": float(fn.mean()), "mean_tp": float(tp.mean()),
        "median_fn": float(np.median(fn)), "median_tp": float(np.median(tp)),
        "delta_mean": float(fn.mean() - tp.mean()),
        "cliffs_delta": d, "p_value": p,
        "direction": "FN-accessible" if d > 0 else "TP-accessible",
    }


def analyse(path: str, args) -> pd.DataFrame:
    df = _load_positives(path, args.region)
    print(f"\n=== {os.path.basename(path)} ===")
    vals, cov = score_sites(df, args.icshape_dir, args.cell_line, args.flank,
                            args.stat, args.ss_thresh)
    et = df["error_type"].to_numpy()
    coverage_bias(cov, et)

    keep = (cov >= args.min_cov) & np.isfinite(vals)
    print(f"\nsites with cov >= {args.min_cov}: {keep.sum()} / {len(keep)} "
          f"({keep.mean():.1%});  stat={args.stat}, flank={args.flank}")
    if keep.sum() < 10:
        print("too few covered sites")
        return pd.DataFrame()

    v, e = vals[keep], et[keep]
    isfn = e == "FN"
    pooled = _row("__pooled__", v, isfn)
    print(f"\npooled: FN n={pooled['n_fn']} mean={pooled['mean_fn']:.4f}  |  "
          f"TP n={pooled['n_tp']} mean={pooled['mean_tp']:.4f}")
    print(f"  delta_mean={pooled['delta_mean']:+.4f}  "
          f"Cliff's delta={pooled['cliffs_delta']:+.4f}  p={pooled['p_value']:.3e}")
    print("  |Cliff's delta| < 0.147 is conventionally 'negligible'.")

    rows = [pooled]
    if args.by_family:
        fam = df["noncodingRNA_fam"].to_numpy().astype(str)[keep]
        for f in np.unique(fam):
            m = fam == f
            sub_fn = int((isfn & m).sum()); sub_tp = int((~isfn & m).sum())
            if sub_fn < args.min_family or sub_tp < args.min_family:
                continue
            r = _row(f, v[m], isfn[m])
            if r:
                rows.append(r)
    res = pd.DataFrame(rows)
    fam_mask = res["fam"] != "__pooled__"
    res["q_value"] = np.nan
    if fam_mask.any():
        res.loc[fam_mask, "q_value"] = bh_fdr(res.loc[fam_mask, "p_value"].to_numpy())
    return res.sort_values(["q_value", "p_value"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True,
                    help="one or more *_errors_v7_restructure.tsv error dumps")
    ap.add_argument("--icshape-dir", default="data/icSHAPE",
                    help="directory of <CELLLINE>-plus.bw / -minus.bw")
    ap.add_argument("--cell-line", default="HEK293T")
    ap.add_argument("--flank", type=int, default=0,
                    help="nt to extend each side of the 50-nt site; 0 = the site "
                         "itself (default). Use e.g. 150 to ask about the "
                         "*neighbourhood* opening rather than the MRE itself")
    ap.add_argument("--stat", default="mean", choices=["mean", "median", "ssfrac"],
                    help="per-site reduction of the covered bases (default mean); "
                         "ssfrac = fraction of measured bases above --ss-thresh")
    ap.add_argument("--ss-thresh", type=float, default=0.5,
                    help="with --stat ssfrac, reactivity above which a base counts "
                         "as single-stranded (default 0.5)")
    ap.add_argument("--min-cov", type=float, default=0.5,
                    help="require this fraction of the window to have icSHAPE data "
                         "(default 0.5); see the coverage-bias table")
    ap.add_argument("--region", default=None,
                    help="restrict to sites whose dominant_region equals this "
                         "(e.g. UTR3, CDS, INTRON, UTR5); default: all sites")
    ap.add_argument("--by-family", action="store_true",
                    help="also test each miRNA family separately (BH across families)")
    ap.add_argument("--min-family", type=int, default=50,
                    help="with --by-family, require this many covered FN and TP "
                         "sites per family (default 50)")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for path in args.input:
        res = analyse(path, args)
        if res.empty:
            continue
        suffix = f"icshape_fn_vs_tp.{args.cell_line}.{args.stat}"
        if args.flank:
            suffix += f".flank{args.flank}"
        if args.region:
            suffix += f".{args.region}"
        base = os.path.basename(path).replace("_errors_v7_restructure.tsv", "")
        out = os.path.join(args.out_dir, f"{base}.{suffix}.tsv")
        res.to_csv(out, sep="\t", index=False)
        print(f"\n-> {out}")
        with pd.option_context("display.max_rows", None, "display.width", 200,
                               "display.float_format", lambda x: f"{x:.4g}"):
            print(res.head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
