#!/usr/bin/env python
"""Does structural accessibility compensate for weak miRNA-MRE pairing?

This is the *only* mechanism by which an icSHAPE channel could help the
sequence-only CNN, and it is worth testing model-free before paying for the
plumbing.  Accessibility cannot act on this dataset through a main effect:
99.4% of negative coordinates are also positive coordinates (negatives are the
same MRE re-paired with a different miRNA), so any MRE-coordinate-derived
track is *identical* for a positive and its negative twin.  Measured directly,
mean icSHAPE reactivity predicts `label` at AUROC 0.497 -- a coin flip.

So the hypothesis has to be an interaction:

    an open (accessible) site can be bound with weaker complementarity than a
    closed one, because AGO2 pays less of an unfolding cost to load it.

THE CONFOUND.  Sequence composition couples both quantities.  Structured RNA is
GC-rich, so GC anti-correlates with accessibility; and chance complementarity to
an average (~50% GC) miRNA is maximised at *matched* composition, so GC also
anti-correlates with pairing count.  Measured on this data both couplings are
real but modest -- Spearman(GC, acc) ~ -0.01, Spearman(GC, pairing) ~ -0.10 --
and, note, they compose to a *positive* spurious corr(pairing, accessibility),
i.e. the naive confound pushes against the hypothesis rather than toward it.
The sign is not the point: a raw corr(pairing, accessibility) among positives is
uninterpretable either way, so every test here is built to cancel the coupling
by holding the MRE (and hence its composition) fixed across the comparison.

Three tests, weakest to strongest
---------------------------------
A. POOLED DIFFERENCE-IN-DIFFERENCES.  rho_pos = Spearman(pairing, accessibility)
   over positives, rho_neg the same over negatives.  Negatives are random miRNAs
   on the same MRE distribution, so rho_neg *is* the chance-complementarity
   baseline -- the GC confound, measured.  The hypothesis predicts
   `delta_rho = rho_pos - rho_neg < 0`: real binders anti-correlate with
   accessibility more strongly than random miRNAs do.  Cancels the confound only
   to first order (it assumes the confound acts equally on both classes).

B. ACCESSIBILITY-STRATIFIED PAIRING.  Within accessibility quintiles, the median
   pairing of positives and of negatives, their gap, and AUROC(pairing -> label).
   The hypothesis predicts both the gap and the AUROC *shrink* as accessibility
   rises: at an open site, complementarity should matter less.

C. PAIRED WITHIN-COORDINATE TEST  <-- the decisive one.  75.7% of training rows
   sit at coordinates carrying both a positive and a negative row.  Within one
   such coordinate the MRE is literally the same string, so accessibility, GC
   content, region, and expression are held *exactly* fixed; the only thing that
   varies is which miRNA was assigned.  Define

       gap(coord) = mean pairing over its positives - mean pairing over its negatives

   i.e. how much more complementary the real binder is than a random miRNA at
   that same site.  The hypothesis predicts `Spearman(gap, accessibility) < 0`:
   open sites need less excess complementarity.  This is confound-free by
   construction.  Its one assumption is that the negative miRNA assigned to a
   coordinate is random with respect to that coordinate, which is how the
   miRBench negatives are generated.

Pairing strength = pair count on the best antiparallel register (the antidiagonal
`i + j = const` maximising the pair count), matching `binding_types.py`'s notion
of the duplex register.  `--pairing wc` counts Watson-Crick only (default);
`--pairing wcgu` also counts G.U wobbles.

Note on strand: only per-site *scalar* reductions of the BigWig are taken here,
so the fact that `gene` is transcript-oriented while `bw.values()` returns
genomic left-to-right order does not matter.  It would matter for a per-position
track -- reverse minus-strand windows before using one.

    python cnn/icshape_pairing_interaction.py \
        --input data/AGO2_eCLIP_Manakov2022_train_v7.tsv \
        --icshape-dir data/icSHAPE --cell-line HEK293T \
        --min-cov 0.5 --region UTR3
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import polars as pl
import pyBigWig
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

try:
    from rbp_enrichment_fn_vs_tp import tsv_chrom_to_fa
except ModuleNotFoundError:  # invoked as cnn.icshape_pairing_interaction
    from cnn.rbp_enrichment_fn_vs_tp import tsv_chrom_to_fa

MAX_MIRNA = 30
MRE_LEN = 50

# A=0 C=1 G=2 U=3, pad/unknown=4 -- same convention as binding_types.py
_LUT = np.full(256, 4, dtype=np.int8)
for _c, _i in zip(b"ACGU", range(4)):
    _LUT[_c] = _i
_LUT[ord("T")] = 3
for _c, _i in zip(b"acgu", range(4)):
    _LUT[_c] = _i
_LUT[ord("t")] = 3

_WC = np.zeros((5, 5), dtype=np.float32)
for _a, _b in ((0, 3), (3, 0), (2, 1), (1, 2)):  # A-U U-A G-C C-G
    _WC[_a, _b] = 1.0
_WCGU = _WC.copy()
for _a, _b in ((2, 3), (3, 2)):                  # G.U wobbles
    _WCGU[_a, _b] = 1.0


def tokenize(seqs: list[str], length: int) -> np.ndarray:
    """(N, length) int8 nucleotide indices, right-padded with 4."""
    out = np.full((len(seqs), length), 4, dtype=np.int8)
    for i, s in enumerate(seqs):
        b = np.frombuffer(str(s).encode(), dtype=np.uint8)[:length]
        out[i, : len(b)] = _LUT[b]
    return out


def best_register_pairs(mi: np.ndarray, ti: np.ndarray, table: np.ndarray,
                        chunk: int = 20000) -> np.ndarray:
    """Pair count on the best antiparallel register, per row.

    Both sequences run 5'->3', so a helical register is an antidiagonal
    `i + j = d`.  Sum the pair table over each of the 79 antidiagonals and take
    the max.  Vectorised as 30 strided adds rather than a scatter.
    """
    n = len(mi)
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        pm = table[mi[s:e, :, None].astype(np.intp),
                   ti[s:e, None, :].astype(np.intp)]     # (b, 30, 50)
        diag = np.zeros((e - s, MAX_MIRNA + MRE_LEN - 1), dtype=np.float32)
        for i in range(MAX_MIRNA):
            diag[:, i:i + MRE_LEN] += pm[:, i, :]
        out[s:e] = diag.max(axis=1)
    return out


def gc_fraction(ti: np.ndarray) -> np.ndarray:
    """GC fraction over the non-pad bases of each MRE -- the confound, quantified."""
    valid = ti != 4
    gc = ((ti == 1) | (ti == 2)) & valid
    return gc.sum(1) / np.maximum(valid.sum(1), 1)


def score_coords(coords: list[tuple[str, int, int, str]], icshape_dir: str,
                 cell_line: str, flank: int, stat: str, ss_thresh: float):
    """Accessibility per *unique coordinate* (they repeat ~1.5x across rows).

    v7 `start` is 1-based, BigWig 0-based half-open -> site spans [start-1, end).
    Reactivity is strand-specific, so the strand picks the BigWig.
    """
    bws = {}
    for sym, name in (("+", "plus"), ("-", "minus")):
        p = os.path.join(icshape_dir, f"{cell_line}-{name}.bw")
        if not os.path.exists(p):
            raise SystemExit(f"missing icSHAPE BigWig: {p}")
        bws[sym] = pyBigWig.open(p)
    chroms = {s: bw.chroms() for s, bw in bws.items()}

    n = len(coords)
    vals = np.full(n, np.nan, dtype=np.float64)
    cov = np.zeros(n, dtype=np.float64)
    for i, (c, s, e, st) in enumerate(coords):
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
        elif stat == "ssfrac":
            vals[i] = (good > ss_thresh).mean()
        else:
            raise SystemExit(f"unknown --stat {stat}")
    for bw in bws.values():
        bw.close()
    return vals, cov


def _boot_spearman_diff(x_a, y_a, x_b, y_b, n_boot: int, rng) -> tuple:
    """Bootstrap CI for Spearman(a) - Spearman(b), resampling within each group."""
    na, nb = len(x_a), len(x_b)
    d = np.empty(n_boot)
    for k in range(n_boot):
        ia = rng.integers(0, na, na)
        ib = rng.integers(0, nb, nb)
        d[k] = (spearmanr(x_a[ia], y_a[ia]).statistic
                - spearmanr(x_b[ib], y_b[ib]).statistic)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _boot_spearman(x, y, n_boot: int, rng) -> tuple:
    n = len(x)
    r = np.empty(n_boot)
    for k in range(n_boot):
        i = rng.integers(0, n, n)
        r[k] = spearmanr(x[i], y[i]).statistic
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def test_a_pooled_did(pair, acc, y, n_boot, rng) -> None:
    print("\n=== A. pooled difference-in-differences ===")
    print("    hypothesis predicts delta_rho < 0 "
          "(real binders anti-correlate more than random miRNAs)")
    p_pos, a_pos = pair[y == 1], acc[y == 1]
    p_neg, a_neg = pair[y == 0], acc[y == 0]
    r_pos = spearmanr(p_pos, a_pos)
    r_neg = spearmanr(p_neg, a_neg)
    print(f"  rho_pos = {r_pos.statistic:+.4f}  (p={r_pos.pvalue:.2e}, n={len(p_pos)})")
    print(f"  rho_neg = {r_neg.statistic:+.4f}  (p={r_neg.pvalue:.2e}, n={len(p_neg)})"
          "   <- the GC / chance-complementarity baseline")
    d = r_pos.statistic - r_neg.statistic
    lo, hi = _boot_spearman_diff(p_pos, a_pos, p_neg, a_neg, n_boot, rng)
    verdict = "SUPPORTS" if hi < 0 else ("CONTRADICTS" if lo > 0 else "NULL")
    print(f"  delta_rho = {d:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   -> {verdict}")


def test_b_strata(pair, acc, y, n_bins: int) -> None:
    print(f"\n=== B. pairing vs label, within accessibility {n_bins}-tiles ===")
    print("    hypothesis predicts the gap and the AUROC SHRINK as accessibility rises")
    edges = np.quantile(acc, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    print(f"{'tile':>4} {'acc range':>16} {'n_pos':>7} {'n_neg':>7} "
          f"{'med_pair_pos':>13} {'med_pair_neg':>13} {'gap':>7} {'AUROC':>7}")
    for b in range(n_bins):
        m = (acc >= edges[b]) & (acc < edges[b + 1])
        yp, pp = y[m], pair[m]
        if (yp == 1).sum() < 20 or (yp == 0).sum() < 20:
            continue
        mp = float(np.median(pp[yp == 1]))
        mn = float(np.median(pp[yp == 0]))
        auc = roc_auc_score(yp, pp)
        print(f"{b + 1:>4} {edges[b]:7.3f}-{edges[b + 1]:<8.3f} "
              f"{int((yp == 1).sum()):>7} {int((yp == 0).sum()):>7} "
              f"{mp:>13.3f} {mn:>13.3f} {mp - mn:>7.3f} {auc:>7.4f}")


def test_c_paired(df: pl.DataFrame, pair: np.ndarray, acc_by_row: np.ndarray,
                  n_boot: int, rng, n_bins: int) -> None:
    """The decisive test: within a coordinate the MRE is identical, so GC,
    region, expression and accessibility are all held exactly fixed."""
    print("\n=== C. paired within-coordinate test (decisive) ===")
    print("    hypothesis predicts Spearman(gap, accessibility) < 0")
    t = df.with_columns(
        pl.Series("pair", pair),
        pl.Series("acc", acc_by_row),
    )
    g = (t.group_by("k")
          .agg(pl.col("pair").filter(pl.col("label") == 1).mean().alias("pp"),
               pl.col("pair").filter(pl.col("label") == 0).mean().alias("pn"),
               pl.col("acc").first().alias("acc"),
               pl.col("label").n_unique().alias("nl"))
          .filter((pl.col("nl") == 2) & pl.col("pp").is_not_null()
                  & pl.col("pn").is_not_null()))
    if g.height < 50:
        print(f"  only {g.height} usable twin coordinates -- underpowered, skipping")
        return
    gap = (g["pp"] - g["pn"]).to_numpy()
    a = g["acc"].to_numpy()
    r = spearmanr(gap, a)
    lo, hi = _boot_spearman(gap, a, n_boot, rng)
    print(f"  twin coordinates: {g.height}")
    print(f"  mean gap (real binder - random miRNA): {gap.mean():+.3f} pairs")
    verdict = "SUPPORTS" if hi < 0 else ("CONTRADICTS" if lo > 0 else "NULL")
    print(f"  Spearman(gap, acc) = {r.statistic:+.4f}  (p={r.pvalue:.2e})"
          f"   95% CI [{lo:+.4f}, {hi:+.4f}]   -> {verdict}")

    edges = np.quantile(a, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    print(f"\n{'tile':>4} {'acc range':>16} {'n_coord':>8} {'mean gap':>9} {'median gap':>11}")
    for b in range(n_bins):
        m = (a >= edges[b]) & (a < edges[b + 1])
        if m.sum() < 20:
            continue
        print(f"{b + 1:>4} {edges[b]:7.3f}-{edges[b + 1]:<8.3f} {int(m.sum()):>8} "
              f"{gap[m].mean():>9.3f} {np.median(gap[m]):>11.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="a v7 TSV with both labels")
    ap.add_argument("--icshape-dir", default="data/icSHAPE")
    ap.add_argument("--cell-line", default="HEK293T")
    ap.add_argument("--flank", type=int, default=0,
                    help="nt to extend each side of the site; 0 = the MRE itself")
    ap.add_argument("--stat", default="mean", choices=["mean", "median", "ssfrac"])
    ap.add_argument("--ss-thresh", type=float, default=0.5)
    ap.add_argument("--min-cov", type=float, default=0.5,
                    help="require this fraction of the window to carry icSHAPE data")
    ap.add_argument("--pairing", default="wc", choices=["wc", "wcgu"],
                    help="count Watson-Crick only (default) or WC + G.U wobbles")
    ap.add_argument("--region", default=None,
                    help="restrict to this dominant_region (e.g. UTR3)")
    ap.add_argument("--max-coords", type=int, default=400000,
                    help="subsample this many unique coordinates (keeps all rows at "
                         "each, so the paired test stays intact); 0 = no cap")
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pl.read_csv(args.input, separator="\t", infer_schema_length=5000)
    if args.region:
        df = df.filter(pl.col("dominant_region") == args.region)
    df = df.with_columns(
        pl.concat_str(["chr", "strand", "start", "end"], separator=":").alias("k"))
    print(f"=== {os.path.basename(args.input)}"
          f"{' [' + args.region + ']' if args.region else ''} ===")
    print(f"rows={df.height}  unique coords={df['k'].n_unique()}")

    if args.max_coords and df["k"].n_unique() > args.max_coords:
        keep = (df.select("k").unique()
                  .sample(args.max_coords, seed=args.seed))
        df = df.join(keep, on="k", how="inner")
        print(f"subsampled to {args.max_coords} coords -> {df.height} rows")

    # accessibility once per unique coordinate, then broadcast back to rows
    uniq = df.select("k", "chr", "start", "end", "strand").unique(subset="k")
    coords = list(zip(uniq["chr"].cast(pl.Utf8), uniq["start"], uniq["end"],
                      uniq["strand"].cast(pl.Utf8)))
    print(f"querying icSHAPE over {len(coords)} unique coordinates ...")
    vals, cov = score_coords(coords, args.icshape_dir, args.cell_line,
                             args.flank, args.stat, args.ss_thresh)
    uniq = uniq.with_columns(pl.Series("acc", vals), pl.Series("cov", cov))
    df = df.join(uniq.select("k", "acc", "cov"), on="k", how="left")

    n_all = df.height
    df = df.filter((pl.col("cov") >= args.min_cov) & pl.col("acc").is_not_null()
                   & pl.col("acc").is_finite())
    print(f"covered rows (cov >= {args.min_cov}): {df.height} / {n_all} "
          f"({df.height / max(n_all, 1):.1%});  stat={args.stat}, flank={args.flank}")
    if df.height < 200:
        raise SystemExit("too few covered rows")

    mi = tokenize(df["noncodingRNA"].to_list(), MAX_MIRNA)
    ti = tokenize(df["gene"].to_list(), MRE_LEN)
    table = _WC if args.pairing == "wc" else _WCGU
    pair = best_register_pairs(mi, ti, table)
    acc = df["acc"].to_numpy()
    y = df["label"].to_numpy()
    gc = gc_fraction(ti)

    print(f"\npairing = best-register {args.pairing.upper()} count;  "
          f"pos median {np.median(pair[y == 1]):.1f}, neg median {np.median(pair[y == 0]):.1f}")

    print("\n--- the confound, measured ---")
    r_gc = spearmanr(gc, acc)
    r_gcp = spearmanr(gc, pair)
    print(f"  Spearman(GC, accessibility) = {r_gc.statistic:+.4f} (p={r_gc.pvalue:.2e})"
          "   structured RNA is GC-rich")
    print(f"  Spearman(GC, pairing)       = {r_gcp.statistic:+.4f} (p={r_gcp.pvalue:.2e})"
          "   chance pairing peaks at composition matched to a ~50% GC miRNA")
    print("  => composition couples both, so a raw corr(pairing, accessibility) is\n"
          "     uninterpretable; tests A/C hold the MRE fixed to cancel it.")

    test_a_pooled_did(pair, acc, y, args.bootstrap, rng)
    test_b_strata(pair, acc, y, args.bins)
    test_c_paired(df, pair, acc, args.bootstrap, rng, args.bins)


if __name__ == "__main__":
    main()
