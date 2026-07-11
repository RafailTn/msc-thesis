#!/usr/bin/env python
"""Does transcript expression separate the CNN's false negatives from its true positives?

Uses RNA-seq coverage from the *same study* as the eCLIP (Manakov 2022,
GSM5918356, HEK293T untransfected, total RNA, stranded) — a direct expression
measurement rather than the eCLIP-derived `cov` proxy.

Why expect anything?  The CNN is **sequence-only**: it never sees expression, so
expression cannot causally drive a prediction.  Any FN/TP separation must come from
how the *labels* were made.  AGO2 eCLIP calls a site when it accumulates enough
chimeric reads, and read count scales with transcript abundance.  On a highly
expressed transcript even a weak, decoy-like site clears the calling threshold; on a
lowly expressed one only a strong site does.  So the positive set on abundant
transcripts is enriched for weakly-paired sites — exactly the sites a complementarity
model misses.  Prediction: **FN sites sit on more highly expressed transcripts than TP
sites.**  If so, expression is indexing label noise, not biology.

THE MEDIATION QUESTION (the point of this script).  FN sites have weaker pairing than
TP sites by construction (that is what makes the model miss them).  And weakly-paired
positives are enriched on abundant transcripts.  So a raw FN>TP expression gap could be
*entirely* explained by pairing strength, telling us nothing new.  The script therefore
always reports the gap **stratified by `n_wc`** (Watson-Crick pair count in the best
register, already in the error dump).  Read the stratified deltas, not the pooled one:

  - gap survives within n_wc strata  -> expression carries FN-risk beyond pairing
  - gap collapses within n_wc strata -> it was pairing all along

Strandedness: `.pos.bw` holds plus-strand genes, `.neg.bw` minus-strand with values
stored **negative** (take abs).  Verified empirically on v7 UTR3 positives of known
strand: 240x signal ratio in the matching file, so the convention is not swapped.

Missingness: unlike icSHAPE, an absent position here means *zero reads*, not "not
measured".  So every site gets a value and there is no coverage filter to bias.

    python cnn/rnaseq_fn_vs_tp.py \
        --input data/manakov_test_errors_v7_restructure.tsv \
        --region UTR3
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
from bisect import bisect_right
from pathlib import Path

import numpy as np
import polars as pl
import pyBigWig
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

try:
    from genomic_utils import bh_fdr, tsv_chrom_to_fa
    from binding_types import _MAX_MI, _MAX_TI, _tok, classify_index_arrays
except ModuleNotFoundError:  # invoked as cnn.rnaseq_fn_vs_tp
    from cnn.genomic_utils import bh_fdr, tsv_chrom_to_fa
    from cnn.binding_types import _MAX_MI, _MAX_TI, _tok, classify_index_arrays

# canonical seed classes: a contiguous seed match anchored near miRNA pos 2-8.
# The rest are the "decoy-like" categories the model is expected to reject.
CANONICAL = ("9mer", "8mer1A", "8mer", "7mer", "7mer1A", "6mer", "offset6mer", "5mer")


def binding_class(df: pl.DataFrame) -> np.ndarray:
    """Per-row canonical binding category, collapsing the .GU / .3prime suffixes."""
    mi = np.stack([_tok(s, _MAX_MI) for s in df["noncodingRNA"].to_list()])
    ti = np.stack([_tok(s, _MAX_TI) for s in df["gene"].to_list()])
    bt = classify_index_arrays(mi, ti)
    return np.array([b if b in ("seedless", "centered", "3prime",
                                "3prime.compensatory") else b.split(".")[0]
                     for b in bt])

PREFIX = ("data/GSM5918356_Expt8_293Tuntfx_RNAseq_total_rep1"
          ".CombinedID.merged.r2.norm")
DEFAULT_GTF = "~/Downloads/hg38/gencode.v47.primary_assembly.annotation.gtf.gz"


# --------------------------------------------------------------------------- #
# MANE Select exon model.  Kept local rather than imported from
# compute_accessibility, which pulls in ViennaRNA at module scope (not in the
# pixi env).  Same precedent as cooperativity_analysis.py owning its copies.
# --------------------------------------------------------------------------- #
def _parse_mane(gtf_path: str):
    """(tx, index, max_exon_len) over MANE_Select exons.  Cached beside the GTF."""
    p = Path(gtf_path).expanduser()
    cache = p.with_name(p.name + ".mane_expr.pkl")
    if cache.exists() and cache.stat().st_mtime >= p.stat().st_mtime:
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    opener = gzip.open if str(p).endswith(".gz") else open
    tx: dict[str, dict] = {}
    with opener(p, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t")
            if f[2] != "exon" or 'tag "MANE_Select"' not in f[8]:
                continue
            tid = f[8].split('transcript_id "', 1)[1].split('"', 1)[0]
            es, ee = int(f[3]), int(f[4])
            d = tx.get(tid)
            if d is None:
                tx[tid] = {"chrom": f[0], "strand": f[6], "ex": [(es, ee)]}
            else:
                d["ex"].append((es, ee))

    buckets: dict[tuple, list] = {}
    max_exon_len = 0
    for tid, d in tx.items():
        d["ex"].sort()
        d["Lt"] = sum(ee - es + 1 for es, ee in d["ex"])
        for es, ee in d["ex"]:
            max_exon_len = max(max_exon_len, ee - es + 1)
            buckets.setdefault((d["chrom"], d["strand"]), []).append((es, ee, tid))

    index = {}
    for key, b in buckets.items():
        b.sort()
        index[key] = (np.array([x[0] for x in b], dtype=np.int64),
                      np.array([x[1] for x in b], dtype=np.int64),
                      [x[2] for x in b])
    out = (tx, index, max_exon_len)
    with open(cache, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    return out


def _find_host_tid(index, max_exon_len, chrom, strand, s, e):
    """transcript_id of the MANE exon fully containing 1-based [s, e], else None."""
    rec = index.get((chrom, strand))
    if rec is None:
        return None
    es_arr, ee_arr, tids = rec
    k = bisect_right(es_arr, s) - 1
    while k >= 0 and (s - es_arr[k]) <= max_exon_len:
        if ee_arr[k] >= e:
            return tids[k]
        k -= 1
    return None


def transcript_expression(df: pl.DataFrame, prefix: str, gtf: str):
    """Mean per-base RNA-seq coverage over the whole MANE mature transcript.

    This is the gene-level abundance estimate that the 50-nt local window cannot
    give: summing across every exon averages out 3'UTR isoform usage (APA), local
    3' bias and single-window mappability dropouts.  Sites not fully inside a MANE
    Select exon (introns, non-MANE genes) return NaN and are dropped downstream.
    """
    tx, index, mx = _parse_mane(gtf)
    bws = {"+": pyBigWig.open(f"{prefix}.pos.bw"),
           "-": pyBigWig.open(f"{prefix}.neg.bw")}
    chroms = {s: b.chroms() for s, b in bws.items()}

    tids = []
    for c, s, e, st in zip(df["chr"], df["start"], df["end"], df["strand"]):
        st = str(st)
        tids.append(None if st not in bws else
                    _find_host_tid(index, mx, tsv_chrom_to_fa(c), st,
                                   int(s), int(e)))

    cache: dict[str, float] = {}
    for tid in {t for t in tids if t is not None}:
        d = tx[tid]
        bw, ch = bws[d["strand"]], chroms[d["strand"]]
        if d["chrom"] not in ch:
            continue
        tot, L = 0.0, 0
        for es, ee in d["ex"]:
            s0, e0 = max(es - 1, 0), min(ee, ch[d["chrom"]])
            if e0 <= s0:
                continue
            try:
                v = bw.values(d["chrom"], s0, e0, numpy=True)
            except (RuntimeError, OverflowError):
                continue
            tot += float(np.abs(np.nan_to_num(v, nan=0.0)).sum())
            L += e0 - s0
        if L:
            cache[tid] = tot / L
    for b in bws.values():
        b.close()
    return np.array([cache.get(t, np.nan) if t else np.nan for t in tids])


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(delta, p) for a vs b.  a=FN, b=TP, so delta>0 means FN more expressed."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan, 1.0
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return float(2.0 * u / (na * nb) - 1.0), float(p)


def score_expression(df: pl.DataFrame, prefix: str, flank: int) -> np.ndarray:
    """Mean RNA-seq coverage over each site (absent positions count as 0 reads).

    v7 `start` is 1-based, BigWig 0-based half-open -> site = [start-1, end).
    """
    bws = {}
    for sym, name in (("+", "pos"), ("-", "neg")):
        p = f"{prefix}.{name}.bw"
        if not os.path.exists(p):
            raise SystemExit(f"missing RNA-seq BigWig: {p}")
        bws[sym] = pyBigWig.open(p)
    chroms = {s: bw.chroms() for s, bw in bws.items()}

    out = np.zeros(df.height)
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
        # absent == zero coverage; minus strand is stored as negative values
        out[i] = np.abs(np.nan_to_num(v, nan=0.0)).mean()
    for bw in bws.values():
        bw.close()
    return out


def mediation_table(v: np.ndarray, isfn: np.ndarray, strata: np.ndarray,
                    title: str, order: list | None = None, min_n: int = 20) -> None:
    """Cliff's delta of `v` (FN vs TP) within each level of `strata`.

    The pooled-vs-within comparison is the mediation read-out: if the confounder
    `strata` explains the pooled gap, the within-stratum deltas collapse to ~0.
    """
    print(f"\n{title}")
    levels = order or sorted(set(strata.tolist()))
    print(f"{'stratum':<22}{'n_FN':>8}{'n_TP':>8}{'delta':>10}{'p':>12}")
    ds, ws, ps, keep = [], [], [], []
    for lv in levels:
        m = strata == lv
        nf, nt = int((m & isfn).sum()), int((m & ~isfn).sum())
        if nf < min_n or nt < min_n:
            continue
        d, p = cliffs_delta(v[m & isfn], v[m & ~isfn])
        ds.append(d); ws.append(nf + nt); ps.append(p); keep.append(lv)
        print(f"{str(lv):<22}{nf:>8}{nt:>8}{d:>+10.4f}{p:>12.2e}")
    if not ds:
        return
    dp, _ = cliffs_delta(v[isfn], v[~isfn])
    within = float(np.average(ds, weights=np.array(ws, dtype=float)))
    qs = bh_fdr(np.array(ps))
    print(f"\n  pooled delta = {dp:+.4f}   weight-averaged within-stratum = {within:+.4f}"
          f"   ({(1 - within / dp) * 100:.0f}% attenuation)" if abs(dp) > 1e-9 else "")
    print(f"  strata significant after BH: {(qs < 0.05).sum()} / {len(qs)}")


def _report(e: np.ndarray, isfn: np.ndarray, label: str) -> dict | None:
    fn, tp = e[isfn], e[~isfn]
    if len(fn) < 20 or len(tp) < 20:
        return None
    d, p = cliffs_delta(fn, tp)
    return {"stratum": label, "n_fn": len(fn), "n_tp": len(tp),
            "med_fn": float(np.median(fn)), "med_tp": float(np.median(tp)),
            "cliffs_delta": d, "p_value": p}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="*_errors_v7_restructure.tsv")
    ap.add_argument("--rnaseq-prefix", default=PREFIX)
    ap.add_argument("--flank", type=int, default=0,
                    help="nt to extend each side of the 50-nt site (default 0)")
    ap.add_argument("--region", default=None, help="restrict to a dominant_region")
    ap.add_argument("--max-rows", type=int, default=250000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gtf", default=DEFAULT_GTF,
                    help="GENCODE GTF with MANE_Select tags, for transcript-level "
                         f"abundance (default {DEFAULT_GTF})")
    ap.add_argument("--no-gtf", action="store_true",
                    help="skip the transcript-level panel")
    args = ap.parse_args()

    df = pl.read_csv(args.input, separator="\t", infer_schema_length=5000)
    df = df.filter(pl.col("error_type").is_in(["FN", "TP"]))
    if args.region:
        df = df.filter(pl.col("dominant_region") == args.region)
    if df.height > args.max_rows:
        df = df.sample(args.max_rows, seed=args.seed)
    print(f"=== {os.path.basename(args.input)}"
          f"{' [' + args.region + ']' if args.region else ''} ===")
    print(f"FN/TP rows: {df.height}")

    e = score_expression(df, args.rnaseq_prefix, args.flank)
    isfn = (df["error_type"] == "FN").to_numpy()
    nwc = df["n_wc"].to_numpy()
    le = np.log1p(e)

    print(f"\nzero-expression sites: {(e == 0).mean():.1%}   "
          f"median coverage: FN {np.median(e[isfn]):.3f}  TP {np.median(e[~isfn]):.3f}")

    print("\n=== A. pooled (confounded by pairing strength -- do not stop here) ===")
    r = _report(le, isfn, "__pooled__")
    print(f"  FN n={r['n_fn']} median log1p={r['med_fn']:.4f}  |  "
          f"TP n={r['n_tp']} median log1p={r['med_tp']:.4f}")
    print(f"  Cliff's delta = {r['cliffs_delta']:+.4f}  p={r['p_value']:.3e}   "
          f"(>0 => FN more expressed)")
    print(f"  AUROC(expression -> is_FN) = {roc_auc_score(isfn, le):.4f}")
    print("  |Cliff's delta| < 0.147 is conventionally 'negligible'.")

    print("\n=== B. the confound, measured: pairing strength FN vs TP ===")
    dn, pn = cliffs_delta(nwc[isfn].astype(float), nwc[~isfn].astype(float))
    print(f"  n_wc: FN median {np.median(nwc[isfn]):.1f}  TP median {np.median(nwc[~isfn]):.1f}"
          f"   Cliff's delta = {dn:+.4f}  p={pn:.2e}")
    print("  (negative = FN are the weaker-paired, as expected)")
    from scipy.stats import spearmanr
    rs = spearmanr(le, nwc.astype(float))
    print(f"  Spearman(expression, n_wc) = {rs.statistic:+.4f} (p={rs.pvalue:.2e})")
    print("  near zero => expression and pairing are near-orthogonal, so the pooled\n"
          "     gap in A is not simply pairing strength wearing a disguise")

    print("\n=== C. expression gap WITHIN n_wc strata (the mediation test) ===")
    print(f"{'n_wc':>8}{'n_FN':>8}{'n_TP':>8}{'med_FN':>10}{'med_TP':>10}"
          f"{'delta':>9}{'p':>11}")
    rows, deltas, weights = [], [], []
    edges = [(0, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13),
             (14, 50)]
    for lo, hi in edges:
        m = (nwc >= lo) & (nwc <= hi)
        rr = _report(le[m], isfn[m], f"{lo}-{hi}" if lo != hi else str(lo))
        if not rr:
            continue
        rows.append(rr)
        deltas.append(rr["cliffs_delta"])
        weights.append(rr["n_fn"] + rr["n_tp"])
        print(f"{rr['stratum']:>8}{rr['n_fn']:>8}{rr['n_tp']:>8}"
              f"{rr['med_fn']:>10.4f}{rr['med_tp']:>10.4f}"
              f"{rr['cliffs_delta']:>+9.4f}{rr['p_value']:>11.2e}")
    if rows:
        w = np.array(weights, dtype=float)
        pooled_within = float(np.average(deltas, weights=w))
        qs = bh_fdr(np.array([r["p_value"] for r in rows]))
        print(f"\n  weight-averaged within-stratum delta = {pooled_within:+.4f}"
              f"   (pooled was {r['cliffs_delta']:+.4f})")
        print(f"  strata significant after BH: {(qs < 0.05).sum()} / {len(qs)}")
        print("  gap survives  -> expression carries FN-risk beyond pairing strength")
        print("  gap collapses -> the pooled signal was pairing strength all along")

    hdr = ("within-region, so composition is fixed" if args.region else
           "WARNING: all regions pooled -- introns have near-zero expression AND a "
           "different FN rate, so this table is region-confounded; rerun with --region")
    print(f"\n=== D. FN rate by expression decile ({hdr}) ===")
    dec = np.floor(np.argsort(np.argsort(le)) / len(le) * 10).astype(int)
    print(f"{'decile':>7}{'median cov':>13}{'n':>9}{'FN rate':>10}{'med n_wc':>10}")
    for d in range(10):
        m = dec == d
        if m.sum() < 50:
            continue
        print(f"{d:>7}{np.median(e[m]):>13.3f}{int(m.sum()):>9}"
              f"{isfn[m].mean():>10.3f}{np.median(nwc[m]):>10.1f}")

    if args.no_gtf:
        return
    gtf = Path(args.gtf).expanduser()
    if not gtf.exists():
        print(f"\n[transcript] GTF not found: {gtf}; pass --gtf or --no-gtf")
        return

    print("\n=== E. transcript-level abundance (MANE Select) vs local coverage ===")
    print("    local 50-nt coverage = transcript abundance x local relative usage.")
    print("    Splitting them says whether the FN signal is ABUNDANCE or APA/3'-bias.")
    tex = transcript_expression(df, args.rnaseq_prefix, str(gtf))
    ok = np.isfinite(tex) & (tex > 0)
    print(f"  sites inside a MANE Select exon: {ok.sum()} / {len(ok)} ({ok.mean():.1%})")
    if ok.sum() < 200:
        print("  too few mapped sites")
        return

    eps = 1e-9
    lt = np.log1p(tex[ok])                                 # abundance
    lr = np.log((e[ok] + eps) / (tex[ok] + eps))           # local relative usage
    ll = le[ok]                                            # local (= lt + lr, up to log1p)
    f = isfn[ok]

    print(f"\n{'component':<28}{'delta':>9}{'p':>12}{'AUROC':>9}")
    for name, v in (("local 50-nt coverage", ll),
                    ("transcript abundance", lt),
                    ("local/transcript (APA-ish)", lr)):
        d, p = cliffs_delta(v[f], v[~f])
        print(f"{name:<28}{d:>+9.4f}{p:>12.2e}{roc_auc_score(f, v):>9.4f}")

    print("\n  mediation of TRANSCRIPT abundance by pairing strength (n_wc):")
    nw = nwc[ok]
    ds, ws = [], []
    for lo, hi in [(0, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13),
                   (14, 50)]:
        m = (nw >= lo) & (nw <= hi)
        if (f[m]).sum() < 20 or (~f[m]).sum() < 20:
            continue
        d, _ = cliffs_delta(lt[m][f[m]], lt[m][~f[m]])
        ds.append(d)
        ws.append(int(m.sum()))
        print(f"    n_wc {lo}-{hi}: delta={d:+.4f}  n={int(m.sum())}")
    if ds:
        dp, _ = cliffs_delta(lt[f], lt[~f])
        print(f"    weight-averaged within-stratum delta = "
              f"{np.average(ds, weights=ws):+.4f}   (pooled {dp:+.4f})")

    print("\n  FN rate by TRANSCRIPT-abundance decile:")
    dd = np.floor(np.argsort(np.argsort(lt)) / len(lt) * 10).astype(int)
    print(f"{'decile':>7}{'median tx cov':>15}{'n':>9}{'FN rate':>10}{'med n_wc':>10}")
    for d in range(10):
        m = dd == d
        if m.sum() < 50:
            continue
        print(f"{d:>7}{np.median(tex[ok][m]):>15.3f}{int(m.sum()):>9}"
              f"{f[m].mean():>10.3f}{np.median(nw[m]):>10.1f}")

    # ---------------------------------------------------------------------- #
    # F. Binding class is the better mediator: n_wc counts Watson-Crick pairs on
    # the best register and is blind to *where* they sit, so a high-n_wc site with
    # no seed looks "strongly paired" to n_wc but not to the model.  Stratifying on
    # the canonical binding category asks the mediation question in the model's own
    # currency.
    # ---------------------------------------------------------------------- #
    print("\n=== F. mediation by BINDING CLASS (rather than n_wc) ===")
    bc = binding_class(df)[ok]
    mediation_table(lt, f, bc,
                    "transcript abundance, FN vs TP, within binding class:")

    print("\n  --- the decoy hypothesis: what ARE the high-n_wc sites? ---")
    print("  If n_wc is a bad proxy for functional pairing, then among strongly")
    print("  'paired' sites the FN should be non-canonical and the TP canonical.")
    hi = nw >= 14
    if hi.sum() > 100:
        canon = np.isin(bc, CANONICAL)
        for name, m in (("n_wc >= 14", hi), ("n_wc <= 9", nw <= 9)):
            cf = canon[m & f].mean() if (m & f).sum() else np.nan
            ct = canon[m & ~f].mean() if (m & ~f).sum() else np.nan
            print(f"    {name:<12} canonical-seed share:  FN {cf:6.1%} (n={int((m & f).sum()):>5})"
                  f"   TP {ct:6.1%} (n={int((m & ~f).sum()):>5})")
        print("\n  abundance delta within n_wc>=14, split by canonical vs not:")
        for name, m in (("canonical", hi & canon), ("non-canonical", hi & ~canon)):
            if (m & f).sum() < 20 or (m & ~f).sum() < 20:
                print(f"    {name:<15} too few")
                continue
            d, p = cliffs_delta(lt[m & f], lt[m & ~f])
            print(f"    {name:<15} delta={d:+.4f}  p={p:.2e}  "
                  f"n_FN={int((m & f).sum())} n_TP={int((m & ~f).sum())}")


if __name__ == "__main__":
    main()
