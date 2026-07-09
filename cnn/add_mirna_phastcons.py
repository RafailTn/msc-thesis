#!/usr/bin/env python
"""Annotate v7 TSVs with a per-position phastCons100way vector over the *miRNA* locus.

Adds two columns, leaving every existing column untouched:

  ``mirna_phastCons100way``        list of ``len(noncodingRNA)`` floats, 5'->3',
                                   aligned position-for-position with the
                                   ``noncodingRNA`` sequence (so index 1..7 is the seed)
  ``mirna_phastCons100way_nloci``  how many genomic loci the vector was averaged over
                                   (0 = unmapped, column is null)

Why this is not a one-liner
---------------------------
The v7 schema has **no miRNA coordinates** -- only a sequence, a name, and a family.
Mapping ``noncodingRNA_name`` to a locus is one-to-many in two separate ways:

1.  42% of rows carry a *multi-mapping* name like
    ``hsa-let-7b-5p|hsa-let-7c-5p|hsa-let-7i-5p`` (the read is compatible with several
    mature miRNAs).  161 of the 425 such groups contain loci of *different lengths*, so
    "just take the first component" silently produces vectors of the wrong length.
2.  Even a single mature name may sit at several genomic loci (117 names, 15% of rows) --
    e.g. the three let-7a copies.  Same mature sequence, different conservation.

Resolution: ask the **genome** which loci actually carry this row's sequence.  For each
candidate locus (every locus of every name component) we extract the strand-corrected
genomic sequence and keep only the loci matching ``noncodingRNA`` exactly.  This resolves
99.8% of (name, sequence) pairs, and guarantees ``len(vector) == len(noncodingRNA)``.
Where several loci match, the mature sequence is identical at each but conservation is
not, so the vectors are averaged elementwise and ``_nloci`` records how many.

Strand: BigWig returns values in genomic left-to-right order.  A minus-strand miRNA's
mature sequence runs right-to-left in genomic coordinates, so the vector is **reversed**
for minus-strand loci.  Without this, half the miRNAs would have their seed at the wrong
end -- the single most damaging silent bug available here.

    python cnn/add_mirna_phastcons.py \
        --input data/AGO2_eCLIP_Manakov2022_{train,test,leftout}_v7.tsv \
        --bigwig data/hg38.phastCons100way.bw \
        --gff3 data/hsa.gff3 \
        --genome ~/Downloads/hg38/GRCh38.primary_assembly.genome.fa
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict

import numpy as np
import polars as pl
import pyBigWig
from pyfaidx import Fasta

_COMP = str.maketrans("ACGTN", "TGCAN")


def parse_mirbase(gff3: str) -> dict[str, list[tuple]]:
    """Name -> [(chrom, start, end, strand), ...] for mature `miRNA` features (1-based)."""
    loci: dict[str, list[tuple]] = defaultdict(list)
    with open(gff3) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "miRNA":
                continue
            m = re.search(r"Name=([^;]+)", f[8])
            if m:
                loci[m.group(1)].append((f[0], int(f[3]), int(f[4]), f[6]))
    return loci


def genomic_seq(fa: Fasta, c: str, s: int, e: int, strand: str) -> str | None:
    """Strand-corrected 5'->3' genomic sequence for 1-based inclusive [s, e]."""
    if c not in fa:
        return None
    x = str(fa[c][s - 1:e]).upper()
    return x.translate(_COMP)[::-1] if strand == "-" else x


def locus_vector(bw, c: str, s: int, e: int, strand: str) -> np.ndarray | None:
    """phastCons over [s, e] (1-based inclusive), returned 5'->3'."""
    ch = bw.chroms()
    if c not in ch:
        return None
    s0, e0 = max(s - 1, 0), min(e, ch[c])
    if e0 <= s0:
        return None
    try:
        v = np.asarray(bw.values(c, s0, e0, numpy=True), dtype=float)
    except (RuntimeError, OverflowError):
        return None
    return v[::-1] if strand == "-" else v


def build_cache(pairs, loci, fa, bw, agg: str = "mean"):
    """(name, seq) -> (vector | None, n_loci).  Vector is nan-masked where the track has
    no data; a position is null only if *every* contributing locus lacks data."""
    reduce = {"mean": np.nanmean, "max": np.nanmax, "min": np.nanmin}[agg]
    cache, stats = {}, defaultdict(int)
    for name, seq in pairs:
        su = seq.upper().replace("U", "T")
        cand = [l for p in name.split("|") if p in loci for l in loci[p]]
        if not cand:
            cache[(name, seq)] = (None, 0)
            stats["no miRBase entry"] += 1
            continue
        exact = [l for l in cand if genomic_seq(fa, *l) == su]
        if not exact:
            cache[(name, seq)] = (None, 0)
            stats["no exact genomic sequence match"] += 1
            continue
        vecs = [v for l in exact if (v := locus_vector(bw, *l)) is not None
                and len(v) == len(su)]
        if not vecs:
            cache[(name, seq)] = (None, 0)
            stats["no bigwig data"] += 1
            continue
        with np.errstate(invalid="ignore"):
            # all-NaN columns legitimately reduce to NaN here (written as None)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                combined = reduce(np.vstack(vecs), axis=0)
        cache[(name, seq)] = (combined, len(vecs))
        stats[f"resolved ({min(len(vecs), 5)} loci)"] += 1
    return cache, stats


def fmt(v: np.ndarray | None) -> str | None:
    """Match the existing gene_phastCons formatting: a Python list literal, 3 decimals.

    A position with no phastCons data is written ``None``, not ``nan``.  The point of
    matching the existing column is that ``ast.literal_eval`` parses it, and ``nan`` is a
    bare Name rather than a literal, so it would raise.  ``np.array(lst, dtype=float)``
    turns the ``None`` back into ``np.nan`` on the way in.
    """
    if v is None:
        return None
    return "[" + ", ".join("None" if not np.isfinite(x) else str(round(float(x), 3))
                           for x in v) + "]"


def check_gene_orientation(df: pl.DataFrame, bw, n: int = 400) -> None:
    """Is the existing `gene_phastCons` in genomic or transcript order on the minus strand?

    `gene` is transcript-oriented (never reverse-complemented), but a BigWig yields genomic
    order.  Correlate the stored 470-way column against our 100-way track in both
    orientations; the tracks are strongly correlated, so the better fit reveals the
    convention.  A mismatch would mean any positional use of gene_phastCons is reversed on
    ~half the rows.
    """
    if "gene_phastCons" not in df.columns:
        return
    sub = df.filter(pl.col("strand").cast(pl.Utf8) == "-").head(n)
    fwd, rev = [], []
    for c, s, e, stored in zip(sub["chr"], sub["start"], sub["end"], sub["gene_phastCons"]):
        c = str(c)
        c = c if c.startswith("chr") else "chr" + c
        try:
            got = np.asarray(bw.values(c, int(s) - 1, int(e), numpy=True), dtype=float)
            ref = np.array([float(x) for x in str(stored).strip("[]").split(",")])
        except Exception:
            continue
        if len(got) != len(ref) or not np.isfinite(got).all():
            continue
        fwd.append(np.corrcoef(ref, got)[0, 1])
        rev.append(np.corrcoef(ref, got[::-1])[0, 1])
    if len(fwd) < 20:
        print("  [orientation] too few usable minus-strand rows")
        return
    f, r = float(np.nanmean(fwd)), float(np.nanmean(rev))
    print(f"  [orientation] minus-strand gene_phastCons vs phastCons100way: "
          f"genomic-order r={f:.3f}   reversed r={r:.3f}  (n={len(fwd)})")
    print("  [orientation] => gene_phastCons is stored in "
          f"{'GENOMIC' if f > r else 'TRANSCRIPT (reversed)'} order."
          + ("  It therefore does NOT align with the transcript-oriented `gene` column."
             if f > r else "  Consistent with the `gene` column."))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True, help="v7 TSVs")
    ap.add_argument("--bigwig", default="data/hg38.phastCons100way.bw")
    ap.add_argument("--gff3", default="data/hsa.gff3")
    ap.add_argument("--genome",
                    default=os.path.expanduser(
                        "~/Downloads/hg38/GRCh38.primary_assembly.genome.fa"))
    ap.add_argument("--col", default="mirna_phastCons100way")
    ap.add_argument("--out-suffix", default="_mirnacons",
                    help="output = <stem><suffix>.tsv (default _mirnacons)")
    ap.add_argument("--check-orientation", action="store_true",
                    help="diagnose whether the existing gene_phastCons is genomic- or "
                         "transcript-ordered on the minus strand")
    ap.add_argument("--agg", default="mean", choices=["mean", "max", "min"],
                    help="how to combine the loci of a multi-locus miRNA (default mean); "
                         "_nloci records how many, so this is always recomputable")
    ap.add_argument("--keep-unmapped", action="store_true",
                    help="keep rows whose miRNA has no miRBase locus, with a null vector "
                         "and _nloci=0.  Default drops them (40 rows across all v7 files: "
                         "hsa-miR-1973 and hsa-miR-378g, both retired from miRBase and "
                         "both noncodingRNA_fam='unknown')")
    args = ap.parse_args()

    for p in (args.bigwig, args.gff3, args.genome):
        if not os.path.exists(p):
            raise SystemExit(f"missing input: {p}")

    loci = parse_mirbase(args.gff3)
    fa = Fasta(args.genome, sequence_always_upper=True, rebuild=False)
    bw = pyBigWig.open(args.bigwig)
    print(f"miRBase: {len(loci)} mature names, "
          f"{sum(len(v) for v in loci.values())} loci")

    frames = {p: pl.read_csv(p, separator="\t", infer_schema_length=5000)
              for p in args.input}
    pairs = sorted({(n, s) for d in frames.values()
                    for n, s in zip(d["noncodingRNA_name"], d["noncodingRNA"])})
    print(f"unique (name, sequence) pairs across {len(frames)} files: {len(pairs)}")

    cache, stats = build_cache(pairs, loci, fa, bw, agg=args.agg)
    print("\nlocus resolution:")
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<34} {v:>5}  ({v / len(pairs):.1%})")

    if args.check_orientation:
        print()
        check_gene_orientation(next(iter(frames.values())), bw)

    print()
    for path, d in frames.items():
        vec, nl = [], []
        for n, s in zip(d["noncodingRNA_name"], d["noncodingRNA"]):
            v, k = cache[(n, s)]
            vec.append(fmt(v))
            nl.append(k)
        d = d.with_columns(pl.Series(args.col, vec, dtype=pl.Utf8),
                           pl.Series(f"{args.col}_nloci", nl, dtype=pl.Int32))
        before = d.height
        if not args.keep_unmapped:
            d = d.filter(pl.col(f"{args.col}_nloci") > 0)
        stem, ext = os.path.splitext(path)
        out = f"{stem}{args.out_suffix}{ext}"
        d.write_csv(out, separator="\t")
        avg = float((d[f"{args.col}_nloci"] > 1).mean())
        print(f"-> {out}")
        print(f"     rows {before:,} -> {d.height:,}  (dropped {before - d.height} unmapped)"
              f"   averaged over >1 locus: {avg:.2%}")
    bw.close()


if __name__ == "__main__":
    main()
