#!/usr/bin/env python3
"""Widen the MRE sequence in a v7 TSV using MANE-transcript / genomic context.

Each MRE in a v7 TSV is the bare ~50-nt `gene` sequence.  For IntaRNA ensemble
folding of the *extended* target you want the MRE plus flanking context.  This
script rewrites the `gene` column to the widened sequence, choosing the flank
source exactly the way the rest of the repo does (see
``cnn/compute_accessibility.py``):

  * **mane**    — if the MRE falls entirely inside a MANE Select exon, the
    ``+/-flank`` nt are taken from the *mature* transcript, so the flanks splice
    correctly across exon junctions (spliced/exonic context).
  * **genomic** — otherwise (intronic, intergenic, non-MANE, or straddling an
    exon boundary) the flanks are taken from the genome, strand-aware pre-mRNA
    context.

The original coordinate columns (`chr/start/end/strand`) are left untouched — an
MRE's genomic locus does not change, only the sequence context we fold does.
Two columns are added:

    mre_offset  0-based start of the original MRE within the new `gene` string
                (so the ``mre_offset : mre_offset+len`` slice is the real MRE).
    mre_region  the same span as a 1-based inclusive IntaRNA ``--tRegion``
                string ("start-end"), so the ensemble scorer can anchor the
                interaction to the original MRE inside the extended target.
    acc_mode    which context was used: "mane", "genomic", or "unmapped"
                (chrom missing / no coords -> `gene` left as-is, offset 0).

Only unique loci are extracted once, then broadcast back to every row that
shares them.

Usage
-----
    python extend_mre.py \\
        --input  IN_v7.tsv \\
        --output IN_v7_extended.tsv \\
        --genome GRCh38.primary_assembly.genome.fa \\
        --gtf    gencode.vXX.primary_assembly.annotation.gtf.gz \\
        --flank  100

Then score the extended file with the local accessibility window (so the now
long target is not folded whole -- that is O(L^3)):

    ./run_ensemble.sh IN_v7_extended.tsv OUT.tsv --tacc-w 150 --tacc-l 100
"""
from __future__ import annotations

import argparse
import gzip
import sys
import time
from bisect import bisect_right

import numpy as np
import polars as pl
from pyfaidx import Fasta

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def rc(s: str) -> str:
    return s.translate(_COMP)[::-1]


def tsv_chrom_to_fa(chrom) -> str:
    """Map the v7 TSV chromosome label to a GENCODE genome contig name."""
    c = str(chrom)
    if c in ("MT", "chrMT", "M"):
        return "chrM"
    return c if c.startswith("chr") else "chr" + c


# --------------------------------------------------------------------------- #
# MANE Select exon model (copied from cnn/compute_accessibility.py so this tool
# stays self-contained / portable to a server without the main project on PATH)
# --------------------------------------------------------------------------- #
def parse_mane(gtf_path: str):
    """Return (tx, index, max_exon_len) describing every MANE Select exon."""
    opener = gzip.open if gtf_path.endswith(".gz") else open
    tx: dict[str, dict] = {}
    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t")
            if f[2] != "exon":
                continue
            attr = f[8]
            if 'tag "MANE_Select"' not in attr:
                continue
            tid = attr.split('transcript_id "', 1)[1].split('"', 1)[0]
            es, ee = int(f[3]), int(f[4])
            d = tx.get(tid)
            if d is None:
                tx[tid] = {"chrom": f[0], "strand": f[6], "ex": [(es, ee)]}
            else:
                d["ex"].append((es, ee))

    index: dict[tuple, list] = {}
    max_exon_len = 0
    for tid, d in tx.items():
        d["ex"].sort()  # ascending genomic
        cum, c = [], 0
        for es, ee in d["ex"]:
            cum.append(c)
            c += ee - es + 1
            max_exon_len = max(max_exon_len, ee - es + 1)
        d["cum"] = cum
        d["Lt"] = c
        key = (d["chrom"], d["strand"])
        bucket = index.setdefault(key, [])
        for (es, ee), cm in zip(d["ex"], cum):
            bucket.append((es, ee, cm, tid))

    flat: dict[tuple, tuple] = {}
    for key, bucket in index.items():
        bucket.sort()  # by es
        es_arr = np.array([b[0] for b in bucket], dtype=np.int64)
        ee_arr = np.array([b[1] for b in bucket], dtype=np.int64)
        meta = [(b[2], b[0], b[3]) for b in bucket]  # (cum, es, tid)
        flat[key] = (es_arr, ee_arr, meta)
    return tx, flat, max_exon_len


def find_host_exon(index, max_exon_len, chrom, strand, s, e):
    """Return (cum, es, tid) of the MANE exon fully containing [s,e], or None."""
    key = (chrom, strand)
    rec = index.get(key)
    if rec is None:
        return None
    es_arr, ee_arr, meta = rec
    j = bisect_right(es_arr, s)  # exons with es <= s are at indices < j
    k = j - 1
    while k >= 0 and (s - es_arr[k]) <= max_exon_len:
        if ee_arr[k] >= e:  # es<=s and ee>=e  => contains [s,e]
            return meta[k]
        k -= 1
    return None


# --------------------------------------------------------------------------- #
# sequence extraction
# --------------------------------------------------------------------------- #
class ContextBuilder:
    def __init__(self, genome_fa, tx, index, max_exon_len, flank):
        self.fa = Fasta(genome_fa, sequence_always_upper=True, rebuild=False)
        self.tx = tx
        self.index = index
        self.max_exon_len = max_exon_len
        self.flank = flank
        self._txseq: dict[str, str] = {}

    def _txseq_for(self, tid: str) -> str:
        seq = self._txseq.get(tid)
        if seq is None:
            d = self.tx[tid]
            chrom = d["chrom"]
            asc = "".join(str(self.fa[chrom][es - 1:ee]) for es, ee in d["ex"])
            seq = asc if d["strand"] == "+" else rc(asc)
            self._txseq[tid] = seq
        return seq

    def window(self, chrom_fa, strand, s, e, mre_dna):
        """Return (window_seq, mre_offset, mode) for one locus.

        s,e     : 1-based inclusive genomic coords of the MRE.
        mre_dna : expected MRE sequence (DNA, transcript/strand orientation).
        """
        flank = self.flank
        mre_len = len(mre_dna)
        # ---- try MANE mature-transcript context ---------------------------- #
        host = find_host_exon(self.index, self.max_exon_len, chrom_fa, strand, s, e)
        if host is not None:
            cum, es, tid = host
            d = self.tx[tid]
            Lt = d["Lt"]
            a_s = cum + (s - es)  # ascending-concat coord of genomic s
            a_e = cum + (e - es)
            tx_lo = a_s if strand == "+" else (Lt - 1 - a_e)
            seqT = self._txseq_for(tid)
            if 0 <= tx_lo and seqT[tx_lo:tx_lo + mre_len] == mre_dna:
                lo = max(0, tx_lo - flank)
                hi = min(Lt, tx_lo + mre_len + flank)
                return seqT[lo:hi], tx_lo - lo, "mane"
        # ---- genomic fallback (strand-aware pre-mRNA context) -------------- #
        clen = len(self.fa[chrom_fa])
        gws = max(0, s - 1 - flank)
        gwe = min(clen, e + flank)
        win_plus = str(self.fa[chrom_fa][gws:gwe])
        if strand == "-":
            win = rc(win_plus)
            off = gwe - e  # right padding on + strand becomes 5' offset
        else:
            win = win_plus
            off = (s - 1) - gws
        return win, off, "genomic"


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="v7 TSV")
    ap.add_argument("--output", required=True, help="extended v7 TSV")
    ap.add_argument("--genome", required=True,
                    help="GRCh38 primary assembly .fa (indexed; pyfaidx builds "
                         "the .fai on first use)")
    ap.add_argument("--gtf", required=True, help="GENCODE annotation .gtf(.gz)")
    ap.add_argument("--flank", type=int, default=100,
                    help="nt of context added on EACH side of the MRE "
                         "(default 100 -> a 50 nt MRE becomes ~250 nt)")
    ap.add_argument("--gene-col", default="gene",
                    help="MRE-sequence column to widen (default: gene)")
    ap.add_argument("--offset-col", default="mre_offset",
                    help="name for the added MRE-offset column")
    ap.add_argument("--region-col", default="mre_region",
                    help="name for the added 1-based --tRegion string column")
    ap.add_argument("--mode-col", default="acc_mode",
                    help="name for the added context-mode column")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N rows")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] parsing MANE exons from {args.gtf}", flush=True)
    tx, index, max_exon_len = parse_mane(args.gtf)
    print(f"[{time.time()-t0:6.1f}s]   {len(tx)} MANE transcripts, "
          f"max exon {max_exon_len} nt", flush=True)

    key_cols = [args.gene_col, "chr", "start", "end", "strand"]
    df = pl.read_csv(args.input, separator="\t", infer_schema_length=10000)
    if args.limit:
        df = df.head(args.limit)
    for c in key_cols:
        if c not in df.columns:
            sys.exit(f"missing column {c!r} in {args.input}")

    loci = (df.select(key_cols)
              .drop_nulls([args.gene_col, "chr", "start", "end", "strand"])
              .unique())
    print(f"[{time.time()-t0:6.1f}s] {df.height} rows, {loci.height} unique loci",
          flush=True)

    builder = ContextBuilder(args.genome, tx, index, max_exon_len, args.flank)

    ext_seq, off_list, reg_list, mode_list = [], [], [], []
    genes, chrs, starts, ends, strands = [], [], [], [], []
    n_mane = n_geno = n_unmapped = 0
    for row in loci.iter_rows(named=True):
        gene = row[args.gene_col]
        chrom_fa = tsv_chrom_to_fa(row["chr"])
        s, e, strand = int(row["start"]), int(row["end"]), row["strand"]
        if gene is None or chrom_fa not in builder.fa:
            win, off, mode = (gene or ""), 0, "unmapped"
            n_unmapped += 1
        else:
            mre_dna = gene.upper().replace("U", "T")
            win, off, mode = builder.window(chrom_fa, strand, s, e, mre_dna)
            if mode == "mane":
                n_mane += 1
            else:
                n_geno += 1
        # 1-based inclusive span of the original MRE within `win`
        mre_len = len(gene) if gene else 0
        reg = f"{off + 1}-{off + mre_len}" if mre_len else ""
        genes.append(gene); chrs.append(row["chr"])
        starts.append(s); ends.append(e); strands.append(strand)
        ext_seq.append(win); off_list.append(off)
        reg_list.append(reg); mode_list.append(mode)
    print(f"[{time.time()-t0:6.1f}s] extended {len(ext_seq)} loci "
          f"(mane={n_mane}, genomic={n_geno}, unmapped={n_unmapped})", flush=True)

    locus_df = pl.DataFrame({
        args.gene_col: genes,
        "chr": chrs,
        "start": starts,
        "end": ends,
        "strand": strands,
        "__ext_seq": ext_seq,
        args.offset_col: off_list,
        args.region_col: reg_list,
        args.mode_col: mode_list,
    }).with_columns(
        pl.col("start").cast(df.schema["start"]),
        pl.col("end").cast(df.schema["end"]),
    )

    out = df.join(locus_df, on=key_cols, how="left")
    # replace gene with the extended sequence where we have one; keep original
    # (offset 0 / unmapped) where the locus could not be mapped
    out = out.with_columns(
        pl.when(pl.col("__ext_seq").is_not_null() & (pl.col("__ext_seq") != ""))
          .then(pl.col("__ext_seq"))
          .otherwise(pl.col(args.gene_col))
          .alias(args.gene_col),
        pl.col(args.offset_col).fill_null(0),
        pl.col(args.region_col).fill_null(""),
        pl.col(args.mode_col).fill_null("unmapped"),
    ).drop("__ext_seq")

    lens = out.select(pl.col(args.gene_col).str.len_chars())
    out.write_csv(args.output, separator="\t")
    print(f"[{time.time()-t0:6.1f}s] wrote {args.output}  ({out.height} rows, "
          f"extended gene length {lens.min().item()}-{lens.max().item()} nt)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
