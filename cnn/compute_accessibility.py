#!/usr/bin/env python
"""Precompute per-MRE accessibility (tAcc) for the miRNA-target CNN.

Each MRE (the 50-nt `gene` sequence in the v7 TSVs) is mapped to its flanking
context and folded locally to obtain a per-base unpaired probability (Pu, in
[0,1]) for the 50 MRE positions.  Pu == 1 means fully accessible/unpaired,
Pu == 0 means paired/inaccessible.

Context selection (decided experimentally):
  * exonic / UTR sites that fall inside a MANE Select exon  -> the MRE +/-FLANK
    nt are taken from the *mature* transcript, so the flanks splice correctly
    across exon junctions (mode = "mane").
  * everything else (intronic ~36%, intergenic, non-MANE genes, or a site that
    straddles an exon/intron boundary) -> MRE +/-FLANK nt from the genome,
    strand-aware pre-mRNA context (mode = "genomic").

Accessibility is computed with ViennaRNA's local (RNAplfold-style) windowed
algorithm so long-range pairs are not over-counted (Tafer/Lange).

The heavy work (folding) is parallelised and done once per *unique* genomic
locus, then broadcast back to every row that shares it.

The output is a CNN-ready TSV: every input column is passed through, the
sequence columns are renamed to what the trainer expects (gene -> mre_sequence,
noncodingRNA -> mirna_sequence), `acc_mode` records the context used, and the
accessibility vector is written to `tAcc` as a comma-separated string that
cnn_branches_mirbind.py reads directly (train with --seq-acc-channel).

    python cnn/compute_accessibility.py \
        --input  data/AGO2_eCLIP_Manakov2022_train_v7.tsv \
        --output data/AGO2_eCLIP_Manakov2022_train_acc.tsv \
        --genome ~/Downloads/hg38/GRCh38.primary_assembly.genome.fa \
        --gtf    ~/Downloads/hg38/gencode.v47.primary_assembly.annotation.gtf.gz \
        --workers 20
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
import time
from bisect import bisect_right
from multiprocessing import Pool

import numpy as np
import polars as pl
from pyfaidx import Fasta

import RNA

MRE_LEN = 50
_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def rc(s: str) -> str:
    return s.translate(_COMP)[::-1]


def tsv_chrom_to_fa(chrom: str) -> str:
    """Map the v7 TSV chromosome label to a GENCODE genome contig name."""
    c = str(chrom)
    if c in ("MT", "chrMT", "M"):
        return "chrM"
    return c if c.startswith("chr") else "chr" + c


# --------------------------------------------------------------------------- #
# MANE Select exon model
# --------------------------------------------------------------------------- #
def parse_mane(gtf_path: str):
    """Return (tx, index, max_exon_len).

    tx[tid]    = {"chrom","strand","ex":[(es,ee)...ascending],"cum":[...],"Lt":int}
    index[(chrom,strand)] = (es_arr, ee_arr, meta_list) sorted by es; meta entry
                            is (cum_offset_of_exon, es, tid).
    """
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
        """Return (window_seq, mre_offset, mode) for a single locus.

        chrom_fa  : genome contig name (chr-prefixed).
        s,e       : 1-based inclusive genomic coords of the MRE.
        mre_dna   : expected MRE sequence (DNA, transcript/strand orientation).
        """
        flank = self.flank
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
            if 0 <= tx_lo and seqT[tx_lo:tx_lo + MRE_LEN] == mre_dna:
                lo = max(0, tx_lo - flank)
                hi = min(Lt, tx_lo + MRE_LEN + flank)
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
# folding (worker side)
# --------------------------------------------------------------------------- #
_WIN = 150
_MAXSPAN = 100


def _init_worker(window_size, max_span):
    global _WIN, _MAXSPAN
    _WIN, _MAXSPAN = window_size, max_span
    # one ViennaRNA process per core; avoid nested OpenMP thread oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _fold_slice(args):
    """args = (window_seq, mre_offset).  Returns float32[50] of Pu over the MRE."""
    seq, off = args
    n = len(seq)
    try:
        # m[i][1] = prob that base i (1-based) is unpaired (RNAplfold-style, local)
        m = RNA.pfl_fold_up(seq, 1, min(_WIN, n), min(_MAXSPAN, n))
    except Exception:
        # degenerate sequence (e.g. runs of N) -> mark as missing
        return np.full(MRE_LEN, np.nan, dtype=np.float32)
    # MRE occupies window[off:off+50] (0-based) -> rows off+1 .. off+50
    hi = min(off + MRE_LEN, n)
    out = np.array([m[i][1] for i in range(off + 1, hi + 1)], dtype=np.float32)
    if out.shape[0] != MRE_LEN:  # clamped window near a transcript end
        out = np.pad(out, (0, MRE_LEN - out.shape[0]), constant_values=np.nan)
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="v7 TSV")
    ap.add_argument("--output", required=True,
                    help="CNN-ready output TSV (read directly by "
                         "cnn_branches_mirbind.py)")
    ap.add_argument("--genome", required=True, help="GRCh38 primary assembly .fa (indexed)")
    ap.add_argument("--gtf", required=True, help="GENCODE annotation .gtf(.gz)")
    ap.add_argument("--flank", type=int, default=150)
    ap.add_argument("--window", type=int, default=150, help="RNAplfold W")
    ap.add_argument("--max-span", type=int, default=100, help="RNAplfold L")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gene-col", default="gene",
                    help="source MRE-sequence column in the input (default: gene)")
    ap.add_argument("--mirna-col", default="noncodingRNA",
                    help="source miRNA-sequence column in the input "
                         "(default: noncodingRNA)")
    ap.add_argument("--mre-out", default="mre_sequence",
                    help="MRE-sequence column name the CNN expects")
    ap.add_argument("--mirna-out", default="mirna_sequence",
                    help="miRNA-sequence column name the CNN expects")
    ap.add_argument("--acc-col", default="tAcc",
                    help="accessibility column name the CNN expects")
    ap.add_argument("--decimals", type=int, default=5,
                    help="rounding for the serialised tAcc vector (default: 5)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N rows")
    args = ap.parse_args()

    if not str(args.output).endswith(".tsv"):
        print(f"  WARNING: --output {args.output!r} does not end in '.tsv'; the "
              f"CNN only tab-splits files ending in .tsv, so rename it or it "
              f"will be misread.", flush=True)

    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] parsing MANE exons from {args.gtf}", flush=True)
    tx, index, max_exon_len = parse_mane(args.gtf)
    print(f"[{time.time()-t0:6.1f}s]   {len(tx)} MANE transcripts, "
          f"max exon {max_exon_len} nt", flush=True)

    cols = [args.gene_col, "chr", "start", "end", "strand"]
    df = pl.read_csv(args.input, separator="\t", columns=None,
                     infer_schema_length=10000)
    if args.limit:
        df = df.head(args.limit)
    for c in cols:
        if c not in df.columns:
            sys.exit(f"missing column {c!r} in {args.input}")

    # unique loci (locus = genomic interval + strand + the MRE sequence itself,
    # because the gene seq is the authoritative target sequence)
    loci = (df.select(cols)
              .drop_nulls(["chr", "start", "end", "strand"])
              .unique())
    print(f"[{time.time()-t0:6.1f}s] {df.height} rows, {loci.height} unique loci",
          flush=True)

    builder = ContextBuilder(args.genome, tx, index, max_exon_len, args.flank)

    windows = []          # (win_seq, off)
    keys = []             # (gene, chr, start, end, strand)
    modes = []
    n_mane = n_geno = 0
    for row in loci.iter_rows(named=True):
        gene = row[args.gene_col]
        if gene is None:
            continue
        mre_dna = gene.upper().replace("U", "T")
        chrom_fa = tsv_chrom_to_fa(row["chr"])
        if chrom_fa not in builder.fa:
            continue
        s, e = int(row["start"]), int(row["end"])
        strand = row["strand"]
        win, off, mode = builder.window(chrom_fa, strand, s, e, mre_dna)
        windows.append((win, off))
        keys.append((gene, row["chr"], s, e, strand))
        modes.append(mode)
        if mode == "mane":
            n_mane += 1
        else:
            n_geno += 1
    print(f"[{time.time()-t0:6.1f}s] built {len(windows)} windows "
          f"(mane={n_mane}, genomic={n_geno}); folding...", flush=True)

    with Pool(args.workers, initializer=_init_worker,
              initargs=(args.window, args.max_span)) as pool:
        tacc = pool.map(_fold_slice, windows, chunksize=256)
    print(f"[{time.time()-t0:6.1f}s] folded {len(tacc)} loci", flush=True)

    locus_df = pl.DataFrame({
        args.gene_col: [k[0] for k in keys],
        "chr": [k[1] for k in keys],
        "start": [k[2] for k in keys],
        "end": [k[3] for k in keys],
        "strand": [k[4] for k in keys],
        args.acc_col: [a.tolist() for a in tacc],
        "acc_mode": modes,
    })
    # normalise join-key dtypes to match df
    locus_df = locus_df.with_columns(
        pl.col("start").cast(df.schema["start"]),
        pl.col("end").cast(df.schema["end"]),
    )
    out = df.join(locus_df, on=cols, how="left")
    cov = out.select(pl.col(args.acc_col).is_not_null().mean()).item()

    # ---- emit a CNN-ready TSV ------------------------------------------------ #
    # Serialise tAcc to a comma-separated string (the CNN's _parse_vector reads
    # it back) and rename the sequence columns to the names the trainer expects.
    out = out.with_columns(
        pl.col(args.acc_col)
          .list.eval(pl.element().round(args.decimals).cast(pl.String))
          .list.join(",")
          .alias(args.acc_col)
    )
    rename = {}
    if args.gene_col != args.mre_out:
        rename[args.gene_col] = args.mre_out
    if args.mirna_col in out.columns and args.mirna_col != args.mirna_out:
        rename[args.mirna_col] = args.mirna_out
    elif args.mirna_col not in out.columns:
        print(f"  WARNING: miRNA column {args.mirna_col!r} not in input; the CNN "
              f"needs a {args.mirna_out!r} column — pass --mirna-col.", flush=True)
    out = out.rename(rename)

    out.write_csv(args.output, separator="\t")
    print(f"[{time.time()-t0:6.1f}s] wrote {args.output}  "
          f"({out.height} rows, tAcc coverage {cov:.3%})", flush=True)


if __name__ == "__main__":
    main()
