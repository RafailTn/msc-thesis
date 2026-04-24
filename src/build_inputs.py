#!/usr/bin/env python3
"""
build_inputs.py — generate FASTA + conservation TSV inputs for predict_target.py

Given a file of (miRNA isoform, gene) pairs for Homo sapiens, fetches:
  * mature miRNA sequences from miRBase (cached locally)
  * gene genomic coordinates + sequences from Ensembl REST (GRCh38 — hg38
    compatible with phastCons470way)
and slides a 50-nt / 25-nt window over the whole gene locus. Output:

  <out>/mirna.fa          — miRNA sequences, one record per output row
  <out>/mre.fa            — 50-nt MRE candidates, one record per output row
  <out>/conservation.tsv  — (chr, start, end, strand) for each MRE, ready for
                            the -conservation_tsv + phastCons470way BigWig path
                            of predict_target.py

The three files are index-aligned (row i of the TSV ↔ record i of each FASTA),
which is what intarna_parallel.py and feature_extraction.py assume.

Input file (TSV by default; use --sep ',' for CSV). The two required columns
default to `mirna_id` and `gene_name` but can be renamed via CLI flags. Each
gene entry can be either an HGNC symbol (e.g. BRCA1) or an Ensembl gene id
(ENSG...). Example:

    mirna_id<TAB>gene_name
    hsa-miR-21-5p<TAB>PTEN
    hsa-let-7a-5p<TAB>ENSG00000136997

Usage:
    python build_inputs.py -i pairs.tsv -o out_dir/
    # then feed the outputs into predict_target.py:
    python predict_target.py \\
        -target_fasta out_dir/mre.fa \\
        -query_fasta  out_dir/mirna.fa \\
        -conservation_tsv out_dir/conservation.tsv \\
        -model /path/to/autogluon_model
"""

import argparse
import csv
import gzip
import sys
import time
from pathlib import Path
from typing import Iterable

import requests


MIRBASE_URL   = "https://mirbase.org/download/mature.fa"
ENSEMBL_BASE  = "https://rest.ensembl.org"
VALID_CHROMS  = {str(i) for i in range(1, 23)} | {'X', 'Y', 'MT'}
RATE_LIMIT_S  = 1 / 14  # Ensembl REST allows ~15 req/s


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ─── miRBase ────────────────────────────────────────────────────────────────

def load_mirbase(cache_path: Path) -> dict[str, str]:
    """Download (if needed) miRBase mature.fa and return {name: DNA_seq}."""
    if not cache_path.exists():
        _log(f"Downloading miRBase mature.fa → {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(MIRBASE_URL, timeout=60)
        r.raise_for_status()
        cache_path.write_bytes(r.content)

    opener = gzip.open if cache_path.suffix == '.gz' else open
    name_to_seq: dict[str, str] = {}
    header: str | None = None
    buf: list[str] = []
    with opener(cache_path, 'rt') as fh:
        for line in fh:
            if line.startswith('>'):
                if header is not None:
                    name_to_seq[header] = ''.join(buf).upper().replace('U', 'T')
                header = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if header is not None:
        name_to_seq[header] = ''.join(buf).upper().replace('U', 'T')

    _log(f"Loaded {len(name_to_seq):,} mature miRNA sequences from miRBase")
    return name_to_seq


# ─── Ensembl REST client ────────────────────────────────────────────────────

class EnsemblClient:
    """Minimal rate-limited client for Ensembl REST (GRCh38)."""

    def __init__(self, base_url: str = ENSEMBL_BASE, timeout: int = 30) -> None:
        self.base = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self._last = 0.0

    def _throttle(self) -> None:
        dt = time.monotonic() - self._last
        if dt < RATE_LIMIT_S:
            time.sleep(RATE_LIMIT_S - dt)
        self._last = time.monotonic()

    def _get(self, path: str, accept: str = 'application/json'):
        self._throttle()
        r = self.session.get(
            f"{self.base}{path}",
            headers={'Accept': accept},
            timeout=self.timeout,
        )
        if r.status_code == 429:
            retry = float(r.headers.get('Retry-After', '1'))
            _log(f"Rate-limited; sleeping {retry:.1f}s")
            time.sleep(retry + 0.1)
            return self._get(path, accept)
        r.raise_for_status()
        return r.json() if accept == 'application/json' else r.text

    def lookup_gene(self, identifier: str) -> dict | None:
        """Return {chrom, start, end, strand} or None if not found on GRCh38."""
        endpoint = (
            f"/lookup/id/{identifier}"
            if identifier.upper().startswith('ENSG')
            else f"/lookup/symbol/homo_sapiens/{identifier}"
        )
        try:
            data = self._get(endpoint)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (400, 404):
                return None
            raise
        if data.get('assembly_name') != 'GRCh38':
            _log(f"  Warning: {identifier} is not on GRCh38; skipping")
            return None
        return {
            'chrom':  str(data['seq_region_name']),
            'start':  int(data['start']),
            'end':    int(data['end']),
            'strand': int(data['strand']),
        }

    def fetch_sequence(self, chrom: str, start: int, end: int, strand: int) -> str:
        """Return genomic DNA on the gene's transcribed (mRNA-sense) strand."""
        text = self._get(
            f"/sequence/region/human/{chrom}:{start}..{end}:{strand}",
            accept='text/plain',
        )
        return ''.join(text.split()).upper()


# ─── Sliding window + coord math ────────────────────────────────────────────

def slide(seq: str, win: int, step: int) -> Iterable[tuple[int, str]]:
    """Yield (offset_0based, window_seq) pairs of length exactly `win`."""
    for i in range(0, len(seq) - win + 1, step):
        yield i, seq[i:i + win]


def window_coords(
    offset: int, win: int, gene_start: int, gene_end: int, strand: int
) -> tuple[int, int]:
    """
    Genomic 1-based inclusive (start, end) for a window at mRNA-sense offset.

    + strand gene: window lives at [gene_start + offset, gene_start + offset + win - 1].
    - strand gene: the Ensembl response with strand=-1 reverse-complements the
      genomic + sequence, so mRNA-sense offset 0 sits at gene_end and offset
      grows leftward on the genome.
    """
    if strand == 1:
        gstart = gene_start + offset
        gend   = gstart + win - 1
    else:
        gend   = gene_end - offset
        gstart = gend - win + 1
    return gstart, gend


# ─── Input parsing ──────────────────────────────────────────────────────────

def read_pairs(path: Path, sep: str, mirna_col: str, gene_col: str):
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter=sep)
        if reader.fieldnames is None:
            raise SystemExit(f"Input file {path} appears to be empty")
        missing = {mirna_col, gene_col} - set(reader.fieldnames)
        if missing:
            raise SystemExit(
                f"Input file is missing required columns: {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            mirna = (row[mirna_col] or '').strip()
            gene  = (row[gene_col]  or '').strip()
            if mirna and gene:
                yield mirna, gene


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('-i', '--input', required=True, type=Path,
                    help='TSV/CSV file with miRNA and gene columns')
    ap.add_argument('-o', '--output', required=True, type=Path,
                    help='Output directory (created if absent)')
    ap.add_argument('--sep', default='\t',
                    help="Input column separator (default: TAB; use ',' for CSV)")
    ap.add_argument('--mirna-col', default='mirna_id',
                    help='Column in the input file holding miRNA isoform names '
                         '(e.g. hsa-miR-21-5p). Default: mirna_id')
    ap.add_argument('--gene-col', default='gene_name',
                    help='Column holding HGNC symbols or Ensembl gene ids. '
                         'Default: gene_name')
    ap.add_argument('--window-size', type=int, default=50,
                    help='MRE window length (default: 50). Downstream feature '
                         'extraction assumes exactly 50 — change at your own risk.')
    ap.add_argument('--step', type=int, default=25,
                    help='Sliding-window step (default: 25)')
    ap.add_argument('--mirbase-fa', type=Path, default=None,
                    help='Cached miRBase mature.fa[.gz] (auto-downloaded if missing)')
    args = ap.parse_args()

    if args.window_size != 50:
        _log("WARNING: downstream feature_extraction.py hardcodes a 50-nt MRE "
             "window; --window-size != 50 will produce features that are "
             "silently misaligned.")

    args.output.mkdir(parents=True, exist_ok=True)
    mirbase_cache = args.mirbase_fa or (args.output / 'mature.fa')

    mirbase = load_mirbase(mirbase_cache)
    ensembl = EnsemblClient()

    pairs = list(read_pairs(args.input, args.sep, args.mirna_col, args.gene_col))
    _log(f"Read {len(pairs):,} (miRNA, gene) pairs from {args.input}")

    # Per-gene cache so Ensembl is hit at most twice per unique gene.
    gene_info: dict[str, dict | None] = {}
    gene_seq:  dict[str, str | None]  = {}

    missing_mirnas: set[str] = set()
    missing_genes:  set[str] = set()
    skipped_chrom:  set[str] = set()
    n_rows = 0

    mirna_fa = args.output / 'mirna.fa'
    mre_fa   = args.output / 'mre.fa'
    cons_tsv = args.output / 'conservation.tsv'

    with mirna_fa.open('w') as fm, mre_fa.open('w') as ft, \
         cons_tsv.open('w', newline='') as fc:
        writer = csv.writer(fc, delimiter='\t', lineterminator='\n')
        writer.writerow(['chr', 'start', 'end', 'strand', 'gene', 'mirna'])

        for mirna_id, gene_name in pairs:
            if mirna_id not in mirbase:
                missing_mirnas.add(mirna_id)
                continue

            if gene_name not in gene_info:
                _log(f"Ensembl lookup: {gene_name}")
                gene_info[gene_name] = ensembl.lookup_gene(gene_name)

            gene = gene_info[gene_name]
            if gene is None:
                missing_genes.add(gene_name)
                continue
            if gene['chrom'] not in VALID_CHROMS:
                skipped_chrom.add(gene['chrom'])
                continue

            if gene_name not in gene_seq:
                gene_seq[gene_name] = ensembl.fetch_sequence(
                    gene['chrom'], gene['start'], gene['end'], gene['strand'],
                )
            seq = gene_seq[gene_name]
            if not seq or len(seq) < args.window_size:
                _log(f"  {gene_name}: sequence shorter than window; skipping")
                continue

            mirna_seq  = mirbase[mirna_id]
            strand_sym = '+' if gene['strand'] == 1 else '-'

            for w_idx, (off, win) in enumerate(slide(seq, args.window_size, args.step)):
                # Drop windows containing non-ACGT bases (N, soft-masked, etc.)
                if set(win) - set('ACGT'):
                    continue
                gstart, gend = window_coords(
                    off, args.window_size, gene['start'], gene['end'], gene['strand'],
                )
                tag = f"{mirna_id}|{gene_name}|w{w_idx}"
                fm.write(f">{tag}\n{mirna_seq}\n")
                ft.write(
                    f">{gene_name}|chr{gene['chrom']}:{gstart}-{gend}:{strand_sym}|w{w_idx}\n"
                    f"{win}\n"
                )
                writer.writerow(
                    [gene['chrom'], gstart, gend, strand_sym, gene_name, mirna_id]
                )
                n_rows += 1

    _log(f"\nWrote {n_rows:,} aligned rows")
    _log(f"  miRNA FASTA      : {mirna_fa}")
    _log(f"  MRE FASTA        : {mre_fa}")
    _log(f"  Conservation TSV : {cons_tsv}")
    if missing_mirnas:
        sample = sorted(missing_mirnas)[:5]
        _log(f"  Skipped {len(missing_mirnas)} unknown miRNA id(s) "
             f"(e.g. {sample}) — not found in miRBase mature.fa")
    if missing_genes:
        sample = sorted(missing_genes)[:5]
        _log(f"  Skipped {len(missing_genes)} unresolved gene(s) "
             f"(e.g. {sample}) — not found in Ensembl GRCh38")
    if skipped_chrom:
        _log(f"  Skipped genes on non-canonical chromosomes: {sorted(skipped_chrom)}")
    if n_rows == 0:
        _log("ERROR: no rows written — check the input file and resolution logs above.")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
