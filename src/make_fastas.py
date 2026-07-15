#!/usr/bin/env python3
"""
Split a v7 TSV into the row-aligned MRE / miRNA FASTA pair that IntaRNA needs.

`intarna_parallel.py` pairs the two FASTAs with `zip`, and `feature_extraction.py`
joins the IntaRNA results back onto the v7 rows by index, so all three files must
stay in the same row order. This script is the only thing that establishes that
order: it writes exactly one record per v7 row, in file order, with the row index
baked into every ID so a misalignment is visible rather than silent.

    python src/make_fastas.py --v7 data/AGO2_eCLIP_Manakov2022_train_v7.tsv \
        --mre-fasta data/train_mre.fa --mirna-fasta data/train_mirna.fa

`gene` (the MRE) is already transcript-oriented on both strands, so it is written
verbatim - do NOT reverse-complement minus-strand rows.
"""

import re
import argparse

import pandas as pd

_ID_SAFE = re.compile(r'[^A-Za-z0-9._-]')


def sanitize(name: str) -> str:
    """IntaRNA takes the ID as a bare command-line token, so keep it boring."""
    return _ID_SAFE.sub('_', str(name)) if name and str(name) != 'nan' else ''


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--v7', required=True)
    p.add_argument('--mre-fasta', required=True)
    p.add_argument('--mirna-fasta', required=True)
    p.add_argument('--mre-col', default='gene')
    p.add_argument('--mirna-col', default='noncodingRNA')
    args = p.parse_args()

    df = pd.read_csv(args.v7, sep='\t')
    for col in (args.mre_col, args.mirna_col):
        if col not in df.columns:
            raise SystemExit(f"ERROR: column '{col}' not in {args.v7}")

    names = (df['noncodingRNA_name'] if 'noncodingRNA_name' in df.columns
             else pd.Series([''] * len(df)))

    with open(args.mre_fasta, 'w') as fm, open(args.mirna_fasta, 'w') as fq:
        for i, (mre, mirna, name) in enumerate(
                zip(df[args.mre_col], df[args.mirna_col], names)):
            mir_id = sanitize(name)
            fm.write(f">mre_{i}\n{str(mre).strip().upper()}\n")
            fq.write(f">{mir_id + '_' if mir_id else 'mir_'}{i}\n"
                     f"{str(mirna).strip().upper()}\n")

    print(f"Wrote {len(df)} records each to {args.mre_fasta} and {args.mirna_fasta}")


if __name__ == '__main__':
    main()
