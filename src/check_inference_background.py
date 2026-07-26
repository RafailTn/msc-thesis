#!/usr/bin/env python3
"""
Guarantee the shuffle-background table covers every miRNA in an inference input.

WHY THIS EXISTS. `feature_extraction.py` fills the four shuffle z-score columns
(`E_z_mirna`, `E_hybrid_z_mirna`, `E_bg_mean_mirna`, `E_bg_sd_mirna`) only when the row's
miRNA is present in `--mirna-background`; a miss defaults them to NaN *silently*. The
model was trained with those columns filled for every miRNA, so a NaN at inference is
off-distribution input, not a neutral "unknown". This script turns that silent miss into
either a loud report or an automatic top-up.

HOW THE TOP-UP STAYS VALID. The whole point of the z-score is that every miRNA is scored
against the *same* panel, so mu/sigma are comparable across miRNAs. A new miRNA must
therefore be run against the identical panel the table was built on - not a freshly
sampled one. `shuffle_background.py` cannot reload a panel (it rebuilds from
target-source+seed each run, and even the same seed drifts if the source file changed),
so this script reuses the panel FASTA that run *saved* (`<background>_panel.fa` by
default) and scores the missing miRNAs against it by calling `shuffle_background`'s own
`background_for_mirna`. That import is deliberate: it keeps "the best site" and the
IntaRNA flags bit-identical to how the existing rows were computed.

WHAT IT DOES NOT NEED. Nothing about the new miRNA's real targets. The background is the
miRNA's energy distribution against shuffled decoy sequence; the real target you want to
predict on contributes its own E through the normal extraction pass, not here.

Usage (report only, exits non-zero if any miRNA is missing):
    python check_inference_background.py \
        --input   data/inference_pairs.tsv --mirna-column noncodingRNA \
        --background data/mirna_background.tsv

Usage (auto-extend the table in place, then continue the pipeline):
    python check_inference_background.py \
        --input   data/inference.fa \
        --background data/mirna_background.tsv \
        --auto-extend
"""

import os
import sys
import csv
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# background_for_mirna / parse_fasta_seqs / normalise live beside this file; importing
# them is what keeps a topped-up row identical to one written by the original run.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shuffle_background import (  # noqa: E402
    background_for_mirna, parse_fasta_seqs, normalise, BACKGROUND_COLS,
)

# Column names that plausibly hold the miRNA sequence, tried in order when the input is a
# table and --mirna-column was not given.
DEFAULT_MIRNA_COLUMNS = ['mirna_sequence', 'noncodingRNA', 'mirna_seq', 'query_seq']


def load_background_sequences(path: str) -> set:
    """The normalised miRNA sequences already covered by the table."""
    covered = set()
    with open(path) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            seq = (row.get('mirna_sequence') or '').upper().replace('T', 'U')
            if seq:
                covered.add(seq)
    return covered


def load_input_mirnas(path: str, column: str = None):
    """Return (unique_sequences, row_count_by_sequence) for the inference input.

    Accepts a FASTA (one miRNA per record) or a TSV/CSV with a miRNA-sequence column.
    Sequences are normalised to the RNA alphabet so the diff against the table cannot
    fail on the T-vs-U mismatch that bit us before.
    """
    if path.endswith(('.fa', '.fasta')):
        seqs = parse_fasta_seqs(path)  # already normalised
        counts = {}
        for s in seqs:
            counts[s] = counts.get(s, 0) + 1
        return set(seqs), counts

    delim = '\t' if path.endswith(('.tsv', '.txt')) else ','
    with open(path) as f:
        reader = csv.DictReader(f, delimiter=delim)
        fields = reader.fieldnames or []
        if column is None:
            column = next((c for c in DEFAULT_MIRNA_COLUMNS if c in fields), None)
            if column is None:
                sys.exit(f"ERROR: none of {DEFAULT_MIRNA_COLUMNS} in {path}; pass "
                         f"--mirna-column. Available: {fields}")
        elif column not in fields:
            sys.exit(f"ERROR: column '{column}' not in {path}. Available: {fields}")

        counts = {}
        for row in reader:
            value = (row.get(column) or '').upper().replace('T', 'U')
            if value:
                counts[value] = counts.get(value, 0) + 1
    return set(counts), counts


def append_background_rows(path: str, new_rows: list):
    """Merge new rows into the table, keeping it sorted by sequence like the builder does.

    Read-modify-write rather than a bare append so the file stays in the exact shape
    `shuffle_background.py` produces (sorted, single header), and a sequence that somehow
    slipped in twice cannot end up duplicated.
    """
    existing = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            existing[row['mirna_sequence']] = row
    for row in new_rows:
        existing[row['mirna_sequence']] = {k: row[k] for k in BACKGROUND_COLS}

    rows = sorted(existing.values(), key=lambda r: r['mirna_sequence'])
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=BACKGROUND_COLS, delimiter='\t')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Check (and optionally extend) shuffle-background coverage for an "
                    "inference input.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--input', required=True,
                        help='inference miRNAs: a FASTA, or a TSV/CSV with a miRNA-'
                             'sequence column (see --mirna-column)')
    parser.add_argument('--mirna-column', default=None,
                        help=f'miRNA-sequence column when --input is a table (tried in '
                             f'order if omitted: {DEFAULT_MIRNA_COLUMNS})')
    parser.add_argument('--background', required=True,
                        help='the mirna_background.tsv to check against / extend')
    parser.add_argument('--panel-fasta', default=None,
                        help='the frozen panel the table was built on (default: '
                             '<background>_panel.fa, as shuffle_background.py writes it). '
                             'Required for --auto-extend.')
    parser.add_argument('--auto-extend', action='store_true',
                        help='score missing miRNAs against the frozen panel and append '
                             'them to --background in place')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--intarna-bin', default='IntaRNA')
    args = parser.parse_args()

    covered = load_background_sequences(args.background)
    input_seqs, counts = load_input_mirnas(args.input, args.mirna_column)
    missing = sorted(input_seqs - covered)
    missing_rows = sum(counts[s] for s in missing)

    print(f"Input miRNAs:        {len(input_seqs)} unique "
          f"({sum(counts.values())} rows)")
    print(f"Covered by table:    {len(input_seqs) - len(missing)}")
    print(f"Missing from table:  {len(missing)} unique ({missing_rows} rows would be NaN)")

    if not missing:
        print("OK: every inference miRNA is covered; z-scores will be filled.")
        return 0

    for s in missing[:10]:
        print(f"  MISSING  {counts[s]:>6d} rows  {s}")
    if len(missing) > 10:
        print(f"  ... and {len(missing) - 10} more")

    if not args.auto_extend:
        print("\nNot extending (no --auto-extend). Re-run with --auto-extend to score "
              "these against the frozen panel, or accept NaN z-scores for these rows.")
        return 1  # non-zero so a pipeline can gate on coverage

    panel_fasta = args.panel_fasta or (os.path.splitext(args.background)[0] + '_panel.fa')
    if not os.path.exists(panel_fasta):
        sys.exit(f"ERROR: panel FASTA not found: {panel_fasta}\n"
                 f"--auto-extend must score against the SAME panel the table was built "
                 f"on. Pass --panel-fasta explicitly, or rebuild the table with "
                 f"shuffle_background.py (which saves the panel beside --output).")

    panel = parse_fasta_seqs(panel_fasta)
    print(f"\n--- Extending: {len(missing)} miRNAs x {len(panel)} panel targets "
          f"(panel {panel_fasta}) ---")

    new_rows = []
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {
            pool.submit(background_for_mirna, seq, f"mir_ext{i}", panel,
                        panel_fasta, args.intarna_bin, args.timeout): seq
            for i, seq in enumerate(missing, 1)
        }
        for done, fut in enumerate(as_completed(futures), 1):
            new_rows.append(fut.result())
            if done % 25 == 0 or done == len(missing):
                print(f"\r  {done}/{len(missing)}", end='', file=sys.stderr, flush=True)
    print(file=sys.stderr)

    append_background_rows(args.background, new_rows)

    usable = [r for r in new_rows if r['n_bg'] >= 2]
    unusable = [r for r in new_rows if r['n_bg'] < 2]
    print(f"--- Appended {len(new_rows)} rows to {args.background} ---")
    print(f"  usable (real z-scores): {len(usable)}")
    if unusable:
        print(f"  still NaN (no panel interaction, n_bg<2): {len(unusable)} - these rows "
              f"will remain NaN even after extension")
    print("Re-run feature_extraction.py --mirna-background to fill the z-columns.")
    return 0 if not unusable else 1


if __name__ == '__main__':
    sys.exit(main())
