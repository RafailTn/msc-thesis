#!/usr/bin/env python3
"""
Per-miRNA background binding energies against dinucleotide-shuffled targets.

WHY. The raw IntaRNA energies conflate two things: "this is a good site" and "this
miRNA is GC-rich and binds everything". The second term is large - across the 1,227
miRNAs of the Manakov train split, the mean energy against shuffled targets spans
-10.05 to -1.14 kcal/mol. A duplex at E = -12 is seven standard deviations beyond
anything an AU-rich miRNA manages against random sequence, and utterly typical for a
GC-rich one. A z-score against a fixed background of shuffled targets removes that term:

    E_z_mirna = (E - mu_mirna) / sigma_mirna

where mu/sigma are this miRNA's energy distribution against a panel of shuffled 50-mers.
What survives is "how much better is this target than generic sequence of the same
composition, in units of this miRNA's own spread".

WHAT THIS DOES AND DOES NOT BUY, given how miRBench builds negatives. Binding sites are
clustered at >90% similarity and, for each miRNA *family*, negatives are sampled from
non-overlapping clusters that the family does not bind (targets must also be Levenshtein
>= 3 from positive counterparts of the same gene). The loop is over miRNAs and the
sampling is over MREs; there are no decoy miRNAs, and complementarity is never a
rejection criterion. Measured consequences on the train split:

  * The miRNA side is balanced by construction - 1,167 of 1,227 miRNAs (95.1%) carry both
    labels, and per-family P(positive) is 0.500 (10th-90th pct 0.484-0.516). Hence
    `E_bg_mean_mirna` on its own is near chance (0.524 AUROC): this table adds no miRNA
    prior, which is exactly what we want from a normaliser.
  * Because negatives are real binding sites of *other* miRNAs, 606,066 of 1,110,112
    targets (54.6%) carry both labels, covering 75.7% of rows. Such same-target pairs
    differ only in the miRNA.

So mu/sigma are constant within one miRNA's rows, and `E_z_mirna` is there an affine
transform of `E` - it adds nothing to ranking targets for a fixed miRNA, which is the
very axis the sampling procedure creates. Its value is elsewhere: across same-target
pairs, where the miRNA is the only thing that differs, and in pooling, where it makes a
single tree split mean the same thing for a GC-rich and an AU-rich miRNA. The measured
gain is a pooled one (miRNA-identity eta^2 on E drops 0.183 -> 0.074 against a 0.057 null;
univariate AUROC 0.705 -> 0.729), consistent with comparability being the mechanism.

WHY THE miRNA SIDE ONLY. Two reasons, both about the target side being the wrong unit.

  * Cost: 1,227 unique miRNAs against 1,110,112 unique targets, so a panel of N=100 is
    ~123k IntaRNA calls one way and ~111M the other.
  * Statistics: that is ~2,034 rows per miRNA against ~2.25 rows per target. A per-target
    mu/sigma would be estimated from roughly two rows - close to a per-row transform, with
    nothing to generalise from, and worthless under a cold-target split.

The symmetric feature is not "wrong", it is just unaffordable and would be badly estimated.
It has its own blind axis anyway, mirroring the one above: constant within a same-target
pair, where `E_z_mirna` is precisely what varies.

THE PANEL. A single panel of dinucleotide-shuffled 50-mers, shared by every miRNA.
Shared, not per-miRNA, so that mu and sigma are measured against identical background
sequence and are therefore comparable *across* miRNAs - which is the whole point.
The panel members are shuffles of real MREs sampled from the dataset, so their
length and dinucleotide composition match the real target population. A
dinucleotide-preserving shuffle (Altschul-Erikson) is used rather than a mononucleotide
one because nearest-neighbour stacking energies are a function of dinucleotides:
a mononucleotide shuffle would leave a composition difference that IntaRNA reads as
a real energy difference.

MATCHING THE FOREGROUND. The IntaRNA flags and the best-site selection here are
identical to the production run (intarna_parallel.py's MFE mode, then
best_intarna.select_best_interaction). If they diverge, the z-score compares two
different quantities and is silently meaningless - so the flags are duplicated from
intarna_parallel.run_intarna deliberately, and `--intarna-flag-check` prints them for
diffing against that function.

Usage:
    python shuffle_background.py \
        --mirna-fasta mirna.fa \
        --target-source data/AGO2_eCLIP_Manakov2022_train_v7.tsv \
        --output data/mirna_background.tsv \
        --panel-size 100 --threads 8

The output is consumed by `feature_extraction.py --mirna-background`.
"""

import os
import sys
import csv
import random
import argparse
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import numpy as np

# best_intarna lives beside this file; importing it rather than reimplementing the
# scoring is what keeps the background's notion of "the best site" identical to the
# foreground's.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from best_intarna import IntaRNAInteraction, select_best_interaction  # noqa: E402


# The MFE-mode column set and flags from intarna_parallel.run_intarna. Duplicated on
# purpose - see MATCHING THE FOREGROUND above.
CSV_COLS = ("id1,start1,end1,id2,start2,end2,subseqDP,hybridDP,"
            "ED1,ED2,E,E_hybrid,E_hybridNorm,E_norm")

INTARNA_FLAGS = [
    '--mode=M', '--tAccW=0', '--tAccL=0', '--qAcc=N', '--outMode=C',
    '--outCsvCols=' + CSV_COLS, '-n', '10', '--outDeltaE=100.0',
    '--outOverlap=B', '--noSeed', '--outMaxE=100', '--outNoLP', '--model=S',
]

# Columns of the background table. Keyed on the miRNA sequence, not its name: the
# sequence is what feature_extraction has in hand, and two names can share a sequence.
BACKGROUND_COLS = [
    'mirna_sequence', 'mirna_id', 'n_bg',
    'E_bg_mean', 'E_bg_sd', 'E_hybrid_bg_mean', 'E_hybrid_bg_sd',
]


# ============================================================================
# DINUCLEOTIDE SHUFFLE
# ============================================================================

def dinuc_shuffle(seq: str, rng: random.Random) -> str:
    """Altschul-Erikson dinucleotide-preserving shuffle.

    Returns a permutation of `seq` with identical dinucleotide counts (and therefore
    identical mononucleotide counts). The construction is the standard one: treat the
    sequence as an Eulerian path in the de Bruijn multigraph over nucleotides, draw a
    random arborescence rooted at the final vertex to fix each vertex's last outgoing
    edge, shuffle the remaining edges freely, and re-walk the path. The first and last
    characters are preserved by construction.
    """
    if len(seq) < 4:
        return seq

    last = seq[-1]
    edges: Dict[str, List[str]] = defaultdict(list)
    for a, b in zip(seq, seq[1:]):
        edges[a].append(b)
    verts = list(edges)

    # Draw last-edges until they form an arborescence into `last`, i.e. until every
    # vertex reaches `last` by following them. Anything else strands part of the graph
    # and the walk below would terminate early.
    for _attempt in range(1000):
        last_edge = {v: rng.choice(edges[v]) for v in verts if v != last}
        if all(_reaches(v, last, last_edge) for v in last_edge):
            break
    else:
        return seq  # pathological; leave unshuffled rather than emit a truncated seq

    out_edges = {}
    for v in verts:
        rest = list(edges[v])
        if v in last_edge:
            rest.remove(last_edge[v])
            rng.shuffle(rest)
            rest.append(last_edge[v])
        else:
            rng.shuffle(rest)
        out_edges[v] = rest

    result = [seq[0]]
    cursor: Dict[str, int] = defaultdict(int)
    current = seq[0]
    for _ in range(len(seq) - 1):
        nxt = out_edges[current][cursor[current]]
        cursor[current] += 1
        result.append(nxt)
        current = nxt
    return ''.join(result)


def _reaches(start: str, target: str, last_edge: Dict[str, str]) -> bool:
    seen = set()
    node = start
    while node != target:
        if node in seen or node not in last_edge:
            return False
        seen.add(node)
        node = last_edge[node]
    return True


# ============================================================================
# IO
# ============================================================================

def parse_fasta_seqs(path: str) -> List[str]:
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                if cur:
                    seqs.append(''.join(cur))
                cur = []
            else:
                cur.append(line.strip())
    if cur:
        seqs.append(''.join(cur))
    return [normalise(s) for s in seqs if s]


def normalise(seq: str) -> str:
    """Upper-case RNA alphabet. Must match feature_extraction.parse_fasta exactly,
    since the background table is joined on the sequence string."""
    return seq.upper().replace('T', 'U')


def load_target_pool(path: str, column: str = 'gene') -> List[str]:
    """Real target sequences to shuffle into the panel. Accepts a FASTA or a v7 TSV."""
    if path.endswith(('.fa', '.fasta')):
        return parse_fasta_seqs(path)

    seqs = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        if column not in (reader.fieldnames or []):
            sys.exit(f"ERROR: column '{column}' not in {path}. "
                     f"Available: {reader.fieldnames}")
        for row in reader:
            value = row.get(column)
            if value:
                seqs.append(normalise(value))
    return seqs


def build_panel(pool: List[str], size: int, seed: int) -> List[str]:
    """Sample `size` real targets and dinucleotide-shuffle each once.

    Sampling without replacement where possible, so the panel is not dominated by one
    sequence's composition. The RNG is seeded so the panel - and therefore every
    z-score computed against it - is reproducible.
    """
    if not pool:
        sys.exit("ERROR: target pool is empty; cannot build a background panel.")

    rng = random.Random(seed)
    picked = (rng.sample(pool, size) if len(pool) >= size
              else [rng.choice(pool) for _ in range(size)])
    return [dinuc_shuffle(s, rng) for s in picked]


def write_panel_fasta(panel: List[str], path: str):
    with open(path, 'w') as f:
        for i, seq in enumerate(panel, 1):
            f.write(f">bg{i}\n{seq}\n")


# ============================================================================
# INTARNA
# ============================================================================

def run_against_panel(mirna_seq: str, mirna_id: str, panel_fasta: str,
                      intarna_bin: str, timeout: int) -> Dict[str, List[IntaRNAInteraction]]:
    """One IntaRNA call for this miRNA against the whole panel (all-vs-all).

    IntaRNA pairs every target in a multi-FASTA with every query, so the panel costs one
    subprocess per miRNA rather than one per (miRNA, panel member) - the difference
    between ~1.2k and ~123k process spawns.
    """
    cmd = [intarna_bin, '--target', panel_fasta, '--query', mirna_seq,
           '--qId=' + mirna_id] + INTARNA_FLAGS
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {}

    by_target: Dict[str, List[IntaRNAInteraction]] = defaultdict(list)
    lines = [l for l in proc.stdout.splitlines() if l.strip() and not l.startswith('#')]
    if not lines:
        return {}

    for row in csv.DictReader(lines, delimiter=';'):
        if row.get('start1') in ('NA', '', None):
            continue
        try:
            by_target[row['id1']].append(IntaRNAInteraction(
                target_id=row['id1'], query_id=row['id2'],
                start_target=int(row['start1']), end_target=int(row['end1']),
                start_query=int(row['start2']), end_query=int(row['end2']),
                subseq_dp=row['subseqDP'], hybrid_dp=row['hybridDP'],
                energy_total=float(row['E']), energy_hybrid=float(row['E_hybrid']),
                energy_ED1=float(row['ED1']), energy_ED2=float(row['ED2']),
                energy_norm=float(row['E_norm']),
                energy_hybrid_norm=float(row['E_hybridNorm']),
            ))
        except (ValueError, KeyError):
            continue
    return by_target


def background_for_mirna(mirna_seq: str, mirna_id: str, panel: List[str],
                         panel_fasta: str, intarna_bin: str,
                         timeout: int) -> Dict[str, object]:
    """mu and sigma of this miRNA's binding energy across the panel."""
    by_target = run_against_panel(mirna_seq, mirna_id, panel_fasta, intarna_bin, timeout)

    energies, hybrids = [], []
    for idx, panel_seq in enumerate(panel, 1):
        interactions = by_target.get(f"bg{idx}")
        if not interactions:
            # No interaction found at all. Skipped rather than imputed: a floor value
            # would be arbitrary and would distort sigma, and `n_bg` records the loss.
            continue
        # Selected exactly as in production - by priority_score, not by energy. If this
        # diverges from best_intarna.main's call the z-score compares two different
        # quantities, so the arguments are kept in step with it deliberately.
        best = select_best_interaction(interactions, panel_seq, mirna_seq,
                                       ensemble=False, seed_last_pos=8)
        if best is None:
            continue
        energies.append(best.energy_total)
        hybrids.append(best.energy_hybrid)

    if len(energies) < 2:
        return {'mirna_sequence': mirna_seq, 'mirna_id': mirna_id, 'n_bg': len(energies),
                'E_bg_mean': float('nan'), 'E_bg_sd': float('nan'),
                'E_hybrid_bg_mean': float('nan'), 'E_hybrid_bg_sd': float('nan')}

    return {
        'mirna_sequence': mirna_seq,
        'mirna_id': mirna_id,
        'n_bg': len(energies),
        'E_bg_mean': round(float(np.mean(energies)), 6),
        'E_bg_sd': round(float(np.std(energies, ddof=1)), 6),
        'E_hybrid_bg_mean': round(float(np.mean(hybrids)), 6),
        'E_hybrid_bg_sd': round(float(np.std(hybrids, ddof=1)), 6),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build a per-miRNA background energy table against "
                    "dinucleotide-shuffled targets.")
    parser.add_argument('--mirna-fasta', required=True,
                        help='miRNA FASTA. Only the unique sequences are run.')
    parser.add_argument('--target-source', required=True,
                        help='v7 TSV or FASTA supplying real targets to shuffle into '
                             'the panel')
    parser.add_argument('--target-column', default='gene',
                        help='target sequence column when --target-source is a TSV')
    parser.add_argument('--output', required=True)
    parser.add_argument('--panel-size', type=int, default=100,
                        help='shuffled targets per miRNA (default 100)')
    parser.add_argument('--seed', type=int, default=0,
                        help='RNG seed for the panel; fixes the background, and so the '
                             'z-scores, across runs')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--intarna-bin', default='IntaRNA')
    parser.add_argument('--panel-fasta', default=None,
                        help='where to write the panel (default: alongside --output). '
                             'Kept so a background table can be audited or extended.')
    parser.add_argument('--intarna-flag-check', action='store_true',
                        help='print the IntaRNA flags used and exit, for diffing '
                             'against intarna_parallel.run_intarna')
    args = parser.parse_args()

    if args.intarna_flag_check:
        print(' '.join(INTARNA_FLAGS))
        return 0

    print("--- Building panel ---")
    pool = load_target_pool(args.target_source, args.target_column)
    print(f"  target pool: {len(pool)} sequences from {args.target_source}")
    panel = build_panel(pool, args.panel_size, args.seed)

    panel_fasta = args.panel_fasta or (os.path.splitext(args.output)[0] + '_panel.fa')
    write_panel_fasta(panel, panel_fasta)
    lengths = {len(s) for s in panel}
    print(f"  panel: {len(panel)} dinucleotide-shuffled targets, lengths {sorted(lengths)}")
    print(f"  written to {panel_fasta} (seed {args.seed})")

    mirnas = sorted(set(parse_fasta_seqs(args.mirna_fasta)))
    print(f"\n--- {len(mirnas)} unique miRNA sequences x {len(panel)} panel targets ---")
    print(f"  {len(mirnas)} IntaRNA calls on {args.threads} threads")

    rows = []
    with ThreadPoolExecutor(max_workers=args.threads) as pool_exec:
        futures = {
            pool_exec.submit(background_for_mirna, seq, f"mir{i}", panel,
                             panel_fasta, args.intarna_bin, args.timeout): seq
            for i, seq in enumerate(mirnas, 1)
        }
        for done, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if done % 50 == 0 or done == len(mirnas):
                print(f"\r  {done}/{len(mirnas)}", end='', file=sys.stderr, flush=True)
    print(file=sys.stderr)

    rows.sort(key=lambda r: r['mirna_sequence'])
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=BACKGROUND_COLS, delimiter='\t')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    complete = [r for r in rows if r['n_bg'] == len(panel)]
    degraded = [r for r in rows if 2 <= r['n_bg'] < len(panel)]
    failed = [r for r in rows if r['n_bg'] < 2]
    print(f"\n--- Wrote {args.output} ---")
    print(f"  full panel:   {len(complete)}")
    print(f"  partial:      {len(degraded)}  (some panel targets found no interaction)")
    print(f"  unusable:     {len(failed)}  (NaN background; z-scores will be NaN)")
    if complete:
        means = [r['E_bg_mean'] for r in complete]
        sds = [r['E_bg_sd'] for r in complete]
        print(f"  E_bg_mean across miRNAs: {min(means):.2f} .. {max(means):.2f}")
        print(f"  E_bg_sd   across miRNAs: {min(sds):.2f} .. {max(sds):.2f}")
        print("  A wide E_bg_mean range is the confound this table exists to remove: "
              "it is how much of E is set by the miRNA alone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
