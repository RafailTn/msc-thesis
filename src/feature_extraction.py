#!/usr/bin/env python3
"""
Feature extraction for miRNA-MRE interactions: the single source of truth.

This module merges what used to be two scripts run back to back:

  * `classify_and_filter_sites_intarna.py` - IntaRNA energies, duplex statistics,
    conservation features, binding-type classification.
  * `seq_features_ml.py`                   - region-specific sequence features.

They are merged because they had drifted: `seq_features_ml.py` still computed
"stacking energies" by sliding a dinucleotide window over each single strand,
which made them a pure function of the miRNA sequence (identical across every
target of the same miRNA) and carried no information about the binding site at
all. This module computes them from the actual IntaRNA duplex instead - see
`duplex_energy_steps`. Keeping one module means the training features and the
inference features cannot diverge again.

ONE OUTPUT MODE. The extractor writes DEFAULT_FEATURES - the featurewiz selection plus
whatever new candidates are under test - and computes only those. The old
`--all-features` superset mode is gone: featurewiz has already judged the superset, and
from here on selection runs against DEFAULT_FEATURES instead, so the ~120 rejected
features cost inference time and feed nothing. `--features-file` overrides the list, and
`--list-features` prints the full vocabulary, so any rejected feature is still reachable
by name (see the DEFAULT_FEATURES comment for why that escape hatch matters).

Computation is GATED to the requested list: whole conservation windows, composition
regions and the duplex-energy block are skipped when nothing asks for them. The gating is
derived from the feature names rather than hand-maintained, and only skips units computed
independently of one another, so a retained feature runs byte-identical code either way.
`--verify-gating` proves it on a sample; `--no-feature-gating` computes everything as a
reference. This is what keeps the single-source-of-truth property that merging the two
predecessor scripts bought: a feature cannot mean one thing during selection and another
during training.

Usage:
    # the normal path - training, selection and inference all use this
    python feature_extraction.py --intarna best.tsv --mre-fasta mre.fa \
        --mirna-fasta mirna.fa --v7 data/..._train_v7.tsv \
        --mirna-background data/mirna_background.tsv \
        --output train.csv

    # reconsider a feature featurewiz rejected, or rebuild the old superset
    python feature_extraction.py --list-features > superset.txt
    python feature_extraction.py ... --features-file superset.txt
"""

import os
import re
import sys
import csv
import json
import argparse
import warnings
import multiprocessing as mp
from collections import Counter
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

try:
    import pyBigWig
    HAS_PYBIGWIG = True
except ImportError:
    HAS_PYBIGWIG = False

try:
    import RNA  # ViennaRNA python bindings; ships with the IntaRNA conda package
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "ViennaRNA python bindings (module 'RNA') are required for the duplex "
        "stacking energies. They come with the intarna conda package; if you are "
        "outside the pixi env, install with `conda install -c bioconda viennarna`."
    ) from _e


# ============================================================================
# SELECTED FEATURES
# ============================================================================
#
# The output of `feature_selection.py` - the intersection of the featurewiz selections
# across the 5 folds. Current as of the rerun that followed the duplex-energy + phyloP
# rewrite (commit "added selected features after rerun"), so the names here mean what
# this module computes today. Mirrored in
# feature_selection_featurewiz/selected_features.json.
#
# This list is the RECORD of what feature selection chose, not the list the extractor
# writes - that is DEFAULT_FEATURES, which adds the candidates still under test. Keeping
# them separate is what lets the training-side A/B hold the selection fixed, and what
# tells the next featurewiz run which features it has already judged.
#
# To refresh: run featurewiz against a DEFAULT_FEATURES extraction, paste the survivors
# here, and empty NEW_CANDIDATE_FEATURES of anything that was kept or dropped.

SELECTED_FEATURES = [
  "Eall",
  "Eall1",
  "binding_type_5mer.mismatch.3prime",
  "binding_type_6mer.mismatch.3prime",
  "central_max_consecutive_drop",
  "consecutive_matches_minus_seed",
  "conservation_range",
  "conservation_variance",
  "effective_3prime_matches",
  "five_prime_flank_conservation_mean",
  "gu_wobbles_in_seed_2_8_pos",
  "mirna_3p_RYR_freq",
  "mirna_3p_RYY_freq",
  "mirna_3p_YRY_freq",
  "mirna_3p_YYR_freq",
  "mirna_3p_YYY_freq",
  "mirna_3p_energy_asymmetry",
  "mirna_3p_energy_gradient",
  "mirna_3p_energy_mean",
  "mirna_3p_energy_std",
  "mirna_3p_ggg_count",
  "mirna_seed_RRR_freq",
  "mirna_seed_RRY_freq",
  "mirna_seed_RYR_freq",
  "mirna_seed_energy_asymmetry",
  "mirna_seed_energy_gradient",
  "mirna_seed_energy_mean",
  "mre_5p_uuu_count",
  "priority_score",
  "seed_au_content",
  "seed_conservation_mean",
  "seed_conservation_min",
  "seed_roughness",
  "three_prime_flank_conservation_mean",
  "total_gu_wobbles",
  "total_matches"
]


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_VALUE = 0.0

# miRNA seed = positions 2-8 (1-based, inclusive); the "3p" region is 9 -> 3' end.
SEED_FIRST_POS = 2
SEED_LAST_POS = 8

MRE_LEN = 50

# The conservation track read out of the v7 TSV.
#
# phyloP replaces phastCons: phastCons saturates (it is a posterior probability
# and piles up at 0.0 and 1.0), while phyloP is a signed log-p-value that keeps
# resolution in both directions - positive = conserved, negative = accelerated.
#
# The catch is that phyloP is signed and unbounded where phastCons was in [0, 1],
# which breaks two of the shape statistics:
#   * entropy needs a fixed histogram range to stay comparable across sites.
#     PHYLOP_HIST_RANGE below is that range; values outside it are clipped.
#   * Gini measures the inequality of a *non-negative* quantity. It is undefined
#     on signed data, so it is dropped from the conservation shape features (it
#     survives for trinucleotide *counts*, which are non-negative by construction).
DEFAULT_CONS_COL = 'gene_phyloP'
PHYLOP_HIST_RANGE = (-5.0, 5.0)
PHASTCONS_HIST_RANGE = (0.0, 1.0)

# v7 columns carried through to the output so the feature CSV needs no separate
# join before feature selection / training.
V7_PASSTHROUGH = [
    'gene', 'noncodingRNA', 'noncodingRNA_name', 'noncodingRNA_fam', 'feature',
    'chr', 'start', 'end', 'strand', 'gene_cluster_ID',
]


# ============================================================================
# IO
# ============================================================================

def parse_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        seq = ""
        for line in f:
            if line.startswith('>'):
                if seq:
                    sequences.append(seq.upper().replace('T', 'U'))
                seq = ""
            else:
                seq += line.strip()
        if seq:
            sequences.append(seq.upper().replace('T', 'U'))
    return sequences


def parse_intarna_results(file_path):
    with open(file_path, 'r') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def get_bigwig_vector(bw_path, chrom, start, end, strand):
    """Read a conservation vector straight from a bigwig, in transcript order.

    Unlike the v7 columns this is genomic-order-corrected here (the reversal for
    minus strand happens below), so its output needs no further reorientation.
    """
    if not HAS_PYBIGWIG:
        return None
    try:
        bw = pyBigWig.open(bw_path)
        scores = bw.values(chrom, int(start) - 1, int(end))
        # For phyloP, 0.0 is the neutral rate - the right fill for "no data".
        scores = np.nan_to_num(np.array(scores, dtype=float), nan=0.0)
        if strand == '-':
            scores = scores[::-1]
        bw.close()
        return scores.tolist()
    except (RuntimeError, ValueError):
        return None


def load_v7(file_path, cons_col: str, bigwig_path: Optional[str],
            reverse_minus_strand: bool = True,
            allow_missing_conservation: bool = False) -> pd.DataFrame:
    """Load the v7 TSV and attach a transcript-oriented conservation vector.

    ORIENTATION. `gene` (the MRE sequence) is transcript-oriented, but the v7
    conservation columns (`gene_phyloP`, `gene_phastCons`) are written in *genomic*
    order. On the plus strand these agree; on the minus strand they are reversed
    relative to each other, so `gene[j]` lines up with `cons[49 - j]`, not `cons[j]`.

    Every conservation feature here is positional - the seed scores are gathered by
    indexing `scores[i]` at MRE positions derived from the duplex - so the vector is
    reversed on minus-strand rows. The previous pipeline did not do this, which
    silently scrambled the conservation features of roughly half the rows.
    (Verified: corr(phyloP[j], phastCons[j]) = +0.50 on both strands, while
    corr(phyloP[49-j], phastCons[j]) = -0.04, so the two tracks share one order.)

    `--cons-no-reverse` opts out, and is correct only for a column already in
    transcript orientation.
    """
    df = pd.read_csv(file_path, sep='\t')

    if 'chr' in df.columns:
        df['chr'] = "chr" + df['chr'].astype(str).str.removeprefix('chr')
        df['chr'] = df['chr'].replace({'chrMT': 'chrM'})

    has_coords = all(c in df.columns for c in ['chr', 'start', 'end', 'strand'])

    if bigwig_path and HAS_PYBIGWIG and has_coords:
        # Already transcript-oriented by get_bigwig_vector.
        df['conservation_vector'] = df.apply(
            lambda r: get_bigwig_vector(bigwig_path, r['chr'], r['start'], r['end'],
                                        r['strand']), axis=1)
        return df

    if cons_col not in df.columns:
        if not allow_missing_conservation:
            raise SystemExit(
                f"ERROR: conservation column '{cons_col}' not in {file_path}. "
                f"Available: "
                f"{[c for c in df.columns if 'phy' in c.lower() or 'cons' in c.lower()]}"
            )
        # Inference without a conservation track: every conservation feature falls back
        # to DEFAULT_VALUE. Never acceptable for training, hence the opt-in flag.
        print(f"  WARNING: no '{cons_col}' column - all conservation features will be "
              f"{DEFAULT_VALUE}.")
        df['conservation_vector'] = [[] for _ in range(len(df))]
        return df

    vecs = df[cons_col].map(parse_conservation_scores)
    if reverse_minus_strand and 'strand' in df.columns:
        minus = (df['strand'] == '-')
        vecs = [v[::-1] if (m and v) else v for v, m in zip(vecs, minus)]
        print(f"  Reversed conservation vector on {int(minus.sum())} minus-strand rows "
              f"(genomic -> transcript order)")
    df['conservation_vector'] = list(vecs)
    return df


def parse_conservation_scores(cons_vector) -> List[float]:
    if cons_vector is None:
        return []
    if isinstance(cons_vector, (list, np.ndarray)):
        return [0.0 if (x is None or np.isnan(x)) else float(x) for x in cons_vector]
    if isinstance(cons_vector, str) and cons_vector not in ('', 'nan', 'None', 'NaN'):
        scores = []
        for s in cons_vector.strip().strip('[]').split(','):
            s = s.strip()
            if s:
                try:
                    scores.append(float(s))
                except ValueError:
                    scores.append(0.0)
        return scores
    return []


# ============================================================================
# DUPLEX VECTOR / POSITION TRACKING
# ============================================================================

def create_duplex_vectors(target_struct: str, query_struct: str) -> str:
    loop = []
    i, j = 0, 0
    match_started = False
    while i < len(target_struct) or j < len(query_struct):
        target_char = target_struct[i] if i < len(target_struct) else '\0'
        query_char = query_struct[j] if j < len(query_struct) else '\0'

        if target_char == '(' and query_char == ')':
            loop.append('1'); match_started = True; i += 1; j += 1
        elif target_char == '.' and query_char == '.':
            loop.append('2' if match_started else 'D'); i += 1; j += 1
        elif target_char == '.' and query_char == ')':
            loop.append('3'); i += 1
        elif target_char == '(' and query_char == '.':
            loop.append('4'); j += 1
        elif target_char == '\0' and query_char == '.':
            loop.append('d'); j += 1
        elif query_char == '\0' and target_char == '.':
            loop.append('e'); i += 1
        else:
            break
    return "".join(loop)


def get_mirna_position_map(total_vec):
    position_map = {}
    mirna_pos = 0
    for vec_idx, char in enumerate(total_vec):
        if char in '124Dd':
            position_map[vec_idx] = mirna_pos
            mirna_pos += 1
        elif char in '3e':
            position_map[vec_idx] = None
        else:
            position_map[vec_idx] = mirna_pos
    return position_map


def get_mre_position_map(total_vec):
    position_map = {}
    mre_pos = 0
    for vec_idx, char in enumerate(total_vec):
        if char in '123De':
            position_map[vec_idx] = mre_pos
            mre_pos += 1
        elif char in '4d':
            position_map[vec_idx] = None
        else:
            position_map[vec_idx] = mre_pos
    return position_map


def get_vector_indices_for_mirna_range(total_vec, mirna_start, mirna_end, mirna_binding_start):
    position_map = get_mirna_position_map(total_vec)
    indices = []
    for vec_idx, mirna_pos_in_binding in position_map.items():
        if mirna_pos_in_binding is not None:
            mirna_pos_in_full = mirna_pos_in_binding + mirna_binding_start
            if mirna_start <= mirna_pos_in_full < mirna_end:
                indices.append(vec_idx)
    return indices


def count_consecutive_matches_in_mirna_region(total_vec, mirna_start, mirna_end, mirna_binding_start):
    indices = get_vector_indices_for_mirna_range(total_vec, mirna_start, mirna_end, mirna_binding_start)
    if not indices:
        return 0
    max_c, curr_c = 0, 0
    for char in (total_vec[i] for i in indices):
        if char == '1':
            curr_c += 1
            max_c = max(max_c, curr_c)
        else:
            curr_c = 0
    return max_c


def count_char_in_mirna_region(total_vec, mirna_start, mirna_end, mirna_binding_start, target_char):
    indices = get_vector_indices_for_mirna_range(total_vec, mirna_start, mirna_end, mirna_binding_start)
    return [total_vec[i] for i in indices].count(target_char) if indices else 0


def get_mirna_region_vector(total_vec, mirna_start, mirna_end, mirna_binding_start):
    indices = get_vector_indices_for_mirna_range(total_vec, mirna_start, mirna_end, mirna_binding_start)
    return ''.join(total_vec[i] for i in indices) if indices else ""


def get_paired_bases_with_positions(total_vec, mre_binding_seq, mirna_binding_seq,
                                    mre_binding_start, mirna_binding_start):
    paired_mre, paired_mir, mirna_positions_in_full = [], [], []
    mre_ptr = mir_ptr = 0
    for char in total_vec:
        if mre_ptr >= len(mre_binding_seq) or mir_ptr >= len(mirna_binding_seq):
            break
        if char == '1':
            paired_mre.append(mre_binding_seq[mre_ptr])
            paired_mir.append(mirna_binding_seq[mir_ptr])
            mirna_positions_in_full.append(mir_ptr + mirna_binding_start)
        if char in '123De':
            mre_ptr += 1
        if char in '124Dd':
            mir_ptr += 1
    return "".join(paired_mre), "".join(paired_mir), mirna_positions_in_full


def get_mre_positions_for_seed(site_data, mirna_binding_start):
    total_vec = site_data['total_vec']
    mre_binding_end = site_data['mre_coord_end'] - 1

    mirna_pos_map = get_mirna_position_map(total_vec)
    mre_pos_map = get_mre_position_map(total_vec)

    seed_mre_positions = []
    for vec_idx in range(len(total_vec)):
        mirna_pos_in_binding = mirna_pos_map.get(vec_idx)
        mre_pos_in_binding = mre_pos_map.get(vec_idx)
        if mirna_pos_in_binding is not None and mre_pos_in_binding is not None:
            mirna_pos_in_full = mirna_pos_in_binding + mirna_binding_start + 1
            if 1 <= mirna_pos_in_full <= 7:
                original_mre_idx = mre_binding_end - mre_pos_in_binding
                if 0 <= original_mre_idx < MRE_LEN:
                    seed_mre_positions.append(original_mre_idx)
    return seed_mre_positions


# ============================================================================
# CONSERVATION FEATURES
# ============================================================================

# No `gini`: it measures inequality of a non-negative quantity and is undefined on
# signed phyloP. See the DEFAULT_CONS_COL comment.
_CONS_SHAPE_KEYS = [
    'skewness', 'kurtosis', 'entropy', 'slope', 'roughness',
    'max_consecutive_drop', 'max_consecutive_rise',
]

_CONS_SUMMARY_KEYS = [
    'seed_conservation_mean', 'seed_conservation_median', 'seed_conservation_max',
    'seed_conservation_min', 'seed_conservation_std',
    'five_prime_flank_conservation_mean', 'three_prime_flank_conservation_mean',
    'full_site_conservation_mean', 'conservation_contrast', 'flank_conservation_diff',
    'conservation_variance', 'conservation_range',
]

_CONS_WINDOWS = ['seed', 'upstream', 'central', 'downstream']


def conservation_feature_names() -> List[str]:
    return list(_CONS_SUMMARY_KEYS) + [
        f'{w}_{k}' for w in _CONS_WINDOWS for k in _CONS_SHAPE_KEYS
    ]


def _compute_vector_shape_features(scores, prefix: str, hist_range, want=None) -> dict:
    n = len(scores)
    if n < 2:
        return {f"{prefix}_{k}": DEFAULT_VALUE for k in _CONS_SHAPE_KEYS}

    def wanted(key: str) -> bool:
        return want is None or f"{prefix}_{key}" in want

    arr = np.asarray(scores, dtype=float)
    mean, std = float(np.mean(arr)), float(np.std(arr))
    out = {}

    if wanted('skewness') or wanted('kurtosis'):
        if std > 0:
            out[f"{prefix}_skewness"] = round(float(np.mean(((arr - mean) / std) ** 3)), 4)
            out[f"{prefix}_kurtosis"] = round(float(np.mean(((arr - mean) / std) ** 4) - 3.0), 4)
        else:
            out[f"{prefix}_skewness"] = DEFAULT_VALUE
            out[f"{prefix}_kurtosis"] = DEFAULT_VALUE

    # Fixed histogram range keeps entropy comparable across sites; phyloP tails are
    # clipped into the outermost bins rather than dropped. The histogram is the single
    # most expensive operation in this function and feeds nothing but entropy, so it is
    # the first thing worth gating.
    if wanted('entropy'):
        counts, _ = np.histogram(np.clip(arr, hist_range[0], hist_range[1]),
                                 bins=10, range=hist_range)
        s = counts.sum()
        if s > 0:
            probs = counts / s
            probs = probs[probs > 0]
            out[f"{prefix}_entropy"] = round(float(-np.sum(probs * np.log2(probs))), 4)
        else:
            out[f"{prefix}_entropy"] = DEFAULT_VALUE

    if not (wanted('slope') or wanted('roughness')
            or wanted('max_consecutive_drop') or wanted('max_consecutive_rise')):
        return out

    positions = np.arange(1, n + 1, dtype=float)
    pos_mean = positions.mean()
    den = float(np.sum((positions - pos_mean) ** 2))
    num = float(np.sum((positions - pos_mean) * (arr - mean)))
    out[f"{prefix}_slope"] = round(num / den if den > 0 else DEFAULT_VALUE, 6)

    diffs = np.diff(arr)
    out[f"{prefix}_roughness"] = round(float(np.mean(np.abs(diffs))), 4)
    out[f"{prefix}_max_consecutive_drop"] = round(float(np.max(-diffs)), 4)
    out[f"{prefix}_max_consecutive_rise"] = round(float(np.max(diffs)), 4)

    return out


def extract_conservation_features(site_data, mirna_binding_start, flank_size=10,
                                  hist_range=PHYLOP_HIST_RANGE, want=None) -> Dict[str, float]:
    """Full conservation superset: seed/flank summaries + shape over 4 windows.

    `want` is the set of feature names the caller will actually use; None means all of
    them. Whole shape windows that contribute nothing to it are skipped - see
    `_resolve_wanted` for why this cannot silently change a retained value.
    """
    features = {k: DEFAULT_VALUE for k in conservation_feature_names()}

    scores = parse_conservation_scores(site_data.get('conservation_vector'))
    if not scores or len(scores) != MRE_LEN:
        return features

    features['conservation_variance'] = round(float(np.var(scores)), 4)
    features['conservation_range'] = round(float(np.max(scores) - np.min(scores)), 4)

    seed_mre_positions = [p for p in get_mre_positions_for_seed(site_data, mirna_binding_start)
                          if 0 <= p < MRE_LEN]

    seed_scores = []
    if seed_mre_positions:
        seed_start, seed_end = min(seed_mre_positions), max(seed_mre_positions)
        seed_scores = [scores[i] for i in seed_mre_positions]

        features['seed_conservation_mean'] = round(float(np.mean(seed_scores)), 4)
        features['seed_conservation_median'] = round(float(np.median(seed_scores)), 4)
        features['seed_conservation_max'] = round(float(np.max(seed_scores)), 4)
        features['seed_conservation_min'] = round(float(np.min(seed_scores)), 4)
        features['seed_conservation_std'] = (round(float(np.std(seed_scores)), 4)
                                             if len(seed_scores) > 1 else DEFAULT_VALUE)

        five_p = scores[max(0, seed_start - flank_size):seed_start]
        features['five_prime_flank_conservation_mean'] = (
            round(float(np.mean(five_p)), 4) if five_p else DEFAULT_VALUE)

        three_p = scores[seed_end + 1:min(MRE_LEN, seed_end + 1 + flank_size)]
        features['three_prime_flank_conservation_mean'] = (
            round(float(np.mean(three_p)), 4) if three_p else DEFAULT_VALUE)

        full = scores[max(0, seed_start - flank_size):min(MRE_LEN, seed_end + 1 + flank_size)]
        features['full_site_conservation_mean'] = (
            round(float(np.mean(full)), 4) if full else DEFAULT_VALUE)

        features['conservation_contrast'] = round(
            features['seed_conservation_mean'] - float(np.mean(scores)), 4)

        # A *difference*, not the old seed/flank ratio: phyloP is signed, so the
        # flank mean crosses zero and a ratio is unstable there (it explodes near
        # zero and flips sign below it). The difference is the signed-data analogue
        # and answers the same question - is the seed more conserved than its flanks.
        flank_mean = (features['five_prime_flank_conservation_mean'] +
                      features['three_prime_flank_conservation_mean']) / 2.0
        features['flank_conservation_diff'] = round(
            features['seed_conservation_mean'] - flank_mean, 4)

    # Shape over the seed sub-vector and the three positional tertiles. Each window is
    # independent of the others, so one contributing nothing to `want` can be skipped
    # outright.
    n = len(scores)
    t1, t2 = n // 3, 2 * (n // 3)
    for window_scores, prefix in [
        (seed_scores, 'seed'),
        (scores[0:t1], 'upstream'),
        (scores[t1:t2], 'central'),
        (scores[t2:n], 'downstream'),
    ]:
        if want is not None and not any(f'{prefix}_{k}' in want for k in _CONS_SHAPE_KEYS):
            continue
        if window_scores:
            features.update(
                _compute_vector_shape_features(window_scores, prefix, hist_range, want))

    return features


# ============================================================================
# DUPLEX NEAREST-NEIGHBOUR STACKING ENERGIES
# ============================================================================

_ENERGY_KEYS = [
    'energy_sum', 'n_steps',
    'energy_mean', 'energy_std', 'energy_min', 'energy_max', 'energy_range',
    'energy_asymmetry', 'energy_gradient', 'energy_volatility', 'energy_max_jump',
    'stability_run_frac', 'energy_oscillation', 'energy_drift',
    '5p_terminal_energy', '3p_terminal_energy',
]

# Energies are a property of the *duplex*, so they exist only on the miRNA regions
# (the axis the duplex is walked along). The MRE regions keep composition only.
_ENERGY_REGIONS = ['mirna_seed', 'mirna_3p']

# Cross-region terms. The seed/3' split is only meaningful when compared, and the
# number of loops differs between rows, so neither region's *mean* carries it.
_ENERGY_CROSS_KEYS = ['duplex_energy_sum_total', 'seed_vs_3p_energy_diff']


def duplex_energy_feature_names() -> List[str]:
    return ([f'{r}_{k}' for r in _ENERGY_REGIONS for k in _ENERGY_KEYS]
            + list(_ENERGY_CROSS_KEYS))


def duplex_energy_steps(site_data) -> List[tuple]:
    """Nearest-neighbour (Turner 2004) decomposition of the IntaRNA duplex.

    IntaRNA reports the duplex as `subseq_dp` ("target&query" subsequences) and
    `hybrid_dp` (their dot-bracket, target using '(' and query using ')'). The two
    strands are antiparallel, so the k-th '(' of the target pairs with the k-th
    ')' of the query counted *from the end*.

    Walking the base pairs 5'->3' along the miRNA, every consecutive pair of base
    pairs encloses exactly one interior loop: a stack when the two pairs are
    adjacent on both strands, otherwise a bulge or an internal loop. Its energy is
    taken from ViennaRNA (the same Turner 2004 parameters IntaRNA itself uses), so
    no thermodynamic constants are hardcoded here.

    Returns one (mirna_pos_5p, mirna_pos_3p, dG) triple per loop, ordered 5'->3'
    along the miRNA, with 1-based miRNA positions. Stacks come out negative
    (stabilising), bulges and internal loops positive.

    Note this is the *interior* of the duplex only. Duplex initiation, terminal
    AU/GU penalties and dangling ends are per-duplex end terms, not per-position
    ones - they are already carried by the Eall / Eall1 features - so summing
    these steps does not reproduce E_hybrid.
    """
    subseq_dp = site_data.get('subseq_dp', '') or ''
    hybrid_dp = site_data.get('hybrid_dp', '') or ''
    if '&' not in subseq_dp or '&' not in hybrid_dp:
        return []

    t_seq, q_seq = subseq_dp.split('&', 1)
    t_dp, q_dp = hybrid_dp.split('&', 1)
    t_seq = t_seq.upper().replace('T', 'U')
    q_seq = q_seq.upper().replace('T', 'U')
    if len(t_seq) != len(t_dp) or len(q_seq) != len(q_dp):
        return []

    opens = [i for i, c in enumerate(t_dp) if c == '(']
    closes = [j for j, c in enumerate(q_dp) if c == ')']
    if len(opens) != len(closes) or len(opens) < 2:
        return []

    # 1-based indices into the concatenated "target&query" fold compound.
    n1 = len(t_seq)
    pairs = [(i + 1, n1 + j + 1) for i, j in zip(opens, reversed(closes))]
    pairs.sort(key=lambda p: p[1])  # ascending query index == 5'->3' along the miRNA

    q_start = safe_int(site_data.get('start_query', 1), 1)
    fc = RNA.fold_compound(t_seq + '&' + q_seq)

    steps = []
    for (i_in, j_in), (i_out, j_out) in zip(pairs, pairs[1:]):
        # (i_out, j_out) encloses (i_in, j_in); neither loop ever spans the strand
        # break, so eval_int_loop is well defined across the '&'.
        dg = fc.eval_int_loop(i_out, j_out, i_in, j_in) / 100.0
        steps.append((j_in - n1 + q_start - 1, j_out - n1 + q_start - 1, dg))
    return steps


def _energy_series_features(energies, region_name: str) -> Dict[str, float]:
    """Reduce an ordered energy series to the per-region energy statistics.

    A statistic that is *undefined* for the series at hand is emitted as NaN, not
    0.0: the standard deviation of a single stack is undefined, not zero, and an
    absent region has no energy profile at all. Using 0.0 would place "undefined"
    mid-distribution (the energies span roughly -3.4 to +4.5), where no single tree
    split can isolate it, and would make an unpaired region indistinguishable from
    a paired one whose terms happen to cancel. NaN keeps it off the numeric axis,
    which the gradient-boosted learners route explicitly.

    `energy_sum` and `n_steps` are the two deliberate exceptions: they are 0 rather
    than NaN on an empty series, because they are not undefined there. A region spanned
    by fewer than two base pairs encloses no interior loop, so it contributes exactly
    0 kcal/mol of interior stacking energy and has exactly 0 of them. `n_steps` sitting
    beside the sum is what keeps "no pairing" distinguishable from "pairing whose terms
    happen to cancel", which is the ambiguity the NaN policy above exists to avoid.

    The sum matters separately from the mean because the number of loops varies row to
    row: a seed with 6 loops averaging -1.5 and a seed with 2 loops averaging -1.5 share
    a mean but total -9.0 against -3.0 kcal/mol. Nothing else in the feature set recovers
    the count - it was previously only implicit in *which* statistics came back NaN.
    """
    out = {f'{region_name}_{k}': np.nan for k in _ENERGY_KEYS}
    n = len(energies)

    # True of the empty series too - see the docstring.
    out[f'{region_name}_n_steps'] = n
    out[f'{region_name}_energy_sum'] = float(np.sum(energies)) if n else 0.0

    if n == 0:
        return out

    # Defined for any non-empty series.
    out[f'{region_name}_energy_mean'] = float(np.mean(energies))
    out[f'{region_name}_energy_min'] = float(np.min(energies))
    out[f'{region_name}_energy_max'] = float(np.max(energies))
    out[f'{region_name}_energy_range'] = float(np.max(energies) - np.min(energies))
    out[f'{region_name}_5p_terminal_energy'] = float(energies[0])
    out[f'{region_name}_3p_terminal_energy'] = float(energies[-1])

    # Need a spread: >= 2 stacks.
    if n >= 2:
        out[f'{region_name}_energy_std'] = float(np.std(energies))

        changes = np.diff(energies)
        out[f'{region_name}_energy_volatility'] = float(np.mean(np.abs(changes)))
        out[f'{region_name}_energy_max_jump'] = float(np.max(np.abs(changes)))
        out[f'{region_name}_energy_drift'] = float(energies[-1] - energies[0])
        out[f'{region_name}_stability_run_frac'] = (
            float(np.sum(np.abs(changes) < 0.3)) / len(changes))

        # Oscillation needs at least two non-flat changes to have a direction to reverse.
        signs = np.sign(changes)
        nonzero_signs = signs[signs != 0]
        if len(nonzero_signs) > 1:
            direction_changes = int(np.sum(np.abs(np.diff(nonzero_signs)) > 0))
            out[f'{region_name}_energy_oscillation'] = (
                float(direction_changes) / (len(nonzero_signs) - 1))

    # Need a trend: >= 3 stacks. Undefined (and numpy-nan) for a constant series.
    if n >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cm = np.corrcoef(np.arange(n), energies)
        out[f'{region_name}_energy_gradient'] = (
            float(cm[0, 1]) if cm.shape == (2, 2) else np.nan)

    # Need two halves to compare: >= 4 stacks.
    if n >= 4:
        half = n // 2
        out[f'{region_name}_energy_asymmetry'] = float(
            np.mean(energies[:half]) - np.mean(energies[half:]))

    return out


def extract_duplex_energy_features(site_data, want=None) -> Dict[str, float]:
    """Duplex stacking-energy features for the miRNA seed and 3' regions.

    A loop is assigned to a region only when *both* of the base pairs it lies
    between fall inside that region, so a step straddling position 8/9 belongs to
    neither.

    A region yields NaN for any statistic it has too few loops to define: none at all
    (fewer than two base pairs in the region - for `mirna_3p` this is the meaningful
    "no 3' supplementary pairing" case), fewer than 2 for a spread, 3 for a trend, 4
    for an asymmetry. The seed is the *more* NaN-prone of the two regions in practice,
    since it spans only positions 2-8 and so can hold at most six loops.
    """
    # The ViennaRNA loop decomposition is shared by both regions and by the cross terms,
    # so it is gated only when the whole block is unused - there is no cheaper partial.
    if want is not None and not any(f in want for f in duplex_energy_feature_names()):
        return {}

    seed_e, three_p_e = [], []
    for p5, p3, dg in duplex_energy_steps(site_data):
        if SEED_FIRST_POS <= p5 and p3 <= SEED_LAST_POS:
            seed_e.append(dg)
        elif p5 > SEED_LAST_POS:
            three_p_e.append(dg)

    out = {}
    out.update(_energy_series_features(seed_e, 'mirna_seed'))
    out.update(_energy_series_features(three_p_e, 'mirna_3p'))

    # A *difference*, not the seed's share of the total. The steps are signed - bulges
    # and internal loops come out positive - so the denominator of a share crosses zero,
    # where the ratio explodes and then flips sign. This is the same call already made
    # for `flank_conservation_diff` on signed phyloP, and it answers the same question:
    # how much of the interior binding energy sits in the seed rather than in 3'
    # supplementary pairing. Both terms are 0-safe, so neither is ever NaN.
    seed_sum = out['mirna_seed_energy_sum']
    three_p_sum = out['mirna_3p_energy_sum']
    out['duplex_energy_sum_total'] = seed_sum + three_p_sum
    out['seed_vs_3p_energy_diff'] = seed_sum - three_p_sum
    return out


# ============================================================================
# SHUFFLE-NORMALISED (z-SCORED) BINDING ENERGIES
# ============================================================================
#
# Raw `E` conflates "this is a good site" with "this miRNA binds everything strongly".
# The second term is large: mean energy against shuffled targets spans -10.05 to -1.14
# kcal/mol across the 1,227 train miRNAs. The z-score against a fixed panel of
# dinucleotide-shuffled targets removes it, leaving how much better this target is than
# generic sequence of the same composition, in units of this miRNA's own spread:
#
#     E_z_mirna = (E - E_bg_mean) / E_bg_sd
#
# WHAT IT BUYS. miRBench samples negatives per miRNA family, from target clusters that
# family does not bind - the loop is over miRNAs, the sampling over MREs, and there are no
# decoy miRNAs. So `E_bg_mean`/`E_bg_sd` are constant within one miRNA's rows, and the
# z-score is an affine transform of `E` there: it adds nothing to ranking targets for a
# fixed miRNA. It pays off across same-target/different-miRNA pairs (75.7% of rows sit on
# a target carrying both labels) and, mostly, in pooling - one tree split then means the
# same thing for a GC-rich and an AU-rich miRNA. Measured pooled: miRNA-identity eta^2 on
# `E` drops 0.183 -> 0.074 (null 0.057), univariate AUROC 0.705 -> 0.729.
#
# `E_bg_mean` / `E_bg_sd` ride along as features in their own right: they are a property
# of the miRNA *sequence*, not an identifier, so they stay meaningful under the
# cold-miRNA-family split. Expect them to carry almost nothing alone (`E_bg_mean_mirna`
# measured 0.524 AUROC) - miRBench balances families across labels, so miRNA avidity is
# not label-predictive. That is the point: the table normalises, it does not smuggle in a
# miRNA prior.
#
# See src/shuffle_background.py for the panel construction and for why only the miRNA side
# is normalised (~111M IntaRNA calls the other way, off ~2.25 rows per target).

_ZSCORE_KEYS = ['E_z_mirna', 'E_hybrid_z_mirna', 'E_bg_mean_mirna', 'E_bg_sd_mirna']


def shuffle_zscore_feature_names() -> List[str]:
    return list(_ZSCORE_KEYS)


def load_mirna_background(path: str) -> Dict[str, dict]:
    """Read shuffle_background.py's table, keyed on the miRNA sequence."""
    background = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            seq = (row.get('mirna_sequence') or '').upper().replace('T', 'U')
            if seq:
                background[seq] = row
    return background


def extract_shuffle_zscore_features(site_data, background: Optional[Dict[str, dict]]
                                    ) -> Dict[str, float]:
    out = {k: np.nan for k in _ZSCORE_KEYS}
    if not background:
        return out

    entry = background.get((site_data.get('mirna_seq') or '').upper().replace('T', 'U'))
    if entry is None:
        return out

    mean = safe_float(entry.get('E_bg_mean'), np.nan)
    sd = safe_float(entry.get('E_bg_sd'), np.nan)
    hybrid_mean = safe_float(entry.get('E_hybrid_bg_mean'), np.nan)
    hybrid_sd = safe_float(entry.get('E_hybrid_bg_sd'), np.nan)

    out['E_bg_mean_mirna'] = mean
    out['E_bg_sd_mirna'] = sd

    # A zero spread means every panel target gave the identical energy, so "how many
    # standard deviations out" has no answer. NaN, not a division by zero.
    if sd and sd > 0:
        out['E_z_mirna'] = (safe_float(site_data.get('E'), np.nan) - mean) / sd
    if hybrid_sd and hybrid_sd > 0:
        out['E_hybrid_z_mirna'] = (
            safe_float(site_data.get('E_hybrid'), np.nan) - hybrid_mean) / hybrid_sd

    return out


# ============================================================================
# REGION SEQUENCE COMPOSITION FEATURES
# ============================================================================

_PUR_PATTERNS = ['RRR', 'RRY', 'RYR', 'RYY', 'YRR', 'YRY', 'YYR', 'YYY']

_COMP_KEYS = ([f'{p}_freq' for p in _PUR_PATTERNS]
              + ['trinuc_entropy', 'unique_trinuc_ratio', 'trinuc_gini',
                 'trinuc_repeat_ratio', 'ggg_count', 'uuu_count'])

_COMP_REGIONS = ['mre_5p', 'mre_3p', 'mirna_seed', 'mirna_3p']


def composition_feature_names() -> List[str]:
    return [f'{r}_{k}' for r in _COMP_REGIONS for k in _COMP_KEYS]


def safe_divide(num, den, default=DEFAULT_VALUE):
    if den == 0 or np.isnan(den) or np.isinf(den):
        return default
    r = num / den
    return default if (np.isnan(r) or np.isinf(r)) else r


def get_regions(seq: str, is_mirna: bool = False) -> Dict[str, str]:
    seq = seq.upper().replace('T', 'U')
    seq_len = len(seq)
    regions = {}
    if is_mirna:
        if seq_len >= 8:
            regions['seed'] = seq[1:8]
        elif seq_len > 1:
            regions['seed'] = seq[1:]
        else:
            regions['seed'] = seq
        if seq_len > 8:
            regions['3p'] = seq[8:]
        elif seq_len >= 3:
            regions['3p'] = seq[-min(3, seq_len):]
        else:
            regions['3p'] = seq
    else:
        mid = max(seq_len // 2, 1)
        regions['5p'] = seq[:mid]
        regions['3p'] = seq[mid:] if seq_len > mid else seq
    return regions


def extract_region_features(seq: str, region_name: str) -> Dict[str, float]:
    """Per-region composition. No energies here - those come from the duplex."""
    out = {f'{region_name}_{k}': DEFAULT_VALUE for k in _COMP_KEYS}
    out[f'{region_name}_ggg_count'] = 0
    out[f'{region_name}_uuu_count'] = 0

    if not seq or len(seq) < 3:
        return out

    n_trinuc = len(seq) - 2
    trinuc_counts = Counter(seq[i:i + 3] for i in range(n_trinuc))

    pur_seq = ''.join('R' if nt in 'AG' else 'Y' for nt in seq)
    pur_trinucs = Counter(pur_seq[i:i + 3] for i in range(len(pur_seq) - 2))
    for pattern in _PUR_PATTERNS:
        out[f'{region_name}_{pattern}_freq'] = safe_divide(
            pur_trinucs.get(pattern, 0), n_trinuc)

    freqs = [c / n_trinuc for c in trinuc_counts.values()]
    entropy = -sum(f * np.log2(f) for f in freqs if f > 0)
    out[f'{region_name}_trinuc_entropy'] = safe_divide(entropy, np.log2(min(n_trinuc, 64)))

    out[f'{region_name}_unique_trinuc_ratio'] = safe_divide(
        len(trinuc_counts), min(n_trinuc, 64))

    # Gini over trinucleotide *counts* - non-negative by construction, so unlike the
    # conservation Gini this one stays well defined.
    if len(trinuc_counts) > 1:
        sorted_counts = sorted(trinuc_counts.values())
        cumsum = np.cumsum(sorted_counts)
        k = len(sorted_counts)
        if cumsum[-1] > 0:
            gini = 1 - 2 * float(np.sum(cumsum)) / (k * cumsum[-1]) + 1 / k
            out[f'{region_name}_trinuc_gini'] = DEFAULT_VALUE if np.isnan(gini) else gini

    repeated = sum(1 for c in trinuc_counts.values() if c > 1)
    out[f'{region_name}_trinuc_repeat_ratio'] = safe_divide(repeated, len(trinuc_counts))

    out[f'{region_name}_ggg_count'] = seq.count('GGG')
    out[f'{region_name}_uuu_count'] = seq.count('UUU')
    return out


def extract_sequence_region_features(mre_seq: str, mirna_seq: str, want=None) -> Dict[str, float]:
    """Composition over the four regions. Regions contributing nothing to `want` are
    skipped; they are computed independently, so dropping one cannot affect another."""
    mre_regions = get_regions(mre_seq) if mre_seq else {'5p': '', '3p': ''}
    mirna_regions = get_regions(mirna_seq, is_mirna=True) if mirna_seq else {'seed': '', '3p': ''}

    out = {}
    for regions, key, name in [
        (mre_regions, '5p', 'mre_5p'),
        (mre_regions, '3p', 'mre_3p'),
        (mirna_regions, 'seed', 'mirna_seed'),
        (mirna_regions, '3p', 'mirna_3p'),
    ]:
        if want is not None and not any(f'{name}_{k}' in want for k in _COMP_KEYS):
            continue
        out.update(extract_region_features(regions.get(key, ''), name))
    return out


# ============================================================================
# HELPERS
# ============================================================================

def get_max_consecutive_char(text, char):
    return max((len(s) for s in re.findall(f"{re.escape(char)}+", text)), default=0)


def safe_float(value, default=0.0):
    if value is None or value in ('NA', ''):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    if value is None or value in ('NA', ''):
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


# ============================================================================
# IntaRNA ENERGY + DUPLEX STATISTICS
# ============================================================================

# From merge_intarna.py. `E`/`E_hybrid`/`ED_*` describe the MFE duplex (the one `hybrid_dp`
# encodes); `Eall*` are pair-level partition-function terms; `E_total` and `P_duplex` are
# computed from those. `P_duplex` is the Boltzmann weight of this duplex within the ensemble
# and is deliberately NOT IntaRNA's site-level `P_E` - see merge_intarna.py.
INTARNA_ENERGY_FEATURES = [
    'E', 'E_hybrid', 'ED_target', 'ED_query', 'E_total',
    'Eall', 'Eall1', 'Eall2', 'Ealltotal', 'P_duplex',
    'Energy_norm', 'Energy_hybrid_norm',
]

# Summaries of the sub-optimal sites IntaRNA reported for the pair but which the
# priority-score selection discarded. Computed in best_intarna.py (SUBOPT_STAT_COLS -
# the two lists must agree, and _check_subopt_columns below fails loudly if they do
# not) and carried through merge_intarna.py untouched, so they are read straight off
# the row here rather than recomputed.
#
# They are censored at IntaRNA's `-n` (10 in intarna_parallel.py), so `subopt_n_sites`
# means "sites, capped at 10". Only comparable across runs sharing `-n`/`--outDeltaE`.
SUBOPT_STAT_FEATURES = [
    'subopt_n_sites',
    'subopt_E_min',
    'subopt_E_gap',
    'subopt_E_mean',
    'subopt_E_std',
    'subopt_n_within_1kcal',
    'subopt_E_delta_selected',
    'subopt_priority_gap',
    'subopt_frac_seedlike',
    'subopt_target_span',
    'subopt_n_distinct_starts',
]

DUPLEX_STAT_FEATURES = [
    'total_matches', 'total_mismatches', 'total_bulges', 'total_mre_bulges',
    'total_mirna_bulges', 'total_gu_wobbles', 'interaction_length', 'match_fraction',
    'seed_matches', 'seed_matches_2_8', 'consecutive_matches_seed',
    'gu_wobbles_in_seed_2_8_pos', 'effective_seed_matches',
    'non_seed_matches', 'consecutive_matches_minus_seed', 'gu_wobbles_minus_seed',
    'effective_3prime_matches', 'priority_score',
    'seed_au_content', 'seed_gc_content', 'nonseed_au_content', 'nonseed_gc_content',
]


def _resolve_wanted(features: Optional[List[str]]) -> Optional[set]:
    """The set of feature names computation may be restricted to, or None for all.

    WHY THIS IS SAFE. Skipping work risks train/serve skew - the exact failure this
    module was merged to prevent - so the gating obeys two rules:

      1. It is DERIVED from the requested names, never hand-maintained. Add a feature to
         the selection and its block switches itself back on; there is no mapping to
         forget to update.
      2. It only ever skips units that are computed *independently* of one another - a
         conservation shape window, a composition region, the duplex-energy block. A
         retained feature is produced by byte-identical code either way, because nothing
         it depends on is shared with what was skipped.

    Rule 2 is what makes the claim checkable rather than merely argued, and it is checked:
    `--verify-gating` recomputes every row ungated and asserts the retained columns match
    exactly. Anything the gating cannot prove safe belongs outside it.

    The duplex-stat block is deliberately NOT gated. `classify_binding_type_detailed`
    reads nine of its intermediates, so `binding_type_*` alone requires essentially all
    of it, and the block is cheap arithmetic over an already-built duplex vector.
    """
    if features is None:
        return None
    wanted = set(features)
    # binding_type_<value> columns are one-hots of the classifier, which needs the whole
    # duplex-stat block - so asking for one means asking for that block, not for a
    # feature named `binding_type_<value>`.
    return wanted


def calculate_intarna_features(site_data, flank_size=10, hist_range=PHYLOP_HIST_RANGE,
                               want=None):
    hybrid_dp = site_data.get('hybrid_dp', '') or ''
    mre_struct, mirna_struct = hybrid_dp.split('&', 1) if '&' in hybrid_dp else ('', '')

    subseq_dp = site_data.get('subseq_dp', '') or ''
    if '&' in subseq_dp:
        a, b = subseq_dp.split('&', 1)
        mre_binding_subseq = a.upper().replace('T', 'U')
        mirna_binding_subseq = b.upper().replace('T', 'U')
    else:
        mre_binding_subseq = mirna_binding_subseq = ''

    mre_coord_start = safe_int(site_data.get('start_target', 1))
    mre_coord_end = safe_int(site_data.get('end_target', 1))
    mirna_coord_start = safe_int(site_data.get('start_query', 1))

    site_data['mre_coord_start'] = mre_coord_start
    site_data['mre_coord_end'] = mre_coord_end
    site_data['mirna_coord_start'] = mirna_coord_start

    mre_binding_start = mre_coord_start - 1
    mirna_binding_start = mirna_coord_start - 1

    for key in INTARNA_ENERGY_FEATURES:
        site_data[key] = safe_float(site_data.get(key, 0))

    # Duplex vector: target structure reversed, since the strands are antiparallel.
    total_vec = create_duplex_vectors(mre_struct[::-1], mirna_struct)
    site_data['total_vec'] = total_vec

    site_data['total_matches'] = total_vec.count('1')
    site_data['total_mismatches'] = total_vec.count('2')
    site_data['total_mre_bulges'] = total_vec.count('3')
    site_data['total_mirna_bulges'] = total_vec.count('4')
    site_data['total_bulges'] = site_data['total_mre_bulges'] + site_data['total_mirna_bulges']
    site_data['interaction_length'] = len(total_vec)
    site_data['match_fraction'] = round(
        safe_divide(site_data['total_matches'], len(total_vec)), 4)

    # Seed region (miRNA positions 1-8, 0-indexed [0, 8))
    seed_region_vec = get_mirna_region_vector(total_vec, 0, 8, mirna_binding_start)
    seed_region_2_8_vec = get_mirna_region_vector(total_vec, 1, 8, mirna_binding_start)

    site_data['seed_matches'] = seed_region_vec.count('1')
    site_data['seed_matches_2_8'] = seed_region_2_8_vec.count('1')
    site_data['consecutive_matches_seed'] = get_max_consecutive_char(seed_region_2_8_vec, '1')
    site_data['total_matches_in_seed_9_pos'] = count_char_in_mirna_region(
        total_vec, 0, 9, mirna_binding_start, '1')
    site_data['mismatch_seed_positions'] = seed_region_vec.count('2')
    site_data['seed_target_bulge_positions'] = seed_region_vec.count('3')
    site_data['seed_mirna_bulge_positions'] = seed_region_vec.count('4')

    non_seed_indices = get_vector_indices_for_mirna_range(total_vec, 8, 100, mirna_binding_start)
    non_seed_vec = ''.join(total_vec[i] for i in non_seed_indices)
    site_data['consecutive_matches_minus_seed'] = get_max_consecutive_char(non_seed_vec, '1')
    site_data['non_seed_matches'] = non_seed_vec.count('1')

    try:
        site_data['start_match_seed'] = mirna_struct.index(')') + 1
    except ValueError:
        site_data['start_match_seed'] = 99

    all_paired_mre, all_paired_mir, mirna_positions = get_paired_bases_with_positions(
        total_vec, mre_binding_subseq[::-1], mirna_binding_subseq,
        mre_binding_start, mirna_binding_start
    )

    _AU = [{'A', 'U'}, {'A', 'T'}, {'U', 'A'}, {'T', 'A'}]
    _GC = [{'G', 'C'}, {'C', 'G'}]
    _GU = [{'G', 'U'}, {'G', 'T'}]

    seed_pairs = [(mre, mir) for mre, mir, pos in
                  zip(all_paired_mre, all_paired_mir, mirna_positions) if 1 <= pos <= 7]
    if seed_pairs:
        site_data['seed_au_content'] = round(
            sum(1 for m, q in seed_pairs if {m, q} in _AU) / len(seed_pairs), 4)
        site_data['seed_gc_content'] = round(
            sum(1 for m, q in seed_pairs if {m, q} in _GC) / len(seed_pairs), 4)
    else:
        site_data['seed_au_content'] = DEFAULT_VALUE
        site_data['seed_gc_content'] = DEFAULT_VALUE

    site_data['gu_wobbles_in_seed'] = sum(1 for m, q in seed_pairs if {m, q} in _GU)
    site_data['gu_wobbles_in_seed_2_8_pos'] = site_data['gu_wobbles_in_seed']

    nonseed_pairs = [(mre, mir) for mre, mir, pos in
                     zip(all_paired_mre, all_paired_mir, mirna_positions) if pos > 7]
    if nonseed_pairs:
        site_data['nonseed_au_content'] = round(
            sum(1 for m, q in nonseed_pairs if {m, q} in _AU) / len(nonseed_pairs), 4)
        site_data['nonseed_gc_content'] = round(
            sum(1 for m, q in nonseed_pairs if {m, q} in _GC) / len(nonseed_pairs), 4)
    else:
        site_data['nonseed_au_content'] = DEFAULT_VALUE
        site_data['nonseed_gc_content'] = DEFAULT_VALUE

    site_data['total_matches_minus_seed'] = len(nonseed_pairs)
    site_data['gu_wobbles_minus_seed'] = sum(1 for m, q in nonseed_pairs if {m, q} in _GU)
    site_data['total_gu_wobbles'] = (site_data['gu_wobbles_in_seed']
                                     + site_data['gu_wobbles_minus_seed'])

    # G:U wobbles only count against the seed when there is more than one.
    priority_seed = site_data['seed_matches']
    if site_data['gu_wobbles_in_seed'] > 1:
        priority_seed -= site_data['gu_wobbles_in_seed']
        site_data['effective_seed_matches'] = priority_seed
    else:
        site_data['effective_seed_matches'] = site_data['seed_matches']

    loop_penalty = (site_data['total_bulges'] + site_data['total_mismatches']) / 3.0
    weight = next((w for w, thr in [(4, 7), (3, 6), (2, 5), (1, 4)] if priority_seed > thr), 0)
    site_data['priority_score'] = round(
        weight * priority_seed + site_data['non_seed_matches']
        - loop_penalty - site_data['gu_wobbles_minus_seed'], 4)

    site_data['effective_3prime_matches'] = (site_data['total_matches_minus_seed']
                                             - site_data['gu_wobbles_minus_seed'])

    site_data.update(extract_duplex_energy_features(site_data, want))
    site_data.update(extract_conservation_features(
        site_data, mirna_binding_start, flank_size, hist_range, want))

    return site_data


# ============================================================================
# BINDING TYPE CLASSIFICATION
# ============================================================================

def classify_binding_type_detailed(d):
    consecutive = d.get('consecutive_matches_seed', 0)
    btype = f"{consecutive}mer" if consecutive >= 5 else "seedless"

    if d.get('start_match_seed', 99) >= 4:
        btype = "seedless"
    if d.get('total_matches_in_seed_9_pos', 0) == 9:
        btype = "9mer"

    if btype == "seedless":
        mirna_binding_start = d.get('mirna_coord_start', 1) - 1
        centered_matches = count_consecutive_matches_in_mirna_region(
            d.get('total_vec', ''), 4, 16, mirna_binding_start)
        if centered_matches >= 8:
            return "centered"
        elif d.get('start_match_seed', 99) >= 13:
            return "3prime"
        elif d.get('effective_3prime_matches', 0) >= 6:
            return "3prime.compensatory"
        return btype

    mirna_seq = d.get('mirna_seq', '')
    start_match = d.get('start_match_seed', 99)

    if btype == "8mer" and start_match == 2 and mirna_seq and mirna_seq[0] == 'A':
        btype = "8mer1A"
    elif btype == "7mer" and start_match == 2 and mirna_seq and mirna_seq[0] == 'A':
        btype = "8mer1A"
    elif btype == "6mer":
        if start_match == 3:
            btype = "offset6mer"
        elif start_match == 2 and mirna_seq and mirna_seq[0] == 'A':
            btype = "7mer1A"

    if 'mer' in btype:
        if d.get('mismatch_seed_positions', 0) > 0:
            btype += ".mismatch"
        if d.get('seed_target_bulge_positions', 0) > 0:
            btype += ".target.bulge"
        if d.get('seed_mirna_bulge_positions', 0) > 0:
            btype += ".mirna.bulge"
        if d.get('gu_wobbles_in_seed', 0) > 0:
            btype += ".GU"
        if d.get('effective_3prime_matches', 0) >= 3:
            btype += ".3prime"

    return btype


# ============================================================================
# FEATURE REGISTRY
# ============================================================================

def all_feature_names() -> List[str]:
    """Every feature this module knows how to compute, in a stable order.

    No longer what gets written by default - see DEFAULT_FEATURES. This is now the
    *vocabulary*: it validates requested names, sizes the gating report, and is what
    `--list-features` prints so a superset run remains one pipe away.
    """
    return (INTARNA_ENERGY_FEATURES
            + SUBOPT_STAT_FEATURES
            + DUPLEX_STAT_FEATURES
            + conservation_feature_names()
            + composition_feature_names()
            + duplex_energy_feature_names()
            + shuffle_zscore_feature_names())


# Features added after the featurewiz run that produced SELECTED_FEATURES, so featurewiz
# has never seen them and cannot have rejected them. Kept OUT of SELECTED_FEATURES on
# purpose: that list is the record of what feature selection actually chose, and folding
# these in would erase the distinction the next selection run needs.
#
# Derived rather than written out, so it cannot drift from the blocks it names.
#
# `subopt_n_sites` is excluded: IntaRNA is run with `-n 10 --outDeltaE 100`, which always
# fills the quota, so the column is constant at 10 on every row (measured) and can only
# ever vary if those flags change.
NEW_CANDIDATE_FEATURES = (
    [c for c in SUBOPT_STAT_FEATURES if c != 'subopt_n_sites']
    + [f'{r}_{k}' for r in _ENERGY_REGIONS for k in ('energy_sum', 'n_steps')]
    + list(_ENERGY_CROSS_KEYS)
    + list(_ZSCORE_KEYS)
)


# What the extractor writes, and computes, unless told otherwise.
#
# The full superset is no longer produced by default. featurewiz has already judged it
# (commit "added selected features after rerun"), and from here on selection runs against
# this list instead: the survivors plus whatever new candidates are under test. So the
# ~120 features it rejected are no longer computed, which is the whole point - they cost
# time at inference and nothing consumes them.
#
# THE ONE THING TO KNOW. featurewiz's rejections were conditional on the set it saw: a
# feature dropped because a correlated competitor beat it may be the better choice once
# that competitor is gone. So this is a ratchet - but a soft one, deliberately. Every
# extraction function is still here and every name is still in all_feature_names(), so any
# rejected feature can be brought back by name:
#
#     python feature_extraction.py --list-features > superset.txt
#     python feature_extraction.py ... --features-file superset.txt
#
# Promote a candidate into SELECTED_FEATURES (and drop it from NEW_CANDIDATE_FEATURES)
# only after a featurewiz run has actually kept it.
DEFAULT_FEATURES = list(SELECTED_FEATURES) + list(NEW_CANDIDATE_FEATURES)


# Named column sets for the training-side A/B, resolved against a default-mode CSV.
# `None` means "every feature column present in the CSV" - which is now DEFAULT_FEATURES,
# so 'all' and 'baseline+new' select the same columns unless you extracted with an
# explicit --features-file. 'baseline' is still the strict subset featurewiz chose, which
# is what makes the A/B meaningful.
FEATURE_SETS = {
    'baseline': lambda: list(SELECTED_FEATURES),
    'baseline+new': lambda: list(SELECTED_FEATURES) + list(NEW_CANDIDATE_FEATURES),
    'all': lambda: None,
}


def feature_set(name: str) -> Optional[List[str]]:
    """Resolve a FEATURE_SETS name to its column list (None = keep everything)."""
    if name not in FEATURE_SETS:
        raise KeyError(f"unknown feature set {name!r}; have {sorted(FEATURE_SETS)}")
    return FEATURE_SETS[name]()


# Identifier / passthrough columns, written in both modes. `energy_source` rides along so
# the MFE fallback is auditable from the feature CSV itself; it is never a model input
# (both feature_selection.py and gluon_training_kfold.py drop it).
ID_COLS = ['target_id', 'query_id', 'binding_type', 'hybrid_dp', 'subseq_dp',
           'mre_sequence', 'mirna_sequence', 'chimeric_sequence', 'mir_fam',
           'energy_source']


# ============================================================================
# MAIN
# ============================================================================

def compute_site_features(site, flank_size, hist_range, background, want=None):
    """Compute one site's features in place.

    The single entry point for per-site computation, so the gated and ungated paths are
    the same code with a different `want` rather than two code paths that could drift.
    """
    calculate_intarna_features(site, flank_size=flank_size, hist_range=hist_range,
                               want=want)
    # Always: `binding_type_*` one-hots are derived from it, and it is cheap.
    site['binding_type'] = classify_binding_type_detailed(site)
    site.update(extract_sequence_region_features(
        site.get('mre_seq', ''), site.get('mirna_seq', ''), want))
    site.update(extract_shuffle_zscore_features(site, background))
    site['mre_sequence'] = site['mre_seq']
    site['mirna_sequence'] = site['mirna_seq']
    return site


# ============================================================================
# PARALLEL EXECUTION
# ============================================================================
#
# The per-site computation is embarrassingly parallel: sites share no state, and
# compute_site_features touches nothing outside the dict it is handed.
#
# PROCESSES, NOT THREADS. The work is numpy reductions over 50-element arrays plus
# SWIG-wrapped ViennaRNA calls. The arrays are far too small for numpy to profitably
# release the GIL, and the ViennaRNA bindings hold it throughout, so a thread pool would
# serialise almost perfectly. Processes cost pickling, which chunking amortises.
#
# Below PARALLEL_MIN_ROWS the pool costs more to start than it saves, so the serial path
# is used - which matters for inference, where a handful of pairs is a normal request.

PARALLEL_MIN_ROWS = 2000
CHUNK_ROWS = 500

# Set once per worker by _worker_init. Read-only after that.
_WORKER_CFG: dict = {}


def _worker_init(flank_size, hist_range, background, want):
    """Ship the read-only config to each worker once, rather than with every chunk."""
    _WORKER_CFG.update(flank_size=flank_size, hist_range=hist_range,
                       background=background, want=want)


def _worker_chunk(chunk):
    return [compute_site_features(s, _WORKER_CFG['flank_size'], _WORKER_CFG['hist_range'],
                                  _WORKER_CFG['background'], _WORKER_CFG['want'])
            for s in chunk]


def compute_all_sites(sites, flank_size, hist_range, background, want, processes=1):
    """Compute every site's features, in parallel when it is worth it.

    Results are written back in input order (`imap` preserves it), so the output is
    identical to the serial path regardless of how many processes are used - there is no
    ordering nondeterminism to reason about downstream.
    """
    if processes <= 1 or len(sites) < PARALLEL_MIN_ROWS:
        if processes > 1:
            print(f"  {len(sites)} rows is below the {PARALLEL_MIN_ROWS}-row threshold; "
                  f"running serially (pool startup would cost more than it saves)")
        for s in sites:
            compute_site_features(s, flank_size, hist_range, background, want)
        return sites

    chunks = (sites[i:i + CHUNK_ROWS] for i in range(0, len(sites), CHUNK_ROWS))
    n_chunks = (len(sites) + CHUNK_ROWS - 1) // CHUNK_ROWS
    print(f"  {processes} processes, {n_chunks} chunks of {CHUNK_ROWS}")

    with mp.Pool(processes=processes, initializer=_worker_init,
                 initargs=(flank_size, hist_range, background, want)) as pool:
        done_rows = 0
        for k, done in enumerate(pool.imap(_worker_chunk, chunks)):
            sites[k * CHUNK_ROWS:k * CHUNK_ROWS + len(done)] = done
            done_rows += len(done)
            if (k + 1) % 20 == 0 or done_rows == len(sites):
                print(f"\r  {done_rows}/{len(sites)}", end='', file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return sites


def _verify_gating(sites, features, flank_size, hist_range, background, sample_size=200):
    """Recompute a sample ungated and assert every retained column is unchanged.

    This is what turns "skipping that block is safe" from an argument into a check. A
    mismatch means the gating dropped something a retained feature depended on, which is
    train/serve skew - so it aborts rather than warns.
    """
    import random
    subset = sites if len(sites) <= sample_size else random.Random(0).sample(sites, sample_size)

    mismatches = []
    for site in subset:
        # A plain copy, with nothing stripped. Removing "output" names would also remove
        # the pass-through inputs that share them - `Eall` and the `subopt_*` columns are
        # read off the IntaRNA row, not computed - and zero them. Recomputation overwrites
        # every retained feature regardless, so stripping buys nothing.
        reference = compute_site_features(
            dict(site), flank_size, hist_range, background, want=None)
        for feature in features:
            got, expected = site.get(feature), reference.get(feature)
            if got != expected and not (
                    isinstance(got, float) and isinstance(expected, float)
                    and np.isnan(got) and np.isnan(expected)):
                mismatches.append((feature, got, expected))

    if mismatches:
        shown = mismatches[:10]
        sys.exit(f"ERROR: feature gating changed {len(mismatches)} value(s) across "
                 f"{len(subset)} sampled rows. This is a gating bug - rerun with "
                 f"--no-feature-gating and report it.\n" +
                 "\n".join(f"  {f}: gated={g!r} ungated={e!r}" for f, g, e in shown))
    print(f"  Gating verified: {len(features)} features identical to the ungated "
          f"computation across {len(subset)} sampled rows")


def _check_subopt_columns(intarna_results):
    """Warn loudly when the IntaRNA table predates the sub-optimal statistics.

    An older best_intarna.py emits none of SUBOPT_STAT_FEATURES. Those rows would read
    back as NaN, which is the right value - but silently, and a whole feature block
    being NaN across a training set is worth one line of output rather than a discovery
    made later from a model that ignores eleven columns.
    """
    if not intarna_results:
        return
    present = set(intarna_results[0].keys())
    missing = [c for c in SUBOPT_STAT_FEATURES if c not in present]
    if missing:
        print(f"  WARNING: {len(missing)}/{len(SUBOPT_STAT_FEATURES)} sub-optimal-site "
              f"columns absent from the IntaRNA table (e.g. {missing[:3]}); they will be "
              f"NaN. Regenerate with the current src/best_intarna.py to populate them.")
    else:
        print(f"  Sub-optimal site statistics: all {len(SUBOPT_STAT_FEATURES)} columns present")


def main():
    parser = argparse.ArgumentParser(
        description='Extract features for miRNA-MRE pairs from an IntaRNA duplex. '
                    'Writes and computes DEFAULT_FEATURES (the featurewiz selection plus '
                    'the candidates under test) unless --features-file says otherwise.')
    # --list-features exits before anything else is read, so the required arguments below
    # must not be enforced for it.
    if '--list-features' in sys.argv:
        for name in all_feature_names():
            print(name)
        return 0

    parser.add_argument('--intarna', required=True, help='best_intarna results TSV')
    parser.add_argument('--mre-fasta', required=True)
    parser.add_argument('--mirna-fasta', required=True)
    parser.add_argument('--v7', required=True,
                        help='v7 TSV: conservation vector, family, label, coordinates')
    parser.add_argument('--output', required=True)
    parser.add_argument('--list-features', action='store_true',
                        help='print every computable feature name, one per line, and exit. '
                             'Pipe into a file and pass it back with --features-file to '
                             'reproduce the old --all-features superset.')
    parser.add_argument('--features-file', default=None,
                        help='JSON/newline list of features to write, overriding '
                             'DEFAULT_FEATURES (e.g. a fresh feature_selection.py run, or '
                             'the --list-features superset)')
    parser.add_argument('--cons-col', default=DEFAULT_CONS_COL,
                        help=f'conservation column in the v7 TSV (default {DEFAULT_CONS_COL})')
    parser.add_argument('--cons-no-reverse', action='store_true',
                        help='do NOT reverse the conservation vector on minus-strand rows. '
                             'Correct only for a column already in transcript order.')
    parser.add_argument('--allow-missing-conservation', action='store_true',
                        help='proceed with zeroed conservation features when the v7 file has '
                             'no conservation column. For inference only - never for training.')
    parser.add_argument('--fallback-report', default=None,
                        help='TSV listing, by chimeric sequence, the pairs for which the '
                             'ensemble run found no interaction, so the partition-function '
                             'energies are NaN (energy_source == "mfe_only")')
    parser.add_argument('--mirna-background', default=None,
                        help='per-miRNA shuffled-target background table from '
                             'src/shuffle_background.py. Without it the four '
                             'shuffle-z-score features are NaN.')
    parser.add_argument('--threads', type=int, default=0,
                        help='worker processes for the per-site computation. 0 (default) '
                             'uses every core; 1 forces the serial path. Output is '
                             'identical either way - results are reordered to match the '
                             'input.')
    parser.add_argument('--no-feature-gating', action='store_true',
                        help='compute every feature even when only a subset is written. '
                             'The gated path is verified identical, so this is an escape '
                             'hatch for debugging, not a correctness switch.')
    parser.add_argument('--verify-gating', action='store_true',
                        help='recompute a sample of rows with gating off and abort if any '
                             'written value differs. Use after changing the feature list.')
    parser.add_argument('--flank-size', type=int, default=10)
    parser.add_argument('--bigwig', default=None,
                        help='read conservation from a bigwig instead of the v7 column')
    args = parser.parse_args()

    if args.threads <= 0:
        args.threads = os.cpu_count() or 1

    hist_range = (PHASTCONS_HIST_RANGE if 'phastcons' in args.cons_col.lower()
                  else PHYLOP_HIST_RANGE)

    print("--- Reading inputs ---")
    intarna_results = parse_intarna_results(args.intarna)
    mre_seqs = parse_fasta(args.mre_fasta)
    mirna_seqs = parse_fasta(args.mirna_fasta)
    v7 = load_v7(args.v7, args.cons_col, args.bigwig,
                 reverse_minus_strand=not args.cons_no_reverse,
                 allow_missing_conservation=args.allow_missing_conservation)

    print(f"  IntaRNA: {len(intarna_results)}   MRE: {len(mre_seqs)}   "
          f"miRNA: {len(mirna_seqs)}   v7: {len(v7)}")

    # The FASTAs and the v7 TSV are strictly row-aligned (make_fastas.py guarantees it).
    if not len(mre_seqs) == len(mirna_seqs) == len(v7):
        sys.exit("ERROR: the MRE FASTA, miRNA FASTA and v7 TSV must be row-aligned. "
                 "Regenerate the FASTAs with src/make_fastas.py.")

    # The IntaRNA table may legitimately be *shorter*: merge_intarna.py inner-joins the
    # MFE duplex against the ensemble energies on coordinates, and a few pairs find no
    # match. Those rows are dropped, so rows are located by pair_index, not by position.
    has_pair_index = bool(intarna_results) and 'pair_index' in intarna_results[0]
    if not has_pair_index and len(intarna_results) != len(v7):
        sys.exit(f"ERROR: IntaRNA table has {len(intarna_results)} rows for {len(v7)} v7 "
                 f"rows and carries no pair_index column, so the two cannot be aligned.")
    if has_pair_index and len(intarna_results) < len(v7):
        print(f"  NOTE: {len(v7) - len(intarna_results)} pair(s) absent from the IntaRNA "
              f"table (dropped by the ensemble merge); they are skipped.")

    print(f"  Conservation track: {args.cons_col} (entropy histogram range {hist_range})")

    _check_subopt_columns(intarna_results)

    background = None
    if args.mirna_background:
        background = load_mirna_background(args.mirna_background)
        print(f"  miRNA background: {len(background)} miRNAs from {args.mirna_background}")
    else:
        print(f"  WARNING: no --mirna-background; {shuffle_zscore_feature_names()} "
              f"will be NaN.")

    print("\n--- Building site data ---")
    sites = []
    for pos, row in enumerate(intarna_results):
        # best_intarna emits a 1-based pair_index; trust it over enumeration order, so
        # an unexpected row count desynchronises loudly instead of silently pairing each
        # duplex with the wrong v7 row (and hence the wrong conservation vector).
        i = safe_int(row['pair_index'], 0) - 1 if 'pair_index' in row else pos
        if not 0 <= i < len(v7):
            sys.exit(f"ERROR: pair_index {row.get('pair_index')} out of range for "
                     f"{len(v7)} v7 rows.")

        if row.get('status') == 'no_interactions':
            continue
        if row.get('start_target') in ('NA', '', None):
            continue

        v7_row = v7.iloc[i]
        site = {k: row.get(k) for k in
                ('target_id', 'query_id', 'start_target', 'end_target',
                 'start_query', 'end_query', 'subseq_dp', 'hybrid_dp')}
        site.update({k: row.get(k, 0) for k in INTARNA_ENERGY_FEATURES})
        # Computed in best_intarna.py, carried through the merge untouched. NaN (not 0)
        # when the column is absent, which _check_subopt_columns has already warned about.
        site.update({k: safe_float(row.get(k), np.nan) for k in SUBOPT_STAT_FEATURES})
        site.update({
            'mre_seq': mre_seqs[i],
            'mirna_seq': mirna_seqs[i],
            'chimeric_sequence': mre_seqs[i] + mirna_seqs[i],
            'conservation_vector': v7_row['conservation_vector'],
            'mir_fam': v7_row.get('noncodingRNA_fam', ''),
            'label': v7_row.get('label', 0),
            # 'mfe_only' where the ensemble run found no interaction at all for this pair,
            # so Eall/Eall1/Eall2/Ealltotal (and E_total/P_E, derived from them) are NaN.
            # Reported, never fed to the model.
            'energy_source': row.get('energy_source', 'ensemble'),
        })
        site.update({c: v7_row.get(c, '') for c in V7_PASSTHROUGH if c in v7.columns})
        sites.append(site)

    print(f"  Valid sites: {len(sites)}")

    # Which feature columns to write. Resolved BEFORE the compute loop, because it is
    # also what the loop is allowed to skip computing (see _resolve_wanted).
    if args.features_file:
        text = open(args.features_file).read()
        features = (json.loads(text) if text.lstrip().startswith('[')
                    else [ln.strip() for ln in text.splitlines() if ln.strip()])
        source = args.features_file
    else:
        features = list(DEFAULT_FEATURES)
        source = (f"DEFAULT_FEATURES ({len(SELECTED_FEATURES)} selected + "
                  f"{len(NEW_CANDIDATE_FEATURES)} candidates)")

    # binding_type is one-hot encoded at selection time, so the list can contain
    # `binding_type_<value>` columns that no computation produces.
    binding_type_cols = [c for c in features if c.startswith('binding_type_')]
    features = [c for c in features if not c.startswith('binding_type_')]

    known = set(all_feature_names())
    unknown = [c for c in features if c not in known]
    if unknown:
        sys.exit(f"ERROR: {len(unknown)} requested feature(s) are not produced by this "
                 f"extractor: {unknown}\nRun --list-features to see the full vocabulary.")

    # Restrict computation to what will actually be written. --no-feature-gating computes
    # everything anyway, which is the reference the gating is verified against.
    want = None if args.no_feature_gating else _resolve_wanted(features)

    print("\n--- Computing features ---")
    print(f"  Feature list: {source}")
    if want is not None:
        total = len(all_feature_names())
        print(f"  Computing {len(features)} of {total} known features "
              f"({total - len(features)} skipped; --no-feature-gating to compute all)")

    compute_all_sites(sites, args.flank_size, hist_range, background, want,
                      processes=args.threads)

    if args.verify_gating and want is not None:
        _verify_gating(sites, features, args.flank_size, hist_range, background)
    elif args.verify_gating:
        print("  --verify-gating is a no-op with gating already off")

    if background:
        matched = sum(1 for s in sites if not np.isnan(s['E_bg_mean_mirna']))
        print(f"  Shuffle background: {matched}/{len(sites)} rows matched a miRNA "
              f"({matched / max(len(sites), 1):.1%})")
        if matched < len(sites):
            print("    Unmatched rows get NaN z-scores. Rebuild the background from a "
                  "miRNA FASTA covering this split if the gap is large.")

    v7_cols = [c for c in V7_PASSTHROUGH if c in v7.columns]
    headers = ID_COLS + v7_cols + features + binding_type_cols + ['label']

    print(f"\n--- Writing {args.output} ---")
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        writer.writeheader()
        for s in sites:
            row = {k: s.get(k, '') for k in ID_COLS + v7_cols}
            row['label'] = s.get('label', 0)
            for feat in features:
                # KeyError, not a 0 default: every feature in `features` is produced
                # unconditionally above, so a miss is a bug, not a missing value.
                # (Genuinely undefined statistics are already NaN - see
                # _energy_series_features.)
                row[feat] = s[feat]
            btype = s.get('binding_type', '')
            for col in binding_type_cols:
                row[col] = 1 if btype == col[len('binding_type_'):] else 0
            writer.writerow(row)

    mode = "from " + ("--features-file" if args.features_file else "DEFAULT_FEATURES")
    print(f"Done. Wrote {len(sites)} rows, {len(features) + len(binding_type_cols)} "
          f"features ({mode}).")

    # Pairs for which the ensemble run found no interaction at all, so the partition-
    # function energies are NaN. Keyed on chimeric_sequence so the set can be joined
    # against any downstream table (it is the same key the training script dedups on).
    fallback = [s for s in sites if s.get('energy_source') == 'mfe_only']
    frac = len(fallback) / len(sites) if sites else 0.0
    print(f"No ensemble interaction: {len(fallback)} / {len(sites)} rows ({frac:.1%})"
          f"{' - see ' + args.fallback_report if args.fallback_report else ''}")
    if len(fallback) and sites:
        pos = sum(1 for s in fallback if safe_int(s.get('label', 0)) == 1)
        base = sum(1 for s in sites if safe_int(s.get('label', 0)) == 1) / len(sites)
        print(f"  positives among them: {pos}/{len(fallback)} ({pos / len(fallback):.1%}) "
              f"vs {base:.1%} overall - a large gap means the NaN pattern is label-biased, "
              f"which the tree models can exploit as a shortcut.")

    if args.fallback_report:
        cols = ['chimeric_sequence', 'target_id', 'query_id', 'mre_sequence',
                'mirna_sequence', 'mir_fam', 'label', 'binding_type',
                'E', 'E_hybrid', 'ED_target']
        with open(args.fallback_report, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter='\t', extrasaction='ignore')
            w.writeheader()
            for s in fallback:
                w.writerow({c: s.get(c, '') for c in cols})
        print(f"  wrote {len(fallback)} rows to {args.fallback_report}")


if __name__ == '__main__':
    main()
