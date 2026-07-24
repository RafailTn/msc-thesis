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

Two output modes:

  * default        - writes only SELECTED_FEATURES. This is what training and
                     `predict_target.py` consume.
  * --all-features - writes the full superset. This is the input to
                     `feature_selection_featurewiz/feature_selection.py`.

Both modes run the identical computation; they differ only in which columns are
written, so a feature cannot mean one thing during selection and another during
training.

Usage:
    # superset, for feature selection
    python feature_extraction.py --intarna best.tsv --mre-fasta mre.fa \
        --mirna-fasta mirna.fa --v7 data/..._train_v7.tsv \
        --output train_all.csv --all-features

    # selected features only, for training / inference
    python feature_extraction.py --intarna best.tsv --mre-fasta mre.fa \
        --mirna-fasta mirna.fa --v7 data/..._train_v7.tsv \
        --output train_selected.csv
"""

import re
import sys
import csv
import json
import argparse
import warnings
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
# The output of `feature_selection.py` (the intersection of the featurewiz
# selections across the 5 folds). Regenerate it with --all-features -> featurewiz,
# then paste the new list here.
#
# STALE as of the duplex-energy + phyloP rewrite: the `mirna_*_energy_*` names
# survive but now mean something different, and the `*_gini` conservation features
# no longer exist (Gini is undefined on signed phyloP - see _CONS_SHAPE_KEYS).
# Until feature selection is rerun, only --all-features is trustworthy.

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


def _compute_vector_shape_features(scores, prefix: str, hist_range) -> dict:
    n = len(scores)
    if n < 2:
        return {f"{prefix}_{k}": DEFAULT_VALUE for k in _CONS_SHAPE_KEYS}

    arr = np.asarray(scores, dtype=float)
    mean, std = float(np.mean(arr)), float(np.std(arr))
    out = {}

    if std > 0:
        out[f"{prefix}_skewness"] = round(float(np.mean(((arr - mean) / std) ** 3)), 4)
        out[f"{prefix}_kurtosis"] = round(float(np.mean(((arr - mean) / std) ** 4) - 3.0), 4)
    else:
        out[f"{prefix}_skewness"] = DEFAULT_VALUE
        out[f"{prefix}_kurtosis"] = DEFAULT_VALUE

    # Fixed histogram range keeps entropy comparable across sites; phyloP tails are
    # clipped into the outermost bins rather than dropped.
    counts, _ = np.histogram(np.clip(arr, hist_range[0], hist_range[1]),
                             bins=10, range=hist_range)
    s = counts.sum()
    if s > 0:
        probs = counts / s
        probs = probs[probs > 0]
        out[f"{prefix}_entropy"] = round(float(-np.sum(probs * np.log2(probs))), 4)
    else:
        out[f"{prefix}_entropy"] = DEFAULT_VALUE

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
                                  hist_range=PHYLOP_HIST_RANGE) -> Dict[str, float]:
    """Full conservation superset: seed/flank summaries + shape over 4 windows."""
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

    # Shape over the seed sub-vector and the three positional tertiles.
    n = len(scores)
    t1, t2 = n // 3, 2 * (n // 3)
    for window_scores, prefix in [
        (seed_scores, 'seed'),
        (scores[0:t1], 'upstream'),
        (scores[t1:t2], 'central'),
        (scores[t2:n], 'downstream'),
    ]:
        if window_scores:
            features.update(_compute_vector_shape_features(window_scores, prefix, hist_range))

    return features


# ============================================================================
# DUPLEX NEAREST-NEIGHBOUR STACKING ENERGIES
# ============================================================================

_ENERGY_KEYS = [
    'energy_mean', 'energy_std', 'energy_min', 'energy_max', 'energy_range',
    'energy_asymmetry', 'energy_gradient', 'energy_volatility', 'energy_max_jump',
    'stability_run_frac', 'energy_oscillation', 'energy_drift',
    '5p_terminal_energy', '3p_terminal_energy',
]

# Energies are a property of the *duplex*, so they exist only on the miRNA regions
# (the axis the duplex is walked along). The MRE regions keep composition only.
_ENERGY_REGIONS = ['mirna_seed', 'mirna_3p']


def duplex_energy_feature_names() -> List[str]:
    return [f'{r}_{k}' for r in _ENERGY_REGIONS for k in _ENERGY_KEYS]


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
    """
    out = {f'{region_name}_{k}': np.nan for k in _ENERGY_KEYS}
    n = len(energies)
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


def extract_duplex_energy_features(site_data) -> Dict[str, float]:
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
    seed_e, three_p_e = [], []
    for p5, p3, dg in duplex_energy_steps(site_data):
        if SEED_FIRST_POS <= p5 and p3 <= SEED_LAST_POS:
            seed_e.append(dg)
        elif p5 > SEED_LAST_POS:
            three_p_e.append(dg)

    out = {}
    out.update(_energy_series_features(seed_e, 'mirna_seed'))
    out.update(_energy_series_features(three_p_e, 'mirna_3p'))
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


def extract_sequence_region_features(mre_seq: str, mirna_seq: str) -> Dict[str, float]:
    mre_regions = get_regions(mre_seq) if mre_seq else {'5p': '', '3p': ''}
    mirna_regions = get_regions(mirna_seq, is_mirna=True) if mirna_seq else {'seed': '', '3p': ''}

    out = {}
    out.update(extract_region_features(mre_regions.get('5p', ''), 'mre_5p'))
    out.update(extract_region_features(mre_regions.get('3p', ''), 'mre_3p'))
    out.update(extract_region_features(mirna_regions.get('seed', ''), 'mirna_seed'))
    out.update(extract_region_features(mirna_regions.get('3p', ''), 'mirna_3p'))
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

DUPLEX_STAT_FEATURES = [
    'total_matches', 'total_mismatches', 'total_bulges', 'total_mre_bulges',
    'total_mirna_bulges', 'total_gu_wobbles', 'interaction_length', 'match_fraction',
    'seed_matches', 'seed_matches_2_8', 'consecutive_matches_seed',
    'gu_wobbles_in_seed_2_8_pos', 'effective_seed_matches',
    'non_seed_matches', 'consecutive_matches_minus_seed', 'gu_wobbles_minus_seed',
    'effective_3prime_matches', 'priority_score',
    'seed_au_content', 'seed_gc_content', 'nonseed_au_content', 'nonseed_gc_content',
]


def calculate_intarna_features(site_data, flank_size=10, hist_range=PHYLOP_HIST_RANGE):
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

    site_data.update(extract_duplex_energy_features(site_data))
    site_data.update(extract_conservation_features(
        site_data, mirna_binding_start, flank_size, hist_range))

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
    """The full numeric superset, in a stable order. Input to feature selection."""
    return (INTARNA_ENERGY_FEATURES
            + DUPLEX_STAT_FEATURES
            + conservation_feature_names()
            + composition_feature_names()
            + duplex_energy_feature_names())


# Identifier / passthrough columns, written in both modes. `energy_source` rides along so
# the MFE fallback is auditable from the feature CSV itself; it is never a model input
# (both feature_selection.py and gluon_training_kfold.py drop it).
ID_COLS = ['target_id', 'query_id', 'binding_type', 'hybrid_dp', 'subseq_dp',
           'mre_sequence', 'mirna_sequence', 'chimeric_sequence', 'mir_fam',
           'energy_source']


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract features for miRNA-MRE pairs from an IntaRNA duplex.')
    parser.add_argument('--intarna', required=True, help='best_intarna results TSV')
    parser.add_argument('--mre-fasta', required=True)
    parser.add_argument('--mirna-fasta', required=True)
    parser.add_argument('--v7', required=True,
                        help='v7 TSV: conservation vector, family, label, coordinates')
    parser.add_argument('--output', required=True)
    parser.add_argument('--all-features', action='store_true',
                        help='write the full superset (input to feature selection) '
                             'instead of only SELECTED_FEATURES')
    parser.add_argument('--features-file', default=None,
                        help='JSON/newline list of features to write, overriding '
                             'SELECTED_FEATURES (e.g. a fresh feature_selection.py run)')
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
    parser.add_argument('--flank-size', type=int, default=10)
    parser.add_argument('--bigwig', default=None,
                        help='read conservation from a bigwig instead of the v7 column')
    args = parser.parse_args()

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

    print("\n--- Computing features ---")
    for s in sites:
        calculate_intarna_features(s, flank_size=args.flank_size, hist_range=hist_range)
        s['binding_type'] = classify_binding_type_detailed(s)
        s.update(extract_sequence_region_features(s.get('mre_seq', ''), s.get('mirna_seq', '')))
        s['mre_sequence'] = s['mre_seq']
        s['mirna_sequence'] = s['mirna_seq']

    # Which feature columns to write.
    if args.all_features:
        features = all_feature_names()
        binding_type_cols = []
    else:
        if args.features_file:
            text = open(args.features_file).read()
            features = (json.loads(text) if text.lstrip().startswith('[')
                        else [ln.strip() for ln in text.splitlines() if ln.strip()])
        else:
            features = list(SELECTED_FEATURES)
        # binding_type is one-hot encoded at selection time, so the selected list can
        # contain `binding_type_<value>` columns that no computation produces.
        binding_type_cols = [c for c in features if c.startswith('binding_type_')]
        features = [c for c in features if not c.startswith('binding_type_')]

        known = set(all_feature_names())
        unknown = [c for c in features if c not in known]
        if unknown:
            sys.exit(f"ERROR: {len(unknown)} selected feature(s) are not produced by this "
                     f"extractor: {unknown}\nRerun feature selection against --all-features.")

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

    mode = "full superset" if args.all_features else "selected"
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
