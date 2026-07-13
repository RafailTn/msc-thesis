#!/usr/bin/env python3
"""
Extract a fixed, pre-selected feature set for miRNA-MRE interactions.

Combines logic from `classify_and_filter_sites_intarna.py` (IntaRNA + duplex +
conservation features and binding-type classification) and
`seq_features_ml.py` (region-specific sequence features), but writes ONLY the
features in SELECTED_FEATURES to the output CSV.

Stacking energies are derived from the IntaRNA duplex, not from the single
strands: see `duplex_energy_steps`.

Usage:
    python extract_selected_features.py \
        --intarna best_intarna_results.tsv \
        --mre-fasta mre_sequences.fasta \
        --mirna-fasta mirna_sequences.fasta \
        --conservation conservation_data.tsv \
        --output features.csv
"""

import os
import re
import sys
import csv
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
# SELECTED FEATURES (final output columns, in addition to identifiers + label)
# ============================================================================

SELECTED_FEATURES = [
    'gu_wobbles_in_seed_2_8_pos', 'mirna_3p_energy_max_jump',
    'mirna_3p_energy_max', 'mirna_3p_YRR_freq', 'mirna_3p_energy_std',
    'binding_type_6mer.mismatch.3prime', 'mirna_3p_stability_run_frac',
    'mirna_3p_energy_gradient', 'mirna_3p_ggg_count',
    'five_prime_flank_conservation_mean', 'mirna_seed_energy_mean',
    'mirna_3p_YYR_freq', 'mirna_3p_energy_min', 'mirna_seed_RRY_freq',
    'mirna_3p_energy_volatility',
    'seed_matches_2_8', 'total_gu_wobbles', 'binding_type_seedless',
    'mirna_seed_energy_range', 'seed_conservation_min',
    'mirna_3p_energy_oscillation', 'Eall',
    'seed_conservation_median', 'mirna_3p_3p_terminal_energy',
    'mirna_3p_RYY_freq',
    'binding_type_6mer.mirna.bulge.3prime', 'mirna_3p_YYY_freq',
    'consecutive_matches_minus_seed', 'Eall1',
    'total_mre_bulges', 'mirna_seed_5p_terminal_energy',
    'mre_3p_unique_trinuc_ratio', 'mirna_seed_energy_volatility',
    'mirna_3p_energy_mean', 'downstream_max_consecutive_drop',
    'three_prime_flank_conservation_mean', 'mirna_3p_YRY_freq',
    'seed_au_content', 'mirna_3p_energy_drift', 'downstream_skewness',
    'binding_type_3prime.compensatory', 'mirna_seed_energy_min',
    'seed_conservation_max', 'total_matches', 'mirna_3p_energy_asymmetry',
    'mirna_seed_energy_asymmetry', 'mirna_3p_5p_terminal_energy',
    'downstream_gini', 'effective_3prime_matches', 'mirna_seed_RRR_freq',
    'binding_type_5mer.mismatch.3prime',
    'conservation_range',
]

BINDING_TYPE_COLS = [c for c in SELECTED_FEATURES if c.startswith('binding_type_')]
BINDING_TYPE_VALUES = [c[len('binding_type_'):] for c in BINDING_TYPE_COLS]


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
    results = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            results.append(row)
    return results


def get_phastcons_vector(bw_path, chrom, start, end, strand):
    if not HAS_PYBIGWIG:
        return None
    try:
        bw = pyBigWig.open(bw_path)
        start, end = int(start) - 1, int(end)
        scores = bw.values(chrom, start, end)
        scores = np.nan_to_num(scores, nan=0.0)
        if strand == '-':
            scores = scores[::-1]
        bw.close()
        return scores.tolist()
    except (RuntimeError, ValueError):
        return None


def parse_conservation_tsv(file_path, bigwig_path: Optional[str]):
    df = pd.read_csv(file_path, sep='\t')

    if 'chr' in df.columns:
        df['chr'] = "chr" + df['chr'].astype(str)
        df['chr'] = df['chr'].replace({'chrMT': 'chrM'})

    if bigwig_path and HAS_PYBIGWIG and all(c in df.columns for c in ['chr', 'start', 'end', 'strand']):
        df['gene_phastCons470'] = df.apply(
            lambda row: get_phastcons_vector(
                bigwig_path, row['chr'], row['start'], row['end'], row['strand']
            ), axis=1)
        cons_vecs = df['gene_phastCons470']
    elif 'conservation_vector' in df.columns:
        cons_vecs = df['conservation_vector']
    elif 'gene_phastCons' in df.columns:
        cons_vecs = df['gene_phastCons']
    else:
        cons_vecs = pd.Series([None] * len(df))

    mir_fam = df['noncodingRNA_fam'] if 'noncodingRNA_fam' in df.columns else pd.Series([''] * len(df))
    labels = df['label'] if 'label' in df.columns else pd.Series([0] * len(df))
    return cons_vecs, mir_fam, labels


def parse_conservation_scores(cons_vector):
    if cons_vector is None:
        return []
    if isinstance(cons_vector, (list, np.ndarray)):
        return [float(x) if not np.isnan(x) else 0.0 for x in cons_vector]
    if isinstance(cons_vector, str) and cons_vector not in ('', 'nan', 'None', 'NaN'):
        cleaned = cons_vector.strip().strip('[]')
        scores = []
        for s in cleaned.split(','):
            s = s.strip()
            if s:
                try:
                    scores.append(float(s))
                except ValueError:
                    scores.append(0.0)
        return scores
    try:
        if pd.isna(cons_vector):
            return []
    except Exception:
        pass
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
    region_chars = [total_vec[i] for i in indices]
    max_c, curr_c = 0, 0
    for char in region_chars:
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
    mre_ptr = 0
    mir_ptr = 0
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
                if 0 <= original_mre_idx < 50:
                    seed_mre_positions.append(original_mre_idx)
    return seed_mre_positions


# ============================================================================
# CONSERVATION VECTOR SHAPE FEATURES
# ============================================================================

_SHAPE_KEYS = [
    'skewness', 'kurtosis', 'entropy', 'gini', 'slope', 'roughness',
    'max_consecutive_drop', 'max_consecutive_rise',
]


def _compute_vector_shape_features(scores: list, prefix: str) -> dict:
    n = len(scores)
    if n < 2:
        return {f"{prefix}_{k}": 0.0 for k in _SHAPE_KEYS}

    arr = np.array(scores, dtype=float)
    mean = np.mean(arr)
    std = np.std(arr)
    out = {}

    if std > 0:
        out[f"{prefix}_skewness"] = round(float(np.mean(((arr - mean) / std) ** 3)), 4)
        out[f"{prefix}_kurtosis"] = round(float(np.mean(((arr - mean) / std) ** 4) - 3.0), 4)
    else:
        out[f"{prefix}_skewness"] = 0.0
        out[f"{prefix}_kurtosis"] = 0.0

    counts, _ = np.histogram(arr, bins=10, range=(0.0, 1.0))
    s = counts.sum()
    if s > 0:
        probs = counts / s
        probs = probs[probs > 0]
        out[f"{prefix}_entropy"] = round(float(-np.sum(probs * np.log2(probs))), 4)
    else:
        out[f"{prefix}_entropy"] = 0.0

    sorted_arr = np.sort(arr)
    idx = np.arange(1, n + 1)
    denom = n * sorted_arr.sum()
    if denom > 0:
        out[f"{prefix}_gini"] = round(
            float((2 * np.sum(idx * sorted_arr)) / denom - (n + 1) / n), 4)
    else:
        out[f"{prefix}_gini"] = 0.0

    positions = np.arange(1, n + 1, dtype=float)
    pos_mean = positions.mean()
    num = np.sum((positions - pos_mean) * (arr - mean))
    den = np.sum((positions - pos_mean) ** 2)
    out[f"{prefix}_slope"] = round(float(num / den) if den > 0 else 0.0, 6)

    diffs = np.abs(np.diff(arr))
    out[f"{prefix}_roughness"] = round(float(np.mean(diffs)), 4)

    signed_diffs = np.diff(arr)
    out[f"{prefix}_max_consecutive_drop"] = round(
        float(np.max(signed_diffs * -1)) if len(signed_diffs) > 0 else 0.0, 4)
    out[f"{prefix}_max_consecutive_rise"] = round(
        float(np.max(signed_diffs)) if len(signed_diffs) > 0 else 0.0, 4)

    return out


def extract_conservation_features(site_data, mirna_binding_start, flank_size=10):
    """Compute only the conservation features required by SELECTED_FEATURES."""
    keys_needed = [
        'seed_conservation_min', 'seed_conservation_median', 'seed_conservation_max',
        'five_prime_flank_conservation_mean', 'three_prime_flank_conservation_mean',
        'conservation_range',
        'downstream_max_consecutive_drop', 'downstream_skewness', 'downstream_gini',
    ]
    features = {k: 0.0 for k in keys_needed}

    scores = parse_conservation_scores(site_data.get('conservation_vector'))
    if not scores or len(scores) != 50:
        return features

    seed_mre_positions = [p for p in get_mre_positions_for_seed(site_data, mirna_binding_start)
                          if 0 <= p < 50]

    if seed_mre_positions:
        seed_start, seed_end = min(seed_mre_positions), max(seed_mre_positions)
        seed_scores = [scores[i] for i in seed_mre_positions]
        if seed_scores:
            features['seed_conservation_median'] = round(np.median(seed_scores), 4)
            features['seed_conservation_max'] = round(np.max(seed_scores), 4)
            features['seed_conservation_min'] = round(np.min(seed_scores), 4)

        five_prime_flank_scores = scores[max(0, seed_start - flank_size):seed_start]
        features['five_prime_flank_conservation_mean'] = round(
            np.mean(five_prime_flank_scores), 4) if five_prime_flank_scores else 0.0

        three_prime_flank_scores = scores[seed_end + 1:min(50, seed_end + 1 + flank_size)]
        features['three_prime_flank_conservation_mean'] = round(
            np.mean(three_prime_flank_scores), 4) if three_prime_flank_scores else 0.0

    features['conservation_range'] = round(np.max(scores) - np.min(scores), 4)

    # Downstream tertile of conservation vector
    n = len(scores)
    t2 = 2 * (n // 3)
    downstream_scores = scores[t2:n]
    if downstream_scores:
        shape = _compute_vector_shape_features(downstream_scores, prefix='downstream')
        for k in ('downstream_max_consecutive_drop', 'downstream_skewness', 'downstream_gini'):
            features[k] = shape.get(k, 0.0)

    return features


# ============================================================================
# SEQUENCE REGION FEATURES (subset from seq_features_ml.py)
# ============================================================================

DEFAULT_VALUE = 0.0

# miRNA seed = positions 2-8 (1-based, inclusive); the "3p" region is 9 -> 3' end.
SEED_FIRST_POS = 2
SEED_LAST_POS = 8


def safe_divide(num, den, default=DEFAULT_VALUE):
    if den == 0 or np.isnan(den) or np.isinf(den):
        return default
    r = num / den
    if np.isnan(r) or np.isinf(r):
        return default
    return r


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
    ones -- they are already carried by the Eall / Eall1 features -- so summing
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


def _energy_series_features(energies, region_name: str, needed: set) -> Dict[str, float]:
    """Reduce an ordered energy series to the per-region `needed` features.

    A statistic that is *undefined* for the series at hand is emitted as NaN, not
    0.0: the standard deviation of a single stack is undefined, not zero, and an
    absent region has no energy profile at all. Using 0.0 would place "undefined"
    mid-distribution (the energies span roughly -3.4 to +4.5), where no single tree
    split can isolate it, and would make an unpaired region indistinguishable from
    a paired one whose terms happen to cancel. NaN keeps it off the numeric axis,
    which the gradient-boosted learners route explicitly.
    """
    out = {f'{region_name}_{f}': np.nan for f in needed}
    n = len(energies)
    if n == 0:
        return out

    # Defined for any non-empty series.
    if 'energy_mean' in needed:
        out[f'{region_name}_energy_mean'] = float(np.mean(energies))
    if 'energy_min' in needed:
        out[f'{region_name}_energy_min'] = float(np.min(energies))
    if 'energy_max' in needed:
        out[f'{region_name}_energy_max'] = float(np.max(energies))
    if 'energy_range' in needed:
        out[f'{region_name}_energy_range'] = float(np.max(energies) - np.min(energies))
    if '5p_terminal_energy' in needed:
        out[f'{region_name}_5p_terminal_energy'] = float(energies[0])
    if '3p_terminal_energy' in needed:
        out[f'{region_name}_3p_terminal_energy'] = float(energies[-1])

    # Need a spread: >= 2 stacks.
    if n >= 2:
        if 'energy_std' in needed:
            out[f'{region_name}_energy_std'] = float(np.std(energies))

        changes = np.diff(energies)
        if 'energy_volatility' in needed:
            out[f'{region_name}_energy_volatility'] = float(np.mean(np.abs(changes)))
        if 'energy_max_jump' in needed:
            out[f'{region_name}_energy_max_jump'] = float(np.max(np.abs(changes)))
        if 'energy_drift' in needed:
            out[f'{region_name}_energy_drift'] = float(energies[-1] - energies[0])
        if 'stability_run_frac' in needed:
            stable_count = int(np.sum(np.abs(changes) < 0.3))
            out[f'{region_name}_stability_run_frac'] = float(stable_count) / len(changes)

        # Oscillation needs at least two non-flat changes to have a direction to reverse.
        if 'energy_oscillation' in needed:
            signs = np.sign(changes)
            nonzero_signs = signs[signs != 0]
            if len(nonzero_signs) > 1:
                direction_changes = int(np.sum(np.abs(np.diff(nonzero_signs)) > 0))
                out[f'{region_name}_energy_oscillation'] = (
                    float(direction_changes) / (len(nonzero_signs) - 1))

    # Need a trend: >= 3 stacks. Undefined (and numpy-nan) for a constant series.
    if 'energy_gradient' in needed and n >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cm = np.corrcoef(np.arange(n), energies)
        corr = cm[0, 1] if cm.shape == (2, 2) else np.nan
        out[f'{region_name}_energy_gradient'] = float(corr)

    # Need two halves to compare: >= 4 stacks.
    if 'energy_asymmetry' in needed and n >= 4:
        half = n // 2
        out[f'{region_name}_energy_asymmetry'] = float(
            np.mean(energies[:half]) - np.mean(energies[half:]))

    return out


def extract_duplex_energy_features(site_data) -> Dict[str, float]:
    """Duplex stacking-energy features for the miRNA seed and 3' regions.

    A loop is assigned to a region only when *both* of the base pairs it lies
    between fall inside that region, which reproduces the region boundaries of the
    old single-strand dinucleotide windows (a step straddling position 8/9 belongs
    to neither). A region with fewer than two base pairs has no loops, so all its
    energy features are NaN -- for `mirna_3p` that is the common and meaningful
    case "no 3' supplementary pairing".
    """
    steps = duplex_energy_steps(site_data)

    seed_e, three_p_e = [], []
    for p5, p3, dg in steps:
        if SEED_FIRST_POS <= p5 and p3 <= SEED_LAST_POS:
            seed_e.append(dg)
        elif p5 > SEED_LAST_POS:
            three_p_e.append(dg)

    out = {}
    out.update(_energy_series_features(seed_e, 'mirna_seed',
                                       DUPLEX_ENERGY_FEATURE_SUBSET['mirna_seed']))
    out.update(_energy_series_features(three_p_e, 'mirna_3p',
                                       DUPLEX_ENERGY_FEATURE_SUBSET['mirna_3p']))
    return out


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


# Energy features: computed from the IntaRNA duplex (see extract_duplex_energy_features).
DUPLEX_ENERGY_FEATURE_SUBSET = {
    'mirna_seed': {'energy_mean', 'energy_range', 'energy_volatility',
                   '5p_terminal_energy', 'energy_min', 'energy_asymmetry'},
    'mirna_3p':   {'energy_max_jump', 'energy_max', 'energy_std',
                   'stability_run_frac', 'energy_gradient', 'energy_min',
                   'energy_volatility', 'energy_oscillation', '3p_terminal_energy',
                   'energy_drift', 'energy_mean', 'energy_asymmetry',
                   '5p_terminal_energy'},
}

# Composition features: properties of a single strand, so they stay sequence-based.
REGION_SEQ_FEATURE_SUBSET = {
    'mre_3p':     {'unique_trinuc_ratio'},
    'mirna_seed': {'RRY_freq', 'RRR_freq'},
    'mirna_3p':   {'YRR_freq', 'YYR_freq', 'RYY_freq', 'YYY_freq', 'YRY_freq',
                   'ggg_count'},
}


def extract_region_features(seq: str, region_name: str, needed: set) -> Dict[str, float]:
    """Compute only the per-region composition features in `needed`. Keys returned
    are `{region_name}_{feature}`."""
    out = {f'{region_name}_{f}': DEFAULT_VALUE for f in needed}
    if 'ggg_count' in needed:
        out[f'{region_name}_ggg_count'] = 0

    if not seq or len(seq) < 3:
        return out

    n_trinuc = len(seq) - 2
    if n_trinuc < 1:
        return out

    # Purine/pyrimidine trinuc frequencies
    pur_keys_needed = {f for f in needed if f.endswith('_freq')}
    if pur_keys_needed:
        pur_seq = ''.join('R' if nt in 'AG' else 'Y' for nt in seq)
        pur_trinucs = Counter(pur_seq[i:i+3] for i in range(len(pur_seq) - 2))
        for f in pur_keys_needed:
            pattern = f[:-len('_freq')]
            out[f'{region_name}_{f}'] = safe_divide(pur_trinucs.get(pattern, 0), n_trinuc)

    # Trinucleotide unique ratio
    if 'unique_trinuc_ratio' in needed:
        trinucs = [seq[i:i+3] for i in range(n_trinuc)]
        out[f'{region_name}_unique_trinuc_ratio'] = safe_divide(
            len(Counter(trinucs)), min(n_trinuc, 64))

    # Motif counts
    if 'ggg_count' in needed:
        out[f'{region_name}_ggg_count'] = seq.count('GGG')

    return out


def extract_sequence_region_features(mre_seq: str, mirna_seq: str) -> Dict[str, float]:
    out = {}
    mre_regions = get_regions(mre_seq, is_mirna=False) if mre_seq else {'5p': '', '3p': ''}
    mirna_regions = get_regions(mirna_seq, is_mirna=True) if mirna_seq else {'seed': '', '3p': ''}

    out.update(extract_region_features(mre_regions.get('3p', ''), 'mre_3p',
                                        REGION_SEQ_FEATURE_SUBSET['mre_3p']))
    out.update(extract_region_features(mirna_regions.get('seed', ''), 'mirna_seed',
                                        REGION_SEQ_FEATURE_SUBSET['mirna_seed']))
    out.update(extract_region_features(mirna_regions.get('3p', ''), 'mirna_3p',
                                        REGION_SEQ_FEATURE_SUBSET['mirna_3p']))
    return out


# ============================================================================
# HELPERS
# ============================================================================

def get_max_consecutive_char(text, char):
    runs = re.findall(f"{re.escape(char)}+", text)
    return max((len(s) for s in runs), default=0)


def safe_float(value, default=0.0):
    if value is None or value == 'NA' or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    if value is None or value == 'NA' or value == '':
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


# ============================================================================
# IntaRNA / DUPLEX FEATURE EXTRACTION (subset)
# ============================================================================

def calculate_intarna_subset_features(site_data, flank_size=10):
    hybrid_dp = site_data.get('hybrid_dp', '')
    if '&' in hybrid_dp:
        mre_struct, mirna_struct = hybrid_dp.split('&', 1)
    else:
        mre_struct, mirna_struct = '', ''

    subseq_dp = site_data.get('subseq_dp', '')
    if '&' in subseq_dp:
        a, b = subseq_dp.split('&', 1)
        mre_binding_subseq = a.upper().replace('T', 'U')
        mirna_binding_subseq = b.upper().replace('T', 'U')
    else:
        mre_binding_subseq = ''
        mirna_binding_subseq = ''

    mre_coord_start = safe_int(site_data.get('start_target', 1))
    mre_coord_end = safe_int(site_data.get('end_target', 1))
    mirna_coord_start = safe_int(site_data.get('start_query', 1))

    site_data['mre_coord_start'] = mre_coord_start
    site_data['mre_coord_end'] = mre_coord_end
    site_data['mirna_coord_start'] = mirna_coord_start

    mre_binding_start = mre_coord_start - 1
    mirna_binding_start = mirna_coord_start - 1

    # Energies (only Eall, Eall1 needed in output, but used in classification too)
    site_data['Eall'] = safe_float(site_data.get('Eall', 0))
    site_data['Eall1'] = safe_float(site_data.get('Eall1', 0))

    # Duplex vector (target reversed for miRNA-mRNA)
    total_vec = create_duplex_vectors(mre_struct[::-1], mirna_struct)
    site_data['total_vec'] = total_vec

    site_data['total_matches'] = total_vec.count('1')
    site_data['total_mre_bulges'] = total_vec.count('3')
    site_data['total_mirna_bulges'] = total_vec.count('4')
    site_data['total_bulges'] = site_data['total_mre_bulges'] + site_data['total_mirna_bulges']
    site_data['total_mismatches'] = total_vec.count('2')

    # Seed analysis (miRNA positions 1-8 in 0-indexed: [0,8))
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

    # Non-seed
    non_seed_indices = get_vector_indices_for_mirna_range(total_vec, 8, 100, mirna_binding_start)
    non_seed_vec = ''.join(total_vec[i] for i in non_seed_indices)
    site_data['consecutive_matches_minus_seed'] = get_max_consecutive_char(non_seed_vec, '1')
    site_data['non_seed_matches'] = non_seed_vec.count('1')

    try:
        site_data['start_match_seed'] = mirna_struct.index(')') + 1
    except ValueError:
        site_data['start_match_seed'] = 99

    # Paired bases for G:U + AU content
    mre_binding_seq_rev = mre_binding_subseq[::-1]
    all_paired_mre, all_paired_mir, mirna_positions = get_paired_bases_with_positions(
        total_vec, mre_binding_seq_rev, mirna_binding_subseq,
        mre_binding_start, mirna_binding_start
    )

    seed_pairs = [(mre, mir) for mre, mir, pos in zip(all_paired_mre, all_paired_mir, mirna_positions)
                  if 1 <= pos <= 7]
    if seed_pairs:
        au_count = sum(1 for mre, mir in seed_pairs
                       if {mre, mir} in [{'A', 'U'}, {'A', 'T'}, {'U', 'A'}, {'T', 'A'}])
        site_data['seed_au_content'] = round(au_count / len(seed_pairs), 4)
    else:
        site_data['seed_au_content'] = 0.0

    paired_mre_seed = ''.join(p[0] for p in seed_pairs)
    paired_mir_seed = ''.join(p[1] for p in seed_pairs)
    site_data['gu_wobbles_in_seed'] = sum(
        1 for i in range(len(paired_mre_seed))
        if {paired_mre_seed[i], paired_mir_seed[i]} in [{'G', 'U'}, {'G', 'T'}]
    )
    site_data['gu_wobbles_in_seed_2_8_pos'] = site_data['gu_wobbles_in_seed']

    nonseed_pairs = [(mre, mir) for mre, mir, pos in zip(all_paired_mre, all_paired_mir, mirna_positions)
                     if pos > 7]
    paired_mre_ns = ''.join(p[0] for p in nonseed_pairs)
    paired_mir_ns = ''.join(p[1] for p in nonseed_pairs)
    site_data['total_matches_minus_seed'] = len(paired_mre_ns)
    site_data['gu_wobbles_minus_seed'] = sum(
        1 for i in range(len(paired_mre_ns))
        if {paired_mre_ns[i], paired_mir_ns[i]} in [{'G', 'U'}, {'G', 'T'}]
    )
    site_data['total_gu_wobbles'] = site_data['gu_wobbles_in_seed'] + site_data['gu_wobbles_minus_seed']

    # effective_3prime_matches
    site_data['effective_3prime_matches'] = (site_data['total_matches_minus_seed']
                                             - site_data['gu_wobbles_minus_seed'])

    # Duplex nearest-neighbour stacking energies (miRNA seed / 3' regions)
    site_data.update(extract_duplex_energy_features(site_data))

    # Conservation features
    site_data.update(extract_conservation_features(site_data, mirna_binding_start, flank_size))

    return site_data


# ============================================================================
# BINDING TYPE CLASSIFICATION (verbatim from classify_and_filter_sites_intarna.py)
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
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract a fixed selected feature set for miRNA-MRE pairs.'
    )
    parser.add_argument('--intarna', required=True)
    parser.add_argument('--mre-fasta', required=True)
    parser.add_argument('--mirna-fasta', required=True)
    parser.add_argument('--conservation', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--flank-size', type=int, default=10)
    parser.add_argument('--bigwig', default=None)
    args = parser.parse_args()

    print("--- Reading inputs ---")
    intarna_results = parse_intarna_results(args.intarna)
    mre_seqs = parse_fasta(args.mre_fasta)
    mirna_seqs = parse_fasta(args.mirna_fasta)
    cons_vecs, mir_fam, labels = parse_conservation_tsv(args.conservation, args.bigwig)

    counts = {
        "IntaRNA": len(intarna_results), "MRE": len(mre_seqs),
        "miRNA": len(mirna_seqs), "Conservation": len(cons_vecs),
    }
    for name, count in counts.items():
        print(f"  {name}: {count}")
    if len(set(counts.values())) != 1:
        print("ERROR: Entry count mismatch")
        sys.exit(1)

    print("\n--- Building site data ---")
    sites = []
    for i, row in enumerate(intarna_results):
        if row.get('status') == 'no_interactions':
            continue
        if row.get('start_target') in ('NA', '', None):
            continue
        sites.append({
            'target_id': row.get('target_id', ''),
            'query_id': row.get('query_id', ''),
            'start_target': row.get('start_target'),
            'end_target': row.get('end_target'),
            'start_query': row.get('start_query'),
            'end_query': row.get('end_query'),
            'subseq_dp': row.get('subseq_dp', ''),
            'hybrid_dp': row.get('hybrid_dp', ''),
            'Eall': row.get('Eall'),
            'Eall1': row.get('Eall1'),
            'mre_seq': mre_seqs[i],
            'mirna_seq': mirna_seqs[i],
            'conservation_vector': cons_vecs.iloc[i] if hasattr(cons_vecs, 'iloc') else cons_vecs[i],
            'mir_fam': mir_fam.iloc[i] if hasattr(mir_fam, 'iloc') else mir_fam[i],
            'label': labels.iloc[i] if hasattr(labels, 'iloc') else labels[i],
        })

    print(f"  Valid sites: {len(sites)}")

    print("\n--- Computing features ---")
    for s in sites:
        calculate_intarna_subset_features(s, flank_size=args.flank_size)
        s['binding_type'] = classify_binding_type_detailed(s)
        s.update(extract_sequence_region_features(s.get('mre_seq', ''), s.get('mirna_seq', '')))

    print(f"\n--- Writing {args.output} ---")
    headers = (['target_id', 'query_id', 'binding_type',
                'mre_sequence', 'mirna_sequence', 'mir_fam']
               + SELECTED_FEATURES + ['label'])

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        writer.writeheader()
        for s in sites:
            row = {
                'target_id': s.get('target_id', ''),
                'query_id': s.get('query_id', ''),
                'binding_type': s.get('binding_type', ''),
                'mre_sequence': s.get('mre_seq', ''),
                'mirna_sequence': s.get('mirna_seq', ''),
                'mir_fam': s.get('mir_fam', ''),
                'label': s.get('label', 0),
            }
            # One-hot encode the requested binding_type values
            btype = s.get('binding_type', '')
            for col, val in zip(BINDING_TYPE_COLS, BINDING_TYPE_VALUES):
                row[col] = 1 if btype == val else 0

            for feat in SELECTED_FEATURES:
                if feat in row:
                    continue
                row[feat] = s.get(feat, 0)
            writer.writerow(row)

    print(f"Done. Wrote {len(sites)} rows with {len(SELECTED_FEATURES)} features.")


if __name__ == '__main__':
    main()
