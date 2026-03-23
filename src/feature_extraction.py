#!/usr/bin/env python3
"""
Unified miRNA feature extractor — streaming

Merges feature_extraction.py + seq_features_ml.py into a single pass that:
  - Computes ONLY the features in TARGET_FEATURES
  - One-hot encodes exactly the five binding-type categories in OHE_BINDING_TYPES
  - Writes the CSV incrementally (no intermediate buffering)

Usage:
    python extract_features_unified.py \
        --intarna  best_intarna_results.tsv \
        --mre-fasta mre_sequences.fasta \
        --mirna-fasta mirna_sequences.fasta \
        --conservation conservation_data.tsv \
        --output features.csv \
        [--bigwig phastcons.bw] \
        [--flank-size 10] \
        [--filter yes]
"""

import re
import sys
import csv
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Optional

try:
    import pyBigWig
    HAS_PYBIGWIG = True
except ImportError:
    HAS_PYBIGWIG = False
    print("Warning: pyBigWig not available — BigWig conservation disabled.", file=sys.stderr)


# ─── Turner NN stacking energies (kcal/mol, 37 °C, 1 M NaCl) ─────────────────
DINUC_ENERGY: Dict[str, float] = {
    'AA': -0.93, 'AU': -1.10, 'AC': -2.24, 'AG': -2.08,
    'UA': -1.33, 'UU': -0.93, 'UC': -2.35, 'UG': -1.30,
    'CA': -2.11, 'CU': -2.08, 'CC': -3.26, 'CG': -2.36,
    'GA': -2.35, 'GU': -1.30, 'GC': -3.42, 'GG': -3.26,
}

# ─── Binding types for one-hot encoding (only these five) ────────────────────
OHE_BINDING_TYPES: List[str] = [
    '5mer.mismatch.3prime',
    '6mer.mirna.bulge.3prime',
    '3prime.compensatory',
    '6mer.mismatch.3prime',
    'seedless',
]

# ─── Fixed (non-optional) output columns ────────────────────────────────────
_FIXED_COLS: List[str] = [
    # chimeric_sequence is always written; other identifiers are optional
    'mre_seq', 'mirna_seq', 'chimeric_sequence',
    # IntaRNA energies
    'Eall', 'Eall2', 'ED_query', 'Energy_hybrid_norm',
    # Duplex stats
    'total_matches', 'total_mre_bulges', 'total_gu_wobbles', 'match_fraction',
    # Seed region
    'seed_matches_2_8', 'gu_wobbles_in_seed_2_8_pos', 'seed_au_content',
    # Non-seed / 3'-supplementary region
    'consecutive_matches_minus_seed', 'nonseed_au_content', 'effective_3prime_matches',
    # Conservation
    'seed_conservation_median', 'seed_conservation_max', 'seed_conservation_min',
    'five_prime_flank_conservation_mean', 'three_prime_flank_conservation_mean',
    'conservation_range', 'downstream_gini',
    # miRNA 3' region — thermodynamic + stability + sequence motifs
    'mirna_3p_energy_max_jump', 'mirna_3p_energy_min', 'mirna_3p_YYY_freq',
    'mirna_3p_energy_mean', 'mirna_3p_energy_asymmetry', 'mirna_3p_stability_run_frac',
    'mirna_3p_energy_max', 'mirna_3p_energy_volatility', 'mirna_3p_energy_oscillation',
    'mirna_3p_energy_drift', 'mirna_3p_energy_range', 'mirna_3p_5p_terminal_energy',
    'mirna_3p_energy_std', 'mirna_3p_ggg_count', 'mirna_3p_YYR_freq',
    'mirna_3p_YRY_freq', 'mirna_3p_energy_gradient', 'mirna_3p_3p_terminal_energy',
    # miRNA seed region — thermodynamic subset + one purine-pattern
    'mirna_seed_RRR_freq', 'mirna_seed_energy_range',
    'mirna_seed_energy_asymmetry', 'mirna_seed_energy_min',
    'mirna_seed_energy_mean', 'mirna_seed_5p_terminal_energy',
    'mirna_seed_3p_terminal_energy','mirna_seed_energy_drift',
    # MRE 3' region — thermodynamic subset + trinuc complexity
    'mre_3p_energy_range', 'mre_3p_energy_volatility',
    'mre_3p_energy_max_jump', 'mre_3p_unique_trinuc_ratio', 'mre_3p_energy_mean',
    # MRE 5' region — thermodynamic subset
    'mre_5p_energy_mean', 'mre_5p_energy_range', 'mre_5p_energy_max_jump',
    # Binding type one-hot columns
    *[f'binding_type_{bt}' for bt in OHE_BINDING_TYPES],
]

# Optional columns — included only when the source data actually contains them.
# Order here determines their position in the output (prepended before fixed cols).
_OPTIONAL_COLS: List[str] = ['mir_fam', 'target_id', 'query_id', 'label']


def build_output_cols(
    has_mir_fam: bool,
    has_target_id: bool,
    has_query_id: bool,
    has_label: bool,
) -> List[str]:
    """
    Return the final ordered column list, inserting optional columns only when
    the corresponding source data was found.  chimeric_sequence is always first
    among the identifier-like columns; label (if present) is appended last.
    """
    optional_front = [c for c, flag in [
        ('mir_fam', has_mir_fam),
        ('target_id', has_target_id),
        ('query_id', has_query_id),
    ] if flag]

    label_col = ['label'] if has_label else []

    return optional_front + _FIXED_COLS + label_col


# =============================================================================
# I/O HELPERS
# =============================================================================

def _rna(seq: str) -> str:
    """Normalise to uppercase RNA."""
    return seq.upper().replace('T', 'U')


def parse_fasta(path: str) -> List[str]:
    """Return an ordered list of RNA sequences from a FASTA file."""
    seqs: List[str] = []
    buf: List[str] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('>'):
                if buf:
                    seqs.append(_rna(''.join(buf)))
                buf = []
            else:
                buf.append(line.strip())
    if buf:
        seqs.append(_rna(''.join(buf)))
    return seqs


def stream_intarna_tsv(path: str):
    """Yield one dict per row from an IntaRNA best-results TSV."""
    with open(path, newline='') as fh:
        yield from csv.DictReader(fh, delimiter='\t')


def intarna_tsv_columns(path: str) -> set:
    """Return the set of column names in an IntaRNA TSV without reading all rows."""
    with open(path, newline='') as fh:
        header = fh.readline()
    return set(header.rstrip('\n').split('\t'))


def load_conservation_tsv(path: str, bigwig_path: Optional[str]):
    """
    Load the conservation TSV.

    Returns
    -------
    cons_vecs   : pd.Series  – conservation vectors
    mir_fam     : pd.Series  – family strings (None-filled if column absent)
    labels      : pd.Series  – label values   (None-filled if column absent)
    has_mir_fam : bool       – True iff 'noncodingRNA_fam' existed
    has_label   : bool       – True iff 'label' existed
    """
    df = pd.read_csv(path, sep='\t')

    if 'chr' in df.columns:
        df['chr'] = 'chr' + df['chr'].astype(str)
        df['chr'] = df['chr'].replace({'chrMT': 'chrM'})

    if (bigwig_path and HAS_PYBIGWIG
            and all(c in df.columns for c in ['chr', 'start', 'end', 'strand'])):
        def _bw_vec(row):
            try:
                bw = pyBigWig.open(bigwig_path)
                s, e = int(row['start']) - 1, int(row['end'])
                vals = np.nan_to_num(bw.values(row['chr'], s, e), nan=0.0)
                bw.close()
                return (vals[::-1] if row['strand'] == '-' else vals).tolist()
            except Exception:
                return None
        cons_vecs = df.apply(_bw_vec, axis=1)
        print("Conservation: using BigWig phastCons")
    elif 'conservation_vector' in df.columns:
        cons_vecs = df['conservation_vector']
        print("Conservation: using column 'conservation_vector'")
    elif 'gene_phastCons' in df.columns:
        cons_vecs = df['gene_phastCons']
        print("Conservation: using column 'gene_phastCons'")
    else:
        cons_vecs = pd.Series([None] * len(df))
        print("Conservation: WARNING — no recognised conservation column found", file=sys.stderr)

    has_mir_fam = 'noncodingRNA_fam' in df.columns
    has_label = 'label' in df.columns
    mir_fam = df['noncodingRNA_fam'] if has_mir_fam else pd.Series([None] * len(df))
    labels = df['label']            if has_label   else pd.Series([None] * len(df))
    return cons_vecs, mir_fam, labels, has_mir_fam, has_label


def _parse_cons_vector(raw) -> List[float]:
    """Parse a conservation vector from string / array / None to a float list."""
    if raw is None:
        return []
    if isinstance(raw, (list, np.ndarray)):
        return [0.0 if np.isnan(x) else float(x) for x in raw]
    if isinstance(raw, str) and raw not in ('', 'nan', 'None', 'NaN'):
        cleaned = raw.strip().strip('[]')
        out: List[float] = []
        for tok in cleaned.split(','):
            tok = tok.strip()
            if tok:
                try:
                    out.append(float(tok))
                except ValueError:
                    out.append(0.0)
        return out
    try:
        if pd.isna(raw):
            return []
    except Exception:
        pass
    return []


def safe_float(v, default: float = 0.0) -> float:
    if v is None or v in ('NA', ''):
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _coerce_str(v, default: str = '') -> str:
    """Return v as a string; converts pandas/numpy NA and None → default."""
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    return str(v)


def safe_int(v, default: int = 0) -> int:
    if v is None or v in ('NA', ''):
        return default
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


# =============================================================================
# DUPLEX VECTOR
# =============================================================================

def create_duplex_vector(target_struct: str, query_struct: str) -> str:
    """
    Build duplex vector from dot-bracket structure strings.
    Codes: '1'=match, '2'=mismatch, '3'=MRE bulge, '4'=miRNA bulge,
           'D'=pre-match gap, 'd'=miRNA dangle, 'e'=MRE dangle.
    """
    loop: List[str] = []
    i, j, started = 0, 0, False
    nt, nq = len(target_struct), len(query_struct)
    while i < nt or j < nq:
        tc = target_struct[i] if i < nt else '\0'
        qc = query_struct[j]  if j < nq else '\0'
        if   tc == '(' and qc == ')': loop.append('1'); started = True; i += 1; j += 1
        elif tc == '.' and qc == '.': loop.append('2' if started else 'D'); i += 1; j += 1
        elif tc == '.' and qc == ')': loop.append('3'); i += 1
        elif tc == '(' and qc == '.': loop.append('4'); j += 1
        elif tc == '\0' and qc == '.': loop.append('d'); j += 1
        elif qc == '\0' and tc == '.': loop.append('e'); i += 1
        else: break
    return ''.join(loop)


# =============================================================================
# POSITION TRACKING (miRNA / MRE within duplex vector)
# =============================================================================

def _mirna_pos_map(vec: str) -> Dict[int, Optional[int]]:
    """vector-index → 0-indexed miRNA position within the binding region."""
    m: Dict[int, Optional[int]] = {}
    pos = 0
    for idx, ch in enumerate(vec):
        if ch in '124Dd':
            m[idx] = pos; pos += 1
        elif ch in '3e':
            m[idx] = None
        else:
            m[idx] = pos
    return m


def _mre_pos_map(vec: str) -> Dict[int, Optional[int]]:
    """vector-index → 0-indexed MRE position within the binding region."""
    m: Dict[int, Optional[int]] = {}
    pos = 0
    for idx, ch in enumerate(vec):
        if ch in '123De':
            m[idx] = pos; pos += 1
        elif ch in '4d':
            m[idx] = None
        else:
            m[idx] = pos
    return m


def _vec_indices_for_mirna_range(vec: str, mir_start: int, mir_end: int,
                                  mir_bind_start: int) -> List[int]:
    """Indices in vec whose miRNA full-sequence position is in [mir_start, mir_end)."""
    pm = _mirna_pos_map(vec)
    return [vi for vi, bp in pm.items()
            if bp is not None and mir_start <= bp + mir_bind_start < mir_end]


def _mirna_region_vec(vec: str, mir_start: int, mir_end: int, mir_bind_start: int) -> str:
    return ''.join(vec[i] for i in _vec_indices_for_mirna_range(vec, mir_start, mir_end, mir_bind_start))


def _max_consecutive(text: str, ch: str) -> int:
    runs = re.findall(re.escape(ch) + '+', text)
    return max((len(r) for r in runs), default=0)


def _count_consec_matches_in_range(vec: str, mir_start: int, mir_end: int,
                                    mir_bind_start: int) -> int:
    return _max_consecutive(_mirna_region_vec(vec, mir_start, mir_end, mir_bind_start), '1')

# =============================================================================
# PAIRED BASES (for GU wobbles + base composition)
# =============================================================================

def _paired_bases(vec: str, mre_bind_seq: str, mir_bind_seq: str,
                  mre_bind_start: int, mir_bind_start: int):
    """
    Return (paired_mre_str, paired_mir_str, mirna_positions_0indexed_in_full).
    """
    pm_mre: List[str] = []
    pm_mir: List[str] = []
    mir_pos: List[int] = []
    mp = qp = 0
    for ch in vec:
        if mp >= len(mre_bind_seq) or qp >= len(mir_bind_seq):
            break
        if ch == '1':
            pm_mre.append(mre_bind_seq[mp])
            pm_mir.append(mir_bind_seq[qp])
            mir_pos.append(qp + mir_bind_start)
        if ch in '123De':
            mp += 1
        if ch in '124Dd':
            qp += 1
    return ''.join(pm_mre), ''.join(pm_mir), mir_pos


# =============================================================================
# CONSERVATION — SEED MRE POSITIONS
# =============================================================================

def _seed_mre_positions(site: dict, mir_bind_start: int) -> List[int]:
    """Map miRNA seed positions (1-7, 1-indexed) to MRE indices in the 50-nt window."""
    vec = site['total_vec']
    mre_bind_end = site['mre_coord_end'] - 1   # 0-indexed
    pmir = _mirna_pos_map(vec)
    pmre = _mre_pos_map(vec)
    out: List[int] = []
    for vi in range(len(vec)):
        mbp = pmir.get(vi)
        mrep = pmre.get(vi)
        if mbp is not None and mrep is not None:
            pos_full_1idx = mbp + mir_bind_start + 1   # 1-indexed full-miRNA position
            if 1 <= pos_full_1idx <= 7:
                orig_mre_idx = mre_bind_end - mrep     # position in 50-nt MRE window
                if 0 <= orig_mre_idx < 50:
                    out.append(orig_mre_idx)
    return out


# =============================================================================
# CONSERVATION FEATURES  (trimmed: only the 7 features we need)
# =============================================================================

def _gini(arr: np.ndarray) -> float:
    """Gini coefficient"""
    n = len(arr)
    if n < 2:
        return 0.0
    s = np.sort(arr)
    total = s.sum()
    if total == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    g = float((2.0 * np.dot(idx, s)) / (n * total) - (n + 1) / n)
    return 0.0 if np.isnan(g) else g


_CONS_ZEROS: Dict[str, float] = {
    'seed_conservation_median': 0.0,
    'seed_conservation_max': 0.0,
    'seed_conservation_min': 0.0,
    'five_prime_flank_conservation_mean': 0.0,
    'three_prime_flank_conservation_mean': 0.0,
    'conservation_range': 0.0,
    'downstream_gini': 0.0,
}


def extract_conservation_features(site: dict, mir_bind_start: int,
                                   flank_size: int = 10) -> Dict[str, float]:
    """Compute only the 7 conservation features present in the target list."""
    scores = _parse_cons_vector(site.get('conservation_vector'))
    if not scores or len(scores) != 50:
        return dict(_CONS_ZEROS)

    arr = np.array(scores, dtype=float)
    n = len(scores)
    t2 = 2 * (n // 3)
    ds = arr[t2:]

    # Features that don't need seed positions
    feats: Dict[str, float] = {
        'conservation_range': round(float(arr.max() - arr.min()), 4),
        'downstream_gini': round(_gini(ds), 4) if len(ds) >= 2 else 0.0,
    }

    seed_pos = [p for p in _seed_mre_positions(site, mir_bind_start) if 0 <= p < 50]
    if not seed_pos:
        feats.update({
            'seed_conservation_median': 0.0,
            'seed_conservation_max': 0.0,
            'seed_conservation_min': 0.0,
            'five_prime_flank_conservation_mean': 0.0,
            'three_prime_flank_conservation_mean': 0.0,
        })
        return feats

    seed_scores = [scores[p] for p in seed_pos]
    seed_start = min(seed_pos)
    seed_end = max(seed_pos)

    feats['seed_conservation_median'] = round(float(np.median(seed_scores)), 4)
    feats['seed_conservation_max'] = round(float(np.max(seed_scores)), 4)
    feats['seed_conservation_min'] = round(float(np.min(seed_scores)), 4)

    five_p  = scores[max(0, seed_start - flank_size):seed_start]
    three_p = scores[seed_end + 1:min(50, seed_end + 1 + flank_size)]
    feats['five_prime_flank_conservation_mean']  = round(float(np.mean(five_p)),  4) if five_p  else 0.0
    feats['three_prime_flank_conservation_mean'] = round(float(np.mean(three_p)), 4) if three_p else 0.0

    return feats


# =============================================================================
# SEQUENCE FEATURES  (targeted: only the features present in the target list)
# =============================================================================

def _dinuc_energies(seq: str) -> List[float]:
    return [DINUC_ENERGY.get(seq[i:i+2], -1.5) for i in range(len(seq) - 1)]


def _energy_stats(energies: List[float], prefix: str) -> Dict[str, float]:
    """
    Thermodynamic + stability features for one sequence region.
    Computes all sub-features needed by any region in our target list.
    """
    feats: Dict[str, float] = {}
    n = len(energies)
    _zero_keys = (
        'energy_mean', 'energy_std', 'energy_min', 'energy_max', 'energy_range',
        'energy_asymmetry', 'energy_gradient',
        'energy_volatility', 'energy_max_jump', 'stability_run_frac',
        'energy_oscillation', 'energy_drift',
        '5p_terminal_energy', '3p_terminal_energy',
    )
    if n == 0:
        return {f'{prefix}_{k}': 0.0 for k in _zero_keys}

    ea = np.array(energies, dtype=float)
    feats[f'{prefix}_energy_mean'] = float(ea.mean())
    feats[f'{prefix}_energy_min'] = float(ea.min())
    feats[f'{prefix}_energy_max'] = float(ea.max())
    feats[f'{prefix}_energy_range'] = float(ea.max() - ea.min())
    feats[f'{prefix}_energy_std'] = float(ea.std()) if n > 1 else 0.0

    # Asymmetry: mean of 5'-half vs 3'-half energies
    if n >= 4:
        h = n // 2
        feats[f'{prefix}_energy_asymmetry'] = float(ea[:h].mean() - ea[h:].mean())
    else:
        feats[f'{prefix}_energy_asymmetry'] = 0.0

    # Gradient via Pearson correlation of position index vs energy
    if n >= 3:
        x = np.arange(n, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mat = np.corrcoef(x, ea)
        val = float(mat[0, 1]) if mat.shape == (2, 2) else 0.0
        feats[f'{prefix}_energy_gradient'] = 0.0 if np.isnan(val) else val
    else:
        feats[f'{prefix}_energy_gradient'] = 0.0

    # Terminal energies
    feats[f'{prefix}_5p_terminal_energy'] = float(ea[0])
    feats[f'{prefix}_3p_terminal_energy'] = float(ea[-1])

    # Stability dynamics (need ≥ 2 dinucleotides)
    if n >= 2:
        diffs = np.diff(ea)
        absdiffs = np.abs(diffs)
        feats[f'{prefix}_energy_volatility'] = float(absdiffs.mean())
        feats[f'{prefix}_energy_max_jump'] = float(absdiffs.max())
        feats[f'{prefix}_stability_run_frac'] = float((absdiffs < 0.3).sum() / len(diffs))
        signs = np.sign(diffs)
        nz    = signs[signs != 0]
        if len(nz) > 1:
            feats[f'{prefix}_energy_oscillation'] = float(
                (np.abs(np.diff(nz)) > 0).sum() / (len(nz) - 1))
        else:
            feats[f'{prefix}_energy_oscillation'] = 0.0
        feats[f'{prefix}_energy_drift'] = float(ea[-1] - ea[0])
    else:
        for k in ('energy_volatility', 'energy_max_jump', 'stability_run_frac',
                  'energy_oscillation', 'energy_drift'):
            feats[f'{prefix}_{k}'] = 0.0

    return feats


def _pur_freq(seq: str, pattern: str, n_trinuc: int) -> float:
    """Frequency of a YR-coded trinucleotide pattern (e.g. 'YYY', 'RRR')."""
    if n_trinuc < 1 or len(seq) < 3:
        return 0.0
    pur_seq = ''.join('R' if nt in 'AG' else 'Y' for nt in seq)
    count   = sum(1 for i in range(len(pur_seq) - 2) if pur_seq[i:i+3] == pattern)
    return count / n_trinuc


def extract_seq_features(mre_seq: str, mirna_seq: str) -> Dict[str, float]:
    """
    Compute only the sequence features that appear in the target feature list.
    Covers four regions: mirna_3p, mirna_seed, mre_3p, mre_5p.
    """
    mre_seq   = _rna(mre_seq)   if mre_seq   else ''
    mirna_seq = _rna(mirna_seq) if mirna_seq else ''

    # ── Region splits ─────────────────────────────────────────────────────────
    # miRNA seed: positions 2-8 (1-indexed) = indices [1:8] (0-indexed)
    if   len(mirna_seq) >= 8: mir_seed = mirna_seq[1:8]
    elif len(mirna_seq) >  1: mir_seed = mirna_seq[1:]
    else:                     mir_seed = mirna_seq

    # miRNA 3': everything after the seed
    if   len(mirna_seq) >  8: mir_3p = mirna_seq[8:]
    elif len(mirna_seq) >= 3: mir_3p = mirna_seq[-min(3, len(mirna_seq)):]
    else:                     mir_3p = mirna_seq

    # MRE halves
    mid = max(len(mre_seq) // 2, 1)
    mre_5p = mre_seq[:mid]
    mre_3p = mre_seq[mid:] if len(mre_seq) > mid else mre_seq

    feats: Dict[str, float] = {}

    # ── mirna_3p ──────────────────────────────────────────────────────────────
    pref = 'mirna_3p'
    if len(mir_3p) >= 2:
        e = _dinuc_energies(mir_3p)
        feats.update(_energy_stats(e, pref))
        nt = max(len(mir_3p) - 2, 0)
        feats[f'{pref}_YYY_freq'] = _pur_freq(mir_3p, 'YYY', nt)
        feats[f'{pref}_YYR_freq'] = _pur_freq(mir_3p, 'YYR', nt)
        feats[f'{pref}_YRY_freq'] = _pur_freq(mir_3p, 'YRY', nt)
        feats[f'{pref}_ggg_count'] = float(mir_3p.count('GGG'))
    else:
        feats.update({k: 0.0 for k in _energy_stats([], pref)})
        feats[f'{pref}_YYY_freq'] = 0.0
        feats[f'{pref}_YYR_freq'] = 0.0
        feats[f'{pref}_YRY_freq'] = 0.0
        feats[f'{pref}_ggg_count'] = 0.0

    # ── mirna_seed ────────────────────────────────────────────────────────────
    # Only the subset of energy stats + RRR_freq needed for this region.
    pref = 'mirna_seed'
    _SEED_ENERGY_KEYS = (
        'energy_range', 'energy_asymmetry', 'energy_min', 'energy_mean',
        '5p_terminal_energy', '3p_terminal_energy', 'energy_drift',
    )
    if len(mir_seed) >= 2:
        eseed = _dinuc_energies(mir_seed)
        _all  = _energy_stats(eseed, pref)
        for k in _SEED_ENERGY_KEYS:
            feats[f'{pref}_{k}'] = _all.get(f'{pref}_{k}', 0.0)
        nt = max(len(mir_seed) - 2, 0)
        feats[f'{pref}_RRR_freq'] = _pur_freq(mir_seed, 'RRR', nt)
    else:
        for k in _SEED_ENERGY_KEYS:
            feats[f'{pref}_{k}'] = 0.0
        feats[f'{pref}_RRR_freq'] = 0.0

    # ── mre_3p ────────────────────────────────────────────────────────────────
    # Only energy subset + unique_trinuc_ratio.
    pref = 'mre_3p'
    _MRE3_ENERGY_KEYS = ('energy_range', 'energy_volatility', 'energy_max_jump', 'energy_mean')
    if len(mre_3p) >= 2:
        e3 = _dinuc_energies(mre_3p)
        _all = _energy_stats(e3, pref)
        for k in _MRE3_ENERGY_KEYS:
            feats[f'{pref}_{k}'] = _all.get(f'{pref}_{k}', 0.0)
        nt = max(len(mre_3p) - 2, 0)
        tc = Counter(mre_3p[i:i+3] for i in range(nt))
        feats[f'{pref}_unique_trinuc_ratio'] = len(tc) / min(nt, 64) if nt > 0 else 0.0
    else:
        for k in _MRE3_ENERGY_KEYS:
            feats[f'{pref}_{k}'] = 0.0
        feats[f'{pref}_unique_trinuc_ratio'] = 0.0

    # ── mre_5p ────────────────────────────────────────────────────────────────
    pref = 'mre_5p'
    _MRE5_ENERGY_KEYS = ('energy_mean', 'energy_range', 'energy_max_jump')
    if len(mre_5p) >= 2:
        e5  = _dinuc_energies(mre_5p)
        _all = _energy_stats(e5, pref)
        for k in _MRE5_ENERGY_KEYS:
            feats[f'{pref}_{k}'] = _all.get(f'{pref}_{k}', 0.0)
    else:
        for k in _MRE5_ENERGY_KEYS:
            feats[f'{pref}_{k}'] = 0.0

    return feats


# =============================================================================
# BINDING TYPE CLASSIFICATION  (logic unchanged from feature_extraction.py)
# =============================================================================

def classify_binding_type(d: dict) -> str:
    consecutive = d.get('consecutive_matches_seed', 0)
    btype = f"{consecutive}mer" if consecutive >= 5 else 'seedless'

    if d.get('start_match_seed', 99) >= 4:
        btype = 'seedless'
    if d.get('total_matches_in_seed_9_pos', 0) == 9:
        btype = '9mer'

    if btype == 'seedless':
        mbst = d.get('mirna_coord_start', 1) - 1
        if _count_consec_matches_in_range(d['total_vec'], 4, 16, mbst) >= 8:
            return 'centered'
        if d.get('start_match_seed', 99) >= 13:
            return '3prime'
        if d.get('effective_3prime_matches', 0) >= 6:
            return '3prime.compensatory'
        return btype

    mirna_seq  = d.get('mirna_seq', '')
    start_match = d.get('start_match_seed', 99)

    if btype == '8mer' and start_match == 2 and mirna_seq and mirna_seq[0] == 'A':
        btype = '8mer1A'
    elif btype == '7mer' and start_match == 2 and mirna_seq and mirna_seq[0] == 'A':
        btype = '8mer1A'
    elif btype == '6mer':
        if   start_match == 3:
            btype = 'offset6mer'
        elif start_match == 2 and mirna_seq and mirna_seq[0] == 'A':
            btype = '7mer1A'

    if 'mer' in btype:
        if d.get('mismatch_seed_positions', 0) > 0: btype += '.mismatch'
        if d.get('seed_target_bulge_positions', 0) > 0: btype += '.target.bulge'
        if d.get('seed_mirna_bulge_positions', 0) > 0: btype += '.mirna.bulge'
        if d.get('gu_wobbles_in_seed', 0) > 0: btype += '.GU'
        if d.get('effective_3prime_matches', 0) >= 3: btype += '.3prime'

    return btype


# =============================================================================
# CORE EXTRACTION  (one site at a time)
# =============================================================================

_AU_PAIRS = [{'A', 'U'}, {'A', 'T'}, {'U', 'A'}, {'T', 'A'}]
_GU_PAIRS = [{'G', 'U'}, {'G', 'T'}]


def _is_au(a: str, b: str) -> bool:
    return {a, b} in _AU_PAIRS


def _is_gu(a: str, b: str) -> bool:
    return {a, b} in _GU_PAIRS


def extract_all_features(site: dict, flank_size: int = 10) -> dict:
    """
    Compute every required feature for *one* site dict.
    Mutates site in place and returns it (avoids copying the dict).
    """
    # ── Parse structures and subsequences ─────────────────────────────────────
    hybrid_dp = site.get('hybrid_dp', '')
    if '&' in hybrid_dp:
        mre_struct, mirna_struct = hybrid_dp.split('&', 1)
    else:
        mre_struct = mirna_struct = ''

    subseq_dp = site.get('subseq_dp', '')
    if '&' in subseq_dp:
        parts = subseq_dp.split('&', 1)
        mre_bind_subseq = _rna(parts[0])
        mirna_bind_subseq = _rna(parts[1])
    else:
        mre_bind_subseq = mirna_bind_subseq = ''

    # ── Coordinates (IntaRNA 1-indexed → 0-indexed internally) ───────────────
    mre_c_start = safe_int(site.get('start_target', site.get('mre_coord_start', 1)))
    mre_c_end = safe_int(site.get('end_target', site.get('mre_coord_end',   1)))
    mir_c_start = safe_int(site.get('start_query', site.get('mirna_coord_start', 1)))
    mir_c_end = safe_int(site.get('end_query', site.get('mirna_coord_end',   1)))

    site['mre_coord_start'] = mre_c_start
    site['mre_coord_end'] = mre_c_end
    site['mirna_coord_start'] = mir_c_start
    site['mirna_coord_end'] = mir_c_end

    mir_bind_start = mir_c_start - 1   # 0-indexed

    # ── IntaRNA energies (only required ones) ─────────────────────────────────
    site['Eall'] = safe_float(site.get('Eall'))
    site['Eall2'] = safe_float(site.get('Eall2'))
    site['ED_query'] = safe_float(site.get('ED_query', site.get('ED2')))
    site['Energy_hybrid_norm'] = safe_float(
        site.get('Energy_hybrid_norm', site.get('energy_hybrid_norm')))

    # ── Duplex vector ─────────────────────────────────────────────────────────
    # IntaRNA target (MRE) structure is stored 5'→3'; reverse for alignment.
    vec = create_duplex_vector(mre_struct[::-1], mirna_struct)
    site['total_vec'] = vec

    site['total_matches'] = vec.count('1')
    site['total_mre_bulges'] = vec.count('3')
    site['match_fraction'] = round(site['total_matches'] / len(vec), 4) if vec else 0.0

    # ── Seed region vectors ───────────────────────────────────────────────────
    sv2_8 = _mirna_region_vec(vec, 1, 8, mir_bind_start)   # positions 2-8
    sv0_9 = _mirna_region_vec(vec, 0, 9, mir_bind_start)   # positions 1-9 (for 9mer check)
    sv0_8 = _mirna_region_vec(vec, 0, 8, mir_bind_start)   # positions 1-8

    site['seed_matches_2_8'] = sv2_8.count('1')
    site['consecutive_matches_seed'] = _max_consecutive(sv2_8, '1')
    site['total_matches_in_seed_9_pos'] = sv0_9.count('1')
    site['mismatch_seed_positions'] = sv0_8.count('2')
    site['seed_target_bulge_positions'] = sv0_8.count('3')
    site['seed_mirna_bulge_positions']  = sv0_8.count('4')

    try:
        site['start_match_seed'] = mirna_struct.index(')') + 1
    except ValueError:
        site['start_match_seed'] = 99

    # ── Non-seed (3' supplementary) region ───────────────────────────────────
    ns_indices = _vec_indices_for_mirna_range(vec, 8, 999, mir_bind_start)
    ns_vec = ''.join(vec[i] for i in ns_indices)
    site['consecutive_matches_minus_seed'] = _max_consecutive(ns_vec, '1')

    # ── Paired bases: GU wobbles + base composition ───────────────────────────
    # MRE binding sequence is stored 5'→3' but the duplex vector was built on
    # the reversed MRE structure, so reverse the subsequence for alignment.
    mre_bind_rev = mre_bind_subseq[::-1]
    pm_mre, pm_mir, mir_pos = _paired_bases(
        vec, mre_bind_rev, mirna_bind_subseq,
        mre_c_start - 1, mir_bind_start,
    )

    # Seed pairs: 1-indexed full-miRNA positions 1-7  (0-indexed 0-6 internally)
    seed_pairs = [(a, b) for a, b, p in zip(pm_mre, pm_mir, mir_pos) if 1 <= p <= 7]
    nonseed_pairs = [(a, b) for a, b, p in zip(pm_mre, pm_mir, mir_pos) if p > 7]

    # Seed AU content
    if seed_pairs:
        au = sum(1 for a, b in seed_pairs if _is_au(a, b))
        site['seed_au_content'] = round(au / len(seed_pairs), 4)
    else:
        site['seed_au_content'] = 0.0

    # GU wobbles in seed (gu_wobbles_in_seed_2_8_pos = alias in original)
    gu_seed = sum(1 for a, b in seed_pairs if _is_gu(a, b))
    site['gu_wobbles_in_seed'] = gu_seed
    site['gu_wobbles_in_seed_2_8_pos'] = gu_seed

    # Non-seed AU content
    if nonseed_pairs:
        au_ns = sum(1 for a, b in nonseed_pairs if _is_au(a, b))
        site['nonseed_au_content'] = round(au_ns / len(nonseed_pairs), 4)
    else:
        site['nonseed_au_content'] = 0.0

    gu_ns = sum(1 for a, b in nonseed_pairs if _is_gu(a, b))
    site['total_gu_wobbles'] = gu_seed + gu_ns
    site['effective_3prime_matches'] = len(nonseed_pairs) - gu_ns

    # ── Conservation ─────────────────────────────────────────────────────────
    site.update(extract_conservation_features(site, mir_bind_start, flank_size))

    # ── Sequence features ─────────────────────────────────────────────────────
    site.update(extract_seq_features(site.get('mre_seq', ''), site.get('mirna_seq', '')))

    # ── Binding type + one-hot encoding ──────────────────────────────────────
    btype = classify_binding_type(site)
    site['binding_type'] = btype
    for bt in OHE_BINDING_TYPES:
        site[f'binding_type_{bt}'] = int(btype == bt)

    return site


# =============================================================================
# OPTIONAL HEURISTIC FILTER  (same logic as original filter_sites_detailed)
# =============================================================================

def passes_filter(site: dict) -> bool:
    btype = site.get('binding_type', '')
    eff3p = site.get('effective_3prime_matches', 0)
    if '5mer' in btype and eff3p <= 5:
        return False
    if ('seedless' in btype or '3prime.compensatory' in btype) and eff3p <= 7:
        return False
    if btype == '3prime' and site.get('consecutive_matches_minus_seed', 0) < 6:
        return False
    if site.get('seed_mirna_bulge_positions', 0) > 2:
        return False
    return True


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Unified miRNA feature extractor — streaming, trimmed-computation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python extract_features_unified.py \\
      --intarna  best_intarna_results.tsv \\
      --mre-fasta mre_sequences.fasta \\
      --mirna-fasta mirna_sequences.fasta \\
      --conservation conservation_data.tsv \\
      --output features.csv
        """,
    )
    ap.add_argument('--intarna', required=True, help='IntaRNA best-results TSV')
    ap.add_argument('--mre-fasta', required=True, help='MRE FASTA (50 nt per entry)')
    ap.add_argument('--mirna-fasta', required=True, help='miRNA FASTA')
    ap.add_argument('--conservation', required=True, help='Conservation TSV')
    ap.add_argument('--output', required=True, help='Output CSV path')
    ap.add_argument('--bigwig', default=None, help='phastCons BigWig (optional)')
    ap.add_argument('--flank-size', type=int, default=10,
                    help='Flank size for conservation features (default: 10)')
    ap.add_argument('--filter',choices=['yes', 'y', 'no', 'n'], default='no',
                    help='Apply heuristic quality filter (default: no)')
    args = ap.parse_args()

    apply_filter = args.filter.lower() in ('y', 'yes')

    # ── Load files that require random access ─────────────────────────────────
    print("--- Reading input files ---")
    mre_seqs = parse_fasta(args.mre_fasta)
    mirna_seqs = parse_fasta(args.mirna_fasta)
    print(f"MRE sequences : {len(mre_seqs):,}")
    print(f"miRNA sequences : {len(mirna_seqs):,}")

    cons_vecs, mir_fam, labels, has_mir_fam, has_label = \
        load_conservation_tsv(args.conservation, args.bigwig)
    print(f"Conservation rows: {len(cons_vecs):,}")
    print(f"mir_fam column: {'found' if has_mir_fam else 'absent — column will be omitted'}")
    print(f"label column: {'found' if has_label   else 'absent — column will be omitted'}")

    # Validate counts
    n_mre, n_mir, n_con = len(mre_seqs), len(mirna_seqs), len(cons_vecs)
    if not (n_mre == n_mir == n_con):
        print(f"\nERROR: count mismatch — MRE={n_mre}, miRNA={n_mir}, conservation={n_con}",
              file=sys.stderr)
        sys.exit(1)
    print("  All file sizes match.\n")

    # ── Streaming extraction + CSV write ──────────────────────────────────────
    print(f"--- Streaming feature extraction → {args.output} ---")
    written = skipped = 0

    # Detect whether target_id / query_id exist in the IntaRNA TSV header
    intarna_header = intarna_tsv_columns(args.intarna)
    has_target_id = 'target_id' in intarna_header
    has_query_id = 'query_id'  in intarna_header
    if not has_target_id: print("  target_id column  : absent — column will be omitted")
    if not has_query_id: print("  query_id column   : absent — column will be omitted")

    output_cols = build_output_cols(has_mir_fam, has_target_id, has_query_id, has_label)
    print(f"\nOutput columns: {len(output_cols)}")

    with open(args.output, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=output_cols, extrasaction='ignore')
        writer.writeheader()

        for i, row in enumerate(stream_intarna_tsv(args.intarna)):
            # Skip absent interactions (row index still advances for FASTA alignment)
            if row.get('status') == 'no_interactions' or row.get('start_target') in ('NA', '', None):
                skipped += 1
                continue

            if i >= n_mre or i >= n_mir or i >= n_con:
                print(f"Warning: index {i} out of range — skipping", file=sys.stderr)
                skipped += 1
                continue

            # Build minimal site dict for this row
            site: dict = {
                'mre_seq': mre_seqs[i],
                'mirna_seq': mirna_seqs[i],
                'chimeric_sequence': mre_seqs[i] + mirna_seqs[i],
                'conservation_vector': (cons_vecs.iloc[i]
                                        if hasattr(cons_vecs, 'iloc') else cons_vecs[i]),
                # Optional identifier / label fields — only populated when present
                **({'target_id': row.get('target_id', '')} if has_target_id else {}),
                **({'query_id': row.get('query_id',  '')} if has_query_id  else {}),
                **({'mir_fam': (mir_fam.iloc[i] if hasattr(mir_fam, 'iloc') else mir_fam[i])}
                   if has_mir_fam else {}),
                **({'label': (labels.iloc[i] if hasattr(labels, 'iloc') else labels[i])}
                   if has_label else {}),
                # IntaRNA structure / coordinate fields
                'hybrid_dp': row.get('hybrid_dp', ''),
                'subseq_dp': row.get('subseq_dp', ''),
                'start_target': row.get('start_target'),
                'end_target': row.get('end_target'),
                'start_query': row.get('start_query'),
                'end_query': row.get('end_query'),
                # Raw energy fields (multiple naming conventions from IntaRNA versions)
                'Eall': row.get('Eall'),
                'Eall2': row.get('Eall2'),
                'ED_query': row.get('ED_query', row.get('ED2')),
                'Energy_hybrid_norm': row.get('Energy_hybrid_norm'),
                'energy_hybrid_norm': row.get('Energy_hybrid_norm'),
            }

            try:
                extract_all_features(site, flank_size=args.flank_size)
            except Exception as exc:
                print(f"Warning: row {i} failed ({exc}) — skipping", file=sys.stderr)
                skipped += 1
                continue

            if apply_filter and not passes_filter(site):
                skipped += 1
                continue

            writer.writerow(site)
            written += 1

            if written % 5_000 == 0:
                print(f"{written:,} rows written…", flush=True)

    print(f"\nDone.")
    print(f"Written : {written:,} rows")
    print(f"Skipped : {skipped}")
    print(f"Output  : {args.output}")
    print(f"Columns : {len(output_cols)}")


if __name__ == '__main__':
    main()
