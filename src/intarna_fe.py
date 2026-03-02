#!/usr/bin/env python3
"""
Feature Extraction for IntaRNA-based miRNA Target Prediction

This script extracts detailed features from IntaRNA interaction predictions,
including energy components, duplex structure features, conservation scores,
and local secondary structure context.

Adapted from RNAduplex feature extraction to work with IntaRNA output format.

Usage:
    python extract_intarna_features.py \
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
import numpy as np
import pandas as pd

# Optional: pyBigWig for conservation scores
try:
    import pyBigWig
    HAS_PYBIGWIG = True
except ImportError:
    HAS_PYBIGWIG = False
    print("Warning: pyBigWig not available. Conservation features from BigWig disabled.", 
          file=sys.stderr)


# ============================================================================
# DUPLEX VECTOR FUNCTIONS (same as in intarna_parallel.py)
# ============================================================================

def create_duplex_vectors(target_struct: str, query_struct: str) -> str:
    """
    Create duplex vector representation from structure strings.
    
    Vector codes:
        '1' = Match (base pair)
        '2' = Mismatch (both unpaired, after match started)
        '3' = Target bulge (target unpaired, query paired)
        '4' = Query bulge (query unpaired, target paired)
        'D' = Dangling (both unpaired, before match starts)
        'd' = Query dangling end
        'e' = Target dangling end
    """
    loop = []
    i, j = 0, 0
    match_started = False
    
    while i < len(target_struct) or j < len(query_struct):
        target_char = target_struct[i] if i < len(target_struct) else '\0'
        query_char = query_struct[j] if j < len(query_struct) else '\0'
        
        if target_char == '(' and query_char == ')':
            loop.append('1')
            match_started = True
            i += 1
            j += 1
        elif target_char == '.' and query_char == '.':
            loop.append('2' if match_started else 'D')
            i += 1
            j += 1
        elif target_char == '.' and query_char == ')':
            loop.append('3')
            i += 1
        elif target_char == '(' and query_char == '.':
            loop.append('4')
            j += 1
        elif target_char == '\0' and query_char == '.':
            loop.append('d')
            j += 1
        elif query_char == '\0' and target_char == '.':
            loop.append('e')
            i += 1
        else:
            break
    
    return "".join(loop)


# ============================================================================
# FASTA AND INPUT PARSING
# ============================================================================

def parse_fasta(file_path):
    """Parse FASTA file and return list of sequences (maintaining order)."""
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


def parse_fasta_with_ids(file_path):
    """Parse FASTA file and return dict of id -> sequence."""
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq).upper().replace('T', 'U')
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq).upper().replace('T', 'U')
    
    return sequences


def parse_contrafold_file(file_path):
    """Parse CONTRAfold output format."""
    structures = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip() == '>structure' and (i + 1) < len(lines):
                structures.append(lines[i + 1].strip())
    return structures


def parse_intarna_results(file_path):
    """Parse IntaRNA best results TSV file."""
    results = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            results.append(row)
    return results


# ============================================================================
# CONSERVATION FUNCTIONS
# ============================================================================

def get_phastcons_vector(bw_path, chrom, start, end, strand):
    """Retrieve phastCons score vector for a genomic region."""
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


def parse_conservation_tsv(file_path, bigwig_path='/home/adam/adam/data/hg38.phastCons470way.bw'):
    """Read conservation TSV and extract vectors."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        
        # DEBUG: Print available columns
        print(f"  Conservation TSV columns: {list(df.columns)}")
        
        # Handle chromosome naming
        if 'chr' in df.columns:
            df['chr'] = "chr" + df['chr'].astype(str)
            df['chr'] = df['chr'].replace({'chrMT': 'chrM'})
        
        # Try to get phastCons if bigwig path provided
        if bigwig_path and HAS_PYBIGWIG and all(c in df.columns for c in ['chr', 'start', 'end', 'strand']):
            df['gene_phastCons470'] = df.apply(
                lambda row: get_phastcons_vector(
                    bigwig_path, row['chr'], row['start'], row['end'], row['strand']
                ), axis=1)
            cons_vecs = df['gene_phastCons470']
            print("  Using: BigWig phastCons")
        elif 'conservation_vector' in df.columns:
            cons_vecs = df['conservation_vector']
            print("  Using column: conservation_vector")
        elif 'gene_phastCons470' in df.columns:
            cons_vecs = df['gene_phastCons470']
            print("  Using column: gene_phastCons470")
        else:
            cons_vecs = pd.Series([None] * len(df))
            print("  WARNING: No conservation column found!")
        
        # DEBUG: Print sample of conservation data
        if len(cons_vecs) > 0:
            sample_val = cons_vecs.iloc[0]
            print(f"  Sample conservation value type: {type(sample_val)}")
            if isinstance(sample_val, str):
                print(f"  Sample value (first 100 chars): {sample_val[:100]}...")
            else:
                print(f"  Sample value: {sample_val}")
        
        # Get optional columns
        mir_fam = df['noncodingRNA_fam'] if 'noncodingRNA_fam' in df.columns else pd.Series([''] * len(df))
        labels = df['label'] if 'label' in df.columns else pd.Series([0] * len(df))
        
        return cons_vecs, mir_fam, labels
        
    except (FileNotFoundError, KeyError) as e:
        print(f"Error processing conservation file '{file_path}': {e}")
        sys.exit(1)


def parse_conservation_scores(cons_vector):
    """Parse conservation vector from various formats."""
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
    # Try pandas NA check
    try:
        if pd.isna(cons_vector):
            return []
    except:
        pass
    return []


# ============================================================================
# POSITION TRACKING FUNCTIONS
# ============================================================================

def get_mirna_position_map(total_vec):
    """Returns mapping: vector_index -> miRNA_position (0-indexed within binding region)."""
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
    """Returns mapping: vector_index -> MRE_position (0-indexed within binding region)."""
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
    """Get vector indices for a miRNA position range in the FULL miRNA."""
    position_map = get_mirna_position_map(total_vec)
    indices = []
    
    for vec_idx, mirna_pos_in_binding in position_map.items():
        if mirna_pos_in_binding is not None:
            mirna_pos_in_full = mirna_pos_in_binding + mirna_binding_start
            if mirna_start <= mirna_pos_in_full < mirna_end:
                indices.append(vec_idx)
    
    return indices


def count_consecutive_matches_in_mirna_region(total_vec, mirna_start, mirna_end, mirna_binding_start):
    """Count max consecutive matches in miRNA position range."""
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
    """Count occurrences of character in miRNA position range."""
    indices = get_vector_indices_for_mirna_range(total_vec, mirna_start, mirna_end, mirna_binding_start)
    return [total_vec[i] for i in indices].count(target_char) if indices else 0


def get_mirna_region_vector(total_vec, mirna_start, mirna_end, mirna_binding_start):
    """Extract portion of total_vec for miRNA position range in FULL miRNA."""
    indices = get_vector_indices_for_mirna_range(total_vec, mirna_start, mirna_end, mirna_binding_start)
    return ''.join(total_vec[i] for i in indices) if indices else ""


# ============================================================================
# PAIRED BASES EXTRACTION
# ============================================================================

def get_paired_bases_with_positions(total_vec, mre_binding_seq, mirna_binding_seq,
                                     mre_binding_start, mirna_binding_start):
    """
    Extract paired bases and track positions in the FULL sequences.
    
    Returns:
        tuple: (paired_mre_str, paired_mir_str, mirna_positions_in_full_seq)
    """
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


# ============================================================================
# CONSERVATION FEATURE EXTRACTION
# ============================================================================

def get_mre_positions_for_seed(site_data, mirna_binding_start):
    """Map miRNA seed positions to MRE positions in full 50nt sequence."""
    total_vec = site_data['total_vec']
    
    mre_binding_start = site_data['mre_coord_start'] - 1
    mre_binding_end = site_data['mre_coord_end'] - 1
    
    mirna_pos_map = get_mirna_position_map(total_vec)
    mre_pos_map = get_mre_position_map(total_vec)
    
    seed_mre_positions = []
    
    for vec_idx in range(len(total_vec)):
        mirna_pos_in_binding = mirna_pos_map.get(vec_idx)
        mre_pos_in_binding = mre_pos_map.get(vec_idx)
        
        if mirna_pos_in_binding is not None and mre_pos_in_binding is not None:
            # Convert to 1-indexed full miRNA position
            # mirna_binding_start is 0-indexed, mirna_pos_in_binding is 0-indexed
            # So we need +1 to get 1-indexed position
            mirna_pos_in_full = mirna_pos_in_binding + mirna_binding_start + 1
            
            if 1 <= mirna_pos_in_full <= 7:  # Seed positions 1-7 (1-indexed)
                original_mre_idx = mre_binding_end - mre_pos_in_binding
                
                if 0 <= original_mre_idx < 50:
                    seed_mre_positions.append(original_mre_idx)
    
    return seed_mre_positions


def extract_conservation_features(site_data, mirna_binding_start, flank_size=10):
    """Extract conservation scores for seed and flanking regions."""
    features = {}
    default_keys = [
        'seed_conservation_mean', 'seed_conservation_median', 'seed_conservation_max',
        'seed_conservation_min', 'seed_conservation_std', 'five_prime_flank_conservation_mean',
        'three_prime_flank_conservation_mean', 'full_site_conservation_mean',
        'conservation_contrast', 'flank_conservation_ratio'
    ]
    
    scores = parse_conservation_scores(site_data.get('conservation_vector'))
    
    if not scores or len(scores) != 50:
        return {k: 0.0 for k in default_keys}
    
    seed_mre_positions = get_mre_positions_for_seed(site_data, mirna_binding_start)
    
    if not seed_mre_positions:
        return {k: 0.0 for k in default_keys}
    
    seed_mre_positions = [p for p in seed_mre_positions if 0 <= p < 50]
    
    if not seed_mre_positions:
        return {k: 0.0 for k in default_keys}
    
    seed_start, seed_end = min(seed_mre_positions), max(seed_mre_positions)
    seed_scores = [scores[i] for i in seed_mre_positions]
    
    if seed_scores:
        features['seed_conservation_mean'] = round(np.mean(seed_scores), 4)
        features['seed_conservation_median'] = round(np.median(seed_scores), 4)
        features['seed_conservation_max'] = round(np.max(seed_scores), 4)
        features['seed_conservation_min'] = round(np.min(seed_scores), 4)
        features['seed_conservation_std'] = round(np.std(seed_scores), 4) if len(seed_scores) > 1 else 0.0
        # features['seed_high_conservation_fraction'] = round(
        #     sum(1 for s in seed_scores if s > 0.7) / len(seed_scores), 4)
    else:
        for k in ['seed_conservation_mean', 'seed_conservation_median', 'seed_conservation_max',
                  'seed_conservation_min', 'seed_conservation_std']:
            features[k] = 0.0
    
    five_prime_flank_scores = scores[max(0, seed_start - flank_size):seed_start]
    features['five_prime_flank_conservation_mean'] = round(
        np.mean(five_prime_flank_scores), 4) if five_prime_flank_scores else 0.0
    
    three_prime_flank_scores = scores[seed_end + 1:min(50, seed_end + 1 + flank_size)]
    features['three_prime_flank_conservation_mean'] = round(
        np.mean(three_prime_flank_scores), 4) if three_prime_flank_scores else 0.0
    
    full_scores = scores[max(0, seed_start - flank_size):min(50, seed_end + 1 + flank_size)]
    features['full_site_conservation_mean'] = round(np.mean(full_scores), 4) if full_scores else 0.0
    
    features['conservation_contrast'] = round(features['seed_conservation_mean'] - np.mean(scores), 4)
    flank_mean = (features['five_prime_flank_conservation_mean'] + 
                  features['three_prime_flank_conservation_mean']) / 2
    features['flank_conservation_ratio'] = round(
        features['seed_conservation_mean'] / flank_mean, 4) if flank_mean > 0 else 0.0
    
    return features


def extract_extended_conservation_features(site_data):
    """Extract conservation pattern features."""
    scores = parse_conservation_scores(site_data.get('conservation_vector'))
    if not scores or len(scores) != 50:
        return {
            'conservation_variance': 0.0, 'conservation_range': 0.0,
            # 'num_conserved_blocks': 0, 'max_conserved_block_length': 0, 'is_highly_conserved': 0
        }
    
    features = {
        'conservation_variance': round(np.var(scores), 4),
        'conservation_range': round(np.max(scores) - np.min(scores), 4)
    }
    return features

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_max_consecutive_char(text, char):
    runs = re.findall(f"{re.escape(char)}+", text)
    return max((len(s) for s in runs), default=0)


def safe_float(value, default=0.0):
    """Safely convert value to float."""
    if value is None or value == 'NA' or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    """Safely convert value to int."""
    if value is None or value == 'NA' or value == '':
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


# ============================================================================
# MAIN FEATURE EXTRACTION FUNCTION FOR INTARNA
# ============================================================================

def calculate_intarna_features(site_data, flank_size=10):
    """
    Calculate extensive features from IntaRNA interaction prediction.
    
    This function extracts features from IntaRNA output format, including:
    - All IntaRNA energy components (E_total, E_hybrid, ED1, ED2)
    - Duplex structure features (matches, mismatches, bulges)
    - Seed region analysis
    - G:U wobble analysis
    - Base composition features
    - Conservation features
    - Local secondary structure context (from CONTRAfold)
    
    Args:
        site_data: Dictionary containing IntaRNA results and auxiliary data
        flank_size: Size of flanking regions for conservation analysis
    
    Returns:
        Updated site_data dictionary with all extracted features
    """
    
    # =========================================================================
    # Parse IntaRNA structure and sequences
    # =========================================================================
    
    # Parse hybrid_dp to get structures: "target_struct&query_struct"
    hybrid_dp = site_data.get('hybrid_dp', '')
    if '&' in hybrid_dp:
        parts = hybrid_dp.split('&')
        mre_struct = parts[0]
        mirna_struct = parts[1]
    else:
        mre_struct = ''
        mirna_struct = ''
    
    site_data['mre_struct'] = mre_struct
    site_data['mirna_struct'] = mirna_struct
    
    # Parse subseq_dp to get binding subsequences: "target_subseq&query_subseq"
    subseq_dp = site_data.get('subseq_dp', '')
    if '&' in subseq_dp:
        parts = subseq_dp.split('&')
        mre_binding_subseq = parts[0].upper().replace('T', 'U')
        mirna_binding_subseq = parts[1].upper().replace('T', 'U')
    else:
        mre_binding_subseq = ''
        mirna_binding_subseq = ''
    
    # Get coordinates (IntaRNA uses 1-indexed)
    mre_coord_start = safe_int(site_data.get('start_target', site_data.get('mre_coord_start', 1)))
    mre_coord_end = safe_int(site_data.get('end_target', site_data.get('mre_coord_end', 1)))
    mirna_coord_start = safe_int(site_data.get('start_query', site_data.get('mirna_coord_start', 1)))
    mirna_coord_end = safe_int(site_data.get('end_query', site_data.get('mirna_coord_end', 1)))
    
    site_data['mre_coord_start'] = mre_coord_start
    site_data['mre_coord_end'] = mre_coord_end
    site_data['mirna_coord_start'] = mirna_coord_start
    site_data['mirna_coord_end'] = mirna_coord_end
    
    # Convert to 0-indexed for internal use
    mre_binding_start = mre_coord_start - 1
    mre_binding_end = mre_coord_end
    mirna_binding_start = mirna_coord_start - 1
    mirna_binding_end = mirna_coord_end
    
    # Get full sequences
    mre_seq = site_data.get('mre_seq', '')
    mirna_seq = site_data.get('mirna_seq', '')
    
    # =========================================================================
    # IntaRNA Energy Features
    # =========================================================================
    
    site_data['E'] = safe_float(site_data.get('E', site_data.get('E', 0)))
    site_data['E_hybrid'] = safe_float(site_data.get('E_hybrid', site_data.get('energy_hybrid', 0)))
    site_data['ED_target'] = safe_float(site_data.get('ED_target', site_data.get('ED1', 0)))
    site_data['ED_query'] = safe_float(site_data.get('ED_query', site_data.get('ED2', 0)))
    site_data['Eall1'] = safe_float(site_data.get('Eall1', 0)) if site_data.get('Eall1') else 0.0
    site_data['Eall2'] = safe_float(site_data.get('Eall2', 0)) if site_data.get('Eall2') else 0.0
    site_data['Eall'] = safe_float(site_data.get('Eall', 0)) if site_data.get('Eall') else 0.0
    site_data['Ealltotal'] = safe_float(site_data.get('Ealltotal', 0)) if site_data.get('Ealltotal') else 0.0
    site_data['E_total'] = safe_float(site_data.get('E_total', 0)) if site_data.get('E_total') else 0.0
    site_data['Energy_norm'] = safe_float(site_data.get('Energy_norm', 0)) if site_data.get('Energy_norm') else 0.0
    site_data['Energy_hybrid_norm'] = safe_float(site_data.get('Energy_hybrid_norm', 0)) if site_data.get('Energy_hybrid_norm') else 0.0
    site_data['P_E'] = safe_float(site_data.get('P_E', 0)) if site_data.get('P_E') else 0.0
    # Energy per nucleotide
    interaction_length = max(1, len(mre_binding_subseq))
   
    # =========================================================================
    # Create Duplex Vector
    # =========================================================================
    
    # For miRNA-mRNA interactions, reverse target structure for proper alignment
    mre_struct_rev = mre_struct[::-1]
    total_vec = create_duplex_vectors(mre_struct_rev, mirna_struct)
    site_data['total_vec'] = total_vec
    
    # =========================================================================
    # Basic Duplex Statistics
    # =========================================================================
    
    site_data['total_matches'] = total_vec.count('1')
    site_data['total_mismatches'] = total_vec.count('2')
    site_data['total_mre_bulges'] = total_vec.count('3')
    site_data['total_mirna_bulges'] = total_vec.count('4')
    site_data['total_bulges'] = site_data['total_mre_bulges'] + site_data['total_mirna_bulges']
    site_data['total_dangles'] = total_vec.count('D') + total_vec.count('d') + total_vec.count('e')
    
    # Interaction length
    site_data['interaction_length'] = len(total_vec)
    site_data['mre_interaction_length'] = mre_coord_end - mre_coord_start + 1
    site_data['mirna_interaction_length'] = mirna_coord_end - mirna_coord_start + 1
    
    # Match fraction
    if len(total_vec) > 0:
        site_data['match_fraction'] = round(site_data['total_matches'] / len(total_vec), 4)
    else:
        site_data['match_fraction'] = 0.0
    
    # =========================================================================
    # Seed Region Features (miRNA positions 1-8)
    # =========================================================================
    
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
    
    # Non-seed (3' supplementary) region
    non_seed_indices = get_vector_indices_for_mirna_range(total_vec, 8, 100, mirna_binding_start)
    non_seed_vec = ''.join(total_vec[i] for i in non_seed_indices)
    site_data['consecutive_matches_minus_seed'] = get_max_consecutive_char(non_seed_vec, '1')
    site_data['non_seed_matches'] = non_seed_vec.count('1')
    
    # Start of seed match
    try:
        site_data['start_match_seed'] = mirna_struct.index(')') + 1
    except ValueError:
        site_data['start_match_seed'] = 99
    
    # =========================================================================
    # Paired Bases Analysis (for G:U wobbles and base composition)
    # =========================================================================
    
    mre_binding_seq_rev = mre_binding_subseq[::-1]
    
    all_paired_mre, all_paired_mir, mirna_positions = get_paired_bases_with_positions(
        total_vec, mre_binding_seq_rev, mirna_binding_subseq,
        mre_binding_start, mirna_binding_start
    )
    
    # Seed pairs (positions 1-7 in full miRNA)
    seed_pairs = [(mre, mir) for mre, mir, pos in zip(all_paired_mre, all_paired_mir, mirna_positions)
                  if 1 <= pos <= 7]
    paired_mre_seed = ''.join(p[0] for p in seed_pairs)
    paired_mir_seed = ''.join(p[1] for p in seed_pairs)
    
    # Seed base composition
    if len(seed_pairs) > 0:
        au_count = sum(1 for mre, mir in seed_pairs
                       if {mre, mir} in [{'A', 'U'}, {'A', 'T'}, {'U', 'A'}, {'T', 'A'}])
        gc_count = sum(1 for mre, mir in seed_pairs
                       if {mre, mir} in [{'G', 'C'}, {'C', 'G'}])
        site_data['seed_au_content'] = round(au_count / len(seed_pairs), 4)
        site_data['seed_gc_content'] = round(gc_count / len(seed_pairs), 4)
    else:
        site_data['seed_au_content'] = 0.0
        site_data['seed_gc_content'] = 0.0
    
    # G:U wobbles in seed
    site_data['gu_wobbles_in_seed'] = sum(
        1 for i in range(len(paired_mre_seed))
        if {paired_mre_seed[i], paired_mir_seed[i]} in [{'G', 'U'}, {'G', 'T'}]
    )
    site_data['gu_wobbles_in_seed_2_8_pos'] = site_data['gu_wobbles_in_seed']  # Alias for compatibility
    
    # Non-seed pairs
    nonseed_pairs = [(mre, mir) for mre, mir, pos in zip(all_paired_mre, all_paired_mir, mirna_positions)
                     if pos > 7]
    paired_mre_nonseed = ''.join(p[0] for p in nonseed_pairs)
    paired_mir_nonseed = ''.join(p[1] for p in nonseed_pairs)
    
    # Non-seed base composition
    if len(nonseed_pairs) > 0:
        nonseed_au_count = sum(1 for mre, mir in nonseed_pairs
                               if {mre, mir} in [{'A', 'U'}, {'A', 'T'}, {'U', 'A'}, {'T', 'A'}])
        nonseed_gc_count = sum(1 for mre, mir in nonseed_pairs
                               if {mre, mir} in [{'G', 'C'}, {'C', 'G'}])
        site_data['nonseed_au_content'] = round(nonseed_au_count / len(nonseed_pairs), 4)
        site_data['nonseed_gc_content'] = round(nonseed_gc_count / len(nonseed_pairs), 4)
    else:
        site_data['nonseed_au_content'] = 0.0
        site_data['nonseed_gc_content'] = 0.0
    
    # G:U wobbles outside seed
    site_data['total_matches_minus_seed'] = len(paired_mre_nonseed)
    site_data['gu_wobbles_minus_seed'] = sum(
        1 for i in range(len(paired_mre_nonseed))
        if {paired_mre_nonseed[i], paired_mir_nonseed[i]} in [{'G', 'U'}, {'G', 'T'}]
    )
    
    # Total G:U wobbles
    site_data['total_gu_wobbles'] = site_data['gu_wobbles_in_seed'] + site_data['gu_wobbles_minus_seed']
   
    # =========================================================================
    # Conservation Features
    # =========================================================================
    
    site_data['conservation_scores_list'] = parse_conservation_scores(
        site_data.get('conservation_vector'))
    site_data.update(extract_conservation_features(site_data, mirna_binding_start, flank_size))
    site_data.update(extract_extended_conservation_features(site_data))
    
    # Priority score (matching R implementation logic)
    # R: priority.seed only subtracts G:U if count > 1
    priority_seed = site_data['seed_matches']
    if site_data['gu_wobbles_in_seed'] > 1:
        priority_seed = priority_seed - site_data['gu_wobbles_in_seed']
    
    priority_loop_penalty = (site_data['total_bulges'] + site_data['total_mismatches']) / 3.0
    
    # R: replacement logic, not cumulative
    if priority_seed > 7:
        priority = 4 * priority_seed + site_data['non_seed_matches'] - priority_loop_penalty - site_data['gu_wobbles_minus_seed']
    elif priority_seed > 6:
        priority = 3 * priority_seed + site_data['non_seed_matches'] - priority_loop_penalty - site_data['gu_wobbles_minus_seed']
    elif priority_seed > 5:
        priority = 2 * priority_seed + site_data['non_seed_matches'] - priority_loop_penalty - site_data['gu_wobbles_minus_seed']
    elif priority_seed > 4:
        priority = 1 * priority_seed + site_data['non_seed_matches'] - priority_loop_penalty - site_data['gu_wobbles_minus_seed']
    else:
        priority = site_data['non_seed_matches'] - priority_loop_penalty - site_data['gu_wobbles_minus_seed']
    
    site_data['priority_score'] = round(priority, 4)
    
    # Effective seed matches (accounting for wobbles, only if > 1)
    if site_data['gu_wobbles_in_seed'] > 1:
        site_data['effective_seed_matches'] = site_data['seed_matches'] - site_data['gu_wobbles_in_seed']
    else:
        site_data['effective_seed_matches'] = site_data['seed_matches']
    
    # Effective 3' matches
    site_data['effective_3prime_matches'] = (site_data['total_matches_minus_seed'] - 
                                              site_data['gu_wobbles_minus_seed'])
    
    return site_data


# ============================================================================
# BINDING TYPE CLASSIFICATION
# ============================================================================

def classify_binding_type_detailed(d):
    """Classify miRNA binding type based on duplex features."""
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


def filter_sites_detailed(site_data):
    """Apply quality filters to binding sites."""
    btype = site_data.get('binding_type', '')
    effective_3prime = site_data.get('effective_3prime_matches', 0)
    
    if "5mer" in btype and effective_3prime <= 5:
        return False
    if ("seedless" in btype or "3prime.compensatory" in btype) and effective_3prime <= 7:
        return False
    if btype == "3prime" and site_data.get('consecutive_matches_minus_seed', 0) < 6:
        return False
    if site_data.get('seed_mirna_bulge_positions', 0) > 2:
        return False
    return True


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def diagnose_position_tracking(all_sites_data):
    """Diagnose position tracking issues in the dataset."""
    print("\n" + "=" * 70)
    print("POSITION TRACKING DIAGNOSTICS")
    print("=" * 70)
    
    n = len(all_sites_data)
    if n == 0:
        print("No sites to analyze.")
        return
    
    stats = {k: 0 for k in ['mre_bulge', 'mirna_bulge', 'pre_match', 'mre_dangle', 'mirna_dangle']}
    sites_with_any_bulge = 0
    
    for site in all_sites_data:
        vec = site.get('total_vec', '')
        has_bulge = False
        
        if '3' in vec:
            stats['mre_bulge'] += 1
            has_bulge = True
        if '4' in vec:
            stats['mirna_bulge'] += 1
            has_bulge = True
        if 'D' in vec:
            stats['pre_match'] += 1
        if 'e' in vec:
            stats['mre_dangle'] += 1
        if 'd' in vec:
            stats['mirna_dangle'] += 1
        
        if has_bulge:
            sites_with_any_bulge += 1
    
    print(f"\nTotal sites: {n}")
    print(f"  MRE bulges ('3'):     {stats['mre_bulge']:>6} ({100*stats['mre_bulge']/n:.1f}%)")
    print(f"  miRNA bulges ('4'):   {stats['mirna_bulge']:>6} ({100*stats['mirna_bulge']/n:.1f}%)")
    print(f"  Pre-match gaps ('D'): {stats['pre_match']:>6} ({100*stats['pre_match']/n:.1f}%)")
    print(f"  MRE dangling ('e'):   {stats['mre_dangle']:>6} ({100*stats['mre_dangle']/n:.1f}%)")
    print(f"  miRNA dangling ('d'): {stats['mirna_dangle']:>6} ({100*stats['mirna_dangle']/n:.1f}%)")
    print(f"  Sites with ANY bulge: {sites_with_any_bulge:>6} ({100*sites_with_any_bulge/n:.1f}%)")
    print("=" * 70 + "\n")


def diagnose_conservation(all_sites_data):
    """Diagnose conservation feature extraction."""
    print("\n" + "=" * 70)
    print("CONSERVATION DIAGNOSTICS")
    print("=" * 70)
    
    if not all_sites_data:
        print("No sites to analyze.")
        return
    
    # Check conservation_scores_list
    cons_list_lengths = [len(s.get('conservation_scores_list', [])) for s in all_sites_data]
    valid = sum(1 for l in cons_list_lengths if l == 50)
    
    print(f"\nConservation scores list lengths:")
    print(f"  Length == 50: {valid}")
    print(f"  Length == 0:  {sum(1 for l in cons_list_lengths if l == 0)}")
    print(f"  Other:        {sum(1 for l in cons_list_lengths if l != 50 and l != 0)}")
    
    if cons_list_lengths:
        unique_lengths = sorted(set(cons_list_lengths))
        print(f"  Unique lengths found: {unique_lengths[:10]}{'...' if len(unique_lengths) > 10 else ''}")
    
    # Check raw conservation_vector
    raw_types = {}
    for s in all_sites_data[:10]:  # Check first 10
        raw_val = s.get('conservation_vector')
        t = type(raw_val).__name__
        raw_types[t] = raw_types.get(t, 0) + 1
    print(f"\nRaw conservation_vector types (first 10): {raw_types}")
    
    # Show a sample
    if all_sites_data:
        sample = all_sites_data[0]
        raw_val = sample.get('conservation_vector')
        print(f"\nSample site conservation_vector type: {type(raw_val)}")
        print(f"Sample site conservation_vector: {str(raw_val)[:150]}...")
        print(f"Sample parsed list length: {len(sample.get('conservation_scores_list', []))}")
        
        # Show coordinates
        print(f"\nSample site coordinates:")
        print(f"  mre_coord_start: {sample.get('mre_coord_start')}")
        print(f"  mre_coord_end: {sample.get('mre_coord_end')}")
        print(f"  mirna_coord_start: {sample.get('mirna_coord_start')}")
        print(f"  mirna_coord_end: {sample.get('mirna_coord_end')}")
        print(f"  total_vec: {sample.get('total_vec', '')[:50]}...")
    
    # Check seed positions (safely)
    seed_positions_counts = []
    for s in all_sites_data[:10]:
        try:
            if s.get('mre_coord_start') and s.get('mre_coord_end') and s.get('total_vec'):
                mirna_binding_start = s.get('mirna_coord_start', 1) - 1
                positions = get_mre_positions_for_seed(s, mirna_binding_start)
                seed_positions_counts.append(len(positions))
            else:
                seed_positions_counts.append(-1)  # Missing data
        except Exception as e:
            seed_positions_counts.append(-2)  # Error
    print(f"\nSeed MRE positions found (first 10 sites): {seed_positions_counts}")
    print("  (-1 = missing data, -2 = error)")
    
    seed_vals = [s.get('seed_conservation_mean', 0) for s in all_sites_data]
    print(f"\nSeed conservation - Mean: {np.mean(seed_vals):.4f}, Median: {np.median(seed_vals):.4f}")
    
    sites_with_seed = sum(1 for s in all_sites_data if s.get('seed_conservation_mean', 0) > 0)
    print(f"Sites with non-zero seed conservation: {sites_with_seed}/{len(all_sites_data)}")
    print("=" * 70 + "\n")


def diagnose_energies(all_sites_data):
    """Diagnose IntaRNA energy features."""
    print("\n" + "=" * 70)
    print("INTARNA ENERGY DIAGNOSTICS")
    print("=" * 70)
    
    if not all_sites_data:
        print("No sites to analyze.")
        return
    
    e_total = [s.get('E', 0) for s in all_sites_data]
    e_hybrid = [s.get('E_hybrid', 0) for s in all_sites_data]
    ed_target = [s.get('ED_target', 0) for s in all_sites_data]
    ed_query = [s.get('ED_query', 0) for s in all_sites_data]
    
    print(f"\nTotal sites: {len(all_sites_data)}")
    print(f"\nE_total:    Min={min(e_total):.2f}, Max={max(e_total):.2f}, Mean={np.mean(e_total):.2f}")
    print(f"E_hybrid:   Min={min(e_hybrid):.2f}, Max={max(e_hybrid):.2f}, Mean={np.mean(e_hybrid):.2f}")
    print(f"ED_target:  Min={min(ed_target):.2f}, Max={max(ed_target):.2f}, Mean={np.mean(ed_target):.2f}")
    print(f"ED_query:   Min={min(ed_query):.2f}, Max={max(ed_query):.2f}, Mean={np.mean(ed_query):.2f}")
    
    # Check energy decomposition consistency
    consistent = sum(1 for s in all_sites_data 
                     if abs(s['E_total'] - (s['E_hybrid'] + s['ED_target'] + s['ED_query'])) < 0.1)
    print(f"\nEnergy decomposition consistent: {consistent}/{len(all_sites_data)}")
    print("=" * 70 + "\n")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract features from IntaRNA miRNA-MRE interaction predictions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python extract_intarna_features.py \\
    --intarna best_intarna_results.tsv \\
    --mre-fasta mre_sequences.fasta \\
    --mirna-fasta mirna_sequences.fasta \\
    --conservation conservation_data.tsv \\
    --filter no \\
    --output features.csv

IntaRNA TSV should contain columns from --select-best output:
  target_id, query_id, start_target, end_target, start_query, end_query,
  subseq_dp, hybrid_dp, E_total, E_hybrid, ED_target, ED_query, etc.
        """
    )
    
    parser.add_argument(
        '--intarna', required=True,
        help='Path to IntaRNA best results TSV file')
    parser.add_argument(
        '--mre-fasta', required=True,
        help='Path to FASTA file containing MRE sequences (50nt each)')
    parser.add_argument(
        '--mirna-fasta', required=True,
        help='Path to FASTA file containing miRNA sequences')
    # parser.add_argument(
    #     '--contrafold', required=True,
    #     help='Path to CONTRAfold secondary structure predictions for MREs')
    parser.add_argument(
        '--conservation', required=True,
        help='Path to TSV file containing conservation vectors and metadata')
    parser.add_argument(
        '--output', required=True,
        help='Path for output CSV file with extracted features')
    parser.add_argument(
        '--filter', choices=['yes', 'y', 'no', 'n'], default='no',
        help='Apply heuristic filters to remove low-quality binding sites (default: no)')
    parser.add_argument(
        '--flank-size', type=int, default=10,
        help='Size of flanking regions for conservation analysis (default: 10)')
    parser.add_argument(
        '--bigwig', default='/home/adam/adam/data/hg38.phastCons470way.bw',
        help='Path to phastCons BigWig file (optional)')
    
    args = parser.parse_args()
    
    try:
        print("--- Stage 0: Reading Input Files ---")
        
        # Read IntaRNA results
        intarna_results = parse_intarna_results(args.intarna)
        print(f"  IntaRNA results: {len(intarna_results)} entries")
        
        # Read FASTA files
        mre_seqs = parse_fasta(args.mre_fasta)
        mirna_seqs = parse_fasta(args.mirna_fasta)
        print(f"  MRE sequences: {len(mre_seqs)}")
        print(f"  miRNA sequences: {len(mirna_seqs)}")
        
        # Read conservation data
        cons_vecs, mir_fam, labels = parse_conservation_tsv(args.conservation, args.bigwig)
        print(f"  Conservation vectors: {len(cons_vecs)}")
        
        # Validate counts match
        counts = {
            "IntaRNA": len(intarna_results),
            "MRE": len(mre_seqs),
            "miRNA": len(mirna_seqs),
            "Conservation": len(cons_vecs)
        }
        
        print("\nPre-flight Check:")
        for name, count in counts.items():
            print(f"  {name}: {count}")
        
        if len(set(counts.values())) != 1:
            print("\nERROR: Entry count mismatch!")
            sys.exit(1)
        print("\nAll files synchronized.\n")
        
        # =====================================================================
        # Build site data structures
        # =====================================================================
        
        print("--- Stage 1: Building Site Data ---")
        all_sites = []
        
        for i, intarna_row in enumerate(intarna_results):
            # Skip entries with no interaction
            if intarna_row.get('status') == 'no_interactions':
                continue
            if intarna_row.get('start_target') in ('NA', '', None):
                continue
            
            site_data = {
                # From IntaRNA
                'target_id': intarna_row.get('target_id', ''),
                'query_id': intarna_row.get('query_id', ''),
                'start_target': intarna_row.get('start_target'),
                'end_target': intarna_row.get('end_target'),
                'start_query': intarna_row.get('start_query'),
                'end_query': intarna_row.get('end_query'),
                'subseq_dp': intarna_row.get('subseq_dp', ''),
                'hybrid_dp': intarna_row.get('hybrid_dp', ''),
                'E': intarna_row.get('E'),
                'E_hybrid': intarna_row.get('E_hybrid'),
                'ED_target': intarna_row.get('ED_target'),
                'ED_query': intarna_row.get('ED_query'),
                'Eall': intarna_row.get('Eall'),
                'Eall1': intarna_row.get('Eall1'),
                'Eall2': intarna_row.get('Eall2'),
                'Ealltotal': intarna_row.get('Ealltotal'),
                'Etotal': intarna_row.get('E_total'),
                'P_E': intarna_row.get('P_E'),
                'energy_hybrid_norm': intarna_row.get('Energy_hybrid_norm'),
                'energy_norm': intarna_row.get('Energy_norm'),
                'total_matches_intarna': intarna_row.get('total_matches', 0),
                'seed_matches_intarna': intarna_row.get('seed_matches', 0),
                'gu_wobbles_seed_intarna': intarna_row.get('gu_wobbles_seed', 0),
                'gu_wobbles_other_intarna': intarna_row.get('gu_wobbles_other', 0),
                
                # From FASTA files
                'mre_seq': mre_seqs[i],
                'mirna_seq': mirna_seqs[i],
                
                # From conservation
                'conservation_vector': cons_vecs.iloc[i] if hasattr(cons_vecs, 'iloc') else cons_vecs[i],
                'mir_fam': mir_fam.iloc[i] if hasattr(mir_fam, 'iloc') else mir_fam[i],
                'label': labels.iloc[i] if hasattr(labels, 'iloc') else labels[i],
                
                # Chimeric sequence
                'chimeric_sequence': mre_seqs[i] + mirna_seqs[i]
            }
            
            all_sites.append(site_data)
        
        print(f"  Valid sites: {len(all_sites)}")
        
        # =====================================================================
        # Feature Extraction
        # =====================================================================
        
        print("\n--- Stage 2: Calculating Features ---")
        featured = [calculate_intarna_features(s, flank_size=args.flank_size) for s in all_sites]
        
        # Diagnostics
        diagnose_energies(featured)
        diagnose_position_tracking(featured)
        diagnose_conservation(featured)
        
        # =====================================================================
        # Binding Type Classification
        # =====================================================================
        
        print("--- Stage 3: Classifying Binding Types ---")
        for site in featured:
            site['binding_type'] = classify_binding_type_detailed(site)
        
        # =====================================================================
        # Filtering
        # =====================================================================
        
        print("--- Stage 4: Filtering ---")
        if args.filter.lower() in ['y', 'yes']:
            final = [s for s in featured if filter_sites_detailed(s)]
            print(f"Filtered: {len(final)}/{len(featured)} sites passed quality filters")
        else:
            final = featured
            print(f"No filtering applied: {len(final)} sites retained")
        
        # =====================================================================
        # Output
        # =====================================================================
        
        print(f"\n--- Stage 5: Writing to CSV ---")
        if not final:
            print("No sites to write.")
            return
        
        # Define output headers (including IntaRNA energy features)
        headers = [
            # Identifiers
            "target_id", "query_id", "binding_type",
            
            # IntaRNA energy features
            "E", "E_hybrid", "ED_target", "ED_query", "Eall", "Eall1", "Eall2", "Ealltotal", "Etotal", "P_E", "Energy_hybrid_norm", "Energy_norm",
            
            # Structure strings
            "hybrid_dp", "subseq_dp", 
            
            # Sequences
            "mre_sequence", "mirna_sequence", "chimeric_sequence", "mir_fam",
            
            # Duplex statistics
            "total_matches", "total_mismatches", "total_bulges",
            "total_mre_bulges", "total_mirna_bulges", "total_gu_wobbles",
            "interaction_length", "match_fraction",
            
            # Seed features
            "seed_matches", "seed_matches_2_8", "consecutive_matches_seed",
            "gu_wobbles_in_seed_2_8_pos", "effective_seed_matches",
            
            
            # Non-seed features
            "non_seed_matches", "consecutive_matches_minus_seed",
            "gu_wobbles_minus_seed",
            "effective_3prime_matches",
            
            # Base composition
            "seed_au_content", "seed_gc_content",
            "nonseed_au_content", "nonseed_gc_content",
            
            # Conservation features
            "seed_conservation_mean", "seed_conservation_median",
            "seed_conservation_max", "seed_conservation_min", "seed_conservation_std",
            "five_prime_flank_conservation_mean", "three_prime_flank_conservation_mean",
            "full_site_conservation_mean", "conservation_contrast",
            "flank_conservation_ratio",
            "conservation_variance", "conservation_range",
           
            # Label
            "label"
        ]
        
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            
            for site in final:
                row = {
                    "target_id": site.get('target_id', ''),
                    "query_id": site.get('query_id', ''),
                    "binding_type": site.get('binding_type', ''),
                    
                    # IntaRNA energies
                    "E": site.get('E', 0),
                    "E_hybrid": site.get('E_hybrid', 0),
                    "ED_target": site.get('ED_target', 0),
                    "ED_query": site.get('ED_query', 0),
                    "Eall": site.get('Eall', 0),
                    "Eall1": site.get('Eall1', 0),
                    "Eall2": site.get('Eall2', 0),
                    "Ealltotal": site.get('Ealltotal', 0),
                    "Etotal": site.get('Etotal', 0),
                    "P_E": site.get('P_E', 0),
                    "Energy_hybrid_norm": site.get('energy_hybrid_norm', 0),
                    "Energy_norm": site.get('energy_norm', 0),
                    
                    # Structures
                    "hybrid_dp": site.get('hybrid_dp', ''),
                    "subseq_dp": site.get('subseq_dp', ''),
                    
                    # Sequences
                    "mre_sequence": site.get('mre_seq', ''),
                    "mirna_sequence": site.get('mirna_seq', ''),
                    "chimeric_sequence": site.get('chimeric_sequence', ''),
                    "mir_fam": site.get('mir_fam', ''),
                    
                    # Duplex stats
                    "total_matches": site.get('total_matches', 0),
                    "total_mismatches": site.get('total_mismatches', 0),
                    "total_bulges": site.get('total_bulges', 0),
                    "total_mre_bulges": site.get('total_mre_bulges', 0),
                    "total_mirna_bulges": site.get('total_mirna_bulges', 0),
                    "total_gu_wobbles": site.get('total_gu_wobbles', 0),
                    "interaction_length": site.get('interaction_length', 0),
                    "match_fraction": site.get('match_fraction', 0),
                    
                    # Seed features
                    "seed_matches": site.get('seed_matches', 0),
                    "seed_matches_2_8": site.get('seed_matches_2_8', 0),
                    "consecutive_matches_seed": site.get('consecutive_matches_seed', 0),
                    "gu_wobbles_in_seed_2_8_pos": site.get('gu_wobbles_in_seed_2_8_pos', 0),
                    "effective_seed_matches": site.get('effective_seed_matches', 0),
                    
                    # Non-seed features
                    "non_seed_matches": site.get('non_seed_matches', 0),
                    "consecutive_matches_minus_seed": site.get('consecutive_matches_minus_seed', 0),
                    "gu_wobbles_minus_seed": site.get('gu_wobbles_minus_seed', 0),
                    "effective_3prime_matches": site.get('effective_3prime_matches', 0),
                    
                    # Base composition
                    "seed_au_content": site.get('seed_au_content', 0),
                    "seed_gc_content": site.get('seed_gc_content', 0),
                    "nonseed_au_content": site.get('nonseed_au_content', 0),
                    "nonseed_gc_content": site.get('nonseed_gc_content', 0),
                    
                    # Conservation
                    "seed_conservation_mean": site.get('seed_conservation_mean', 0),
                    "seed_conservation_median": site.get('seed_conservation_median', 0),
                    "seed_conservation_max": site.get('seed_conservation_max', 0),
                    "seed_conservation_min": site.get('seed_conservation_min', 0),
                    "seed_conservation_std": site.get('seed_conservation_std', 0),
                    "five_prime_flank_conservation_mean": site.get('five_prime_flank_conservation_mean', 0),
                    "three_prime_flank_conservation_mean": site.get('three_prime_flank_conservation_mean', 0),
                    "full_site_conservation_mean": site.get('full_site_conservation_mean', 0),
                    "conservation_contrast": site.get('conservation_contrast', 0),
                    "flank_conservation_ratio": site.get('flank_conservation_ratio', 0),
                    "conservation_variance": site.get('conservation_variance', 0),
                    "conservation_range": site.get('conservation_range', 0),
                    
                    # Label
                    "label": site.get('label', 0)
                }
                writer.writerow(row)
        
        print(f"\nSuccess! Feature extraction complete.")
        print(f"   Output saved to: {args.output}")
        print(f"   Total sites: {len(final)}")
        print(f"   Features extracted: {len(headers)}")
        
    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e.filename}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
