import os
import re
import sys
import csv
import argparse
import numpy as np
import pandas as pd
import pyBigWig
from best_duplex import parse_fasta, create_duplex_vectors

# ============================================================================
# UTILITY AND PARSING FUNCTIONS
# ============================================================================

def get_phastcons_vector(bw_path, chrom, start, end, strand):
    """Retrieves a phastCons score vector for a given genomic region."""
    try:
        bw = pyBigWig.open(bw_path)
        start, end = int(start)-1, int(end)
        scores = bw.values(chrom, start, end)
        scores = np.nan_to_num(scores, nan=0.0)
        if strand == '-':
            scores = scores[::-1]
        bw.close()
        return scores.tolist()
    except (RuntimeError, ValueError):
        return None

def parse_conservation_tsv(file_path):
    """Reads TSV and extracts conservation vectors."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        df['chr'] = "chr" + df['chr'].astype(str)
        df['chr'] = df['chr'].replace({'chrMT': 'chrM'})
        df['gene_phastCons470'] = df.apply(
            lambda row: get_phastcons_vector(
                '/home/adam/adam/data/hg38.phastCons470way.bw', 
                row['chr'], row['start'], row['end'], row['strand']
            ), axis=1) 
        return (df['gene_phastCons470'], df['noncodingRNA_fam'], df['label'])
    except (FileNotFoundError, KeyError) as e:
        print(f"Error processing conservation file '{file_path}': {e}")
        sys.exit(1)

# ============================================================================
# POSITION TRACKING FUNCTIONS (FIXED FOR BINDING REGIONS)
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
    """
    Get vector indices for a miRNA position range in the FULL miRNA.
    
    Args:
        mirna_start: Start position in FULL miRNA (0-indexed)
        mirna_end: End position in FULL miRNA (0-indexed, exclusive)
        mirna_binding_start: Where binding starts in full miRNA (0-indexed)
    """
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
# PAIRED BASES EXTRACTION (FIXED FOR BINDING REGIONS)
# ============================================================================

def get_paired_bases_with_positions(total_vec, mre_binding_seq, mirna_binding_seq, 
                                     mre_binding_start, mirna_binding_start):
    """
    Extracts paired bases and tracks positions in the FULL sequences.
    
    Args:
        total_vec: Duplex vector (represents binding region only)
        mre_binding_seq: MRE binding region sequence (already reversed)
        mirna_binding_seq: miRNA binding region sequence
        mre_binding_start: 0-indexed position where binding starts in full MRE
        mirna_binding_start: 0-indexed position where binding starts in full miRNA
    
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
# CONSERVATION FEATURE EXTRACTION (FIXED)
# ============================================================================

def parse_conservation_scores(cons_vector):
    """Parse conservation vector from various formats."""
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
    return []

def get_mre_positions_for_seed(site_data, mirna_binding_start):
    """Maps miRNA seed positions to MRE positions in full 50nt sequence."""
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
            mirna_pos_in_full = mirna_pos_in_binding + mirna_binding_start
            
            if 1 <= mirna_pos_in_full <= 7:
                original_mre_idx = mre_binding_end - mre_pos_in_binding
                
                if 0 <= original_mre_idx < 50:
                    seed_mre_positions.append(original_mre_idx)
    
    return seed_mre_positions

def extract_conservation_features(site_data, mirna_binding_start, flank_size=10):
    """Extracts conservation scores for seed and flanking regions."""
    features = {}
    default_keys = ['seed_conservation_mean', 'seed_conservation_median', 'seed_conservation_max',
                    'seed_conservation_min', 'seed_conservation_std', 'five_prime_flank_conservation_mean',
                    'three_prime_flank_conservation_mean', 'full_site_conservation_mean',
                    'conservation_contrast', 'seed_high_conservation_fraction', 'flank_conservation_ratio']
    
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
        features['seed_high_conservation_fraction'] = round(sum(1 for s in seed_scores if s > 0.7) / len(seed_scores), 4)
    else:
        for k in ['seed_conservation_mean', 'seed_conservation_median', 'seed_conservation_max',
                  'seed_conservation_min', 'seed_conservation_std', 'seed_high_conservation_fraction']:
            features[k] = 0.0
    
    five_prime_flank_scores = scores[max(0, seed_start - flank_size):seed_start]
    features['five_prime_flank_conservation_mean'] = round(np.mean(five_prime_flank_scores), 4) if five_prime_flank_scores else 0.0
    
    three_prime_flank_scores = scores[seed_end + 1:min(50, seed_end + 1 + flank_size)]
    features['three_prime_flank_conservation_mean'] = round(np.mean(three_prime_flank_scores), 4) if three_prime_flank_scores else 0.0
    
    full_scores = scores[max(0, seed_start - flank_size):min(50, seed_end + 1 + flank_size)]
    features['full_site_conservation_mean'] = round(np.mean(full_scores), 4) if full_scores else 0.0
    
    features['conservation_contrast'] = round(features['seed_conservation_mean'] - np.mean(scores), 4)
    flank_mean = (features['five_prime_flank_conservation_mean'] + features['three_prime_flank_conservation_mean']) / 2
    features['flank_conservation_ratio'] = round(features['seed_conservation_mean'] / flank_mean, 4) if flank_mean > 0 else 0.0
    
    return features

def extract_extended_conservation_features(site_data):
    """Extracts conservation pattern features."""
    scores = parse_conservation_scores(site_data.get('conservation_vector'))
    if not scores or len(scores) != 50:
        return {'conservation_variance': 0.0, 'conservation_range': 0.0}
    
    features = {
        'conservation_variance': round(np.var(scores), 4),
        'conservation_range': round(np.max(scores) - np.min(scores), 4)
    }
    return features

# ============================================================================
# FEATURE CALCULATION
# ============================================================================

def get_max_consecutive_char(text, char):
    runs = re.findall(f"{re.escape(char)}+", text)
    return max((len(s) for s in runs), default=0)

def calculate_detailed_features(site_data, flank_size=10):
    """Calculates extensive features for a duplex."""
    
    # Extract binding regions
    mre_binding_start = site_data['mre_coord_start'] - 1
    mre_binding_end = site_data['mre_coord_end']
    mirna_binding_start = site_data['mirna_coord_start'] - 1
    mirna_binding_end = site_data['mirna_coord_end']
    
    mre_binding_seq = site_data['mre_seq'][mre_binding_start:mre_binding_end]
    mirna_binding_seq = site_data['mirna_seq'][mirna_binding_start:mirna_binding_end]
    
    # Create duplex vector
    mre_struct_rev = site_data['mre_struct'][::-1]
    total_vec = create_duplex_vectors(mre_struct_rev, site_data['mirna_struct'])
    site_data['total_vec'] = total_vec
    
    # Seed features
    seed_region_vec = get_mirna_region_vector(total_vec, 0, 8, mirna_binding_start)
    seed_region_2_8_vec = get_mirna_region_vector(total_vec, 1, 8, mirna_binding_start)
    
    site_data['consecutive_matches_seed'] = get_max_consecutive_char(seed_region_2_8_vec, '1')
    site_data['total_matches_in_seed_9_pos'] = count_char_in_mirna_region(total_vec, 0, 9, mirna_binding_start, '1')
    site_data['mismatch_seed_positions'] = seed_region_vec.count('2')
    site_data['seed_target_bulge_positions'] = seed_region_vec.count('3')
    site_data['seed_mirna_bulge_positions'] = seed_region_vec.count('4')
    
    non_seed_indices = get_vector_indices_for_mirna_range(total_vec, 8, 100, mirna_binding_start)
    non_seed_vec = ''.join(total_vec[i] for i in non_seed_indices)
    site_data['consecutive_matches_minus_seed'] = get_max_consecutive_char(non_seed_vec, '1')
    
    try:
        site_data['start_match_seed'] = site_data['mirna_struct'].index(')') + 1
    except ValueError:
        site_data['start_match_seed'] = 99
    
    # Paired bases
    all_paired_mre, all_paired_mir, mirna_positions = get_paired_bases_with_positions(
        total_vec, mre_binding_seq[::-1], mirna_binding_seq, mre_binding_start, mirna_binding_start)
    
    seed_pairs = [(mre, mir) for mre, mir, pos in zip(all_paired_mre, all_paired_mir, mirna_positions) if 1 <= pos <= 7]
    paired_mre_seed = ''.join(p[0] for p in seed_pairs)
    paired_mir_seed = ''.join(p[1] for p in seed_pairs)
    
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
    # ====================================================
    
    nonseed_pairs = [(mre, mir) for mre, mir, pos in zip(all_paired_mre, all_paired_mir, mirna_positions) if pos > 7]
    paired_mre_nonseed = ''.join(p[0] for p in nonseed_pairs)
    paired_mir_nonseed = ''.join(p[1] for p in nonseed_pairs)
    
    # ========== NEW: AU and GC content in non-seed ==========
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

    site_data['gu_wobbles_in_seed_2_8_pos'] = sum(
        1 for i in range(len(paired_mre_seed)) if {paired_mre_seed[i], paired_mir_seed[i]} in [{'G', 'U'}, {'G', 'T'}])
    
    site_data['total_matches_minus_seed'] = len(paired_mre_nonseed)
    site_data['gu_wobbles_minus_seed'] = sum(
        1 for i in range(len(paired_mre_nonseed)) if {paired_mre_nonseed[i], paired_mir_nonseed[i]} in [{'G', 'U'}, {'G', 'T'}])

    contrafold_struct = site_data['contrafold_struct']
    site_data['mre_accessibility_score'] = round(contrafold_struct.count('.') / len(contrafold_struct), 4) if contrafold_struct else 'N/A'
    
    site_data['conservation_scores_list'] = parse_conservation_scores(site_data['conservation_vector'])
    site_data.update(extract_conservation_features(site_data, mirna_binding_start, flank_size))
    site_data.update(extract_extended_conservation_features(site_data))
    
    return site_data

# ============================================================================
# CLASSIFICATION
# ============================================================================

def classify_binding_type_detailed(d):
    consecutive = d['consecutive_matches_seed']
    btype = f"{consecutive}mer" if consecutive >= 5 else "seedless"

    if d['start_match_seed'] >= 4:
        btype = "seedless"
    if d['total_matches_in_seed_9_pos'] == 9:
        btype = "9mer"
        
    if btype == "seedless":
        mirna_binding_start = d.get('mirna_coord_start', 1) - 1
        centered_matches = count_consecutive_matches_in_mirna_region(d['total_vec'], 4, 16, mirna_binding_start)
        if centered_matches >= 8:
            return "centered"
        elif d['start_match_seed'] >= 13:
            return "3prime"
        elif (d['total_matches_minus_seed'] - d['gu_wobbles_minus_seed']) >= 6:
            return "3prime.compensatory"
        return btype

    if btype == "8mer" and d['start_match_seed'] == 2 and d['mirna_seq'][0] == 'A':
        btype = "8mer1A"
    elif btype == "7mer" and d['start_match_seed'] == 2 and d['mirna_seq'][0] == 'A':
        btype = "8mer1A"
    elif btype == "6mer":
        if d['start_match_seed'] == 3:
            btype = "offset6mer"
        elif d['start_match_seed'] == 2 and d['mirna_seq'][0] == 'A':
            btype = "7mer1A"
    
    if 'mer' in btype:
        if d['mismatch_seed_positions'] > 0:
            btype += ".mismatch"
        if d['seed_target_bulge_positions'] > 0:
            btype += ".target.bulge"
        if d['seed_mirna_bulge_positions'] > 0:
            btype += ".mirna.bulge"
        if d['gu_wobbles_in_seed_2_8_pos'] > 0:
            btype += ".GU"
        if (d['total_matches_minus_seed'] - d['gu_wobbles_minus_seed']) >= 3:
            btype += ".3prime"
    return btype

def filter_sites_detailed(site_data):
    btype = site_data['binding_type']
    if "5mer" in btype and (site_data['total_matches_minus_seed'] - site_data['gu_wobbles_minus_seed']) <= 5:
        return False
    if ("seedless" in btype or "3prime.compensatory" in btype) and \
       (site_data['total_matches_minus_seed'] - site_data['gu_wobbles_minus_seed']) <= 7:
        return False
    if btype == "3prime" and site_data['consecutive_matches_minus_seed'] < 6:
        return False
    if site_data['seed_mirna_bulge_positions'] > 2:
        return False
    return True
# ============================================================================
# DIAGNOSTICS
# ============================================================================
def diagnose_position_tracking(all_sites_data):
    """Diagnose position tracking issues in the dataset."""
    print("\n" + "="*70)
    print("POSITION TRACKING DIAGNOSTICS")
    print("="*70)
    
    n = len(all_sites_data)
    stats = {k: 0 for k in ['mre_bulge', 'mirna_bulge', 'pre_match', 'mre_dangle', 'mirna_dangle']}
    sites_with_any_bulge = 0  # NEW: count unique sites
    
    for site in all_sites_data:
        vec = site.get('total_vec', '')
        has_bulge = False  # NEW: track if this site has any bulge
        
        if '3' in vec: 
            stats['mre_bulge'] += 1
            has_bulge = True
        if '4' in vec: 
            stats['mirna_bulge'] += 1
            has_bulge = True
        if 'D' in vec: stats['pre_match'] += 1
        if 'e' in vec: stats['mre_dangle'] += 1
        if 'd' in vec: stats['mirna_dangle'] += 1
        
        if has_bulge:  # NEW: count unique sites with bulges
            sites_with_any_bulge += 1
    
    print(f"\nTotal sites: {n}")
    print(f"  MRE bulges ('3'):     {stats['mre_bulge']:>6} ({100*stats['mre_bulge']/n:.1f}%)")
    print(f"  miRNA bulges ('4'):   {stats['mirna_bulge']:>6} ({100*stats['mirna_bulge']/n:.1f}%)")
    print(f"  Pre-match gaps ('D'): {stats['pre_match']:>6} ({100*stats['pre_match']/n:.1f}%)")
    print(f"  MRE dangling ('e'):   {stats['mre_dangle']:>6} ({100*stats['mre_dangle']/n:.1f}%)")
    print(f"  miRNA dangling ('d'): {stats['mirna_dangle']:>6} ({100*stats['mirna_dangle']/n:.1f}%)")
    print(f"  Sites with ANY bulge: {sites_with_any_bulge:>6} ({100*sites_with_any_bulge/n:.1f}%)")  # NEW
    
    pct = 100 * sites_with_any_bulge / n  # FIXED
    print(f"\n{'='*70}")
    if pct < 5:
        print(f"LOW IMPACT: {pct:.1f}% of sites have bulges")
    elif pct < 15:
        print(f"MEDIUM IMPACT: {pct:.1f}% of sites have bulges")
    else:
        print(f"HIGH IMPACT: {pct:.1f}% of sites have bulges")
    print("="*70 + "\n")

def diagnose_conservation(all_sites_data):
    """Diagnose conservation feature extraction."""
    print("\n" + "="*70)
    print("CONSERVATION DIAGNOSTICS")
    print("="*70)
    
    valid = sum(1 for s in all_sites_data if len(s.get('conservation_scores_list', [])) == 50)
    seed_vals = [s.get('seed_conservation_mean', 0) for s in all_sites_data]
    
    print(f"\nValid conservation data: {valid}/{len(all_sites_data)}")
    print(f"Seed conservation - Mean: {np.mean(seed_vals):.4f}, Median: {np.median(seed_vals):.4f}")
    
    # Check if seed positions are reasonable
    sites_with_seed = sum(1 for s in all_sites_data if s.get('seed_conservation_mean', 0) > 0)
    print(f"Sites with non-zero seed conservation: {sites_with_seed}/{len(all_sites_data)}")
    print("="*70 + "\n")

def diagnose_zero_conservation_root_cause(all_sites_data):
    """
    Comprehensive diagnostic to find WHY seed conservation is zero.
    """
    print("\n" + "="*70)
    print("ROOT CAUSE ANALYSIS: ZERO SEED CONSERVATION")
    print("="*70)
    
    # Analyze a few problematic sites in detail
    zero_seed_sites = [s for s in all_sites_data if s.get('seed_conservation_mean', 0) == 0][:5]
    
    if not zero_seed_sites:
        print("No zero-conservation sites found!")
        return
    
    print(f"\nAnalyzing {len(zero_seed_sites)} examples with zero seed conservation:\n")
    
    for idx, site in enumerate(zero_seed_sites):
        print(f"{'='*70}")
        print(f"EXAMPLE {idx+1}")
        print(f"{'='*70}")
        print(f"Chr: {site.get('chr')}, Pos: {site.get('start')}-{site.get('end')}, Strand: {site.get('strand')}")
        print(f"miRNA binding: {site.get('mirna_coord_start')}-{site.get('mirna_coord_end')}")
        print(f"MRE binding: {site.get('mre_coord_start')}-{site.get('mre_coord_end')}")
        
        # Check 1: Conservation vector validity
        cons_scores = parse_conservation_scores(site.get('conservation_vector'))
        print(f"\nCheck 1 - Conservation Vector:")
        print(f"  Length: {len(cons_scores)} (expected: 50)")
        if len(cons_scores) == 50:
            print(f"  Mean: {np.mean(cons_scores):.3f}")
            print(f"  Max: {np.max(cons_scores):.3f}")
            print(f"  Non-zero positions: {sum(1 for s in cons_scores if s > 0)}/50")
            print(f"  First 10: {[round(s, 2) for s in cons_scores[:10]]}")
        else:
            print(f"  ❌ INVALID LENGTH!")
            continue
        
        # Check 2: Seed region mapping
        mirna_binding_start = site.get('mirna_coord_start', 1) - 1
        total_vec = site.get('total_vec', '')
        
        print(f"\nCheck 2 - Seed Region Mapping:")
        print(f"  total_vec length: {len(total_vec)}")
        print(f"  miRNA binding starts at position: {mirna_binding_start} (0-indexed in full miRNA)")
        
        # Get seed indices
        seed_indices = get_vector_indices_for_mirna_range(total_vec, 0, 8, mirna_binding_start)
        print(f"  Seed covers vector indices: {seed_indices}")
        
        if not seed_indices:
            print(f"  ❌ NO SEED INDICES FOUND!")
            print(f"  Possible reasons:")
            print(f"    - miRNA binding starts after position 8 (binding_start={mirna_binding_start})")
            print(f"    - No matched pairs in seed region")
            continue
        
        # Get MRE positions for seed
        seed_mre_positions = get_mre_positions_for_seed(site, mirna_binding_start)
        print(f"  Seed maps to MRE positions (in 50nt): {seed_mre_positions}")
        
        if not seed_mre_positions:
            print(f"  ❌ SEED DOES NOT MAP TO ANY MRE POSITIONS!")
            continue
        
        # Check 3: Conservation at seed positions
        print(f"\nCheck 3 - Conservation at Seed Positions:")
        seed_cons_values = [cons_scores[p] for p in seed_mre_positions if 0 <= p < 50]
        print(f"  Conservation values: {[round(v, 3) for v in seed_cons_values]}")
        
        if all(v == 0 for v in seed_cons_values):
            print(f"  ❌ ALL ZERO - Possible causes:")
            print(f"    A) Genuinely non-conserved region (biological)")
            print(f"    B) Wrong positions extracted from BigWig")
            print(f"    C) Coordinate system mismatch")
            
            # Check if OTHER positions have conservation
            non_seed_cons = [cons_scores[i] for i in range(50) if i not in seed_mre_positions]
            if non_seed_cons:
                print(f"  Non-seed conservation mean: {np.mean(non_seed_cons):.3f}")
                if np.mean(non_seed_cons) > 0.1:
                    print(f"  ⚠️ NON-SEED HAS CONSERVATION BUT SEED DOESN'T - LIKELY INDEXING ERROR!")
        
        # Check 4: Binding region conservation
        print(f"\nCheck 4 - Whole Binding Region:")
        mre_binding_start = site.get('mre_coord_start', 1) - 1
        mre_binding_end = site.get('mre_coord_end', 50)
        binding_cons = cons_scores[mre_binding_start:mre_binding_end]
        print(f"  Binding region [{mre_binding_start}:{mre_binding_end}] conservation: {np.mean(binding_cons):.3f}")
        
        flanks = cons_scores[:mre_binding_start] + cons_scores[mre_binding_end:]
        if flanks:
            print(f"  Flanking regions conservation: {np.mean(flanks):.3f}")
            if np.mean(binding_cons) < np.mean(flanks) * 0.5:
                print(f"  ⚠️ BINDING LESS CONSERVED THAN FLANKS - POSSIBLY BACKWARDS!")
        
        print()
    
    print("="*70)
    print("SUMMARY OF LIKELY CAUSES:")
    print("1. If 'NO SEED INDICES': miRNA binding starts too late (after position 8)")
    print("2. If 'ALL ZERO' but non-seed has conservation: INDEXING MISMATCH")
    print("3. If everything is zero: genuinely non-conserved OR BigWig extraction issue")
    print("4. If binding < flanks consistently: sequences/conservation ORIENTATION MISMATCH")
    print("="*70 + "\n")

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Extract features from miRNA-MRE binding interactions for machine learning.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python script.py \\
    --duplex best_duplex_output.txt \\
    --mre-fasta mre_sequences.fasta \\
    --mirna-fasta mirna_sequences.fasta \\
    --conservation conservation_data.tsv \\
    --filter yes \\
    --output features.csv

Note: All input files must have the same number of entries (synchronized).
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--duplex',
        required=True,
        help='Path to best duplex file (output from RNAduplex processing)'
    )
    
    parser.add_argument(
        '--mre-fasta',
        required=True,
        help='Path to FASTA file containing MRE sequences (50nt each)'
    )
    
    parser.add_argument(
        '--mirna-fasta',
        required=True,
        help='Path to FASTA file containing miRNA sequences'
    )
    
    parser.add_argument(
        '--conservation',
        required=True,
        help='Path to TSV file containing conservation vectors and metadata'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Path for output CSV file with extracted features'
    )
    
    # Optional arguments
    parser.add_argument(
        '--filter',
        choices=['yes', 'y', 'no', 'n'],
        default='no',
        help='Apply heuristic filters to remove low-quality binding sites (default: no)'
    )
    
    parser.add_argument(
        '--flank-size',
        type=int,
        default=10,
        help='Size of flanking regions for conservation analysis (default: 10)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Extract values from args
    duplex_file = args.duplex
    mre_fasta = args.mre_fasta
    mirna_fasta = args.mirna_fasta
    cons_file = args.conservation
    output_csv = args.output
    do_filter = args.filter
    flank_size = args.flank_size

    try:
        print("--- Stage 0: Reading Input Files ---")
        with open(os.path.abspath(duplex_file), 'r') as f:
            raw_data = f.read()
        
        entries = [e for e in raw_data.strip().split('-----------') if e.strip()]
        mre_seqs = parse_fasta(mre_fasta)
        mirna_seqs = parse_fasta(mirna_fasta)
        cons_vecs, mir_fam, labels = parse_conservation_tsv(cons_file)
        
        counts = {"Duplex": len(entries), "MRE": len(mre_seqs), "miRNA": len(mirna_seqs)}
        
        print("\nPre-flight Check:")
        for name, count in counts.items():
            print(f"  {name}: {count}")
        
        if len(set(counts.values())) != 1:
            print("\nERROR: Entry count mismatch!")
            sys.exit(1)
        print("\nAll files synchronized.\n")
        
        all_sites = []
        line_re = re.compile(r"(.+?)\s+(\d+),(\d+)\s+:\s+(\d+),(\d+)\s+\((.+)\)")
        
        for i, entry in enumerate(entries):
            lines = entry.strip().split('\n')
            header, struct_line = lines[0], lines[2]
            m = line_re.match(struct_line.strip())
            if not m:
                continue
            full_struct, mre_s, mre_e, mir_s, mir_e, mfe = m.groups()
            mre_struct, mirna_struct = full_struct.split('&')
            
            all_sites.append({
                "header": header, "full_structure": full_struct, "mfe": float(mfe),
                "mre_struct": mre_struct, "mirna_struct": mirna_struct,
                "mre_seq": mre_seqs[i], "mirna_seq": mirna_seqs[i],
                "conservation_vector": cons_vecs[i],
                "mir_fam": mir_fam[i], "label": labels[i],
                "mre_coord_start": int(mre_s), "mre_coord_end": int(mre_e),
                "mirna_coord_start": int(mir_s), "mirna_coord_end": int(mir_e),
                "chimeric_sequence": mre_seqs[i] + mirna_seqs[i]
            })

        print("--- Stage 1: Calculating Features ---")
        featured = [calculate_detailed_features(s, flank_size=flank_size) for s in all_sites]
        
        diagnose_position_tracking(featured)
        diagnose_conservation(featured)
        diagnose_zero_conservation_root_cause(featured)

        print("--- Stage 2: Classifying Binding Types ---")
        for site in featured:
            site['binding_type'] = classify_binding_type_detailed(site)

        print("--- Stage 3: Filtering ---")
        if do_filter.lower() in ['y', 'yes']:
            final = [s for s in featured if filter_sites_detailed(s)]
            print(f"Filtered: {len(final)}/{len(featured)} sites passed quality filters")
        else:
            final = featured
            print(f"No filtering applied: {len(final)} sites retained")

        print(f"--- Stage 4: Writing to CSV ---")
        if not final:
            print("No sites to write.")
            return

        headers = [
            "interaction_header", "binding_type", "mre_accessibility_score", "contrafold_struct",
            "conservation_vector", "mfe", "full_structure", "mre_sequence", "mirna_sequence",
            "chimeric_sequence", "mir_fam", "consecutive_matches_seed", "gu_wobbles_in_seed_2_8_pos", "mismatch_seed_positions",
            "seed_conservation_mean", "seed_conservation_median", "seed_conservation_max",
            "seed_conservation_min", "seed_conservation_std", "five_prime_flank_conservation_mean",
            "three_prime_flank_conservation_mean", "full_site_conservation_mean",
            "conservation_contrast", "seed_high_conservation_fraction", "flank_conservation_ratio",
            "conservation_variance", "conservation_range",
            "seed_au_content", "seed_gc_content", "nonseed_au_content", "nonseed_gc_content",  
            "label"
        ]
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for site in final:
                writer.writerow({
                    "interaction_header": site['header'],
                    "binding_type": site['binding_type'],
                    "conservation_vector": site['conservation_vector'],
                    "mfe": site['mfe'],
                    "full_structure": site['full_structure'],
                    "mre_sequence": site["mre_seq"],
                    "mirna_sequence": site["mirna_seq"],
                    "chimeric_sequence": site["chimeric_sequence"],
                    "mir_fam": site["mir_fam"], "consecutive_matches_seed": site['consecutive_matches_seed'],
                    "gu_wobbles_in_seed_2_8_pos": site['gu_wobbles_in_seed_2_8_pos'],
                    "mismatch_seed_positions": site['mismatch_seed_positions'],
                    "seed_conservation_mean": site.get('seed_conservation_mean', 0.0),
                    "seed_conservation_median": site.get('seed_conservation_median', 0.0),
                    "seed_conservation_max": site.get('seed_conservation_max', 0.0),
                    "seed_conservation_min": site.get('seed_conservation_min', 0.0),
                    "seed_conservation_std": site.get('seed_conservation_std', 0.0),
                    "five_prime_flank_conservation_mean": site.get('five_prime_flank_conservation_mean', 0.0),
                    "three_prime_flank_conservation_mean": site.get('three_prime_flank_conservation_mean', 0.0),
                    "full_site_conservation_mean": site.get('full_site_conservation_mean', 0.0),
                    "conservation_contrast": site.get('conservation_contrast', 0.0),
                    "flank_conservation_ratio": site.get('flank_conservation_ratio', 0.0),
                    "conservation_variance": site.get('conservation_variance', 0.0),
                    "conservation_range": site.get('conservation_range', 0.0),
                    "seed_au_content": site.get('seed_au_content', 0.0),
                    "seed_gc_content": site.get('seed_gc_content', 0.0),
                    "nonseed_au_content": site.get('nonseed_au_content', 0.0),  # NEW
                    "nonseed_gc_content": site.get('nonseed_gc_content', 0.0),  # NEW
                    "label": site['label']
                })
        
        print(f"\nSuccess! Feature extraction complete.")
        print(f"   Output saved to: {output_csv}")
        print(f"   Total sites: {len(final)}")
        print(f"   Flank size used: {flank_size} nt")
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
