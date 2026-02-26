"""
Region-Specific Sequence Features for miRNA and MRE Sequences

Features are computed for specific regions:
- MRE: 5' half, 3' half (2 regions)
- miRNA: seed (positions 2-8), 3' region (2 regions)

For each region:
- Purine/pyrimidine patterns (8 features)
- Entropy & complexity (4 features)
- Thermodynamic features (8 features)
- Stability dynamics (5 features)
- Terminal energies (2 features)
- Motif counts (3 features)

Total: ~30 features/region × 4 regions = ~120 features
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
from itertools import product
import polars as pl
import warnings
import sys

# Track problematic sequences for diagnostics
PROBLEM_COUNT = 0
MAX_PRINT = 20  # Maximum number of problematic sequences to print


# Turner nearest-neighbor stacking energies (kcal/mol, 37°C, 1M NaCl)
DINUC_ENERGY = {
    'AA': -0.93, 'AU': -1.10, 'AC': -2.24, 'AG': -2.08,
    'UA': -1.33, 'UU': -0.93, 'UC': -2.35, 'UG': -1.30,
    'CA': -2.11, 'CU': -2.08, 'CC': -3.26, 'CG': -2.36,
    'GA': -2.35, 'GU': -1.30, 'GC': -3.42, 'GG': -3.26,
}

# Reference statistics for z-scores
_ALL_ENERGIES = list(DINUC_ENERGY.values())
DINUC_ENERGY_MEAN = np.mean(_ALL_ENERGIES)
DINUC_ENERGY_STD = np.std(_ALL_ENERGIES)

# Default value for missing/invalid features
DEFAULT_VALUE = 0.0


def print_warning_sequence(mre_seq, mirna_seq, issue, warning_type):
    """Print sequence that triggered a warning in real-time."""
    global PROBLEM_COUNT
    
    if PROBLEM_COUNT < MAX_PRINT:
        print(f"\n⚠️  WARNING #{PROBLEM_COUNT + 1} - {warning_type}")
        print(f"   Issue: {issue}")
        print(f"   MRE  (len={len(mre_seq) if mre_seq else 0:2d}): {mre_seq[:60] if mre_seq else 'EMPTY'}")
        print(f"   miRNA(len={len(mirna_seq) if mirna_seq else 0:2d}): {mirna_seq[:60] if mirna_seq else 'EMPTY'}")
        
        # Show region splits
        if mre_seq and len(mre_seq) >= 3:
            mre_regions = get_regions(mre_seq, is_mirna=False)
            print(f"   MRE regions: 5p={mre_regions['5p'][:30]} | 3p={mre_regions['3p'][:30]}")
        
        if mirna_seq and len(mirna_seq) >= 3:
            mirna_regions = get_regions(mirna_seq, is_mirna=True)
            print(f"   miRNA regions: seed={mirna_regions.get('seed', '')[:30]} | 3p={mirna_regions.get('3p', '')[:30]}")
        
        sys.stdout.flush()  # Force immediate output
    elif PROBLEM_COUNT == MAX_PRINT:
        print(f"\n... (suppressing further warnings, {MAX_PRINT} samples shown)")
        sys.stdout.flush()
    
    PROBLEM_COUNT += 1


def safe_divide(numerator, denominator, default=DEFAULT_VALUE, context=None):
    """Safely divide, returning default if denominator is 0 or invalid."""
    if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
        if context and 'mre_seq' in context and 'mirna_seq' in context:
            print_warning_sequence(
                context['mre_seq'], 
                context['mirna_seq'],
                f"Division by zero: {numerator}/{denominator}",
                "DIVISION BY ZERO"
            )
        return default
    result = numerator / denominator
    if np.isnan(result) or np.isinf(result):
        if context and 'mre_seq' in context and 'mirna_seq' in context:
            print_warning_sequence(
                context['mre_seq'],
                context['mirna_seq'],
                f"Invalid result: {numerator}/{denominator} = {result}",
                "NaN/INF RESULT"
            )
        return default
    return result


def get_regions(seq: str, is_mirna: bool = False) -> Dict[str, str]:
    """
    Split sequence into regions.
    
    For MRE: 5' half, 3' half
    For miRNA: seed (positions 2-8), 3' region (rest after seed)
    
    Returns:
        Dictionary mapping region names to subsequences
    """
    seq = seq.upper().replace('T', 'U')
    seq_len = len(seq)
    
    regions = {}
    
    if is_mirna:
        # Seed region: positions 2-8 (1-indexed) = indices 1-7 (0-indexed)
        if seq_len >= 8:
            regions['seed'] = seq[1:8]
        elif seq_len > 1:
            regions['seed'] = seq[1:]
        else:
            regions['seed'] = seq
        
        # 3' region: everything after seed
        if seq_len > 8:
            regions['3p'] = seq[8:]
        elif seq_len >= 3:
            regions['3p'] = seq[-min(3, seq_len):]
        else:
            regions['3p'] = seq
    else:
        # MRE regions: simple halves
        mid = max(seq_len // 2, 1)
        
        regions['5p'] = seq[:mid]
        regions['3p'] = seq[mid:] if seq_len > mid else seq
    
    return regions


def extract_region_features(seq: str, region_name: str, context: Dict = None) -> Dict[str, float]:
    """
    Extract features for a single region.
    
    Returns ~30 features per region.
    """
    if not seq or len(seq) < 3:
        if context and len(seq) > 0 and len(seq) < 3:
            print_warning_sequence(
                context.get('mre_seq', ''),
                context.get('mirna_seq', ''),
                f"Region '{region_name}' too short: {seq} (len={len(seq)})",
                "SHORT REGION"
            )
        return get_empty_region_features(region_name)
    
    features = {}
    n_trinuc = len(seq) - 2
    n_dinuc = len(seq) - 1
    
    if n_trinuc < 1 or n_dinuc < 1:
        if context:
            print_warning_sequence(
                context.get('mre_seq', ''),
                context.get('mirna_seq', ''),
                f"Region '{region_name}' insufficient for features: {seq} (len={len(seq)})",
                "INSUFFICIENT LENGTH"
            )
        return get_empty_region_features(region_name)
    
    trinucs = [seq[i:i+3] for i in range(n_trinuc)]
    trinuc_counts = Counter(trinucs)
    
    dinucs = [seq[i:i+2] for i in range(n_dinuc)]
    dinuc_counts = Counter(dinucs)
    
    # =================================================================
    # 1. PURINE/PYRIMIDINE PATTERNS (8 features)
    # =================================================================
    pur_seq = ''.join('R' if nt in 'AG' else 'Y' for nt in seq)
    
    if len(pur_seq) >= 3:
        pur_trinucs = Counter(pur_seq[i:i+3] for i in range(len(pur_seq) - 2))
        pur_patterns = ['RRR', 'RRY', 'RYR', 'RYY', 'YRR', 'YRY', 'YYR', 'YYY']
        for pattern in pur_patterns:
            features[f'{region_name}_{pattern}_freq'] = safe_divide(
                pur_trinucs.get(pattern, 0), n_trinuc, context=context
            )
    else:
        pur_patterns = ['RRR', 'RRY', 'RYR', 'RYY', 'YRR', 'YRY', 'YYR', 'YYY']
        for pattern in pur_patterns:
            features[f'{region_name}_{pattern}_freq'] = DEFAULT_VALUE
    
    # =================================================================
    # 2. ENTROPY & COMPLEXITY (4 features)
    # =================================================================
    # Trinucleotide entropy
    if len(trinuc_counts) > 0 and n_trinuc > 0:
        freqs = [c / n_trinuc for c in trinuc_counts.values()]
        entropy = -sum(f * np.log2(f) for f in freqs if f > 0)
        max_entropy = np.log2(min(n_trinuc, 64))
        features[f'{region_name}_trinuc_entropy'] = safe_divide(entropy, max_entropy, context=context)
    else:
        features[f'{region_name}_trinuc_entropy'] = DEFAULT_VALUE
    
    # Unique trinucleotide ratio
    features[f'{region_name}_unique_trinuc_ratio'] = safe_divide(
        len(trinuc_counts), min(n_trinuc, 64), context=context
    )
    
    # Gini coefficient
    if len(trinuc_counts) > 1:
        sorted_counts = sorted(trinuc_counts.values())
        cumsum = np.cumsum(sorted_counts)
        if cumsum[-1] > 0 and len(sorted_counts) > 0:
            gini = 1 - 2 * np.sum(cumsum) / (len(sorted_counts) * cumsum[-1]) + 1 / len(sorted_counts)
            features[f'{region_name}_trinuc_gini'] = gini if not np.isnan(gini) else DEFAULT_VALUE
        else:
            features[f'{region_name}_trinuc_gini'] = DEFAULT_VALUE
    else:
        features[f'{region_name}_trinuc_gini'] = DEFAULT_VALUE
    
    # Repeat ratio
    if len(trinuc_counts) > 0:
        repeated = sum(1 for c in trinuc_counts.values() if c > 1)
        features[f'{region_name}_trinuc_repeat_ratio'] = safe_divide(repeated, len(trinuc_counts), context=context)
    else:
        features[f'{region_name}_trinuc_repeat_ratio'] = DEFAULT_VALUE
    
    # =================================================================
    # 3. THERMODYNAMIC FEATURES (8 features)
    # =================================================================
    energies = [DINUC_ENERGY.get(d, -1.5) for d in dinucs]
    
    if len(energies) > 0:
        features[f'{region_name}_energy_mean'] = np.mean(energies)
        features[f'{region_name}_energy_std'] = np.std(energies) if len(energies) > 1 else DEFAULT_VALUE
        features[f'{region_name}_energy_min'] = np.min(energies)
        features[f'{region_name}_energy_max'] = np.max(energies)
        features[f'{region_name}_energy_range'] = np.max(energies) - np.min(energies)
        
    else:
        features[f'{region_name}_energy_mean'] = DEFAULT_VALUE
        features[f'{region_name}_energy_std'] = DEFAULT_VALUE
        features[f'{region_name}_energy_min'] = DEFAULT_VALUE
        features[f'{region_name}_energy_max'] = DEFAULT_VALUE
        features[f'{region_name}_energy_range'] = DEFAULT_VALUE
    
    # Asymmetry within region
    if n_dinuc >= 4:
        half = n_dinuc // 2
        first_half_energy = np.mean(energies[:half])
        second_half_energy = np.mean(energies[half:])
        features[f'{region_name}_energy_asymmetry'] = first_half_energy - second_half_energy
    else:
        features[f'{region_name}_energy_asymmetry'] = DEFAULT_VALUE
    
    # Energy gradient
    if n_dinuc >= 3:
        x = np.arange(n_dinuc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr_matrix = np.corrcoef(x, energies)
            corr = corr_matrix[0, 1] if corr_matrix.shape == (2, 2) else 0
            features[f'{region_name}_energy_gradient'] = corr if not np.isnan(corr) else DEFAULT_VALUE
    else:
        features[f'{region_name}_energy_gradient'] = DEFAULT_VALUE
    
    # =================================================================
    # 4. STABILITY DYNAMICS (5 features)
    # =================================================================
    if len(energies) >= 2:
        changes = np.diff(energies)
        
        features[f'{region_name}_energy_volatility'] = np.mean(np.abs(changes))
        features[f'{region_name}_energy_max_jump'] = np.max(np.abs(changes))
        
        stable_threshold = 0.3
        stable_count = np.sum(np.abs(changes) < stable_threshold)
        features[f'{region_name}_stability_run_frac'] = safe_divide(stable_count, len(changes), context=context)
        
        # Oscillation
        signs = np.sign(changes)
        nonzero_signs = signs[signs != 0]
        if len(nonzero_signs) > 1:
            direction_changes = np.sum(np.abs(np.diff(nonzero_signs)) > 0)
            features[f'{region_name}_energy_oscillation'] = safe_divide(
                direction_changes, len(nonzero_signs) - 1, context=context
            )
        else:
            features[f'{region_name}_energy_oscillation'] = DEFAULT_VALUE
        
        features[f'{region_name}_energy_drift'] = energies[-1] - energies[0]
    else:
        features[f'{region_name}_energy_volatility'] = DEFAULT_VALUE
        features[f'{region_name}_energy_max_jump'] = DEFAULT_VALUE
        features[f'{region_name}_stability_run_frac'] = DEFAULT_VALUE
        features[f'{region_name}_energy_oscillation'] = DEFAULT_VALUE
        features[f'{region_name}_energy_drift'] = DEFAULT_VALUE
    
    # =================================================================
    # 5. TERMINAL FEATURES (2 features)
    # =================================================================
    energies_list = [DINUC_ENERGY.get(d, -1.5) for d in dinucs]
    features[f'{region_name}_5p_terminal_energy'] = energies_list[0] if energies_list else DEFAULT_VALUE
    features[f'{region_name}_3p_terminal_energy'] = energies_list[-1] if energies_list else DEFAULT_VALUE
    
    # =================================================================
    # 6. MOTIF-BASED FEATURES (3 features)
    # =================================================================
    features[f'{region_name}_ggg_count'] = seq.count('GGG')
    features[f'{region_name}_uuu_count'] = seq.count('UUU')
    return features


def get_empty_region_features(region_name: str) -> Dict[str, float]:
    """Return empty features for a region."""
    features = {}
    
    # Purine/pyrimidine
    pur_patterns = ['RRR', 'RRY', 'RYR', 'RYY', 'YRR', 'YRY', 'YYR', 'YYY']
    for pattern in pur_patterns:
        features[f'{region_name}_{pattern}_freq'] = DEFAULT_VALUE
    
    # Entropy & complexity
    features[f'{region_name}_trinuc_entropy'] = DEFAULT_VALUE
    features[f'{region_name}_unique_trinuc_ratio'] = DEFAULT_VALUE
    features[f'{region_name}_trinuc_gini'] = DEFAULT_VALUE
    features[f'{region_name}_trinuc_repeat_ratio'] = DEFAULT_VALUE
    
    # Thermodynamic
    features[f'{region_name}_energy_mean'] = DEFAULT_VALUE
    features[f'{region_name}_energy_std'] = DEFAULT_VALUE
    features[f'{region_name}_energy_min'] = DEFAULT_VALUE
    features[f'{region_name}_energy_max'] = DEFAULT_VALUE
    features[f'{region_name}_energy_range'] = DEFAULT_VALUE
    features[f'{region_name}_energy_asymmetry'] = DEFAULT_VALUE
    features[f'{region_name}_energy_gradient'] = DEFAULT_VALUE
    
    # Stability dynamics
    features[f'{region_name}_energy_volatility'] = DEFAULT_VALUE
    features[f'{region_name}_energy_max_jump'] = DEFAULT_VALUE
    features[f'{region_name}_stability_run_frac'] = DEFAULT_VALUE
    features[f'{region_name}_energy_oscillation'] = DEFAULT_VALUE
    features[f'{region_name}_energy_drift'] = DEFAULT_VALUE
    
    # Terminal
    features[f'{region_name}_5p_terminal_energy'] = DEFAULT_VALUE
    features[f'{region_name}_3p_terminal_energy'] = DEFAULT_VALUE
    
    # Motif-based
    features[f'{region_name}_ggg_count'] = 0
    features[f'{region_name}_uuu_count'] = 0
    
    return features


def extract_sequence_features(seq: str, prefix: str, is_mirna: bool = False, context: Dict = None) -> Dict[str, float]:
    """
    Extract region-specific features for a sequence.
    
    Parameters:
        seq: RNA sequence
        prefix: 'mre' or 'mirna'
        is_mirna: If True, uses seed region; otherwise uses 5' region
        context: Dictionary with mre_seq and mirna_seq for warning messages
    
    Returns:
        Dictionary with features for all regions
    """
    if not seq or len(seq) < 3:
        if context and seq and len(seq) > 0:
            print_warning_sequence(
                context.get('mre_seq', ''),
                context.get('mirna_seq', ''),
                f"{prefix.upper()} sequence too short: {seq} (len={len(seq)})",
                "SEQUENCE TOO SHORT"
            )
        
        if is_mirna:
            empty = {}
            empty.update(get_empty_region_features(f'{prefix}_seed'))
            empty.update(get_empty_region_features(f'{prefix}_3p'))
            return empty
        else:
            empty = {}
            empty.update(get_empty_region_features(f'{prefix}_5p'))
            empty.update(get_empty_region_features(f'{prefix}_3p'))
            return empty
    
    regions = get_regions(seq, is_mirna)
    features = {}
    
    if is_mirna:
        # miRNA: seed, 3p
        for region_key, region_name in [('seed', 'seed'), ('3p', '3p')]:
            region_seq = regions.get(region_key, '')
            region_features = extract_region_features(region_seq, f'{prefix}_{region_name}', context=context)
            features.update(region_features)
    else:
        # MRE: 5p, 3p
        for region_key, region_name in [('5p', '5p'), ('3p', '3p')]:
            region_seq = regions.get(region_key, '')
            region_features = extract_region_features(region_seq, f'{prefix}_{region_name}', context=context)
            features.update(region_features)
    
    return features


def extract_both_sequence_features(mre_seq: str, mirna_seq: str) -> Dict[str, float]:
    """
    Extract region-specific features for both MRE and miRNA.
    
    Returns:
        Dictionary with ~120 features (30 features × 4 regions)
    """
    context = {'mre_seq': mre_seq, 'mirna_seq': mirna_seq}
    features = {}
    
    # MRE features (5p, 3p)
    mre_features = extract_sequence_features(mre_seq, prefix='mre', is_mirna=False, context=context)
    features.update(mre_features)
    
    # miRNA features (seed, 3p)
    mirna_features = extract_sequence_features(mirna_seq, prefix='mirna', is_mirna=True, context=context)
    features.update(mirna_features)
    
    return features


def get_feature_names() -> List[str]:
    """Return list of all feature names."""
    sample = extract_both_sequence_features("ACGUACGUACGUACGUACGU", "UGCAUGCAUGCAUGCAUGCA")
    return sorted(sample.keys())


def add_sequence_features(
    df: pl.DataFrame,
    mre_col: str = 'mre_sequence',
    mirna_col: str = 'mirna_sequence'
) -> pl.DataFrame:
    """
    Add region-specific sequence features to a Polars DataFrame.
    """
    all_features = []
    
    mre_sequences = df[mre_col].to_list()
    mirna_sequences = df[mirna_col].to_list()
    
    # Reset problem counter
    global PROBLEM_COUNT
    PROBLEM_COUNT = 0
    
    print(f"\nProcessing {len(mre_sequences):,} sequences...")
    print(f"(Showing first {MAX_PRINT} problematic sequences)\n")
    
    for i, (mre_seq, mirna_seq) in enumerate(zip(mre_sequences, mirna_sequences)):
        if not mre_seq or not isinstance(mre_seq, str):
            mre_seq = ""
        if not mirna_seq or not isinstance(mirna_seq, str):
            mirna_seq = ""
            
        features = extract_both_sequence_features(mre_seq, mirna_seq)
        all_features.append(features)
        
        # Progress indicator every 1000 sequences
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1:,} sequences... ({PROBLEM_COUNT} problematic)")
            sys.stdout.flush()
    
    feature_df = pl.DataFrame(all_features)
    
    return pl.concat([df, feature_df], how='horizontal')


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description='Extract region-specific sequence features for MRE and miRNA sequences.'
    )
    parser.add_argument('input_file', nargs='?', help='Input file (parquet, csv, tsv)')
    parser.add_argument('-o', '--output', help='Output file path', default=None)
    parser.add_argument('--mre-col', help='MRE sequence column name', default='mre_sequence')
    parser.add_argument('--mirna-col', help='miRNA sequence column name', default='mirna_sequence')
    parser.add_argument('--max-warnings', type=int, default=20, help='Maximum warnings to print (default: 20)')
    parser.add_argument('--test', action='store_true', help='Run test mode')
    
    args = parser.parse_args()
    
    # Update MAX_PRINT from args
    MAX_PRINT = args.max_warnings
    
    if args.test or not args.input_file:
        print("Region-Specific Sequence Feature Extraction - Test Mode")
        print("=" * 70)
        
        # Test with problematic sequences
        print(f"\n--- Testing Problematic Sequences ---")
        test_cases = [
            ("ACGUACGUACGUACGU", "UGCAUGCAUGCAUGCA"),  # Normal
            ("ACG", "UGCAU"),  # Very short
            ("ACGUAC", "UGCAUGCA"),  # Short MRE
            ("ACGUACGUACGU", "UGC"),  # Short miRNA
            ("", "UGCAUGCA"),  # Empty MRE
            ("ACGUACGU", ""),  # Empty miRNA
        ]
        
        for mre, mirna in test_cases:
            print(f"\nTesting: MRE(len={len(mre)}), miRNA(len={len(mirna)})")
            features = extract_both_sequence_features(mre, mirna)
        
        print(f"\n\nTotal problematic sequences: {PROBLEM_COUNT}")
        sys.exit(0)
    
    # Process file
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(f"{input_path.stem}_region_features")
    
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"MRE sequence column: {args.mre_col}")
    print(f"miRNA sequence column: {args.mirna_col}")
    print("-" * 70)
    
    # Read file
    suffix = input_path.suffix.lower()
    if suffix == '.parquet':
        df = pl.read_parquet(input_path)
    elif suffix == '.csv':
        df = pl.read_csv(input_path, schema_overrides={"total_vec":pl.String})
    elif suffix in ['.tsv', '.txt']:
        df = pl.read_csv(input_path, separator='\t')
    else:
        print(f"Error: Unsupported format: {suffix}")
        sys.exit(1)
    
    print(f"Loaded {len(df):,} rows")
    
    # Check columns exist
    if args.mre_col not in df.columns:
        print(f"Error: Column '{args.mre_col}' not found")
        print(f"Available columns: {df.columns}")
        sys.exit(1)
    
    if args.mirna_col not in df.columns:
        print(f"Error: Column '{args.mirna_col}' not found")
        print(f"Available columns: {df.columns}")
        sys.exit(1)
    
    # Extract features
    print("Extracting region-specific sequence features...")
    result = add_sequence_features(df, args.mre_col, args.mirna_col)
    
    n_features = len(get_feature_names())
    print(f"\n{'='*70}")
    print(f"Completed! Added {n_features} features")
    print(f"Total problematic sequences: {PROBLEM_COUNT}")
    print("  - MRE: 5' half, 3' half (~60 features)")
    print("  - miRNA: seed (pos 2-8), 3' region (~60 features)")
    
    # Save
    out_suffix = output_path.suffix.lower()
    if out_suffix == '.parquet':
        result.write_parquet(output_path)
    else:
        result.write_csv(output_path)
    
    print(f"\nSaved to {output_path}")
