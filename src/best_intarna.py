#!/usr/bin/env python3
"""
IntaRNA Best Duplex Selector

Scores and selects the best interaction per sequence pair from existing
IntaRNA output based on biological criteria (seed quality, G:U wobbles, etc.)

This script does NOT run IntaRNA - it only processes existing output files.

Usage:
    python best_intarna.py \
        --intarna intarna_output.csv \
        --mre-fasta mre_sequences.fasta \
        --mirna-fasta mirna_sequences.fasta \
        --output best_results.tsv
"""

import argparse
import sys
import csv
import math
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Dict


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FastaRecord:
    """Represents a single FASTA sequence record."""
    header: str
    sequence: str
    
    @property
    def id(self) -> str:
        return self.header.lstrip('>').split()[0]


@dataclass
class IntaRNAInteraction:
    """Container for a single IntaRNA interaction result."""
    target_id: str
    query_id: str
    start_target: int
    end_target: int
    start_query: int
    end_query: int
    subseq_dp: str
    hybrid_dp: str
    energy_total: float
    energy_hybrid: float = 0.0
    energy_ED1: float = 0.0
    energy_ED2: float = 0.0
    energy_all1: float = 0.0    
    energy_all2: float = 0.0    
    energy_all: float = 0.0    
    energy_all_total: float = 0.0    
    energy_total_total: float = 0.0    
    energy_norm: float = 0.0    
    energy_hybrid_norm: float = 0.0    
    p_e: float = 0.0    
    # Scoring fields
    priority_score: float = 0.0
    total_matches: int = 0
    total_mismatches: int = 0
    seed_matches: int = 0
    gu_wobbles_seed: int = 0
    gu_wobbles_other: int = 0
    total_vec: str = ""


@dataclass 
class PairResult:
    """Container for all interactions found for a sequence pair."""
    pair_index: int
    target_id: str
    query_id: str
    target_length: int
    query_length: int
    target_seq: str = ""
    query_seq: str = ""
    interactions: List[IntaRNAInteraction] = field(default_factory=list)
    best_interaction: Optional[IntaRNAInteraction] = None
    status: str = "success"


# =============================================================================
# SUB-OPTIMAL SITE STATISTICS
# =============================================================================
#
# IntaRNA is run with `-n 10 --outDeltaE 100`, so it reports up to ten sub-optimal
# sites per pair. Everything but the winner used to be discarded. These columns
# summarise the discarded ones: how many alternative sites the pair has, how
# dominant the chosen one is, and how dispersed they are along the target.
#
# CAVEAT for interpretation: `subopt_n_sites` saturates at IntaRNA's `-n`, so it is
# "number of sites, censored at 10", not an unbounded count. Any comparison across
# runs is only valid if `-n` and `--outDeltaE` match.

# Energy window (kcal/mol) for "how many sites are nearly as good as the best one".
NEAR_OPTIMAL_KCAL = 1.0

# A site counts as seed-like at >= 6 paired bases in the seed region (a 6mer seed),
# using the same extended-seed definition as calculate_interaction_score.
SEEDLIKE_MIN_MATCHES = 6

SUBOPT_STAT_COLS = [
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


def compute_subopt_stats(interactions: List[IntaRNAInteraction],
                         best: Optional[IntaRNAInteraction]) -> Dict[str, float]:
    """Summarise the full set of sub-optimal sites reported for one pair.

    All interactions must already have been scored (calculate_interaction_score
    mutates them in place, so a prior select_best_interaction call is enough).

    Statistics that need at least two sites to be defined (a gap, a spread) are
    emitted as NaN when the pair has only one, matching the convention in
    feature_extraction._energy_series_features: NaN keeps "undefined" off the
    numeric axis instead of parking it mid-distribution at 0.0. Counts and spans
    are genuinely 0 for a single site and are emitted as such.

    Note that `best` is selected by priority_score, not by energy, so the selected
    site need not be the energy-minimal one - `subopt_E_delta_selected` measures
    exactly that disagreement, in kcal/mol.
    """
    out = {c: float('nan') for c in SUBOPT_STAT_COLS}
    if not interactions:
        return out

    energies = sorted(i.energy_total for i in interactions)
    n = len(energies)
    e_min = energies[0]

    out['subopt_n_sites'] = n
    out['subopt_E_min'] = round(e_min, 4)
    out['subopt_E_mean'] = round(sum(energies) / n, 4)
    out['subopt_n_within_1kcal'] = sum(1 for e in energies if e <= e_min + NEAR_OPTIMAL_KCAL)

    if n >= 2:
        out['subopt_E_gap'] = round(energies[1] - e_min, 4)
        mean = sum(energies) / n
        out['subopt_E_std'] = round(math.sqrt(sum((e - mean) ** 2 for e in energies) / n), 4)

        priorities = sorted((i.priority_score for i in interactions), reverse=True)
        out['subopt_priority_gap'] = round(priorities[0] - priorities[1], 4)

    if best is not None:
        out['subopt_E_delta_selected'] = round(best.energy_total - e_min, 4)

    out['subopt_frac_seedlike'] = round(
        sum(1 for i in interactions if i.seed_matches >= SEEDLIKE_MIN_MATCHES) / n, 4)

    starts = [i.start_target for i in interactions]
    out['subopt_target_span'] = max(starts) - min(starts)
    out['subopt_n_distinct_starts'] = len(set(starts))

    return out


# =============================================================================
# DUPLEX VECTOR FUNCTIONS
# =============================================================================

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


def get_paired_bases(total_vec: str, target_seq: str, query_seq: str,
                     target_start: int, query_start: int) -> tuple:
    """Extract paired nucleotide bases from the duplex."""
    paired_target = []
    paired_query = []
    
    target_ptr = target_start - 1
    query_ptr = query_start - 1
    
    for char in total_vec:
        if target_ptr >= len(target_seq) or query_ptr >= len(query_seq):
            break
        
        if char == '1':
            paired_target.append(target_seq[target_ptr])
            paired_query.append(query_seq[query_ptr])
            target_ptr += 1
            query_ptr += 1
        elif char in '2D':
            target_ptr += 1
            query_ptr += 1
        elif char in '3e':
            target_ptr += 1
        elif char in '4d':
            query_ptr += 1
    
    return "".join(paired_target), "".join(paired_query)


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def calculate_interaction_score(interaction: IntaRNAInteraction,
                                 full_target_seq: str,
                                 full_query_seq: str,
                                 seed_last_pos: int = 8) -> IntaRNAInteraction:
    """
    Calculate scoring features for an IntaRNA interaction.

    `seed_last_pos` is the LAST miRNA position counted as seed, 1-based and inclusive -
    not a length, despite what this parameter used to be called. The distinction is not
    academic: the formula below is `end - start + 1`, so the two readings agree only when
    the duplex starts at miRNA position 1, which is 33% of real duplexes. Naming it a
    length is what let 8 and 9 both look defensible in different parts of this file.

    8 means positions 1-8, matching feature_extraction's `seed_matches`
    (`get_mirna_region_vector(total_vec, 0, 8, ...)`). Position 9 belongs to the separate
    9mer test there, not to the seed.

    Scoring logic matches the original R implementation exactly:
    1. binding.seed.region = seed_last_pos - mirna.start + 1 (adjusted for where binding starts)
    2. priority.seed = matches in seed region
       - Extended by target bulge count (max 2)
       - Only subtract G:U wobbles at positions 2-8 if count > 1
    3. priority.loop = (target_bulges + mirna_bulges + mismatches) / 3
    4. priority = center + 3prime - loop_penalty - GU_outside_seed
       - Multiplied by seed quality (replacement, not cumulative)
    """
    # Parse structure from hybridDP: "target_struct&query_struct"
    if '&' not in interaction.hybrid_dp:
        return interaction
    
    parts = interaction.hybrid_dp.split('&')
    if len(parts) != 2:
        return interaction
    
    target_struct = parts[0]
    query_struct = parts[1]
    
    # Parse subsequences from subseqDP: "target_subseq&query_subseq"
    if '&' in interaction.subseq_dp:
        subseq_parts = interaction.subseq_dp.split('&')
        target_subseq = subseq_parts[0].upper().replace('T', 'U')
        query_subseq = subseq_parts[1].upper().replace('T', 'U')
    else:
        target_subseq = full_target_seq[interaction.start_target-1:interaction.end_target].upper().replace('T', 'U')
        query_subseq = full_query_seq[interaction.start_query-1:interaction.end_query].upper().replace('T', 'U')
    
    # Get miRNA binding coordinates (1-indexed)
    mirna_start = interaction.start_query
    mirna_end = interaction.end_query
    
    # Reverse target structure for proper miRNA-mRNA alignment
    target_struct_reversed = target_struct[::-1]
    target_subseq_reversed = target_subseq[::-1]
    
    # Create duplex vector
    total_vec = create_duplex_vectors(target_struct_reversed, query_struct)
    interaction.total_vec = total_vec
    
    # Count basic features
    interaction.total_matches = total_vec.count('1')
    interaction.total_mismatches = total_vec.count('2')
    
    # Count bulges (total in entire duplex)
    target_bulge_positions = total_vec.count('3')
    mirna_bulge_positions = total_vec.count('4')
    
    # === R LOGIC: Calculate binding.seed.region ===
    # binding.seed.region := seed.length - mirna.start + 1
    # [mirna.end < seed.length, binding.seed.region := mirna.end - mirna.start + 2]
    # [binding.seed.region <= 0, binding.seed.region := 1]
    binding_seed_region = seed_last_pos - mirna_start + 1
    if mirna_end < seed_last_pos:
        binding_seed_region = mirna_end - mirna_start + 2
    if binding_seed_region <= 0:
        binding_seed_region = 1
    
    # Find seed end index in duplex vector based on binding_seed_region
    query_pos_counter = 0
    seed_end_idx = -1
    
    for idx, char in enumerate(total_vec):
        if char in '124Dd':  # Characters that consume miRNA position
            query_pos_counter += 1
        if query_pos_counter == binding_seed_region:
            seed_end_idx = idx + 1
            break
    
    if seed_end_idx == -1:
        seed_end_idx = len(total_vec)
    
    seed_region_vec = total_vec[:seed_end_idx]
    center_prime3_vec = total_vec[seed_end_idx:]
    
    # Count target bulges in seed region
    target_bulge_in_seed = seed_region_vec.count('3')
    
    # === Get paired bases with miRNA position tracking for G:U analysis ===
    paired_target_bases = []
    paired_query_bases = []
    mirna_positions = []  # Track which miRNA position each pair corresponds to
    
    target_ptr = 0
    query_ptr = 0
    mirna_pos = mirna_start  # Start from actual miRNA binding start position
    
    for char in total_vec:
        if target_ptr >= len(target_subseq_reversed) or query_ptr >= len(query_subseq):
            break
        
        if char == '1':  # Match
            paired_target_bases.append(target_subseq_reversed[target_ptr])
            paired_query_bases.append(query_subseq[query_ptr])
            mirna_positions.append(mirna_pos)
            target_ptr += 1
            query_ptr += 1
            mirna_pos += 1
        elif char in '2D':  # Mismatch or Dangling
            target_ptr += 1
            query_ptr += 1
            mirna_pos += 1
        elif char in '3e':  # Target bulge
            target_ptr += 1
        elif char in '4d':  # miRNA bulge
            query_ptr += 1
            mirna_pos += 1
    
    # === Count seed matches (base count) ===
    num_matches_in_seed = seed_region_vec.count('1')
    
    # === R LOGIC: Extend seed region for target bulges (max 2) ===
    if target_bulge_in_seed == 1:
        extended_seed_end = min(seed_end_idx + 1, len(total_vec))
        extended_seed_vec = total_vec[:extended_seed_end]
        num_matches_in_seed = extended_seed_vec.count('1')
    elif target_bulge_in_seed >= 2:
        extended_seed_end = min(seed_end_idx + 2, len(total_vec))
        extended_seed_vec = total_vec[:extended_seed_end]
        num_matches_in_seed = extended_seed_vec.count('1')
    
    interaction.seed_matches = num_matches_in_seed
    
    # === R LOGIC: Count G:U wobbles at positions 2-8 only ===
    gu_wobbles_in_seed_2_8 = 0
    for i in range(len(paired_target_bases)):
        if i < len(paired_query_bases) and i < len(mirna_positions):
            mirna_pos = mirna_positions[i]
            # Only count G:U at positions 2-8 (not position 1, not position 9)
            if 2 <= mirna_pos <= 8:
                pair = {paired_target_bases[i], paired_query_bases[i]}
                if pair == {'G', 'U'}:
                    gu_wobbles_in_seed_2_8 += 1
    interaction.gu_wobbles_seed = gu_wobbles_in_seed_2_8
    
    # === Count G:U wobbles outside seed (positions > 8) ===
    gu_wobbles_outside_seed = 0
    for i in range(len(paired_target_bases)):
        if i < len(paired_query_bases) and i < len(mirna_positions):
            mirna_pos = mirna_positions[i]
            if mirna_pos > 8:
                pair = {paired_target_bases[i], paired_query_bases[i]}
                if pair == {'G', 'U'}:
                    gu_wobbles_outside_seed += 1
    interaction.gu_wobbles_other = gu_wobbles_outside_seed
    
    # === R LOGIC: priority_seed only subtracts G:U if count > 1 ===
    priority_seed = num_matches_in_seed
    if gu_wobbles_in_seed_2_8 > 1:
        priority_seed = priority_seed - gu_wobbles_in_seed_2_8
    
    # === Calculate priority_loop penalty ===
    priority_loop_penalty = (target_bulge_positions + mirna_bulge_positions + interaction.total_mismatches) / 3.0
    
    # === Count matches in center and 3' regions ===
    total_matches_center_prime3 = center_prime3_vec.count('1')
    
    # === R LOGIC: Final priority (replacement, not cumulative) ===
    if priority_seed > 7:
        priority = 4 * priority_seed + total_matches_center_prime3 - priority_loop_penalty - gu_wobbles_outside_seed
    elif priority_seed > 6:
        priority = 3 * priority_seed + total_matches_center_prime3 - priority_loop_penalty - gu_wobbles_outside_seed
    elif priority_seed > 5:
        priority = 2 * priority_seed + total_matches_center_prime3 - priority_loop_penalty - gu_wobbles_outside_seed
    elif priority_seed > 4:
        priority = 1 * priority_seed + total_matches_center_prime3 - priority_loop_penalty - gu_wobbles_outside_seed
    else:
        priority = total_matches_center_prime3 - priority_loop_penalty - gu_wobbles_outside_seed
    
    interaction.priority_score = priority
    
    return interaction


def select_best_interaction(interactions: List[IntaRNAInteraction],
                            full_target_seq: str,
                            full_query_seq: str,
                            ensemble: bool = True,
                            seed_last_pos: int = 8) -> Optional[IntaRNAInteraction]:
    """
    Score all interactions and select the best one.

    Selection priority (sorted descending):
    1. priority_score
    2. total_matches
    3. -energy_total (more negative = better)

    `ensemble` only swaps which energy breaks a tie. In an MFE-mode run it is a
    distinction without a difference: `Ealltotal` is not in that run's requested column
    set, so it parses as 0.0 on every interaction (verified across 40,000 of them) and
    the ensemble branch sorts on a constant. Ties then fall through to Python's stable
    sort, i.e. IntaRNA's emission order, i.e. ascending E - which is exactly what the
    non-ensemble branch computes. Both branches select the same duplex.
    """
    if not interactions:
        return None

    # Score all interactions
    scored = []
    for inter in interactions:
        scored_inter = calculate_interaction_score(
            inter, full_target_seq, full_query_seq, seed_last_pos
        )
        scored.append(scored_inter)
    
    # Sort by priority_score, total_matches, then energy
    if ensemble:
        scored.sort(
            key=lambda x: (x.priority_score, x.total_matches, -x.energy_all_total),
            reverse=True
        )
    else:
        scored.sort(
            key=lambda x: (x.priority_score, x.total_matches, -x.energy_total),
            reverse=True
        )

    
    return scored[0]


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def parse_fasta(filepath: str) -> Iterator[FastaRecord]:
    """Parse a FASTA file and yield FastaRecord objects."""
    header = None
    sequence_lines = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('>'):
                if header is not None:
                    yield FastaRecord(header, ''.join(sequence_lines))
                header = line
                sequence_lines = []
            else:
                seq = line.replace(' ', '').upper().replace('T', 'U')
                sequence_lines.append(seq)
        
        if header is not None:
            yield FastaRecord(header, ''.join(sequence_lines))


def parse_intarna_csv(filepath: str, verbose: bool = False) -> Dict[int, List[IntaRNAInteraction]]:
    """
    Parse IntaRNA CSV output file and group interactions by pair index.
    
    Supports both:
    - Direct IntaRNA CSV output (semicolon-delimited)
    - Script TSV output (tab-delimited with pair_index column)
    """
    pair_interactions: Dict[int, List[IntaRNAInteraction]] = {}
    
    with open(filepath, 'r') as f:
        # Read first line to detect format
        first_line = f.readline().strip()
        f.seek(0)
        
        # Detect delimiter and format
        if '\t' in first_line and 'pair_index' in first_line:
            # Script TSV output format
            delimiter = '\t'
            has_pair_index = True
        elif ';' in first_line:
            # Direct IntaRNA CSV output
            delimiter = ';'
            has_pair_index = False
        else:
            # Try tab first, then comma
            delimiter = '\t' if '\t' in first_line else ','
            has_pair_index = 'pair_index' in first_line
        
        # Filter out comment lines
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith('#')]
        
        if not lines:
            return pair_interactions
        
        reader = csv.DictReader(lines, delimiter=delimiter)
        
        current_pair = 0
        last_target_id = None
        last_query_id = None
        
        for row in reader:
            # Get pair index
            if has_pair_index and 'pair_index' in row:
                pair_idx = int(row['pair_index'])
            else:
                # Infer pair index from target/query id changes
                target_id = row.get('id1', row.get('target_id', ''))
                query_id = row.get('id2', row.get('query_id', ''))
                
                if target_id != last_target_id or query_id != last_query_id:
                    current_pair += 1
                    last_target_id = target_id
                    last_query_id = query_id
                
                pair_idx = current_pair
            
            # Skip invalid rows
            start_target = row.get('start1', row.get('start_target', 'NA'))
            if start_target in ('NA', '', None):
                continue
            
            try:
                # Parse energy values
                e_total = float(row.get('E', row.get('E_total', 0)) or 0)
                e_hybrid = float(row.get('E_hybrid', 0) or 0)
                ed1 = float(row.get('ED1', row.get('ED_target', 0)) or 0)
                ed2 = float(row.get('ED2', row.get('ED_query', 0)) or 0)
                energy_all1 = float(row.get('Eall1', 0)) if row.get('Eall1') else 0.0
                energy_all2 = float(row.get('Eall2', 0)) if row.get('Eall2') else 0.0
                energy_all = float(row.get('Eall', 0)) if row.get('Eall') else 0.0
                energy_all_total = float(row.get('Ealltotal', 0)) if row.get('Ealltotal') else 0.0
                energy_total_total = float(row.get('E_total', 0)) if row.get('E_total') else 0.0
                energy_norm = float(row.get('Energy_norm', 0)) if row.get('Energy_norm') else 0.0
                energy_hybrid_norm = float(row.get('Energy_hybrid_norm', 0)) if row.get('Energy_hybrid_norm') else 0.0
                p_e = float(row.get('P_E', 0)) if row.get('P_E') else 0.0
                
                interaction = IntaRNAInteraction(
                    target_id=row.get('id1', row.get('target_id', '')),
                    query_id=row.get('id2', row.get('query_id', '')),
                    start_target=int(row.get('start1', row.get('start_target', 0))),
                    end_target=int(row.get('end1', row.get('end_target', 0))),
                    start_query=int(row.get('start2', row.get('start_query', 0))),
                    end_query=int(row.get('end2', row.get('end_query', 0))),
                    subseq_dp=row.get('subseqDP', row.get('subseq_dp', '')),
                    hybrid_dp=row.get('hybridDP', row.get('hybrid_dp', '')),
                    energy_total=e_total,
                    energy_hybrid=e_hybrid,
                    energy_ED1=ed1,
                    energy_ED2=ed2,
                    energy_all1=energy_all1,
                    energy_all2=energy_all2,
                    energy_all=energy_all,
                    energy_all_total=energy_all_total, 
                    energy_total_total=energy_total_total, 
                    energy_norm=energy_norm, 
                    energy_hybrid_norm=energy_hybrid_norm, 
                    p_e=p_e 
                )
                
                if pair_idx not in pair_interactions:
                    pair_interactions[pair_idx] = []
                pair_interactions[pair_idx].append(interaction)
                
            except (ValueError, KeyError) as e:
                if verbose:
                    print(f"Warning: Could not parse row: {e}", file=sys.stderr)
                continue
    
    return pair_interactions


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def write_best_results_tsv(results: List[PairResult], output_file: str):
    """Write best interaction per pair to TSV file.

    Rows are built by column name rather than positionally: the no-interaction
    branch used to emit 25 fields against a 30-column header, which silently shifted
    `status` into the `gu_wobbles_seed` column. Those rows are dropped downstream
    (feature_extraction skips `status == 'no_interactions'`), so it never bit, but
    adding columns positionally would have made the drift worse.

    Undefined numeric values are written as the literal `nan`, not `NA`: that is what
    merge_intarna's `na_rep='nan'` round-trips and what feature_extraction's
    safe_float parses back to NaN. `NA` would be coerced to 0.0 there.
    """
    headers = [
        'pair_index',
        'target_id', 'query_id',
        'target_length', 'query_length',
        'start_target', 'end_target',
        'start_query', 'end_query',
        'subseq_dp', 'hybrid_dp',
        'E', 'E_hybrid', 'ED_target', 'ED_query',
        'E_total', 'Eall', 'Eall1', 'Eall2', 'Ealltotal','P_E',
        'Energy_norm', 'Energy_hybrid_norm',
        'priority_score', 'total_matches', 'seed_matches',
        'gu_wobbles_seed', 'gu_wobbles_other',
    ] + SUBOPT_STAT_COLS + [
        'total_vec',
        'status'
    ]

    def fmt(value):
        if isinstance(value, float) and math.isnan(value):
            return 'nan'
        return value

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter='\t',
                                restval='nan', extrasaction='ignore')
        writer.writeheader()

        for pair in results:
            row = {
                'pair_index': pair.pair_index,
                'target_id': pair.target_id,
                'query_id': pair.query_id,
                'target_length': pair.target_length,
                'query_length': pair.query_length,
                'status': pair.status,
            }
            # Present for every pair, including those with no interaction at all
            # (all-NaN there, since there is nothing to summarise).
            row.update({k: fmt(v) for k, v in
                        compute_subopt_stats(pair.interactions,
                                             pair.best_interaction).items()})

            inter = pair.best_interaction
            if inter:
                row.update({
                    'target_id': inter.target_id,
                    'query_id': inter.query_id,
                    'start_target': inter.start_target,
                    'end_target': inter.end_target,
                    'start_query': inter.start_query,
                    'end_query': inter.end_query,
                    'subseq_dp': inter.subseq_dp,
                    'hybrid_dp': inter.hybrid_dp,
                    'E': f"{inter.energy_total:.2f}",
                    'E_hybrid': f"{inter.energy_hybrid:.2f}",
                    'ED_target': f"{inter.energy_ED1:.2f}",
                    'ED_query': f"{inter.energy_ED2:.2f}",
                    'E_total': f"{inter.energy_total_total:.2f}",
                    'Eall': f"{inter.energy_all:.2f}",
                    'Eall1': f"{inter.energy_all1:.2f}",
                    'Eall2': f"{inter.energy_all2:.2f}",
                    'Ealltotal': f"{inter.energy_all_total:.2f}",
                    'P_E': f"{inter.p_e:.2f}",
                    'Energy_norm': f"{inter.energy_norm:.2f}",
                    'Energy_hybrid_norm': f"{inter.energy_hybrid_norm:.2f}",
                    'priority_score': f"{inter.priority_score:.2f}",
                    'total_matches': inter.total_matches,
                    'seed_matches': inter.seed_matches,
                    'gu_wobbles_seed': inter.gu_wobbles_seed,
                    'gu_wobbles_other': inter.gu_wobbles_other,
                    'total_vec': inter.total_vec,
                })

            writer.writerow(row)


def print_summary(results: List[PairResult]):
    """Print summary of results."""
    print("\n" + "=" * 80)
    print("Best Duplex Selection Summary")
    print("=" * 80)
    
    total = len(results)
    with_best = sum(1 for r in results if r.best_interaction)
    
    print(f"\nTotal pairs: {total}")
    print(f"  With best selection: {with_best}")
    print(f"  No interactions: {total - with_best}")
    
    if with_best > 0:
        scores = [r.best_interaction.priority_score for r in results if r.best_interaction]
        energies = [r.best_interaction.energy_total for r in results if r.best_interaction]
        seed_matches = [r.best_interaction.seed_matches for r in results if r.best_interaction]
        
        print(f"\nPriority Score: Min={min(scores):.2f}, Max={max(scores):.2f}, Mean={sum(scores)/len(scores):.2f}")
        print(f"E_total:        Min={min(energies):.2f}, Max={max(energies):.2f}, Mean={sum(energies)/len(energies):.2f}")
        print(f"Seed matches:   Min={min(seed_matches)}, Max={max(seed_matches)}, Mean={sum(seed_matches)/len(seed_matches):.1f}")
    
    print("\n" + "-" * 80)
    print(f"{'Pair':<6} {'Target':<12} {'Query':<12} {'E_total':<10} {'Score':<8} {'Seed':<6} {'GU_s':<6}")
    print("-" * 80)
    
    for r in results[:20]:  # Show first 20
        if r.best_interaction:
            b = r.best_interaction
            print(f"{r.pair_index:<6} {b.target_id[:12]:<12} {b.query_id[:12]:<12} "
                  f"{b.energy_total:<10.2f} {b.priority_score:<8.2f} "
                  f"{b.seed_matches:<6} {b.gu_wobbles_seed:<6}")
        else:
            print(f"{r.pair_index:<6} {r.target_id[:12]:<12} {r.query_id[:12]:<12} "
                  f"{'N/A':<10} {'N/A':<8} {'N/A':<6} {'N/A':<6}")
    
    if total > 20:
        print(f"... and {total - 20} more pairs")
    
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Select best IntaRNA interaction per pair based on biological scoring.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script scores and selects the best interaction from existing IntaRNA output.
It does NOT run IntaRNA - use intarna_parallel.py for that.

Scoring criteria (in order of priority):
  1. priority_score: Composite score favoring seed quality
  2. total_matches: Number of base pairs
  3. energy_total: More negative = better

Input formats supported:
  - Direct IntaRNA CSV output (semicolon-delimited)
  - intarna_parallel.py TSV output (tab-delimited)

Examples:
    # From direct IntaRNA output
    python select_best_intarna.py \\
        --intarna intarna_raw.csv \\
        --mre-fasta mre.fasta \\
        --mirna-fasta mirna.fasta \\
        --output best.tsv

    # From intarna_parallel.py output (all suboptimals)
    python select_best_intarna.py \\
        --intarna all_interactions.tsv \\
        --mre-fasta mre.fasta \\
        --mirna-fasta mirna.fasta \\
        --output best.tsv
        """
    )
    
    parser.add_argument(
        '--intarna', '-i', required=True,
        help='Path to IntaRNA output file (CSV or TSV)')
    parser.add_argument(
        '--mre-fasta', '-t', required=True,
        help='Path to FASTA file containing target/MRE sequences')
    parser.add_argument(
        '--mirna-fasta', '-q', required=True,
        help='Path to FASTA file containing query/miRNA sequences')
    parser.add_argument(
        '--output', '-o', required=True,
        help='Path for output TSV file with best selections')
    parser.add_argument(
        '--seed-last-pos', '--seed-length', type=int, default=8, dest='seed_last_pos',
        help='Last miRNA position counted as seed when scoring; 1-based and inclusive, '
             'NOT a length (default: 8, i.e. positions 1-8, matching '
             'feature_extraction seed_matches). --seed-length is kept as an alias.')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Print progress information')
    parser.add_argument(
        '--no-summary', action='store_true',
        help='Suppress summary output')
    parser.add_argument(
        '--ensemble', action='store_true',
        help='ensemble mode')
    
    args = parser.parse_args()
    
    # Load sequences
    print("Loading sequences...", file=sys.stderr)
    targets = list(parse_fasta(args.mre_fasta))
    queries = list(parse_fasta(args.mirna_fasta))
    
    print(f"  Targets: {len(targets)}", file=sys.stderr)
    print(f"  Queries: {len(queries)}", file=sys.stderr)
    
    # Create sequence lookup by index and by ID
    target_seqs_by_idx = {i: t.sequence for i, t in enumerate(targets, 1)}
    query_seqs_by_idx = {i: q.sequence for i, q in enumerate(queries, 1)}
    target_seqs_by_id = {t.id: t.sequence for t in targets}
    query_seqs_by_id = {q.id: q.sequence for q in queries}
    target_ids_by_idx = {i: t.id for i, t in enumerate(targets, 1)}
    query_ids_by_idx = {i: q.id for i, q in enumerate(queries, 1)}
    
    # Parse IntaRNA output
    print(f"Parsing IntaRNA output: {args.intarna}", file=sys.stderr)
    pair_interactions = parse_intarna_csv(args.intarna, args.verbose)
    
    total_interactions = sum(len(v) for v in pair_interactions.values())
    print(f"  Pairs: {len(pair_interactions)}", file=sys.stderr)
    print(f"  Total interactions: {total_interactions}", file=sys.stderr)
    
    # Select best for each pair
    print("Scoring and selecting best interactions...", file=sys.stderr)
    results = []
    
    for pair_idx in sorted(pair_interactions.keys()):
        interactions = pair_interactions[pair_idx]
        
        if not interactions:
            continue
        
        # Get sequences (try by ID first, then by index)
        first_inter = interactions[0]
        target_seq = target_seqs_by_id.get(first_inter.target_id, 
                                            target_seqs_by_idx.get(pair_idx, ''))
        query_seq = query_seqs_by_id.get(first_inter.query_id,
                                          query_seqs_by_idx.get(pair_idx, ''))
        
        # Get IDs
        target_id = first_inter.target_id or target_ids_by_idx.get(pair_idx, f'target_{pair_idx}')
        query_id = first_inter.query_id or query_ids_by_idx.get(pair_idx, f'query_{pair_idx}')
        
        # Select best
        # Keyword arguments, deliberately. This call used to pass args.seed_length
        # positionally into `ensemble`, so the flag never reached seed_length at all and
        # every run to date used the default 8. That is why the flag's default is now 8:
        # wiring it up while leaving it at 9 would have silently re-picked the duplex on
        # 14% of rows and invalidated every feature CSV. Verified byte-identical.
        best = select_best_interaction(interactions, target_seq, query_seq,
                                       ensemble=args.ensemble,
                                       seed_last_pos=args.seed_last_pos)
        
        pair_result = PairResult(
            pair_index=pair_idx,
            target_id=target_id,
            query_id=query_id,
            target_length=len(target_seq),
            query_length=len(query_seq),
            target_seq=target_seq,
            query_seq=query_seq,
            interactions=interactions,
            best_interaction=best,
            status='success' if best else 'no_interactions'
        )
        results.append(pair_result)
        
        if args.verbose and pair_idx % 100 == 0:
            print(f"  Processed {pair_idx} pairs...", file=sys.stderr)
    
    # Handle pairs with no interactions in the file
    max_pair_in_file = max(pair_interactions.keys()) if pair_interactions else 0
    num_sequences = min(len(targets), len(queries))
    
    for pair_idx in range(1, num_sequences + 1):
        if pair_idx not in pair_interactions:
            target_id = target_ids_by_idx.get(pair_idx, f'target_{pair_idx}')
            query_id = query_ids_by_idx.get(pair_idx, f'query_{pair_idx}')
            target_seq = target_seqs_by_idx.get(pair_idx, '')
            query_seq = query_seqs_by_idx.get(pair_idx, '')
            
            pair_result = PairResult(
                pair_index=pair_idx,
                target_id=target_id,
                query_id=query_id,
                target_length=len(target_seq),
                query_length=len(query_seq),
                target_seq=target_seq,
                query_seq=query_seq,
                interactions=[],
                best_interaction=None,
                status='no_interactions'
            )
            results.append(pair_result)
    
    # Sort by pair index
    results.sort(key=lambda x: x.pair_index)
    
    # Write output
    print(f"Writing results to: {args.output}", file=sys.stderr)
    write_best_results_tsv(results, args.output)
    
    # Print summary
    if not args.no_summary:
        print_summary(results)
    
    print(f"\nDone! Best selections written to: {args.output}", file=sys.stderr)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
