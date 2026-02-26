#!/usr/bin/env python3
"""
RNAduplex Best Structure Selector

Parses RNAduplex output and selects the best structure for each entry
based on biological scoring criteria matching the original R implementation.

Usage:
    python best_duplex.py <duplex_file> <mre_fasta> <mirna_fasta> <output_file>
"""

import re
import sys


def parse_fasta(file_path):
    """
    A simple FASTA parser that reads sequences into a list, maintaining order.
    """
    sequences = []
    with open(file_path, 'r') as f:
        seq = ""
        for line in f:
            if line.startswith('>'):
                if seq:
                    sequences.append(seq.upper().replace('U', 'T'))
                seq = ""
            else:
                seq += line.strip()
        if seq:  # Append the last sequence
            sequences.append(seq.upper().replace('U', 'T'))
    return sequences


def get_paired_bases(total_vec, mre_seq, mirna_seq, mre_coord_start, mirna_coord_start):
    """
    Extracts the actual nucleotide bases that are paired ('1') in the duplex.
    """
    paired_mre = []
    paired_mir = []
    # Adjust for 1-based indexing of FASTA and coordinates
    mre_ptr = mre_coord_start - 1
    mir_ptr = mirna_coord_start - 1
    for char in total_vec:
        if mre_ptr >= len(mre_seq) or mir_ptr >= len(mirna_seq):
            break
        if char == '1':  # Match
            paired_mre.append(mre_seq[mre_ptr])
            paired_mir.append(mirna_seq[mir_ptr])
            mre_ptr += 1
            mir_ptr += 1
        elif char in '2D':  # Mismatch or Dangling
            mre_ptr += 1
            mir_ptr += 1
        elif char in '3e':  # MRE Bulge
            mre_ptr += 1
        elif char in '4d':  # miRNA Bulge
            mir_ptr += 1
    return "".join(paired_mre), "".join(paired_mir)


def create_duplex_vectors(mre_struct, mirna_struct):
    """
    A precise translation of the C function's logic to create duplex vectors.
    """
    loop, i, j = [], 0, 0
    match_started = False
    while i < len(mre_struct) or j < len(mirna_struct):
        mre_char = mre_struct[i] if i < len(mre_struct) else '\0'
        mirna_char = mirna_struct[j] if j < len(mirna_struct) else '\0'
        if mre_char == '(' and mirna_char == ')':
            loop.append('1')
            match_started = True
            i += 1
            j += 1
        elif mre_char == '.' and mirna_char == '.':
            loop.append('2' if match_started else 'D')
            i += 1
            j += 1
        elif mre_char == '.' and mirna_char == ')':
            loop.append('3')
            i += 1  # MRE Bulge
        elif mre_char == '(' and mirna_char == '.':
            loop.append('4')
            j += 1  # miRNA Bulge
        elif mre_char == '\0' and mirna_char == '.':
            loop.append('d')
            j += 1  # miRNA dangling end
        elif mirna_char == '\0' and mre_char == '.':
            loop.append('e')
            i += 1  # MRE dangling end
        else:
            break
    return "".join(loop)


def calculate_score_and_features(structure_data, seed_length=9):
    """
    Calculates features and priority score matching the original R implementation exactly.
    
    R logic:
    1. binding.seed.region = seed.length - mirna.start + 1 (adjusted for where binding starts)
    2. priority.seed = matches in seed (extended by target bulge count, max 2)
       - Only subtract G:U wobbles at positions 2-8 if count > 1
    3. priority.loop = (target_bulges + mirna_bulges + mismatches) / 3
    4. priority = center + 3prime - loop_penalty - GU_outside_seed
       - Multiplied by seed quality (replacement, not cumulative)
    """
    mre_struct_reversed = structure_data['mre_struct'][::-1]
    total_vec = create_duplex_vectors(mre_struct_reversed, structure_data['mirna_struct'])
    structure_data['total_vec'] = total_vec
    structure_data['total_matches'] = total_vec.count('1')
    structure_data['total_mismatches'] = total_vec.count('2')
    
    # Count bulges (total in entire duplex)
    target_bulge_positions = total_vec.count('3')
    mirna_bulge_positions = total_vec.count('4')
    
    # Get miRNA binding coordinates (1-indexed)
    mirna_start = structure_data['mirna_coord_start']
    mirna_end = structure_data['mirna_coord_end']
    
    # === R LOGIC: Calculate binding.seed.region ===
    # binding.seed.region := seed.length - mirna.start + 1
    # [mirna.end < seed.length, binding.seed.region := mirna.end - mirna.start + 2]
    # [binding.seed.region <= 0, binding.seed.region := 1]
    binding_seed_region = seed_length - mirna_start + 1
    if mirna_end < seed_length:
        binding_seed_region = mirna_end - mirna_start + 2
    if binding_seed_region <= 0:
        binding_seed_region = 1
    
    # Find seed end index in duplex vector
    mirna_pos_counter = 0
    seed_end_idx = -1
    for idx, char in enumerate(total_vec):
        if char in '124Dd':
            mirna_pos_counter += 1
        if mirna_pos_counter == binding_seed_region:
            seed_end_idx = idx + 1
            break
    if seed_end_idx == -1:
        seed_end_idx = len(total_vec)
    
    seed_region_total = total_vec[:seed_end_idx]
    center_prime3_region_total = total_vec[seed_end_idx:]
    
    # Count target bulges in seed region
    target_bulge_in_seed = seed_region_total.count('3')
    
    # Base seed matches count
    num_matches_in_seed = seed_region_total.count('1')
    
    # === R LOGIC: Extend seed region for target bulges (max 2) ===
    if target_bulge_in_seed == 1:
        extended_seed_end = min(seed_end_idx + 1, len(total_vec))
        extended_seed_vec = total_vec[:extended_seed_end]
        num_matches_in_seed = extended_seed_vec.count('1')
    elif target_bulge_in_seed >= 2:
        extended_seed_end = min(seed_end_idx + 2, len(total_vec))
        extended_seed_vec = total_vec[:extended_seed_end]
        num_matches_in_seed = extended_seed_vec.count('1')
    
    # === Get paired bases with miRNA position tracking ===
    mre_seq_reversed = structure_data['mre_seq'][::-1]
    mirna_seq = structure_data['mirna_seq']
    
    # Calculate starting positions for traversal
    # MRE: binding starts at (length - mre_coord_end) in reversed sequence
    mre_ptr = len(structure_data['mre_seq']) - structure_data['mre_coord_end']
    mir_ptr = mirna_start - 1  # 0-indexed
    mirna_pos = mirna_start  # Track actual miRNA position (1-indexed)
    
    paired_mre_bases = []
    paired_mir_bases = []
    mirna_positions = []
    
    for char in total_vec:
        if mre_ptr >= len(mre_seq_reversed) or mir_ptr >= len(mirna_seq):
            break
        if char == '1':  # Match
            paired_mre_bases.append(mre_seq_reversed[mre_ptr])
            paired_mir_bases.append(mirna_seq[mir_ptr])
            mirna_positions.append(mirna_pos)
            mre_ptr += 1
            mir_ptr += 1
            mirna_pos += 1
        elif char in '2D':  # Mismatch or Dangling
            mre_ptr += 1
            mir_ptr += 1
            mirna_pos += 1
        elif char in '3e':  # MRE Bulge
            mre_ptr += 1
        elif char in '4d':  # miRNA Bulge
            mir_ptr += 1
            mirna_pos += 1
    
    # === R LOGIC: Count G:U wobbles at positions 2-8 only ===
    gu_wobbles_in_seed_2_8 = 0
    for i in range(len(paired_mre_bases)):
        if i < len(paired_mir_bases) and i < len(mirna_positions):
            mirna_pos = mirna_positions[i]
            # Only count G:U at positions 2-8 (not position 1, not position 9)
            if 2 <= mirna_pos <= 8:
                pair = {paired_mre_bases[i], paired_mir_bases[i]}
                if pair == {'G', 'T'}:  # Using T since FASTA parser converts U to T
                    gu_wobbles_in_seed_2_8 += 1
    
    # === Count G:U wobbles outside seed (positions > 8) ===
    gu_wobbles_minus_seed = 0
    for i in range(len(paired_mre_bases)):
        if i < len(paired_mir_bases) and i < len(mirna_positions):
            mirna_pos = mirna_positions[i]
            if mirna_pos > 8:
                pair = {paired_mre_bases[i], paired_mir_bases[i]}
                if pair == {'G', 'T'}:
                    gu_wobbles_minus_seed += 1
    
    # === R LOGIC: priority.seed only subtracts G:U if count > 1 ===
    priority_seed = num_matches_in_seed
    if gu_wobbles_in_seed_2_8 > 1:
        priority_seed = priority_seed - gu_wobbles_in_seed_2_8
    
    # Matches in center + 3' region
    total_matches_center_prime3 = center_prime3_region_total.count('1')
    
    # === R LOGIC: priority.loop uses total bulges ===
    priority_loop = (target_bulge_positions + mirna_bulge_positions + structure_data['total_mismatches']) / 3.0
    
    # === R LOGIC: Final priority (replacement, not cumulative) ===
    if priority_seed > 7:
        priority = 4 * priority_seed + total_matches_center_prime3 - priority_loop - gu_wobbles_minus_seed
    elif priority_seed > 6:
        priority = 3 * priority_seed + total_matches_center_prime3 - priority_loop - gu_wobbles_minus_seed
    elif priority_seed > 5:
        priority = 2 * priority_seed + total_matches_center_prime3 - priority_loop - gu_wobbles_minus_seed
    elif priority_seed > 4:
        priority = 1 * priority_seed + total_matches_center_prime3 - priority_loop - gu_wobbles_minus_seed
    else:
        priority = total_matches_center_prime3 - priority_loop - gu_wobbles_minus_seed
    
    structure_data['priority_score'] = priority
    return structure_data


def select_best_structure_for_entry(entry_lines, mre_seq, mirna_seq):
    """
    Parses, scores, and selects the best structure for one entry.
    """
    parsed_structures = []
    line_regex = re.compile(r"(.+?)\s+(\d+),(\d+)\s+:\s+(\d+),(\d+)\s+\((.+)\)")
    for line in entry_lines:
        match = line_regex.match(line.strip())
        if not match:
            continue
        full_struct, mre_start, mre_end, mir_start, mir_end, mfe = match.groups()
        mre_struct, mirna_struct = full_struct.split('&')
        parsed_structures.append({
            "original_line": line,
            "mre_struct": mre_struct,
            "mirna_struct": mirna_struct,
            "mfe": float(mfe),
            "mre_seq": mre_seq,
            "mirna_seq": mirna_seq,
            "mre_coord_start": int(mre_start),
            "mre_coord_end": int(mre_end),
            "mirna_coord_start": int(mir_start),
            "mirna_coord_end": int(mir_end)
        })
    if not parsed_structures:
        return "No valid structures found."
    
    scored_structures = [calculate_score_and_features(s) for s in parsed_structures]
    
    # Selection: max priority, then max total_matches, then min mfe (most negative)
    best_structure = sorted(
        scored_structures,
        key=lambda x: (x['priority_score'], x['total_matches'], -x['mfe']),
        reverse=True
    )[0]
    return best_structure['original_line']


def process_rna_duplex_output(raw_text, mre_sequences, mirna_sequences):
    """
    Main function to process the entire raw output, entry by entry.
    """
    # Filter out empty entries from trailing separators
    entries = [entry for entry in raw_text.strip().split('------------------------------') if entry.strip()]
    print(f"\nFound {len(entries)} duplex entries and {len(mre_sequences)} MRE sequences.")
    if len(entries) > len(mre_sequences) or len(entries) > len(mirna_sequences):
        print("Warning: Mismatch between number of entries and sequences. Check FASTA files.")
    best_structures = []
    for i, entry in enumerate(entries):
        if not entry.strip():
            continue
        lines = entry.strip().split('\n')
        header_info = lines[0]
        structure_lines = [line for line in lines[3:] if line.strip()]
        if not structure_lines:
            continue
        # Pass the corresponding sequences to the selection function
        if i < len(mre_sequences) and i < len(mirna_sequences):
            best_line = select_best_structure_for_entry(structure_lines, mre_sequences[i], mirna_sequences[i])
            best_structures.append(f"{header_info}\n>BEST_STRUCTURE:\n{best_line}\n")
    return "-----------\n".join(best_structures)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python select_best_duplex.py <duplex_file> <mre_fasta> <mirna_fasta> <output_file>")
        sys.exit(1)
    duplex_file, mre_fasta, mirna_fasta, output_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    try:
        print(f"Reading duplex results from '{duplex_file}'...")
        with open(duplex_file, 'r') as f:
            raw_duplex_data = f.read()
        print(f"Reading MRE sequences from '{mre_fasta}'...")
        mre_sequences = parse_fasta(mre_fasta)
        print(f"Reading miRNA sequences from '{mirna_fasta}'...")
        mirna_sequences = parse_fasta(mirna_fasta)
        print("Processing entries to find the best structures...")
        best_results = process_rna_duplex_output(raw_duplex_data, mre_sequences, mirna_sequences)
        with open(output_file, 'w') as f:
            f.write(best_results)
        print(f"\nSuccess! Results have been saved to '{output_file}'.")
    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e.filename}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
