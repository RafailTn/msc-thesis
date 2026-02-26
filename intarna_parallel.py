#!/usr/bin/env python3
"""
IntaRNA v3.x Parallel Processor (Multithreaded)

This script parses two FASTA files (mRNA sequences and miRNA sequences),
pairs them using the zip function, runs IntaRNA for each pair in parallel,
and extracts multiple suboptimal interactions with full energy decomposition
including accessibility.

Usage:
    python intarna_threaded.py mrna.fasta mirna.fasta [options]
"""

from asyncio import ensure_future
import subprocess
import argparse
import sys
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Iterator, List, Optional, Tuple
import os


@dataclass
class FastaRecord:
    """Represents a single FASTA sequence record."""
    header: str
    sequence: str
    
    @property
    def id(self) -> str:
        """Extract the sequence ID from the header."""
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
    def __str__(self) -> str:
        if self.energy_hybrid != 0 or self.energy_ED1 != 0 or self.energy_ED2 != 0:
            return (f"E={self.energy_total:.2f} "
                    f"(E_hybrid={self.energy_hybrid:.2f} + "
                    f"ED1={self.energy_ED1:.2f} + "
                    f"ED2={self.energy_ED2:.2f})")
        else:
            return f"E={self.energy_total:.2f}"


@dataclass 
class PairResult:
    """Container for all interactions found for a sequence pair."""
    pair_index: int
    target_id: str
    query_id: str
    target_length: int
    query_length: int
    interactions: List[IntaRNAInteraction] = field(default_factory=list)
    status: str = "success"
    error: str = ""
    raw_output: str = ""


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


def run_intarna(target_seq: str, query_seq: str,
                target_id: str = "target",
                query_id: str = "query",
                num_subopt: int = 10,
                delta_e: float = 100.0,
                ensemble:bool = False,
                debug: bool = False) -> tuple:
    """
    Run IntaRNA v3.x on a pair of sequences.
    
    Returns:
        Tuple of (stdout, stderr, command_used)
    """
    # Request specific columns including energy breakdown
    if ensemble:
        csv_cols = "id1,start1,end1,id2,start2,end2,subseqDP,hybridDP,P_E,Eall,Eall1,Eall2,Etotal,EallTotal,ED1,ED2,E,E_hybrid,E_hybridNorm,E_norm"
    else:
        csv_cols = "id1,start1,end1,id2,start2,end2,subseqDP,hybridDP,ED1,ED2,E,E_hybrid,E_hybridNorm,E_norm"

    
    cmd = [
        'IntaRNA',
        '--target', target_seq,
        '--query', query_seq,
        '--tId=' + target_id,
        '--qId=' + query_id,
        '--mode=M',
        '--accW=0',
        '--accL=0',
        '--outMode=C',
        '--outCsvCols=' + csv_cols,
        '-n', str(num_subopt),
        '--outDeltaE=' + str(delta_e),
        '--outOverlap=B',
        '--noSeed',
        '--outMaxE=100',
        '--outNoLP'
    ]
    if ensemble:
        cmd.append('--model=P')
    else:
        cmd.append('--model=S')
    
    cmd_str = ' '.join(cmd)
    
    if debug:
        print(f"DEBUG: Running command:\n  {cmd_str[:300]}...", file=sys.stderr)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if debug:
            print(f"DEBUG: Return code: {result.returncode}", file=sys.stderr)
            print(f"DEBUG: stdout:\n{result.stdout[:1500] if result.stdout else '(empty)'}", file=sys.stderr)
            if result.stderr:
                print(f"DEBUG: stderr:\n{result.stderr[:500]}", file=sys.stderr)
        
        return result.stdout, result.stderr, cmd_str
        
    except subprocess.TimeoutExpired:
        return "", "ERROR: IntaRNA timed out", cmd_str
    except FileNotFoundError:
        return "", "ERROR: IntaRNA not found.", cmd_str


def parse_intarna_csv(csv_output: str, target_id: str, query_id: str,
                      debug: bool = False) -> List[IntaRNAInteraction]:
    """Parse IntaRNA CSV output to extract interactions."""
    interactions = []
    
    if not csv_output or not csv_output.strip():
        if debug:
            print("DEBUG: Empty CSV output", file=sys.stderr)
        return interactions
    
    # Filter out comment/info lines (start with #)
    lines = [l for l in csv_output.strip().split('\n') if not l.startswith('#')]
    
    if debug:
        print(f"DEBUG: CSV has {len(lines)} data lines", file=sys.stderr)
        for i, line in enumerate(lines[:5]):
            print(f"DEBUG: Line {i}: {line[:150]}", file=sys.stderr)
    
    if len(lines) < 2:
        return interactions
    
    # IntaRNA uses semicolon delimiter
    try:
        reader = csv.DictReader(lines, delimiter=';')
        
        for row in reader:
            if debug:
                print(f"DEBUG: Parsing row: {dict(row)}", file=sys.stderr)
            
            try:
                # Get energy values, defaulting to 0 if not present
                e_total = float(row.get('E', 0)) if row.get('E') else 0.0
                e_hybrid = float(row.get('E_hybrid', 0)) if row.get('E_hybrid') else 0.0
                ed1 = float(row.get('ED1', 0)) if row.get('ED1') else 0.0
                ed2 = float(row.get('ED2', 0)) if row.get('ED2') else 0.0
                energy_norm = float(row.get('E_norm', 0)) if row.get('E_norm') else 0.0
                energy_hybrid_norm = float(row.get('E_hybridNorm', 0)) if row.get('E_hybridNorm') else 0.0
                energy_all1 = float(row.get('Eall1', 0)) if row.get('Eall1') else 0.0
                energy_all2 = float(row.get('Eall2', 0)) if row.get('Eall2') else 0.0
                energy_all = float(row.get('Eall', 0)) if row.get('Eall') else 0.0
                energy_all_total = float(row.get('EallTotal', 0)) if row.get('EallTotal') else 0.0
                energy_total_total = float(row.get('Etotal', 0)) if row.get('Etotal') else 0.0
                p_e = float(row.get('P_E', 0)) if row.get('P_E') else 0.0
                interaction = IntaRNAInteraction(
                    target_id=row.get('id1', target_id),
                    query_id=row.get('id2', query_id),
                    start_target=int(row.get('start1', 0)),
                    end_target=int(row.get('end1', 0)),
                    start_query=int(row.get('start2', 0)),
                    end_query=int(row.get('end2', 0)),
                    subseq_dp=row.get('subseqDP', ''),
                    hybrid_dp=row.get('hybridDP', ''),
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
                interactions.append(interaction)
                
            except (ValueError, KeyError) as e:
                if debug:
                    print(f"DEBUG: Failed to parse row: {e}", file=sys.stderr)
                continue
                
    except Exception as e:
        if debug:
            print(f"DEBUG: CSV parsing failed: {e}", file=sys.stderr)
    
    return interactions


def process_single_pair(
    pair_index: int,
    target: FastaRecord,
    query: FastaRecord,
    num_subopt: int,
    delta_e: float,
    debug: bool,
    ensemble:bool=False
) -> PairResult:
    """
    Process a single target-query pair through IntaRNA.
    
    This function is designed to be called by multiple threads.
    """
    pair_result = PairResult(
        pair_index=pair_index,
        target_id=target.id,
        query_id=query.id,
        target_length=len(target.sequence),
        query_length=len(query.sequence)
    )
    
    stdout, stderr, cmd = run_intarna(
        target.sequence, 
        query.sequence,
        target.id,
        query.id,
        num_subopt,
        delta_e,
        ensemble,
        debug
    )
    
    pair_result.raw_output = stdout
    
    # Check for errors (but ignore INFO messages)
    error_lines = [l for l in stderr.split('\n') if 'error' in l.lower() and not l.startswith('#')]
    if error_lines:
        pair_result.status = "failed"
        pair_result.error = '\n'.join(error_lines)
    elif not stdout.strip() or all(l.startswith('#') for l in stdout.strip().split('\n')):
        pair_result.status = "no_interactions"
    else:
        interactions = parse_intarna_csv(stdout, target.id, query.id, debug)
        pair_result.interactions = interactions
        
        if not interactions:
            pair_result.status = "no_interactions"
    
    return pair_result


class ProgressTracker:
    """Thread-safe progress tracking."""
    
    def __init__(self, total: int, verbose: bool = False, debug: bool = False):
        self.total = total
        self.completed = 0
        self.verbose = verbose
        self.debug = debug
        self.lock = Lock()
    
    def report(self, result: PairResult):
        """Report completion of a pair (thread-safe)."""
        with self.lock:
            self.completed += 1
            progress_pct = (self.completed / self.total) * 100
            
            if self.verbose or self.debug:
                status_msg = ""
                if result.status == "failed":
                    status_msg = f"FAILED: {result.error[:100]}"
                elif result.status == "no_interactions":
                    status_msg = "No interactions"
                else:
                    status_msg = f"Found {len(result.interactions)} interaction(s)"
                
                print(f"[{self.completed}/{self.total} {progress_pct:5.1f}%] "
                      f"Pair {result.pair_index}: {result.target_id} vs {result.query_id} -> {status_msg}",
                      file=sys.stderr)
            else:
                # Simple progress indicator
                print(f"\rProgress: {self.completed}/{self.total} ({progress_pct:.1f}%)", 
                      end='', file=sys.stderr)


def process_sequence_pairs(target_file: str, query_file: str,
                          num_subopt: int = 10,
                          delta_e: float = 15.0,
                          output_file: Optional[str] = None,
                          verbose: bool = False,
                          debug: bool = False,
                          ensemble:bool = False,
                          num_threads: int = 4) -> List[PairResult]:
    """
    Process pairs of target and query sequences through IntaRNA using multiple threads.
    
    Args:
        target_file: Path to FASTA file with target sequences
        query_file: Path to FASTA file with query sequences
        num_subopt: Number of suboptimal interactions to report
        delta_e: Energy range from MFE in kcal/mol
        output_file: Optional path for TSV output
        verbose: Print progress information
        debug: Print detailed debug info
        num_threads: Number of parallel threads to use
    
    Returns:
        List of PairResult objects
    """
    target_records = list(parse_fasta(target_file))
    query_records = list(parse_fasta(query_file))
    
    print(f"Loaded {len(target_records)} target sequences from {target_file}", file=sys.stderr)
    print(f"Loaded {len(query_records)} query sequences from {query_file}", file=sys.stderr)
    
    if len(target_records) != len(query_records):
        print(f"Warning: Sequence counts differ. Processing {min(len(target_records), len(query_records))} pairs.", 
              file=sys.stderr)
    
    # Prepare pairs
    pairs = list(zip(target_records, query_records))
    total_pairs = len(pairs)
    
    print(f"Processing {total_pairs} pairs using {num_threads} threads...", file=sys.stderr)
    
    # Progress tracking
    tracker = ProgressTracker(total_pairs, verbose, debug)
    
    # Results dict to maintain order
    results_dict = {}
    
    # Process pairs in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        future_to_index = {}
        for i, (target, query) in enumerate(pairs, 1):
            future = executor.submit(
                process_single_pair,
                i,
                target,
                query,
                num_subopt,
                delta_e,
                debug,
                ensemble
            )
            future_to_index[future] = i
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            pair_index = future_to_index[future]
            try:
                result = future.result()
                results_dict[pair_index] = result
                tracker.report(result)
            except Exception as e:
                # Handle unexpected errors
                error_result = PairResult(
                    pair_index=pair_index,
                    target_id=pairs[pair_index-1][0].id,
                    query_id=pairs[pair_index-1][1].id,
                    target_length=len(pairs[pair_index-1][0].sequence),
                    query_length=len(pairs[pair_index-1][1].sequence),
                    status="failed",
                    error=str(e)
                )
                results_dict[pair_index] = error_result
                tracker.report(error_result)
    
    # Clear progress line if not verbose
    if not verbose and not debug:
        print("", file=sys.stderr)
    
    # Sort results by original pair index
    results = [results_dict[i] for i in sorted(results_dict.keys())]
    
    if output_file:
        write_results_tsv(results, output_file)
        print(f"\nResults written to: {output_file}", file=sys.stderr)
    
    return results


def write_results_tsv(results: List[PairResult], output_file: str):
    """Write results to a TSV file."""
    headers = [
        'pair_index', 'interaction_rank',
        'target_id', 'query_id',
        'target_length', 'query_length',
        'start_target', 'end_target',
        'start_query', 'end_query',
        'subseq_dp', 'hybrid_dp',
        'E', 'E_hybrid', 'ED_target', 'ED_query',
        'E_total', 'Eall', 'Eall1', 'Eall2', 'Ealltotal','P_E',
        'Energy_norm', 'Energy_hybrid_norm',
        'status'
    ]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(headers)
        
        for pair in results:
            if pair.interactions:
                for rank, inter in enumerate(pair.interactions, 1):
                    writer.writerow([
                        pair.pair_index, rank,
                        inter.target_id, inter.query_id,
                        pair.target_length, pair.query_length,
                        inter.start_target, inter.end_target,
                        inter.start_query, inter.end_query,
                        inter.subseq_dp, inter.hybrid_dp,
                        f"{inter.energy_total:.2f}",
                        f"{inter.energy_hybrid:.2f}",
                        f"{inter.energy_ED1:.2f}",
                        f"{inter.energy_ED2:.2f}",
                        f"{inter.energy_total_total:.2f}",
                        f"{inter.energy_all:.2f}",
                        f"{inter.energy_all1:.2f}",
                        f"{inter.energy_all2:.2f}",
                        f"{inter.energy_all_total:.2f}",
                        f"{inter.p_e:.2f}",
                        f"{inter.energy_norm:.2f}",
                        f"{inter.energy_hybrid_norm:.2f}",
                        pair.status
                    ])
            else:
                writer.writerow([
                    pair.pair_index, 0,
                    pair.target_id, pair.query_id,
                    pair.target_length, pair.query_length,
                    'NA', 'NA', 'NA', 'NA', 'NA',
                    'NA', 'NA', 'NA','NA','NA',
                    'NA','NA','NA','NA','NA',
                    pair.status
                ])


def print_results_summary(results: List[PairResult]):
    """Print a summary of the results."""
    print("\n" + "="*80)
    print("IntaRNA Results Summary")
    print("="*80)
    
    total_pairs = len(results)
    successful = sum(1 for r in results if r.interactions)
    no_inter = sum(1 for r in results if r.status == "no_interactions")
    failed = sum(1 for r in results if r.status == "failed")
    total_interactions = sum(len(r.interactions) for r in results)
    
    print(f"\nTotal sequence pairs: {total_pairs}")
    print(f"  With interactions: {successful}")
    print(f"  No interactions found: {no_inter}")
    print(f"  Failed: {failed}")
    print(f"Total interactions found: {total_interactions}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description='Run IntaRNA v3.x on paired target/query sequences from FASTA files (multithreaded).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with 4 threads (default)
    python intarna_parallel.py mrna.fasta mirna.fasta
        """
    )
    
    parser.add_argument('target_fasta', 
                        help='FASTA file containing target/mRNA sequences')
    parser.add_argument('query_fasta', 
                        help='FASTA file containing query/miRNA sequences')
    parser.add_argument('-n', '--num-subopt', type=int, default=10,
                        help='Number of suboptimal interactions per pair (default: 10)')
    parser.add_argument('--deltaE', type=float, default=100.0,
                        help='Energy range from MFE in kcal/mol (default: 100)')
    parser.add_argument('-o', '--output', 
                        help='Output file for results (TSV format)')
    parser.add_argument('-t', '--threads', type=int, default=4,
                        help='Number of parallel threads (default: 4)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print progress information')
    parser.add_argument('--debug', action='store_true',
                        help='Print detailed debug information')
    parser.add_argument('--no-summary', action='store_true',
                        help='Suppress summary output')
    parser.add_argument('--ensemble', action='store_true',
                        help='Use ensemble mode of IntaRNA')
    
    args = parser.parse_args()
    
    # Validate thread count
    if args.threads < 1:
        print("Error: Thread count must be at least 1", file=sys.stderr)
        return 1
    
    # Use CPU count as reasonable upper bound suggestion
    cpu_count = os.cpu_count() or 4
    if args.threads > cpu_count * 2:
        print(f"Warning: Using {args.threads} threads on a system with {cpu_count} CPUs. "
              f"Consider using {cpu_count} to {cpu_count * 2} threads for optimal performance.",
              file=sys.stderr)
    
    results = process_sequence_pairs(
        args.target_fasta,
        args.query_fasta,
        args.num_subopt,
        args.deltaE,
        args.output,
        args.verbose,
        args.debug,
        args.ensemble,
        args.threads
    )
    
    if not args.no_summary:
        print_results_summary(results)
    
    if all(r.interactions for r in results):
        return 0
    elif any(r.interactions for r in results):
        return 1
    else:
        return 2


if __name__ == '__main__':
    sys.exit(main())
