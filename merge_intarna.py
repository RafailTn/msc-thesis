#!/usr/bin/env python3
"""
Merge MFE and ensemble IntaRNA outputs based on matching coordinates.

Keeps structure features from MFE, energies from ensemble.
"""

import pandas as pd
import argparse
from pathlib import Path


def merge_intarna_outputs(
    mfe_file: str,
    ensemble_file: str,
    output_file: str,
    sep: str = '\t',
    id_cols: list = None,
    coord_cols: list = None,
    ensemble_cols: list = None
) -> dict:
    """
    Merge MFE and ensemble IntaRNA outputs on matching coordinates.
    
    Args:
        mfe_file: Path to MFE mode output (has structure features)
        ensemble_file: Path to ensemble mode output (has Eall, P_E, etc.)
        output_file: Path for merged output
        sep: Column separator
        id_cols: Columns identifying the pair (e.g., ['id1', 'id2'])
        coord_cols: Coordinate columns to match on
        ensemble_cols: Columns to take from ensemble output
    
    Returns:
        Stats dictionary
    """
    # Defaults
    if coord_cols is None:
        coord_cols = ['start1', 'end1', 'start2', 'end2']
    
    if ensemble_cols is None:
        ensemble_cols = ['Eall', 'Eall1', 'Eall2', 'EallTotal', 'P_E']
    
    if id_cols is None:
        id_cols = ['id1', 'id2']
    
    # Load files
    print(f"Loading MFE file: {mfe_file}")
    mfe_df = pd.read_csv(mfe_file, sep=sep)
    print(f"  Rows: {len(mfe_df)}")
    
    print(f"Loading ensemble file: {ensemble_file}")
    ens_df = pd.read_csv(ensemble_file, sep=sep)
    print(f"  Rows: {len(ens_df)}")
    
    # Check which columns exist
    available_id_cols = [c for c in id_cols if c in mfe_df.columns and c in ens_df.columns]
    available_coord_cols = [c for c in coord_cols if c in mfe_df.columns and c in ens_df.columns]
    available_ensemble_cols = [c for c in ensemble_cols if c in ens_df.columns]
    
    print(f"\nID columns: {available_id_cols}")
    print(f"Coordinate columns: {available_coord_cols}")
    print(f"Ensemble columns to merge: {available_ensemble_cols}")
    
    if not available_coord_cols:
        raise ValueError(f"No coordinate columns found. MFE has: {mfe_df.columns.tolist()}")
    
    if not available_ensemble_cols:
        raise ValueError(f"No ensemble columns found. Ensemble has: {ens_df.columns.tolist()}")
    
    # Create merge key
    merge_cols = available_id_cols + available_coord_cols
    
    # Prepare ensemble df - only keep merge keys + ensemble columns
    ens_subset = ens_df[merge_cols + available_ensemble_cols].copy()
    
    # Remove duplicate ensemble columns from MFE if they exist
    mfe_cols_to_drop = [c for c in available_ensemble_cols if c in mfe_df.columns]
    if mfe_cols_to_drop:
        print(f"\nDropping from MFE (will use ensemble values): {mfe_cols_to_drop}")
        mfe_df = mfe_df.drop(columns=mfe_cols_to_drop)
    
    # Merge
    print(f"\nMerging on: {merge_cols}")
    merged_df = pd.merge(
        mfe_df,
        ens_subset,
        on=merge_cols,
        how='inner'
    )
    
    # Stats
    stats = {
        'mfe_rows': len(mfe_df),
        'ensemble_rows': len(ens_df),
        'merged_rows': len(merged_df),
        'mfe_only': len(mfe_df) - len(merged_df),
        'ensemble_only': len(ens_df) - len(merged_df),
        'match_rate_mfe': len(merged_df) / len(mfe_df) * 100 if len(mfe_df) > 0 else 0,
        'match_rate_ensemble': len(merged_df) / len(ens_df) * 100 if len(ens_df) > 0 else 0,
    }
    
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"MFE rows:      {stats['mfe_rows']}")
    print(f"Ensemble rows: {stats['ensemble_rows']}")
    print(f"Merged rows:   {stats['merged_rows']}")
    print(f"Match rate:    {stats['match_rate_mfe']:.1f}% of MFE, {stats['match_rate_ensemble']:.1f}% of ensemble")
    
    # Save
    merged_df.to_csv(output_file, sep=sep, index=False)
    print(f"\nSaved to: {output_file}")
    
    return merged_df, stats


def main():
    parser = argparse.ArgumentParser(
        description="Merge MFE and ensemble IntaRNA outputs on matching coordinates"
    )
    parser.add_argument(
        "--mfe", "-m",
        required=True,
        help="MFE mode output file (has structure)"
    )
    parser.add_argument(
        "--ensemble", "-e",
        required=True,
        help="Ensemble mode output file (has Eall, P_E)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output merged file"
    )
    parser.add_argument(
        "--sep",
        default="\t",
        help="Column separator (default: tab)"
    )
    parser.add_argument(
        "--id-cols",
        nargs="+",
        default=["target_id", "query_id"],
        help="ID columns (default: id1 id2)"
    )
    parser.add_argument(
        "--coord-cols",
        nargs="+",
        default=["start_target", "end_target", "start_query", "end_query"],
        help="Coordinate columns (default: start1 end1 start2 end2)"
    )
    parser.add_argument(
        "--ensemble-cols",
        nargs="+",
        default=["E", "E_hybrid", "ED_target", "ED_query", "Eall", "Eall1", "Eall2", "E_total", "Ealltotal", "P_E", "Energy_hybrid_norm", "Energy_norm"],
        help="Columns to take from ensemble output"
    )
    
    args = parser.parse_args()
    
    merge_intarna_outputs(
        mfe_file=args.mfe,
        ensemble_file=args.ensemble,
        output_file=args.output,
        sep=args.sep,
        id_cols=args.id_cols,
        coord_cols=args.coord_cols,
        ensemble_cols=args.ensemble_cols
    )


if __name__ == "__main__":
    main()
