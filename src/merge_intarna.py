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
    ensemble_cols: list = None,
    fallback_on_mismatch: bool = False,
    mismatch_log: str = None,
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
    merged_df['coord_mismatch'] = False

    # -------------------------------------------------------------------------
    # Optional fallback: recover pairs where MFE and ensemble never agreed on
    # any set of coordinates.  For those pairs, take each mode's single
    # best-energy row and join them on ID columns only.
    n_mismatch_recovered = 0
    if fallback_on_mismatch and available_id_cols:
        # IDs that survived the exact-coordinate merge
        if len(merged_df) > 0:
            merged_id_set = set(
                zip(*[merged_df[c].astype(str).tolist() for c in available_id_cols])
            )
        else:
            merged_id_set = set()

        mfe_id_tuples = list(
            zip(*[mfe_df[c].astype(str).tolist() for c in available_id_cols])
        )
        unmatched_mask = pd.Series([t not in merged_id_set for t in mfe_id_tuples])
        unmatched_mfe = mfe_df[unmatched_mask.values].copy()

        print(f"\n  MFE pairs with no coordinate match : {unmatched_mask.sum()}")

        if len(unmatched_mfe) > 0:
            # Drop ensemble columns that were already stripped from mfe_df earlier
            unmatched_mfe = unmatched_mfe.drop(
                columns=[c for c in available_ensemble_cols if c in unmatched_mfe.columns],
                errors='ignore',
            )

            # Best MFE row per pair (lowest E; fall back to first row)
            if 'E' in unmatched_mfe.columns:
                unmatched_mfe = unmatched_mfe.copy()
                unmatched_mfe['_E_num'] = pd.to_numeric(unmatched_mfe['E'], errors='coerce')
                best_mfe_idx = unmatched_mfe.groupby(available_id_cols)['_E_num'].idxmin()
                best_mfe = unmatched_mfe.loc[best_mfe_idx].drop(columns=['_E_num'])
            else:
                best_mfe = unmatched_mfe.groupby(available_id_cols, as_index=False).first()

            # Best ensemble row per pair (lowest E; fall back to first row)
            ens_tmp = ens_df.copy()
            if 'E' in ens_tmp.columns:
                ens_tmp['_E_num'] = pd.to_numeric(ens_tmp['E'], errors='coerce')
                best_ens_idx = ens_tmp.groupby(available_id_cols)['_E_num'].idxmin()
                best_ens = ens_tmp.loc[best_ens_idx].drop(columns=['_E_num'])
            else:
                best_ens = ens_tmp.groupby(available_id_cols, as_index=False).first()

            ens_fallback_cols = [c for c in available_ensemble_cols if c in best_ens.columns]
            best_ens_subset = best_ens[available_id_cols + ens_fallback_cols].copy()

            fallback_df = pd.merge(best_mfe, best_ens_subset, on=available_id_cols, how='inner')
            fallback_df['coord_mismatch'] = True
            n_mismatch_recovered = len(fallback_df)

            merged_df = pd.concat([merged_df, fallback_df], ignore_index=True)

            print(f"  Recovered via ID-only fallback     : {n_mismatch_recovered}")

            if mismatch_log and n_mismatch_recovered > 0:
                fallback_df[available_id_cols].to_csv(mismatch_log, sep='\t', index=False)
                print(f"  Mismatch log written to            : {mismatch_log}")
        else:
            print("  No coordinate mismatches found.")

    # Stats (compute against the coord-exact count, before any fallback rows)
    n_exact = (merged_df['coord_mismatch'] == False).sum()  # noqa: E712
    stats = {
        'mfe_rows': len(mfe_df),
        'ensemble_rows': len(ens_df),
        'merged_rows': len(merged_df),
        'exact_coord_matches': int(n_exact),
        'fallback_recovered': n_mismatch_recovered,
        'match_rate_mfe': n_exact / len(mfe_df) * 100 if len(mfe_df) > 0 else 0,
        'match_rate_ensemble': n_exact / len(ens_df) * 100 if len(ens_df) > 0 else 0,
    }
    
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"MFE rows:             {stats['mfe_rows']}")
    print(f"Ensemble rows:        {stats['ensemble_rows']}")
    print(f"Coord-exact matches:  {stats['exact_coord_matches']}")
    print(f"Fallback recovered:   {stats['fallback_recovered']}")
    print(f"Total merged rows:    {stats['merged_rows']}")
    print(f"Match rate:           {stats['match_rate_mfe']:.1f}% of MFE, {stats['match_rate_ensemble']:.1f}% of ensemble")
    
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
    parser.add_argument(
        "--fallback-on-mismatch",
        action="store_true",
        help=(
            "For pairs where MFE and ensemble never agree on any coordinates, "
            "fall back to an ID-only merge using each mode's lowest-E row. "
            "Adds a 'coord_mismatch' column (True for recovered pairs)."
        ),
    )
    parser.add_argument(
        "--mismatch-log",
        default=None,
        help="Write recovered mismatch pair IDs (target_id, query_id) to this TSV file.",
    )

    args = parser.parse_args()

    merge_intarna_outputs(
        mfe_file=args.mfe,
        ensemble_file=args.ensemble,
        output_file=args.output,
        sep=args.sep,
        id_cols=args.id_cols,
        coord_cols=args.coord_cols,
        ensemble_cols=args.ensemble_cols,
        fallback_on_mismatch=args.fallback_on_mismatch,
        mismatch_log=args.mismatch_log,
    )


if __name__ == "__main__":
    main()
