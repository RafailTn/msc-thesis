#!/usr/bin/env python3
"""
Combine the MFE and ensemble IntaRNA runs into one row per pair.

The two runs answer different questions, and the columns split cleanly along that line:

  * The MFE run reports a concrete duplex - `hybrid_dp`, coordinates, and the energies
    OF THAT DUPLEX (`E`, `E_hybrid`, `ED_target`, `ED_query`, `Energy_norm`,
    `Energy_hybrid_norm`). Everything downstream - `total_vec`, the binding-type call,
    the Turner stacking energies - is computed from this structure, so these energies are
    taken from here and the `E` column means "the energy of the duplex we actually used".

    (The ensemble run also emits an `E`, but it is NOT that: its `hybrid_dp` is degenerate
    - `(......(&)......)`, just the outermost pair - because in partition-function mode
    there is no single structure to report. Its `E` is the ensemble free energy of the
    site. Taking it would put an energy in the row that describes a different object from
    the `hybrid_dp` beside it, which is what the previous coordinate-based merge did.)

  * The ensemble run contributes the partition-function quantities `Eall`, `Eall1`,
    `Eall2`, `Ealltotal`. These are properties of the SEQUENCE PAIR, not of any duplex:
    `Eall1`/`Eall2` are each strand's own intramolecular folding partition function, and
    `Eall = -RT ln( sum over ALL interactions exp(-E_i/RT) )` already contains every
    possible duplex, at every coordinate, with every structure. Verified: they are
    bit-identical whether IntaRNA is asked for 1, 3 or 10 sites, while `E` fans out across
    those sites; and the partial sum over the top-N reported sites converges up to `Eall`
    from above (-6.68 at N=1 -> -6.8882 at N=400, against a reported Eall of -6.89).

So the join is on the PAIR, not on coordinates. The old coordinate join demanded a
positional match in order to fetch values that are constant across all positions, and
silently dropped the ~3.5% of pairs where the MFE-best site happened not to appear among
the ensemble's suboptimals - a biased loss, since those are exactly the pairs where the
two models disagree about where the best site is.

`E_total` is then IntaRNA's own documented identity, evaluated at our `E`:

    E_total = E + Eall1 + Eall2          # IntaRNA: "total energy of an interaction including
                                         # the ensemble energies of intra-molecular structure
                                         # formation (E+Eall1+Eall2)"     [max err 0.000000]

`P_duplex` is deliberately NOT called `P_E`, because it is not IntaRNA's `P_E`:

    P_duplex = exp(-(E - Eall) / RT) = exp(-E/RT) / Zall

IntaRNA's `P_E` is the "probability of an interaction (site) within the considered ensemble",
i.e. `Z(S)/Zall`, where mode P reports `E(S) = -RT log Z(S)` - the partition function of the
whole SITE, summed over every structure in it (which is also why mode P's `hybrid_dp` is
degenerate: it "abstracts from individual inter-molecular base pairing"). Feeding the MFE
structure's energy into that ratio instead gives the probability of THAT ONE DUPLEX, which is
a different quantity - strictly smaller, since `Z(S) >= exp(-E_mfe/RT)`.

That is the quantity we actually want here: every other feature in the row describes the MFE
duplex, so its Boltzmann weight within the ensemble is the coherent companion. The ratio is
legitimate because `ED1`/`ED2` are identical between the two modes (measured: 0.000 difference),
so `E` and `Eall` are on the same energy scale. But it is a *different* number from IntaRNA's
`P_E`, so it gets a different name.

Coverage is 100% of pairs. A pair is only incomplete if the ensemble run found no
interaction for it at all, in which case the four pair-level columns (and the two derived
from them) are NaN and the row is tagged `energy_source == 'mfe_only'`;
`feature_extraction.py --fallback-report` lists those by chimeric sequence.
"""

import argparse

import numpy as np
import pandas as pd

# Gas constant x IntaRNA's default temperature (37 C), in kcal/mol. These are the exact
# constants IntaRNA/ViennaRNA use (RNA.GASCONST = 1.98717 cal/(mol K), RNA.K0 = 273.15),
# verified by solving RT = -Eall / ln(Zall) from IntaRNA's own output: 0.616321 across every
# row, matching this value. Getting them right makes P_duplex reproduce IntaRNA's P_E to ~1e-4.
RT_37C = 1.98717e-3 * (273.15 + 37.0)

# Properties of the sequence pair, not of any one duplex. Fetched per pair.
PAIR_LEVEL_COLS = ['Eall', 'Eall1', 'Eall2', 'Ealltotal']

# Computed from the columns above rather than looked up. `P_duplex` is NOT IntaRNA's `P_E`
# (see the module docstring); the MFE-mode `P_E` column is a 0.0 placeholder and is dropped.
DERIVED_COLS = ['E_total', 'P_duplex']
STALE_MFE_COLS = ['P_E']


def merge_intarna_outputs(
    mfe_file: str,
    ensemble_file: str,
    output_file: str,
    sep: str = '\t',
    pair_cols: list = None,
) -> tuple:
    if pair_cols is None:
        pair_cols = ['pair_index']

    print(f"Loading MFE file:      {mfe_file}")
    mfe_df = pd.read_csv(mfe_file, sep=sep)
    print(f"  Rows: {len(mfe_df)}")

    print(f"Loading ensemble file: {ensemble_file}")
    ens_df = pd.read_csv(ensemble_file, sep=sep)
    print(f"  Rows: {len(ens_df)}")

    missing = [c for c in pair_cols if c not in mfe_df.columns or c not in ens_df.columns]
    if missing:
        raise ValueError(f"Pair key column(s) {missing} not present in both files. "
                         f"MFE has {mfe_df.columns.tolist()}")

    if 'status' in ens_df.columns:
        ens_df = ens_df[ens_df['status'] != 'no_interactions']

    available = [c for c in PAIR_LEVEL_COLS if c in ens_df.columns]
    if not available:
        raise ValueError(f"None of {PAIR_LEVEL_COLS} in the ensemble file. Was it run "
                         f"with --ensemble? MFE mode does not compute them.")

    # The pair-level claim is load-bearing, so check it rather than trust it: if any of
    # these varies across a pair's suboptimal sites, it is not pair-level and this whole
    # merge is invalid.
    nunique = ens_df.groupby(pair_cols)[available].nunique()
    varying = {c: int((nunique[c] > 1).sum()) for c in available if (nunique[c] > 1).any()}
    if varying:
        raise ValueError(
            f"Expected {available} to be constant within a pair, but they vary: {varying}. "
            f"They would then be site-specific and a pair-level join would mix sites. "
            f"Investigate before proceeding."
        )
    print(f"  Verified pair-level (constant across each pair's sites): {available}")

    ens_pair = ens_df[pair_cols + available].drop_duplicates(subset=pair_cols)

    # Take nothing else from the ensemble: E/E_hybrid/ED_* stay as the MFE run reported
    # them, so they describe the same duplex as hybrid_dp. The MFE run's placeholder 0.0
    # columns for the ensemble-only quantities are dropped rather than carried.
    drop = [c for c in available + STALE_MFE_COLS if c in mfe_df.columns]
    merged_df = pd.merge(mfe_df.drop(columns=drop), ens_pair, on=pair_cols, how='left')

    has_ens = merged_df['Eall'].notna()
    merged_df['energy_source'] = np.where(has_ens, 'ensemble', 'mfe_only')

    # IntaRNA's own identity, evaluated at our E (the MFE duplex).
    merged_df['E_total'] = merged_df['E'] + merged_df['Eall1'] + merged_df['Eall2']

    # Boltzmann weight of THIS duplex within the whole interaction ensemble. Not IntaRNA's
    # site-level P_E - see the module docstring. Eall <= E by construction, so the exponent
    # is <= 0; the clip only guards 2-decimal storage rounding pushing E a hair below Eall.
    merged_df['P_duplex'] = np.clip(
        np.exp(-(merged_df['E'] - merged_df['Eall']) / RT_37C), 0.0, 1.0)

    n_missing = int((~has_ens).sum())
    stats = {
        'mfe_rows': len(mfe_df),
        'merged_rows': len(merged_df),
        'with_ensemble': int(has_ens.sum()),
        'mfe_only': n_missing,
    }

    print(f"\n{'=' * 50}")
    print("RESULTS")
    print(f"{'=' * 50}")
    print(f"Pairs in:              {stats['mfe_rows']}")
    print(f"Pairs out:             {stats['merged_rows']}  (pair-level join, nothing dropped)")
    print(f"  with ensemble terms: {stats['with_ensemble']} "
          f"({stats['with_ensemble'] / max(len(merged_df), 1):.1%})")
    print(f"  mfe_only:            {n_missing}"
          + (f"  -> no ensemble interaction found; {PAIR_LEVEL_COLS + DERIVED_COLS} are NaN"
             if n_missing else ""))
    print("Duplex energies (E, E_hybrid, ED_*) taken from MFE, matching hybrid_dp.")
    print("E_total = E + Eall1 + Eall2 (IntaRNA's identity).")
    print("P_duplex = exp(-(E - Eall)/RT): Boltzmann weight of THIS duplex in the ensemble. "
          "Not IntaRNA's site-level P_E, which is dropped.")

    # na_rep so NaN survives the round-trip through csv.DictReader in feature_extraction;
    # an empty field would be read back as 0.0.
    merged_df.to_csv(output_file, sep=sep, index=False, na_rep='nan')
    print(f"\nSaved to: {output_file}")

    return merged_df, stats


def main():
    parser = argparse.ArgumentParser(
        description="Combine MFE and ensemble IntaRNA outputs on the sequence pair.")
    parser.add_argument("--mfe", "-m", required=True,
                        help="MFE-mode output, one best duplex per pair (best_intarna.py)")
    parser.add_argument("--ensemble", "-e", required=True,
                        help="Ensemble-mode output (intarna_parallel.py --ensemble). Only the "
                             "pair-level columns are read, and those are identical on every "
                             "reported site, so `-n 1` is sufficient - suboptimals here are "
                             "wasted work. A larger -n is accepted and ignored.")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--sep", default="\t")
    parser.add_argument("--pair-cols", nargs="+", default=["pair_index"],
                        help="Columns identifying a sequence pair (default: pair_index)")

    args = parser.parse_args()

    merge_intarna_outputs(
        mfe_file=args.mfe,
        ensemble_file=args.ensemble,
        output_file=args.output,
        sep=args.sep,
        pair_cols=args.pair_cols,
    )


if __name__ == "__main__":
    main()
