#!/usr/bin/env python3
"""
Model-free test of the neighbor structural-opening hypothesis.

Hypothesis
----------
A bound miRNA can remodel local target secondary structure and raise
accessibility for a *nearby* site, so a positive that has a structure-opening
neighbour should be able to bind with *weaker direct pairing* (and lower own-site
accessibility) than a positive with no such neighbour.

This makes a falsifiable, leakage-free prediction that needs no model:

    positives WITH a nearby distinct site  →  systematically WEAKER pairing
    positives WITHOUT one                  →  stronger pairing

If that enrichment is absent, the mechanism is not operating in this dataset and
a coordinate-based second pass cannot conjure it. If it is present, the faithful
feature is neighbour-constrained accessibility (RNAplfold), not the model's own
predictions.

"Neighbour" = a *distinct* positive site (different genomic coordinate) within a
window on the same chr+strand. The relevant distance is set by the AGO2 footprint
(~50–60 nt, so co-occupancy needs ≳60–80 nt clearance) and the local folding
domain (~70–200 nt), NOT the 13–35 nt seed-to-seed *repression* spacing — two
RISC complexes cannot co-occupy that close.

Pairing strength is summarised by `_duplex_stats` from the CNN module (seed
pairs, Watson–Crick / wobble / mismatch counts, longest paired run) at each
pair's strongest antiparallel register — the same deterministic statistics the
error analysis uses, so "weak pairing" means the same thing here as there.

Usage
-----
  python cnn/cooperativity_analysis.py \
      --input data/AGO2_CLASH_Hejret2023_train_v7.tsv

  # also characterise false negatives, if you have a predictions TSV
  # (predict_cnn.py output: needs `prediction` + `label` + sequence columns)
  python cnn/cooperativity_analysis.py \
      --input data/AGO2_CLASH_Hejret2023_test_v7.tsv \
      --pred  predictions/hejret_test.tsv

  # dump the per-positive table (neighbour flags + duplex stats) for plotting
  python cnn/cooperativity_analysis.py --input ... --out-table coop_sites.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Flat when run from cnn/, package-style when imported as cnn.*
try:
    from cnn_branches_mirbind import (
        _encode_seqs, _duplex_stats, _parse_vector, MRE_LEN, MAX_MIRNA,
    )
except ImportError:
    from cnn.cnn_branches_mirbind import (  # type: ignore
        _encode_seqs, _duplex_stats, _parse_vector, MRE_LEN, MAX_MIRNA,
    )

# Default neighbour windows (centre-to-centre nt). The first is the headline
# window used for the main with/without split; the rest are reported as a sweep.
WINDOWS = (150, 80, 200, 300)

# Pairing-strength metrics and the direction the hypothesis predicts for the
# WITH-neighbour group. "weaker" means a structure-opening neighbour lets a site
# bind despite less direct complementarity; n_mm is the one that should go UP.
#   metric        -> ("lower" | "higher") expected for WITH-neighbour positives
PAIRING_METRICS = {
    "seed_pairs":  "lower",
    "total_pairs": "lower",     # n_wc + n_gu
    "n_wc":        "lower",
    "max_run":     "lower",
    "n_mm":        "higher",
}


def _read_table(path: str | Path) -> pd.DataFrame:
    sep = "\t" if str(path).endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)


def _mean_track(raw) -> float:
    """Mean of a per-base conservation track stored as a scalar or CSV string."""
    try:
        return float(raw)
    except (TypeError, ValueError):
        v = _parse_vector(raw, MRE_LEN)
        nz = v[v != 0.0]
        return float(nz.mean()) if nz.size else float("nan")


def _neighbour_flags(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """For every positive row, flag whether a *distinct* positive site sits
    within `window` nt (centre-to-centre) on the same chr+strand, and whether
    that neighbour is the same / a different miRNA family.

    Returned frame is indexed like `df` (positives only) with columns
    has_nb / has_nb_difffam / has_nb_samefam, n_nb (distinct neighbour coords).
    """
    pos = df[df["label"] == 1].copy()
    pos["_center"] = (pos["start"].to_numpy() + pos["end"].to_numpy()) // 2
    fam_col = "noncodingRNA_fam" if "noncodingRNA_fam" in pos.columns else None

    has_nb   = np.zeros(len(pos), dtype=bool)
    has_diff = np.zeros(len(pos), dtype=bool)
    has_same = np.zeros(len(pos), dtype=bool)
    n_nb     = np.zeros(len(pos), dtype=np.int32)
    row_pos  = {idx: i for i, idx in enumerate(pos.index)}

    for _, g in pos.groupby(["chr", "strand"], sort=False):
        # Collapse to distinct coordinates; remember each coord's family set and
        # which rows live there so flags can be written back per row.
        coords = g.groupby("_center")
        centers = np.array(sorted(coords.groups.keys()), dtype=np.int64)
        fams_at = {
            c: (set(sub[fam_col]) if fam_col else set())
            for c, sub in coords
        }
        rows_at = {c: list(sub.index) for c, sub in coords}

        for ci, c in enumerate(centers):
            lo = np.searchsorted(centers, c - window, "left")
            hi = np.searchsorted(centers, c + window, "right")
            nb_centers = [centers[j] for j in range(lo, hi) if j != ci]
            if not nb_centers:
                continue
            nb_fams = set().union(*(fams_at[nc] for nc in nb_centers))
            for idx in rows_at[c]:
                i = row_pos[idx]
                has_nb[i] = True
                n_nb[i]   = len(nb_centers)
                if fam_col:
                    own = next(iter(fams_at[c])) if len(fams_at[c]) == 1 else None
                    f = g.loc[idx, fam_col]
                    has_diff[i] = bool(nb_fams - {f})
                    has_same[i] = f in nb_fams

    return pd.DataFrame(
        {"has_nb": has_nb, "has_nb_difffam": has_diff,
         "has_nb_samefam": has_same, "n_nb": n_nb},
        index=pos.index,
    )


def _duplex_frame(df: pd.DataFrame, mirna_col: str, mre_col: str) -> pd.DataFrame:
    mi = _encode_seqs(df[mirna_col].astype(str).tolist(), MAX_MIRNA)
    ti = _encode_seqs(df[mre_col].astype(str).tolist(),   MRE_LEN)
    stats = _duplex_stats(mi, ti)
    out = pd.DataFrame(stats, index=df.index)
    out["total_pairs"] = out["n_wc"] + out["n_gu"]
    return out


def _cliffs_delta(a: np.ndarray, b: np.ndarray, u: float) -> float:
    """Cliff's delta from the Mann–Whitney U (group a vs b). In [-1, 1];
    positive means a tends to exceed b."""
    n1, n2 = len(a), len(b)
    return 2.0 * u / (n1 * n2) - 1.0 if n1 and n2 else float("nan")


def _compare(with_nb: pd.DataFrame, without_nb: pd.DataFrame,
             metrics: dict[str, str], label: str) -> None:
    n1, n0 = len(with_nb), len(without_nb)
    print(f"\n{label}")
    print(f"  WITH neighbour: n={n1:>7}   WITHOUT: n={n0:>7}")
    if n1 == 0 or n0 == 0:
        print("  (one group empty — nothing to compare)")
        return
    print(f"  {'metric':<12} {'med(with)':>10} {'med(wo)':>9} "
          f"{'Δmean':>8} {'δ':>7} {'p':>10}  predicted  verdict")
    for m, direction in metrics.items():
        a = with_nb[m].to_numpy(float)
        b = without_nb[m].to_numpy(float)
        dmean = a.mean() - b.mean()
        if HAS_SCIPY:
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            delta = _cliffs_delta(a, b, u)
        else:
            p, delta = float("nan"), float("nan")
        # Hypothesis holds if the WITH group moves in the predicted direction
        # AND the shift is significant.
        moved = (dmean < 0) if direction == "lower" else (dmean > 0)
        sig   = (p < 0.05) if p == p else False
        verdict = "✓ supports" if (moved and sig) else (
                  "· n.s." if moved else "✗ opposes")
        print(f"  {m:<12} {np.median(a):>10.2f} {np.median(b):>9.2f} "
              f"{dmean:>+8.2f} {delta:>+7.3f} {p:>10.2e}  "
              f"{direction:<9}  {verdict}")


def _fn_characterisation(df: pd.DataFrame, flags: pd.DataFrame,
                         duplex: pd.DataFrame, pred_path: str,
                         mirna_col: str, mre_col: str) -> None:
    """Among true positives, split FN vs TP and ask whether FNs are the
    neighbour-having, weak-paired sites the hypothesis points to."""
    pred = _read_table(pred_path)
    need = {"prediction", mirna_col, mre_col}
    miss = need - set(pred.columns)
    if miss:
        print(f"\n[--pred] skipping FN characterisation; missing {miss} "
              f"in {pred_path}")
        return
    key = [mirna_col, mre_col]
    pred = pred.drop_duplicates(subset=key)[key + ["prediction"]]
    pos = df[df["label"] == 1].merge(pred, on=key, how="left")
    pos.index = df[df["label"] == 1].index
    pos = pos.join(flags).join(duplex)
    scored = pos.dropna(subset=["prediction"])
    if scored.empty:
        print("\n[--pred] no positives matched the predictions file.")
        return
    fn = scored[scored["prediction"] == 0]
    tp = scored[scored["prediction"] == 1]
    print("\n" + "=" * 72)
    print(f"FALSE-NEGATIVE characterisation  (matched positives: {len(scored)})")
    print(f"  FN={len(fn)}  TP={len(tp)}  recall={len(tp)/max(len(scored),1):.3f}")
    if len(fn) and len(tp):
        print(f"  neighbour rate   FN={fn['has_nb'].mean():.3f}  "
              f"TP={tp['has_nb'].mean():.3f}   "
              f"(hypothesis: FN >= TP if opening rescues weak sites it misses)")
        # Are FNs weaker-paired than TPs (expected regardless), and is the
        # neighbour effect on FN-rate concentrated in weak-paired sites?
        _compare(fn, tp, PAIRING_METRICS, "FN vs TP pairing strength")
        weak = scored[scored["seed_pairs"] <= 4]
        if len(weak) > 20:
            wr_nb  = 1 - weak[weak["has_nb"]]["prediction"].mean()
            wr_non = 1 - weak[~weak["has_nb"]]["prediction"].mean()
            print(f"\n  Among weak-seed positives (seed_pairs<=4, n={len(weak)}): "
                  f"FN-rate WITH nb={wr_nb:.3f}  WITHOUT nb={wr_non:.3f}")
            print("  (hypothesis: a structure-opening neighbour should LOWER the "
                  "FN-rate of weak-seed sites)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="v7 TSV with coordinates + label.")
    ap.add_argument("--mirna-col", default="noncodingRNA")
    ap.add_argument("--mre-col",   default="gene")
    ap.add_argument("--window", type=int, default=WINDOWS[0],
                    help=f"headline neighbour window in nt (default {WINDOWS[0]}).")
    ap.add_argument("--pred", default=None,
                    help="predict_cnn.py output TSV; enables FN characterisation.")
    ap.add_argument("--out-table", default=None,
                    help="write the per-positive table (flags + duplex stats).")
    args = ap.parse_args()

    df = _read_table(args.input)
    for col in ("label", "chr", "start", "end", "strand",
                args.mirna_col, args.mre_col):
        if col not in df.columns:
            sys.exit(f"ERROR: missing column {col!r}. Have: {list(df.columns)}")
    n_pos = int((df["label"] == 1).sum())
    print(f"{Path(args.input).name}: {len(df)} rows, {n_pos} positive")

    # Neighbour-window sweep (prevalence only) so the headline split is in context.
    windows = [args.window] + [w for w in WINDOWS if w != args.window]
    print("\nNeighbour prevalence among positives (distinct site, same chr+strand):")
    flags_by_w = {}
    for w in windows:
        f = _neighbour_flags(df, w)
        flags_by_w[w] = f
        rate = f["has_nb"].mean()
        dr   = f["has_nb_difffam"].mean()
        sr   = f["has_nb_samefam"].mean()
        print(f"  within {w:>4} nt: any={rate:6.3f}  "
              f"diff-family={dr:6.3f}  same-family={sr:6.3f}")

    flags  = flags_by_w[args.window]
    duplex = _duplex_frame(df[df["label"] == 1], args.mirna_col, args.mre_col)
    joined = flags.join(duplex)

    # Optional per-MRE conservation, if present (sanity: openers might prefer
    # less-structured, less-conserved regions).
    for con in ("gene_phastCons", "gene_phyloP"):
        if con in df.columns:
            joined[con] = df.loc[joined.index, con].map(_mean_track).to_numpy()

    print("\n" + "=" * 72)
    print(f"PAIRING STRENGTH: positives WITH vs WITHOUT a neighbour "
          f"(window {args.window} nt)")
    print("  prediction: opening lets WITH-neighbour sites bind on WEAKER pairing")
    _compare(joined[joined["has_nb"]], joined[~joined["has_nb"]],
             PAIRING_METRICS, "any neighbour")
    # The structure-opener case the hypothesis is really about: a *different*
    # family bound nearby (not same-miRNA cooperativity).
    _compare(joined[joined["has_nb_difffam"]], joined[~joined["has_nb"]],
             PAIRING_METRICS, "different-family neighbour (vs no neighbour)")

    if args.pred:
        _fn_characterisation(df, flags, duplex, args.pred,
                             args.mirna_col, args.mre_col)

    if args.out_table:
        joined.insert(0, "label", 1)
        joined.to_csv(args.out_table, sep="\t")
        print(f"\nwrote per-positive table -> {args.out_table}")


if __name__ == "__main__":
    main()
