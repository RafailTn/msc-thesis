#!/usr/bin/env python3
"""
Nunique (read-support) distributions by pairing extent.

Motivation: comparing Nunique between the classifier's categorical binding types
(seedless / 3'-compensatory vs canonical) shows only a negligible gap, and the
sign *flips inside* the canonical group — weak canonical subtypes (5mer,
6mer.GU.3prime) sit at or below seedless, while subtypes with long 3'
supplementary pairing (7mer.3prime, 9mer.3prime) sit well above it.  So the
ordering variable is pairing *extent*, not seed canonicity.  These plots put
extent on the x-axis directly.

Extent axes (best antiparallel register, same tables/argmax as binding_types.py):
  eff_3prime   WC pairs at miRNA position 9+  — 3' supplementary extent
  total_pairs  WC+GU pairs over the whole register — overall extent
  seed_run     longest contiguous paired run in the seed (miRNA pos 2-8)

Input is the tidy table written by the companion extractor (one row per positive:
dataset, seed_run, eff_3prime, total_pairs, n_wc, binding_type, Nunique,
error_type).  Positives only — Nunique is structurally 0 for label==0 rows.

Outputs (per --outdir, with an ``_<error_type>`` suffix when --error-type is given):
  nunique_dist_by_3prime_extent.png   stacked Nunique composition per extent bin
  nunique_dist_by_total_pairs.png     same, on overall pairing extent
  nunique_trend_by_extent.png         mean Nunique + P(Nunique>=2) vs extent
  nunique_heatmap_seed_x_3prime.png   mean Nunique over seed_run x eff_3prime
  nunique_by_pairing_extent.tsv       the table view (every plotted value)

Usage:
    python plot_nunique_by_pairing_extent.py --data pairing_extent.tsv.gz
    python plot_nunique_by_pairing_extent.py --data … --error-type FN
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- palette (validated: ordinal ramp + 3-slot categorical, light surface) ----
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
AXIS      = "#c3c2b7"
# Ordinal blue ramp for the Nunique bins (steps 250/350/450/550/650) — order
# carries meaning, so one hue with monotone lightness, not categorical hues.
NUNIQUE_RAMP = ["#5598e7", "#2a78d6", "#1c5cab", "#104281"]
# Categorical slots 1-3 for the three dataset splits.
SERIES = {"train (OOF)": "#2a78d6", "test": "#eb6834", "leftout": "#1baf7a"}
# Sequential blue 100->700 for the heatmap (continuous magnitude).
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

# The plotted composition is the *tail*: Nunique==1 holds ~75-90% of every bin,
# so stacking it swamps a 0-100% axis and hides the gradient entirely.  Stacking
# only 2/3/4/5+ on an axis that still starts at zero shows the same information
# without truncating a part-to-whole scale; the Nunique==1 share is the
# complement and is direct-labelled.
NUNIQUE_BINS   = [2, 3, 4, 5]              # 5 = "5+"
NUNIQUE_LABELS = ["2", "3", "4", "5+"]
DATASET_ORDER  = ["train (OOF)", "test", "leftout"]
MIN_N = 500  # extent bins below this are pooled into the end bins

# What the plotted rows are, for titles and axis labels — "positives" by default,
# or the selected error type (e.g. "FN") when --error-type narrows the slice.
GROUP = "positives"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  SURFACE,
        "axes.facecolor":    SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family":       "sans-serif",
        "font.size":         9,
        "text.color":        INK,
        "axes.labelcolor":   INK_2,
        "axes.edgecolor":    AXIS,
        "axes.linewidth":    0.8,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "xtick.labelcolor":  INK_2,
        "ytick.labelcolor":  INK_2,
        "grid.color":        GRID,
        "grid.linewidth":    0.8,
        "grid.linestyle":    "-",
        "axes.grid":         False,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "legend.frameon":    False,
    })


def _bin_range(s: pd.Series, hard_cap: int, min_n: int = MIN_N) -> tuple[int, int]:
    """Pick (floor, cap) so every retained extent bin holds >= min_n rows.

    Both tails are pooled rather than dropped: the extreme extents are sparse
    (a handful of rows at 0 WC pairs), and a full-width bar or a plotted mean
    built on n=8 invites exactly the over-reading these plots exist to prevent.
    """
    counts = s.value_counts()
    cap = hard_cap
    while cap > 1 and counts.reindex(range(cap, int(s.max()) + 1),
                                     fill_value=0).sum() < min_n:
        cap -= 1
    floor = 0
    while floor < cap - 1 and counts.reindex(range(0, floor + 1),
                                             fill_value=0).sum() < min_n:
        floor += 1
    return floor, cap


def bin_extent(s: pd.Series, floor: int, cap: int) -> pd.Series:
    """Clip an extent count into floor..cap; the end bins mean '<=' and '>='."""
    return s.clip(lower=floor, upper=cap)


def _bin_labels(index, floor: int, cap: int) -> list[str]:
    out = []
    for v in index:
        v = int(v)
        out.append(f"≤{v}" if v == floor and floor > 0
                   else f"{v}+" if v == cap else f"{v}")
    return out


def composition(df: pd.DataFrame, col: str, floor: int, cap: int) -> pd.DataFrame:
    """Per-extent-bin shares of Nunique in {2,3,4,5+} (+ n, mean, share at 1).

    Shares are of *all* rows in the bin, so the four stacked segments sum to
    P(Nunique>=2) and the bar's headroom is the Nunique==1 share.
    """
    d = df.copy()
    d["extent"] = bin_extent(d[col], floor, cap)
    d["nu_bin"] = d["Nunique"].clip(upper=NUNIQUE_BINS[-1])
    tab = (d.groupby(["extent", "nu_bin"]).size()
             .unstack("nu_bin", fill_value=0)
             .reindex(columns=NUNIQUE_BINS, fill_value=0))
    tab.columns = NUNIQUE_LABELS
    total = d.groupby("extent").size()
    out = tab.div(total, axis=0)
    out["n"]         = total
    out["mean"]      = d.groupby("extent")["Nunique"].mean()
    out["frac_eq_1"] = d.groupby("extent")["Nunique"].apply(lambda x: (x == 1).mean())
    out["frac_ge_2"] = out[NUNIQUE_LABELS].sum(axis=1)
    return out


def plot_composition(data: pd.DataFrame, col: str, xlabel: str, title: str,
                     out: Path, hard_cap: int) -> pd.DataFrame:
    """Small multiples: exceedance curves per extent bin, one panel per split.

    One line per read-support threshold — the share of rows with Nunique >= k
    for k in 2..5 — rather than a stacked composition.  A stack of four thin
    segments made the reader compare segment heights across neighbouring bars to
    see a trend; as curves the same numbers read as slope, each line answers a
    plain question ("what fraction has at least k reads?"), and the four
    thresholds no longer have to share one bar's height.
    """
    ranges = {ds: _bin_range(data.loc[data["dataset"] == ds, col], hard_cap)
              for ds in DATASET_ORDER}
    tabs = {ds: composition(data.loc[data["dataset"] == ds], col, *ranges[ds])
            for ds in DATASET_ORDER}
    # Exceedance = reverse-cumulative of the per-bin shares, so the top line is
    # P(Nunique>=2) and each lower line is a strictly rarer subset of it.
    curves = {ds: t[NUNIQUE_LABELS].iloc[:, ::-1].cumsum(axis=1).iloc[:, ::-1]
              for ds, t in tabs.items()}
    ymax = max(c.to_numpy().max() for c in curves.values()) * 1.14

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), sharey=True)
    for ax, ds in zip(axes, DATASET_ORDER):
        tab, cur = tabs[ds], curves[ds]
        floor, cap = ranges[ds]
        x = np.arange(len(tab))
        # "5+" is already the top Nunique bin, so its threshold reads "≥ 5".
        placed: list[float] = []
        for lab, color in zip(NUNIQUE_LABELS, NUNIQUE_RAMP):
            ys = cur[lab].to_numpy()
            ax.plot(x, ys, color=color, lw=2, solid_joinstyle="round",
                    solid_capstyle="round", marker="o", ms=4.5,
                    markeredgecolor=SURFACE, markeredgewidth=2,
                    label=f"≥ {lab.rstrip('+')}")
            # Direct-label each curve at its right end — but skip a label that
            # would land on one already placed.  Nudging them apart would detach
            # them from their lines; where curves converge the legend carries it.
            end = ys[-1]
            if all(abs(end - p) > 0.035 * ymax for p in placed):
                ax.annotate(f"≥ {lab.rstrip('+')}", (x[-1], end), xytext=(5, 0),
                            textcoords="offset points", va="center", fontsize=7.5,
                            color=INK_2)
                placed.append(end)
        ax.set_xticks(x, _bin_labels(tab.index, floor, cap), fontsize=8)
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.4, len(tab) - 0.4)
        ax.margins(x=0.10)
        ax.set_xlabel(xlabel)
        ax.set_title(f"{ds}   n={int(tab['n'].sum()):,}", fontsize=10, color=INK,
                     loc="left")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(f"Share of {GROUP} at or above threshold")
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Nunique", loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.015), fontsize=9, title_fontsize=9)
    fig.suptitle(f"{title} — {GROUP}", fontsize=12, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}", file=sys.stderr)

    rows = []
    for ds, tab in tabs.items():
        floor, cap = ranges[ds]
        t = tab.reset_index().rename(columns={"extent": "bin"})
        # The plotted values themselves, so the table view matches the curves.
        for lab in NUNIQUE_LABELS:
            t[f"frac_ge_{lab.rstrip('+')}"] = curves[ds][lab].to_numpy()
        t.insert(0, "axis", col)
        t.insert(0, "dataset", ds)
        t["bin_label"] = _bin_labels(t["bin"], floor, cap)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def plot_trend(data: pd.DataFrame, out: Path) -> None:
    """Mean Nunique and P(Nunique>=2) vs pairing extent, all three splits.

    Two panels share one y-scale each; the three splits are separate series
    (categorical identity), direct-labelled at the line ends because aqua sits
    below 3:1 on this surface.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    specs = [("eff_3prime", 10, "3' supplementary pairing (WC pairs, miRNA pos 9+)"),
             ("total_pairs", 20, "Total pairing extent (WC+GU pairs in register)")]

    for ax, (col, hard_cap, xlabel) in zip(axes, specs):
        # One bin range per axis (from the pooled data) so all three splits sit
        # on the same x grid and the pooled end bins can be labelled honestly;
        # a per-split range would put a different meaning behind each tick.
        floor, cap = _bin_range(data[col], hard_cap)
        for ds in DATASET_ORDER:
            d = data.loc[data["dataset"] == ds]
            g = (d.assign(extent=bin_extent(d[col], floor, cap))
                  .groupby("extent")["Nunique"])
            mean, n, sd = g.mean(), g.size(), g.std(ddof=1)
            keep = (n >= MIN_N).to_numpy()
            xs = mean.index.to_numpy()[keep]
            ys = mean.to_numpy()[keep]
            err = (1.96 * sd / np.sqrt(n)).to_numpy()[keep]
            color = SERIES[ds]
            ax.fill_between(xs, ys - err, ys + err, color=color, alpha=0.10,
                            linewidth=0)
            ax.plot(xs, ys, color=color, lw=2, solid_joinstyle="round",
                    solid_capstyle="round", marker="o", ms=4.5,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=ds)
            ax.annotate(ds, (xs[-1], ys[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8,
                        color=INK_2)
        ticks = list(range(floor, cap + 1))
        step = 1 if len(ticks) <= 12 else 2
        ticks = ticks[::step] if cap in ticks[::step] else ticks[::step] + [cap]
        ax.set_xticks(ticks, _bin_labels(ticks, floor, cap), fontsize=8)
        ax.set_xlabel(xlabel)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.margins(x=0.18)
    axes[0].set_ylabel("Mean Nunique (95% CI)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.015), fontsize=9)
    fig.suptitle(f"Mean Nunique vs pairing extent — {GROUP} "
                 f"(bins with n < {MIN_N} omitted)",
                 fontsize=12, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}", file=sys.stderr)


def plot_heatmap(data: pd.DataFrame, out: Path, dataset: str = "train (OOF)",
                 min_cell: int = 200) -> pd.DataFrame:
    """Mean Nunique over the seed_run x eff_3prime grid for one split.

    Separates the two candidate drivers: reading across a row shows what 3'
    extent does at fixed seed strength, reading down a column shows what seed
    strength does at fixed 3' extent.
    """
    d = data.loc[data["dataset"] == dataset].copy()
    d["e3"] = bin_extent(d["eff_3prime"], 0, 9)
    grp = d.groupby(["seed_run", "e3"])["Nunique"]
    mean, n = grp.mean().unstack("e3"), grp.size().unstack("e3")
    mean = mean.where(n >= min_cell)
    # Drop rows/columns left entirely empty by the min_cell mask — an all-blank
    # lane is a tick the reader has to rule out for nothing.
    mean = mean.dropna(axis=0, how="all").dropna(axis=1, how="all")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    im = ax.imshow(mean.to_numpy(), cmap=SEQ_CMAP, aspect="auto", origin="lower")
    ax.set_xticks(range(len(mean.columns)),
                  [f"{int(c)}" if c < 9 else "9+" for c in mean.columns])
    ax.set_yticks(range(len(mean.index)), [int(v) for v in mean.index])
    ax.set_xlabel("3' supplementary pairing (WC pairs, miRNA pos 9+)")
    ax.set_ylabel("Seed contiguous run (miRNA pos 2–8)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    # Every cell is labelled here on purpose: this is the table view of the grid
    # (9 x 8 cells), so no value is gated behind the color scale.
    vals = mean.to_numpy()
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if (v - lo) / (hi - lo) > 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Mean Nunique", color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelcolor=INK_2)
    # State the span: the ramp always stretches over whatever range the cells
    # occupy, so without it a narrow-range grid (FN-only, say) looks as
    # structured as a wide one.
    ax.set_title(f"Mean Nunique by seed strength x 3' extent — {dataset}, {GROUP}\n"
                 f"(cells with n < {min_cell} left blank; "
                 f"colour spans {lo:.2f}–{hi:.2f})",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}", file=sys.stderr)

    tidy = (grp.agg(["size", "mean"]).reset_index()
               .rename(columns={"e3": "eff_3prime_bin", "size": "n",
                                "mean": "mean_Nunique"}))
    tidy.insert(0, "dataset", dataset)
    return tidy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True,
                    help="Tidy per-positive table from the extractor (.tsv/.tsv.gz).")
    ap.add_argument("--outdir", default="results",
                    help="Directory to write plots into (default: results).")
    ap.add_argument("--error-type", default=None,
                    help="Restrict to one error type (e.g. FN, TP). Outputs get an "
                         "_<error_type> suffix so they sit beside the unfiltered run.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _style()

    data = pd.read_csv(args.data, sep="\t", low_memory=False)
    data = data.loc[data["dataset"].isin(DATASET_ORDER)]

    global GROUP
    suffix = ""
    if args.error_type:
        known = sorted(data["error_type"].dropna().unique())
        if args.error_type not in known:
            print(f"ERROR: error_type {args.error_type!r} not in the data "
                  f"(have: {', '.join(known)}).", file=sys.stderr)
            return 1
        data = data.loc[data["error_type"] == args.error_type]
        GROUP = args.error_type
        suffix = f"_{args.error_type}"
    print(f"Loaded {len(data):,} rows ({GROUP}) from {args.data}", file=sys.stderr)

    t1 = plot_composition(data, "eff_3prime",
                          "3' supplementary pairing (WC pairs, miRNA pos 9+)",
                          "Nunique distribution by 3' supplementary pairing extent",
                          outdir / f"nunique_dist_by_3prime_extent{suffix}.png", 10)
    t2 = plot_composition(data, "total_pairs",
                          "Total pairing extent (WC+GU pairs in register)",
                          "Nunique distribution by total pairing extent",
                          outdir / f"nunique_dist_by_total_pairs{suffix}.png", 20)
    plot_trend(data, outdir / f"nunique_trend_by_extent{suffix}.png")
    t3 = plot_heatmap(data, outdir / f"nunique_heatmap_seed_x_3prime{suffix}.png")

    tsv = outdir / f"nunique_by_pairing_extent{suffix}.tsv"
    with tsv.open("w") as fh:
        pd.concat([t1, t2], ignore_index=True).to_csv(fh, sep="\t", index=False)
        fh.write("\n")
        t3.to_csv(fh, sep="\t", index=False)
    print(f"Wrote {tsv}", file=sys.stderr)

    # Console summary: the endpoints of each extent axis, per split.
    for col, hard_cap in (("eff_3prime", 10), ("total_pairs", 20)):
        print(f"\n{col}:", file=sys.stderr)
        for ds in DATASET_ORDER:
            d = data.loc[data["dataset"] == ds]
            floor, cap = _bin_range(d[col], hard_cap)
            tab = composition(d, col, floor, cap)
            lo, hi = tab.index.min(), tab.index.max()
            print(f"  {ds:<12} bin {int(lo)}: mean={tab.loc[lo, 'mean']:.3f} "
                  f"P(>=2)={tab.loc[lo, 'frac_ge_2']:.3f} n={int(tab.loc[lo, 'n']):,}"
                  f"  ->  bin {int(hi)}+: mean={tab.loc[hi, 'mean']:.3f} "
                  f"P(>=2)={tab.loc[hi, 'frac_ge_2']:.3f} n={int(tab.loc[hi, 'n']):,}",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
