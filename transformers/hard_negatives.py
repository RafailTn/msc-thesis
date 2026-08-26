"""
dG-matched hard-negative mining for miRNA:MRE training.

Motivation
----------
Weak / 3'-compensatory positives and many negatives are indistinguishable by
hybridisation free energy. In the default training set those energy-matched
negatives are diluted among millions of trivially-separable random pairings, so
the gradient is dominated by easy examples and the decision boundary is never
forced into the region where the classes actually overlap.

Two modes are provided:

  resample : keep plain BCE, but oversample (or upweight) the negatives whose dG
             is closest to that of a hard-stratum positive. No architecture or
             loss change -- the cheapest possible test of whether separability
             exists in the overlap region at all.

  paired   : emit batches laid out as [pos, neg_1..neg_k, pos, neg_1..neg_k, ...]
             so that each anchor positive sits alongside its own dG-matched
             negatives. This is what makes a margin or InfoNCE term meaningful;
             with random in-batch negatives those losses are close to free.

Mining is always restricted to the indices you pass in, so call it with the
TRAIN indices of a fold only -- never the full frame -- or hard negatives will
leak across the fold boundary.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Sampler

HARD_STRATA_DEFAULT = ("weak_seed", "seedless")


# --------------------------------------------------------------- energy proxy
_PAIR_OK = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
_WOBBLE = {("G", "U"), ("U", "G")}


def proxy_energy(mirna: str, mre: str) -> float:
    """
    Crude stand-in for a real hybridisation dG, for smoke tests only.

    Best ungapped complementarity score over all offsets, negated so that lower
    (more negative) means more stable, matching the sign convention of RNAduplex
    or IntaRNA. Do NOT use this for real experiments -- pass a proper
    transcript-aware IntaRNA / RNAduplex energy via --energy-col instead.
    """
    m = mirna.upper().replace("T", "U")
    t = mre.upper().replace("T", "U")[::-1]
    best = 0.0
    for off in range(max(1, len(t) - len(m) + 1)):
        sc = 0.0
        for k, ch in enumerate(m):
            if off + k >= len(t):
                break
            pair = (ch, t[off + k])
            if pair in _PAIR_OK:
                sc += 2.0 if pair in {("G", "C"), ("C", "G")} else 1.5
            elif pair in _WOBBLE:
                sc += 0.7
        best = max(best, sc)
    return -best


def ensure_energy(df: pd.DataFrame, energy_col: str | None,
                  mirna_col="noncodingRNA", mre_col="gene") -> str:
    """Return the name of a usable energy column, computing a proxy if needed."""
    if energy_col and energy_col in df.columns:
        return energy_col
    warnings.warn(
        f"energy column {energy_col!r} not found -- falling back to proxy_energy(). "
        "This is a complementarity heuristic, NOT a thermodynamic dG. Results from "
        "hard-negative mining on the proxy are not publishable; supply a real "
        "IntaRNA/RNAduplex column via --energy-col.",
        RuntimeWarning,
    )
    df["_proxy_dG"] = [proxy_energy(a, b) for a, b in zip(df[mirna_col], df[mre_col])]
    return "_proxy_dG"


# ------------------------------------------------------------------- mining
def mine_hard_negatives(
    df: pd.DataFrame,
    indices: np.ndarray,
    energy_col: str,
    k: int = 4,
    strata_col: str | None = "seed_stratum",
    hard_strata: tuple[str, ...] = HARD_STRATA_DEFAULT,
    label_col: str = "label",
    max_dg_gap: float | None = None,
) -> dict[int, list[int]]:
    """
    For each hard-stratum positive within `indices`, find its k dG-nearest
    negatives (also within `indices`).

    Returns {positive_row_index: [negative_row_index, ...]} using ORIGINAL df
    row positions, so the mapping survives being handed to a Sampler.

    max_dg_gap: if set, drop matches further than this in dG. Prevents pairing a
    positive with a "nearest" negative that is not actually energy-matched when
    the local density of negatives is sparse.
    """
    sub = df.iloc[indices]
    y = sub[label_col].to_numpy()
    e = sub[energy_col].to_numpy(dtype=np.float64)

    if strata_col and strata_col in sub.columns:
        is_hard = sub[strata_col].isin(hard_strata).to_numpy()
    else:
        warnings.warn("no strata column -- mining against ALL positives")
        is_hard = np.ones(len(sub), dtype=bool)

    pos_local = np.flatnonzero((y == 1) & is_hard)
    neg_local = np.flatnonzero(y == 0)
    if len(pos_local) == 0 or len(neg_local) == 0:
        return {}

    tree = cKDTree(e[neg_local].reshape(-1, 1))
    kq = min(k, len(neg_local))
    dist, nn = tree.query(e[pos_local].reshape(-1, 1), k=kq)
    if kq == 1:
        dist, nn = dist[:, None], nn[:, None]

    out: dict[int, list[int]] = {}
    for row, (d_row, n_row) in enumerate(zip(dist, nn)):
        keep = n_row if max_dg_gap is None else n_row[d_row <= max_dg_gap]
        if len(keep) == 0:
            continue
        out[int(indices[pos_local[row]])] = [int(indices[neg_local[j]]) for j in keep]
    return out


def mining_report(df: pd.DataFrame, mined: dict[int, list[int]], energy_col: str) -> str:
    if not mined:
        return "no hard negatives mined"
    gaps = []
    for p, negs in mined.items():
        ep = df[energy_col].iat[p]
        gaps.extend(abs(df[energy_col].iat[n] - ep) for n in negs)
    gaps = np.asarray(gaps)
    n_neg = len({n for v in mined.values() for n in v})
    return (f"mined {len(mined)} anchor positives -> {n_neg} distinct negatives; "
            f"|ddG| median {np.median(gaps):.3f} p90 {np.quantile(gaps, .9):.3f}")


# ------------------------------------------------------------------ samplers
def resample_weights(
    n_rows: int,
    mined: dict[int, list[int]],
    boost: float = 8.0,
    base: float = 1.0,
) -> torch.DoubleTensor:
    """
    Sampling weights for WeightedRandomSampler: mined hard negatives (and their
    anchor positives) get `boost`x the weight of everything else.
    """
    w = np.full(n_rows, base, dtype=np.float64)
    for p, negs in mined.items():
        w[p] = boost
        for n in negs:
            w[n] = boost
    return torch.as_tensor(w, dtype=torch.double)


class PairedHardNegSampler(Sampler[list[int]]):
    """
    Yields batches laid out as
        [p_1, n_1^1..n_1^k, p_2, n_2^1..n_2^k, ...]
    with `anchors_per_batch` anchors. Reshape the logits to
    (anchors_per_batch, 1 + k) and column 0 is the positive.

    Anchors are shuffled every epoch and their negatives resampled from the
    mined pool, so the model sees varied matched negatives per anchor.
    """

    def __init__(self, mined: dict[int, list[int]], k: int = 4,
                 anchors_per_batch: int = 32, seed: int = 0, drop_last: bool = True):
        self.anchors = np.array(sorted(mined.keys()))
        self.pool = {p: np.array(v) for p, v in mined.items()}
        self.k = k
        self.apb = anchors_per_batch
        self.rng = np.random.default_rng(seed)
        self.drop_last = drop_last

    @property
    def batch_size(self) -> int:
        return self.apb * (1 + self.k)

    def __len__(self) -> int:
        n = len(self.anchors) // self.apb
        return n if self.drop_last else int(np.ceil(len(self.anchors) / self.apb))

    def __iter__(self):
        order = self.rng.permutation(len(self.anchors))
        for s in range(0, len(order), self.apb):
            chunk = order[s : s + self.apb]
            if self.drop_last and len(chunk) < self.apb:
                break
            batch: list[int] = []
            for a in chunk:
                p = int(self.anchors[a])
                cand = self.pool[p]
                negs = self.rng.choice(cand, size=self.k, replace=len(cand) < self.k)
                batch.append(p)
                batch.extend(int(x) for x in negs)
            yield batch


# --------------------------------------------------------------------- losses
def paired_margin_loss(logits: torch.Tensor, k: int, margin: float = 1.0) -> torch.Tensor:
    """
    logits: (anchors*(1+k),) laid out by PairedHardNegSampler.
    Hinge requiring each anchor positive to outscore its own dG-matched negatives.
    """
    z = logits.view(-1, 1 + k)
    pos, neg = z[:, :1], z[:, 1:]
    return torch.relu(margin - (pos - neg)).mean()


def paired_infonce(logits: torch.Tensor, k: int, temperature: float = 0.1) -> torch.Tensor:
    """
    InfoNCE over the score itself: the anchor positive must be the argmax within
    its own group of dG-matched candidates. Sharper than the margin form and
    needs no separate projection head.
    """
    z = logits.view(-1, 1 + k) / temperature
    tgt = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
    return torch.nn.functional.cross_entropy(z, tgt)
