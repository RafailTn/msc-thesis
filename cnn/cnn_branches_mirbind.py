#!/usr/bin/env python3
"""
Single-branch CNN for miRNA–MRE interaction classification.
Sequence branch uses miRBind-style 2D Watson-Crick complementarity matrix + 2D CNN.

Architecture
------------

Sequence branch (miRBind-style)
    Build a 2D binary matrix M of shape (MAX_MIRNA × MRE_LEN) where M[i,j]=1 if
    miRNA position i and MRE position j can form a Watson–Crick (or G·U wobble)
    base pair, 0 otherwise.  A 2D CNN of 6 convolutional blocks (5×5 kernels,
    BatchNorm2d, LeakyReLU, optional MaxPool2d, Dropout) processes the matrix,
    followed by an AdaptiveAvgPool2d and two dense blocks to produce a fixed-size
    embedding.  This directly encodes base-pairing potential geometry, matching
    the core idea from:
        Klimentova et al. (2022). miRBind: A Deep Learning Method for miRNA
        Binding Classification. Genes, 13(12), 2323.
    The embedding is fed straight into the classifier head.

Expected CSV columns
--------------------
Required:
    mre_sequence       – nucleotide string (≤50 nt)
    mirna_sequence     – nucleotide string (≤30 nt)
    label              – 0 or 1  (omit for inference)

Usage
-----
  python cnn_branches_mirbind.py train \\
      --train data/train.csv --val data/val.csv \\
      --out checkpoints/cnn_mirbind.pt --epochs 40

  python cnn_branches_mirbind.py predict \\
      --checkpoint checkpoints/cnn_mirbind.pt \\
      --input data/test.csv --output predictions.tsv

  python cnn_branches_mirbind.py predict-ensemble \\
      --checkpoints checkpoints/cnn_mirbind_fold*.pt \\
      --inputs data/test1.csv data/test2.csv \\
      --output-dir predictions/
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Shared binding-type classifier (one definition across training undersampling
# and error_analysis.py).  Imported flat when run as a script from cnn/,
# package-style when imported as cnn.cnn_branches_mirbind.
try:
    import binding_types as _bt
except ImportError:
    from cnn import binding_types as _bt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MRE_LEN   = 50
MAX_MIRNA = 30

# Watson-Crick + G·U wobble complementarity (RNA)
_WC_PAIRS: set[tuple[str, str]] = {
    ("A", "U"), ("U", "A"),
    ("G", "C"), ("C", "G"),
    ("G", "U"), ("U", "G"),   # wobble
}

# Vectorised lookup table: _NUC_IDX maps nucleotide → int index (unknown → 4)
_NUC_IDX: dict[str, int] = {"A": 0, "C": 1, "G": 2, "U": 3}
# 5×5 so that index 4 (unknown) always maps to 0
_WC_TABLE = np.zeros((5, 5), dtype=np.float32)
for _a, _b in _WC_PAIRS:
    _WC_TABLE[_NUC_IDX[_a], _NUC_IDX[_b]] = 1.0

# Per-cell pairing-TYPE lookup tables for the multi-channel sequence branch.
# Each is 5×5; index 4 (pad/unknown) maps to all-zero so padding cells carry no
# signal in any channel.
_PAIR_AU = np.zeros((5, 5), dtype=np.float32)   # A·U Watson–Crick only
for _a, _b in {("A", "U"), ("U", "A")}:
    _PAIR_AU[_NUC_IDX[_a], _NUC_IDX[_b]] = 1.0

_PAIR_GC = np.zeros((5, 5), dtype=np.float32)   # G·C Watson–Crick only
for _a, _b in {("G", "C"), ("C", "G")}:
    _PAIR_GC[_NUC_IDX[_a], _NUC_IDX[_b]] = 1.0

# Combined Watson–Crick (A·U + G·C), used by the 3-channel "multi" encoding and
# by the deterministic duplex statistics.
_PAIR_WC = _PAIR_AU + _PAIR_GC

_PAIR_GU = np.zeros((5, 5), dtype=np.float32)   # G·U wobble only
for _a, _b in {("G", "U"), ("U", "G")}:
    _PAIR_GU[_NUC_IDX[_a], _NUC_IDX[_b]] = 1.0

_PAIR_MM = np.zeros((5, 5), dtype=np.float32)   # mismatch: both real, not a pair
for _i in range(4):
    for _j in range(4):
        if _PAIR_WC[_i, _j] == 0.0 and _PAIR_GU[_i, _j] == 0.0:
            _PAIR_MM[_i, _j] = 1.0

# (3, 5, 5) stack: Watson–Crick / wobble / mismatch channels.
_PAIR_TABLE = np.stack([_PAIR_WC, _PAIR_GU, _PAIR_MM])

# (4, 5, 5) stack: A·U / G·C / wobble / mismatch — the WC channel of _PAIR_TABLE
# split into its weak (A·U, 2 H-bonds) and strong (G·C, 3 H-bonds) components.
_PAIR_TABLE4 = np.stack([_PAIR_AU, _PAIR_GC, _PAIR_GU, _PAIR_MM])

# Graded pairing-STRENGTH map (single dense 5×5 channel): a monotonic stability
# ordering G·C (3 H-bonds) > A·U (2) > G·U wobble (1) > mismatch (0).  Pad/unknown
# (index 4) stays 0.  Used to chemistry-initialise the learnable "embed" pairing
# representation so it starts from this prior instead of sparse one-hot channels.
_PAIR_STRENGTH = (3.0 * _PAIR_GC + 2.0 * _PAIR_AU
                  + 1.0 * _PAIR_GU + 0.0 * _PAIR_MM).astype(np.float32)   # (5, 5)


def _chem_init_pair_embed(dim: int, seed: int = 0) -> np.ndarray:
    """Chemistry-initialised (dim, 5, 5) lookup for the learnable pairing embedding.

    Channel 0 is seeded with the graded pairing-strength prior (_PAIR_STRENGTH);
    any extra channels start as small Gaussian noise so the network can learn
    further distinctions (e.g. mismatch sub-types) on top of the prior.  The pad
    index (4) rows/cols are zeroed so padding cells carry no signal at init.
    """
    if dim < 1:
        raise ValueError(f"pair_embed_dim must be >= 1, got {dim}")
    rng = np.random.default_rng(seed)
    table = (0.02 * rng.standard_normal((dim, 5, 5))).astype(np.float32)
    table[0] = _PAIR_STRENGTH
    table[:, 4, :] = 0.0
    table[:, :, 4] = 0.0
    return np.ascontiguousarray(table)


# ASCII byte → nucleotide index lookup (default 4 = unknown/padding).
# Lets us tokenise whole sequences with a single vectorised gather instead of
# a per-character Python loop.  Handles upper/lowercase and T→U.
_ASCII_IDX = np.full(256, 4, dtype=np.int8)
for _c, _i in _NUC_IDX.items():
    _ASCII_IDX[ord(_c)] = _i
    _ASCII_IDX[ord(_c.lower())] = _i
_ASCII_IDX[ord("T")] = _NUC_IDX["U"]
_ASCII_IDX[ord("t")] = _NUC_IDX["U"]

# Bump when the on-disk preprocessing cache format changes.
# v2: dropped the tspot and energy arrays (tspot/energy branches removed).
# v3: dropped the conservation and eclip arrays (those branches removed).
# v4: added the per-MRE accessibility vector (tAcc) for the MRE-axis channel.
# v5: added the per-MRE conservation vector (phastCons/phyloP) MRE-axis channel.
# v6: added the per-pair leakage-free neighbour-count scalar (classifier head).
# v7: dropped the tAcc/conservation MRE-axis vectors (acc/con channels removed).
_CACHE_VERSION = 7


def _norm_seq(seq: str) -> str:
    return seq.upper().replace("T", "U")


def _encode_seqs(seqs: list[str], length: int) -> np.ndarray:
    """Tokenise nucleotide strings into a fixed-length int8 index matrix.

    Returns shape (N, length); each entry is the nucleotide index (0–3) or 4
    for unknown/padding.  Padding indices map to an all-zero row/column of
    _WC_TABLE, so the Watson–Crick matrix built downstream is identical to the
    previous per-sample construction — just computed once, in bulk, and the
    actual matrix assembled on-device in the model's forward pass.
    """
    out = np.full((len(seqs), length), 4, dtype=np.int8)
    for r, s in enumerate(seqs):
        b = np.frombuffer(s.encode("ascii", "ignore")[:length], dtype=np.uint8)
        out[r, :b.shape[0]] = _ASCII_IDX[b]
    return out


def _set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and Torch RNGs for reproducible training.

    Covers weight init, DataLoader shuffling, the WeightedRandomSampler and
    dropout. Pass deterministic=True to also force cuDNN's deterministic
    algorithms (slower, but removes the last source of run-to-run variation).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _read_table(path: str | Path) -> pd.DataFrame:
    sep = "\t" if str(path).endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)


def _dedup_pairs(df: pd.DataFrame, mirna_col: str, mre_col: str,
                 mode: str) -> pd.DataFrame:
    """Drop duplicate (miRNA, MRE) sequence pairs.

    mode="first" keeps the first occurrence of each duplicated pair; mode="none"
    drops every row belonging to a duplicated pair (so only pairs that appear
    exactly once survive). Matching is on the raw (mirna_col, mre_col) strings.
    """
    subset  = [mirna_col, mre_col]
    missing = [c for c in subset if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: --dedup needs columns {subset}; missing {missing}. "
                 f"Available: {list(df.columns)}")
    keep   = "first" if mode == "first" else False
    before = len(df)
    out    = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    print(f"  [dedup:{mode}] {before} -> {len(out)} rows "
          f"({before - len(out)} dropped on ({mirna_col}, {mre_col}))")
    return out


def _parse_vector(raw, length: int) -> np.ndarray:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return np.zeros(length, dtype=np.float32)
    if isinstance(raw, (list, np.ndarray)):
        vals = np.array(raw, dtype=np.float32)
    else:
        raw = str(raw).strip().strip("[]")
        if not raw or raw in ("nan", "None", "NaN"):
            return np.zeros(length, dtype=np.float32)
        try:
            vals = np.fromstring(raw, sep=",", dtype=np.float32)
        except Exception:
            return np.zeros(length, dtype=np.float32)
    if vals.size == 0:
        return np.zeros(length, dtype=np.float32)
    out = np.zeros(length, dtype=np.float32)
    out[:min(len(vals), length)] = vals[:min(len(vals), length)]
    return out


# ---------------------------------------------------------------------------
# Preprocessing cache
#
# Parsing a multi-GB CSV with pandas and re-tokenising every sequence on each
# run is the dominant start-up cost (and RAM spike).  We cache the parsed
# arrays to a sidecar .cnncache.npz next to the source file and reload that
# instead whenever it is present and up to date.
# ---------------------------------------------------------------------------

def _cache_path(path: str | Path) -> Path:
    return Path(str(path) + ".cnncache.npz")


def _save_cache(path: str | Path, mre_col: str, mirna_col: str,
                nbr_col: str, ds: "MiRNAInteractionDataset") -> None:
    cp = _cache_path(path)
    try:
        arrays = dict(
            version=np.array([_CACHE_VERSION]),
            mre_col=np.array([mre_col]),
            mirna_col=np.array([mirna_col]),
            nbr_col=np.array([nbr_col or ""]),
            dims=np.array([MAX_MIRNA, MRE_LEN]),
            has_nbr=np.array([ds.nbr is not None]),
            mirna_idx=ds.mirna_idx,
            mre_idx=ds.mre_idx,
            labels=ds.labels,
        )
        if ds.nbr is not None:
            arrays["nbr"] = ds.nbr
        np.savez(cp, **arrays)
        print(f"  [cache] wrote {cp.name}")
    except Exception as e:   # caching is best-effort; never fail training over it
        print(f"  [cache] could not write {cp.name}: {e}")


def _load_cache(path: str | Path, mre_col: str, mirna_col: str,
                nbr_col: str) -> Optional[dict]:
    cp = _cache_path(path)
    if not cp.exists():
        return None
    try:
        if cp.stat().st_mtime < Path(path).stat().st_mtime:
            return None   # source changed after the cache was written
        z = np.load(cp, allow_pickle=False)
        if (int(z["version"][0]) != _CACHE_VERSION
                or str(z["mre_col"][0]) != mre_col
                or str(z["mirna_col"][0]) != mirna_col
                or str(z["nbr_col"][0]) != (nbr_col or "")
                or list(z["dims"]) != [MAX_MIRNA, MRE_LEN]):
            z.close()
            return None
        data = {
            "mirna_idx":  z["mirna_idx"],
            "mre_idx":    z["mre_idx"],
            "labels":     z["labels"],
            "nbr":        z["nbr"] if bool(z["has_nbr"][0]) else None,
        }
        z.close()
        return data
    except Exception as e:
        print(f"  [cache] ignoring unreadable cache {cp.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MiRNAInteractionDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        has_labels: bool = True,
        mre_col: str = "mre_sequence",
        mirna_col: str = "mirna_sequence",
        nbr_col: str = "",
        cache: bool = True,
    ) -> None:
        cached = (_load_cache(path, mre_col, mirna_col, nbr_col)
                  if cache else None)
        if cached is not None:
            print(f"  [cache] loaded {_cache_path(path).name}")
            self.has_labels = has_labels
            self._set_arrays(**cached)
        else:
            self._init(_read_table(path), has_labels, mre_col, mirna_col, nbr_col)
            if cache:
                _save_cache(path, mre_col, mirna_col, nbr_col, self)

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        has_labels: bool = True,
        mre_col: str = "mre_sequence",
        mirna_col: str = "mirna_sequence",
        nbr_col: str = "",
    ) -> "MiRNAInteractionDataset":
        obj = cls.__new__(cls)
        obj._init(df, has_labels, mre_col, mirna_col, nbr_col)
        return obj

    def _init(
        self,
        df: pd.DataFrame,
        has_labels: bool,
        mre_col: str,
        mirna_col: str,
        nbr_col: str = "",
    ) -> None:
        self.has_labels = has_labels
        self._build_arrays(df, has_labels, mre_col, mirna_col, nbr_col)

    def _build_arrays(
        self,
        df: pd.DataFrame,
        has_labels: bool,
        mre_col: str,
        mirna_col: str,
        nbr_col: str = "",
    ) -> None:
        # Tokenise sequences once into int8 index matrices; the Watson–Crick
        # matrix is assembled on-device in the model forward pass.
        self.mirna_idx = _encode_seqs(df[mirna_col].astype(str).tolist(), MAX_MIRNA)  # (N, 30)
        self.mre_idx   = _encode_seqs(df[mre_col].astype(str).tolist(),   MRE_LEN)    # (N, 50)

        # Per-pair leakage-free neighbour count: number of distinct
        # confident-positive sites near this pair's MRE (a global scalar, not an
        # MRE-axis vector).  Precomputed offline by the `neighbor-counts`
        # subcommand from out-of-fold predictions and stored as a plain column;
        # parsed only when present, else the model — if it requests the feature —
        # sees a count of 0.  Stored as (N,) float32.
        if nbr_col and nbr_col in df.columns:
            self.nbr = df[nbr_col].to_numpy(dtype=np.float32)
            np.nan_to_num(self.nbr, copy=False)
        else:
            self.nbr = None

        if has_labels and "label" in df.columns:
            self.labels = df["label"].astype(int).values
        else:
            self.labels = np.zeros(len(df), dtype=np.int64)

    def _set_arrays(
        self,
        mirna_idx: np.ndarray,
        mre_idx: np.ndarray,
        labels: np.ndarray,
        nbr: Optional[np.ndarray] = None,
    ) -> None:
        """Populate arrays from a loaded cache."""
        self.mirna_idx  = mirna_idx
        self.mre_idx    = mre_idx
        self.nbr        = nbr
        self.labels = labels if self.has_labels else np.zeros(len(mirna_idx), dtype=np.int64)

    def subset(self, mask: np.ndarray) -> None:
        """Restrict the dataset in place to rows where ``mask`` is True.

        Keeps every per-sample array (tokens, labels, optional neighbour count)
        aligned, so the dataset stays internally consistent after e.g.
        binding-type filtering.
        """
        mask = np.asarray(mask, dtype=bool)
        self.mirna_idx = self.mirna_idx[mask]
        self.mre_idx   = self.mre_idx[mask]
        self.labels    = self.labels[mask]
        if self.nbr is not None:
            self.nbr = self.nbr[mask]

    def __len__(self) -> int:
        return len(self.mirna_idx)

    def __getitem__(self, idx: int):
        nbr = float(self.nbr[idx]) if self.nbr is not None else 0.0
        return (
            torch.from_numpy(self.mirna_idx[idx]),          # (MAX_MIRNA,) int8
            torch.from_numpy(self.mre_idx[idx]),            # (MRE_LEN,)   int8
            nbr,                                            # scalar neighbour count
            int(self.labels[idx]),
        )


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------

# Activation factory for the 2D sequence branch (conv + dense blocks). The
# vector branches keep their own GELU; only the miRBind branch is tunable here.
_ACTIVATIONS = {
    "leaky_relu": lambda: nn.LeakyReLU(0.1, inplace=True),
    "relu":       lambda: nn.ReLU(inplace=True),
    "gelu":       lambda: nn.GELU(),
    "silu":       lambda: nn.SiLU(inplace=True),
    "elu":        lambda: nn.ELU(inplace=True),
    "selu":       lambda: nn.SELU(inplace=True),
}


def _make_activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name]()
    except KeyError:
        raise ValueError(
            f"activation must be one of {list(_ACTIVATIONS)}, got {name!r}")


class GeMDownsample2d(nn.Module):
    """GeM pooling as a fixed-stride 2D downsampler — a drop-in for MaxPool2d.

    GeM(x) = ( mean_window x_i^p )^(1/p) over each kernel window: p=1 is average
    pooling, p→∞ approaches max pooling, and p is learnable so the block tunes
    how peaky its downsampling is. Inputs are clamped to ≥eps because the
    mean-of-powers is only defined for non-negative values (standard GeM
    assumption), so any negatives from the preceding activation are floored.
    """

    def __init__(self, kernel_size: int = 2, stride: int = 2,
                 p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, self.kernel_size, self.stride)
        return x.pow(1.0 / self.p)


class Conv2dBlock(nn.Module):
    """One miRBind-style 2D convolutional block.

    Conv2d (5×5, same padding) → BatchNorm2d → activation →
    [MaxPool2d / GeMDownsample2d (2,2)] → Dropout.

    BatchNorm precedes the activation so the (non-negative) activation output —
    not the zero-centred BN output — is what feeds the pool; this keeps GeM's
    non-negativity assumption intact. The conv runs bias-free since the following
    BatchNorm re-centres and makes a conv bias redundant.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.3,
                 pool: bool = True, block_pool: str = "max",
                 activation: str = "leaky_relu") -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_ch),
            _make_activation(activation),
        ]
        if pool:
            if block_pool == "gem":
                layers.append(GeMDownsample2d(2, 2))
            elif block_pool == "max":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                raise ValueError(
                    f"block_pool must be 'max' or 'gem', got {block_pool!r}")
        layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseBlock(nn.Module):
    """One miRBind-style dense block: Linear → BatchNorm1d → activation → Dropout.

    BatchNorm precedes the activation (mirroring Conv2dBlock); the linear runs
    bias-free since the following BatchNorm makes its bias redundant.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3,
                 activation: str = "leaky_relu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            _make_activation(activation),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GeM2d(nn.Module):
    """Generalized-mean pooling over the spatial dims: (B, C, H, W) → (B, C, 1, 1).

    GeM(x) = ( mean_i x_i^p )^(1/p).  p=1 recovers average pooling, p→∞
    approaches max pooling; p is learnable so the network tunes how peaky the
    pooling is.  The 2D pairing map is mostly empty (the duplex is a small
    contiguous block), so emphasising the strong region beats averaging it away.
    Inputs are clamped to ≥eps because the mean-of-powers is only defined for
    non-negative values (standard GeM assumption).  Translation-invariant, so it
    preserves the property that makes the 2D branch generalise.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.pow(1.0 / self.p)


class MiRBindSeqBranch(nn.Module):
    """miRBind-style sequence branch.

    Takes the 2D WC-complementarity matrix (B, 1, MAX_MIRNA, MRE_LEN) and
    produces a fixed-size embedding of shape (B, out_dim).

    Architecture (mirroring Klimentova et al. 2022):
      - n_conv_blocks Conv2d blocks (5×5, BN2d, activation, Dropout)
        First n_pool_blocks blocks downsample (MaxPool2d or GeMDownsample2d 2×2)
        Remaining blocks have no pooling (spatial dims small by this point)
      - global pool (GeM or adaptive-avg) → flatten
      - 2 dense blocks → out_dim

    The 2D input height is MAX_MIRNA (30), which halves with each pooling block
    (30→15→7→3→1), so n_pool_blocks must not exceed 4 — a 5th pool would reduce a
    size-1 dimension to 0. n_pool_blocks is also capped at n_conv_blocks.
    """

    def __init__(
        self,
        n_filters: int = 64,
        out_dim:   int = 128,
        dropout:   float = 0.3,
        in_ch:     int = 1,
        pool:      str = "gem",
        n_conv_blocks: int = 6,
        n_pool_blocks: int = 4,
        block_pool: str = "max",
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        if n_pool_blocks > n_conv_blocks:
            raise ValueError(
                f"n_pool_blocks ({n_pool_blocks}) must be <= n_conv_blocks "
                f"({n_conv_blocks}).")
        if n_pool_blocks > 4:
            raise ValueError(
                f"n_pool_blocks ({n_pool_blocks}) exceeds 4: the miRNA-axis "
                f"height (30) only halves to 1 after 4 poolings.")

        conv_blocks: list[nn.Module] = []
        ch = in_ch
        for i in range(n_conv_blocks):
            pool_block = i < n_pool_blocks
            conv_blocks.append(Conv2dBlock(
                ch, n_filters, dropout=dropout, pool=pool_block,
                block_pool=block_pool, activation=activation))
            ch = n_filters
        self.conv_blocks = nn.Sequential(*conv_blocks)

        if pool == "gem":
            self.global_pool: nn.Module = GeM2d()
        elif pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError(f"pool must be 'gem' or 'avg', got {pool!r}")
        pooled_dim = n_filters

        hidden = max(n_filters * 2, out_dim)
        self.dense = nn.Sequential(
            DenseBlock(pooled_dim, hidden, dropout=dropout, activation=activation),
            DenseBlock(hidden,    out_dim, dropout=dropout, activation=activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, MAX_MIRNA, MRE_LEN)
        h = self.conv_blocks(x)                # (B, n_filters, H', W')
        h = self.global_pool(h).flatten(1)     # (B, n_filters)
        return self.dense(h)                   # (B, out_dim)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MiRBindCNN(nn.Module):
    """Single-branch CNN with a miRBind-style 2D sequence branch.

    Sequence branch (miRBind)
        2D WC-complementarity matrix → 6 Conv2d blocks → dense → seq_dim
        embedding, fed directly into the classifier head.

    Parameters
    ----------
    seq_filters : int
        Number of 2D conv filters in the miRBind sequence branch.
    seq_dim : int
        Output embedding size of the sequence branch (after dense layers).
    seq_dropout : float
        Dropout inside the miRBind 2D CNN blocks and the classifier head.
    seq_pool : str
        Global pool for the sequence branch: "gem" or "avg".
    """

    def __init__(
        self,
        seq_filters:      int   = 64,
        seq_dim:          int   = 128,
        seq_dropout:      float = 0.3,
        seq_pairing:      str   = "multi",
        pair_embed_dim:   int   = 3,
        seq_pool:         str   = "gem",
        seq_nbr_feature:  bool  = False,
        n_conv_blocks:    int   = 6,
        n_pool_blocks:    int   = 4,
        block_pool:       str   = "max",
        activation:       str   = "leaky_relu",
    ) -> None:
        super().__init__()

        # On-device pairing-matrix lookup, gathered in forward() on the same
        # device as the model.  "binary" = single WC(+wobble) channel (legacy);
        # "multi" = separate Watson–Crick / G·U wobble / mismatch channels;
        # "multi4" = WC split into A·U and G·C, i.e. A·U / G·C / wobble / mismatch;
        # "embed" = a learnable, chemistry-initialised dense (pair_embed_dim,5,5)
        # lookup — a low-dim continuous representation of each nucleotide pair that
        # avoids the sparse one-hot channels and lets the network learn its own
        # pairing distinctions on top of the strength prior.
        #
        # The fixed tables are constants → registered as non-persistent buffers
        # (rebuilt at construction, kept out of the state_dict so older checkpoints
        # stay loadable).  The "embed" table is learned → an nn.Parameter that IS
        # saved.  In both cases forward() masks pad cells, so padding carries no
        # signal regardless of the learned values.
        self._pair_learnable = False
        if seq_pairing == "embed":
            self.pair_table = nn.Parameter(
                torch.from_numpy(_chem_init_pair_embed(pair_embed_dim)))
            self._pair_learnable = True
            n_pair_ch = pair_embed_dim
        else:
            if seq_pairing == "multi":
                pair_table = _PAIR_TABLE                # (3, 5, 5)
            elif seq_pairing == "multi4":
                pair_table = _PAIR_TABLE4               # (4, 5, 5)
            elif seq_pairing == "binary":
                pair_table = _WC_TABLE[np.newaxis]      # (1, 5, 5)
            else:
                raise ValueError(
                    "seq_pairing must be 'binary', 'multi', 'multi4' or 'embed', "
                    f"got {seq_pairing!r}")
            self.register_buffer(
                "pair_table",
                torch.from_numpy(np.ascontiguousarray(pair_table)),
                persistent=False)
            n_pair_ch = pair_table.shape[0]

        # ── miRBind 2D sequence branch ───────────────────────────────────────
        self.seq_branch = MiRBindSeqBranch(
            n_filters=seq_filters, out_dim=seq_dim, dropout=seq_dropout,
            in_ch=n_pair_ch, pool=seq_pool,
            n_conv_blocks=n_conv_blocks, n_pool_blocks=n_pool_blocks,
            block_pool=block_pool, activation=activation)

        # ── Leakage-free neighbour-count scalar (classifier head) ─────────────
        # A per-pair global feature: the number of distinct confident-positive
        # sites near this pair's MRE (precomputed offline from out-of-fold
        # predictions; see the `neighbor-counts` subcommand).  It does NOT belong
        # in the 2D conv stack — it is constant over the pairing map — so it is
        # injected into the classifier head instead.  log1p compresses the raw
        # 0..~40 count and a BatchNorm1d normalises it onto the embedding's
        # scale.  A sample with no neighbour column (or genuinely no neighbours)
        # sees count 0, so the head degrades gracefully to the no-neighbour
        # baseline.
        self.seq_nbr_feature = seq_nbr_feature

        # ── Classifier ────────────────────────────────────────────────────────
        # No dropout before the first Linear: the seq-branch's final DenseBlock
        # already applies dropout, and only a LayerNorm+GELU (no linear) sits
        # between it and here, so a second dropout would be redundant.
        #
        # Without the neighbour feature the head is unchanged (byte-identical
        # state_dict, so existing checkpoints keep loading).  With it, the seq
        # embedding is LayerNorm+GELU'd on its own and the neighbour scalar is
        # concatenated before the first Linear (kept out of that LayerNorm so it
        # is not renormalised against the 128-d embedding).
        if seq_nbr_feature:
            self.nbr_norm = nn.BatchNorm1d(1)
            self.seq_head_norm = nn.Sequential(nn.LayerNorm(seq_dim), nn.GELU())
            self.classifier = nn.Sequential(
                nn.Linear(seq_dim + 1, seq_dim // 2),
                nn.GELU(),
                nn.Dropout(seq_dropout),
                nn.Linear(seq_dim // 2, 1),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(seq_dim),
                nn.GELU(),
                nn.Linear(seq_dim, seq_dim // 2),
                nn.GELU(),
                nn.Dropout(seq_dropout),
                nn.Linear(seq_dim // 2, 1),
            )

    def forward(
        self,
        mi:     torch.Tensor,            # (B, MAX_MIRNA)  int nucleotide indices
        ti:     torch.Tensor,            # (B, MRE_LEN)    int nucleotide indices
        nbr:    Optional[torch.Tensor] = None,  # (B,) float neighbour count
    ) -> torch.Tensor:          # (B,) logits

        # ── miRBind 2D sequence branch ───────────────────────────────────────
        # Assemble the pairing matrix on-device from the nucleotide-index
        # vectors: a broadcasted gather into pair_table giving one channel per
        # pairing type, (B, C_pair, MAX_MIRNA, MRE_LEN).  This keeps the
        # per-sample CPU work in the data loader down to a slice copy.
        mi = mi.long()
        ti = ti.long()
        pair = self.pair_table[:, mi[:, :, None], ti[:, None, :]]  # (C_pair, B, 30, 50)
        wc_mat = pair.movedim(0, 1).contiguous()                   # (B, C_pair, 30, 50)
        # Pad cells (mi/ti == index 4) must carry no signal.  Fixed tables zero
        # them by construction; the learnable embedding does not, so mask it.
        if self._pair_learnable:
            valid = ((mi < 4)[:, :, None] & (ti < 4)[:, None, :])  # (B, 30, 50)
            wc_mat = wc_mat * valid.unsqueeze(1).to(wc_mat.dtype)
        h_seq = self.seq_branch(wc_mat)         # (B, seq_dim)

        if self.seq_nbr_feature:
            h = self.seq_head_norm(h_seq)       # (B, seq_dim)
            if nbr is None:
                nbr = h.new_zeros(h.shape[0])
            # log1p-compress the count, then BatchNorm onto the embedding scale.
            nf = torch.log1p(nbr.to(h.dtype).clamp(min=0.0)).unsqueeze(1)  # (B, 1)
            nf = self.nbr_norm(nf)
            return self.classifier(torch.cat([h, nf], dim=1)).squeeze(-1)
        return self.classifier(h_seq).squeeze(-1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _binary_metrics(logits: np.ndarray, labels: np.ndarray,
                    threshold: float = 0.5) -> dict:
    probs = 1.0 / (1.0 + np.exp(-logits))
    return _metrics_from_probs(probs, labels, threshold)


def _metrics_from_probs(probs: np.ndarray, labels: np.ndarray,
                        threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy  = (tp + tn) / max(tp + fp + fn + tn, 1)

    out = dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1)
    if HAS_SKLEARN and len(np.unique(labels)) == 2:
        out["auroc"] = roc_auc_score(labels, probs)
        out["auprc"] = average_precision_score(labels, probs)
    return out


# ---------------------------------------------------------------------------
# DataLoader helper
# ---------------------------------------------------------------------------

def _make_loader(dataset: MiRNAInteractionDataset, batch_size: int,
                 shuffle: bool, num_workers: int,
                 balance: bool = False,
                 weights: Optional[np.ndarray] = None,
                 persistent_workers: Optional[bool] = None) -> DataLoader:
    sampler = None
    # Explicit per-sample weights (e.g. negative binding-type undersampling) take
    # precedence over --balance: they already encode the desired class ratio.
    if weights is not None and shuffle and dataset.has_labels:
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights), replacement=True)
        shuffle = False
    elif balance and shuffle and dataset.has_labels:
        labels  = dataset.labels
        counts  = np.bincount(labels)
        weights = 1.0 / counts[labels]
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights).double(),
            num_samples=len(weights), replacement=True)
        shuffle = False
    # Default: keep workers alive across epochs.  Callers that rebuild loaders
    # repeatedly (e.g. an Optuna study) should pass persistent_workers=False so a
    # previous trial's live worker iterator is not inherited by the next trial's
    # forked workers (-> "AssertionError: can only test a child process").
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    else:
        persistent_workers = persistent_workers and num_workers > 0
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        sampler=sampler, num_workers=num_workers,
        pin_memory=True, drop_last=(shuffle and sampler is None),
        persistent_workers=persistent_workers,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _compute_loss(logits: torch.Tensor, labels: torch.Tensor,
                  pos_weight: Optional[torch.Tensor],
                  gamma: float) -> torch.Tensor:
    """BCE with optional focal modulation.

    gamma=0  → standard BCEWithLogitsLoss (identical to before).
    gamma>0  → focal loss: (1-pt)^gamma * BCE per sample, then mean.
               Compatible with pos_weight: the class weight is applied
               inside BCE before the focal factor scales each sample.
    """
    if gamma == 0.0:
        return F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight)
    bce = F.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=pos_weight, reduction="none")
    pt  = torch.sigmoid(logits)
    pt  = torch.where(labels == 1, pt, 1.0 - pt)   # prob of the correct class
    return ((1.0 - pt) ** gamma * bce).mean()


@torch.no_grad()
def evaluate(model: MiRBindCNN, loader: DataLoader,
             device: torch.device,
             pos_weight: Optional[torch.Tensor] = None,
             gamma: float = 0.0) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    total_loss, n_batches  = 0.0, 0

    for mi, ti, nbr, labels in loader:
        mi     = mi.to(device,     non_blocking=True)
        ti     = ti.to(device,     non_blocking=True)
        nbr    = nbr.to(device,    non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        logits = model(mi, ti, nbr)
        loss   = _compute_loss(logits, labels, pos_weight, gamma)
        total_loss += loss.item()
        n_batches  += 1
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels).astype(int)
    metrics   = _binary_metrics(logits_np, labels_np)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


@torch.no_grad()
def predict_logits(model: MiRBindCNN, loader: DataLoader,
                   device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits, labels) for every sample in `loader` in dataset order.

    Used for fold ensembling: a fixed (shuffle=False) loader yields samples in a
    stable order across folds, so per-fold probabilities can be averaged
    element-wise before scoring.
    """
    model.eval()
    all_logits, all_labels = [], []
    for mi, ti, nbr, labels in loader:
        mi     = mi.to(device,     non_blocking=True)
        ti     = ti.to(device,     non_blocking=True)
        nbr    = nbr.to(device,    non_blocking=True)
        logits = model(mi, ti, nbr)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels).astype(int)


def _duplex_stats(mirna_idx: np.ndarray, mre_idx: np.ndarray) -> dict:
    """Per-pair duplex summary at the strongest antiparallel register.

    Deterministic functions of the two sequences (no model needed) for slicing
    errors: within the best contiguous complementary register, the counts of
    Watson-Crick / G·U-wobble / mismatch positions in the duplex span, the
    longest contiguous paired run, the seed-region (miRNA positions 2-8) pair
    count, and the real (unpadded) sequence lengths.
    """
    n = len(mirna_idx)
    keys = ("n_wc", "n_gu", "n_mm", "max_run", "seed_pairs", "mirna_len", "mre_len")
    out = {k: np.zeros(n, dtype=np.int32) for k in keys}
    seed_pos = np.arange(1, 8)                      # miRNA positions 2..8 (0-indexed 1..7)
    P, Q     = np.indices((MAX_MIRNA, MRE_LEN))
    d_flat   = (P + Q).ravel()
    ndiag    = MAX_MIRNA + MRE_LEN - 1

    for s in range(n):
        mi = mirna_idx[s].astype(np.intp)
        ti = mre_idx[s].astype(np.intp)
        out["mirna_len"][s] = int((mi != 4).sum())
        out["mre_len"][s]   = int((ti != 4).sum())

        wc   = _PAIR_WC[mi[:, None], ti[None, :]]   # (30, 50); 0 at padded indices
        gu   = _PAIR_GU[mi[:, None], ti[None, :]]
        pair = (wc + gu).ravel()
        pair_d = np.bincount(d_flat, weights=pair, minlength=ndiag)
        if pair_d.max() == 0:                       # no complementarity at all
            continue
        best = int(np.argmax(pair_d))               # diagonal p+q with most pairs

        ps   = np.arange(max(0, best - (MRE_LEN - 1)), min(MAX_MIRNA - 1, best) + 1)
        qs   = best - ps
        real = (mi[ps] != 4) & (ti[qs] != 4)
        wcl  = _PAIR_WC[mi[ps], ti[qs]].astype(bool)
        gul  = _PAIR_GU[mi[ps], ti[qs]].astype(bool)
        pl   = (wcl | gul) & real
        hits = np.flatnonzero(pl)
        if hits.size == 0:
            continue
        lo, hi   = hits[0], hits[-1]                # duplex span = first..last pair
        seg_real = real[lo:hi + 1]
        out["n_wc"][s] = int(wcl[lo:hi + 1][seg_real].sum())
        out["n_gu"][s] = int(gul[lo:hi + 1][seg_real].sum())
        out["n_mm"][s] = int((seg_real & ~(wcl | gul)[lo:hi + 1]).sum())

        run = best_run = 0                          # longest contiguous paired run
        for v in pl[lo:hi + 1]:
            run = run + 1 if v else 0
            best_run = max(best_run, run)
        out["max_run"][s] = best_run

        sq = best - seed_pos                         # seed pairs at this register
        ok = (seed_pos < MAX_MIRNA) & (sq >= 0) & (sq < MRE_LEN)
        sp = mi[seed_pos[ok]]
        sq = sq[ok]
        out["seed_pairs"][s] = int(
            ((_PAIR_WC[sp, ti[sq]] + _PAIR_GU[sp, ti[sq]]) > 0).sum())
    return out


# ---------------------------------------------------------------------------
# Leakage-free neighbour-count feature
#
# Real AGO2 target sites cluster spatially while the dissimilarity-sampled
# negatives do not, so the number of *other* confident-positive sites near a
# pair's MRE predicts the true label even at fixed seed strength — a signal the
# sequence branch cannot see (it has no coordinates).  This stays leakage-free
# because (a) the count is over OTHER coordinates, never the row's own, and (b)
# the "confident-positive" set comes from out-of-fold predictions the caller
# supplies (so a row's own label never enters its feature).  The graded count
# (dose-response), not a binary flag, is what carries the signal.
# ---------------------------------------------------------------------------

def _neighbor_counts(df: pd.DataFrame, score: np.ndarray, conf: float,
                     window: int, min_sep: int, chr_col: str, strand_col: str,
                     start_col: str, end_col: str) -> np.ndarray:
    """Per-row count of *distinct* confident-positive neighbour sites in a band.

    For each row, counts the distinct genomic coordinates (centre = (start+end)//2)
    on the same chr+strand whose ``score >= conf`` and whose centre lies in the
    band ``min_sep <= |Δcentre| <= window``.  ``score`` is a per-row confidence
    (e.g. ``interaction_probability`` from a held-out / out-of-fold prediction
    pass).

    ``min_sep`` sets a lower bound that drops too-close neighbours.  The row's own
    coordinate (Δ=0) is always excluded, so ``min_sep=0`` counts every distinct
    neighbour in ``(0, window]``.  Two AGO2 footprints (~50–60 nt) cannot
    co-occupy, and sites <~50 nt apart share overlapping 50-mer MRE fragments
    (near-duplicate sequences the seq branch already sees), so ``min_sep≈60``
    isolates the independent-clustering signal — empirically a steeper
    per-neighbour dose-response than the close-inclusive band.  Returns an (N,)
    int32 array.
    """
    centers = ((df[start_col].to_numpy(dtype=np.int64)
                + df[end_col].to_numpy(dtype=np.int64)) // 2)
    conf_mask = np.asarray(score, dtype=np.float64) >= conf
    counts = np.zeros(len(df), dtype=np.int32)
    grp = pd.DataFrame({
        "chr":    df[chr_col].astype(str).to_numpy(),
        "strand": df[strand_col].astype(str).to_numpy(),
        "c":      centers,
        "conf":   conf_mask,
        "row":    np.arange(len(df)),
    })
    # Exclude the near band |Δ| < max(min_sep, 1) — which always covers the row's
    # own Δ=0 coordinate, so a site never counts itself regardless of min_sep.
    thr = max(min_sep, 1)
    for _, sub in grp.groupby(["chr", "strand"], sort=False):
        conf_centers = np.unique(sub.loc[sub["conf"], "c"].to_numpy())
        if conf_centers.size == 0:
            continue
        rc   = sub["c"].to_numpy()
        lo   = np.searchsorted(conf_centers, rc - window, side="left")
        hi   = np.searchsorted(conf_centers, rc + window, side="right")
        nlo  = np.searchsorted(conf_centers, rc - (thr - 1), side="left")
        nhi  = np.searchsorted(conf_centers, rc + (thr - 1), side="right")
        counts[sub["row"].to_numpy()] = ((hi - lo) - (nhi - nlo)).astype(np.int32)
    return counts


# ---------------------------------------------------------------------------
# MANE-Select transcript model — transcript-aware / hybrid neighbour counting
#
# Genomic `_neighbor_counts` measures linear distance, which conflates relations
# that differ on the processed mRNA: two sites 100 nt apart on the genome can
# straddle a splice junction (far apart — or non-co-existent — on the mature
# mRNA), and an intronic site only exists in the pre-mRNA.  These helpers map an
# MRE onto MANE-Select transcript (spliced) coordinates and count neighbours only
# within the SAME transcript by spliced distance: introns collapsed, cross-
# junction / wrong-isoform pairs excluded.  A 50-mer is "exonic" only when a
# single MANE exon FULLY contains [start,end] AND the spliced transcript sequence
# at the mapped offset equals the MRE sequence (U->T) — the same routing the
# accessibility precompute uses for its `acc_mode`.  Straddlers / intronic /
# intergenic / sequence-mismatch rows are unmapped; in hybrid mode they fall back
# to the genomic count.  (Kept here in the core module so the `neighbor-counts`
# builder and cooperativity_analysis share one implementation.)
# ---------------------------------------------------------------------------

_DNA_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _rc_dna(s: str) -> str:
    return s.translate(_DNA_COMP)[::-1]


def _tsv_chrom_to_fa(chrom: str) -> str:
    """v7 TSV chromosome label (`6`, `MT`) -> GENCODE contig (`chr6`, `chrM`)."""
    c = str(chrom)
    if c in ("MT", "chrMT", "M"):
        return "chrM"
    return c if c.startswith("chr") else "chr" + c


def _parse_mane_gtf(gtf_path):
    """Parse MANE-Select exons -> (tx, index, max_exon_len); pickle-cached.

    tx[tid] = {"chrom","strand","ex":[(es,ee)...asc],"cum":[...],"Lt":int};
    index[(chrom,strand)] = (es_arr, ee_arr, meta) sorted by es, meta entry
    (cum_offset, es, tid).  Cache <gtf>.mane_nbr.pkl, rebuilt when GTF is newer.
    """
    import gzip
    import pickle
    gtf_path = Path(gtf_path)
    cache = gtf_path.with_name(gtf_path.name + ".mane_nbr.pkl")
    if cache.exists() and cache.stat().st_mtime >= gtf_path.stat().st_mtime:
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    tx: dict = {}
    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t")
            if len(f) < 9 or f[2] != "exon" or 'tag "MANE_Select"' not in f[8]:
                continue
            tid = f[8].split('transcript_id "', 1)[1].split('"', 1)[0]
            es, ee = int(f[3]), int(f[4])
            d = tx.get(tid)
            if d is None:
                tx[tid] = {"chrom": f[0], "strand": f[6], "ex": [(es, ee)]}
            else:
                d["ex"].append((es, ee))

    raw: dict = {}
    max_exon_len = 0
    for tid, d in tx.items():
        d["ex"].sort()                                   # genomic ascending
        cum, c = [], 0
        for es, ee in d["ex"]:
            cum.append(c)
            c += ee - es + 1
            max_exon_len = max(max_exon_len, ee - es + 1)
        d["cum"] = cum
        d["Lt"] = c
        for (es, ee), cm in zip(d["ex"], cum):
            raw.setdefault((d["chrom"], d["strand"]), []).append((es, ee, cm, tid))

    index: dict = {}
    for key, bucket in raw.items():
        bucket.sort()                                    # by exon start
        index[key] = (
            np.array([b[0] for b in bucket], dtype=np.int64),
            np.array([b[1] for b in bucket], dtype=np.int64),
            [(b[2], b[0], b[3]) for b in bucket],        # (cum, es, tid)
        )
    with open(cache, "wb") as fh:
        pickle.dump((tx, index, max_exon_len), fh)
    return tx, index, max_exon_len


def _find_host_exon(index, max_exon_len, chrom, strand, s, e):
    """(cum, es, tid) of the MANE exon fully containing [s,e], or None."""
    from bisect import bisect_right
    rec = index.get((chrom, strand))
    if rec is None:
        return None
    es_arr, ee_arr, meta = rec
    j = bisect_right(es_arr, s)
    k = j - 1
    while k >= 0 and (s - es_arr[k]) <= max_exon_len:
        if ee_arr[k] >= e:
            return meta[k]
        k -= 1
    return None


class _TxContext:
    """Lazily concatenated spliced MANE transcript sequences (for the guard)."""

    def __init__(self, genome_fa, tx):
        from pyfaidx import Fasta
        self.fa = Fasta(genome_fa, sequence_always_upper=True, rebuild=False)
        self.tx = tx
        self._seq: dict = {}

    def txseq(self, tid: str) -> str:
        s = self._seq.get(tid)
        if s is None:
            d = self.tx[tid]
            asc = "".join(str(self.fa[d["chrom"]][es - 1:ee]) for es, ee in d["ex"])
            s = asc if d["strand"] == "+" else _rc_dna(asc)
            self._seq[tid] = s
        return s


def _map_rows_to_tx(df, tx, index, max_exon_len, ctx,
                    chr_col, strand_col, start_col, end_col, mre_col):
    """Map each row to its MANE host transcript (full containment + seq guard).

    Returns (tids[object], txpos[int64 5'-spliced coord], mapped[bool],
    n_nohost, n_seqfail).  txpos is a constant 25-nt offset from the centre, so
    it is fine as the neighbour anchor (only |Δ| matters)."""
    s_arr = df[start_col].to_numpy(np.int64)
    e_arr = df[end_col].to_numpy(np.int64)
    chrom = df[chr_col].astype(str).to_numpy()
    strand = df[strand_col].astype(str).to_numpy()
    mre = (df[mre_col].astype(str).str.upper()
           .str.replace("U", "T", regex=False).to_numpy())

    n = len(df)
    tids = np.empty(n, dtype=object)
    txpos = np.full(n, -1, dtype=np.int64)
    mapped = np.zeros(n, dtype=bool)
    n_nohost = n_seqfail = 0
    for j in range(n):
        chrom_fa = _tsv_chrom_to_fa(chrom[j])
        if chrom_fa not in ctx.fa:
            n_nohost += 1
            continue
        host = _find_host_exon(index, max_exon_len, chrom_fa, strand[j],
                               int(s_arr[j]), int(e_arr[j]))
        if host is None:
            n_nohost += 1
            continue
        cum, es, tid = host
        Lt = tx[tid]["Lt"]
        a_s = cum + (int(s_arr[j]) - es)
        a_e = cum + (int(e_arr[j]) - es)
        tlo = a_s if strand[j] == "+" else (Lt - 1 - a_e)
        if 0 <= tlo and ctx.txseq(tid)[tlo:tlo + MRE_LEN] == mre[j]:
            tids[j] = tid
            txpos[j] = tlo
            mapped[j] = True
        else:
            n_seqfail += 1
    return tids, txpos, mapped, n_nohost, n_seqfail


def _neighbor_counts_transcript(df, score, conf, window, min_sep,
                                tx, index, max_exon_len, ctx,
                                chr_col, strand_col, start_col, end_col, mre_col):
    """Distinct confident-positive neighbours within a SPLICED band [min_sep,
    window] along the same MANE host transcript.  Each row has at most one host,
    so counts assign directly.  Returns (counts, mapped, n_nohost, n_seqfail)."""
    centers = ((df[start_col].to_numpy(np.int64)
                + df[end_col].to_numpy(np.int64)) // 2)
    conf_mask = np.asarray(score, float) >= conf
    tids, txpos, mapped, n_nohost, n_seqfail = _map_rows_to_tx(
        df, tx, index, max_exon_len, ctx,
        chr_col, strand_col, start_col, end_col, mre_col)

    counts = np.zeros(len(df), dtype=np.int32)
    sel = np.where(mapped)[0]
    if sel.size == 0:
        return counts, mapped, n_nohost, n_seqfail

    thr = max(min_sep, 1)
    long = pd.DataFrame({"tid": tids[sel], "row": sel, "pos": txpos[sel],
                         "center": centers[sel], "conf": conf_mask[sel]})
    for _, sub in long.groupby("tid", sort=False):
        cdf = sub[sub["conf"]].drop_duplicates("center")
        if cdf.empty:
            continue
        order = np.argsort(cdf["pos"].to_numpy())
        cpos = cdf["pos"].to_numpy()[order]
        ccen = cdf["center"].to_numpy()[order]
        rpos = sub["pos"].to_numpy()
        rrow = sub["row"].to_numpy()
        rcen = sub["center"].to_numpy()
        lo = np.searchsorted(cpos, rpos - window, "left")
        hi = np.searchsorted(cpos, rpos + window, "right")
        nlo = np.searchsorted(cpos, rpos - (thr - 1), "left")
        nhi = np.searchsorted(cpos, rpos + (thr - 1), "right")
        for k in range(len(rrow)):
            if hi[k] == lo[k]:
                continue
            neigh = np.concatenate((ccen[lo[k]:nlo[k]], ccen[nhi[k]:hi[k]]))
            neigh = neigh[neigh != rcen[k]]
            counts[rrow[k]] = neigh.size                 # unique centres already
    return counts, mapped, n_nohost, n_seqfail


# ---------------------------------------------------------------------------
# Negative binding-type undersampling
#
# Label each pair with the project's canonical binding type (the shared
# binding_types classifier — same definition error_analysis.py uses), then build
# WeightedRandomSampler weights that draw negatives of chosen types less often
# while holding the positive:negative sampling ratio fixed — so the model sees
# fewer of e.g. seedless / 3'-compensatory decoys without changing the class
# balance it trains against.
# ---------------------------------------------------------------------------

# Targetable (non-canonical) categories, from the shared classifier.
BINDING_TYPES = _bt.UNDERSAMPLE_CATEGORIES


def _binding_types(mirna_idx: np.ndarray, mre_idx: np.ndarray,
                   cache_path: str | Path | None = None) -> np.ndarray:
    """Per-row canonical binding categories for the given token arrays.

    Classification is pure-Python per row (slow on millions of rows), so when a
    ``cache_path`` (the source file) is given the result is memoised to a
    ``<path>.bindtype.npz`` sidecar, keyed on the classifier version, row count
    and source mtime.  CV folds pass no path and are classified fresh.
    """
    if cache_path is not None:
        cp = Path(str(cache_path) + ".bindtype.npz")
        try:
            if cp.exists() and cp.stat().st_mtime >= Path(cache_path).stat().st_mtime:
                z = np.load(cp, allow_pickle=False)
                if (int(z["version"][0]) == _bt.CLASSIFIER_VERSION
                        and int(z["n"][0]) == len(mirna_idx)):
                    types = z["types"]
                    z.close()
                    print(f"  [bindtype] loaded {cp.name}")
                    return types
                z.close()
        except Exception as e:
            print(f"  [bindtype] ignoring unreadable cache {cp.name}: {e}")

    types = _bt.classify_index_arrays(mirna_idx, mre_idx)

    if cache_path is not None:
        try:
            np.savez(cp, version=np.array([_bt.CLASSIFIER_VERSION]),
                     n=np.array([len(mirna_idx)]), types=types)
            print(f"  [bindtype] wrote {cp.name}")
        except Exception as e:
            print(f"  [bindtype] could not write {cp.name}: {e}")
    return types


def _keep_binding_mask(binding_types: np.ndarray,
                       keep_cats: list[str]) -> np.ndarray:
    """Boolean keep-mask for rows whose binding type matches one of ``keep_cats``.

    A row matches a token when its classified type equals the token or is a
    sub-type of it: ``label == token`` or ``label.startswith(token + ".")``.  So
    ``seedless`` / ``3prime.compensatory`` match exactly, while a base canonical
    token like ``8mer1A`` also keeps its suffixed variants (``8mer1A.GU``,
    ``8mer1A.GU.3prime``).  Exits if a token matches no row (typo guard),
    printing the binding types actually present.
    """
    bt = np.asarray(binding_types)
    mask = np.zeros(len(bt), dtype=bool)
    unmatched = []
    for tok in keep_cats:
        m = (bt == tok) | np.char.startswith(bt, tok + ".")
        if not m.any():
            unmatched.append(tok)
        mask |= m
    if unmatched:
        present = sorted(set(bt.tolist()))
        sys.exit(f"ERROR: --keep-binding-type: token(s) {unmatched} match no rows. "
                 f"Binding types present: {present}")
    return mask


def _filter_binding_types(ds: "MiRNAInteractionDataset", keep_cats: list[str],
                          split: str,
                          cache_path: str | Path | None = None) -> None:
    """Restrict ``ds`` in place to rows whose binding type is in ``keep_cats``.

    Classifies every row with the shared binding_types classifier (memoised to
    the ``<path>.bindtype.npz`` sidecar when ``cache_path`` is given, since the
    full-set count still matches the source file at this point), then subsets the
    dataset and logs the kept positive/negative split and per-type distribution.
    ``split`` is only a label for the log line.
    """
    types  = _binding_types(ds.mirna_idx, ds.mre_idx, cache_path=cache_path)
    mask   = _keep_binding_mask(types, keep_cats)
    before = len(ds)
    ds.subset(mask)
    uniq, counts = np.unique(types[mask], return_counts=True)
    dist = " ".join(f"{t}({c})" for t, c in
                    sorted(zip(uniq.tolist(), counts.tolist()), key=lambda x: -x[1]))
    pos = int(ds.labels.sum())
    print(f"  [keep-binding-type:{split}] {before} -> {len(ds)} rows "
          f"({pos} pos / {len(ds) - pos} neg); kept: {dist}")


def _parse_undersample_spec(tokens) -> dict:
    """Parse ``--undersample-neg-type`` tokens into {category: factor}.

    Each token is ``CATEGORY`` or ``CATEGORY:FACTOR``.  FACTOR is the
    sampling-weight multiplier (0 = drop, 1 = no change); a bare ``CATEGORY``
    defaults to 0.0.  Different factors per category are allowed (e.g. the
    distribution-matching values differ by type).
    """
    spec: dict = {}
    for tok in tokens:
        cat, sep, raw = tok.partition(":")
        if sep:
            try:
                factor = float(raw)
            except ValueError:
                sys.exit(f"ERROR: --undersample-neg-type {tok!r}: factor "
                         f"{raw!r} is not a number.")
        else:
            factor = 0.0
        if cat not in BINDING_TYPES:
            sys.exit(f"ERROR: --undersample-neg-type {tok!r}: unknown category "
                     f"{cat!r}; choose from {list(BINDING_TYPES)}.")
        if factor < 0:
            sys.exit(f"ERROR: --undersample-neg-type {tok!r}: factor must be "
                     f">= 0, got {factor}.")
        spec[cat] = factor
    return spec


def _undersample_weights(labels: np.ndarray, binding_types: np.ndarray,
                         type_factors: dict,
                         balance: bool) -> Optional[np.ndarray]:
    """Per-sample WeightedRandomSampler weights that under-represent negatives by
    a per-category factor (0 = effectively drop, 1 = no change) drawn from
    ``type_factors`` ({category: factor}), holding the positive:negative sampling
    ratio fixed.

    The target neg:pos mass ratio is 1.0 when ``balance`` is set (class-balanced)
    and the natural n_neg/n_pos otherwise.  Within negatives, each targeted
    category is scaled by its factor and the rest absorb the freed mass, so the
    overall class ratio is unchanged — only the *composition* of the negatives
    shifts.
    """
    labels = np.asarray(labels)
    is_pos = labels == 1
    is_neg = ~is_pos
    n_pos, n_neg = int(is_pos.sum()), int(is_neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None

    factor = np.ones(len(labels), dtype=np.float64)
    for cat, a in type_factors.items():
        factor[binding_types == cat] = a
    factor[is_pos] = 1.0                       # never touch positives

    raw_neg  = factor[is_neg]
    neg_mass = raw_neg.sum()
    if neg_mass <= 0:                          # every negative targeted with factor 0
        print("  [undersample] all negatives targeted with factor 0; skipping "
              "(would leave no negatives to sample)")
        return None

    R = 1.0 if balance else n_neg / n_pos      # target neg:pos mass ratio
    w = np.empty(len(labels), dtype=np.float64)
    w[is_pos] = 1.0 / n_pos                     # positive mass = 1
    w[is_neg] = raw_neg * (R / neg_mass)        # negative mass = R
    parts = [f"{cat}×{a}({int(((binding_types == cat) & is_neg).sum())})"
             for cat, a in type_factors.items()]
    print(f"  [undersample] {' '.join(parts)} of {n_neg} neg "
          f"(target neg:pos mass = {R:.3f})")
    return w


def _train_sampler_weights(train_ds: "MiRNAInteractionDataset",
                           args: argparse.Namespace,
                           cache_path: str | Path | None = None
                           ) -> Optional[np.ndarray]:
    """Sampler weights for negative-binding-type undersampling, or None when the
    feature (``--undersample-neg-type``) is disabled."""
    tokens = getattr(args, "undersample_neg_type", None)
    if not tokens:
        return None
    type_factors  = _parse_undersample_spec(tokens)
    binding_types = _binding_types(train_ds.mirna_idx, train_ds.mre_idx,
                                   cache_path=cache_path)
    return _undersample_weights(
        train_ds.labels, binding_types, type_factors, balance=args.balance)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _model_args_from_cli(args: argparse.Namespace) -> dict:
    return {
        "seq_filters":      args.seq_filters,
        "seq_dim":          args.seq_dim,
        "seq_dropout":      args.seq_dropout,
        "seq_pairing":      args.seq_pairing,
        "pair_embed_dim":   args.pair_embed_dim,
        "seq_pool":         args.seq_pool,
        "seq_nbr_feature":  args.seq_nbr_feature,
        "n_conv_blocks":    args.n_conv_blocks,
        "n_pool_blocks":    args.n_pool_blocks,
        "block_pool":       args.block_pool,
        "activation":       args.activation,
    }


class ModelEMA:
    """Exponential moving average of model weights.

    Shadows the full ``state_dict`` (parameters *and* buffers), so any
    normalization running statistics are averaged alongside the weights and no
    separate stats-recompute pass is needed. Non-persistent buffers (e.g. the
    pairing lookup table) are not in ``state_dict`` and are left untouched.

    Evaluate with the averaged weights via ``apply_to`` / ``restore``; the raw
    training weights are unaffected between updates.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        self._backup: Optional[dict] = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:                       # counters (e.g. num_batches_tracked)
                s.copy_(v)

    def state_dict(self, model: nn.Module) -> dict:
        """EMA weights cast to the model's dtypes, ready for ``load_state_dict``."""
        target = model.state_dict()
        return {k: self.shadow[k].to(target[k].dtype) for k in target}

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.state_dict(model))

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if self._backup is not None:
            model.load_state_dict(self._backup)
            self._backup = None


def _train_one_run(
    model: MiRBindCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_ds: MiRNAInteractionDataset,
    model_args: dict,
    args: argparse.Namespace,
    device: torch.device,
    out_path: Path,
) -> float:
    gamma = getattr(args, "focal_gamma", 0.0)

    pos_weight: Optional[torch.Tensor] = None
    if not args.balance and gamma == 0.0:
        n_pos = int(train_ds.labels.sum())
        n_neg = len(train_ds.labels) - n_pos
        if n_pos > 0 and n_neg > 0:
            pw = n_neg / n_pos
            pos_weight = torch.tensor([pw], device=device)
            print(f"  BCEWithLogitsLoss pos_weight = {pw:.3f}")

    if gamma > 0.0:
        print(f"  Focal loss enabled (gamma={gamma}, pos_weight disabled)")

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps  = args.epochs * max(1, len(train_loader))
    warmup_steps = min(args.warmup_steps, total_steps // 10)
    if warmup_steps > 0:
        sched = torch.optim.lr_scheduler.SequentialLR(
            optim,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optim, start_factor=1e-6, end_factor=1.0,
                    total_iters=warmup_steps),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim, T_max=total_steps - warmup_steps),
            ],
            milestones=[warmup_steps],
        )
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps)

    ema = ModelEMA(model, decay=args.ema_decay) if getattr(args, "ema", False) else None
    if ema is not None:
        print(f"  weight EMA enabled (decay={args.ema_decay})")

    no_val = getattr(args, "no_val", False)
    best_val = -float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        seen = 0
        train_logits_buf, train_labels_buf = [], []

        for mi, ti, nbr, labels in train_loader:
            mi     = mi.to(device,     non_blocking=True)
            ti     = ti.to(device,     non_blocking=True)
            nbr    = nbr.to(device,    non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            logits = model(mi, ti, nbr)
            loss   = _compute_loss(logits, labels, pos_weight, gamma)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            if ema is not None:
                ema.update(model)

            running_loss += loss.item() * mi.size(0)
            seen += mi.size(0)
            train_logits_buf.append(logits.detach().cpu().numpy())
            train_labels_buf.append(labels.cpu().numpy().astype(int))

        train_loss    = running_loss / max(seen, 1)
        train_metrics = _binary_metrics(
            np.concatenate(train_logits_buf), np.concatenate(train_labels_buf))

        # No-validation mode: train the full epoch budget on all data, never
        # evaluate or early-stop, and save the (EMA, if enabled) weights at the
        # final epoch. Used to refit on the entire training set after CV.
        if no_val:
            dt = time.time() - t0
            print(f"[{epoch:03d}/{args.epochs}] "
                  f"train_loss={train_loss:.4f}  "
                  f"train_f1={train_metrics['f1']:.4f}  "
                  f"train_acc={train_metrics['accuracy']:.4f}  ({dt:.1f}s)")
            if HAS_WANDB and wandb.run is not None:
                log_dict = {
                    "epoch":          epoch,
                    "lr":             sched.get_last_lr()[0],
                    "train/loss":     train_loss,
                    "train/f1":       train_metrics["f1"],
                    "train/accuracy": train_metrics["accuracy"],
                }
                if "auroc" in train_metrics:
                    log_dict["train/auroc"] = train_metrics["auroc"]
                if "auprc" in train_metrics:
                    log_dict["train/auprc"] = train_metrics["auprc"]
                wandb.log(log_dict)
            if epoch == args.epochs:
                sel_variant = "ema" if ema is not None else "raw"
                sel_state   = (ema.state_dict(model) if ema is not None
                               else model.state_dict())
                torch.save({
                    "model_state":  sel_state,
                    "model_args":   model_args,
                    "val_metrics":  None,
                    "epoch":        epoch,
                    "weights":      sel_variant,
                }, out_path)
                print(f"  → checkpoint saved ({sel_variant}, final epoch, "
                      f"no validation): {out_path}")
            continue

        val_metrics   = evaluate(model, val_loader, device, pos_weight, gamma)

        def _ckpt_val(vm: dict) -> float:
            return vm.get(args.checkpoint_metric, -vm["loss"])

        # Candidate checkpoints for this epoch: the raw weights, and (if enabled)
        # the EMA weights. Keep whichever scores higher on the validation metric.
        sel_variant, sel_metrics = "raw", val_metrics
        val_metrics_ema = None
        if ema is not None:
            ema.apply_to(model)
            val_metrics_ema = evaluate(model, val_loader, device, pos_weight, gamma)
            ema.restore(model)
            if _ckpt_val(val_metrics_ema) > _ckpt_val(val_metrics):
                sel_variant, sel_metrics = "ema", val_metrics_ema

        dt       = time.time() - t0
        ckpt_val = _ckpt_val(sel_metrics)
        improved = ckpt_val > best_val

        log = (f"[{epoch:03d}/{args.epochs}] "
               f"train_loss={train_loss:.4f}  "
               f"val_loss={val_metrics['loss']:.4f}  "
               f"val_f1={val_metrics['f1']:.4f}  "
               f"val_acc={val_metrics['accuracy']:.4f}")
        if "auroc" in val_metrics:
            log += f"  val_auroc={val_metrics['auroc']:.4f}"
        if "auprc" in val_metrics:
            log += f"  val_auprc={val_metrics['auprc']:.4f}"
        if val_metrics_ema is not None and "auprc" in val_metrics_ema:
            log += f"  ema_auprc={val_metrics_ema['auprc']:.4f}"
        log += f"  ({dt:.1f}s)" + (f" *[{sel_variant}]" if improved else "")
        print(log)

        if HAS_WANDB and wandb.run is not None:
            log_dict: dict = {
                "epoch":           epoch,
                "lr":              sched.get_last_lr()[0],
                "train/loss":      train_loss,
                "train/f1":        train_metrics["f1"],
                "train/accuracy":  train_metrics["accuracy"],
                "val/loss":        val_metrics["loss"],
                "val/f1":          val_metrics["f1"],
                "val/accuracy":    val_metrics["accuracy"],
            }
            for split, metrics in (("train", train_metrics), ("val", val_metrics)):
                if "auroc" in metrics:
                    log_dict[f"{split}/auroc"] = metrics["auroc"]
                if "auprc" in metrics:
                    log_dict[f"{split}/auprc"] = metrics["auprc"]
            wandb.log(log_dict)

        if improved:
            best_val = ckpt_val
            patience_counter = 0
            sel_state = (ema.state_dict(model) if sel_variant == "ema"
                         else model.state_dict())
            torch.save({
                "model_state":  sel_state,
                "model_args":   model_args,
                "val_metrics":  sel_metrics,
                "epoch":        epoch,
                "weights":      sel_variant,
            }, out_path)
            print(f"  → checkpoint saved ({sel_variant}): {out_path}")
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"Early stopping after {patience_counter} epochs without improvement.")
                break

    if no_val:
        print(f"\nTrained {args.epochs} epochs on the full set (no validation).")
        return float("nan")
    print(f"\nBest {args.checkpoint_metric} = {best_val:.4f}")
    if HAS_WANDB and wandb.run is not None:
        wandb.run.summary[f"best_{args.checkpoint_metric}"] = best_val
    return best_val


def _run_single(args: argparse.Namespace, device: torch.device) -> None:
    if not args.val and not args.no_val:
        sys.exit("ERROR: --val is required when --folds is not set "
                 "(or pass --no-val to train on the full set without validation).")
    if args.val and args.no_val:
        print("WARNING: --no-val is set; ignoring --val and training on the full set.")

    cache = not args.no_cache
    print("Loading training data ...")
    if args.dedup:
        # Dedup mutates the frame, so the path-keyed cache would be stale;
        # read + dedup + build from the deduped DataFrame instead.
        train_df = _dedup_pairs(_read_table(args.train),
                                args.mirna_col, args.mre_col, args.dedup)
        train_ds = MiRNAInteractionDataset.from_df(
            train_df, has_labels=True,
            mre_col=args.mre_col, mirna_col=args.mirna_col, nbr_col=args.nbr_col)
    else:
        train_ds = MiRNAInteractionDataset(
            args.train, has_labels=True,
            mre_col=args.mre_col, mirna_col=args.mirna_col, nbr_col=args.nbr_col, cache=cache)
    print(f"  train samples : {len(train_ds)}")
    print(f"  positives     : {int(train_ds.labels.sum())} / {len(train_ds.labels)}")
    if args.seq_nbr_feature and train_ds.nbr is None:
        print(f"  WARNING: --seq-nbr-feature set but column {args.nbr_col!r} not "
              f"found in {args.train}; the neighbour count will be all zeros "
              f"(head degrades to the no-neighbour baseline). Materialise the "
              f"column first with the `neighbor-counts` subcommand, or drop the "
              f"flag.")

    val_loader = None
    if not args.no_val:
        print("Loading validation data ...")
        val_ds = MiRNAInteractionDataset(
            args.val, has_labels=True,
            mre_col=args.mre_col, mirna_col=args.mirna_col, nbr_col=args.nbr_col, cache=cache)
        print(f"  val samples   : {len(val_ds)}")
        if args.keep_binding_type and args.keep_binding_type_val:
            _filter_binding_types(val_ds, args.keep_binding_type, "val",
                                  cache_path=(args.val if cache else None))
        val_loader = _make_loader(val_ds,   args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)

    # Cache binding-type labels keyed to the source file, but not when --dedup is
    # active (the deduped row set no longer matches the on-disk file).
    bt_cache_path = args.train if (cache and not args.dedup) else None
    if args.keep_binding_type:
        _filter_binding_types(train_ds, args.keep_binding_type, "train",
                              cache_path=bt_cache_path)
        # The filtered row set no longer matches the on-disk file, so the
        # bindtype sidecar can't be reused/written for the undersample pass.
        bt_cache_path = None
    train_weights = _train_sampler_weights(train_ds, args, cache_path=bt_cache_path)
    train_loader = _make_loader(train_ds, args.batch_size, shuffle=True,
                                num_workers=args.num_workers, balance=args.balance,
                                weights=train_weights)

    model_args = _model_args_from_cli(args)
    model = MiRBindCNN(**model_args).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    if HAS_WANDB and getattr(args, "wandb_project", None):
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or None,
            group=args.wandb_group or None,
            config={**model_args, "epochs": args.epochs, "batch_size": args.batch_size,
                    "lr": args.lr, "weight_decay": args.weight_decay},
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _train_one_run(model, train_loader, val_loader, train_ds,
                   model_args, args, device, out_path)

    if HAS_WANDB and wandb.run is not None:
        wandb.finish()


def _run_kfold(args: argparse.Namespace, device: torch.device) -> None:
    if not HAS_SKLEARN:
        sys.exit("ERROR: scikit-learn is required for --folds.")

    print(f"Loading data for {args.folds}-fold stratified group cross-validation ...")
    df = _read_table(args.train)
    print(f"  {len(df)} rows")
    if args.dedup:
        df = _dedup_pairs(df, args.mirna_col, args.mre_col, args.dedup)

    # Binding-type filter: restrict the whole pool before the CV split, so every
    # fold (train and held-out) is drawn from the chosen subdomain.  --keep-
    # binding-type-val does not apply here (there is no separate held-out val).
    if args.keep_binding_type:
        mi = _encode_seqs(df[args.mirna_col].astype(str).tolist(), MAX_MIRNA)
        ti = _encode_seqs(df[args.mre_col].astype(str).tolist(),   MRE_LEN)
        types = _binding_types(
            mi, ti, cache_path=(args.train if not args.dedup else None))
        mask  = _keep_binding_mask(types, args.keep_binding_type)
        df    = df[mask].reset_index(drop=True)
        print(f"  [keep-binding-type] filtered to {len(df)} rows "
              f"matching {args.keep_binding_type}")

    if args.family_col not in df.columns:
        sys.exit(f"ERROR: --family-col '{args.family_col}' not found. "
                 f"Available: {list(df.columns)}")

    groups = df[args.family_col].fillna("unknown").astype(str).values
    n_families = len(set(groups))
    print(f"  {n_families} unique miRNA families → {args.folds} folds")

    out_path   = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_args = _model_args_from_cli(args)
    # shuffle=True + a fixed random_state makes the fold assignment reproducible
    # across runs (random_state is ignored by StratifiedGroupKFold unless shuffle).
    gkf        = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed)
    print(f"  split seed: {args.seed}")
    fold_scores: list[float] = []
    test_paths  = args.test or []
    fold_test_metrics: dict[str, list[dict]] = {Path(p).stem: [] for p in test_paths}
    # Per-fold probabilities on each test set, accumulated for fold ensembling.
    # Test loaders use shuffle=False, so sample order is identical across folds
    # and probabilities can be averaged element-wise before scoring.
    ensemble_probs:  dict[str, list[np.ndarray]] = {Path(p).stem: [] for p in test_paths}
    ensemble_labels: dict[str, np.ndarray] = {}
    # Out-of-fold predictions for the whole --train set: each row is scored once,
    # by the fold whose held-out split contains it (so by a model that never
    # trained on it). NaN-initialised; the union of val splits covers every row.
    oof_probs = (np.full(len(df), np.nan, dtype=np.float64)
                 if args.oof_out else None)

    for fold, (train_idx, val_idx) in enumerate(
            gkf.split(df, y=df["label"].values, groups=groups), 1):
        val_families = sorted(set(groups[val_idx]))
        print(f"\n{'='*60}")
        print(f"Fold {fold}/{args.folds}  train={len(train_idx)}  val={len(val_idx)}")
        preview = val_families[:8]
        suffix  = " ..." if len(val_families) > 8 else ""
        print(f"  val families ({len(val_families)}): {preview}{suffix}")
        print(f"{'='*60}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)

        train_ds = MiRNAInteractionDataset.from_df(
            train_df, has_labels=True,
            mre_col=args.mre_col, mirna_col=args.mirna_col, nbr_col=args.nbr_col)
        val_ds   = MiRNAInteractionDataset.from_df(
            val_df, has_labels=True,
            mre_col=args.mre_col, mirna_col=args.mirna_col, nbr_col=args.nbr_col)

        print(f"  train positives: {int(train_ds.labels.sum())} / {len(train_ds.labels)}")
        print(f"  val   positives: {int(val_ds.labels.sum())} / {len(val_ds.labels)}")

        train_weights = _train_sampler_weights(train_ds, args)
        train_loader = _make_loader(train_ds, args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, balance=args.balance,
                                    weights=train_weights)
        val_loader   = _make_loader(val_ds,   args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)

        fold_model_args = dict(model_args)
        model = MiRBindCNN(**fold_model_args).to(device)
        if fold == 1:
            print(f"Model parameters: "
                  f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

        if HAS_WANDB and getattr(args, "wandb_project", None):
            _group = args.wandb_group or args.wandb_run_name or out_path.stem
            _name  = (f"{args.wandb_run_name}_fold{fold}"
                      if args.wandb_run_name else f"{out_path.stem}_fold{fold}")
            wandb.init(
                project=args.wandb_project, entity=args.wandb_entity or None,
                name=_name, group=_group,
                config={**fold_model_args, "fold": fold, "n_folds": args.folds,
                        "epochs": args.epochs, "batch_size": args.batch_size,
                        "lr": args.lr, "weight_decay": args.weight_decay},
            )

        fold_out = out_path.parent / f"{out_path.stem}_fold{fold}{out_path.suffix}"
        score    = _train_one_run(model, train_loader, val_loader, train_ds,
                                  fold_model_args, args, device, fold_out)
        fold_scores.append(score)

        if test_paths or oof_probs is not None:
            best_ckpt = torch.load(fold_out, map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt["model_state"])

        if oof_probs is not None:
            # val_loader is shuffle=False over df.iloc[val_idx], so probs align
            # row-for-row with val_idx -> scatter them back to the global buffer.
            val_logits, _ = predict_logits(model, val_loader, device)
            oof_probs[val_idx] = 1.0 / (1.0 + np.exp(-val_logits))

        if test_paths:
            print(f"\n  Test-set evaluation (fold {fold} best checkpoint):")
            for test_path in test_paths:
                test_name = Path(test_path).stem
                test_ds   = MiRNAInteractionDataset.from_df(
                    _read_table(test_path),
                    has_labels=True,
                    mre_col=args.mre_col, mirna_col=args.mirna_col, nbr_col=args.nbr_col)
                test_loader = _make_loader(
                    test_ds, args.batch_size, shuffle=False,
                    num_workers=args.num_workers)
                logits, labels = predict_logits(model, test_loader, device)
                probs   = 1.0 / (1.0 + np.exp(-logits))
                metrics = _metrics_from_probs(probs, labels)
                fold_test_metrics[test_name].append(metrics)
                ensemble_probs[test_name].append(probs)
                ensemble_labels[test_name] = labels
                row = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"    [{test_name}]  {row}")

        if HAS_WANDB and wandb.run is not None:
            wandb.finish()

    print(f"\n{'='*60}")
    print(f"{args.folds}-fold CV results ({args.checkpoint_metric}):")
    for k, s in enumerate(fold_scores, 1):
        print(f"  fold {k}: {s:.4f}")
    mean, std = float(np.mean(fold_scores)), float(np.std(fold_scores))
    print(f"  mean  : {mean:.4f} ± {std:.4f}")

    if fold_test_metrics:
        print(f"\nTest-set summary across folds:")
        for test_name, metrics_list in fold_test_metrics.items():
            print(f"  {test_name}:")
            for metric_key in metrics_list[0]:
                vals      = [m[metric_key] for m in metrics_list]
                per_fold  = "  ".join(f"{v:.4f}" for v in vals)
                print(f"    {metric_key:<20s} folds=[{per_fold}]  "
                      f"mean={np.mean(vals):.4f} ± {np.std(vals):.4f}")

        # Fold ensemble: average per-fold probabilities, then score once.
        # Valid on the test sets (their rows are unseen by every fold); this is
        # NOT applied to the per-fold validation metric, where each row would be
        # scored by models that trained on it.
        print(f"\nFold-ensemble test results ({args.folds} folds, prob. average):")
        for test_name, probs_list in ensemble_probs.items():
            if not probs_list:
                continue
            mean_probs = np.mean(np.stack(probs_list), axis=0)
            ens        = _metrics_from_probs(mean_probs, ensemble_labels[test_name])
            print(f"  {test_name}:")
            for metric_key, v in ens.items():
                indiv     = [m[metric_key] for m in fold_test_metrics[test_name]]
                mean_indiv = float(np.mean(indiv))
                delta      = v - mean_indiv
                print(f"    {metric_key:<20s} ensemble={v:.4f}  "
                      f"(mean-of-folds={mean_indiv:.4f}, Δ={delta:+.4f})")

    if oof_probs is not None:
        n_missing = int(np.isnan(oof_probs).sum())
        oof_df = df.copy()
        oof_df["interaction_probability"] = np.nan_to_num(oof_probs, nan=0.0)
        oof_path = Path(args.oof_out)
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        oof_df.to_csv(oof_path, sep="\t", index=False)
        msg = f"\nWrote out-of-fold predictions for {len(df)} train rows → {oof_path}"
        if n_missing:
            msg += f"  ({n_missing} rows left unscored → 0.0)"
        print(msg)
        print("  Feed it to `neighbor-counts --score-col interaction_probability` "
              "for a leakage-free neighbour column.")

    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Train / predict subcommands
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    print(f"Device: {device}")
    _set_global_seed(args.seed, deterministic=args.deterministic)
    print(f"Global seed: {args.seed}"
          f"{' (deterministic cuDNN)' if args.deterministic else ''}")
    if args.folds and args.no_val:
        sys.exit("ERROR: --no-val cannot be combined with --folds "
                 "(cross-validation requires held-out folds).")
    if args.folds:
        _run_kfold(args, device)
    else:
        _run_single(args, device)


def _write_error_dump(path: str | Path, df_in: pd.DataFrame,
                      ds: "MiRNAInteractionDataset", probs: np.ndarray,
                      preds: np.ndarray, labels: np.ndarray) -> None:
    """Write a per-sample TSV for error analysis and print a quick breakdown.

    Loader order is preserved (shuffle=False), so rows align with `df_in` and
    the dataset arrays. Adds prediction columns, a TP/TN/FP/FN tag, and
    deterministic duplex stats.
    """
    adf = df_in.copy()
    adf["prob"]    = probs
    adf["pred"]    = preds
    adf["label"]   = labels
    adf["correct"] = (preds == labels).astype(int)

    et = np.full(len(adf), "??", dtype=object)
    et[(labels == 1) & (preds == 1)] = "TP"
    et[(labels == 0) & (preds == 0)] = "TN"
    et[(labels == 0) & (preds == 1)] = "FP"
    et[(labels == 1) & (preds == 0)] = "FN"
    adf["error_type"] = et

    for k, v in _duplex_stats(ds.mirna_idx, ds.mre_idx).items():
        adf[k] = v

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adf.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote error-analysis dump ({len(adf)} rows) → {out_path}")

    # Quick breakdown by outcome to orient the analysis.
    print("  by outcome:  count   mean_prob   n_wc  seed_pairs  max_run")
    for tag in ("TP", "FN", "TN", "FP"):
        m = et == tag
        if not m.any():
            continue
        print(f"    {tag}: {int(m.sum()):8d}   {adf['prob'][m].mean():8.3f}   "
              f"{adf['n_wc'][m].mean():5.1f}  {adf['seed_pairs'][m].mean():9.1f}  "
              f"{adf['max_run'][m].mean():7.1f}")


def _load_ckpt_model(checkpoint: str | Path,
                     device: torch.device) -> tuple[MiRBindCNN, dict]:
    """Load a trained checkpoint into an eval-mode model.

    Returns ``(model, ckpt)``. Checkpoints are self-contained: each stores its
    own ``model_args``, so a single checkpoint — or one fold of a k-fold run —
    can be applied to any input independently.
    """
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
    margs = dict(ckpt["model_args"])
    # backward-compat defaults: checkpoints predating these features used a
    # single WC(+wobble) channel with average pooling, so default to that when
    # the keys are absent (newer checkpoints carry their own values).
    for key, val in [("seq_pairing", "binary"), ("seq_pool", "avg"),
                     ("seq_nbr_feature", False),
                     ("n_conv_blocks", 6), ("n_pool_blocks", 4),
                     ("block_pool", "max"), ("activation", "leaky_relu")]:
        margs.setdefault(key, val)
    # Drop keys for removed branches (tspot / energy, conservation / eclip vector
    # branches, and the removed accessibility / conservation / positional channels
    # and attention pooling) so older checkpoints still reconstruct — their saved
    # weights for those branches, if any, are ignored and such checkpoints must be
    # retrained.
    for dead in ("use_tspot", "use_energy", "energy_dim",
                 "use_conservation", "use_eclip",
                 "vec_channels", "vec_blocks", "vec_kernel_size",
                 "vec_dropout", "norm",
                 "pool_heads", "seq_pos_channels", "seq_acc_channel",
                 "seq_con_channel", "con_transform", "con_scale",
                 "con_median", "con_iqr"):
        margs.pop(dead, None)
    model = MiRBindCNN(**margs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def cmd_predict(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    model, ckpt = _load_ckpt_model(args.checkpoint, device)
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch')}, "
          f"val_metrics={ckpt.get('val_metrics')})")

    ds = MiRNAInteractionDataset(
        args.input, has_labels=True,
        mre_col=args.mre_col, mirna_col=args.mirna_col, nbr_col=args.nbr_col, cache=not args.no_cache)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for mi, ti, nbr, labels in loader:
            mi     = mi.to(device)
            ti     = ti.to(device)
            nbr    = nbr.to(device)
            logits = model(mi, ti, nbr)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend((probs >= args.threshold).astype(int).tolist())
            all_labels.extend(labels.numpy().tolist())

    df_in = _read_table(args.input)
    df_in["interaction_probability"] = all_probs
    df_in["prediction"] = all_preds

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_in.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(df_in)} rows → {out_path}")

    if "label" in df_in.columns and len(np.unique(all_labels)) == 2:
        logits_np = np.log(np.array(all_probs) / (1.0 - np.array(all_probs) + 1e-9))
        metrics   = _binary_metrics(logits_np, np.array(all_labels), args.threshold)
        print("\nTest-set metrics:")
        for k, v in metrics.items():
            print(f"  {k:20s} = {v:.4f}")

    if args.error_dump:
        _write_error_dump(args.error_dump, df_in, ds,
                          np.array(all_probs), np.array(all_preds, dtype=int),
                          np.array(all_labels, dtype=int))


def cmd_neighbor_counts(args: argparse.Namespace) -> None:
    """Materialise the leakage-free neighbour-count column on an input TSV.

    Reads a table that already carries genomic coordinates and a per-row
    confidence score (e.g. ``interaction_probability`` from a held-out / k-fold
    out-of-fold prediction pass), computes the per-row count of distinct
    confident-positive neighbour sites, and writes the table back with the new
    column added.  Feed the resulting file to ``train``/``predict`` with
    ``--seq-nbr-feature --nbr-col <out-col>``.

    Leakage note: use OUT-OF-FOLD predictions for the score on training data —
    a model's in-sample confident calls on its own training neighbours are
    over-optimistic and would inflate the feature relative to inference.

    ``--mode`` selects how neighbours are counted:
      genomic    — linear distance on the same chr+strand (default; no GTF).
      transcript — spliced distance within the same MANE host transcript;
                   intronic/intergenic/straddling rows score 0.
      hybrid     — exon-mapped rows take the transcript count, the rest fall back
                   to the genomic count (mirrors the accessibility `acc_mode`).
    transcript/hybrid need ``--gtf`` + ``--genome`` and the MRE-sequence column
    (``--mre-col``) for the spliced-sequence guard.  Use the SAME mode/band for
    training and inference so the count distribution matches.
    """
    df = _read_table(args.input)
    need = [args.chr_col, args.strand_col, args.start_col, args.end_col,
            args.score_col]
    if args.mode in ("transcript", "hybrid"):
        need.append(args.mre_col)
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: neighbor-counts needs columns {need}; missing "
                 f"{missing}. Available: {list(df.columns)}")

    score = df[args.score_col].to_numpy(dtype=float)
    gen = _neighbor_counts(
        df, score, conf=args.conf, window=args.window, min_sep=args.min_sep,
        chr_col=args.chr_col, strand_col=args.strand_col,
        start_col=args.start_col, end_col=args.end_col)

    if args.mode == "genomic":
        counts = gen
    else:
        for path, flag in ((args.gtf, "--gtf"), (args.genome, "--genome")):
            if not path:
                sys.exit(f"ERROR: --mode {args.mode} needs {flag}.")
            if not Path(path).expanduser().exists():
                sys.exit(f"ERROR: {flag} not found: {path}")
        tx, index, max_exon_len = _parse_mane_gtf(Path(args.gtf).expanduser())
        ctx = _TxContext(str(Path(args.genome).expanduser()), tx)
        txc, mapped, n_nohost, n_seqfail = _neighbor_counts_transcript(
            df, score, conf=args.conf, window=args.window, min_sep=args.min_sep,
            tx=tx, index=index, max_exon_len=max_exon_len, ctx=ctx,
            chr_col=args.chr_col, strand_col=args.strand_col,
            start_col=args.start_col, end_col=args.end_col, mre_col=args.mre_col)
        counts = txc if args.mode == "transcript" else \
            np.where(mapped, txc, gen).astype(np.int32)
        print(f"  {int(mapped.sum()):,}/{len(df):,} sites "
              f"({100*mapped.mean():.1f}%) map to a MANE host transcript; "
              f"fallback {n_nohost:,} no-host + {n_seqfail:,} seq-mismatch.")

    df[args.out_col] = counts

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)

    nz = int((counts > 0).sum())
    print(f"Wrote {len(df)} rows → {out_path}")
    print(f"  neighbour column {args.out_col!r} (mode={args.mode}): conf>={args.conf} "
          f"in band [{args.min_sep}, {args.window}] nt")
    print(f"  {nz} rows ({100*nz/max(len(df),1):.1f}%) have >=1 neighbour; "
          f"max={int(counts.max())}, mean={counts.mean():.2f}")


def cmd_nbr_report(args: argparse.Namespace) -> None:
    """Summarise the neighbour-count distribution of one or more TSVs.

    For each input prints the count distribution (mean / %≥1 / max) and a
    bucketed breakdown (0, 1-2, 3-6, ≥7); when a label column is present it also
    shows the dose-response P(label=1) per bucket. Use it to check that the
    inference column built by `infer` matches the training column's
    distribution — if the buckets diverge, the confident-positive detector or
    `conf` differs between train and inference and the feature is miscalibrated.
    """
    buckets = [(0, 0, "0"), (1, 2, "1-2"), (3, 6, "3-6"), (7, None, ">=7")]
    for path in args.inputs:
        df = _read_table(path)
        if args.nbr_col not in df.columns:
            print(f"[{Path(path).stem}]  no column {args.nbr_col!r} "
                  f"(have {list(df.columns)[:8]}…); skipping")
            continue
        c   = df[args.nbr_col].to_numpy(dtype=np.float64)
        lab = (df[args.label_col].to_numpy(dtype=np.float64)
               if args.label_col in df.columns else None)
        nz  = float((c >= 1).mean())
        print(f"\n[{Path(path).stem}]  n={len(c):,}")
        print(f"  {args.nbr_col}: mean={c.mean():.3f}  std={c.std():.3f}  "
              f"%>=1={100*nz:.1f}%  max={int(c.max()) if len(c) else 0}")
        hdr = f"  {'bucket':<8s} {'n':>10s} {'frac':>8s}"
        if lab is not None:
            hdr += f" {'P(label=1)':>11s}"
        print(hdr)
        for lo, hi, name in buckets:
            m = (c >= lo) if hi is None else ((c >= lo) & (c <= hi))
            n = int(m.sum())
            row = f"  {name:<8s} {n:>10,d} {100*n/max(len(c),1):>7.1f}%"
            if lab is not None:
                row += (f" {lab[m].mean():>11.3f}" if n else f" {'—':>11s}")
            print(row)


def cmd_predict_ensemble(args: argparse.Namespace) -> None:
    """Average several fold checkpoints over one or more test sets.

    For each test set, every checkpoint's probabilities are computed in a fixed
    (shuffle=False) order and averaged element-wise, then scored once — the same
    fold-ensembling done at the end of a k-fold training run, but applied to
    already-trained checkpoints on new test sets. Writes a ``<name>_ensemble.tsv``
    per test set (per-fold probs + averaged probability + thresholded prediction).
    """
    device = torch.device(args.device)

    models: list[tuple[str, MiRBindCNN]] = []
    for ckpt_path in args.checkpoints:
        model, ckpt = _load_ckpt_model(ckpt_path, device)
        name = Path(ckpt_path).stem
        models.append((name, model))
        print(f"Loaded {name} (epoch {ckpt.get('epoch')}, "
              f"val_metrics={ckpt.get('val_metrics')})")
    print(f"\nEnsembling {len(models)} checkpoint(s) over "
          f"{len(args.inputs)} test set(s).")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for test_path in args.inputs:
        test_name = Path(test_path).stem
        print(f"\n{'='*60}\n[{test_name}]  {test_path}\n{'='*60}")
        # Encode the test set once (arrays cached); all folds share it.
        ds = MiRNAInteractionDataset(
            test_path, has_labels=True,
            mre_col=args.mre_col, mirna_col=args.mirna_col,
            nbr_col=args.nbr_col, cache=not args.no_cache)

        fold_probs: list[np.ndarray] = []
        per_fold: list[tuple[str, dict]] = []
        labels: Optional[np.ndarray] = None
        for name, model in models:
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)
            logits, labels = predict_logits(model, loader, device)
            probs   = 1.0 / (1.0 + np.exp(-logits))
            metrics = _metrics_from_probs(probs, labels, args.threshold)
            fold_probs.append(probs)
            per_fold.append((name, metrics))
            row = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"  {name:24s} {row}")

        mean_probs = np.mean(np.stack(fold_probs), axis=0)
        ens        = _metrics_from_probs(mean_probs, labels, args.threshold)

        print(f"\n  Fold-ensemble ({len(models)} ckpts, prob. average):")
        for metric_key, v in ens.items():
            indiv      = [m[metric_key] for _, m in per_fold]
            mean_indiv = float(np.mean(indiv))
            print(f"    {metric_key:<20s} ensemble={v:.4f}  "
                  f"(mean-of-folds={mean_indiv:.4f}, Δ={v - mean_indiv:+.4f})")

        df_out = _read_table(test_path)
        for name, probs in zip([n for n, _ in models], fold_probs):
            df_out[f"prob_{name}"] = probs
        df_out["interaction_probability"] = mean_probs
        df_out["prediction"] = (mean_probs >= args.threshold).astype(int)
        out_path = out_dir / f"{test_name}_ensemble.tsv"
        df_out.to_csv(out_path, sep="\t", index=False)
        print(f"  Wrote {len(df_out)} rows → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="miRBind-style 2D sequence branch CNN for miRNA–MRE classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ─────────────────────────────────────────────────────────────────
    tr = sub.add_parser("train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    tr.add_argument("--train",      required=True)
    tr.add_argument("--val",        default=None)
    tr.add_argument("--no-val",     action="store_true", dest="no_val",
                    help="Train on the entire --train set with no validation: "
                         "run the full --epochs budget (no early stopping) and "
                         "save the EMA weights (if --ema, else raw) at the final "
                         "epoch. Mutually exclusive with --folds.")
    tr.add_argument("--folds",      type=int, default=None)
    tr.add_argument("--seed",       type=int, default=42,
                    help="Global RNG seed (Python/NumPy/Torch): seeds the "
                         "StratifiedGroupKFold split, weight init, DataLoader "
                         "shuffling, the sampler and dropout. Same seed + same "
                         "data -> reproducible run.")
    tr.add_argument("--deterministic", action="store_true",
                    help="Also force cuDNN deterministic algorithms (slower, "
                         "but removes the last run-to-run variation on GPU).")
    tr.add_argument("--dedup",      choices=["first", "none"], default=None,
                    help="Drop duplicate (miRNA, MRE) sequence pairs from the "
                         "--train data before training. 'first' keeps the first "
                         "occurrence of each duplicated pair; 'none' drops every "
                         "row of a duplicated pair (keeps only unique pairs). "
                         "--val/--test inputs are left untouched.")
    tr.add_argument("--family-col", default="mirna_family", dest="family_col")
    tr.add_argument("--oof-out", default=None, dest="oof_out",
                    help="With --folds, write the --train table back to this path "
                         "with an out-of-fold `interaction_probability` column "
                         "(each row scored by the fold that held it out). Use it "
                         "as the leakage-free score for the `neighbor-counts` "
                         "subcommand. Ignored without --folds.")
    tr.add_argument("--test",       nargs="+", default=None, metavar="FILE")
    tr.add_argument("--out",        default="checkpoints/cnn_mirbind.pt")
    tr.add_argument("--mre-col",    default="mre_sequence",   dest="mre_col")
    tr.add_argument("--mirna-col",  default="mirna_sequence", dest="mirna_col")
    tr.add_argument("--nbr-col",    default="neighbor_count",  dest="nbr_col",
                    help="Column holding the per-pair leakage-free neighbour "
                         "count (materialise it with the `neighbor-counts` "
                         "subcommand). Used only when --seq-nbr-feature is set.")
    # Architecture
    tr.add_argument("--seq-filters",     type=int,   default=64,
                    dest="seq_filters",
                    help="Filters in the 2D miRBind conv blocks.")
    tr.add_argument("--seq-dim",         type=int,   default=128,
                    dest="seq_dim",
                    help="Output embedding size of the sequence branch.")
    tr.add_argument("--seq-dropout",     type=float, default=0.3, dest="seq_dropout",
                    help="Dropout in the 2D miRBind conv blocks and classifier head.")
    tr.add_argument("--seq-pairing", choices=["binary", "multi", "multi4", "embed"],
                    default="multi", dest="seq_pairing",
                    help="2D pairing matrix encoding: single WC(+wobble) channel "
                         "(binary), separate WC/wobble/mismatch channels (multi), "
                         "A·U/G·C/wobble/mismatch (multi4, WC split by strength), or "
                         "a learnable chemistry-initialised dense embedding (embed).")
    tr.add_argument("--pair-embed-dim", type=int, default=3, dest="pair_embed_dim",
                    help="Channels of the learnable pairing embedding when "
                         "--seq-pairing embed (ignored otherwise). Channel 0 is "
                         "initialised to the graded pairing-strength prior.")
    tr.add_argument("--seq-pool", choices=["avg", "gem"], default="gem",
                    dest="seq_pool",
                    help="Global pooling for the 2D sequence branch: average or "
                         "GeM (learnable power-mean).")
    tr.add_argument("--seq-nbr-feature", action="store_true", dest="seq_nbr_feature",
                    help="Feed the leakage-free neighbour-count scalar (from "
                         "--nbr-col) into the classifier head: a per-pair global "
                         "feature (count of nearby confident-positive sites) that "
                         "captures spatial clustering the sequence branch cannot "
                         "see. Off by default. Materialise the column first with "
                         "the `neighbor-counts` subcommand.")
    tr.add_argument("--n-conv-blocks", type=int, default=6, dest="n_conv_blocks",
                    help="Number of Conv2d blocks in the 2D sequence branch.")
    tr.add_argument("--n-pool-blocks", type=int, default=4, dest="n_pool_blocks",
                    help="How many of the first conv blocks downsample (2×2). "
                         "Must be <= --n-conv-blocks and <= 4 (the miRNA-axis "
                         "height of 30 only halves to 1 after 4 poolings).")
    tr.add_argument("--block-pool", choices=["max", "gem"], default="max",
                    dest="block_pool",
                    help="Downsampling pool inside the conv blocks: max pooling "
                         "or a strided learnable GeM.")
    tr.add_argument("--activation",
                    choices=["leaky_relu", "relu", "gelu", "silu", "elu", "selu"],
                    default="leaky_relu",
                    help="Activation for the 2D sequence branch conv/dense blocks.")
    # Training
    tr.add_argument("--epochs",       type=int,   default=40)
    tr.add_argument("--batch-size",   type=int,   default=256)
    tr.add_argument("--lr",           type=float, default=1e-3)
    tr.add_argument("--weight-decay", type=float, default=1e-4)
    tr.add_argument("--warmup-steps", type=int,   default=200)
    tr.add_argument("--num-workers",  type=int,   default=8)
    tr.add_argument("--patience",     type=int,   default=10)
    tr.add_argument("--ema",          action="store_true",
                    help="Track an exponential moving average of the weights and "
                         "checkpoint whichever of raw/EMA scores higher on val.")
    tr.add_argument("--ema-decay",    type=float, default=0.999, dest="ema_decay",
                    help="EMA decay; effective horizon ~1/(1-decay) steps. Lower "
                         "it (e.g. 0.99) for short runs so the average keeps up.")
    tr.add_argument("--focal-gamma",  type=float, default=0.0, dest="focal_gamma",
                    help="Focal loss gamma (default: 0 = standard BCE). "
                         "gamma=2 is the standard choice; higher values (3-4) "
                         "concentrate more gradient on hard examples. "
                         "When gamma>0, the loss-level pos_weight is disabled "
                         "(focal already handles imbalance); pass --balance to "
                         "use sampler oversampling instead.")
    tr.add_argument("--balance",      action="store_true")
    tr.add_argument("--undersample-neg-type", nargs="+", default=None,
                    metavar="TYPE[:FACTOR]", dest="undersample_neg_type",
                    help="Under-represent negatives of these canonical binding "
                         "categories via the training sampler, keeping the "
                         "positive:negative ratio fixed. Each token is CATEGORY "
                         "or CATEGORY:FACTOR, where FACTOR is the sampling-weight "
                         "multiplier (0 = drop, 1 = no change); a bare CATEGORY "
                         "defaults to 0.0. Per-category factors may differ. "
                         "Categories come from the shared binding_types classifier "
                         f"(same one error_analysis uses): {', '.join(BINDING_TYPES)}. "
                         "Example: --undersample-neg-type seedless:0.55 "
                         "3prime.compensatory:0.8")
    tr.add_argument("--keep-binding-type", nargs="+", default=None,
                    metavar="CATEGORY", dest="keep_binding_type",
                    help="Train only on samples whose classified binding type "
                         "matches one of these categories (both positives and "
                         "negatives are filtered). A sample matches a token when "
                         "its type equals the token or is a sub-type of it (token "
                         "'8mer1A' also keeps '8mer1A.GU.3prime'); 'seedless' and "
                         "'3prime.compensatory' match exactly. Categories come "
                         "from the shared binding_types classifier (same one "
                         "error_analysis uses). Exits if a token matches no row, "
                         "printing the types present. By default only the training "
                         "set is filtered (see --keep-binding-type-val). With "
                         "--folds the whole pool is filtered before the CV split. "
                         "Example: --keep-binding-type seedless 3prime.compensatory")
    tr.add_argument("--keep-binding-type-val", action="store_true",
                    dest="keep_binding_type_val",
                    help="Also restrict the validation set to --keep-binding-type "
                         "categories, so val metrics reflect the trained "
                         "subdomain. Off by default (val stays the full mixed "
                         "set). No effect with --folds, where the whole dataset is "
                         "filtered before splitting.")
    tr.add_argument("--no-cache",     action="store_true", dest="no_cache",
                    help="Disable the preprocessing .cnncache.npz sidecar files.")
    tr.add_argument("--checkpoint-metric",
                    choices=["auroc", "auprc", "f1", "accuracy"], default="auprc")
    tr.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    tr.add_argument("--wandb-project",  default=None, dest="wandb_project")
    tr.add_argument("--wandb-entity",   default=None, dest="wandb_entity")
    tr.add_argument("--wandb-run-name", default=None, dest="wandb_run_name")
    tr.add_argument("--wandb-group",    default=None, dest="wandb_group")

    # ── predict ───────────────────────────────────────────────────────────────
    pr = sub.add_parser("predict", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pr.add_argument("--checkpoint", required=True)
    pr.add_argument("--input",      required=True)
    pr.add_argument("--output",     required=True)
    pr.add_argument("--error-dump", default=None, dest="error_dump",
                    help="Also write a per-sample error-analysis TSV (probs, "
                         "error type, duplex stats, aux-feature summaries) to "
                         "this path. Requires labels in the input; no training.")
    pr.add_argument("--threshold",  type=float, default=0.5)
    pr.add_argument("--batch-size", type=int,   default=256)
    pr.add_argument("--num-workers", type=int,  default=4, dest="num_workers")
    pr.add_argument("--mre-col",   default="mre_sequence",   dest="mre_col")
    pr.add_argument("--mirna-col", default="mirna_sequence", dest="mirna_col")
    pr.add_argument("--nbr-col",   default="neighbor_count",  dest="nbr_col",
                    help="Per-pair neighbour-count column (used if the checkpoint "
                         "was trained with --seq-nbr-feature). Materialise it with "
                         "the `neighbor-counts` subcommand.")
    pr.add_argument("--no-cache",  action="store_true", dest="no_cache",
                    help="Disable the preprocessing .cnncache.npz sidecar files.")
    pr.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")

    # ----- predict-ensemble -------------------------------------------------
    pe = sub.add_parser("predict-ensemble",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                        help="Average several fold checkpoints over test set(s).")
    pe.add_argument("--checkpoints", required=True, nargs="+", metavar="CKPT",
                    help="Fold checkpoints to ensemble (e.g. *_fold1.pt ...).")
    pe.add_argument("--inputs", required=True, nargs="+", metavar="FILE",
                    help="Test CSV(s) to score; need a 'label' column for metrics.")
    pe.add_argument("--output-dir", default="predictions", dest="output_dir",
                    help="Directory for <testname>_ensemble.tsv outputs.")
    pe.add_argument("--threshold",  type=float, default=0.5)
    pe.add_argument("--batch-size", type=int,   default=256)
    pe.add_argument("--num-workers", type=int,  default=4, dest="num_workers")
    pe.add_argument("--mre-col",   default="mre_sequence",   dest="mre_col")
    pe.add_argument("--mirna-col", default="mirna_sequence", dest="mirna_col")
    pe.add_argument("--nbr-col",   default="neighbor_count",  dest="nbr_col",
                    help="Per-pair neighbour-count column (used if the "
                         "checkpoints were trained with --seq-nbr-feature).")
    pe.add_argument("--no-cache",  action="store_true", dest="no_cache",
                    help="Disable the preprocessing .cnncache.npz sidecar files.")
    pe.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")

    # ----- neighbor-counts --------------------------------------------------
    nc = sub.add_parser(
        "neighbor-counts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Materialise the leakage-free neighbour-count column from a "
             "predictions TSV (coords + a confidence score) for "
             "--seq-nbr-feature training.")
    nc.add_argument("--input",  required=True,
                    help="TSV with genomic coordinates and a per-row confidence "
                         "score (e.g. a predict / out-of-fold output).")
    nc.add_argument("--output", required=True,
                    help="Output TSV (input plus the neighbour-count column).")
    nc.add_argument("--score-col", default="interaction_probability",
                    dest="score_col",
                    help="Per-row confidence used to flag confident-positive "
                         "neighbour sites. Use OUT-OF-FOLD scores for training "
                         "data to stay leakage-free.")
    nc.add_argument("--conf",   type=float, default=0.8,
                    help="A site counts as a confident-positive neighbour when "
                         "its score >= this threshold.")
    nc.add_argument("--window", type=int, default=150,
                    help="Upper bound of the neighbour band in nt (centre-to-"
                         "centre, same chr+strand). ~150 covers the AGO2 "
                         "footprint + local folding domain.")
    nc.add_argument("--min-sep", type=int, default=0, dest="min_sep",
                    help="Lower bound of the neighbour band in nt: drop "
                         "neighbours closer than this. The row's own coordinate "
                         "is always excluded. ~60 isolates independent clustering "
                         "(two AGO2 footprints can't co-occupy < ~60 nt, and "
                         "closer sites share overlapping MRE fragments) and gives "
                         "a steeper per-neighbour signal; 0 counts all distinct "
                         "neighbours. Must match between training and inference.")
    nc.add_argument("--out-col", default="neighbor_count", dest="out_col",
                    help="Name of the neighbour-count column to write.")
    nc.add_argument("--mode", choices=["genomic", "transcript", "hybrid"],
                    default="genomic",
                    help="Neighbour distance frame: linear genomic (default), "
                         "spliced within the same MANE transcript, or hybrid "
                         "(transcript where exonic, genomic fallback elsewhere — "
                         "mirrors the accessibility acc_mode). transcript/hybrid "
                         "need --gtf + --genome. Use the same mode for train and "
                         "inference.")
    nc.add_argument("--gtf", default=None,
                    help="GENCODE GTF (MANE_Select tag); required for "
                         "--mode transcript/hybrid.")
    nc.add_argument("--genome", default=None,
                    help="GRCh38 primary-assembly .fa (indexed) for the spliced-"
                         "sequence guard; required for --mode transcript/hybrid.")
    nc.add_argument("--mre-col", default="gene", dest="mre_col",
                    help="MRE-sequence column for the spliced-sequence guard "
                         "(transcript/hybrid modes).")
    nc.add_argument("--chr-col",    default="chr",    dest="chr_col")
    nc.add_argument("--strand-col", default="strand", dest="strand_col")
    nc.add_argument("--start-col",  default="start",  dest="start_col")
    nc.add_argument("--end-col",    default="end",    dest="end_col")

    # ----- nbr-report -------------------------------------------------------
    nr = sub.add_parser(
        "nbr-report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Print the neighbour-count distribution (and dose-response if "
             "labels present) for one or more TSVs — compare train vs "
             "inference columns.")
    nr.add_argument("--inputs", required=True, nargs="+", metavar="FILE",
                    help="TSV(s) carrying the neighbour-count column.")
    nr.add_argument("--nbr-col", default="neighbor_count", dest="nbr_col",
                    help="Neighbour-count column to summarise.")
    nr.add_argument("--label-col", default="label", dest="label_col",
                    help="Label column for the per-bucket dose-response "
                         "(skipped if absent).")

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict-ensemble":
        cmd_predict_ensemble(args)
    elif args.command == "neighbor-counts":
        cmd_neighbor_counts(args)
    elif args.command == "nbr-report":
        cmd_nbr_report(args)
    else:
        cmd_predict(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
