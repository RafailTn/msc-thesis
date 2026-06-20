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
    LeakyReLU, BatchNorm2d, optional MaxPool2d, Dropout) processes the matrix,
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
_CACHE_VERSION = 3


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
                ds: "MiRNAInteractionDataset") -> None:
    cp = _cache_path(path)
    try:
        np.savez(
            cp,
            version=np.array([_CACHE_VERSION]),
            mre_col=np.array([mre_col]),
            mirna_col=np.array([mirna_col]),
            dims=np.array([MAX_MIRNA, MRE_LEN]),
            mirna_idx=ds.mirna_idx,
            mre_idx=ds.mre_idx,
            labels=ds.labels,
        )
        print(f"  [cache] wrote {cp.name}")
    except Exception as e:   # caching is best-effort; never fail training over it
        print(f"  [cache] could not write {cp.name}: {e}")


def _load_cache(path: str | Path, mre_col: str, mirna_col: str) -> Optional[dict]:
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
                or list(z["dims"]) != [MAX_MIRNA, MRE_LEN]):
            z.close()
            return None
        data = {
            "mirna_idx":  z["mirna_idx"],
            "mre_idx":    z["mre_idx"],
            "labels":     z["labels"],
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
        cache: bool = True,
    ) -> None:
        cached = _load_cache(path, mre_col, mirna_col) if cache else None
        if cached is not None:
            print(f"  [cache] loaded {_cache_path(path).name}")
            self.has_labels = has_labels
            self._set_arrays(**cached)
        else:
            self._init(_read_table(path), has_labels, mre_col, mirna_col)
            if cache:
                _save_cache(path, mre_col, mirna_col, self)

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        has_labels: bool = True,
        mre_col: str = "mre_sequence",
        mirna_col: str = "mirna_sequence",
    ) -> "MiRNAInteractionDataset":
        obj = cls.__new__(cls)
        obj._init(df, has_labels, mre_col, mirna_col)
        return obj

    def _init(
        self,
        df: pd.DataFrame,
        has_labels: bool,
        mre_col: str,
        mirna_col: str,
    ) -> None:
        self.has_labels = has_labels
        self._build_arrays(df, has_labels, mre_col, mirna_col)

    def _build_arrays(
        self,
        df: pd.DataFrame,
        has_labels: bool,
        mre_col: str,
        mirna_col: str,
    ) -> None:
        # Tokenise sequences once into int8 index matrices; the Watson–Crick
        # matrix is assembled on-device in the model forward pass.
        self.mirna_idx = _encode_seqs(df[mirna_col].astype(str).tolist(), MAX_MIRNA)  # (N, 30)
        self.mre_idx   = _encode_seqs(df[mre_col].astype(str).tolist(),   MRE_LEN)    # (N, 50)

        if has_labels and "label" in df.columns:
            self.labels = df["label"].astype(int).values
        else:
            self.labels = np.zeros(len(df), dtype=np.int64)

    def _set_arrays(
        self,
        mirna_idx: np.ndarray,
        mre_idx: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Populate arrays from a loaded cache."""
        self.mirna_idx  = mirna_idx
        self.mre_idx    = mre_idx
        self.labels = labels if self.has_labels else np.zeros(len(mirna_idx), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.mirna_idx)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.mirna_idx[idx]),          # (MAX_MIRNA,) int8
            torch.from_numpy(self.mre_idx[idx]),            # (MRE_LEN,)   int8
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

    Conv2d (5×5, same padding) → activation → BatchNorm2d →
    [MaxPool2d / GeMDownsample2d (2,2)] → Dropout.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.3,
                 pool: bool = True, block_pool: str = "max",
                 activation: str = "leaky_relu") -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2),
            _make_activation(activation),
            nn.BatchNorm2d(out_ch),
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
    """One miRBind-style dense block: Linear → activation → BatchNorm1d → Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3,
                 activation: str = "leaky_relu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            _make_activation(activation),
            nn.BatchNorm1d(out_dim),
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


class AttentionPool2d(nn.Module):
    """Content-based attention pooling over a 2D feature map: (B,C,H,W) -> (B, heads*C).

    A 1×1 scoring conv produces ``heads`` score maps; each is softmaxed over the
    H·W spatial cells and used to take a weighted sum of the C-dim feature
    vectors, yielding one pooled vector per head (concatenated on output).

    Unlike GeM (a fixed power-mean that collapses to the single strongest
    region), the weights are learned and content-dependent, so the softmax can
    place mass on several disjoint regions at once.  With multiple heads each can
    specialise — e.g. one to the seed block and one to the 3′-supplementary
    block — so both paired regions of a 3′-compensatory duplex reach the
    classifier instead of being averaged away.

    The scores are derived from features alone (no positional encoding) and the
    pooling is a weighted sum over positions, so the output is permutation- /
    translation-invariant, preserving the property that makes the 2D branch
    generalise.  Note this only helps if the pre-pool map keeps spatial
    resolution: with many pooling blocks the map is already ~1×k and there is
    little to attend over, so pair attention pooling with fewer --n-pool-blocks.
    """

    def __init__(self, channels: int, heads: int = 1) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError(f"pool_heads must be >= 1, got {heads}")
        self.heads = heads
        self.score = nn.Conv2d(channels, heads, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn  = self.score(x).flatten(2).softmax(dim=-1)    # (B, heads, H*W)
        feats = x.flatten(2)                                # (B, C,     H*W)
        pooled = torch.einsum("bhn,bcn->bhc", attn, feats)  # (B, heads, C)
        return pooled.flatten(1)                            # (B, heads*C)


class MiRBindSeqBranch(nn.Module):
    """miRBind-style sequence branch.

    Takes the 2D WC-complementarity matrix (B, 1, MAX_MIRNA, MRE_LEN) and
    produces a fixed-size embedding of shape (B, out_dim).

    Architecture (mirroring Klimentova et al. 2022):
      - n_conv_blocks Conv2d blocks (5×5, activation, BN2d, Dropout)
        First n_pool_blocks blocks downsample (MaxPool2d or GeMDownsample2d 2×2)
        Remaining blocks have no pooling (spatial dims small by this point)
      - global pool (GeM, adaptive-avg, or multi-head attention) → flatten
      - 2 dense blocks → out_dim

    The 2D input height is MAX_MIRNA (30), which halves with each pooling block
    (30→15→7→3→1), so n_pool_blocks must not exceed 4 — a 5th pool would reduce a
    size-1 dimension to 0. n_pool_blocks is also capped at n_conv_blocks.

    With pool="attention" the global pool is AttentionPool2d, which keeps
    pool_heads weighted views of the feature map (so the pooled width feeding the
    dense head is n_filters * pool_heads).
    """

    def __init__(
        self,
        n_filters: int = 64,
        out_dim:   int = 128,
        dropout:   float = 0.3,
        in_ch:     int = 1,
        pool:      str = "gem",
        pool_heads: int = 1,
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
            pooled_dim = n_filters
        elif pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            pooled_dim = n_filters
        elif pool == "attention":
            self.global_pool = AttentionPool2d(n_filters, heads=pool_heads)
            pooled_dim = n_filters * pool_heads
        else:
            raise ValueError(
                f"pool must be 'gem', 'avg' or 'attention', got {pool!r}")

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
        Global pool for the sequence branch: "gem", "avg" or "attention".
    pool_heads : int
        Number of attention heads when seq_pool="attention" (ignored otherwise).
    """

    def __init__(
        self,
        seq_filters:      int   = 64,
        seq_dim:          int   = 128,
        seq_dropout:      float = 0.3,
        seq_pairing:      str   = "multi",
        pair_embed_dim:   int   = 3,
        seq_pool:         str   = "gem",
        pool_heads:       int   = 1,
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
            in_ch=n_pair_ch, pool=seq_pool, pool_heads=pool_heads,
            n_conv_blocks=n_conv_blocks, n_pool_blocks=n_pool_blocks,
            block_pool=block_pool, activation=activation)

        # ── Classifier ────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(seq_dim),
            nn.GELU(),
            nn.Dropout(seq_dropout),
            nn.Linear(seq_dim, seq_dim // 2),
            nn.GELU(),
            nn.Dropout(seq_dropout),
            nn.Linear(seq_dim // 2, 1),
        )

    def forward(
        self,
        mi:     torch.Tensor,   # (B, MAX_MIRNA)  int nucleotide indices
        ti:     torch.Tensor,   # (B, MRE_LEN)    int nucleotide indices
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
        if self._pair_learnable:
            # Fixed tables zero pad rows/cols by construction; the learnable
            # embedding does not, so mask any cell touching pad index 4 (mi/ti==4).
            valid = ((mi < 4)[:, :, None] & (ti < 4)[:, None, :])  # (B, 30, 50)
            wc_mat = wc_mat * valid.unsqueeze(1).to(wc_mat.dtype)
        h_seq = self.seq_branch(wc_mat)         # (B, seq_dim)

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
                 persistent_workers: Optional[bool] = None) -> DataLoader:
    sampler = None
    if balance and shuffle and dataset.has_labels:
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

    for mi, ti, labels in loader:
        mi     = mi.to(device,     non_blocking=True)
        ti     = ti.to(device,     non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        logits = model(mi, ti)
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
    for mi, ti, labels in loader:
        mi     = mi.to(device,     non_blocking=True)
        ti     = ti.to(device,     non_blocking=True)
        logits = model(mi, ti)
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
        "pool_heads":       args.pool_heads,
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
    pos_weight: Optional[torch.Tensor] = None
    if not args.balance:
        n_pos = int(train_ds.labels.sum())
        n_neg = len(train_ds.labels) - n_pos
        if n_pos > 0 and n_neg > 0:
            pw = n_neg / n_pos
            pos_weight = torch.tensor([pw], device=device)
            print(f"  BCEWithLogitsLoss pos_weight = {pw:.3f}")

    gamma = getattr(args, "focal_gamma", 0.0)
    if gamma > 0.0:
        print(f"  Focal loss enabled (gamma={gamma})")

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

        for mi, ti, labels in train_loader:
            mi     = mi.to(device,     non_blocking=True)
            ti     = ti.to(device,     non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            logits = model(mi, ti)
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
            mre_col=args.mre_col, mirna_col=args.mirna_col)
    else:
        train_ds = MiRNAInteractionDataset(
            args.train, has_labels=True,
            mre_col=args.mre_col, mirna_col=args.mirna_col, cache=cache)
    print(f"  train samples : {len(train_ds)}")
    print(f"  positives     : {int(train_ds.labels.sum())} / {len(train_ds.labels)}")

    val_loader = None
    if not args.no_val:
        print("Loading validation data ...")
        val_ds = MiRNAInteractionDataset(
            args.val, has_labels=True,
            mre_col=args.mre_col, mirna_col=args.mirna_col, cache=cache)
        print(f"  val samples   : {len(val_ds)}")
        val_loader = _make_loader(val_ds,   args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)

    train_loader = _make_loader(train_ds, args.batch_size, shuffle=True,
                                num_workers=args.num_workers, balance=args.balance)

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
            mre_col=args.mre_col, mirna_col=args.mirna_col)
        val_ds   = MiRNAInteractionDataset.from_df(
            val_df, has_labels=True,
            mre_col=args.mre_col, mirna_col=args.mirna_col)

        print(f"  train positives: {int(train_ds.labels.sum())} / {len(train_ds.labels)}")
        print(f"  val   positives: {int(val_ds.labels.sum())} / {len(val_ds.labels)}")

        train_loader = _make_loader(train_ds, args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, balance=args.balance)
        val_loader   = _make_loader(val_ds,   args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)

        model = MiRBindCNN(**model_args).to(device)
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
                config={**model_args, "fold": fold, "n_folds": args.folds,
                        "epochs": args.epochs, "batch_size": args.batch_size,
                        "lr": args.lr, "weight_decay": args.weight_decay},
            )

        fold_out = out_path.parent / f"{out_path.stem}_fold{fold}{out_path.suffix}"
        score    = _train_one_run(model, train_loader, val_loader, train_ds,
                                  model_args, args, device, fold_out)
        fold_scores.append(score)

        if test_paths:
            best_ckpt = torch.load(fold_out, map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt["model_state"])
            print(f"\n  Test-set evaluation (fold {fold} best checkpoint):")
            for test_path in test_paths:
                test_name = Path(test_path).stem
                test_ds   = MiRNAInteractionDataset.from_df(
                    _read_table(test_path),
                    has_labels=True,
                    mre_col=args.mre_col, mirna_col=args.mirna_col)
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
                     ("pool_heads", 1),
                     ("n_conv_blocks", 6), ("n_pool_blocks", 4),
                     ("block_pool", "max"), ("activation", "leaky_relu")]:
        margs.setdefault(key, val)
    # Drop keys for removed branches (tspot / energy, and the conservation /
    # eclip vector branches) so older checkpoints still reconstruct — their
    # saved weights for those branches, if any, are ignored and such
    # checkpoints must be retrained.
    for dead in ("use_tspot", "use_energy", "energy_dim",
                 "use_conservation", "use_eclip",
                 "vec_channels", "vec_blocks", "vec_kernel_size",
                 "vec_dropout", "norm"):
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
        mre_col=args.mre_col, mirna_col=args.mirna_col, cache=not args.no_cache)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for mi, ti, labels in loader:
            mi     = mi.to(device)
            ti     = ti.to(device)
            logits = model(mi, ti)
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
            cache=not args.no_cache)

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
    tr.add_argument("--test",       nargs="+", default=None, metavar="FILE")
    tr.add_argument("--out",        default="checkpoints/cnn_mirbind.pt")
    tr.add_argument("--mre-col",    default="mre_sequence",   dest="mre_col")
    tr.add_argument("--mirna-col",  default="mirna_sequence", dest="mirna_col")
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
    tr.add_argument("--seq-pool", choices=["avg", "gem", "attention"], default="gem",
                    dest="seq_pool",
                    help="Global pooling for the 2D sequence branch: average, "
                         "GeM (learnable power-mean), or multi-head content-based "
                         "attention pooling. Attention can aggregate disjoint "
                         "paired regions (e.g. seed + 3′ supplementary), but needs "
                         "spatial resolution — pair it with fewer --n-pool-blocks.")
    tr.add_argument("--pool-heads", type=int, default=1, dest="pool_heads",
                    help="Attention heads when --seq-pool attention (ignored "
                         "otherwise). Each head can specialise to a different "
                         "paired region; pooled width becomes seq_filters * heads.")
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
                         "Compatible with --balance and pos_weight.")
    tr.add_argument("--balance",      action="store_true")
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
    pe.add_argument("--no-cache",  action="store_true", dest="no_cache",
                    help="Disable the preprocessing .cnncache.npz sidecar files.")
    pe.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict-ensemble":
        cmd_predict_ensemble(args)
    else:
        cmd_predict(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
