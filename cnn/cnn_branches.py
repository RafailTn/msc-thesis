#!/usr/bin/env python3
"""
Three-branch CNN for miRNA–MRE interaction classification.

Architecture
------------

Branch 1 – Sequence
    One-hot encode miRNA (padded to max_mirna_len, default 30 nt) and MRE
    (padded to mre_len, default 50 nt) then concatenate along the length
    dimension. A 5th binary channel marks which positions belong to the
    miRNA (1) vs MRE (0) so the model knows the boundary. Processed through
    a dilated residual CNN tower with attention pooling.

Branch 2 – Conservation
    PhastCons per-base scores (mre_len values in [0,1]) processed through a
    smaller dilated CNN tower with attention pooling.

Branch 3 – eCLIP
    Per-base AGO2-eCLIP probability vector (mre_len values) predicted by the
    TwoComponentEclip model from mirna_eqtl
    (src/predict_two_component.py, "predicted_probs" column). Same CNN
    architecture as Branch 2 with attention pooling.

Branch 4 – IntaRNA spot probability
    Per-base interaction probability vector (mre_len values) from IntaRNA
    ensemble mode. For each target position i, the probability that i is
    covered by any interaction in the partition-function ensemble: Zi / Z.
    Generate with ``--model=P --out=tSpotProb:FILE`` added to the existing
    ensemble IntaRNA call — no extra computation beyond what is already paid
    for Eall / P_E. Same CNN architecture as Branches 2 and 3.

Energy gate
    Two IntaRNA scalars (Eall, P_E) are processed by a small MLP that
    produces a gating vector applied multiplicatively to the concatenated
    branch embeddings before the final binary classifier. Eall encodes
    absolute binding-energy magnitude and P_E the overall interaction
    probability — information the spatial tSpotProb vector does not directly
    express.

Expected CSV columns
--------------------
Required:
    mre_sequence       – nucleotide string (≤50 nt; longer sequences are
                         center-cropped to mre_len)
    mirna_sequence     – nucleotide string (≤30 nt; right-padded with N)
    label              – 0 or 1  (omit for inference)

At least one of the vector columns (zero-filled when missing):
    conservation_vector  – comma-separated floats (mre_len values, phastCons)
    eclip_probs          – comma-separated floats (mre_len values)
                           Produce via mirna_eqtl/src/predict_two_component.py

Vector branch 4 (zero-filled when absent):
    tspot_probs          – comma-separated floats (mre_len values)
                           IntaRNA ensemble run with --out=tSpotProb:FILE

Energy scalars (all optional, zero-filled when absent):
    Eall, P_E

Usage
-----
  # Train
  python cnn_branches.py train \\
      --train data/train.csv --val data/val.csv \\
      --out checkpoints/cnn_branches.pt \\
      --epochs 40 --batch-size 64

  # Predict
  python cnn_branches.py predict \\
      --checkpoint checkpoints/cnn_branches.pt \\
      --input data/test.csv --output predictions.tsv \\
      --threshold 0.5
"""

from __future__ import annotations

import argparse
import math
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

MRE_LEN      = 50   # fixed MRE window length
MAX_MIRNA    = 30   # miRNA padded to this length
SEQ_LEN      = MRE_LEN + MAX_MIRNA   # 80 — full sequence branch input

ENERGY_COLS = ["Eall", "P_E"]
N_ENERGY = len(ENERGY_COLS)

# One-hot: A=0, C=1, G=2, T/U=3; N/other → all-zero row
_BASE_IDX = np.full(256, -1, dtype=np.int8)
for ch, i in (("A",0),("C",1),("G",2),("T",3),("U",3),
              ("a",0),("c",1),("g",2),("t",3),("u",3)):
    _BASE_IDX[ord(ch)] = i


def _one_hot(seq: str, length: int) -> np.ndarray:
    """One-hot encode *seq*, right-padded / right-cropped to *length*.

    Returns shape (4, length) float32.  Unknown bases → all-zero column.
    """
    seq = seq.upper().replace("T", "U")
    if len(seq) > length:
        seq = seq[:length]
    arr = np.frombuffer((seq + "N" * (length - len(seq))).encode("ascii"),
                        dtype=np.uint8)
    idx = _BASE_IDX[arr]
    out = np.zeros((4, length), dtype=np.float32)
    valid = idx >= 0
    out[idx[valid], np.nonzero(valid)[0]] = 1.0
    return out


def _read_table(path: str | Path) -> pd.DataFrame:
    sep = "\t" if str(path).endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)


def _parse_vector(raw, length: int) -> np.ndarray:
    """Parse a comma-separated float string (or list/ndarray) into a fixed-size
    float32 array of shape (length,).  Pads with 0 or crops as needed."""
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
    n = min(len(vals), length)
    out[:n] = vals[:n]
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MiRNAInteractionDataset(Dataset):
    """Load miRNA–MRE interaction data from a CSV for the three-branch CNN.

    Parameters
    ----------
    path : str or Path
        Input CSV file.
    energy_stats : dict or None
        Dict with keys ``mean`` and ``std`` (each a 1-D ndarray of length
        N_ENERGY). When None, statistics are computed from this split and
        stored in ``self.energy_stats`` for later serialisation.
    has_labels : bool
        Whether the CSV contains a ``label`` column.
    """

    def __init__(
        self,
        path: str | Path,
        energy_stats: Optional[dict] = None,
        has_labels: bool = True,
        mre_col: str = "mre_sequence",
        mirna_col: str = "mirna_sequence",
    ) -> None:
        self._init(_read_table(path), energy_stats, has_labels, mre_col, mirna_col)

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        energy_stats: Optional[dict] = None,
        has_labels: bool = True,
        mre_col: str = "mre_sequence",
        mirna_col: str = "mirna_sequence",
    ) -> "MiRNAInteractionDataset":
        obj = cls.__new__(cls)
        obj._init(df, energy_stats, has_labels, mre_col, mirna_col)
        return obj

    def _init(
        self,
        df: pd.DataFrame,
        energy_stats: Optional[dict],
        has_labels: bool,
        mre_col: str = "mre_sequence",
        mirna_col: str = "mirna_sequence",
    ) -> None:
        self.has_labels = has_labels

        # Sequences
        self.mre_seqs    = df[mre_col].astype(str).tolist()
        self.mirna_seqs  = df[mirna_col].astype(str).tolist()

        # Conservation vector
        if "conservation_vector" in df.columns:
            self.cons_vecs = df["conservation_vector"].tolist()
        else:
            self.cons_vecs = [None] * len(df)

        # eCLIP probability vector
        if "eclip_probs" in df.columns:
            self.eclip_vecs = df["eclip_probs"].tolist()
        else:
            self.eclip_vecs = [None] * len(df)

        # IntaRNA tSpotProb vector
        if "tspot_probs" in df.columns:
            self.tspot_vecs = df["tspot_probs"].tolist()
        else:
            self.tspot_vecs = [None] * len(df)

        # Energy features
        energy_mat = np.zeros((len(df), N_ENERGY), dtype=np.float32)
        for j, col in enumerate(ENERGY_COLS):
            if col in df.columns:
                energy_mat[:, j] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
        self.energy_raw = energy_mat

        # Normalise energy features
        if energy_stats is None:
            mean = self.energy_raw.mean(axis=0)
            std  = self.energy_raw.std(axis=0)
            std[std < 1e-8] = 1.0
            self.energy_stats = {"mean": mean, "std": std}
        else:
            self.energy_stats = energy_stats
        self.energy = (self.energy_raw - self.energy_stats["mean"]) / self.energy_stats["std"]

        # Labels
        if has_labels and "label" in df.columns:
            self.labels = df["label"].astype(int).values
        else:
            self.labels = np.zeros(len(df), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.mre_seqs)

    def __getitem__(self, idx: int):
        # --- Sequence branch (5 × SEQ_LEN) -----------------------------------
        mirna_oh  = _one_hot(self.mirna_seqs[idx], MAX_MIRNA)   # (4, 30)
        mre_oh    = _one_hot(self.mre_seqs[idx],   MRE_LEN)     # (4, 50)
        seq_oh    = np.concatenate([mirna_oh, mre_oh], axis=1)  # (4, 80)

        # Segment indicator channel: 1 = miRNA, 0 = MRE
        seg = np.zeros((1, SEQ_LEN), dtype=np.float32)
        seg[0, :MAX_MIRNA] = 1.0
        seq_input = np.concatenate([seq_oh, seg], axis=0)       # (5, 80)

        # --- Conservation branch (1 × MRE_LEN) --------------------------------
        cons  = _parse_vector(self.cons_vecs[idx],  MRE_LEN)[None, :]  # (1, 50)

        # --- eCLIP branch (1 × MRE_LEN) ----------------------------------------
        eclip = _parse_vector(self.eclip_vecs[idx], MRE_LEN)[None, :]  # (1, 50)

        # --- tSpotProb branch (1 × MRE_LEN) ------------------------------------
        tspot = _parse_vector(self.tspot_vecs[idx], MRE_LEN)[None, :]  # (1, 50)

        # --- Energy (N_ENERGY,) ------------------------------------------------
        energy = self.energy[idx].astype(np.float32)

        return (
            torch.from_numpy(seq_input),
            torch.from_numpy(cons),
            torch.from_numpy(eclip),
            torch.from_numpy(tspot),
            torch.from_numpy(energy),
            int(self.labels[idx]),
        )


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------

class _LayerNorm1d(nn.Module):
    """LayerNorm for (B, C, L) conv tensors — normalises over the channel dim."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class DilatedResBlock(nn.Module):
    """Pre-activation dilated residual block (CPU/GPU compatible)."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3,
                 dropout: float = 0.1, norm: str = "batch") -> None:
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        norm_cls = _LayerNorm1d if norm == "layer" else nn.BatchNorm1d
        self.net = nn.Sequential(
            norm_cls(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            norm_cls(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class AttentionPool(nn.Module):
    """Soft attention pooling: (B, C, L) → (B, C)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Conv1d(channels, 1, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        w = self.score(h).squeeze(1).softmax(dim=-1)   # (B, L)
        return (h * w.unsqueeze(1)).sum(dim=-1)         # (B, C)


class CrossAttention(nn.Module):
    """Bidirectional cross-attention between miRNA and MRE feature sequences.

    MRE positions attend to miRNA positions and vice versa; both results are
    returned with residual connections and layer normalisation.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.mre_to_mirna = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.mirna_to_mre = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_mre   = nn.LayerNorm(dim)
        self.norm_mirna = nn.LayerNorm(dim)

    def forward(
        self,
        h_mirna: torch.Tensor,   # (B, C, MAX_MIRNA)
        h_mre:   torch.Tensor,   # (B, C, MRE_LEN)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # MultiheadAttention expects (B, L, C)
        mirna_t = h_mirna.transpose(1, 2)   # (B, 30, C)
        mre_t   = h_mre.transpose(1, 2)     # (B, 50, C)

        mre_ctx,   _ = self.mre_to_mirna(mre_t,   mirna_t, mirna_t)
        mirna_ctx, _ = self.mirna_to_mre(mirna_t, mre_t,   mre_t)

        mre_out   = self.norm_mre  (mre_t   + mre_ctx  ).transpose(1, 2)  # (B, C, 50)
        mirna_out = self.norm_mirna(mirna_t + mirna_ctx ).transpose(1, 2)  # (B, C, 30)
        return mirna_out, mre_out


# ---------------------------------------------------------------------------
# Three-branch CNN
# ---------------------------------------------------------------------------

class ThreeBranchCNN(nn.Module):
    """Four-branch CNN with an IntaRNA energy gate for miRNA target classification.

    Branches
    --------
    1. Sequence   — dilated CNN on one-hot miRNA ++ MRE (5 × SEQ_LEN)
    2. Conservation — dilated CNN on phastCons vector (1 × MRE_LEN)
    3. eCLIP      — dilated CNN on AGO2-eCLIP probability vector (1 × MRE_LEN)
    4. tSpotProb  — dilated CNN on IntaRNA per-base interaction probability (1 × MRE_LEN)

    Branches 2–4 share weights in the projection layer (``vec_proj``) but
    have independent stems, towers, and attention pools.

    Parameters
    ----------
    seq_channels : int
        Channel width of the sequence branch CNN tower.
    seq_blocks : int
        Number of dilated residual blocks in the sequence branch.
    vec_channels : int
        Channel width of the three vector branches (shared).
    vec_blocks : int
        Number of dilated residual blocks in each vector branch.
    energy_dim : int
        Hidden size of the energy MLP (produces both the embedding and gate).
    kernel_size : int
        Stem convolution kernel size for all branches.
    dropout : float
        Dropout rate inside residual blocks and the classifier head.
    norm : str
        "batch" or "layer" normalisation.
    """

    def __init__(
        self,
        seq_channels:     int   = 128,
        seq_blocks:       int   = 6,
        vec_channels:     int   = 64,
        vec_blocks:       int   = 3,
        energy_dim:       int   = 64,
        kernel_size:      int   = 7,
        dropout:          float = 0.15,
        norm:             str   = "batch",
        use_conservation: bool  = True,
        use_eclip:        bool  = True,
        use_tspot:        bool  = True,
        use_energy:       bool  = True,
        num_heads:        int   = 4,
    ) -> None:
        super().__init__()
        self.use_conservation = use_conservation
        self.use_eclip  = use_eclip
        self.use_tspot  = use_tspot
        self.use_energy = use_energy

        if seq_channels % num_heads != 0:
            raise ValueError(
                f"seq_channels ({seq_channels}) must be divisible by "
                f"num_heads ({num_heads}).")

        # ── Branch 1: sequence — separate miRNA / MRE stems + cross-attention ──
        # Separate stems learn molecule-specific local patterns; the tower is
        # shared (same RNA alphabet, keeps parameter count in check).
        self.mirna_stem   = nn.Conv1d(4, seq_channels, kernel_size, padding="same")
        self.mre_seq_stem = nn.Conv1d(4, seq_channels, kernel_size, padding="same")
        self.seq_tower = nn.ModuleList([
            DilatedResBlock(seq_channels, dilation=2**i,
                            kernel_size=3, dropout=dropout, norm=norm)
            for i in range(seq_blocks)
        ])
        # Learned positional embeddings — one vector per position
        self.mirna_pe = nn.Embedding(MAX_MIRNA, seq_channels)
        self.mre_pe   = nn.Embedding(MRE_LEN,   seq_channels)

        self.cross_attn   = CrossAttention(seq_channels, num_heads=num_heads,
                                           dropout=dropout)
        self.mirna_pool   = AttentionPool(seq_channels)
        self.mre_seq_pool = AttentionPool(seq_channels)
        # Two pooled vectors → project back to seq_channels
        self.seq_proj = nn.Sequential(
            nn.LayerNorm(seq_channels * 2),
            nn.GELU(),
            nn.Linear(seq_channels * 2, seq_channels),
        )

        # ── Branch 2: conservation vector (1-channel, optional) ───────────────
        if use_conservation:
            self.cons_stem = nn.Conv1d(1, vec_channels, kernel_size, padding="same")
            self.cons_tower = nn.ModuleList([
                DilatedResBlock(vec_channels, dilation=2**i,
                                kernel_size=3, dropout=dropout, norm=norm)
                for i in range(vec_blocks)
            ])
            self.cons_pool = AttentionPool(vec_channels)

        # ── Branch 3: eCLIP vector (1-channel, optional) ──────────────────────
        if use_eclip:
            self.eclip_stem = nn.Conv1d(1, vec_channels, kernel_size, padding="same")
            self.eclip_tower = nn.ModuleList([
                DilatedResBlock(vec_channels, dilation=2**i,
                                kernel_size=3, dropout=dropout, norm=norm)
                for i in range(vec_blocks)
            ])
            self.eclip_pool = AttentionPool(vec_channels)

        # ── Branch 4: IntaRNA tSpotProb vector (1-channel, optional) ─────────
        if use_tspot:
            self.tspot_stem = nn.Conv1d(1, vec_channels, kernel_size, padding="same")
            self.tspot_tower = nn.ModuleList([
                DilatedResBlock(vec_channels, dilation=2**i,
                                kernel_size=3, dropout=dropout, norm=norm)
                for i in range(vec_blocks)
            ])
            self.tspot_pool = AttentionPool(vec_channels)

        # Shared projection for all active vector branches after attention pooling
        n_vec = sum([use_conservation, use_eclip, use_tspot])
        if n_vec > 0:
            self.vec_proj = nn.Sequential(
                nn.LayerNorm(vec_channels),
                nn.GELU(),
                nn.Linear(vec_channels, vec_channels),
            )

        combined_dim = seq_channels + n_vec * vec_channels

        # ── Energy MLP (gate + embedding, optional) ───────────────────────────
        if use_energy:
            self.energy_embed = nn.Sequential(
                nn.Linear(N_ENERGY, energy_dim),
                nn.LayerNorm(energy_dim),
                nn.GELU(),
                nn.Linear(energy_dim, energy_dim),
                nn.GELU(),
            )
            # Gate projects the energy embedding to the combined_dim size
            self.energy_gate = nn.Sequential(
                nn.Linear(energy_dim, combined_dim),
                nn.Sigmoid(),
            )
            fused_dim = combined_dim + energy_dim
        else:
            fused_dim = combined_dim

        # ── Classifier ────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, 1),
        )

    def _run_tower(self, x: torch.Tensor, tower: nn.ModuleList) -> torch.Tensor:
        for block in tower:
            x = block(x)
        return x

    def forward(
        self,
        seq:    torch.Tensor,   # (B, 5, SEQ_LEN)
        cons:   torch.Tensor,   # (B, 1, MRE_LEN)
        eclip:  torch.Tensor,   # (B, 1, MRE_LEN)
        tspot:  torch.Tensor,   # (B, 1, MRE_LEN)
        energy: torch.Tensor,   # (B, N_ENERGY)
    ) -> torch.Tensor:          # (B,) logits

        # ── Branch 1: sequence (cross-attention between miRNA and MRE) ──────────
        mirna_oh = seq[:, :4, :MAX_MIRNA]    # (B, 4, 30)
        mre_oh   = seq[:, :4, MAX_MIRNA:]    # (B, 4, 50)

        h_mirna = self._run_tower(self.mirna_stem(mirna_oh),   self.seq_tower)
        h_mre_s = self._run_tower(self.mre_seq_stem(mre_oh),   self.seq_tower)

        mirna_pos = torch.arange(MAX_MIRNA, device=seq.device)
        mre_pos   = torch.arange(MRE_LEN,   device=seq.device)
        h_mirna = h_mirna + self.mirna_pe(mirna_pos).T.unsqueeze(0)  # (1, C, 30)
        h_mre_s = h_mre_s + self.mre_pe(mre_pos).T.unsqueeze(0)     # (1, C, 50)

        h_mirna_ctx, h_mre_ctx = self.cross_attn(h_mirna, h_mre_s)

        h_seq = self.seq_proj(torch.cat([
            self.mirna_pool(h_mirna_ctx),    # (B, seq_channels)
            self.mre_seq_pool(h_mre_ctx),    # (B, seq_channels)
        ], dim=1))                           # (B, seq_channels)

        # ── Branch 2: conservation (optional) ────────────────────────────────
        parts = [h_seq]
        if self.use_conservation:
            h_cons = self.cons_stem(cons)
            h_cons = self._run_tower(h_cons, self.cons_tower)
            parts.append(self.vec_proj(self.cons_pool(h_cons)))

        # ── Branch 3: eCLIP (optional) ────────────────────────────────────────
        if self.use_eclip:
            h_eclip = self.eclip_stem(eclip)
            h_eclip = self._run_tower(h_eclip, self.eclip_tower)
            parts.append(self.vec_proj(self.eclip_pool(h_eclip)))

        # ── Branch 4: tSpotProb (optional) ───────────────────────────────────
        if self.use_tspot:
            h_tspot = self.tspot_stem(tspot)
            h_tspot = self._run_tower(h_tspot, self.tspot_tower)
            parts.append(self.vec_proj(self.tspot_pool(h_tspot)))

        combined = torch.cat(parts, dim=1)                    # (B, combined_dim)

        # ── Energy gate (optional) ────────────────────────────────────────────
        if self.use_energy:
            e_emb = self.energy_embed(energy)                 # (B, energy_dim)
            gate  = self.energy_gate(e_emb)                   # (B, combined_dim)
            fused = torch.cat([combined * gate, e_emb], dim=1)
        else:
            fused = combined

        return self.classifier(fused).squeeze(-1)             # (B,)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _binary_metrics(logits: np.ndarray, labels: np.ndarray,
                    threshold: float = 0.5) -> dict:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy  = (tp + tn) / max(tp + fp + fn + tn, 1)

    out = dict(accuracy=accuracy, precision=precision,
               recall=recall, f1=f1)
    if HAS_SKLEARN and len(np.unique(labels)) == 2:
        out["auroc"] = roc_auc_score(labels, probs)
        out["auprc"] = average_precision_score(labels, probs)
    return out


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def _make_loader(dataset: MiRNAInteractionDataset,
                 batch_size: int,
                 shuffle: bool,
                 num_workers: int,
                 balance: bool = False) -> DataLoader:
    sampler = None
    if balance and shuffle and dataset.has_labels:
        labels = dataset.labels
        counts = np.bincount(labels)
        weights = 1.0 / counts[labels]
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights).double(),
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(shuffle and sampler is None),
    )


@torch.no_grad()
def evaluate(model: ThreeBranchCNN, loader: DataLoader,
             device: torch.device, pos_weight: Optional[torch.Tensor] = None
             ) -> dict:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0
    n_batches  = 0

    for seq, cons, eclip, tspot, energy, labels in loader:
        seq    = seq.to(device,    non_blocking=True)
        cons   = cons.to(device,   non_blocking=True)
        eclip  = eclip.to(device,  non_blocking=True)
        tspot  = tspot.to(device,  non_blocking=True)
        energy = energy.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        logits = model(seq, cons, eclip, tspot, energy)
        loss   = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight)

        total_loss += loss.item()
        n_batches  += 1
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels).astype(int)
    metrics = _binary_metrics(logits_np, labels_np)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


# ---------------------------------------------------------------------------
# Train subcommand helpers
# ---------------------------------------------------------------------------

def _model_args_from_cli(args: argparse.Namespace) -> dict:
    return {
        "seq_channels":     args.seq_channels,
        "seq_blocks":       args.seq_blocks,
        "vec_channels":     args.vec_channels,
        "vec_blocks":       args.vec_blocks,
        "energy_dim":       args.energy_dim,
        "kernel_size":      args.kernel_size,
        "dropout":          args.dropout,
        "norm":             args.norm,
        "use_conservation": not args.no_conservation,
        "use_eclip":        not args.no_eclip,
        "use_tspot":        not args.no_tspot,
        "use_energy":       not args.no_energy,
        "num_heads":        args.num_heads,
    }


def _train_one_run(
    model: ThreeBranchCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_ds: MiRNAInteractionDataset,
    model_args: dict,
    args: argparse.Namespace,
    device: torch.device,
    out_path: Path,
) -> float:
    """Run the training loop; save best checkpoint to out_path.

    Returns the best value of args.checkpoint_metric.
    """
    pos_weight: Optional[torch.Tensor] = None
    if not args.balance:
        n_pos = int(train_ds.labels.sum())
        n_neg = len(train_ds.labels) - n_pos
        if n_pos > 0 and n_neg > 0:
            pw = n_neg / n_pos
            pos_weight = torch.tensor([pw], device=device)
            print(f"  BCEWithLogitsLoss pos_weight = {pw:.3f}")

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

    best_val = -float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        seen = 0
        train_logits_buf: list[np.ndarray] = []
        train_labels_buf: list[np.ndarray] = []

        for seq, cons, eclip, tspot, energy, labels in train_loader:
            seq    = seq.to(device,    non_blocking=True)
            cons   = cons.to(device,   non_blocking=True)
            eclip  = eclip.to(device,  non_blocking=True)
            tspot  = tspot.to(device,  non_blocking=True)
            energy = energy.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            logits = model(seq, cons, eclip, tspot, energy)
            loss   = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            running_loss += loss.item() * seq.size(0)
            seen += seq.size(0)
            train_logits_buf.append(logits.detach().cpu().numpy())
            train_labels_buf.append(labels.cpu().numpy().astype(int))

        train_loss = running_loss / max(seen, 1)
        train_metrics = _binary_metrics(
            np.concatenate(train_logits_buf),
            np.concatenate(train_labels_buf),
        )
        val_metrics = evaluate(model, val_loader, device, pos_weight)
        dt = time.time() - t0

        ckpt_val = val_metrics.get(args.checkpoint_metric, -val_metrics["loss"])
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
        log += f"  ({dt:.1f}s)" + (" *" if improved else "")
        print(log)

        if HAS_WANDB and wandb.run is not None:
            log_dict: dict = {
                "epoch":           epoch,
                "lr":              sched.get_last_lr()[0],
                "train/loss":      train_loss,
                "train/f1":        train_metrics["f1"],
                "train/accuracy":  train_metrics["accuracy"],
                "train/precision": train_metrics["precision"],
                "train/recall":    train_metrics["recall"],
                "val/loss":        val_metrics["loss"],
                "val/f1":          val_metrics["f1"],
                "val/accuracy":    val_metrics["accuracy"],
                "val/precision":   val_metrics["precision"],
                "val/recall":      val_metrics["recall"],
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
            torch.save({
                "model_state":  model.state_dict(),
                "model_args":   model_args,
                "energy_stats": train_ds.energy_stats,
                "val_metrics":  val_metrics,
                "epoch":        epoch,
            }, out_path)
            print(f"  → checkpoint saved: {out_path}")
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"Early stopping after {patience_counter} epochs without improvement.")
                break

    print(f"\nBest {args.checkpoint_metric} = {best_val:.4f}")
    if HAS_WANDB and wandb.run is not None:
        wandb.run.summary[f"best_{args.checkpoint_metric}"] = best_val
    return best_val


def _run_single(args: argparse.Namespace, device: torch.device) -> None:
    if not args.val:
        sys.exit("ERROR: --val is required when --folds is not set.")

    print("Loading training data ...")
    train_ds = MiRNAInteractionDataset(
        args.train, energy_stats=None, has_labels=True,
        mre_col=args.mre_col, mirna_col=args.mirna_col)
    print(f"  train samples : {len(train_ds)}")
    print(f"  positives     : {int(train_ds.labels.sum())} / {len(train_ds.labels)}")

    print("Loading validation data ...")
    val_ds = MiRNAInteractionDataset(
        args.val, energy_stats=train_ds.energy_stats, has_labels=True,
        mre_col=args.mre_col, mirna_col=args.mirna_col)
    print(f"  val samples   : {len(val_ds)}")

    train_loader = _make_loader(train_ds, args.batch_size, shuffle=True,
                                num_workers=args.num_workers, balance=args.balance)
    val_loader   = _make_loader(val_ds,   args.batch_size, shuffle=False,
                                num_workers=args.num_workers)

    model_args = _model_args_from_cli(args)
    model = ThreeBranchCNN(**model_args).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    if HAS_WANDB and getattr(args, "wandb_project", None):
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or None,
            group=args.wandb_group or None,
            config={
                **model_args,
                "epochs": args.epochs, "batch_size": args.batch_size,
                "lr": args.lr, "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps, "patience": args.patience,
                "balance": args.balance, "checkpoint_metric": args.checkpoint_metric,
            },
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _train_one_run(model, train_loader, val_loader, train_ds,
                   model_args, args, device, out_path)

    if HAS_WANDB and wandb.run is not None:
        wandb.finish()


def _run_kfold(args: argparse.Namespace, device: torch.device) -> None:
    if not HAS_SKLEARN:
        sys.exit("ERROR: scikit-learn is required for --folds. "
                 "Install with: pip install scikit-learn")

    print(f"Loading data for {args.folds}-fold stratified group cross-validation ...")
    df = _read_table(args.train)
    print(f"  {len(df)} rows")

    if args.family_col not in df.columns:
        sys.exit(f"ERROR: --family-col '{args.family_col}' not found in input. "
                 f"Available columns: {list(df.columns)}")

    groups = df[args.family_col].fillna("unknown").astype(str).values
    n_families = len(set(groups))
    print(f"  {n_families} unique miRNA families → {args.folds} folds")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_args = _model_args_from_cli(args)
    gkf = StratifiedGroupKFold(n_splits=args.folds)
    fold_scores: list[float] = []
    test_paths = args.test or []
    # keyed by test file stem; each value is a list of metric dicts (one per fold)
    fold_test_metrics: dict[str, list[dict]] = {
        Path(p).stem: [] for p in test_paths
    }

    for fold, (train_idx, val_idx) in enumerate(
            gkf.split(df, y=df["label"].values, groups=groups), 1):
        val_families = sorted(set(groups[val_idx]))
        print(f"\n{'='*60}")
        print(f"Fold {fold}/{args.folds}  "
              f"train={len(train_idx)}  val={len(val_idx)}")
        preview = val_families[:8]
        suffix  = " ..." if len(val_families) > 8 else ""
        print(f"  val families ({len(val_families)}): {preview}{suffix}")
        print(f"{'='*60}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)

        train_ds = MiRNAInteractionDataset.from_df(
            train_df, energy_stats=None, has_labels=True,
            mre_col=args.mre_col, mirna_col=args.mirna_col)
        val_ds   = MiRNAInteractionDataset.from_df(
            val_df, energy_stats=train_ds.energy_stats, has_labels=True,
            mre_col=args.mre_col, mirna_col=args.mirna_col)

        print(f"  train positives: {int(train_ds.labels.sum())} / {len(train_ds.labels)}")
        print(f"  val   positives: {int(val_ds.labels.sum())} / {len(val_ds.labels)}")

        train_loader = _make_loader(train_ds, args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, balance=args.balance)
        val_loader   = _make_loader(val_ds,   args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)

        model = ThreeBranchCNN(**model_args).to(device)
        if fold == 1:
            print(f"Model parameters: "
                  f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

        if HAS_WANDB and getattr(args, "wandb_project", None):
            _group = args.wandb_group or args.wandb_run_name or out_path.stem
            _name  = (f"{args.wandb_run_name}_fold{fold}"
                      if args.wandb_run_name else f"{out_path.stem}_fold{fold}")
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                name=_name,
                group=_group,
                config={
                    **model_args, "fold": fold, "n_folds": args.folds,
                    "epochs": args.epochs, "batch_size": args.batch_size,
                    "lr": args.lr, "weight_decay": args.weight_decay,
                    "warmup_steps": args.warmup_steps, "patience": args.patience,
                    "balance": args.balance, "checkpoint_metric": args.checkpoint_metric,
                },
            )

        fold_out = out_path.parent / f"{out_path.stem}_fold{fold}{out_path.suffix}"
        score = _train_one_run(model, train_loader, val_loader, train_ds,
                               model_args, args, device, fold_out)
        fold_scores.append(score)

        # ── Evaluate on external test sets using the best checkpoint ──────────
        if test_paths:
            best_ckpt = torch.load(fold_out, map_location=device,
                                   weights_only=False)
            model.load_state_dict(best_ckpt["model_state"])
            print(f"\n  Test-set evaluation (fold {fold} best checkpoint):")
            for test_path in test_paths:
                test_name = Path(test_path).stem
                test_ds = MiRNAInteractionDataset.from_df(
                    _read_table(test_path),
                    energy_stats=train_ds.energy_stats,
                    has_labels=True,
                    mre_col=args.mre_col,
                    mirna_col=args.mirna_col,
                )
                test_loader = _make_loader(
                    test_ds, args.batch_size, shuffle=False,
                    num_workers=args.num_workers)
                metrics = evaluate(model, test_loader, device)
                fold_test_metrics[test_name].append(metrics)
                row = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"    [{test_name}]  {row}")
                if HAS_WANDB and wandb.run is not None:
                    for k, v in metrics.items():
                        wandb.run.summary[f"test_{test_name}/{k}"] = v

        if HAS_WANDB and wandb.run is not None:
            wandb.finish()

    # ── Cross-fold summary ────────────────────────────────────────────────────
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
                vals = [m[metric_key] for m in metrics_list]
                per_fold = "  ".join(f"{v:.4f}" for v in vals)
                print(f"    {metric_key:<20s} folds=[{per_fold}]  "
                      f"mean={np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Train subcommand
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    print(f"Device: {device}")
    if args.folds:
        _run_kfold(args, device)
    else:
        _run_single(args, device)


# ---------------------------------------------------------------------------
# Predict subcommand
# ---------------------------------------------------------------------------

def cmd_predict(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    margs = ckpt["model_args"]
    margs.setdefault("use_conservation", True)   # backward compat with old checkpoints
    margs.setdefault("use_eclip", True)
    margs.setdefault("use_tspot", True)
    margs.setdefault("use_energy", True)
    margs.setdefault("num_heads", 4)
    energy_stats = ckpt["energy_stats"]

    model = ThreeBranchCNN(**margs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch')}, "
          f"val_metrics={ckpt.get('val_metrics')})")

    has_labels = True   # will be corrected after loading
    ds = MiRNAInteractionDataset(
        args.input, energy_stats=energy_stats, has_labels=has_labels,
        mre_col=args.mre_col, mirna_col=args.mirna_col)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2)

    all_probs:  list[float] = []
    all_preds:  list[int]   = []
    all_labels: list[int]   = []

    with torch.no_grad():
        for seq, cons, eclip, tspot, energy, labels in loader:
            seq    = seq.to(device)
            cons   = cons.to(device)
            eclip  = eclip.to(device)
            tspot  = tspot.to(device)
            energy = energy.to(device)
            logits = model(seq, cons, eclip, tspot, energy)
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
        metrics = _binary_metrics(logits_np, np.array(all_labels), args.threshold)
        print("\nTest-set metrics:")
        for k, v in metrics.items():
            print(f"  {k:20s} = {v:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Three-branch CNN for miRNA–MRE interaction classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ─────────────────────────────────────────────────────────────────
    tr = sub.add_parser("train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    tr.add_argument("--train",      required=True,
                    help="Training CSV/TSV (full dataset when --folds is set)")
    tr.add_argument("--val",        default=None,
                    help="Validation CSV/TSV. Required unless --folds is set.")
    tr.add_argument("--folds",      type=int, default=None,
                    help="Number of StratifiedGroupKFold folds (uses --family-col as groups, "
                         "stratifies by label). Saves one checkpoint per fold.")
    tr.add_argument("--family-col", default="mirna_family", dest="family_col",
                    help="Column with miRNA family labels for StratifiedGroupKFold.")
    tr.add_argument("--test", nargs="+", default=None, metavar="FILE",
                    help="One or more test CSV/TSV files evaluated after each fold "
                         "(only used with --folds). Energy stats come from the "
                         "training fold, not the test file.")
    tr.add_argument("--out",        default="checkpoints/cnn_branches.pt")
    tr.add_argument("--mre-col",   default="mre_sequence", dest="mre_col",
                    help="Column name for the MRE nucleotide sequence.")
    tr.add_argument("--mirna-col", default="mirna_sequence", dest="mirna_col",
                    help="Column name for the miRNA nucleotide sequence.")
    # Architecture
    tr.add_argument("--seq-channels",  type=int,   default=128)
    tr.add_argument("--seq-blocks",    type=int,   default=6)
    tr.add_argument("--vec-channels",  type=int,   default=64)
    tr.add_argument("--vec-blocks",    type=int,   default=3)
    tr.add_argument("--energy-dim",    type=int,   default=64)
    tr.add_argument("--kernel-size",   type=int,   default=7)
    tr.add_argument("--dropout",       type=float, default=0.15)
    tr.add_argument("--norm", choices=["batch", "layer"], default="batch")
    tr.add_argument("--num-heads",     type=int,   default=4,  dest="num_heads",
                    help="Number of attention heads in cross-attention "
                         "(seq_channels must be divisible by this).")
    tr.add_argument("--no-conservation", action="store_true",
                    help="Disable the conservation branch.")
    tr.add_argument("--no-eclip", action="store_true",
                    help="Disable the eCLIP branch.")
    tr.add_argument("--no-tspot", action="store_true",
                    help="Disable the IntaRNA tSpotProb branch.")
    tr.add_argument("--no-energy", action="store_true",
                    help="Disable the IntaRNA energy gate (Eall / P_E scalars).")
    # Training
    tr.add_argument("--epochs",        type=int,   default=40)
    tr.add_argument("--batch-size",    type=int,   default=64)
    tr.add_argument("--lr",            type=float, default=1e-3)
    tr.add_argument("--weight-decay",  type=float, default=1e-4)
    tr.add_argument("--warmup-steps",  type=int,   default=200)
    tr.add_argument("--num-workers",   type=int,   default=2)
    tr.add_argument("--patience",      type=int,   default=10,
                    help="Early stopping patience. 0 = disabled.")
    tr.add_argument("--balance",       action="store_true",
                    help="Use weighted sampler to balance classes.")
    tr.add_argument("--checkpoint-metric",
                    choices=["auroc", "auprc", "f1", "accuracy"],
                    default="auprc",
                    help="Validation metric to use for checkpointing.")
    tr.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    # W&B
    tr.add_argument("--wandb-project",  default=None, dest="wandb_project",
                    help="W&B project name. W&B logging is disabled when not set.")
    tr.add_argument("--wandb-entity",   default=None, dest="wandb_entity",
                    help="W&B entity (user or team). Uses your W&B default if not set.")
    tr.add_argument("--wandb-run-name", default=None, dest="wandb_run_name",
                    help="W&B run display name. Auto-generated by W&B if not set.")
    tr.add_argument("--wandb-group",    default=None, dest="wandb_group",
                    help="W&B run group. For k-fold, folds are grouped automatically "
                         "under this name (falls back to --wandb-run-name or output stem).")

    # ── predict ───────────────────────────────────────────────────────────────
    pr = sub.add_parser("predict", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pr.add_argument("--checkpoint", required=True)
    pr.add_argument("--input",      required=True, help="Input CSV")
    pr.add_argument("--output",     required=True, help="Output TSV")
    pr.add_argument("--threshold",  type=float, default=0.5)
    pr.add_argument("--batch-size", type=int,   default=256)
    pr.add_argument("--mre-col",   default="mre_sequence", dest="mre_col",
                    help="Column name for the MRE nucleotide sequence.")
    pr.add_argument("--mirna-col", default="mirna_sequence", dest="mirna_col",
                    help="Column name for the miRNA nucleotide sequence.")
    pr.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    else:
        cmd_predict(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
