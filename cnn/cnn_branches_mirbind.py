#!/usr/bin/env python3
"""
Three-branch CNN for miRNA–MRE interaction classification.
Sequence branch uses miRBind-style 2D Watson-Crick complementarity matrix + 2D CNN.

Architecture
------------

Branch 1 – Sequence (miRBind-style)
    Build a 2D binary matrix M of shape (MAX_MIRNA × MRE_LEN) where M[i,j]=1 if
    miRNA position i and MRE position j can form a Watson–Crick (or G·U wobble)
    base pair, 0 otherwise.  A 2D CNN of 6 convolutional blocks (5×5 kernels,
    LeakyReLU, BatchNorm2d, optional MaxPool2d, Dropout) processes the matrix,
    followed by an AdaptiveAvgPool2d and two dense blocks to produce a fixed-size
    embedding.  This directly encodes base-pairing potential geometry, matching
    the core idea from:
        Klimentova et al. (2022). miRBind: A Deep Learning Method for miRNA
        Binding Classification. Genes, 13(12), 2323.

Branch 2 – Conservation
    PhastCons per-base scores (mre_len values in [0,1]) processed through a
    smaller dilated CNN tower with attention pooling.

Branch 3 – eCLIP
    Per-base AGO2-eCLIP probability vector processed by the same architecture.

Branch 4 – IntaRNA tSpotProb
    Per-base interaction probability vector from IntaRNA ensemble mode.

Energy gate
    Two IntaRNA scalars (Eall, P_E) gated multiplicatively onto the fused embedding.

Expected CSV columns
--------------------
Required:
    mre_sequence       – nucleotide string (≤50 nt)
    mirna_sequence     – nucleotide string (≤30 nt)
    label              – 0 or 1  (omit for inference)

Optional vector columns (zero-filled when missing):
    conservation_vector, eclip_probs, tspot_probs

Optional scalars (zero-filled when missing):
    Eall, P_E

Usage
-----
  python cnn_branches_mirbind.py train \\
      --train data/train.csv --val data/val.csv \\
      --out checkpoints/cnn_mirbind.pt --epochs 40

  python cnn_branches_mirbind.py predict \\
      --checkpoint checkpoints/cnn_mirbind.pt \\
      --input data/test.csv --output predictions.tsv
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

MRE_LEN   = 50
MAX_MIRNA = 30

ENERGY_COLS = ["Eall", "P_E"]
N_ENERGY    = len(ENERGY_COLS)

# Watson-Crick + G·U wobble complementarity (RNA)
_WC_PAIRS: set[tuple[str, str]] = {
    ("A", "U"), ("U", "A"),
    ("G", "C"), ("C", "G"),
    ("G", "U"), ("U", "G"),   # wobble
}


def _norm_seq(seq: str) -> str:
    return seq.upper().replace("T", "U")


def _wc_matrix(mirna_seq: str, mre_seq: str,
               mirna_len: int = MAX_MIRNA,
               mre_len: int = MRE_LEN) -> np.ndarray:
    """Binary Watson-Crick complementarity matrix.

    Returns shape (1, mirna_len, mre_len) float32.
    M[0, i, j] = 1 if mirna[i] can base-pair with mre[j], else 0.
    miRNA is cropped/padded to mirna_len; MRE to mre_len.
    """
    mirna = _norm_seq(mirna_seq)[:mirna_len]
    mre   = _norm_seq(mre_seq)[:mre_len]

    mat = np.zeros((mirna_len, mre_len), dtype=np.float32)
    for i, mb in enumerate(mirna):
        for j, tb in enumerate(mre):
            if (mb, tb) in _WC_PAIRS:
                mat[i, j] = 1.0
    return mat[np.newaxis]   # (1, mirna_len, mre_len)


def _read_table(path: str | Path) -> pd.DataFrame:
    sep = "\t" if str(path).endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)


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
# Dataset
# ---------------------------------------------------------------------------

class MiRNAInteractionDataset(Dataset):
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
        mre_col: str,
        mirna_col: str,
    ) -> None:
        self.has_labels  = has_labels
        self.mre_seqs    = df[mre_col].astype(str).tolist()
        self.mirna_seqs  = df[mirna_col].astype(str).tolist()

        self.cons_vecs   = df["conservation_vector"].tolist() if "conservation_vector" in df.columns else [None] * len(df)
        self.eclip_vecs  = df["eclip_probs"].tolist()         if "eclip_probs"         in df.columns else [None] * len(df)
        self.tspot_vecs  = df["tspot_probs"].tolist()         if "tspot_probs"         in df.columns else [None] * len(df)

        energy_mat = np.zeros((len(df), N_ENERGY), dtype=np.float32)
        for j, col in enumerate(ENERGY_COLS):
            if col in df.columns:
                energy_mat[:, j] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
        self.energy_raw = energy_mat

        if energy_stats is None:
            mean = self.energy_raw.mean(axis=0)
            std  = self.energy_raw.std(axis=0)
            std[std < 1e-8] = 1.0
            self.energy_stats = {"mean": mean, "std": std}
        else:
            self.energy_stats = energy_stats
        self.energy = (self.energy_raw - self.energy_stats["mean"]) / self.energy_stats["std"]

        if has_labels and "label" in df.columns:
            self.labels = df["label"].astype(int).values
        else:
            self.labels = np.zeros(len(df), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.mre_seqs)

    def __getitem__(self, idx: int):
        # --- Sequence branch: miRBind 2D WC-complementarity matrix ------------
        wc_mat = _wc_matrix(self.mirna_seqs[idx], self.mre_seqs[idx])  # (1, 30, 50)

        # --- Vector branches --------------------------------------------------
        cons  = _parse_vector(self.cons_vecs[idx],  MRE_LEN)[None, :]
        eclip = _parse_vector(self.eclip_vecs[idx], MRE_LEN)[None, :]
        tspot = _parse_vector(self.tspot_vecs[idx], MRE_LEN)[None, :]

        energy = self.energy[idx].astype(np.float32)

        return (
            torch.from_numpy(wc_mat),
            torch.from_numpy(cons),
            torch.from_numpy(eclip),
            torch.from_numpy(tspot),
            torch.from_numpy(energy),
            int(self.labels[idx]),
        )


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------

class Conv2dBlock(nn.Module):
    """One miRBind-style 2D convolutional block.

    Conv2d (5×5, same padding) → LeakyReLU → BatchNorm2d →
    [MaxPool2d(2,2)] → Dropout.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.3,
                 pool: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(out_ch),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseBlock(nn.Module):
    """One miRBind-style dense block: Linear → LeakyReLU → BatchNorm1d → Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MiRBindSeqBranch(nn.Module):
    """miRBind-style sequence branch.

    Takes the 2D WC-complementarity matrix (B, 1, MAX_MIRNA, MRE_LEN) and
    produces a fixed-size embedding of shape (B, out_dim).

    Architecture (mirroring Klimentova et al. 2022):
      - 6 Conv2d blocks (5×5, LeakyReLU, BN2d, Dropout=0.3)
        Blocks 0–3: include MaxPool2d(2,2) to downsample
        Blocks 4–5: no pooling (spatial dims too small by this point)
      - AdaptiveAvgPool2d((1,1)) → flatten
      - 2 dense blocks → out_dim
    """

    N_CONV_BLOCKS  = 6
    N_POOL_BLOCKS  = 4   # first 4 blocks have MaxPool2d

    def __init__(
        self,
        n_filters: int = 64,
        out_dim:   int = 128,
        dropout:   float = 0.3,
    ) -> None:
        super().__init__()

        conv_blocks: list[nn.Module] = []
        in_ch = 1
        for i in range(self.N_CONV_BLOCKS):
            pool = i < self.N_POOL_BLOCKS
            conv_blocks.append(Conv2dBlock(in_ch, n_filters, dropout=dropout, pool=pool))
            in_ch = n_filters
        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        hidden = max(n_filters * 2, out_dim)
        self.dense = nn.Sequential(
            DenseBlock(n_filters, hidden, dropout=dropout),
            DenseBlock(hidden,    out_dim, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, MAX_MIRNA, MRE_LEN)
        h = self.conv_blocks(x)                # (B, n_filters, H', W')
        h = self.global_pool(h).flatten(1)     # (B, n_filters)
        return self.dense(h)                   # (B, out_dim)


# ---------------------------------------------------------------------------
# 1D dilated CNN blocks for the vector branches (unchanged from original)
# ---------------------------------------------------------------------------

class _LayerNorm1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class DilatedResBlock(nn.Module):
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
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Conv1d(channels, 1, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        w = self.score(h).squeeze(1).softmax(dim=-1)
        return (h * w.unsqueeze(1)).sum(dim=-1)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MiRBindCNN(nn.Module):
    """Four-branch CNN with miRBind-style 2D sequence branch.

    Branch 1 — Sequence (miRBind)
        2D WC-complementarity matrix → 6 Conv2d blocks → dense → seq_dim embedding.

    Branches 2–4 — Conservation / eCLIP / tSpotProb
        Per-base 1D vectors → dilated residual CNN → attention pooling.

    Energy gate
        Two IntaRNA scalars (Eall, P_E) multiplicatively gate the fused embedding.

    Parameters
    ----------
    seq_filters : int
        Number of 2D conv filters in the miRBind sequence branch.
    seq_dim : int
        Output embedding size of the sequence branch (after dense layers).
    vec_channels : int
        Channel width of the three 1D vector branches.
    vec_blocks : int
        Dilated residual blocks per vector branch.
    energy_dim : int
        Hidden size of the energy MLP / gate.
    vec_kernel_size : int
        Stem kernel size for the vector branches.
    seq_dropout : float
        Dropout inside the miRBind 2D CNN blocks.
    vec_dropout : float
        Dropout inside the 1D dilated blocks.
    norm : str
        "batch" or "layer" for the 1D vector branches.
    """

    def __init__(
        self,
        seq_filters:      int   = 64,
        seq_dim:          int   = 128,
        vec_channels:     int   = 64,
        vec_blocks:       int   = 3,
        energy_dim:       int   = 64,
        vec_kernel_size:  int   = 7,
        seq_dropout:      float = 0.3,
        vec_dropout:      float = 0.15,
        norm:             str   = "batch",
        use_conservation: bool  = True,
        use_eclip:        bool  = True,
        use_tspot:        bool  = True,
        use_energy:       bool  = True,
    ) -> None:
        super().__init__()
        self.use_conservation = use_conservation
        self.use_eclip  = use_eclip
        self.use_tspot  = use_tspot
        self.use_energy = use_energy

        # ── Branch 1: miRBind 2D sequence branch ─────────────────────────────
        self.seq_branch = MiRBindSeqBranch(
            n_filters=seq_filters, out_dim=seq_dim, dropout=seq_dropout)

        # ── Branches 2–4: 1D dilated CNN vector branches ─────────────────────
        def _make_vec_branch():
            stem  = nn.Conv1d(1, vec_channels, vec_kernel_size, padding="same")
            tower = nn.ModuleList([
                DilatedResBlock(vec_channels, dilation=2**i,
                                kernel_size=3, dropout=vec_dropout, norm=norm)
                for i in range(vec_blocks)
            ])
            pool  = AttentionPool(vec_channels)
            return stem, tower, pool

        if use_conservation:
            self.cons_stem,  self.cons_tower,  self.cons_pool  = _make_vec_branch()
        if use_eclip:
            self.eclip_stem, self.eclip_tower, self.eclip_pool = _make_vec_branch()
        if use_tspot:
            self.tspot_stem, self.tspot_tower, self.tspot_pool = _make_vec_branch()

        n_vec = sum([use_conservation, use_eclip, use_tspot])
        if n_vec > 0:
            self.vec_proj = nn.Sequential(
                nn.LayerNorm(vec_channels),
                nn.GELU(),
                nn.Linear(vec_channels, vec_channels),
            )

        combined_dim = seq_dim + n_vec * vec_channels

        # ── Energy gate ───────────────────────────────────────────────────────
        if use_energy:
            self.energy_embed = nn.Sequential(
                nn.Linear(N_ENERGY, energy_dim),
                nn.LayerNorm(energy_dim),
                nn.GELU(),
                nn.Linear(energy_dim, energy_dim),
                nn.GELU(),
            )
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
            nn.Dropout(vec_dropout),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(vec_dropout),
            nn.Linear(fused_dim // 2, 1),
        )

    def _run_tower(self, x, tower):
        for block in tower:
            x = block(x)
        return x

    def forward(
        self,
        wc_mat: torch.Tensor,   # (B, 1, MAX_MIRNA, MRE_LEN)
        cons:   torch.Tensor,   # (B, 1, MRE_LEN)
        eclip:  torch.Tensor,   # (B, 1, MRE_LEN)
        tspot:  torch.Tensor,   # (B, 1, MRE_LEN)
        energy: torch.Tensor,   # (B, N_ENERGY)
    ) -> torch.Tensor:          # (B,) logits

        # ── Branch 1: miRBind 2D sequence branch ─────────────────────────────
        h_seq = self.seq_branch(wc_mat)         # (B, seq_dim)

        # ── Branches 2–4 ─────────────────────────────────────────────────────
        parts = [h_seq]

        if self.use_conservation:
            h = self._run_tower(self.cons_stem(cons), self.cons_tower)
            parts.append(self.vec_proj(self.cons_pool(h)))

        if self.use_eclip:
            h = self._run_tower(self.eclip_stem(eclip), self.eclip_tower)
            parts.append(self.vec_proj(self.eclip_pool(h)))

        if self.use_tspot:
            h = self._run_tower(self.tspot_stem(tspot), self.tspot_tower)
            parts.append(self.vec_proj(self.tspot_pool(h)))

        combined = torch.cat(parts, dim=1)      # (B, combined_dim)

        # ── Energy gate ───────────────────────────────────────────────────────
        if self.use_energy:
            e_emb = self.energy_embed(energy)
            gate  = self.energy_gate(e_emb)
            fused = torch.cat([combined * gate, e_emb], dim=1)
        else:
            fused = combined

        return self.classifier(fused).squeeze(-1)


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
                 balance: bool = False) -> DataLoader:
    sampler = None
    if balance and shuffle and dataset.has_labels:
        labels  = dataset.labels
        counts  = np.bincount(labels)
        weights = 1.0 / counts[labels]
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights).double(),
            num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        sampler=sampler, num_workers=num_workers,
        pin_memory=True, drop_last=(shuffle and sampler is None),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: MiRBindCNN, loader: DataLoader,
             device: torch.device,
             pos_weight: Optional[torch.Tensor] = None) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    total_loss, n_batches  = 0.0, 0

    for wc_mat, cons, eclip, tspot, energy, labels in loader:
        wc_mat = wc_mat.to(device, non_blocking=True)
        cons   = cons.to(device,   non_blocking=True)
        eclip  = eclip.to(device,  non_blocking=True)
        tspot  = tspot.to(device,  non_blocking=True)
        energy = energy.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        logits = model(wc_mat, cons, eclip, tspot, energy)
        loss   = F.binary_cross_entropy_with_logits(logits, labels,
                                                     pos_weight=pos_weight)
        total_loss += loss.item()
        n_batches  += 1
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels).astype(int)
    metrics   = _binary_metrics(logits_np, labels_np)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _model_args_from_cli(args: argparse.Namespace) -> dict:
    return {
        "seq_filters":      args.seq_filters,
        "seq_dim":          args.seq_dim,
        "vec_channels":     args.vec_channels,
        "vec_blocks":       args.vec_blocks,
        "energy_dim":       args.energy_dim,
        "vec_kernel_size":  args.vec_kernel_size,
        "seq_dropout":      args.seq_dropout,
        "vec_dropout":      args.vec_dropout,
        "norm":             args.norm,
        "use_conservation": not args.no_conservation,
        "use_eclip":        not args.no_eclip,
        "use_tspot":        not args.no_tspot,
        "use_energy":       not args.no_energy,
    }


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
        train_logits_buf, train_labels_buf = [], []

        for wc_mat, cons, eclip, tspot, energy, labels in train_loader:
            wc_mat = wc_mat.to(device, non_blocking=True)
            cons   = cons.to(device,   non_blocking=True)
            eclip  = eclip.to(device,  non_blocking=True)
            tspot  = tspot.to(device,  non_blocking=True)
            energy = energy.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            logits = model(wc_mat, cons, eclip, tspot, energy)
            loss   = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            running_loss += loss.item() * wc_mat.size(0)
            seen += wc_mat.size(0)
            train_logits_buf.append(logits.detach().cpu().numpy())
            train_labels_buf.append(labels.cpu().numpy().astype(int))

        train_loss    = running_loss / max(seen, 1)
        train_metrics = _binary_metrics(
            np.concatenate(train_logits_buf), np.concatenate(train_labels_buf))
        val_metrics   = evaluate(model, val_loader, device, pos_weight)
        dt            = time.time() - t0

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

    if args.family_col not in df.columns:
        sys.exit(f"ERROR: --family-col '{args.family_col}' not found. "
                 f"Available: {list(df.columns)}")

    groups = df[args.family_col].fillna("unknown").astype(str).values
    n_families = len(set(groups))
    print(f"  {n_families} unique miRNA families → {args.folds} folds")

    out_path   = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_args = _model_args_from_cli(args)
    gkf        = StratifiedGroupKFold(n_splits=args.folds)
    fold_scores: list[float] = []
    test_paths  = args.test or []
    fold_test_metrics: dict[str, list[dict]] = {Path(p).stem: [] for p in test_paths}

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
                    energy_stats=train_ds.energy_stats, has_labels=True,
                    mre_col=args.mre_col, mirna_col=args.mirna_col)
                test_loader = _make_loader(
                    test_ds, args.batch_size, shuffle=False,
                    num_workers=args.num_workers)
                metrics = evaluate(model, test_loader, device)
                fold_test_metrics[test_name].append(metrics)
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

    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Train / predict subcommands
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    print(f"Device: {device}")
    if args.folds:
        _run_kfold(args, device)
    else:
        _run_single(args, device)


def cmd_predict(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    ckpt   = torch.load(args.checkpoint, map_location=device, weights_only=False)
    margs  = ckpt["model_args"]
    # backward-compat defaults
    for key, val in [("use_conservation", True), ("use_eclip", True),
                     ("use_tspot", True), ("use_energy", True)]:
        margs.setdefault(key, val)

    energy_stats = ckpt["energy_stats"]
    model = MiRBindCNN(**margs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch')}, "
          f"val_metrics={ckpt.get('val_metrics')})")

    ds = MiRNAInteractionDataset(
        args.input, energy_stats=energy_stats, has_labels=True,
        mre_col=args.mre_col, mirna_col=args.mirna_col)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for wc_mat, cons, eclip, tspot, energy, labels in loader:
            wc_mat = wc_mat.to(device)
            cons   = cons.to(device)
            eclip  = eclip.to(device)
            tspot  = tspot.to(device)
            energy = energy.to(device)
            logits = model(wc_mat, cons, eclip, tspot, energy)
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
    tr.add_argument("--folds",      type=int, default=None)
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
    tr.add_argument("--vec-channels",    type=int,   default=64,  dest="vec_channels")
    tr.add_argument("--vec-blocks",      type=int,   default=3,   dest="vec_blocks")
    tr.add_argument("--energy-dim",      type=int,   default=64,  dest="energy_dim")
    tr.add_argument("--vec-kernel-size", type=int,   default=7,   dest="vec_kernel_size")
    tr.add_argument("--seq-dropout",     type=float, default=0.3, dest="seq_dropout",
                    help="Dropout in the 2D miRBind conv blocks.")
    tr.add_argument("--vec-dropout",     type=float, default=0.15, dest="vec_dropout",
                    help="Dropout in the 1D vector branch blocks.")
    tr.add_argument("--norm", choices=["batch", "layer"], default="batch")
    tr.add_argument("--no-conservation", action="store_true")
    tr.add_argument("--no-eclip",        action="store_true")
    tr.add_argument("--no-tspot",        action="store_true")
    tr.add_argument("--no-energy",       action="store_true")
    # Training
    tr.add_argument("--epochs",       type=int,   default=40)
    tr.add_argument("--batch-size",   type=int,   default=64)
    tr.add_argument("--lr",           type=float, default=1e-3)
    tr.add_argument("--weight-decay", type=float, default=1e-4)
    tr.add_argument("--warmup-steps", type=int,   default=200)
    tr.add_argument("--num-workers",  type=int,   default=2)
    tr.add_argument("--patience",     type=int,   default=10)
    tr.add_argument("--balance",      action="store_true")
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
    pr.add_argument("--threshold",  type=float, default=0.5)
    pr.add_argument("--batch-size", type=int,   default=256)
    pr.add_argument("--mre-col",   default="mre_sequence",   dest="mre_col")
    pr.add_argument("--mirna-col", default="mirna_sequence", dest="mirna_col")
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
