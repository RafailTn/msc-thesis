"""Two-tower miRNA-MRE interaction model with a bottlenecked fusion.

Motivation
----------
The single-stream model (``transformer_imp.py``) concatenates ``MRE ++ miRNA``
into one sequence and lets all-to-all self-attention mix the two strands from
layer 0. That lets the network memorise the raw (miRNA_i, MRE_j) co-occurrence
table and overfits.

This module instead encodes each strand *independently* and compresses it to a
small fixed-size summary vector (``bottleneck_dim``) BEFORE any cross-strand
interaction. The bottleneck is the primary regulariser: fusion happens on
low-dimensional *representations*, not on the raw per-position sequences, so the
model must learn transferable interaction features rather than the co-occurrence
table.

Hard invariants respected (see task spec):
  * No base-pairing is enforced anywhere: no alignment, no folding/MFE, no
    seed-position (2-8) masks. Pairing stays fully learnable.
  * The MRE is NOT reverse-complemented by default (data is stored native
    5'->3', transcript-oriented). See ``mre_orientation`` below.
  * Sequence lengths / channel counts are read from config/data, not hardcoded.

Drop-in contract
----------------
Matches the existing single-stream models exactly so it swaps into their
training loops with a one-line change:
  * batch is the 2-tuple ``(x, labels)`` produced by ``collate_fn_onehot``,
    with ``x`` of shape ``(B, L, 5)`` == ``[A, C, G, T, segment_id]``;
  * ``forward(x) -> (B,)`` logits;
  * ``BCEWithLogitsLoss`` by default (plus optional label smoothing / GCE / SCE).

The two strands are recovered *inside* the model by splitting on the segment-id
channel, so the dataset/collate stay untouched (no new batch schema).
"""

from __future__ import annotations

import argparse

import torch
import polars as ps
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import AveragePrecision, AUROC, MatthewsCorrCoef
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from utils import OneHotDataset, collate_fn_onehot, DnaOneHotEncoder, AttentionPool
from dotenv import load_dotenv

load_dotenv()


# --------------------------------------------------------------------------- #
# Stream splitting: recover the two strands from the concatenated one-hot batch
# --------------------------------------------------------------------------- #
def split_streams(
    x: torch.Tensor,
    mre_orientation: str = "native",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the concatenated ``(B, L, 5)`` batch into two left-aligned strands.

    The collated batch is ``MRE(seg=0) ++ miRNA(seg=1) ++ pad`` per row. We use
    the segment-id channel (index 4) and one-hot validity to carve out each
    strand and re-left-align it into its own tensor, so each tower gets its own
    positions starting at 0 (its own learned PE is meaningful).

    Returns ``(mre_x, mre_pad, mirna_x, mirna_pad)`` where the ``*_x`` tensors
    carry only the nucleotide channels ``[A, C, G, T]`` (segment id is dropped,
    it is constant within a strand) and ``*_pad`` are ``True`` at padding.

    Parameters
    ----------
    mre_orientation : {"native", "rc"}
        ``native`` (default) feeds the MRE as stored (5'->3', transcript
        oriented). ``rc`` reverse-complements it. Tradeoff: RC makes canonical
        Watson-Crick pairing an *identity* match between aligned registers and
        re-privileges that register, biasing what the fusion can learn; leaving
        it native keeps canonical / G.U wobble / bulge pairing fully learnable.
    """
    seg = x[..., -1]                       # (B, L) segment id
    valid = x[..., :-1].sum(dim=-1) > 0    # (B, L) True where a real nucleotide
    is_mre = valid & (seg < 0.5)
    is_mir = valid & (seg > 0.5)
    mre_len = is_mre.sum(dim=1)            # (B,)
    mir_len = is_mir.sum(dim=1)            # (B,)

    B, L, _ = x.shape
    nuc = x[..., :-1]                       # (B, L, nuc_dim), drop segment id
    device = x.device
    ar = torch.arange(L, device=device)

    # MRE occupies [0, mre_len) already (it is the first block, left-aligned).
    Lm = int(mre_len.max().item()) if B else 0
    Lm = max(Lm, 1)
    mre_x = nuc[:, :Lm, :]
    mre_pad = ar[:Lm].unsqueeze(0) >= mre_len.unsqueeze(1)     # (B, Lm)

    # miRNA occupies [mre_len, mre_len + mir_len); its start varies per row, so
    # gather to left-align it into its own tensor.
    Ln = int(mir_len.max().item()) if B else 0
    Ln = max(Ln, 1)
    pos = torch.arange(Ln, device=device).unsqueeze(0)         # (1, Ln)
    src = (mre_len.unsqueeze(1) + pos).clamp(max=L - 1)         # (B, Ln)
    mir_pad = pos >= mir_len.unsqueeze(1)                       # (B, Ln)
    gather_idx = src.unsqueeze(-1).expand(-1, -1, nuc.size(-1))
    mir_x = torch.gather(nuc, 1, gather_idx)                    # (B, Ln, nuc_dim)

    if mre_orientation == "rc":
        mre_x = _reverse_complement(mre_x, mre_len)
    elif mre_orientation != "native":
        raise ValueError(f"mre_orientation must be 'native' or 'rc', got {mre_orientation!r}")

    # Zero out padding positions so masked-out gather artefacts never leak.
    mre_x = mre_x.masked_fill(mre_pad.unsqueeze(-1), 0.0)
    mir_x = mir_x.masked_fill(mir_pad.unsqueeze(-1), 0.0)
    return mre_x, mre_pad, mir_x, mir_pad


def _reverse_complement(nuc: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Reverse-complement a left-aligned ``[A, C, G, T]`` one-hot tensor.

    Complement = flip the 4 nucleotide channels (A<->T, C<->G). Reverse = flip
    the valid prefix of each row (per-row length, keeping padding at the tail).
    """
    comp = nuc.flip(dims=[-1])                       # [A,C,G,T] -> [T,G,C,A]
    B, Lm, _ = comp.shape
    ar = torch.arange(Lm, device=comp.device).unsqueeze(0)     # (1, Lm)
    rev_src = (lengths.unsqueeze(1) - 1 - ar).clamp(min=0)      # (B, Lm)
    idx = rev_src.unsqueeze(-1).expand(-1, -1, comp.size(-1))
    return torch.gather(comp, 1, idx)

class TransformerBlock(nn.Module):
    """
    Single pre-LayerNorm transformer block:
      x → norm → MHA → residual → norm → FFN → residual
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + self.drop(attn_out)
        # FFN with pre-norm
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

# --------------------------------------------------------------------------- #
# Towers
# --------------------------------------------------------------------------- #
class Tower(nn.Module):
    """Independent per-strand encoder ending in a low-dim bottleneck vector.

    one-hot -> linear proj + learned PE -> N pre-norm transformer blocks ->
    attention pool -> linear bottleneck. Nothing here sees the other strand.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int,
        bottleneck_dim: int,
    ):
        super().__init__()
        self.encoder = DnaOneHotEncoder(
            input_dim=input_dim, emb_size=d_model, max_seq_len=max_seq_len, dropout=dropout
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.pool = AttentionPool(d_model)
        # The bottleneck: compress the pooled summary to a small vector. This
        # is the capacity control that forces the fusion to work on
        # representations rather than raw co-occurrence.
        self.bottleneck = nn.Linear(d_model, bottleneck_dim)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, input_dim); pad_mask: (B, L) True = padding
        h = self.encoder(x)
        for block in self.blocks:
            h = block(h, key_padding_mask=pad_mask)
        h = self.norm(h)
        summary = self.pool(h, pad_mask)      # (B, d_model)
        return self.bottleneck(summary)       # (B, bottleneck_dim)


# --------------------------------------------------------------------------- #
# Fusion of the two bottleneck vectors
# --------------------------------------------------------------------------- #
class CrossAttentionFusion(nn.Module):
    """Perceiver-style fusion: a few learned latents cross-attend to the two
    strand summaries. Dropout lives here; the latent stays low-dimensional.

    Note on the optional 'learnable complementarity bias' (task, optional): a
    per-base bias b_ij = f(base_i, base_j) needs *per-position* cross-attention
    between the strands, which is exactly what the bottleneck deliberately
    discards. Adding it would defeat the bottleneck, so it is intentionally not
    wired here; pairing remains learnable inside each tower instead.
    """

    def __init__(
        self,
        bottleneck_dim: int,
        fusion_dim: int,
        nhead: int,
        num_latents: int,
        dropout: float,
    ):
        super().__init__()
        self.in_proj = nn.Linear(bottleneck_dim, fusion_dim)
        self.latents = nn.Parameter(torch.randn(num_latents, fusion_dim) * 0.02)
        self.norm_q = nn.LayerNorm(fusion_dim)
        self.norm_kv = nn.LayerNorm(fusion_dim)
        self.attn = nn.MultiheadAttention(fusion_dim, nhead, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(fusion_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )
        self.drop = nn.Dropout(dropout)
        self.out_dim = fusion_dim

    def forward(self, z_mir: torch.Tensor, z_mre: torch.Tensor) -> torch.Tensor:
        B = z_mir.size(0)
        kv = torch.stack([self.in_proj(z_mir), self.in_proj(z_mre)], dim=1)   # (B, 2, F)
        q = self.latents.unsqueeze(0).expand(B, -1, -1)                       # (B, Nl, F)
        q_n, kv_n = self.norm_q(q), self.norm_kv(kv)
        attn_out, _ = self.attn(q_n, kv_n, kv_n)
        q = q + self.drop(attn_out)
        q = q + self.drop(self.ffn(self.norm_ffn(q)))
        return q.mean(dim=1)                                                  # (B, F)


class ConcatMLPFusion(nn.Module):
    """Simplest fusion fallback: concat the two summaries and MLP. Dropout here."""

    def __init__(self, bottleneck_dim: int, fusion_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * bottleneck_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = fusion_dim

    def forward(self, z_mir: torch.Tensor, z_mre: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_mir, z_mre], dim=-1))


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class TwoTowerDNA(nn.Module):
    """Two independent towers -> bottleneck -> fusion -> MLP head -> logit.

    Input:  (B, L, nuc_dim + 1)   concatenated one-hot with segment id
    Output: (B,)                  scalar logit per pair
    """

    def __init__(
        self,
        nuc_dim: int = 4,
        mirna_d_model: int = 128,
        mirna_nhead: int = 4,
        mirna_num_layers: int = 2,
        mirna_dim_feedforward: int = 512,
        mirna_max_seq_len: int = 64,
        mre_d_model: int = 128,
        mre_nhead: int = 4,
        mre_num_layers: int = 2,
        mre_dim_feedforward: int = 512,
        mre_max_seq_len: int = 256,
        bottleneck_dim: int = 64,
        tower_dropout: float = 0.3,
        fusion: str = "cross_attention",
        fusion_dim: int = 128,
        fusion_nhead: int = 4,
        fusion_num_latents: int = 4,
        fusion_dropout: float = 0.3,
        head_hidden: int = 128,
        head_dropout: float = 0.3,
        mre_orientation: str = "native",
    ):
        super().__init__()
        self.nuc_dim = nuc_dim
        self.mre_orientation = mre_orientation

        # Towers do NOT share weights: miRNA and MRE differ in length regime and
        # biological role.
        self.mirna_tower = Tower(
            input_dim=nuc_dim, d_model=mirna_d_model, nhead=mirna_nhead,
            num_layers=mirna_num_layers, dim_feedforward=mirna_dim_feedforward,
            dropout=tower_dropout, max_seq_len=mirna_max_seq_len,
            bottleneck_dim=bottleneck_dim,
        )
        self.mre_tower = Tower(
            input_dim=nuc_dim, d_model=mre_d_model, nhead=mre_nhead,
            num_layers=mre_num_layers, dim_feedforward=mre_dim_feedforward,
            dropout=tower_dropout, max_seq_len=mre_max_seq_len,
            bottleneck_dim=bottleneck_dim,
        )

        if fusion == "cross_attention":
            self.fusion = CrossAttentionFusion(
                bottleneck_dim=bottleneck_dim, fusion_dim=fusion_dim,
                nhead=fusion_nhead, num_latents=fusion_num_latents, dropout=fusion_dropout,
            )
        elif fusion == "concat_mlp":
            self.fusion = ConcatMLPFusion(
                bottleneck_dim=bottleneck_dim, fusion_dim=fusion_dim, dropout=fusion_dropout,
            )
        else:
            raise ValueError(f"fusion must be 'cross_attention' or 'concat_mlp', got {fusion!r}")

        self.head = nn.Sequential(
            nn.LayerNorm(self.fusion.out_dim),
            nn.Linear(self.fusion.out_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, nuc_dim + 1); the last channel is the segment id.
        if x.size(-1) != self.nuc_dim + 1:
            raise ValueError(
                f"expected {self.nuc_dim + 1} channels (nuc_dim={self.nuc_dim} + segment id), "
                f"got {x.size(-1)}"
            )
        mre_x, mre_pad, mir_x, mir_pad = split_streams(x, self.mre_orientation)
        z_mre = self.mre_tower(mre_x, mre_pad)       # (B, bottleneck_dim)
        z_mir = self.mirna_tower(mir_x, mir_pad)     # (B, bottleneck_dim)
        fused = self.fusion(z_mir, z_mre)            # (B, fusion_dim)
        return self.head(fused).squeeze(-1)          # (B,)


# --------------------------------------------------------------------------- #
# Noise-robust / smoothed losses (negatives contain accidental seed matches)
# --------------------------------------------------------------------------- #
class SmoothedBCEWithLogits(nn.Module):
    """BCEWithLogits with optional symmetric label smoothing on the targets."""

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.eps = label_smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.eps > 0:
            targets = targets * (1 - self.eps) + 0.5 * self.eps
        return self.bce(logits, targets)


class BinaryGCELoss(nn.Module):
    """Generalized Cross Entropy (Zhang & Sabuncu 2018), binary form.

    Interpolates MAE (q->1, noise-robust) and CE (q->0). Tolerates label noise
    such as negatives that carry accidental seed matches.
    """

    def __init__(self, q: float = 0.7, eps: float = 1e-7):
        super().__init__()
        self.q = q
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)      # prob of the (soft) true class
        pt = pt.clamp(min=self.eps, max=1.0)
        return ((1 - pt.pow(self.q)) / self.q).mean()


class BinarySCELoss(nn.Module):
    """Symmetric Cross Entropy (Wang et al. 2019): alpha*CE + beta*RCE."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.bce(logits, targets)
        p = torch.sigmoid(logits).clamp(min=self.eps, max=1 - self.eps)
        y = targets.clamp(min=self.eps, max=1 - self.eps)
        # Reverse CE: swap the roles of prediction and (clamped) target.
        rce = -(p * torch.log(y) + (1 - p) * torch.log(1 - y)).mean()
        return self.alpha * ce + self.beta * rce


def build_criterion(
    loss: str = "bce",
    label_smoothing: float = 0.0,
    gce_q: float = 0.7,
    sce_alpha: float = 1.0,
    sce_beta: float = 1.0,
) -> nn.Module:
    if loss == "bce":
        return SmoothedBCEWithLogits(label_smoothing=label_smoothing)
    if loss == "gce":
        return BinaryGCELoss(q=gce_q)
    if loss == "sce":
        return BinarySCELoss(alpha=sce_alpha, beta=sce_beta)
    raise ValueError(f"loss must be one of 'bce', 'gce', 'sce', got {loss!r}")


# --------------------------------------------------------------------------- #
# Lightning module — mirrors TransformerDNALightning for a drop-in swap
# --------------------------------------------------------------------------- #
class TwoTowerLightning(pl.LightningModule):
    def __init__(
        self,
        nuc_dim: int = 4,
        # miRNA tower
        mirna_d_model: int = 128,
        mirna_nhead: int = 4,
        mirna_num_layers: int = 2,
        mirna_dim_feedforward: int = 512,
        mirna_max_seq_len: int = 64,
        # MRE tower
        mre_d_model: int = 128,
        mre_nhead: int = 4,
        mre_num_layers: int = 2,
        mre_dim_feedforward: int = 512,
        mre_max_seq_len: int = 256,
        # bottleneck / fusion / head
        bottleneck_dim: int = 64,
        tower_dropout: float = 0.3,
        fusion: str = "cross_attention",
        fusion_dim: int = 128,
        fusion_nhead: int = 4,
        fusion_num_latents: int = 4,
        fusion_dropout: float = 0.3,
        head_hidden: int = 128,
        head_dropout: float = 0.3,
        mre_orientation: str = "native",
        # loss
        loss: str = "bce",
        label_smoothing: float = 0.0,
        gce_q: float = 0.7,
        sce_alpha: float = 1.0,
        sce_beta: float = 1.0,
        # optim
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = TwoTowerDNA(
            nuc_dim=nuc_dim,
            mirna_d_model=mirna_d_model, mirna_nhead=mirna_nhead,
            mirna_num_layers=mirna_num_layers, mirna_dim_feedforward=mirna_dim_feedforward,
            mirna_max_seq_len=mirna_max_seq_len,
            mre_d_model=mre_d_model, mre_nhead=mre_nhead,
            mre_num_layers=mre_num_layers, mre_dim_feedforward=mre_dim_feedforward,
            mre_max_seq_len=mre_max_seq_len,
            bottleneck_dim=bottleneck_dim, tower_dropout=tower_dropout,
            fusion=fusion, fusion_dim=fusion_dim, fusion_nhead=fusion_nhead,
            fusion_num_latents=fusion_num_latents, fusion_dropout=fusion_dropout,
            head_hidden=head_hidden, head_dropout=head_dropout,
            mre_orientation=mre_orientation,
        )
        self.criterion = build_criterion(
            loss=loss, label_smoothing=label_smoothing,
            gce_q=gce_q, sce_alpha=sce_alpha, sce_beta=sce_beta,
        )
        self.train_ap    = AveragePrecision(task='binary')
        self.val_ap      = AveragePrecision(task='binary')
        self.test_ap     = AveragePrecision(task='binary')
        self.train_auroc = AUROC(task='binary')
        self.val_auroc   = AUROC(task='binary')
        self.test_auroc  = AUROC(task='binary')
        self.train_mcc   = MatthewsCorrCoef(task='binary')
        self.val_mcc     = MatthewsCorrCoef(task='binary')
        self.test_mcc    = MatthewsCorrCoef(task='binary')
        self.final_test_ap    = AveragePrecision(task='binary')
        self.final_test_auroc = AUROC(task='binary')
        self.final_test_mcc   = MatthewsCorrCoef(task='binary')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 5) == (B, L, nuc_dim + segment id)
        return self.model(x)

    def _shared_step(self, batch, batch_idx):
        x, labels = batch
        logits = self(x)
        loss = self.criterion(logits, labels)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        return loss, preds, probs, labels

    def training_step(self, batch, batch_idx):
        loss, preds, probs, labels = self._shared_step(batch, batch_idx)
        self.train_ap(probs, labels.int())
        self.train_auroc(probs, labels.int())
        self.train_mcc(preds, labels.int())
        self.log('train_loss',  loss,             on_step=True,  on_epoch=True, prog_bar=True)
        self.log('train_ap',    self.train_ap,    on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_mcc',   self.train_mcc,   on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, probs, labels = self._shared_step(batch, batch_idx)
        self.val_ap(probs, labels.int())
        self.val_auroc(probs, labels.int())
        self.val_mcc(preds, labels.int())
        self.log('val_loss',  loss,           on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_ap',    self.val_ap,    on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mcc',   self.val_mcc,   on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        loss, preds, probs, labels = self._shared_step(batch, batch_idx)
        prefix = "test" if dataloader_idx == 0 else "leftout"
        if dataloader_idx == 0:
            self.test_ap(probs, labels.int())
            self.test_auroc(probs, labels.int())
            self.test_mcc(preds, labels.int())
            ap_metric, auroc_metric, mcc_metric = self.test_ap, self.test_auroc, self.test_mcc
        else:
            self.final_test_ap(probs, labels.int())
            self.final_test_auroc(probs, labels.int())
            self.final_test_mcc(preds, labels.int())
            ap_metric, auroc_metric, mcc_metric = self.final_test_ap, self.final_test_auroc, self.final_test_mcc

        self.log(f'{prefix}_loss',  loss,         on_step=False, on_epoch=True)
        self.log(f'{prefix}_ap',    ap_metric,    on_step=False, on_epoch=True)
        self.log(f'{prefix}_auroc', auroc_metric, on_step=False, on_epoch=True)
        self.log(f'{prefix}_mcc',   mcc_metric,   on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch['input_ids'] if isinstance(batch, dict) else batch[0]
        logits = self(x)
        probs  = torch.sigmoid(logits)
        preds  = (probs > 0.5).long()
        return {'predictions': preds, 'probabilities': probs}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1},
        }


# --------------------------------------------------------------------------- #
# Lightweight model registry / factory (there is no argparse/Hydra in the repo;
# this keeps the new model selectable by name without touching existing files).
# --------------------------------------------------------------------------- #
MODEL_REGISTRY = {
    "two_tower": TwoTowerLightning,
}


def build_model(name: str = "two_tower", **cfg) -> pl.LightningModule:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; choices: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**cfg)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI for the two-tower training run.

    Every model/loss knob defaults to the ``TwoTowerLightning.__init__``
    default, so passing no flags reproduces the original hardcoded behavior.
    """
    p = argparse.ArgumentParser(
        description="Train the two-tower miRNA-MRE model (StratifiedGroupKFold over mir_fam).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Loss / label noise (the reason the CLI exists)
    g = p.add_argument_group("loss")
    g.add_argument("--loss", choices=["bce", "gce", "sce"], default="bce",
                   help="bce=BCE(+label smoothing); gce=Generalized CE; sce=Symmetric CE (noise-robust)")
    g.add_argument("--label-smoothing", type=float, default=0.0, help="only used by --loss bce")
    g.add_argument("--gce-q", type=float, default=0.7, help="only used by --loss gce")
    g.add_argument("--sce-alpha", type=float, default=1.0, help="only used by --loss sce")
    g.add_argument("--sce-beta", type=float, default=1.0, help="only used by --loss sce")

    # Architecture / regularization
    g = p.add_argument_group("model")
    g.add_argument("--mre-orientation", choices=["native", "rc"], default="native",
                   help="native = feed MRE as stored (5'->3'); rc = reverse-complement it")
    g.add_argument("--bottleneck-dim", type=int, default=64, help="main capacity control")
    g.add_argument("--fusion", choices=["cross_attention", "concat_mlp"], default="cross_attention")
    g.add_argument("--fusion-dropout", type=float, default=0.3)
    g.add_argument("--tower-dropout", type=float, default=0.3)
    g.add_argument("--head-dropout", type=float, default=0.3)

    # Run / trainer
    g = p.add_argument_group("run")
    g.add_argument("--data-dir", default="../data", help="dir holding the AGO2_eCLIP_Manakov2022_*_v7.tsv files")
    g.add_argument("--out-dir", default="../models", help="where checkpoints are written")
    g.add_argument("--epochs", type=int, default=25)
    g.add_argument("--batch-size", type=int, default=256)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight-decay", type=float, default=1e-2)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--n-splits", type=int, default=5)
    g.add_argument("--folds", type=int, default=None,
                   help="run only the first N folds (default: all n_splits)")
    g.add_argument("--limit-rows", type=int, default=None,
                   help="subsample the train df to N rows (quick local test runs)")
    g.add_argument("--precision", default="auto",
                   help="Lightning precision; 'auto' = 16-mixed on GPU, 32-true on CPU")
    g.add_argument("--wandb", dest="wandb", action="store_true", default=True, help="log to Weights & Biases (default)")
    g.add_argument("--no-wandb", dest="wandb", action="store_false", help="disable W&B logging")
    g.add_argument("--wandb-project", default="two-tower-mirna-bottleneck")

    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    # Mirrors transformer_imp.main() but CLI-driven; swap in for the single-stream
    # model by pointing the training loop at build_model('two_tower', ...).
    #
    # Reads the local v7 TSVs. Their columns differ from OneHotDataset's
    # expectations, so rename: gene -> mre_sequence (the MRE target site, stored
    # native 5'->3', transcript-oriented), noncodingRNA -> mirna_sequence,
    # noncodingRNA_fam -> mir_fam (StratifiedGroupKFold grouping key).
    args = parse_args(argv)

    # Model config assembled from the CLI; unspecified knobs fall back to the
    # TwoTowerLightning defaults.
    model_cfg = dict(
        loss=args.loss, label_smoothing=args.label_smoothing,
        gce_q=args.gce_q, sce_alpha=args.sce_alpha, sce_beta=args.sce_beta,
        mre_orientation=args.mre_orientation,
        bottleneck_dim=args.bottleneck_dim,
        fusion=args.fusion, fusion_dropout=args.fusion_dropout,
        tower_dropout=args.tower_dropout, head_dropout=args.head_dropout,
        learning_rate=args.lr, weight_decay=args.weight_decay,
    )

    # 'auto' precision: 16-mixed only makes sense on GPU; use 32-true on CPU.
    precision = args.precision
    if precision == "auto":
        precision = "16-mixed" if torch.cuda.is_available() else "32-true"

    def load_v7(path: str, with_fam: bool):
        rename = {'gene': 'mre_sequence', 'noncodingRNA': 'mirna_sequence'}
        cols = ['gene', 'noncodingRNA', 'label']
        if with_fam:
            cols.insert(2, 'noncodingRNA_fam')
            rename['noncodingRNA_fam'] = 'mir_fam'
        d = ps.read_csv(path, separator='\t', columns=cols).rename(rename)
        return d.unique(subset=['mre_sequence', 'mirna_sequence'], keep='none')

    df            = load_v7(f'{args.data_dir}/AGO2_eCLIP_Manakov2022_train_v7.tsv',   with_fam=True)
    test_df       = load_v7(f'{args.data_dir}/AGO2_eCLIP_Manakov2022_test_v7.tsv',    with_fam=False)
    final_test_df = load_v7(f'{args.data_dir}/AGO2_eCLIP_Manakov2022_leftout_v7.tsv', with_fam=False)
    if args.limit_rows is not None:
        # Random sample, not head(): the TSVs are sorted by miRNA family, so the
        # first N rows collapse to a few families / one class and produce an
        # empty val fold. Sampling keeps the subset representative. Applied to
        # all splits so a quick run also has a quick test phase.
        def _subsample(d):
            return d.sample(n=min(args.limit_rows, d.height), shuffle=True, seed=42)
        df, test_df, final_test_df = _subsample(df), _subsample(test_df), _subsample(final_test_df)

    # NOTE (cold-split): the split below is cold-miRNA-*family* (groups=mir_fam)
    # but NOT cold-target. Genes/MREs can recur across train and val. See the
    # task summary for the flag.
    n_folds = args.n_splits if args.folds is None else min(args.folds, args.n_splits)
    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(sgkf.split(df, df['label'], groups=df['mir_fam'])):
        if i >= n_folds:
            break
        final_train_data, final_val_data = df[train_idx], df[val_idx]

        early_stop_callback = EarlyStopping(monitor="val_ap", patience=7, mode="max")
        checkpoint_callback = ModelCheckpoint(
            monitor="val_ap", mode="max",
            dirpath=args.out_dir,
            filename=f"TwoTower-{args.fusion}-{args.loss}-{args.mre_orientation}-bn{args.bottleneck_dim}_fold{i}"
        )

        train_dataset      = OneHotDataset(final_train_data)
        eval_dataset       = OneHotDataset(final_val_data)
        test_dataset       = OneHotDataset(test_df)
        final_test_dataset = OneHotDataset(final_test_df)

        train_dataloader      = DataLoader(train_dataset,      batch_size=args.batch_size, collate_fn=collate_fn_onehot, shuffle=True,  num_workers=args.num_workers)
        val_dataloader        = DataLoader(eval_dataset,       batch_size=args.batch_size, collate_fn=collate_fn_onehot, shuffle=False, num_workers=args.num_workers)
        test_dataloader       = DataLoader(test_dataset,       batch_size=args.batch_size, collate_fn=collate_fn_onehot, shuffle=False, num_workers=args.num_workers)
        final_test_dataloader = DataLoader(final_test_dataset, batch_size=args.batch_size, collate_fn=collate_fn_onehot, shuffle=False, num_workers=args.num_workers)

        logger = False
        if args.wandb:
            logger = WandbLogger(
                project=args.wandb_project,
                name=f"fold-{i}",
                log_model=False,
            )

        model = build_model("two_tower", **model_cfg)
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            callbacks=[early_stop_callback, checkpoint_callback],
            accelerator='auto',
            precision=precision,
            logger=logger,
        )

        trainer.fit(model, train_dataloader, val_dataloader)
        trainer.test(model, [test_dataloader, final_test_dataloader])
        if args.wandb:
            logger.experiment.finish()


if __name__ == "__main__":
    main()
