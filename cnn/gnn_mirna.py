#!/usr/bin/env python3
"""
gnn_mirna.py — graph neural network alternative to the 2D-CNN for miRNA–MRE
interaction classification.

Motivation
----------
The CNN encodes the duplex as a 2D Watson-Crick complementarity image and pools
over it. A 3′-compensatory site (weak seed + strong 3′ supplementary pairing)
appears as *two disjoint* paired blocks, which a translation-invariant conv +
global pool struggles to aggregate. A graph represents those two blocks natively:
each is just a cluster of base-pair edges, and message passing can integrate both
regardless of how far apart they sit on the sequences.

Graph
-----
Nodes  : the miRNA nucleotides (≤30) and the MRE nucleotides (≤50), 80 nodes max.
Edges  : two relation types —
         * backbone  — consecutive nucleotides within each strand (sequence order)
         * base-pair — candidate WC / G·U pairs between a miRNA and an MRE node,
                       weighted by graded pairing strength (G·C > A·U > G·U).
The whole graph is built on-device inside forward() from the same integer
nucleotide-index tensors the CNN uses, so this script reuses the CNN's dataset,
cache, training loop, k-fold CV, metrics and error-analysis code unchanged — only
the model differs. Pure PyTorch (dense batched message passing over ≤80 nodes);
no torch_geometric dependency.

Usage
-----
  # seeded k-fold CV (mirrors the CNN CLI)
  python cnn/gnn_mirna.py train --train train.tsv --folds 5 --seed 42 \\
      --gnn-pool attention --pool-heads 2 --hidden 128 --n-layers 3 \\
      --out checkpoints/gnn.pt --test test.tsv

  # single split
  python cnn/gnn_mirna.py train --train train.tsv --val val.tsv --out checkpoints/gnn.pt

  # score new pairs (+ optional error dump)
  python cnn/gnn_mirna.py predict --checkpoint checkpoints/gnn.pt \\
      --input pairs.tsv --output preds.tsv --error-dump errors.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse the CNN's data + training machinery wholesale: the model is the only
# new piece. These helpers are model-agnostic — they call model(mi, ti).
from cnn_branches_mirbind import (  # noqa: E402
    MAX_MIRNA, MRE_LEN, _PAIR_STRENGTH,
    MiRNAInteractionDataset,
    _make_loader, _make_activation,
    _train_one_run, evaluate, predict_logits,
    _binary_metrics, _metrics_from_probs,
    _read_table, _dedup_pairs, _set_global_seed,
    _write_error_dump,
    HAS_SKLEARN, HAS_WANDB,
)

if HAS_WANDB:
    import wandb  # noqa: E402
if HAS_SKLEARN:
    from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

N_NODES = MAX_MIRNA + MRE_LEN   # miRNA nodes [0, MAX_MIRNA) + MRE nodes [MAX_MIRNA, N)


# ---------------------------------------------------------------------------
# Static graph scaffolding (shape-only, independent of the sequences)
# ---------------------------------------------------------------------------

def _backbone_template() -> np.ndarray:
    """(N, N) symmetric adjacency of within-strand consecutive nucleotides.

    No edge crosses the miRNA→MRE boundary; cross-strand links are base-pair
    edges, added per-sample in forward()."""
    a = np.zeros((N_NODES, N_NODES), dtype=np.float32)
    for i in range(MAX_MIRNA - 1):                       # miRNA backbone
        a[i, i + 1] = a[i + 1, i] = 1.0
    for j in range(MAX_MIRNA, N_NODES - 1):              # MRE backbone
        a[j, j + 1] = a[j + 1, j] = 1.0
    return a


def _node_meta() -> tuple[np.ndarray, np.ndarray]:
    """Per-node strand id (0=miRNA, 1=MRE) and within-strand normalised position."""
    strand = np.concatenate([np.zeros(MAX_MIRNA, dtype=np.int64),
                             np.ones(MRE_LEN,  dtype=np.int64)])
    pos    = np.concatenate([np.arange(MAX_MIRNA, dtype=np.float32) / (MAX_MIRNA - 1),
                             np.arange(MRE_LEN,  dtype=np.float32) / (MRE_LEN - 1)])
    return strand, pos.astype(np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _RelationalGNNLayer(nn.Module):
    """One relational message-passing step over the two edge types.

    h' = h + dropout( act( LN( W_self h + Â_bb h W_bb + Â_pair h W_pair ) ) )

    Â_* are the row-normalised backbone / base-pair adjacencies. Separate weight
    matrices per relation let backbone (sequence context) and base-pairing
    (cross-strand structure) be combined differently — an R-GCN-style update.
    """

    def __init__(self, dim: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.w_self = nn.Linear(dim, dim)
        self.w_bb   = nn.Linear(dim, dim, bias=False)
        self.w_pair = nn.Linear(dim, dim, bias=False)
        self.norm   = nn.LayerNorm(dim)
        self.act    = _make_activation(activation)
        self.drop   = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, a_bb: torch.Tensor,
                a_pair: torch.Tensor) -> torch.Tensor:
        msg = (self.w_self(h)
               + torch.bmm(a_bb,   self.w_bb(h))
               + torch.bmm(a_pair, self.w_pair(h)))
        return h + self.drop(self.act(self.norm(msg)))


class _NodeGeMPool(nn.Module):
    """Generalized-mean readout over (masked) nodes: (B, N, D) → (B, D).

    GeM(x) = ( mean_i x_i^p )^(1/p) over the real nodes; p=1 recovers mean,
    p→∞ approaches max, and p is learnable. Inputs are clamped to ≥eps (the
    power-mean is only defined for non-negative values), mirroring the CNN's
    GeM2d. Pad nodes are dropped *after* the clamp so they contribute nothing.

    Like the CNN's GeM, this is a single power-mean: it emphasises the strongest
    region but cannot keep two disjoint paired regions (seed + 3′) separate the
    way multi-head attention can.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, h: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        xp    = h.clamp(min=self.eps).pow(self.p) * real.unsqueeze(-1)
        denom = real.sum(1, keepdim=True).clamp_min(1.0)
        return (xp.sum(1) / denom).pow(1.0 / self.p)


class _NodeAttentionPool(nn.Module):
    """Multi-head content-based attention readout over (masked) nodes.

    Each head learns its own soft selection over nodes, so different heads can
    focus on different paired regions (e.g. seed vs 3′ supplementary) and the
    graph embedding keeps both. Pooled width = heads * dim.
    """

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError(f"pool_heads must be >= 1, got {heads}")
        self.score = nn.Linear(dim, heads)

    def forward(self, h: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        # h: (B, N, D); real: (B, N) bool/float node mask
        s = self.score(h)                                         # (B, N, heads)
        s = s.masked_fill(~real.bool().unsqueeze(-1), float("-inf"))
        a = s.softmax(dim=1)                                      # over nodes
        pooled = torch.einsum("bnh,bnd->bhd", a, h)              # (B, heads, D)
        return pooled.flatten(1)                                  # (B, heads*D)


class MiRNAGraphNet(nn.Module):
    """Relational GNN over the miRNA–MRE base-pairing graph.

    forward(mi, ti) -> (B,) logits, matching the CNN's interface so the shared
    training / evaluation / prediction helpers work unchanged.

    Parameters
    ----------
    hidden : int          node embedding / message width.
    n_layers : int        number of message-passing steps (graph depth ≈ how far
                          signal propagates between paired regions).
    dropout : float       dropout in the message-passing layers and head.
    gnn_pool : str        graph readout: "mean", "max" or "attention".
    pool_heads : int      attention heads when gnn_pool="attention".
    out_dim : int         embedding size after the dense projection.
    activation : str      activation for layers/dense.
    """

    def __init__(
        self,
        hidden:     int   = 128,
        n_layers:   int   = 3,
        dropout:    float = 0.3,
        gnn_pool:   str   = "mean",
        pool_heads: int   = 1,
        out_dim:    int   = 128,
        activation: str   = "leaky_relu",
    ) -> None:
        super().__init__()
        self.gnn_pool = gnn_pool

        # Node inputs: nucleotide identity + strand + within-strand position.
        self.nuc_emb    = nn.Embedding(5, hidden, padding_idx=4)   # ACGU + pad(4)
        self.strand_emb = nn.Embedding(2, hidden)                  # miRNA / MRE
        self.pos_proj   = nn.Linear(1, hidden)

        self.layers = nn.ModuleList(
            [_RelationalGNNLayer(hidden, dropout, activation) for _ in range(n_layers)])

        if gnn_pool == "attention":
            self.readout: nn.Module = _NodeAttentionPool(hidden, pool_heads)
            pooled_dim = hidden * pool_heads
        elif gnn_pool == "gem":
            self.readout = _NodeGeMPool()
            pooled_dim = hidden
        elif gnn_pool in ("mean", "max"):
            self.readout = nn.Identity()
            pooled_dim = hidden
        else:
            raise ValueError(
                f"gnn_pool must be 'mean', 'max', 'gem' or 'attention', "
                f"got {gnn_pool!r}")

        self.dense = nn.Sequential(
            nn.Linear(pooled_dim, out_dim),
            _make_activation(activation),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, 1),
        )

        # Constant scaffolding → non-persistent buffers (rebuilt at construction,
        # kept out of the state_dict so checkpoints stay small and portable).
        strand, pos = _node_meta()
        self.register_buffer("bb_template",
                             torch.from_numpy(_backbone_template()), persistent=False)
        self.register_buffer("strand_ids", torch.from_numpy(strand), persistent=False)
        self.register_buffer("pos_vec",
                             torch.from_numpy(pos).unsqueeze(-1), persistent=False)
        self.register_buffer("strength",
                             torch.from_numpy(
                                 np.ascontiguousarray(_PAIR_STRENGTH, dtype=np.float32)),
                             persistent=False)

    def forward(self, mi: torch.Tensor, ti: torch.Tensor) -> torch.Tensor:
        mi = mi.long()
        ti = ti.long()
        B  = mi.size(0)
        idx  = torch.cat([mi, ti], dim=1)                 # (B, N) nucleotide indices
        real = (idx != 4).float()                         # (B, N) 1 = real, 0 = pad

        # ── node features ────────────────────────────────────────────────────
        h = (self.nuc_emb(idx)
             + self.strand_emb(self.strand_ids).unsqueeze(0)
             + self.pos_proj(self.pos_vec).unsqueeze(0))
        h = h * real.unsqueeze(-1)                         # zero pad nodes

        # ── adjacencies (B, N, N) ────────────────────────────────────────────
        # backbone: static template gated by real nodes at both ends
        a_bb = self.bb_template.unsqueeze(0) * real.unsqueeze(1) * real.unsqueeze(2)
        # base-pair: graded strength between miRNA i and MRE j (0 at pad / mismatch)
        s = self.strength[mi[:, :, None], ti[:, None, :]]  # (B, MAX_MIRNA, MRE_LEN)
        a_pair = h.new_zeros(B, N_NODES, N_NODES)
        a_pair[:, :MAX_MIRNA, MAX_MIRNA:] = s
        a_pair[:, MAX_MIRNA:, :MAX_MIRNA] = s.transpose(1, 2)
        # row-normalise (mean aggregation); empty rows stay zero
        a_bb   = a_bb   / a_bb.sum(-1,   keepdim=True).clamp_min(1.0)
        a_pair = a_pair / a_pair.sum(-1, keepdim=True).clamp_min(1e-6)

        # ── message passing ──────────────────────────────────────────────────
        for layer in self.layers:
            h = layer(h, a_bb, a_pair)
            h = h * real.unsqueeze(-1)                     # keep pad nodes silent

        # ── readout ──────────────────────────────────────────────────────────
        if self.gnn_pool in ("attention", "gem"):
            pooled = self.readout(h, real)
        elif self.gnn_pool == "mean":
            denom  = real.sum(1, keepdim=True).clamp_min(1.0)
            pooled = (h * real.unsqueeze(-1)).sum(1) / denom
        else:  # max
            masked = h.masked_fill(~real.bool().unsqueeze(-1), float("-inf"))
            pooled = masked.max(dim=1).values

        return self.classifier(self.dense(pooled)).squeeze(-1)


# ---------------------------------------------------------------------------
# CLI → model args / checkpoint loading
# ---------------------------------------------------------------------------

def _model_args_from_cli(args: argparse.Namespace) -> dict:
    return {
        "hidden":     args.hidden,
        "n_layers":   args.n_layers,
        "dropout":    args.dropout,
        "gnn_pool":   args.gnn_pool,
        "pool_heads": args.pool_heads,
        "out_dim":    args.out_dim,
        "activation": args.activation,
    }


def _load_ckpt_model(checkpoint: str | Path,
                     device: torch.device) -> tuple[MiRNAGraphNet, dict]:
    """Load a trained GNN checkpoint into an eval-mode model."""
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
    margs = dict(ckpt["model_args"])
    for key, val in [("hidden", 128), ("n_layers", 3), ("dropout", 0.3),
                     ("gnn_pool", "mean"), ("pool_heads", 1),
                     ("out_dim", 128), ("activation", "leaky_relu")]:
        margs.setdefault(key, val)
    model = MiRNAGraphNet(**margs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Training orchestration (mirrors the CNN's single-split / k-fold runners,
# swapping in MiRNAGraphNet; reuses _train_one_run for the epoch loop)
# ---------------------------------------------------------------------------

def _run_single(args: argparse.Namespace, device: torch.device) -> None:
    if not args.val and not args.no_val:
        sys.exit("ERROR: --val is required when --folds is not set "
                 "(or pass --no-val to train on the full set without validation).")
    if args.val and args.no_val:
        print("WARNING: --no-val is set; ignoring --val and training on the full set.")

    cache = not args.no_cache
    print("Loading training data ...")
    if args.dedup:
        train_df = _dedup_pairs(_read_table(args.train),
                                args.mirna_col, args.mre_col, args.dedup)
        train_ds = MiRNAInteractionDataset.from_df(
            train_df, has_labels=True, mre_col=args.mre_col, mirna_col=args.mirna_col)
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
        val_loader = _make_loader(val_ds, args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)

    train_loader = _make_loader(train_ds, args.batch_size, shuffle=True,
                                num_workers=args.num_workers, balance=args.balance)

    model_args = _model_args_from_cli(args)
    model = MiRNAGraphNet(**model_args).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    if HAS_WANDB and getattr(args, "wandb_project", None):
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity or None,
            name=args.wandb_run_name or None, group=args.wandb_group or None,
            config={**model_args, "epochs": args.epochs, "batch_size": args.batch_size,
                    "lr": args.lr, "weight_decay": args.weight_decay})

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
    print(f"  {len(set(groups))} unique miRNA families → {args.folds} folds")

    out_path   = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_args = _model_args_from_cli(args)
    gkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    print(f"  split seed: {args.seed}")

    fold_scores: list[float] = []
    test_paths  = args.test or []
    fold_test_metrics: dict[str, list[dict]] = {Path(p).stem: [] for p in test_paths}
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
            train_df, has_labels=True, mre_col=args.mre_col, mirna_col=args.mirna_col)
        val_ds   = MiRNAInteractionDataset.from_df(
            val_df, has_labels=True, mre_col=args.mre_col, mirna_col=args.mirna_col)
        print(f"  train positives: {int(train_ds.labels.sum())} / {len(train_ds.labels)}")
        print(f"  val   positives: {int(val_ds.labels.sum())} / {len(val_ds.labels)}")

        train_loader = _make_loader(train_ds, args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, balance=args.balance)
        val_loader   = _make_loader(val_ds, args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)

        model = MiRNAGraphNet(**model_args).to(device)
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
                        "lr": args.lr, "weight_decay": args.weight_decay})

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
                    _read_table(test_path), has_labels=True,
                    mre_col=args.mre_col, mirna_col=args.mirna_col)
                test_loader = _make_loader(test_ds, args.batch_size, shuffle=False,
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
    print(f"  mean  : {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    if fold_test_metrics:
        print(f"\nTest-set summary across folds:")
        for test_name, metrics_list in fold_test_metrics.items():
            print(f"  {test_name}:")
            for metric_key in metrics_list[0]:
                vals     = [m[metric_key] for m in metrics_list]
                per_fold = "  ".join(f"{v:.4f}" for v in vals)
                print(f"    {metric_key:<20s} folds=[{per_fold}]  "
                      f"mean={np.mean(vals):.4f} ± {np.std(vals):.4f}")

        print(f"\nFold-ensemble test results ({args.folds} folds, prob. average):")
        for test_name, probs_list in ensemble_probs.items():
            if not probs_list:
                continue
            mean_probs = np.mean(np.stack(probs_list), axis=0)
            ens        = _metrics_from_probs(mean_probs, ensemble_labels[test_name])
            print(f"  {test_name}:")
            for metric_key, v in ens.items():
                indiv      = [m[metric_key] for m in fold_test_metrics[test_name]]
                mean_indiv = float(np.mean(indiv))
                print(f"    {metric_key:<20s} ensemble={v:.4f}  "
                      f"(mean-of-folds={mean_indiv:.4f}, Δ={v - mean_indiv:+.4f})")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Subcommands
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


def cmd_predict(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model, ckpt = _load_ckpt_model(args.checkpoint, device)
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch')}, "
          f"val_metrics={ckpt.get('val_metrics')})")

    has_labels = args.error_dump is not None
    df_in = _read_table(args.input)
    ds = MiRNAInteractionDataset.from_df(
        df_in, has_labels=has_labels,
        mre_col=args.mre_col, mirna_col=args.mirna_col)
    loader = _make_loader(ds, args.batch_size, shuffle=False,
                          num_workers=args.num_workers)

    logits, labels = predict_logits(model, loader, device)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= args.threshold).astype(int)

    out = df_in.copy()
    out["interaction_probability"] = probs
    out["prediction"] = preds
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(out)} rows → {out_path}  "
          f"({int(preds.sum())} predicted positive @ thr={args.threshold})")

    if args.error_dump is not None:
        if "label" not in df_in.columns:
            sys.exit("ERROR: --error-dump needs a 'label' column in --input.")
        print(f"\nTest metrics: "
              + "  ".join(f"{k}={v:.4f}" for k, v in
                          _binary_metrics(logits, labels, args.threshold).items()))
        _write_error_dump(args.error_dump, df_in, ds, probs, preds, labels)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Graph neural network for miRNA–MRE interaction classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ───────────────────────────────────────────────────────────────
    tr = sub.add_parser("train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    tr.add_argument("--train",      required=True)
    tr.add_argument("--val",        default=None)
    tr.add_argument("--no-val",     action="store_true", dest="no_val",
                    help="Train on the entire --train set with no validation "
                         "(full --epochs budget, save final-epoch weights). "
                         "Mutually exclusive with --folds.")
    tr.add_argument("--folds",      type=int, default=None)
    tr.add_argument("--seed",       type=int, default=42,
                    help="Global RNG seed: split, init, shuffling, dropout.")
    tr.add_argument("--deterministic", action="store_true",
                    help="Force cuDNN deterministic algorithms (slower).")
    tr.add_argument("--dedup",      choices=["first", "none"], default=None,
                    help="Drop duplicate (miRNA, MRE) pairs before training.")
    tr.add_argument("--family-col", default="mirna_family", dest="family_col")
    tr.add_argument("--test",       nargs="+", default=None, metavar="FILE")
    tr.add_argument("--out",        default="checkpoints/gnn_mirna.pt")
    tr.add_argument("--mre-col",    default="mre_sequence",   dest="mre_col")
    tr.add_argument("--mirna-col",  default="mirna_sequence", dest="mirna_col")
    # Architecture
    tr.add_argument("--hidden",     type=int, default=128,
                    help="Node embedding / message width.")
    tr.add_argument("--n-layers",   type=int, default=3, dest="n_layers",
                    help="Message-passing steps (how far signal propagates "
                         "between paired regions).")
    tr.add_argument("--dropout",    type=float, default=0.3,
                    help="Dropout in message-passing layers and head.")
    tr.add_argument("--gnn-pool",   choices=["mean", "max", "gem", "attention"],
                    default="mean", dest="gnn_pool",
                    help="Graph readout over nodes: mean, max, GeM (learnable "
                         "power-mean, like the CNN), or multi-head attention. "
                         "'attention' (with --pool-heads) can keep disjoint "
                         "paired regions separate; GeM is the matched baseline "
                         "for comparing against the CNN's GeM pool.")
    tr.add_argument("--pool-heads", type=int, default=1, dest="pool_heads",
                    help="Attention heads when --gnn-pool attention "
                         "(pooled width = heads * hidden).")
    tr.add_argument("--out-dim",    type=int, default=128, dest="out_dim",
                    help="Embedding size after the dense projection.")
    tr.add_argument("--activation",
                    choices=["leaky_relu", "relu", "gelu", "silu", "elu", "selu"],
                    default="leaky_relu")
    # Training (names match the CNN so _train_one_run consumes them)
    tr.add_argument("--epochs",       type=int,   default=40)
    tr.add_argument("--batch-size",   type=int,   default=256)
    tr.add_argument("--lr",           type=float, default=1e-3)
    tr.add_argument("--weight-decay", type=float, default=1e-4)
    tr.add_argument("--warmup-steps", type=int,   default=200)
    tr.add_argument("--num-workers",  type=int,   default=8)
    tr.add_argument("--patience",     type=int,   default=10)
    tr.add_argument("--ema",          action="store_true",
                    help="Track an EMA of the weights and checkpoint whichever "
                         "of raw/EMA scores higher on val.")
    tr.add_argument("--ema-decay",    type=float, default=0.999, dest="ema_decay")
    tr.add_argument("--focal-gamma",  type=float, default=0.0, dest="focal_gamma",
                    help="Focal loss gamma (0 = BCE). When >0, loss-level "
                         "pos_weight is disabled (focal handles imbalance).")
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
                         "error type, duplex stats). Requires a 'label' column.")
    pr.add_argument("--threshold",   type=float, default=0.5)
    pr.add_argument("--batch-size",  type=int,   default=256)
    pr.add_argument("--num-workers", type=int,   default=4, dest="num_workers")
    pr.add_argument("--mre-col",     default="mre_sequence",   dest="mre_col")
    pr.add_argument("--mirna-col",   default="mirna_sequence", dest="mirna_col")
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
