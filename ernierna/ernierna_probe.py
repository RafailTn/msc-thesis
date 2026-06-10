#!/usr/bin/env python3
"""
RNA foundation model linear probe for miRNA–MRE interaction classification.

The backbone (multimolecule/ernierna by default) is kept FROZEN; only a small
MLP classification head is trained on the mean-pooled sequence-pair embeddings.

ERNIE-RNA is an RNA-native BERT model pretrained on 23M non-coding RNA
sequences with secondary-structure awareness, making it more appropriate for
miRNA–MRE interaction tasks than DNA-based models.

Workflow
--------
1. embed  – pass a raw CSV through the backbone and save (embeddings, labels,
             groups) to a compressed .npz file.
2. train  – load a .npz produced by `embed` and train the MLP head.
3. predict – load a pre-extracted .npz (or raw CSV) plus a head checkpoint
             and write per-sample interaction probabilities.

Sequence encoding
-----------------
Both sequences are RNA-normalised (T→U) and fed as a BERT pair:
    [CLS] mirna_seq [SEP] mre_seq [SEP]
Mean pooling over all non-padding token embeddings is used as the pair
representation (768 dims for ERNIE-RNA).

Embeddings are stored as float16 in the .npz to halve disk usage; they are
cast back to float32 at training time.

Usage
-----
  # Step 1 – extract embeddings once (GPU recommended)
  python ernierna/ernierna_probe.py embed \\
      --input  data/manakov_train_cnn.csv \\
      --output data/train_emb.npz \\
      --batch-size 128

  python ernierna/ernierna_probe.py embed \\
      --input  data/manakov_test_cnn.csv \\
      --output data/test_emb.npz \\
      --batch-size 128

  # Step 2 – train MLP head on pre-extracted embeddings
  python ernierna/ernierna_probe.py train \\
      --train data/train_emb.npz \\
      --val   data/test_emb.npz \\
      --out   checkpoints/rna_probe.pt

  # Step 2 (alternative) – k-fold CV from a single .npz
  python ernierna/ernierna_probe.py train \\
      --train data/train_emb.npz \\
      --folds 5 \\
      --out   checkpoints/rna_probe.pt

  # Step 3 – predict
  python ernierna/ernierna_probe.py predict \\
      --checkpoint checkpoints/rna_probe.pt \\
      --input      data/test_emb.npz \\
      --output     predictions.tsv

  # Sanity-check baseline: randomly initialised backbone
  python ernierna/ernierna_probe.py embed \\
      --input data/manakov_train_cnn.csv \\
      --output data/train_emb_random.npz \\
      --random-init

  # Use a different backbone (e.g. RNA-FM)
  python ernierna/ernierna_probe.py embed \\
      --input  data/manakov_train_cnn.csv \\
      --output data/train_emb_rnafm.npz \\
      --model  multimolecule/rnafm
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

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

DEFAULT_MODEL = "multimolecule/ernierna"
EMBED_DIM     = 768   # ERNIE-RNA hidden size (also 768 for RNABERT, RNA-MSM, BiRNA-BERT)
MAX_LENGTH    = 128   # miRNA ~22 nt + MRE 50 nt + special tokens << 128


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _norm_rna(seq: str) -> str:
    """Normalise DNA/RNA to uppercase RNA (T → U) for RNA-native models."""
    return seq.upper().replace("T", "U")


def _read_table(path: str | Path) -> "pd.DataFrame":
    sep = "\t" if str(path).endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)


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
# Backbone loader
# ---------------------------------------------------------------------------

def load_backbone(model_name: str = DEFAULT_MODEL,
                  random_init: bool = False,
                  device: torch.device = torch.device("cpu")):
    """Load tokenizer and frozen RNA backbone (default: multimolecule/ernierna)."""
    if not HAS_TRANSFORMERS:
        sys.exit("ERROR: transformers not found. pip install transformers")

    # multimolecule registers RnaTokenizer / ErnieRnaModel etc. into the HF
    # Auto registries on import; without this, AutoTokenizer can't resolve them.
    try:
        import multimolecule  # noqa: F401
    except ImportError:
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if random_init:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model  = AutoModel.from_config(config)
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  backbone: {n_params:.1f}M parameters (frozen)")
    return tokenizer, model.to(device)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    mirna_seqs: list[str],
    mre_seqs:   list[str],
    tokenizer,
    backbone:   nn.Module,
    device:     torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Extract mean-pooled RNA backbone embeddings for miRNA-MRE sequence pairs.

    Sequences are DNA-normalised to RNA (T→U) and fed as a BERT pair:
        [CLS] mirna_seq [SEP] mre_seq [SEP]

    Mean pooling is applied over all non-padding token positions.
    Returns float32 array of shape (N, EMBED_DIM).
    """
    n = len(mirna_seqs)
    embeddings = np.zeros((n, EMBED_DIM), dtype=np.float32)

    mirna_norm = [_norm_rna(s) for s in mirna_seqs]
    mre_norm   = [_norm_rna(s) for s in mre_seqs]

    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        batch_mirna = mirna_norm[start:end]
        batch_mre   = mre_norm[start:end]

        encoding = tokenizer(
            batch_mirna,
            batch_mre,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden  = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state

        # Mean pool over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden    = (hidden * mask_expanded).sum(dim=1)
        count         = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled        = (sum_hidden / count).cpu().float().numpy()

        embeddings[start:end] = pooled
        print(f"\r  {end}/{n}", end="", file=sys.stderr)

    print("", file=sys.stderr)
    return embeddings


# ---------------------------------------------------------------------------
# MLP classification head
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """
    Small MLP trained on top of frozen RNA backbone mean-pooled embeddings.

    Architecture: in_dim → hidden → hidden//2 → 1 (binary logit)
    """

    def __init__(self, in_dim: int = EMBED_DIM, hidden: int = 256,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# DataLoader helper
# ---------------------------------------------------------------------------

def _make_tensor_loader(
    embeddings: np.ndarray,
    labels:     np.ndarray,
    batch_size: int,
    shuffle:    bool,
    balance:    bool = False,
) -> DataLoader:
    X  = torch.from_numpy(embeddings).float()
    y  = torch.from_numpy(labels).long()
    ds = TensorDataset(X, y)

    sampler = None
    if balance and shuffle:
        counts  = np.bincount(labels)
        weights = 1.0 / counts[labels]
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights).double(),
            num_samples=len(weights), replacement=True)
        shuffle = False

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      sampler=sampler, pin_memory=True,
                      drop_last=(shuffle and sampler is None))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:      MLPHead,
    loader:     DataLoader,
    device:     torch.device,
    pos_weight: Optional[torch.Tensor] = None,
) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    total_loss, n_batches  = 0.0, 0

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        logits = model(X)
        loss   = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        total_loss += loss.item()
        n_batches  += 1
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.cpu().numpy().astype(int))

    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels).astype(int)
    metrics   = _binary_metrics(logits_np, labels_np)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_head(
    train_emb:    np.ndarray,
    train_labels: np.ndarray,
    val_emb:      np.ndarray,
    val_labels:   np.ndarray,
    args:         argparse.Namespace,
    device:       torch.device,
    out_path:     Path,
) -> float:
    model = MLPHead(hidden=args.hidden, dropout=args.dropout).to(device)
    n_head_params = sum(p.numel() for p in model.parameters())
    print(f"  head parameters: {n_head_params:,}")

    train_loader = _make_tensor_loader(
        train_emb, train_labels, args.batch_size, shuffle=True, balance=args.balance)
    val_loader   = _make_tensor_loader(
        val_emb,   val_labels,   args.batch_size, shuffle=False)

    pos_weight: Optional[torch.Tensor] = None
    if not args.balance:
        n_pos = int(train_labels.sum())
        n_neg = len(train_labels) - n_pos
        if n_pos > 0 and n_neg > 0:
            pw         = n_neg / n_pos
            pos_weight = torch.tensor([pw], device=device)
            print(f"  BCEWithLogitsLoss pos_weight = {pw:.3f}")

    optim        = torch.optim.AdamW(
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

    best_val         = -float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0           = time.time()
        running_loss = 0.0
        seen         = 0
        train_logits_buf, train_labels_buf = [], []

        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()

            logits = model(X)
            loss   = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pos_weight)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            running_loss += loss.item() * X.size(0)
            seen         += X.size(0)
            train_logits_buf.append(logits.detach().cpu().numpy())
            train_labels_buf.append(y.cpu().numpy().astype(int))

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
                "epoch":          epoch,
                "lr":             sched.get_last_lr()[0],
                "train/loss":     train_loss,
                "train/f1":       train_metrics["f1"],
                "train/accuracy": train_metrics["accuracy"],
                "val/loss":       val_metrics["loss"],
                "val/f1":         val_metrics["f1"],
                "val/accuracy":   val_metrics["accuracy"],
            }
            for split, mets in (("train", train_metrics), ("val", val_metrics)):
                if "auroc" in mets:
                    log_dict[f"{split}/auroc"] = mets["auroc"]
                if "auprc" in mets:
                    log_dict[f"{split}/auprc"] = mets["auprc"]
            wandb.log(log_dict)

        if improved:
            best_val         = ckpt_val
            patience_counter = 0
            torch.save({
                "head_state":     model.state_dict(),
                "head_args":      {"hidden": args.hidden, "dropout": args.dropout},
                "val_metrics":    val_metrics,
                "epoch":          epoch,
                "backbone_model": DEFAULT_MODEL,
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


# ---------------------------------------------------------------------------
# .npz I/O
# ---------------------------------------------------------------------------

def _load_npz(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return (data["embeddings"].astype(np.float32),
            data["labels"].astype(np.int64),
            data["groups"].astype(str))


# ---------------------------------------------------------------------------
# Subcommand: embed
# ---------------------------------------------------------------------------

def cmd_embed(args: argparse.Namespace) -> None:
    if not HAS_PANDAS:
        sys.exit("ERROR: pandas not found. pip install pandas")

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Backbone: {args.model}  (random_init={args.random_init})")
    tokenizer, backbone = load_backbone(args.model, args.random_init, device)

    df = _read_table(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    mirna_seqs = df[args.mirna_col].astype(str).tolist()
    mre_seqs   = df[args.mre_col].astype(str).tolist()

    labels = (df["label"].astype(int).values
              if "label" in df.columns
              else np.full(len(df), -1, dtype=np.int64))
    groups = (df[args.family_col].fillna("unknown").astype(str).values
              if args.family_col in df.columns
              else np.array(["unknown"] * len(df)))

    print(f"Extracting embeddings (batch_size={args.batch_size}) ...")
    embeddings = extract_embeddings(
        mirna_seqs, mre_seqs, tokenizer, backbone, device, args.batch_size)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Store as float16 (~3.5 GB for 2.5 M × 768 vs. ~7 GB float32)
    np.savez_compressed(out_path,
                        embeddings=embeddings.astype(np.float16),
                        labels=labels,
                        groups=groups)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved {out_path}  shape={embeddings.shape}  size={size_mb:.0f} MB")


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    device   = torch.device(args.device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    if args.folds:
        if not HAS_SKLEARN:
            sys.exit("ERROR: scikit-learn is required for --folds.")

        print(f"Loading embeddings for {args.folds}-fold cross-validation ...")
        emb, labels, groups = _load_npz(args.train)
        n_groups = len(np.unique(groups))
        print(f"  {len(emb)} samples  |  {n_groups} families  |  "
              f"pos={int(labels.sum())}")

        gkf         = StratifiedGroupKFold(n_splits=args.folds)
        fold_scores: list[float] = []

        for fold, (train_idx, val_idx) in enumerate(
                gkf.split(emb, y=labels, groups=groups), 1):
            val_families = sorted(set(groups[val_idx]))
            print(f"\n{'='*60}")
            print(f"Fold {fold}/{args.folds}  "
                  f"train={len(train_idx)}  val={len(val_idx)}")
            preview = val_families[:8]
            suffix  = " ..." if len(val_families) > 8 else ""
            print(f"  val families ({len(val_families)}): {preview}{suffix}")
            print(f"{'='*60}")
            print(f"  train positives: {int(labels[train_idx].sum())} / {len(train_idx)}")
            print(f"  val   positives: {int(labels[val_idx].sum())} / {len(val_idx)}")

            if HAS_WANDB and getattr(args, "wandb_project", None):
                _group = args.wandb_group or args.wandb_run_name or out_path.stem
                _name  = (f"{args.wandb_run_name}_fold{fold}"
                          if args.wandb_run_name else f"{out_path.stem}_fold{fold}")
                wandb.init(
                    project=args.wandb_project, entity=args.wandb_entity or None,
                    name=_name, group=_group,
                    config={"fold": fold, "n_folds": args.folds,
                            "hidden": args.hidden, "dropout": args.dropout,
                            "epochs": args.epochs, "lr": args.lr,
                            "weight_decay": args.weight_decay},
                )

            fold_out = out_path.parent / f"{out_path.stem}_fold{fold}{out_path.suffix}"
            score    = _train_head(
                emb[train_idx], labels[train_idx],
                emb[val_idx],   labels[val_idx],
                args, device, fold_out)
            fold_scores.append(score)

            if HAS_WANDB and wandb.run is not None:
                wandb.finish()

        print(f"\n{'='*60}")
        print(f"{args.folds}-fold CV results ({args.checkpoint_metric}):")
        for k, s in enumerate(fold_scores, 1):
            print(f"  fold {k}: {s:.4f}")
        mean = float(np.mean(fold_scores))
        std  = float(np.std(fold_scores))
        print(f"  mean  : {mean:.4f} ± {std:.4f}")
        print(f"{'='*60}")

    else:
        if not args.val:
            sys.exit("ERROR: --val is required when --folds is not set.")

        print("Loading training embeddings ...")
        train_emb, train_labels, _ = _load_npz(args.train)
        print(f"  {len(train_emb)} samples  |  pos={int(train_labels.sum())}")

        print("Loading validation embeddings ...")
        val_emb, val_labels, _     = _load_npz(args.val)
        print(f"  {len(val_emb)} samples  |  pos={int(val_labels.sum())}")

        if HAS_WANDB and getattr(args, "wandb_project", None):
            wandb.init(
                project=args.wandb_project, entity=args.wandb_entity or None,
                name=args.wandb_run_name or None, group=args.wandb_group or None,
                config={"hidden": args.hidden, "dropout": args.dropout,
                        "epochs": args.epochs, "lr": args.lr,
                        "weight_decay": args.weight_decay},
            )

        _train_head(train_emb, train_labels, val_emb, val_labels,
                    args, device, out_path)

        if HAS_WANDB and wandb.run is not None:
            wandb.finish()


# ---------------------------------------------------------------------------
# Subcommand: predict
# ---------------------------------------------------------------------------

def cmd_predict(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    head_args = ckpt["head_args"]
    model     = MLPHead(**head_args).to(device)
    model.load_state_dict(ckpt["head_state"])
    model.eval()
    print(f"Loaded checkpoint  epoch={ckpt.get('epoch')}  "
          f"val_metrics={ckpt.get('val_metrics')}")

    p = Path(args.input)
    if p.suffix == ".npz":
        emb, labels, _ = _load_npz(args.input)
    else:
        # Raw CSV: extract on the fly
        if not HAS_PANDAS:
            sys.exit("ERROR: pandas not found. pip install pandas")
        model_name = ckpt.get("backbone_model", ckpt.get("dnabert2_model", DEFAULT_MODEL))
        print(f"Extracting embeddings with {model_name} ...")
        tokenizer, backbone = load_backbone(model_name, device=device)
        df         = _read_table(args.input)
        mirna_seqs = df[args.mirna_col].astype(str).tolist()
        mre_seqs   = df[args.mre_col].astype(str).tolist()
        emb        = extract_embeddings(mirna_seqs, mre_seqs, tokenizer, backbone,
                                        device, args.batch_size)
        labels     = (df["label"].astype(int).values
                      if "label" in df.columns
                      else np.full(len(df), -1, dtype=np.int64))

    loader = DataLoader(
        TensorDataset(torch.from_numpy(emb).float()),
        batch_size=args.batch_size, shuffle=False)

    all_probs: list[float] = []
    with torch.no_grad():
        for (X,) in loader:
            X = X.to(device)
            probs = torch.sigmoid(model(X)).cpu().numpy()
            all_probs.extend(probs.tolist())

    all_probs_np = np.array(all_probs)
    all_preds    = (all_probs_np >= args.threshold).astype(int)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if p.suffix == ".npz" or not HAS_PANDAS:
        lines = ["interaction_probability\tprediction\tlabel"]
        lines += [f"{prob:.6f}\t{pred}\t{lbl}"
                  for prob, pred, lbl in zip(all_probs_np, all_preds, labels)]
        out_path.write_text("\n".join(lines) + "\n")
    else:
        df["interaction_probability"] = all_probs_np
        df["prediction"]              = all_preds
        df.to_csv(out_path, sep="\t", index=False)

    print(f"Wrote {len(all_probs)} rows → {out_path}")

    valid = labels >= 0
    if HAS_SKLEARN and len(np.unique(labels[valid])) == 2:
        logits_np = np.log(all_probs_np[valid] / (1.0 - all_probs_np[valid] + 1e-9))
        metrics   = _binary_metrics(logits_np, labels[valid], args.threshold)
        print("\nTest-set metrics:")
        for k, v in metrics.items():
            print(f"  {k:20s} = {v:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="RNA foundation model linear probe for miRNA–MRE interaction classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    _device_default = "cuda" if torch.cuda.is_available() else "cpu"

    # ── embed ──────────────────────────────────────────────────────────────────
    em = sub.add_parser("embed",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                        help="Extract frozen RNA backbone embeddings from a CSV.")
    em.add_argument("--input",      required=True,
                    help="Input CSV/TSV with sequence columns")
    em.add_argument("--output",     required=True,
                    help="Output .npz file (float16 embeddings + labels + groups)")
    em.add_argument("--model",      default=DEFAULT_MODEL,
                    help="HuggingFace model ID or local path")
    em.add_argument("--mre-col",    default="gene",             dest="mre_col",
                    help="Column name for MRE/target sequence")
    em.add_argument("--mirna-col",  default="noncodingRNA",     dest="mirna_col",
                    help="Column name for miRNA/query sequence")
    em.add_argument("--family-col", default="noncodingRNA_fam", dest="family_col",
                    help="Column used as group key for k-fold splits")
    em.add_argument("--batch-size", type=int, default=64,  dest="batch_size",
                    help="Sequences per forward pass through the backbone")
    em.add_argument("--random-init", action="store_true", dest="random_init",
                    help="Randomly initialise backbone weights (control baseline)")
    em.add_argument("--device", default=_device_default)

    # ── train ──────────────────────────────────────────────────────────────────
    tr = sub.add_parser("train",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                        help="Train MLP head on pre-extracted .npz embeddings.")
    tr.add_argument("--train",   required=True, metavar="NPZ",
                    help=".npz produced by `embed` (used for training, or entire "
                         "dataset when --folds is set)")
    tr.add_argument("--val",     default=None, metavar="NPZ",
                    help=".npz for validation (required unless --folds is set)")
    tr.add_argument("--folds",   type=int, default=None,
                    help="Run stratified-group k-fold CV instead of train/val split")
    tr.add_argument("--out",     default="checkpoints/rna_probe.pt",
                    help="Output checkpoint path (.pt)")
    # Head architecture
    tr.add_argument("--hidden",  type=int,   default=256,
                    help="Hidden size of the MLP head")
    tr.add_argument("--dropout", type=float, default=0.3)
    # Training
    tr.add_argument("--epochs",           type=int,   default=30)
    tr.add_argument("--batch-size",       type=int,   default=512,  dest="batch_size")
    tr.add_argument("--lr",               type=float, default=3e-4)
    tr.add_argument("--weight-decay",     type=float, default=1e-4, dest="weight_decay")
    tr.add_argument("--warmup-steps",     type=int,   default=200,  dest="warmup_steps")
    tr.add_argument("--patience",         type=int,   default=10)
    tr.add_argument("--balance",          action="store_true",
                    help="Use WeightedRandomSampler to balance classes each epoch")
    tr.add_argument("--checkpoint-metric",
                    choices=["auroc", "auprc", "f1", "accuracy"], default="auprc",
                    dest="checkpoint_metric")
    tr.add_argument("--wandb-project",    default=None, dest="wandb_project")
    tr.add_argument("--wandb-entity",     default=None, dest="wandb_entity")
    tr.add_argument("--wandb-run-name",   default=None, dest="wandb_run_name")
    tr.add_argument("--wandb-group",      default=None, dest="wandb_group")
    tr.add_argument("--device", default=_device_default)

    # ── predict ────────────────────────────────────────────────────────────────
    pr = sub.add_parser("predict",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                        help="Predict with a saved head checkpoint.")
    pr.add_argument("--checkpoint", required=True)
    pr.add_argument("--input",      required=True,
                    help=".npz (pre-extracted) or raw .csv/.tsv "
                         "(backbone will be loaded for on-the-fly extraction)")
    pr.add_argument("--output",     required=True,
                    help="Output TSV with interaction_probability and prediction columns")
    pr.add_argument("--threshold",  type=float, default=0.5)
    pr.add_argument("--batch-size", type=int,   default=512, dest="batch_size")
    pr.add_argument("--mre-col",    default="gene",         dest="mre_col")
    pr.add_argument("--mirna-col",  default="noncodingRNA", dest="mirna_col")
    pr.add_argument("--device", default=_device_default)

    args = parser.parse_args()
    {
        "embed":   cmd_embed,
        "train":   cmd_train,
        "predict": cmd_predict,
    }[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
