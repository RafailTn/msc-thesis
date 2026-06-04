#!/usr/bin/env python3
"""
Optuna hyperparameter search for ThreeBranchCNN.

Maximises validation Average Precision Score (AUPRC).
Uses Optuna's MedianPruner to cut unpromising trials early.
The study is persisted to a SQLite file so it can be resumed.

Usage
-----
    # start a new search (single train/val split)
    python optuna_cnn.py \\
        --train data/train.csv --val data/val.csv \\
        --trials 100 --epochs 25 \\
        --study-name cnn_search --storage optuna_cnn.db

    # resume (same --study-name + --storage)
    python optuna_cnn.py \\
        --train data/train.csv --val data/val.csv \\
        --trials 50 \\
        --study-name cnn_search --storage optuna_cnn.db

    # time-based stopping instead of trial count
    python optuna_cnn.py \\
        --train data/train.csv --val data/val.csv \\
        --timeout 3600 \\
        --study-name cnn_search --storage optuna_cnn.db
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Import reusable pieces from the sibling module.
sys.path.insert(0, str(Path(__file__).parent))
from cnn_branches import (
    MiRNAInteractionDataset,
    ThreeBranchCNN,
    _make_loader,
    evaluate,
)


# ---------------------------------------------------------------------------
# Training loop (trial-aware)
# ---------------------------------------------------------------------------

def _train_trial(
    trial: optuna.Trial,
    model: ThreeBranchCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_ds: MiRNAInteractionDataset,
    hparams: dict,
    device: torch.device,
    epochs: int,
    patience: int,
) -> float:
    """Train for *epochs* epochs, report AUPRC to Optuna each epoch.

    Returns the best AUPRC achieved.  Raises TrialPruned if Optuna decides
    the trial is unpromising.
    """
    pos_weight = None
    if not hparams["balance"]:
        n_pos = int(train_ds.labels.sum())
        n_neg = len(train_ds.labels) - n_pos
        if n_pos > 0 and n_neg > 0:
            pos_weight = torch.tensor([n_neg / n_pos], device=device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
    )
    total_steps  = epochs * max(1, len(train_loader))
    warmup_steps = min(hparams["warmup_steps"], total_steps // 10)

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
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=total_steps)

    best_auprc       = -float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

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

        val_metrics = evaluate(model, val_loader, device, pos_weight)
        auprc = val_metrics.get("auprc", -val_metrics["loss"])

        print(f"  trial={trial.number}  epoch={epoch:03d}/{epochs}"
              f"  train_loss={running_loss/max(seen,1):.4f}"
              f"  val_auprc={auprc:.4f}"
              f"  val_auroc={val_metrics.get('auroc', float('nan')):.4f}"
              f"  val_f1={val_metrics['f1']:.4f}")

        # Report to Optuna and check for pruning
        trial.report(auprc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if auprc > best_auprc:
            best_auprc = auprc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience > 0 and patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

    return best_auprc


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(
    train_ds: MiRNAInteractionDataset,
    val_ds:   MiRNAInteractionDataset,
    device:   torch.device,
    epochs:   int,
    patience: int,
    num_workers: int,
):
    def objective(trial: optuna.Trial) -> float:
        # ── Architecture ──────────────────────────────────────────────────
        seq_channels = trial.suggest_categorical("seq_channels", [64, 128, 256])
        seq_blocks   = trial.suggest_int("seq_blocks", 3, 8)
        vec_channels = trial.suggest_categorical("vec_channels", [32, 64, 128])
        vec_blocks   = trial.suggest_int("vec_blocks", 2, 5)
        energy_dim   = trial.suggest_categorical("energy_dim", [32, 64, 128])
        kernel_size  = trial.suggest_categorical("kernel_size", [3, 5, 7, 9])
        dropout      = trial.suggest_float("dropout", 0.05, 0.4)
        norm         = trial.suggest_categorical("norm", ["batch", "layer"])
        # num_heads must divide seq_channels; choose from valid divisors
        num_heads    = trial.suggest_categorical(
            "num_heads", [h for h in [2, 4, 8] if seq_channels % h == 0])

        # ── Training ──────────────────────────────────────────────────────
        lr            = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay  = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        batch_size    = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        warmup_steps  = trial.suggest_int("warmup_steps", 0, 500, step=50)
        balance       = trial.suggest_categorical("balance", [True, False])

        hparams = dict(
            lr=lr, weight_decay=weight_decay,
            warmup_steps=warmup_steps, balance=balance,
        )

        model_args = dict(
            seq_channels=seq_channels, seq_blocks=seq_blocks,
            vec_channels=vec_channels, vec_blocks=vec_blocks,
            energy_dim=energy_dim, kernel_size=kernel_size,
            dropout=dropout, norm=norm, num_heads=num_heads,
        )

        model = ThreeBranchCNN(**model_args).to(device)

        train_loader = _make_loader(
            train_ds, batch_size, shuffle=True,
            num_workers=num_workers, balance=balance)
        val_loader = _make_loader(
            val_ds, batch_size, shuffle=False,
            num_workers=num_workers)

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"\n[Trial {trial.number}] params={n_params:.2f}M  "
              + "  ".join(f"{k}={v}" for k, v in {**model_args, **hparams}.items()))

        best_auprc = _train_trial(
            trial, model, train_loader, val_loader,
            train_ds, hparams, device, epochs, patience,
        )
        return best_auprc

    return objective


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Optuna search for ThreeBranchCNN — maximises val AUPRC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train",       required=True,  help="Training CSV/TSV")
    p.add_argument("--val",         required=True,  help="Validation CSV/TSV")
    p.add_argument("--mre-col",     default="mre_sequence",   dest="mre_col")
    p.add_argument("--mirna-col",   default="mirna_sequence", dest="mirna_col")
    p.add_argument("--trials",      type=int,   default=100,
                   help="Number of Optuna trials. Ignored when --timeout is set.")
    p.add_argument("--timeout",     type=float, default=None,
                   help="Stop search after this many seconds (overrides --trials).")
    p.add_argument("--epochs",      type=int,   default=25,
                   help="Epochs per trial (keep shorter than full training).")
    p.add_argument("--patience",    type=int,   default=7,
                   help="Per-trial early stopping patience. 0 = disabled.")
    p.add_argument("--study-name",  default="cnn_auprc",  dest="study_name")
    p.add_argument("--storage",     default="optuna_cnn.db",
                   help="SQLite file for persistent study (relative to cwd).")
    p.add_argument("--num-workers", type=int,   default=2,  dest="num_workers")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--best-out",    default="best_hparams.json", dest="best_out",
                   help="JSON file to write best hyperparameters.")
    p.add_argument("--startup-trials", type=int, default=10, dest="startup_trials",
                   help="Random trials before TPE kicks in.")
    p.add_argument("--pruner-warmup", type=int, default=5, dest="pruner_warmup",
                   help="Epochs before MedianPruner is allowed to prune.")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    print("Loading datasets ...")
    train_ds = MiRNAInteractionDataset(
        args.train, energy_stats=None, has_labels=True,
        mre_col=args.mre_col, mirna_col=args.mirna_col)
    val_ds = MiRNAInteractionDataset(
        args.val, energy_stats=train_ds.energy_stats, has_labels=True,
        mre_col=args.mre_col, mirna_col=args.mirna_col)
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    storage_url = f"sqlite:///{args.storage}"
    sampler = TPESampler(n_startup_trials=args.startup_trials, seed=42)
    pruner  = MedianPruner(
        n_startup_trials=args.startup_trials,
        n_warmup_steps=args.pruner_warmup,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    print(f"Study '{args.study_name}' loaded — "
          f"{len(study.trials)} existing trial(s).")

    objective = make_objective(
        train_ds, val_ds, device,
        epochs=args.epochs,
        patience=args.patience,
        num_workers=args.num_workers,
    )

    study.optimize(
        objective,
        n_trials=args.trials if args.timeout is None else None,
        timeout=args.timeout,
        catch=(Exception,),
    )

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"  AUPRC = {study.best_value:.4f}")
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k:<20s} = {v}")

    best_out = Path(args.best_out)
    best_out.parent.mkdir(parents=True, exist_ok=True)
    with open(best_out, "w") as fh:
        json.dump({"auprc": study.best_value, **study.best_params}, fh, indent=2)
    print(f"\nBest params written → {best_out}")

    # Top-5 trials
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value or -1, reverse=True)
    print("\nTop-5 completed trials:")
    for t in completed[:5]:
        print(f"  #{t.number:<4d}  auprc={t.value:.4f}  "
              + "  ".join(f"{k}={v}" for k, v in t.params.items()))

    return 0


if __name__ == "__main__":
    sys.exit(main())
