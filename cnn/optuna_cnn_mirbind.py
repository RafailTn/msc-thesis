#!/usr/bin/env python3
"""
Optuna hyperparameter search for MiRBindCNN.

Maximises validation AUPRC.  Uses MedianPruner to cut unpromising trials.
The study is persisted to a SQLite file so it can be resumed.

Usage
-----
    # explicit val file
    python optuna_cnn_mirbind.py \\
        --train data/train.csv --val data/val.csv \\
        --trials 100 --epochs 25 \\
        --study-name mirbind_search --storage optuna_mirbind.db

    # auto val split via StratifiedGroupKFold
    python optuna_cnn_mirbind.py \\
        --train data/train.csv \\
        --family-col noncodingRNA_fam --val-folds 5 --val-fold 0 \\
        --trials 100 --epochs 25 \\
        --study-name mirbind_search --storage optuna_mirbind.db

    # resume
    python optuna_cnn_mirbind.py \\
        --train data/train.csv --val data/val.csv \\
        --trials 50 \\
        --study-name mirbind_search --storage optuna_mirbind.db

    # time-based stopping
    python optuna_cnn_mirbind.py \\
        --train data/train.csv --val data/val.csv \\
        --timeout 3600 \\
        --study-name mirbind_search --storage optuna_mirbind.db
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from cnn_branches_mirbind import (
    MiRNAInteractionDataset,
    MiRBindCNN,
    _make_loader,
    _read_table,
    evaluate,
)


# ---------------------------------------------------------------------------
# Training loop (trial-aware)
# ---------------------------------------------------------------------------

def _train_trial(
    trial: optuna.Trial,
    model: MiRBindCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_ds: MiRNAInteractionDataset,
    hparams: dict,
    device: torch.device,
    epochs: int,
    patience: int,
    checkpoint_path: Path,
    model_args: dict,
) -> float:
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

        for mi, ti, acc, con, nbr, labels in train_loader:
            mi     = mi.to(device,     non_blocking=True)
            ti     = ti.to(device,     non_blocking=True)
            acc    = acc.to(device,    non_blocking=True)
            con    = con.to(device,    non_blocking=True)
            nbr    = nbr.to(device,    non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            logits = model(mi, ti, acc, con, nbr)
            loss   = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight)

            # Some sampled configs (e.g. GeM pooling with a large learnable p,
            # or an aggressive lr) diverge to NaN/inf.  Prune the trial instead
            # of letting the non-finite logits reach the metric functions, where
            # sklearn would raise "Input contains NaN" and fail the trial.
            if not torch.isfinite(loss):
                print(f"  trial={trial.number}  epoch={epoch:03d}: "
                      f"non-finite loss ({loss.item()}); pruning trial.")
                raise optuna.TrialPruned()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            running_loss += loss.item() * mi.size(0)
            seen += mi.size(0)

        val_metrics = evaluate(model, val_loader, device, pos_weight)
        auprc = val_metrics.get("auprc", -val_metrics["loss"])

        print(f"  trial={trial.number}  epoch={epoch:03d}/{epochs}"
              f"  train_loss={running_loss/max(seen,1):.4f}"
              f"  val_auprc={auprc:.4f}"
              f"  val_auroc={val_metrics.get('auroc', float('nan')):.4f}"
              f"  val_f1={val_metrics['f1']:.4f}")

        trial.report(auprc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if auprc > best_auprc:
            best_auprc = auprc
            patience_counter = 0
            torch.save({
                "model_state":  model.state_dict(),
                "model_args":   model_args,
                "trial":        trial.number,
                "val_auprc":    best_auprc,
            }, checkpoint_path)
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
    ckpt_dir: Path,
    seq_pairing:      str   = "multi",
    pair_embed_dim:   int   = 3,
    seq_pool:         str   = "gem",
):
    def objective(trial: optuna.Trial) -> float:
        # ── Sequence branch (2D miRBind CNN) ──────────────────────────────
        seq_filters  = trial.suggest_categorical("seq_filters",  [32, 64, 128])
        seq_dim      = trial.suggest_categorical("seq_dim",      [64, 128, 256])
        seq_dropout  = trial.suggest_float("seq_dropout", 0.1, 0.5)
        block_pool   = trial.suggest_categorical("block_pool", ["max", "gem"])
        # GeM is a power-mean and assumes non-negative input.  The global pool is
        # GeM (seq_pool below) and per-block pooling may be GeM too; the model now
        # runs Conv/Linear -> BatchNorm -> activation, so the activation output
        # (not the zero-centred BN output) is what feeds the pool.  Restrict the
        # search to (near-)non-negative activations so GeM's assumption holds --
        # leaky_relu/gelu/elu/selu would push negative values straight into GeM,
        # re-introducing the clamp-to-eps loss the BN reorder removed.  relu is
        # strictly >=0; silu has only a small bounded negative tail (~-0.28),
        # negligible next to the pre-fix ~50% BN leak.
        activation   = trial.suggest_categorical("activation", ["relu", "silu"])
        # The miRNA-axis height (30) only halves to 1 after 4 poolings, so cap
        # n_pool_blocks at 4; conv depth then ranges from that up to 8.
        n_pool_blocks = trial.suggest_int("n_pool_blocks", 2, 4)
        n_conv_blocks = trial.suggest_int("n_conv_blocks", n_pool_blocks, 8)

        # ── Training ──────────────────────────────────────────────────────
        lr           = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size   = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
        warmup_steps = trial.suggest_int("warmup_steps", 0, 500, step=50)
        balance      = trial.suggest_categorical("balance", [True, False])

        hparams = dict(
            lr=lr, weight_decay=weight_decay,
            warmup_steps=warmup_steps, balance=balance,
        )

        model_args = dict(
            seq_filters=seq_filters, seq_dim=seq_dim, seq_dropout=seq_dropout,
            n_conv_blocks=n_conv_blocks, n_pool_blocks=n_pool_blocks,
            block_pool=block_pool, activation=activation,
            # Fixed (not part of the search space, so existing studies still
            # resume); recorded so checkpoints reconstruct the right branch and
            # input channels.
            seq_pairing=seq_pairing, pair_embed_dim=pair_embed_dim,
            seq_pool=seq_pool,
        )

        model = MiRBindCNN(**model_args).to(device)

        # persistent_workers=False: this objective rebuilds loaders every trial,
        # and pruned trials abandon them mid-iteration.  Persistent workers would
        # leave a live worker iterator that the next trial's forked workers can
        # inherit -> "AssertionError: can only test a child process".
        train_loader = _make_loader(
            train_ds, batch_size, shuffle=True,
            num_workers=num_workers, balance=balance,
            persistent_workers=False)
        val_loader = _make_loader(
            val_ds, batch_size, shuffle=False,
            num_workers=num_workers, persistent_workers=False)

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"\n[Trial {trial.number}] params={n_params:.2f}M  "
              + "  ".join(f"{k}={v}" for k, v in {**model_args, **hparams}.items()))

        try:
            best_auprc = _train_trial(
                trial, model, train_loader, val_loader,
                train_ds, hparams, device, epochs, patience,
                checkpoint_path=ckpt_dir / f"trial_{trial.number}.pt",
                model_args=model_args,
            )
        finally:
            # Tear loaders (and their workers) down before the next trial forks
            # new ones, even when this trial is pruned or raises.
            del train_loader, val_loader
            gc.collect()
        return best_auprc

    return objective


# ---------------------------------------------------------------------------
# Retrain best config and evaluate on held-out test files
# ---------------------------------------------------------------------------

def _load_and_test(
    study:    optuna.Study,
    train_ds: MiRNAInteractionDataset,
    args:     argparse.Namespace,
    device:   torch.device,
) -> None:
    best_trial = study.best_trial
    ckpt_path  = Path(args.ckpt_dir) / f"trial_{best_trial.number}.pt"

    if not ckpt_path.exists():
        print(f"WARNING: checkpoint not found at {ckpt_path} — skipping test evaluation.")
        return

    print(f"\n{'='*60}")
    print(f"Loading best checkpoint: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MiRBindCNN(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  trial={ckpt['trial']}  val_auprc={ckpt['val_auprc']:.4f}")

    if args.best_model_out:
        out_path = Path(args.best_model_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(ckpt_path, out_path)
        print(f"  Copied → {out_path}")

    batch_size   = study.best_params["batch_size"]

    print(f"\nTest-set evaluation (trial #{ckpt['trial']}, val_auprc={ckpt['val_auprc']:.4f}):")
    for test_path in args.test:
        test_ds = MiRNAInteractionDataset(
            test_path, has_labels=True,
            mre_col=args.mre_col, mirna_col=args.mirna_col)
        test_loader = _make_loader(test_ds, batch_size, shuffle=False,
                                   num_workers=args.num_workers)
        metrics = evaluate(model, test_loader, device)
        print(f"  [{Path(test_path).stem}]  "
              + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    print("="*60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Optuna search for MiRBindCNN — maximises val AUPRC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train",       required=True,  help="Training CSV/TSV")
    p.add_argument("--val",         default=None,
                   help="Validation CSV/TSV. Mutually exclusive with --val-folds.")
    p.add_argument("--val-folds",   type=int, default=5, dest="val_folds",
                   help="Total folds for StratifiedGroupKFold auto-split.")
    p.add_argument("--val-fold",    type=int, default=0, dest="val_fold",
                   help="Which fold index to use as validation (0-based).")
    p.add_argument("--family-col",  default="noncodingRNA_fam", dest="family_col",
                   help="Column for StratifiedGroupKFold groups.")
    p.add_argument("--mre-col",     default="mre_sequence",   dest="mre_col")
    p.add_argument("--mirna-col",   default="mirna_sequence", dest="mirna_col")
    p.add_argument("--trials",      type=int,   default=100,
                   help="Number of Optuna trials. Ignored when --timeout is set.")
    p.add_argument("--timeout",     type=float, default=None,
                   help="Stop after this many seconds (overrides --trials).")
    p.add_argument("--epochs",      type=int,   default=25,
                   help="Epochs per trial.")
    p.add_argument("--patience",    type=int,   default=7,
                   help="Per-trial early stopping patience. 0 = disabled.")
    p.add_argument("--study-name",  default="mirbind_auprc", dest="study_name")
    p.add_argument("--storage",     default="optuna_mirbind.db",
                   help="SQLite file for persistent study.")
    p.add_argument("--num-workers", type=int,   default=8,  dest="num_workers")
    p.add_argument("--no-cache",    action="store_true", dest="no_cache",
                   help="Disable the preprocessing .cnncache.npz sidecar files.")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--best-out",    default="best_hparams_mirbind.json", dest="best_out",
                   help="JSON file for best hyperparameters.")
    p.add_argument("--startup-trials", type=int, default=10, dest="startup_trials",
                   help="Random trials before TPE kicks in.")
    p.add_argument("--pruner-warmup", type=int, default=5, dest="pruner_warmup",
                   help="Epochs before MedianPruner is allowed to prune.")
    p.add_argument("--seq-pairing", choices=["binary", "multi", "multi4", "embed"],
                   default="multi", dest="seq_pairing",
                   help="2D pairing encoding fixed for ALL trials (not tuned): "
                        "binary, multi (WC/wobble/mismatch), multi4 "
                        "(A·U/G·C/wobble/mismatch), or embed (learnable "
                        "chemistry-initialised dense embedding). Changing this "
                        "does not alter the search space, so existing studies "
                        "still resume.")
    p.add_argument("--pair-embed-dim", type=int, default=3, dest="pair_embed_dim",
                   help="Channels of the learnable pairing embedding when "
                        "--seq-pairing embed (ignored otherwise). Fixed for all "
                        "trials; not part of the search space.")
    # ── Pooling: fixed for ALL trials (like --seq-pairing), so toggling it does
    # not change the search space and existing studies resume.
    p.add_argument("--seq-pool", choices=["avg", "gem"], default="gem",
                   dest="seq_pool",
                   help="Global pooling for the 2D branch, fixed for all trials.")
    p.add_argument("--test",          nargs="+", default=None, metavar="FILE",
                   help="Test CSV/TSV files to evaluate with the best checkpoint.")
    p.add_argument("--ckpt-dir",      default=None, dest="ckpt_dir",
                   help="Directory for per-trial checkpoints. "
                        "Defaults to <storage_stem>_trials/ next to --storage.")
    p.add_argument("--best-model-out", default=None, dest="best_model_out",
                   help="Copy the best trial checkpoint here.")
    args = p.parse_args()

    if args.val and args.val_fold != 0:
        sys.exit("ERROR: --val-fold is only used when --val is not provided.")

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else \
               Path(args.storage).parent / (Path(args.storage).stem + "_trials")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Persist the resolved path so _load_and_test (which reads args.ckpt_dir)
    # finds the per-trial checkpoints even when --ckpt-dir was defaulted.
    args.ckpt_dir = str(ckpt_dir)

    device = torch.device(args.device)
    print(f"Device: {device}")

    print("Loading datasets ...")
    if args.val:
        train_ds = MiRNAInteractionDataset(
            args.train, has_labels=True,
            mre_col=args.mre_col, mirna_col=args.mirna_col, cache=not args.no_cache)
        val_ds = MiRNAInteractionDataset(
            args.val, has_labels=True,
            mre_col=args.mre_col, mirna_col=args.mirna_col, cache=not args.no_cache)
        print(f"  train={len(train_ds)}  val={len(val_ds)}")
    else:
        df = _read_table(args.train)
        if args.family_col not in df.columns:
            sys.exit(f"ERROR: --family-col '{args.family_col}' not found. "
                     f"Available: {list(df.columns)}")
        groups = df[args.family_col].fillna("unknown").astype(str).values
        labels = df["label"].values
        sgkf   = StratifiedGroupKFold(n_splits=args.val_folds)
        splits = list(sgkf.split(df, y=labels, groups=groups))
        if args.val_fold >= len(splits):
            sys.exit(f"ERROR: --val-fold {args.val_fold} out of range "
                     f"(only {len(splits)} folds).")
        train_idx, val_idx = splits[args.val_fold]
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)
        train_ds = MiRNAInteractionDataset.from_df(
            train_df, has_labels=True,
            mre_col=args.mre_col, mirna_col=args.mirna_col)
        val_ds = MiRNAInteractionDataset.from_df(
            val_df, has_labels=True,
            mre_col=args.mre_col, mirna_col=args.mirna_col)
        print(f"  StratifiedGroupKFold: fold {args.val_fold}/{args.val_folds}  "
              f"train={len(train_ds)}  val={len(val_ds)}")

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
        ckpt_dir=ckpt_dir,
        seq_pairing=args.seq_pairing,
        pair_embed_dim=args.pair_embed_dim,
        seq_pool=args.seq_pool,
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
        print(f"    {k:<25s} = {v}")

    best_out = Path(args.best_out)
    best_out.parent.mkdir(parents=True, exist_ok=True)
    with open(best_out, "w") as fh:
        json.dump({"auprc": study.best_value, **study.best_params}, fh, indent=2)
    print(f"\nBest params written → {best_out}")

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value or -1, reverse=True)
    print("\nTop-5 completed trials:")
    for t in completed[:5]:
        print(f"  #{t.number:<4d}  auprc={t.value:.4f}  "
              + "  ".join(f"{k}={v}" for k, v in t.params.items()))

    if args.test:
        _load_and_test(study, train_ds, args, device)

    return 0


if __name__ == "__main__":
    sys.exit(main())
