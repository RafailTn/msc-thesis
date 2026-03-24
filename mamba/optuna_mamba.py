import torch
import polars as ps
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from mamba_ssm import Mamba2
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import AveragePrecision, AUROC, MatthewsCorrCoef
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from utils import OneHotDataset, collate_fn_onehot, DnaOneHotEncoder, AttentionPool
from mamba_imp import MambaBlock, MambaDNA, MambaDNALightning 
from dotenv import load_dotenv

load_dotenv()

# ── constants ────────────────────────────────────────────────────────────────
DATA_PATH  = '/home/adam/adam/data/AGO2eCLIPManakov2022trainimprovedwfeatures.csv'
N_TRIALS   = 50
N_EPOCHS   = 15   # shorter than full training to keep search fast
MONITOR    = 'val_ap'

# ── data (load once, reuse across all trials) ────────────────────────────────
def load_data():
    df = ps.read_csv(DATA_PATH,
                     columns=['mre_sequence', 'mirna_sequence', 'mir_fam', 'label'])
    df = df.unique(subset=['mre_sequence', 'mirna_sequence'], keep='none')
    # Use a single held-out fold (fold 0) for the search — fast and consistent
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(sgkf.split(df, df['label'], groups=df['mir_fam']))
    return df[train_idx], df[val_idx]


# ── objective ────────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial, train_df, val_df) -> float:
    # ── suggest hyperparameters ──────────────────────────────────────────────
    d_model    = trial.suggest_categorical('d_model',    [32, 64, 128])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout    = trial.suggest_float('dropout',  0.1, 0.3, step=0.05)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    # ── dataloaders ──────────────────────────────────────────────────────────
    train_loader = DataLoader(OneHotDataset(train_df), batch_size=256,
                              collate_fn=collate_fn_onehot, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(OneHotDataset(val_df),   batch_size=256,
                              collate_fn=collate_fn_onehot, shuffle=False, num_workers=4)

    # ── model & trainer ──────────────────────────────────────────────────────
    model = MambaDNALightning(d_model=d_model, num_layers=num_layers, dropout=dropout, learning_rate=learning_rate, weight_decay=weight_decay)

    callbacks = [
        # Report val_ap to Optuna each epoch and prune underperforming trials early
        PyTorchLightningPruningCallback(trial, monitor=MONITOR),
        EarlyStopping(monitor=MONITOR, patience=5, mode='max'),
    ]

    trainer = pl.Trainer(
        max_epochs=N_EPOCHS,
        callbacks=callbacks,
        accelerator='auto',
        precision='16-mixed',
        enable_progress_bar=False,   # cleaner output when running many trials
        enable_model_summary=False,
        logger=False,                # no per-trial logging overhead
    )
    trainer.fit(model, train_loader, val_loader)

    # Return the best val_ap seen during this trial
    return trainer.callback_metrics[MONITOR].item()


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    train_df, val_df = load_data()

    # MedianPruner stops trials whose intermediate val_ap is below the median
    # of completed trials at the same epoch — cuts wasted compute significantly
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    study = optuna.create_study(
        direction='maximize',
        pruner=pruner,
        study_name='mamba-dna-hpo_add',
        storage='sqlite:///mamba_hpo_add.db',   # persists results; safe to Ctrl-C and resume
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, train_df, val_df),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print("\n─── Best trial ───────────────────────────────")
    print(f"  val_ap     : {study.best_value:.4f}")
    print(f"  d_model    : {study.best_params['d_model']}")
    print(f"  num_layers : {study.best_params['num_layers']}")
    print(f"  dropout    : {study.best_params['dropout']:.2f}")
    print(f"  learning_rate    : {study.best_params['learning_rate']:.2f}")
    print(f"  weight_decay    : {study.best_params['weight_decay']:.2f}")

if __name__ == "__main__":
    main()
