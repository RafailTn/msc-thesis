import torch
import polars as ps
from mamba_ssm import Mamba2
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import AveragePrecision, AUROC, MatthewsCorrCoef
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────
# Dataset & collate  (replaces OneHotDataset)
# ─────────────────────────────────────────────

def separate_cols_chim(df):
    return df['mre_sequence'], df['mirna_sequence'], df['label']


class SeparatedIndexDataset(Dataset):
    """
    Returns MRE and miRNA as separate integer index tensors.
    A=0, C=1, G=2, T/U=3, unknown=4
    Padding token index is 5 (used in collate).
    """
    VOCAB = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}

    def __init__(self, df):
        df = df.to_pandas().reset_index(drop=True)
        self.seqs_mre, self.seqs_mirna, self.labels = separate_cols_chim(df)

    def encode(self, seq: str) -> torch.Tensor:
        return torch.tensor(
            [self.VOCAB.get(c, 4) for c in seq], dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mr = self.encode(self.seqs_mre[idx])
        mi = self.encode(self.seqs_mirna[idx])
        y  = torch.tensor(self.labels[idx], dtype=torch.float32)
        return mr, mi, y


def collate_fn_separated(batch):
    """Pads MRE and miRNA separately. Padding token = 5."""
    mrs, mis, labels = zip(*batch)
    mrs    = pad_sequence(mrs, batch_first=True, padding_value=5)   # (B, L_mr)
    mis    = pad_sequence(mis, batch_first=True, padding_value=5)   # (B, L_mi)
    labels = torch.tensor(labels, dtype=torch.float32)
    return (mrs, mis), labels


# ─────────────────────────────────────────────
# RC helper
# ─────────────────────────────────────────────

def get_rc(indices: torch.Tensor) -> torch.Tensor:
    """
    Reverse complement for integer-encoded DNA.
    A(0)↔T(3),  C(1)↔G(2),  unknown(4) and padding(5) stay unchanged.
    Returns the sequence reversed along dim=1.
    """
    rc = indices.clone()
    rc[indices == 0] = 3
    rc[indices == 3] = 0
    rc[indices == 1] = 2
    rc[indices == 2] = 1
    return torch.flip(rc, dims=[1])


# ─────────────────────────────────────────────
# Model building blocks
# ─────────────────────────────────────────────

class AttentionPool(nn.Module):
    """
    Learned weighted sum over sequence positions.
    Ignores padding positions via an explicit boolean mask.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x:    (B, L, D)
        # mask: (B, L) bool — True marks padding positions
        scores = self.attn(x)                                       # (B, L, 1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1), float('-inf'))
        weights = torch.softmax(scores, dim=1)                      # (B, L, 1)
        return (weights * x).sum(dim=1)                             # (B, D)


class MambaBlock(nn.Module):
    """
    Single residual Mamba2 block: pre-LayerNorm → Mamba2 → residual → dropout.
    """
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float):
        super().__init__()
        self.norm    = nn.LayerNorm(d_model)
        self.mamba   = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mamba(self.norm(x)))


class MambaDNA(nn.Module):
    """
    Stacked Mamba2 encoder with attention pooling.

    Input:  (B, L, d_model)   — embedded by the Lightning module
    Output: (B, 1)            — raw logit
    """
    def __init__(
        self,
        d_model:    int   = 128,
        d_state:    int   = 16,
        d_conv:     int   = 4,
        expand:     int   = 2,
        num_layers: int   = 3,
        dropout:    float = 0.2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pool = AttentionPool(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.pool(x, mask)      # (B, D)
        return self.head(x)         # (B, 1)


# ─────────────────────────────────────────────
# Lightning module
# ─────────────────────────────────────────────

class MambaDNALightning(pl.LightningModule):
    """
    Bidirectional RC Mamba model for miRNA target classification.

    Input per sample: two separate integer index tensors (MRE, miRNA).
    Forward pass builds:
        MRE | SEP | miRNA | SEP | rc(miRNA) | SEP | rc(MRE)
    giving the model both strand orientations explicitly.
    """

    def __init__(
        self,
        d_model:       int   = 128,
        d_state:       int   = 16,
        d_conv:        int   = 4,
        expand:        int   = 2,
        num_layers:    int   = 3,
        dropout:       float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay:  float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()

        # vocab: A=0, C=1, G=2, T=3, unknown=4, padding=5  → 6 tokens
        # dropout after embedding acts like input regularisation
        self.embedding   = nn.Embedding(6, d_model, padding_idx=5)
        self.emb_dropout = nn.Dropout(dropout)

        self.model = MambaDNA(
            d_model=d_model, d_state=d_state, d_conv=d_conv,
            expand=expand, num_layers=num_layers, dropout=dropout,
        )

        self.criterion = nn.BCEWithLogitsLoss()

        # ── metrics: train / val ──
        self.train_ap    = AveragePrecision(task='binary')
        self.val_ap      = AveragePrecision(task='binary')
        self.train_auroc = AUROC(task='binary')
        self.val_auroc   = AUROC(task='binary')
        self.train_mcc   = MatthewsCorrCoef(task='binary')
        self.val_mcc     = MatthewsCorrCoef(task='binary')

        # ── metrics: test set 1 ──
        self.test_ap    = AveragePrecision(task='binary')
        self.test_auroc = AUROC(task='binary')
        self.test_mcc   = MatthewsCorrCoef(task='binary')

        # ── metrics: test set 2 (left-out / final) ──
        self.final_test_ap    = AveragePrecision(task='binary')
        self.final_test_auroc = AUROC(task='binary')
        self.final_test_mcc   = MatthewsCorrCoef(task='binary')

    # ── forward ──────────────────────────────

    def forward(self, batch_x: tuple) -> torch.Tensor:
        mr, mi = batch_x                                    # (B, L_mr), (B, L_mi)

        rc_mi = get_rc(mi)                                  # (B, L_mi) reversed
        rc_mr = get_rc(mr)                                  # (B, L_mr) reversed

        # embed all four segments + apply input dropout
        e_mr   = self.emb_dropout(self.embedding(mr))      # (B, L_mr, D)
        e_mi   = self.emb_dropout(self.embedding(mi))      # (B, L_mi, D)
        e_rcmi = self.emb_dropout(self.embedding(rc_mi))   # (B, L_mi, D)
        e_rcmr = self.emb_dropout(self.embedding(rc_mr))   # (B, L_mr, D)

        # zero-vector SEP token — signals segment boundary to Mamba state
        sep = torch.zeros(mr.shape[0], 1, self.hparams.d_model, device=self.device)

        # final sequence: MRE | SEP | miRNA | SEP | rc(miRNA) | SEP | rc(MRE)
        x = torch.cat([e_mr, sep, e_mi, sep, e_rcmi, sep, e_rcmr], dim=1)

        # padding mask — True wherever the original index was the padding token (5)
        # rc sequences have the same lengths as their originals so we reuse the masks
        sep_mask = torch.zeros(mr.shape[0], 1, dtype=torch.bool, device=self.device)
        pad_mr   = (mr == 5)                               # (B, L_mr)
        pad_mi   = (mi == 5)                               # (B, L_mi)
        pad_mask = torch.cat([
            pad_mr, sep_mask,
            pad_mi, sep_mask,
            pad_mi, sep_mask,   # rc(miRNA) same length as miRNA
            pad_mr,             # rc(MRE)   same length as MRE
        ], dim=1)                                          # (B, total_L)

        logits = self.model(x, pad_mask)
        return logits.squeeze(-1)                          # (B,)

    # ── shared step ──────────────────────────

    def _shared_step(self, batch, batch_idx):
        batch_x, labels = batch
        logits = self(batch_x)
        loss   = self.criterion(logits, labels)
        probs  = torch.sigmoid(logits)
        preds  = (probs > 0.5).float()
        return loss, preds, probs, labels

    # ── train / val ──────────────────────────

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

    # ── test (two datasets) ──────────────────

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        loss, preds, probs, labels = self._shared_step(batch, batch_idx)

        if dataloader_idx == 0:
            prefix = 'test'
            self.test_ap(probs, labels.int())
            self.test_auroc(probs, labels.int())
            self.test_mcc(preds, labels.int())
            ap_m, auroc_m, mcc_m = self.test_ap, self.test_auroc, self.test_mcc
        else:
            prefix = 'final_test'
            self.final_test_ap(probs, labels.int())
            self.final_test_auroc(probs, labels.int())
            self.final_test_mcc(preds, labels.int())
            ap_m, auroc_m, mcc_m = self.final_test_ap, self.final_test_auroc, self.final_test_mcc

        self.log(f'{prefix}_loss',  loss,    on_step=False, on_epoch=True)
        self.log(f'{prefix}_ap',    ap_m,    on_step=False, on_epoch=True)
        self.log(f'{prefix}_auroc', auroc_m, on_step=False, on_epoch=True)
        self.log(f'{prefix}_mcc',   mcc_m,   on_step=False, on_epoch=True)
        return loss

    # ── optimiser ────────────────────────────

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


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    df = ps.read_csv(
        '/home/adam/adam/data/AGO2eCLIPManakov2022trainimprovedwfeatures.csv',
        columns=['mre_sequence', 'mirna_sequence', 'mir_fam', 'label'],
    ).unique(subset=['mre_sequence', 'mirna_sequence'], keep='none')

    test_df = ps.read_csv(
        '/home/adam/adam/data/AGO2eCLIPManakov2022testimprovedwfeatures.csv',
        columns=['mre_sequence', 'mirna_sequence', 'label'],
    ).unique(subset=['mre_sequence', 'mirna_sequence'], keep='none')

    final_test_df = ps.read_csv(
        '/home/adam/adam/data/AGO2eCLIPManakov2022leftoutimprovedwfeatures.csv',
        columns=['mre_sequence', 'mirna_sequence', 'label'],
    ).unique(subset=['mre_sequence', 'mirna_sequence'], keep='none')

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for i, (train_idx, val_idx) in enumerate(sgkf.split(df, df['label'], groups=df['mir_fam'])):
        final_train_data = df[train_idx]
        final_val_data   = df[val_idx]

        train_dataset      = SeparatedIndexDataset(final_train_data)
        eval_dataset       = SeparatedIndexDataset(final_val_data)
        test_dataset       = SeparatedIndexDataset(test_df)
        final_test_dataset = SeparatedIndexDataset(final_test_df)

        train_dataloader      = DataLoader(train_dataset,      batch_size=256, collate_fn=collate_fn_separated, shuffle=True,  num_workers=4)
        val_dataloader        = DataLoader(eval_dataset,       batch_size=256, collate_fn=collate_fn_separated, shuffle=False, num_workers=4)
        test_dataloader       = DataLoader(test_dataset,       batch_size=256, collate_fn=collate_fn_separated, shuffle=False, num_workers=4)
        final_test_dataloader = DataLoader(final_test_dataset, batch_size=256, collate_fn=collate_fn_separated, shuffle=False, num_workers=4)

        # callbacks recreated each fold so internal state is fresh
        early_stop_callback = EarlyStopping(monitor='val_ap', patience=7, mode='max')
        checkpoint_callback = ModelCheckpoint(
            monitor='val_ap', mode='max',
            dirpath='/home/adam/eli-adam/models/',
            filename=f'Mamba-chim-RC-fold{i}',
        )

        wandb_logger = WandbLogger(
            project='mamba-mirna',
            name=f'fold-{i}',
            log_model=False,
        )

        model = MambaDNALightning()

        trainer = pl.Trainer(
            max_epochs=25,
            callbacks=[early_stop_callback, checkpoint_callback],
            accelerator='auto',
            precision='16-mixed',
            logger=wandb_logger,
        )

        trainer.fit(model, train_dataloader, val_dataloader)

        # both test sets in one call → test_step receives dataloader_idx 0 and 1
        trainer.test(model, [test_dataloader, final_test_dataloader])

        wandb_logger.experiment.finish()


if __name__ == '__main__':
    main()
