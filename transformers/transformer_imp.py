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


class TransformerDNA(nn.Module):
    """
    Stacked vanilla Transformer encoder for DNA/RNA binary classification.

    Input:  (batch, seq_len, d_model)   — already projected by DnaOneHotEncoder
    Output: scalar logit per sequence
    """
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.pool = AttentionPool(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # mask: (B, L), True = padding position
        for block in self.blocks:
            x = block(x, key_padding_mask=mask)
        x = self.norm(x)
        x = self.pool(x, mask)
        return self.head(x)


class TransformerDNALightning(pl.LightningModule):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()
        # input_dim=5: [A, C, G, T, segment_id]; max_seq_len=256 covers chimeric seqs
        self.encoder = DnaOneHotEncoder(input_dim=5, emb_size=d_model, max_seq_len=256, dropout=dropout)
        self.model = TransformerDNA(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.criterion = nn.BCEWithLogitsLoss()
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
        # x: (B, L, 5)
        pad_mask = (x.sum(dim=-1) == 0)   # (B, L), True = padding
        x = self.encoder(x)               # (B, L, d_model)
        x = self.model(x, pad_mask)       # (B, 1)
        return x.squeeze(-1)              # (B,)

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
        self.log('train_loss',  loss,            on_step=True,  on_epoch=True, prog_bar=True)
        self.log('train_ap',    self.train_ap,   on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_mcc',   self.train_mcc,  on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, probs, labels = self._shared_step(batch, batch_idx)
        self.val_ap(probs, labels.int())
        self.val_auroc(probs, labels.int())
        self.val_mcc(preds, labels.int())
        self.log('val_loss',  loss,          on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_ap',    self.val_ap,   on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mcc',   self.val_mcc,  on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        loss, preds, probs, labels = self._shared_step(batch, batch_idx)
        prefix = "test" if dataloader_idx == 0 else "leftout"
        if dataloader_idx == 0:
            self.test_ap(probs, labels.int())
            self.test_auroc(probs, labels.int())
            self.test_mcc(preds, labels.int())
            ap_metric    = self.test_ap
            auroc_metric = self.test_auroc
            mcc_metric   = self.test_mcc
        else:
            self.final_test_ap(probs, labels.int())
            self.final_test_auroc(probs, labels.int())
            self.final_test_mcc(preds, labels.int())
            ap_metric    = self.final_test_ap
            auroc_metric = self.final_test_auroc
            mcc_metric   = self.final_test_mcc

        self.log(f'{prefix}_loss',  loss,         on_step=False, on_epoch=True)
        self.log(f'{prefix}_ap',    ap_metric,    on_step=False, on_epoch=True)
        self.log(f'{prefix}_auroc', auroc_metric, on_step=False, on_epoch=True)
        self.log(f'{prefix}_mcc',   mcc_metric,   on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch['input_ids']
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


def main():
    df = ps.read_csv('/home/adam/adam/data/AGO2eCLIPManakov2022trainimprovedwfeatures.csv', columns=['mre_sequence', 'mirna_sequence', 'mir_fam', 'label'])
    df = df.unique(subset=['mre_sequence', 'mirna_sequence'], keep='none')
    test_df = ps.read_csv('/home/adam/adam/data/AGO2eCLIPManakov2022testimprovedwfeatures.csv', columns=['mre_sequence', 'mirna_sequence', 'label'])
    test_df = test_df.unique(subset=['mre_sequence', 'mirna_sequence'], keep='none')
    final_test_df = ps.read_csv('/home/adam/adam/data/AGO2eCLIPManakov2022leftoutimprovedwfeatures.csv', columns=['mre_sequence', 'mirna_sequence', 'label'])
    final_test_df = final_test_df.unique(subset=['mre_sequence', 'mirna_sequence'], keep='none')

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(sgkf.split(df, df['label'], groups=df['mir_fam'])):
        final_train_data, final_val_data = df[train_idx], df[val_idx]

        early_stop_callback = EarlyStopping(monitor="val_ap", patience=7, mode="max")
        checkpoint_callback = ModelCheckpoint(
            monitor="val_ap", mode="max",
            dirpath="/home/adam/eli-adam/models/",
            filename=f"Transformer-chim-OH_dropout03_weightedpool_fold{i}"
        )

        print(final_train_data.filter(ps.col('label') == 0).height)
        print(final_train_data.filter(ps.col('label') == 1).height)

        train_dataset      = OneHotDataset(final_train_data)
        eval_dataset       = OneHotDataset(final_val_data)
        test_dataset       = OneHotDataset(test_df)
        final_test_dataset = OneHotDataset(final_test_df)

        train_dataloader      = DataLoader(train_dataset,      batch_size=256, collate_fn=collate_fn_onehot, shuffle=True,  num_workers=4)
        val_dataloader        = DataLoader(eval_dataset,       batch_size=256, collate_fn=collate_fn_onehot, shuffle=False, num_workers=4)
        test_dataloader       = DataLoader(test_dataset,       batch_size=256, collate_fn=collate_fn_onehot, shuffle=False, num_workers=4)
        final_test_dataloader = DataLoader(final_test_dataset, batch_size=256, collate_fn=collate_fn_onehot, shuffle=False, num_workers=4)

        wandb_logger = WandbLogger(
            project="transformer-mirna-dropout03-weightedpool",
            name=f"fold-{i}",
            log_model=False,
        )

        model = TransformerDNALightning()
        trainer = pl.Trainer(
            max_epochs=25,
            callbacks=[early_stop_callback, checkpoint_callback],
            accelerator='auto',
            precision='16-mixed',
            logger=wandb_logger,
        )

        trainer.fit(model, train_dataloader, val_dataloader)
        trainer.test(model, [test_dataloader, final_test_dataloader])
        wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
