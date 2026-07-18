"""Smoke test for the two-tower model.

Instantiates TwoTowerLightning from a config dict, pulls ONE real batch from
the actual dataloader (OneHotDataset + collate_fn_onehot) built from real v7
data, runs a forward pass, prints the output shape, and confirms backward()
runs. It does NOT launch training.

Run from the transformers/ directory (so `from utils import ...` resolves):
    cd transformers && pixi run python smoke_test_two_tower.py
"""

import polars as ps
import torch
from torch.utils.data import DataLoader

from utils import OneHotDataset, collate_fn_onehot
from two_tower_imp import build_model, split_streams

# A real v7 dataset shipped in the repo. Columns: gene (=MRE, native 5'->3'),
# noncodingRNA (=miRNA), label. The training main() uses different column names
# (mre_sequence/mirna_sequence) from another machine's CSVs, so we rename here to
# match what OneHotDataset expects.
DATA_TSV = "../data/AGO2_eCLIP_Manakov2022_test_v7.tsv"
N_ROWS = 512
BATCH_SIZE = 32


def load_batch():
    df = ps.read_csv(DATA_TSV, separator="\t", columns=["gene", "noncodingRNA", "label"])
    df = df.rename({"gene": "mre_sequence", "noncodingRNA": "mirna_sequence"})
    df = df.unique(subset=["mre_sequence", "mirna_sequence"], keep="none").head(N_ROWS)
    dataset = OneHotDataset(df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_onehot, shuffle=False)
    return next(iter(loader))


def main():
    torch.manual_seed(0)
    x, labels = load_batch()
    print(f"[batch] x: shape={tuple(x.shape)} dtype={x.dtype}   labels: shape={tuple(labels.shape)} dtype={labels.dtype}")
    print(f"[batch] label balance: {int(labels.sum())} pos / {labels.numel()} total")

    # Sanity-check the stream split recovers two non-empty strands.
    mre_x, mre_pad, mir_x, mir_pad = split_streams(x, "native")
    mre_lens = (~mre_pad).sum(1)
    mir_lens = (~mir_pad).sum(1)
    print(f"[split] MRE tensor {tuple(mre_x.shape)}  len min/max={int(mre_lens.min())}/{int(mre_lens.max())}")
    print(f"[split] miRNA tensor {tuple(mir_x.shape)}  len min/max={int(mir_lens.min())}/{int(mir_lens.max())}")

    # Small config so it runs fast on CPU; exercises the real code paths.
    cfg = dict(
        nuc_dim=4,
        mirna_d_model=64, mirna_nhead=4, mirna_num_layers=2, mirna_dim_feedforward=128,
        mirna_max_seq_len=64,
        mre_d_model=64, mre_nhead=4, mre_num_layers=2, mre_dim_feedforward=128,
        mre_max_seq_len=256,
        bottleneck_dim=32,
        fusion="cross_attention", fusion_dim=64, fusion_nhead=4, fusion_num_latents=4,
        loss="bce", label_smoothing=0.05,
    )
    model = build_model("two_tower", **cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] TwoTowerLightning instantiated: {n_params:,} params, fusion={cfg['fusion']}")

    # Forward
    model.train()
    logits = model(x)
    print(f"[forward] logits shape={tuple(logits.shape)} (expected ({x.shape[0]},)) dtype={logits.dtype}")
    assert logits.shape == (x.shape[0],), "output shape mismatch"

    # Loss + backward
    loss = model.criterion(logits, labels)
    print(f"[loss] {loss.item():.4f}")
    loss.backward()
    grad_norm = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None))
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"[backward] ok — grad global-norm={grad_norm.item():.4f}, params with grad={n_with_grad}")

    # Quick check that the concat_mlp fallback and rc orientation also build/run.
    alt = build_model("two_tower", **{**cfg, "fusion": "concat_mlp", "mre_orientation": "rc", "loss": "gce"})
    alt_logits = alt(x)
    alt_loss = alt.criterion(alt_logits, labels)
    alt_loss.backward()
    print(f"[alt] concat_mlp + rc + gce: logits {tuple(alt_logits.shape)}, loss {alt_loss.item():.4f} — backward ok")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
