#!/usr/bin/env python3
"""
predict_cnn.py — score (optionally unlabeled) miRNA–MRE pairs with one
sequence-only CNN checkpoint.

Reads sequences from a TSV/CSV, de-duplicates on the chimeric (miRNA+MRE)
sequence keeping the first occurrence, scores the unique pairs with one
cnn_branches_mirbind.py checkpoint, and writes predictions.

No feature preparation (IntaRNA / conservation / eCLIP) is run: the checkpoint
is assumed to use only the 2D sequence branch, so the auxiliary feature columns
are ignored (zero-filled) by the model. No 'label' column is required and no
metrics are computed.

Usage
-----
  python cnn/predict_cnn.py \\
      --checkpoint checkpoints/cnn_seqonly.pt \\
      --input pairs.tsv -o predictions.tsv \\
      --mre-col mre_sequence --mirna-col mirna_sequence
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from cnn_branches_mirbind import (  # noqa: E402
    MiRNAInteractionDataset,
    predict_logits,
    _load_ckpt_model,
)


def chimeric_key(mirna: str, mre: str) -> str:
    """Dedup key: miRNA+MRE, upper-case, T→U (RNA-normalised)."""
    norm = lambda s: str(s).upper().replace("T", "U")
    return norm(mirna) + norm(mre)


def dedup_chimeric(df: pd.DataFrame, mre_col: str, mirna_col: str
                   ) -> tuple[pd.DataFrame, int]:
    """Drop rows whose (miRNA+MRE) chimeric sequence already appeared, keeping
    the first occurrence. Returns ``(deduped_df, n_removed)``."""
    chim = (df[mirna_col].astype(str).str.upper().str.replace("T", "U", regex=False)
            + df[mre_col].astype(str).str.upper().str.replace("T", "U", regex=False))
    mask = ~chim.duplicated(keep="first")
    return df[mask].reset_index(drop=True), int((~mask).sum())


def score_dataframe(
    checkpoint: str | Path,
    df: pd.DataFrame,
    *,
    device: str = "cpu",
    mre_col: str = "mre_sequence",
    mirna_col: str = "mirna_sequence",
    batch_size: int = 256,
    num_workers: int = 4,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    """Score an in-memory DataFrame of sequences with one checkpoint.

    Returns ``(df, ckpt)`` where *df* is the input plus ``interaction_probability``
    and ``prediction`` columns. Auxiliary feature columns may be absent — a
    sequence-only model ignores them (they are zero-filled). No 'label' column is
    required.
    """
    dev = torch.device(device)
    model, ckpt = _load_ckpt_model(checkpoint, dev)

    ds = MiRNAInteractionDataset.from_df(
        df, has_labels=False,
        mre_col=mre_col, mirna_col=mirna_col)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    logits, _ = predict_logits(model, loader, dev)
    probs = 1.0 / (1.0 + np.exp(-logits))

    out = df.copy()
    out["interaction_probability"] = probs
    out["prediction"] = (probs >= threshold).astype(int)
    return out, ckpt


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Sequence-only cnn_branches_mirbind.py checkpoint (.pt).")
    p.add_argument("--input", required=True,
                   help="TSV/CSV with miRNA and MRE sequence columns.")
    p.add_argument("-o", "--output", required=True,
                   help="Output TSV of (de-duplicated) predictions.")
    p.add_argument("--sep", default="\t",
                   help="Separator of the --input file.")
    p.add_argument("--mre-col", default="mre_sequence", dest="mre_col")
    p.add_argument("--mirna-col", default="mirna_sequence", dest="mirna_col")
    p.add_argument("--no-dedup", action="store_true", dest="no_dedup",
                   help="Score every input row instead of de-duplicating on the "
                        "chimeric (miRNA+MRE) sequence.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4, dest="num_workers")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    df = pd.read_csv(args.input, sep=args.sep)
    print(f"Read {len(df)} rows from {args.input}")
    for col in (args.mre_col, args.mirna_col):
        if col not in df.columns:
            sys.exit(f"ERROR: required column '{col}' not found in input. "
                     f"Available: {list(df.columns)}")

    if not args.no_dedup:
        df, n_removed = dedup_chimeric(df, args.mre_col, args.mirna_col)
        if n_removed:
            print(f"  dedup: removed {n_removed} duplicate chimeric sequences "
                  f"→ {len(df)} unique rows")

    out, ckpt = score_dataframe(
        args.checkpoint, df, device=args.device,
        mre_col=args.mre_col, mirna_col=args.mirna_col,
        batch_size=args.batch_size, num_workers=args.num_workers,
        threshold=args.threshold)
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch')}, "
          f"val_metrics={ckpt.get('val_metrics')})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)
    n_pos = int((out["prediction"] == 1).sum())
    print(f"Wrote {len(out)} rows → {out_path}  "
          f"({n_pos} predicted positive @ thr={args.threshold})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
