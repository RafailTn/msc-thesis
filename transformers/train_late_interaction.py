"""
Training harness for LateInteractionMiRNA.

Two modes:
  --mode single   train on a train TSV, evaluate on one or more test TSVs
  --mode kfold    GroupKFold over miRNA family on the train TSV (optionally also
                  evaluating each fold's model on the external test sets)

Reports metrics STRATIFIED BY SEED TYPE, because aggregate AP is not the quantity
of interest here -- recall on the 3'-compensatory / seedless stratum is.

Example
-------
python train_late_interaction.py --mode kfold --folds 5 \
    --train AGO2_eCLIP_Manakov2022_train.tsv.gz \
    --test  AGO2_eCLIP_Manakov2022_test.tsv.gz AGO2_CLASH_Hejret2023_test.tsv.gz \
    --epochs 8 --batch-size 256 --out runs/li_kfold
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

from hard_negatives import (
    PairedHardNegSampler,
    ensure_energy,
    mine_hard_negatives,
    mining_report,
    paired_infonce,
    paired_margin_loss,
    resample_weights,
)
from late_interaction_mirna import (
    MAX_MIRNA,
    MRE_LEN,
    PAD_IDX,
    LateInteractionMiRNA,
    bce_loss,
    register_profile,
)

# ----------------------------------------------------------------- tokenising
_ASCII = np.full(256, PAD_IDX, dtype=np.int8)
for _c, _i in zip("ACGU", range(4)):
    _ASCII[ord(_c)] = _i
_ASCII[ord("T")] = 3  # T -> U


def tokenise(seqs: pd.Series, length: int) -> np.ndarray:
    out = np.full((len(seqs), length), PAD_IDX, dtype=np.int8)
    for i, s in enumerate(seqs.astype(str).str.upper().values):
        b = np.frombuffer(s.encode(), dtype=np.uint8)[:length]
        out[i, : len(b)] = _ASCII[b]
    return out


# --------------------------------------------------------------- seed typing
def seed_stratum(mirna: str, mre: str) -> str:
    """
    Coarse seed classification for stratified reporting.
    Longest contiguous Watson-Crick/wobble run within miRNA positions 2-8,
    scanned against the reverse-complemented target.
    """
    comp = {"A": "U", "U": "A", "G": "C", "C": "G"}
    wob = {("G", "U"), ("U", "G")}
    m = mirna.upper().replace("T", "U")
    t = mre.upper().replace("T", "U")[::-1]
    seed = m[1:8]
    best = 0
    for off in range(max(1, len(t) - len(seed) + 1)):
        run = cur = 0
        for k, ch in enumerate(seed):
            if off + k >= len(t):
                break
            pair = (ch, t[off + k])
            if comp.get(ch) == t[off + k] or pair in wob:
                cur += 1
                run = max(run, cur)
            else:
                cur = 0
        best = max(best, run)
    if best >= 7:
        return "canonical_7_8mer"
    if best == 6:
        return "canonical_6mer"
    if best >= 4:
        return "weak_seed"
    return "seedless"


# ------------------------------------------------------------------ dataset
class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mirna_col="noncodingRNA", mre_col="gene",
                 label_col="label"):
        self.m = tokenise(df[mirna_col], MAX_MIRNA)
        self.t = tokenise(df[mre_col], MRE_LEN)
        self.y = df[label_col].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.m[i].astype(np.int64)),
            torch.from_numpy(self.t[i].astype(np.int64)),
            torch.tensor(self.y[i]),
        )


# ------------------------------------------------------------------- config
@dataclass
class Cfg:
    d_model: int = 128
    d_interact: int = 64
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 8
    batch_size: int = 256
    patience: int = 3
    seed: int = 42
    # hard-negative settings
    hard_neg: str = "off"          # off | resample | paired
    k_neg: int = 4
    boost: float = 8.0
    max_dg_gap: float | None = None
    lambda_margin: float = 0.0
    lambda_infonce: float = 0.0
    margin: float = 1.0


# ------------------------------------------------------------- train / eval
def make_model(cfg: Cfg, device) -> nn.Module:
    return LateInteractionMiRNA(
        d_model=cfg.d_model,
        d_interact=cfg.d_interact,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
    ).to(device)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs, ys, maxsims = [], [], []
    for m, t, y in loader:
        s = model(m.to(device), t.to(device))
        probs.append(torch.sigmoid(s.logit).cpu().numpy())
        maxsims.append(s.maxsim.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys), np.concatenate(maxsims)


def train_one(model, train_loader, val_loader, cfg: Cfg, device, log=print):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    best_ap, best_state, bad = -1.0, None, 0

    paired = cfg.hard_neg == "paired"
    for ep in range(cfg.epochs):
        model.train()
        tot = n = 0
        for m, t, y in train_loader:
            m, t, y = m.to(device), t.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            s = model(m, t)
            loss = bce_loss(s, y)
            if paired:
                # batch is [pos, neg*k] repeated -- see PairedHardNegSampler
                if cfg.lambda_margin:
                    loss = loss + cfg.lambda_margin * paired_margin_loss(
                        s.logit, cfg.k_neg, cfg.margin)
                if cfg.lambda_infonce:
                    loss = loss + cfg.lambda_infonce * paired_infonce(
                        s.logit, cfg.k_neg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(y)
            n += len(y)
        sched.step()

        p, yv, _ = predict(model, val_loader, device)
        ap = average_precision_score(yv, p)
        log(f"  epoch {ep+1:>2}  train_loss {tot/n:.4f}  val_AP {ap:.4f}")

        if ap > best_ap:
            best_ap, bad = ap, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"  early stop (patience {cfg.patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_ap


def evaluate(model, df: pd.DataFrame, cfg: Cfg, device, name: str, log=print) -> dict:
    loader = DataLoader(PairDataset(df), batch_size=cfg.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    p, y, maxsim = predict(model, loader, device)

    res = {
        "set": name,
        "n": int(len(y)),
        "AP": float(average_precision_score(y, p)),
        "ROC_AUC": float(roc_auc_score(y, p)),
        "strata": {},
    }

    # stratified recall at the 0.5 threshold -- the metric that actually matters
    if "seed_stratum" in df.columns:
        pos = y == 1
        for st in sorted(df["seed_stratum"].unique()):
            m = (df["seed_stratum"].to_numpy() == st) & pos
            if m.sum() < 50:
                continue
            res["strata"][st] = {
                "n_pos": int(m.sum()),
                "recall@0.5": float((p[m] >= 0.5).mean()),
                "mean_prob": float(p[m].mean()),
            }

    # register profile on the missed positives -- is the encoder finding 3' partners?
    fn = (y == 1) & (p < 0.5)
    if fn.sum() >= 50:
        roles = register_profile(type("S", (), {"maxsim": torch.from_numpy(maxsim[fn])})())
        res["FN_register_profile"] = {k: float(v.mean()) for k, v in roles.items()}
    tp = (y == 1) & (p >= 0.5)
    if tp.sum() >= 50:
        roles = register_profile(type("S", (), {"maxsim": torch.from_numpy(maxsim[tp])})())
        res["TP_register_profile"] = {k: float(v.mean()) for k, v in roles.items()}

    log(f"[{name}] n={res['n']}  AP={res['AP']:.4f}  ROC-AUC={res['ROC_AUC']:.4f}")
    for st, d in res["strata"].items():
        log(f"    {st:<20} n_pos={d['n_pos']:>7}  recall@0.5={d['recall@0.5']:.4f}")
    return res


# --------------------------------------------------------------------- data
def load(path: str, add_strata: bool) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if add_strata:
        df["seed_stratum"] = [
            seed_stratum(a, b) for a, b in zip(df["noncodingRNA"], df["gene"])
        ]
    return df


# --------------------------------------------------------------------- main
def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--mode", choices=["single", "kfold"], default="single")
    ap_.add_argument("--train", required=True)
    ap_.add_argument("--test", nargs="*", default=[])
    ap_.add_argument("--folds", type=int, default=5)
    ap_.add_argument("--group-col", default="noncodingRNA_fam")
    ap_.add_argument("--val-frac", type=float, default=0.1,
                     help="single mode: family-grouped validation split off train")
    ap_.add_argument("--epochs", type=int, default=8)
    ap_.add_argument("--batch-size", type=int, default=256)
    ap_.add_argument("--lr", type=float, default=3e-4)
    ap_.add_argument("--no-strata", action="store_true")
    ap_.add_argument("--hard-neg", choices=["off", "resample", "paired"], default="off")
    ap_.add_argument("--energy-col", default="energy",
                     help="column with hybridisation dG (IntaRNA/RNAduplex). "
                          "A proxy is computed if absent -- smoke tests only.")
    ap_.add_argument("--k-neg", type=int, default=4)
    ap_.add_argument("--boost", type=float, default=8.0)
    ap_.add_argument("--max-dg-gap", type=float, default=None)
    ap_.add_argument("--lambda-margin", type=float, default=0.0)
    ap_.add_argument("--lambda-infonce", type=float, default=0.0)
    ap_.add_argument("--margin", type=float, default=1.0)
    ap_.add_argument("--out", default="runs/late_interaction")
    args = ap_.parse_args()

    cfg = Cfg(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
              hard_neg=args.hard_neg, k_neg=args.k_neg, boost=args.boost,
              max_dg_gap=args.max_dg_gap, lambda_margin=args.lambda_margin,
              lambda_infonce=args.lambda_infonce, margin=args.margin)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    logf = open(out / "log.txt", "w")

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"device={device}  mode={args.mode}  cfg={asdict(cfg)}")

    train_df = load(args.train, not args.no_strata)
    test_dfs = {Path(p).name: load(p, not args.no_strata) for p in args.test}
    energy_col = (ensure_energy(train_df, args.energy_col)
                  if cfg.hard_neg != "off" else None)
    if cfg.hard_neg != "off":
        log(f"hard-neg mode={cfg.hard_neg}  energy_col={energy_col}  k={cfg.k_neg}")
    log(f"train n={len(train_df)}  " + "  ".join(f"{k} n={len(v)}" for k, v in test_dfs.items()))

    groups = train_df[args.group_col].fillna("__UNASSIGNED__").to_numpy()
    n_unassigned = int((groups == "__UNASSIGNED__").sum())
    if n_unassigned:
        log(f"WARNING: {n_unassigned} rows lack {args.group_col}; "
            "they form one pseudo-group and will not leak across folds")

    results = []

    def loader_for(df, shuffle):
        return DataLoader(PairDataset(df), batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=2, pin_memory=True, drop_last=shuffle)

    def train_loader_for(idx: np.ndarray):
        """
        Build the training loader for a set of ORIGINAL row indices, applying the
        selected hard-negative strategy. Mining happens inside `idx` only, so no
        hard negative can cross a fold boundary.
        """
        sub = train_df.iloc[idx].reset_index(drop=True)
        ds = PairDataset(sub)

        if cfg.hard_neg == "off":
            return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)

        local = np.arange(len(sub))
        mined = mine_hard_negatives(
            sub, local, energy_col=energy_col, k=cfg.k_neg,
            hard_strata=("weak_seed", "seedless"), max_dg_gap=cfg.max_dg_gap,
        )
        log("  " + mining_report(sub, mined, energy_col))
        if not mined:
            log("  no hard negatives -- falling back to plain shuffling")
            return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)

        if cfg.hard_neg == "resample":
            w = resample_weights(len(sub), mined, boost=cfg.boost)
            sampler = torch.utils.data.WeightedRandomSampler(
                w, num_samples=len(sub), replacement=True)
            return DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler,
                              num_workers=2, pin_memory=True, drop_last=True)

        apb = max(1, cfg.batch_size // (1 + cfg.k_neg))
        bs = PairedHardNegSampler(mined, k=cfg.k_neg, anchors_per_batch=apb,
                                  seed=cfg.seed)
        log(f"  paired batches: {apb} anchors x (1+{cfg.k_neg}) = {bs.batch_size} rows, "
            f"{len(bs)} batches/epoch")
        return DataLoader(ds, batch_sampler=bs, num_workers=2, pin_memory=True)

    if args.mode == "single":
        gss = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=cfg.seed)
        tr_i, va_i = next(gss.split(train_df, groups=groups))
        log(f"single run: train={len(tr_i)}  val={len(va_i)} "
            f"({len(set(groups[va_i]))} held-out families)")

        model = make_model(cfg, device)
        best = train_one(model, train_loader_for(tr_i),
                         loader_for(train_df.iloc[va_i], False), cfg, device, log)
        log(f"best val AP {best:.4f}")
        torch.save(model.state_dict(), out / "model.pt")

        results.append(evaluate(model, train_df.iloc[va_i], cfg, device, "val", log))
        for name, df in test_dfs.items():
            results.append(evaluate(model, df, cfg, device, name, log))

    else:
        gkf = GroupKFold(n_splits=args.folds)
        oof = np.zeros(len(train_df))
        for k, (tr_i, va_i) in enumerate(gkf.split(train_df, groups=groups), 1):
            log(f"\n=== fold {k}/{args.folds}  train={len(tr_i)} val={len(va_i)} "
                f"({len(set(groups[va_i]))} held-out families) ===")
            model = make_model(cfg, device)
            best = train_one(model, train_loader_for(tr_i),
                             loader_for(train_df.iloc[va_i], False), cfg, device, log)
            log(f"fold {k} best val AP {best:.4f}")
            torch.save(model.state_dict(), out / f"model_fold{k}.pt")

            p, _, _ = predict(model, loader_for(train_df.iloc[va_i], False), device)
            oof[va_i] = p

            r = evaluate(model, train_df.iloc[va_i], cfg, device, f"fold{k}_val", log)
            results.append(r)
            for name, df in test_dfs.items():
                results.append(evaluate(model, df, cfg, device, f"fold{k}_{name}", log))

        train_df["oof_pred"] = oof
        train_df[["noncodingRNA", "gene", "label", "oof_pred"]
                 + (["seed_stratum"] if "seed_stratum" in train_df else [])
                 ].to_csv(out / "oof_predictions.tsv.gz", sep="\t", index=False)

        y = train_df["label"].to_numpy()
        log(f"\n=== OOF overall  AP={average_precision_score(y, oof):.4f}  "
            f"ROC-AUC={roc_auc_score(y, oof):.4f} ===")
        if "seed_stratum" in train_df:
            for st in sorted(train_df["seed_stratum"].unique()):
                m = (train_df["seed_stratum"] == st).to_numpy() & (y == 1)
                if m.sum() >= 50:
                    log(f"    OOF {st:<20} n_pos={m.sum():>7}  "
                        f"recall@0.5={(oof[m] >= 0.5).mean():.4f}")

    with open(out / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    log(f"\nwrote {out/'results.json'}")
    logf.close()


if __name__ == "__main__":
    main()
