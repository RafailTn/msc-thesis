#!/usr/bin/env bash
# Thin wrapper around cnn/cnn_branches_mirbind.py that trains the CNN either with
# stratified-group k-fold CV or once on the entire training set (no validation).
#
#   bash cnn/run_train.sh kfold      # K-fold CV (default)  -> *_fold{1..K}.pt
#   bash cnn/run_train.sh full       # one run on all of --train, no val split
#
# Everything is overridable via environment variables, e.g.
#   FOLDS=10 EPOCHS=60 bash cnn/run_train.sh kfold
#   OUT=checkpoints/cnn.pt bash cnn/run_train.sh full
#
# Set OOF_OUT to also write per-row out-of-fold predictions (kfold only) — each
# row scored by the fold that held it out. That table is the leakage-free input
# to cnn/cooperativity_analysis.py:
#   OOF_OUT=data/train_oof.tsv bash cnn/run_train.sh kfold
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root (pipeline/)

MODE="${1:-kfold}"                           # kfold | full

PY="${PY:-dependencies/.pixi/envs/default/bin/python}"
SCRIPT="cnn/cnn_branches_mirbind.py"

# ── data + output ──────────────────────────────────────────────────────────
TRAIN="${TRAIN:-data/AGO2_eCLIP_Manakov2022_train_v7.tsv}"
TESTS="${TESTS:-data/AGO2_eCLIP_Manakov2022_test_v7.tsv data/AGO2_eCLIP_Manakov2022_leftout_v7.tsv}"
OUT="${OUT:-checkpoints/cnn_mirbind.pt}"
FOLDS="${FOLDS:-5}"
OOF_OUT="${OOF_OUT:-}"                 # non-empty -> --oof-out (kfold only)

# ── v7 TSV column mapping (gene = MRE sequence) ────────────────────────────
MRE_COL="${MRE_COL:-gene}"
MIRNA_COL="${MIRNA_COL:-noncodingRNA}"
FAMILY_COL="${FAMILY_COL:-noncodingRNA_fam}"

# ── training hyper-params ──────────────────────────────────────────────────
EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH:-256}"
LR="${LR:-1e-3}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
METRIC="${METRIC:-auprc}"

# ── model architecture / loss ──────────────────────────────────────────────
SEQ_PAIRING="${SEQ_PAIRING:-embed}"    # binary | multi | multi4 | embed
PAIR_EMBED_DIM="${PAIR_EMBED_DIM:-16}" # learned pair-embed dim (embed pairing only)
SEQ_POOL="${SEQ_POOL:-gem}"            # avg | gem
ACTIVATION="${ACTIVATION:-relu}"       # leaky_relu | relu | gelu | silu | elu | selu
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"      # 0 = BCE; 2 = standard focal
DETERMINISTIC="${DETERMINISTIC:-1}"    # 1 -> --deterministic

# ── optional feature flags (off by default) ────────────────────────────────
EMA="${EMA:-1}"        # 1 -> --ema

# Shared feature flags (used by every mode's training argv).
feature_flags=()
[[ "$EMA" == "1" ]] && feature_flags+=( --ema )

# Architecture / loss flags. Not passed to predict — the design is baked into
# the checkpoint.
model_flags=(
  --seq-pairing "$SEQ_PAIRING" --seq-pool "$SEQ_POOL"
  --activation "$ACTIVATION" --focal-gamma "$FOCAL_GAMMA"
)
[[ "$SEQ_PAIRING" == "embed" ]] && model_flags+=( --pair-embed-dim "$PAIR_EMBED_DIM" )
[[ "$DETERMINISTIC" == "1" ]]   && model_flags+=( --deterministic )

# ── assemble argv (kfold | full) ───────────────────────────────────────────
args=(
  "$SCRIPT" train
  --train "$TRAIN"
  --out "$OUT"
  --mre-col "$MRE_COL" --mirna-col "$MIRNA_COL" --family-col "$FAMILY_COL"
  --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR"
  --seed "$SEED" --device "$DEVICE" --checkpoint-metric "$METRIC"
)

# shellcheck disable=SC2206
test_arr=( $TESTS )
[[ ${#test_arr[@]} -gt 0 ]] && args+=( --test "${test_arr[@]}" )

args+=( "${feature_flags[@]}" "${model_flags[@]}" )

case "$MODE" in
  kfold)
    args+=( --folds "$FOLDS" )
    [[ -n "$OOF_OUT" ]] && args+=( --oof-out "$OOF_OUT" )
    ;;
  full)  args+=( --no-val ) ;;
  *) echo "ERROR: mode must be 'kfold' or 'full' (got '$MODE')" >&2; exit 2 ;;
esac

echo "[run_train] mode=$MODE  train=$TRAIN  out=$OUT"
echo "[run_train] $PY ${args[*]}"
exec "$PY" "${args[@]}"
