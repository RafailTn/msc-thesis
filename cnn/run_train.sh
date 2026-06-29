#!/usr/bin/env bash
# Thin wrapper around cnn/cnn_branches_mirbind.py that trains the CNN either with
# stratified-group k-fold CV or once on the entire training set (no validation),
# and can prepare the leakage-free neighbour-count column.
#
#   bash cnn/run_train.sh kfold      # K-fold CV (default)  -> *_fold{1..K}.pt
#   bash cnn/run_train.sh full       # one run on all of --train, no val split
#   bash cnn/run_train.sh nbr-prep   # base k-fold -> OOF preds -> neighbor_count
#   bash cnn/run_train.sh infer      # base scorer -> neighbor_count -> final model
#   bash cnn/run_train.sh report     # compare neighbor_count distros (train vs infer)
#
# infer scores new data end to end (e.g. test/leftout/genome scan):
#   INPUT=data/..._test_v7.tsv FINAL=checkpoints/cnn_nbr.pt \
#     bash cnn/run_train.sh infer
# It uses the SAME conf/window/min_sep band as nbr-prep so the count
# distribution matches training. BASE_CKPTS defaults to the nbr-prep base
# fold-ensemble; CON/ACC must match how the base + final models were trained.
#
# Everything is overridable via environment variables, e.g.
#   FOLDS=10 EPOCHS=60 bash cnn/run_train.sh kfold
#   NBR=1 CON=1 OUT=checkpoints/cnn_nbr.pt bash cnn/run_train.sh full
#
# Neighbour feature (NBR=1): the --train file must already carry the
# `neighbor_count` column, leakage-free ONLY if built from OUT-OF-FOLD scores.
# `nbr-prep` automates that: it trains a base model (no nbr feature) with k-fold,
# writes per-row out-of-fold predictions, then materialises the neighbour-count
# column into ${NBR_TRAIN}. Afterwards re-train with that file:
#   bash cnn/run_train.sh nbr-prep
#   NBR=1 TRAIN=<NBR_TRAIN> OUT=checkpoints/cnn_nbr.pt bash cnn/run_train.sh kfold
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

# ── v7 TSV column mapping (gene = MRE sequence) ────────────────────────────
MRE_COL="${MRE_COL:-gene}"
MIRNA_COL="${MIRNA_COL:-noncodingRNA}"
FAMILY_COL="${FAMILY_COL:-noncodingRNA_fam}"
CON_COL="${CON_COL:-gene_phyloP}"
ACC_COL="${ACC_COL:-tAcc}"
NBR_COL="${NBR_COL:-neighbor_count}"

# ── training hyper-params ──────────────────────────────────────────────────
EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH:-256}"
LR="${LR:-1e-3}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
METRIC="${METRIC:-auprc}"

# ── optional feature channels (off by default) ─────────────────────────────
NBR="${NBR:-0}"        # 1 -> --seq-nbr-feature (needs neighbor_count column)
CON="${CON:-0}"        # 1 -> --seq-con-channel  (conservation, gene_phyloP)
ACC="${ACC:-0}"        # 1 -> --seq-acc-channel  (accessibility, needs tAcc col)
EMA="${EMA:-1}"        # 1 -> --ema

# ── nbr-prep params (build the leakage-free neighbour-count column) ─────────
OOF_OUT="${OOF_OUT:-${TRAIN%.tsv}_oof.tsv}"     # train + out-of-fold preds
NBR_TRAIN="${NBR_TRAIN:-${TRAIN%.tsv}_nbr.tsv}" # train + neighbor_count column
BASE_OUT="${BASE_OUT:-${OUT%.pt}_base.pt}"      # base (no-nbr) fold checkpoints
CONF="${CONF:-0.8}"
WINDOW="${WINDOW:-150}"
MIN_SEP="${MIN_SEP:-60}"
CHR_COL="${CHR_COL:-chr}"
STRAND_COL="${STRAND_COL:-strand}"
START_COL="${START_COL:-start}"
END_COL="${END_COL:-end}"

# ── infer params (two-pass scoring on new data) ────────────────────────────
INPUT="${INPUT:-}"                              # data to score (required for infer)
FINAL="${FINAL:-$OUT}"                          # final neighbour-model checkpoint
INFER_OUT="${INFER_OUT:-}"                      # final predictions output
# Base scorer for the column: the k-fold base checkpoints from `nbr-prep`
# (ensemble). Override with a single full-trained base, or a custom glob/list.
BASE_CKPTS="${BASE_CKPTS:-${BASE_OUT%.pt}_fold*.pt}"

# Shared feature flags (used by every mode's training argv).
feature_flags=()
[[ "$EMA" == "1" ]] && feature_flags+=( --ema )
[[ "$CON" == "1" ]] && feature_flags+=( --seq-con-channel --con-col "$CON_COL" --con-transform robust )
[[ "$ACC" == "1" ]] && feature_flags+=( --seq-acc-channel )

# Channel column names a checkpoint needs at predict time (the channels
# themselves are baked into the checkpoint; predict only needs the columns).
predict_chan=()
[[ "$CON" == "1" ]] && predict_chan+=( --con-col "$CON_COL" )
[[ "$ACC" == "1" ]] && predict_chan+=( --acc-col "$ACC_COL" )

# ── nbr-prep: base k-fold (OOF preds) -> neighbor-counts column -------------
if [[ "$MODE" == "nbr-prep" ]]; then
  base_args=(
    "$SCRIPT" train
    --train "$TRAIN" --out "$BASE_OUT"
    --mre-col "$MRE_COL" --mirna-col "$MIRNA_COL" --family-col "$FAMILY_COL"
    --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR"
    --seed "$SEED" --device "$DEVICE" --checkpoint-metric "$METRIC"
    --folds "$FOLDS" --oof-out "$OOF_OUT"
    "${feature_flags[@]}"
  )
  # shellcheck disable=SC2206
  test_arr=( $TESTS )
  [[ ${#test_arr[@]} -gt 0 ]] && base_args+=( --test "${test_arr[@]}" )

  echo "[nbr-prep] 1/2 base k-fold ($FOLDS folds) -> OOF preds: $OOF_OUT"
  echo "[nbr-prep] $PY ${base_args[*]}"
  "$PY" "${base_args[@]}"

  nc_args=(
    "$SCRIPT" neighbor-counts
    --input "$OOF_OUT" --output "$NBR_TRAIN"
    --score-col interaction_probability
    --conf "$CONF" --window "$WINDOW" --min-sep "$MIN_SEP"
    --out-col "$NBR_COL"
    --chr-col "$CHR_COL" --strand-col "$STRAND_COL"
    --start-col "$START_COL" --end-col "$END_COL"
  )
  echo "[nbr-prep] 2/2 neighbor-counts (conf>=$CONF, band [$MIN_SEP,$WINDOW]) -> $NBR_TRAIN"
  echo "[nbr-prep] $PY ${nc_args[*]}"
  "$PY" "${nc_args[@]}"

  echo "[nbr-prep] done. Train the neighbour model with:"
  echo "    NBR=1 TRAIN=$NBR_TRAIN OUT=checkpoints/cnn_nbr.pt bash cnn/run_train.sh kfold"
  exit 0
fi

# ── infer: base scorer -> neighbor-counts -> final neighbour model ─────────
if [[ "$MODE" == "infer" ]]; then
  [[ -n "$INPUT" ]] || { echo "ERROR: infer needs INPUT=<data.tsv>" >&2; exit 2; }
  [[ -e "$FINAL" ]] || { echo "ERROR: final model FINAL='$FINAL' not found "    \
                              "(set FINAL=checkpoints/cnn_nbr.pt)" >&2; exit 2; }
  : "${INFER_OUT:=${INPUT%.tsv}_nbrpred.tsv}"
  workdir="$(dirname "$INFER_OUT")"; mkdir -p "$workdir"
  base="${INPUT%.tsv}"
  scored="${base}_basescore.tsv"      # input + base interaction_probability
  withnbr="${base}_withnbr.tsv"       # + neighbor_count column

  # shellcheck disable=SC2206
  base_ckpts=( $BASE_CKPTS )
  [[ ${#base_ckpts[@]} -ge 1 && -e "${base_ckpts[0]}" ]] \
    || { echo "ERROR: no base checkpoints matched BASE_CKPTS='$BASE_CKPTS'" >&2; exit 2; }

  # 1/3 — base scoring (ensemble if >1 checkpoint, else single predict)
  if [[ ${#base_ckpts[@]} -gt 1 ]]; then
    echo "[infer] 1/3 base ensemble (${#base_ckpts[@]} ckpts) scoring $INPUT"
    pe_args=(
      "$SCRIPT" predict-ensemble
      --checkpoints "${base_ckpts[@]}" --inputs "$INPUT" --output-dir "$workdir"
      --mre-col "$MRE_COL" --mirna-col "$MIRNA_COL"
      --batch-size "$BATCH" --device "$DEVICE" "${predict_chan[@]}"
    )
    echo "[infer] $PY ${pe_args[*]}"
    "$PY" "${pe_args[@]}"
    scored="$workdir/$(basename "$base")_ensemble.tsv"
  else
    echo "[infer] 1/3 base (single ckpt ${base_ckpts[0]}) scoring $INPUT"
    pr_args=(
      "$SCRIPT" predict --checkpoint "${base_ckpts[0]}"
      --input "$INPUT" --output "$scored"
      --mre-col "$MRE_COL" --mirna-col "$MIRNA_COL"
      --batch-size "$BATCH" --device "$DEVICE" "${predict_chan[@]}"
    )
    echo "[infer] $PY ${pr_args[*]}"
    "$PY" "${pr_args[@]}"
  fi

  # 2/3 — materialise neighbor_count (SAME band/conf as nbr-prep training)
  nc_args=(
    "$SCRIPT" neighbor-counts --input "$scored" --output "$withnbr"
    --score-col interaction_probability
    --conf "$CONF" --window "$WINDOW" --min-sep "$MIN_SEP" --out-col "$NBR_COL"
    --chr-col "$CHR_COL" --strand-col "$STRAND_COL"
    --start-col "$START_COL" --end-col "$END_COL"
  )
  echo "[infer] 2/3 neighbor-counts (conf>=$CONF, band [$MIN_SEP,$WINDOW]) -> $withnbr"
  echo "[infer] $PY ${nc_args[*]}"
  "$PY" "${nc_args[@]}"

  # 3/3 — final neighbour model consumes the column
  final_args=(
    "$SCRIPT" predict --checkpoint "$FINAL"
    --input "$withnbr" --output "$INFER_OUT" --nbr-col "$NBR_COL"
    --mre-col "$MRE_COL" --mirna-col "$MIRNA_COL"
    --batch-size "$BATCH" --device "$DEVICE" "${predict_chan[@]}"
  )
  echo "[infer] 3/3 final neighbour model ($FINAL) -> $INFER_OUT"
  echo "[infer] $PY ${final_args[*]}"
  "$PY" "${final_args[@]}"

  echo "[infer] done -> $INFER_OUT  (intermediates: $scored, $withnbr)"
  exit 0
fi

# ── report: compare neighbour-count distributions (train vs inference) ─────
if [[ "$MODE" == "report" ]]; then
  # REPORT_FILES defaults to the train column + the test/leftout *_withnbr.tsv.
  default_files="$NBR_TRAIN"
  for t in $TESTS; do default_files="$default_files ${t%.tsv}_withnbr.tsv"; done
  REPORT_FILES="${REPORT_FILES:-$default_files}"
  # shellcheck disable=SC2206
  files=( $REPORT_FILES )
  present=()
  for f in "${files[@]}"; do [[ -e "$f" ]] && present+=( "$f" ) || echo "[report] skip (missing): $f"; done
  [[ ${#present[@]} -ge 1 ]] || { echo "ERROR: no report files found (set REPORT_FILES=...)" >&2; exit 2; }
  rp_args=( "$SCRIPT" nbr-report --inputs "${present[@]}" --nbr-col "$NBR_COL" )
  echo "[report] $PY ${rp_args[*]}"
  exec "$PY" "${rp_args[@]}"
fi

# ── assemble argv (kfold | full) ───────────────────────────────────────────
args=(
  "$SCRIPT" train
  --train "$TRAIN"
  --out "$OUT"
  --mre-col "$MRE_COL" --mirna-col "$MIRNA_COL" --family-col "$FAMILY_COL"
  --nbr-col "$NBR_COL"
  --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR"
  --seed "$SEED" --device "$DEVICE" --checkpoint-metric "$METRIC"
)

# shellcheck disable=SC2206
test_arr=( $TESTS )
[[ ${#test_arr[@]} -gt 0 ]] && args+=( --test "${test_arr[@]}" )

args+=( "${feature_flags[@]}" )
[[ "$NBR" == "1" ]] && args+=( --seq-nbr-feature )

case "$MODE" in
  kfold) args+=( --folds "$FOLDS" ) ;;
  full)  args+=( --no-val ) ;;
  *) echo "ERROR: mode must be 'kfold', 'full', 'nbr-prep', 'infer' or 'report' (got '$MODE')" >&2; exit 2 ;;
esac

echo "[run_train] mode=$MODE  train=$TRAIN  out=$OUT  nbr=$NBR con=$CON acc=$ACC"
echo "[run_train] $PY ${args[*]}"
exec "$PY" "${args[@]}"
