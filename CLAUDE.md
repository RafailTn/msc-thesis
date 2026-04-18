# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ML pipeline for predicting miRNA target sites (MREs). It trains and runs an AutoGluon tabular model on features derived from RNA-RNA interaction folding (IntaRNA), sequence properties, and conservation scores. The dataset comes from AGO2-eCLIP experiments (Manakov et al. 2022), preprocessed by Gresova et al. 2025. The model achieves ~83–84% Average Precision and ~81% ROC-AUC.

## Setup

Uses [Pixi](https://pixi.sh/) as the package manager (Python 3.12, conda + pip):

```bash
cd dependencies
pixi install
pixi shell
```

Key dependencies: AutoGluon 1.4.0, IntaRNA 3.4.1, Polars, Pandas, PyBigWig, PyRanges, SHAP.

## Running the Pipeline

```bash
python src/predict_target.py \
  -target_fasta <MRE.fasta> \
  -query_fasta <miRNA.fasta> \
  -conservation_tsv <coords.tsv> \
  -model models/autogluon_final_model \
  -o results.tsv \
  [--bigwig phastcons.bw] \
  [--threshold 0.5] \
  [--explain] \
  [--threads 4] \
  [--keep_files]
```

- Conservation TSV can include pre-extracted scores, or a BigWig (hg38 phastCons470way) can be provided/auto-downloaded from UCSC.
- `--explain` computes SHAP values (caps at 200 positive predictions sampled).
- `--keep_files` preserves intermediate files for debugging.

## Architecture & Data Flow

`predict_target.py` orchestrates the full pipeline in sequence:

1. **IntaRNA** (`intarna_parallel.py`) — runs RNA-RNA interaction folding in two modes in parallel using Python threads: MFE (minimum free energy, for structure) and Ensemble (suboptimal interactions, for thermodynamic ensemble energies).
2. **Merge** (`merge_intarna.py`) — joins MFE structural output with ensemble energy output on sequence pair identifiers.
3. **Best duplex selection** (`best_intarna.py`) — scores each candidate interaction using biological criteria (seed region match at positions 2–8, G:U wobbles, binding type classification) and keeps the top-ranked duplex per pair.
4. **Feature extraction** (`feature_extraction.py`) — streaming CSV writer that computes 45+ features: IntaRNA structural features, seed region properties, miRNA 3′ and MRE 5′/3′ region features, conservation scores (BigWig or TSV), one-hot binding type encoding, and dinucleotide stacking energies (Turner nearest-neighbor model).
5. **Prediction** — AutoGluon `TabularPredictor` loaded from `models/autogluon_final_model/` predicts interaction probability.
6. **Explainability** (optional) — SHAP analysis and permutation importance on predicted positives.

## Training

```bash
# Full dataset training
python training/gluon_train_total.py

# K-fold cross-validation with misclassification tracking
python training/gluon_training_kfold.py
```

## Key Directories

| Path | Purpose |
|------|---------|
| `src/` | Core pipeline scripts |
| `training/` | Model training and cross-validation |
| `models/autogluon_final_model/` | Trained AutoGluon predictor (directory, not a file) |
| `mamba/` | Experimental PyTorch/Mamba deep learning (not in main pipeline) |
| `rnaduplex_features/` | Alternative feature source (RNAduplex), not used in main pipeline |
| `tarbase_cleaning_and_matching/` | TarBase validation data preprocessing for false-negative correction |
| `feature_selection_featurewiz/` | Feature engineering experiments |
| `report/` | LaTeX thesis manuscript |
| `dependencies/` | Pixi environment definition |

## Design Notes

- Processing is **streaming** (row-by-row CSV writing in feature extraction) to keep memory footprint low for large datasets.
- IntaRNA is called as a subprocess; parallelism is thread-based via `intarna_parallel.py`.
- The model directory `models/autogluon_final_model/` must be passed as a path (AutoGluon loads its own internal structure from that directory).
- There is no formal test suite — this is a research/thesis project.
