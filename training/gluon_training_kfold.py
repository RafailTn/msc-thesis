import argparse
import ast
import json
import sys
import pandas as pd
import polars as pl
import numpy as np
from autogluon.tabular import TabularPredictor
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from autogluon.core.metrics import make_scorer
from sklearn.metrics import fbeta_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold
from typing import List, Optional, Dict, Tuple

# Shared with gluon_train_total.py so the two entry points cannot disagree about what a
# named feature set means; the definitions themselves live in src/feature_extraction.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_sets import (  # noqa: E402
    FEATURE_SETS, feature_set, select_feature_columns, COLS2DROP, SEQUENCE_COLS,
)


def get_misclassified(
    df_with_sequences: pd.DataFrame, 
    df_for_prediction: pd.DataFrame, 
    predictor: TabularPredictor, 
    fold: int, 
    dataset_name: str, 
    output_dir: str,
    save: bool = True
) -> pd.DataFrame:
    """
    Identify and save misclassified samples with their original sequences.
    
    Args:
        df_with_sequences: Original dataframe WITH sequence columns preserved
        df_for_prediction: Processed dataframe for prediction (without sequence cols)
        predictor: Trained AutoGluon predictor
        fold: Current fold number
        dataset_name: Name for the output file (e.g., 'train', 'val', 'test')
        output_dir: Directory to save results
        save: Whether to save to CSV
        
    Returns:
        DataFrame of misclassified samples with predictions
    """
    y_true = df_for_prediction['label'].values
    y_pred = predictor.predict(df_for_prediction).values
    y_pred_proba = predictor.predict_proba(df_for_prediction)[1].values
    
    # Find misclassified indices
    misclassified_mask = y_true != y_pred
    
    # Get misclassified samples from the original df (with sequences)
    misclassified_df = df_with_sequences.loc[misclassified_mask].copy()
    misclassified_df['predicted_label'] = y_pred[misclassified_mask]
    misclassified_df['predicted_proba'] = y_pred_proba[misclassified_mask]
    misclassified_df['fold'] = fold
    misclassified_df['dataset'] = dataset_name
    
    # Also categorize error type
    misclassified_df['error_type'] = np.where(
        misclassified_df['label'] == 1, 
        'false_negative',  # Was positive, predicted negative
        'false_positive'   # Was negative, predicted positive
    )
    
    if save:
        os.makedirs(output_dir, exist_ok=True)
        outpath = os.path.join(output_dir, f'misclassified_{dataset_name}_fold{fold}.csv')
        misclassified_df.to_csv(outpath, index=False)
        print(f"Saved {len(misclassified_df)} misclassified samples to {outpath}")
    
    # Print summary
    n_fp = (misclassified_df['error_type'] == 'false_positive').sum()
    n_fn = (misclassified_df['error_type'] == 'false_negative').sum()
    print(f"  {dataset_name} fold {fold}: {len(misclassified_df)} errors ({n_fp} FP, {n_fn} FN)")
    
    return misclassified_df


def aggregate_misclassified(output_dir: str, n_folds: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Aggregate misclassified samples across all folds and identify consistently misclassified samples.
    
    Args:
        output_dir: Directory containing misclassified CSV files
        n_folds: Number of folds
        
    Returns:
        Dictionary with aggregated DataFrames for each dataset
    """
    results = {}
    
    for dataset_name in ['train', 'val', 'test', 'final_test']:
        files = [
            os.path.join(output_dir, f'misclassified_{dataset_name}_fold{i}.csv')
            for i in range(n_folds)
        ]
        existing_files = [f for f in files if os.path.exists(f)]
        
        if not existing_files:
            continue
            
        all_misclassified = pd.concat([pd.read_csv(f) for f in existing_files], ignore_index=True)
        
        # Count how often each sequence was misclassified
        misclass_counts = all_misclassified.groupby('chimeric_sequence').agg({
            'fold': 'count',
            'predicted_proba': ['mean', 'std'],
            'error_type': lambda x: x.mode().iloc[0] if len(x) > 0 else None,
            'label': 'first'
        }).reset_index()
        
        misclass_counts.columns = [
            'chimeric_sequence', 'times_misclassified', 
            'mean_pred_proba', 'std_pred_proba', 
            'dominant_error_type', 'true_label'
        ]
        misclass_counts = misclass_counts.sort_values('times_misclassified', ascending=False)
        
        # Save aggregated results
        agg_path = os.path.join(output_dir, f'aggregated_misclassified_{dataset_name}.csv')
        misclass_counts.to_csv(agg_path, index=False)
        
        # Also save consistently misclassified (3+ folds for val, always for test)
        if dataset_name == 'val':
            consistent = misclass_counts[misclass_counts['times_misclassified'] >= 3]
            consistent_path = os.path.join(output_dir, f'consistently_misclassified_{dataset_name}.csv')
            consistent.to_csv(consistent_path, index=False)
            print(f"\n{dataset_name}: {len(consistent)} samples misclassified in 3+ folds")
        
        results[dataset_name] = misclass_counts
        print(f"Aggregated {len(all_misclassified)} total misclassifications for {dataset_name}")
    
    return results


def evaluate_df(df: pd.DataFrame, predictor: TabularPredictor) -> Tuple[dict, float]:
    """Evaluate a dataframe and return metrics including average precision score."""
    eval_metrics = predictor.evaluate(df)
    y_pred_proba = predictor.predict_proba(df)[1]
    y_true = df['label']
    ap_score = average_precision_score(y_true, y_pred_proba)
    return eval_metrics, ap_score


def evaluate_gluon(
    train_data: pd.DataFrame, 
    val_data: pd.DataFrame, 
    test_data: pd.DataFrame, 
    additional_test_data: Optional[List[pd.DataFrame]], 
    outf_path: str, 
    predictor: TabularPredictor,
    fold: int,
    decision_threshold: float|None = None
) -> Dict[str, Tuple[dict, float]]:
    """Evaluate predictor on all datasets, write per-fold results, and return them.

    The returned dict is keyed by the same dataset labels used in the per-fold lines
    ('Train', 'Val', 'Small_test', 'Final_test_{i}'), each mapping to
    (metric_dict, average_precision). main() accumulates these across folds to write a
    mean/std block, so the aggregate lines line up with the per-fold ones.
    """
    if decision_threshold:
        predictor.set_decision_threshold(decision_threshold)
    else:
        calibrated_threshold = predictor.calibrate_decision_threshold(metric="f1")
        predictor.set_decision_threshold(calibrated_threshold)
    train_eval, ap_score_train = evaluate_df(train_data, predictor)
    fold_eval, ap_score_eval = evaluate_df(val_data, predictor)
    test_eval, ap_score_test = evaluate_df(test_data, predictor)

    results: Dict[str, Tuple[dict, float]] = {
        'Train': (train_eval, ap_score_train),
        'Val': (fold_eval, ap_score_eval),
        'Small_test': (test_eval, ap_score_test),
    }

    with open(outf_path, 'a') as outfile:
        outfile.write(f"Train_results_fold{fold}: {train_eval}, APS: {ap_score_train}\n")
        outfile.write(f"Val_results_fold{fold}: {fold_eval}, APS: {ap_score_eval}\n")
        outfile.write(f"Small_test_results_fold{fold}: {test_eval}, APS: {ap_score_test}\n")

        if additional_test_data:
            for i, add_test in enumerate(additional_test_data):
                add_test_eval, ap_score_add = evaluate_df(add_test, predictor)
                outfile.write(f"Final_test_{i}_results_fold{fold}: {add_test_eval}, APS: {ap_score_add}\n")
                results[f'Final_test_{i}'] = (add_test_eval, ap_score_add)

    return results


def write_metric_means(
    outf_path: str,
    fold_results: List[Dict[str, Tuple[dict, float]]],
) -> None:
    """Append a cross-fold mean/std block for every metric to the results file.

    `fold_results` is the per-fold output of evaluate_gluon. For each dataset and each
    metric key, the mean and population std are taken across the folds that reported it
    (a metric absent from a fold is skipped, not counted as zero), plus the average
    precision. The lines mirror the per-fold format - a metric dict then `APS:` - so the
    same parser reads both.
    """
    if not fold_results:
        return

    # Preserve the dataset order of the first fold rather than sorting, so the block reads
    # train / val / test / leftout like the per-fold lines above it.
    datasets: List[str] = list(fold_results[0].keys())

    with open(outf_path, 'a') as outfile:
        outfile.write(f"\n{'='*60}\n")
        outfile.write(f"MEAN +/- STD ACROSS {len(fold_results)} FOLDS\n")
        outfile.write(f"{'='*60}\n")
        for ds in datasets:
            metric_dicts = [fr[ds][0] for fr in fold_results if ds in fr]
            aps_values = [fr[ds][1] for fr in fold_results if ds in fr]
            keys = sorted({k for d in metric_dicts for k in d})

            mean_dict, std_dict = {}, {}
            for k in keys:
                vals = [d[k] for d in metric_dicts
                        if k in d and d[k] is not None and not pd.isna(d[k])]
                if vals:
                    mean_dict[k] = float(np.mean(vals))
                    std_dict[k] = float(np.std(vals))

            aps_mean = float(np.mean(aps_values)) if aps_values else float('nan')
            aps_std = float(np.std(aps_values)) if aps_values else float('nan')
            outfile.write(f"{ds}_mean: {mean_dict}, APS: {aps_mean}\n")
            outfile.write(f"{ds}_std:  {std_dict}, APS: {aps_std}\n")


def preprocess_dataframe(
    df: pd.DataFrame,
    cols2drop: List[str],
    sequence_cols: List[str],
    features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess dataframe: drop duplicates, drop columns, preserve sequences.

    `features` None keeps every column except cols2drop (the historical behaviour);
    a list selects exactly those instead.

    Returns:
        Tuple of (processed_df, df_with_sequences)
    """
    df = df.drop_duplicates(subset=['chimeric_sequence'], keep=False)
    # Preserve sequences BEFORE dropping columns
    available_seq_cols = [c for c in sequence_cols if c in df.columns]
    df_with_sequences = df[available_seq_cols + ['label']].copy()
    if features is not None:
        df = select_feature_columns(df, features)
    else:
        # Drop columns that exist in the dataframe
        cols_to_drop = [c for c in cols2drop if c in df.columns]
        df = df.drop(columns=cols_to_drop)
    family_counts = df['mir_fam'].value_counts().clip(lower=100)
    total_samples = len(df)
    # Weight = Total / (n_families * count_of_this_family)
    # This scales weights so they sum up to roughly len(df)
    n_families = len(family_counts)
    df['weights'] = df['mir_fam'].map(lambda x: total_samples / (n_families * family_counts[x])) 
    df['label'] = df['label'].astype(int)
    return df, df_with_sequences


def main(
    train_df_path: str,
    test_df_path: str,
    leftout_df_path: str,
    model_path: str,
    label_col: str,
    eval_metric: str,
    time_limit: int,
    misclassified_output_dir: str,
    results_output_path: str,
    n_folds: int = 5,
    feature_set_name: str = 'all',
    features_json: str = None,
):
    print("Starting Training...")

    if features_json is not None:
        # A/B a fresh feature_selection.py list without first promoting it into
        # SELECTED_FEATURES. The JSON path is the traceable record of what was compared.
        with open(features_json) as fh:
            features = json.load(fh)
        print(f"Feature set: {features_json} ({len(features)} features from JSON)")
    else:
        features = feature_set(feature_set_name)
        print(f"Feature set: {feature_set_name} "
              f"({'all columns present' if features is None else str(len(features)) + ' features'})")

    cols2drop = COLS2DROP
    # Sequence columns to preserve for misclassification analysis
    sequence_cols = SEQUENCE_COLS

    # Load and preprocess training data
    df_raw = pd.read_csv(train_df_path)
    df, df_with_sequences = preprocess_dataframe(df_raw, cols2drop, sequence_cols, features)
    df_pl = pl.from_pandas(df)

    # Load and preprocess test data
    final_test_raw = pd.read_csv(test_df_path)
    final_test_data, final_test_with_seq = preprocess_dataframe(
        final_test_raw, cols2drop, sequence_cols, features
    )

    final_final_test_raw = pd.read_csv(leftout_df_path)
    final_final_test_data, final_final_test_with_seq = preprocess_dataframe(
        final_final_test_raw, cols2drop, sequence_cols, features
    )
    # Setup cross-validation
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    y = df_pl['label'].to_numpy()
    groups = df_pl['mir_fam'].to_numpy()
    X = df_pl.to_numpy()
    
    # Create output directory for misclassified samples
    os.makedirs(misclassified_output_dir, exist_ok=True)
    
    # Store all misclassified for aggregation
    all_misclassified = {
        'train': [],
        'val': [],
        'test': [],
        'final_test': []
    }

    # Per-fold metric dicts, accumulated to write cross-fold means at the end.
    fold_results: List[Dict[str, Tuple[dict, float]]] = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}")
        print(f"{'='*60}")
        # Split data
        train_fold = df_pl[train_idx].drop('mir_fam')
        val_fold = df_pl[val_idx].drop('mir_fam')
        # Get corresponding sequence dataframes (reset index to match)
        train_with_seq = df_with_sequences.iloc[train_idx].reset_index(drop=True)
        val_with_seq = df_with_sequences.iloc[val_idx].reset_index(drop=True)
        # Train predictor
        predictor = TabularPredictor(
            label=label_col,
            eval_metric=eval_metric,
            sample_weight='weights',
            weight_evaluation=True,
            path=f'{model_path}{fold}',
        )
        
        predictor.fit(
            train_data=train_fold.to_pandas(),
            tuning_data=val_fold.to_pandas(),
            presets='best_quality',
            num_bag_folds=0,
            num_stack_levels=0,
            time_limit=time_limit,
        )
        
        # Convert to pandas for evaluation
        train_fold_pd = train_fold.to_pandas()
        val_fold_pd = val_fold.to_pandas()
        
        # Evaluate and save metrics
        fold_results.append(evaluate_gluon(
            train_data=train_fold_pd,
            val_data=val_fold_pd,
            test_data=final_test_data,
            additional_test_data=[final_final_test_data],
            outf_path=results_output_path,
            predictor=predictor,
            fold=fold
        ))
        
        # Get misclassified samples for each dataset
        print(f"\nMisclassification Analysis for Fold {fold}:")
        
        misclass_train = get_misclassified(
            train_with_seq, train_fold_pd, predictor, fold, 'train', misclassified_output_dir
        )
        all_misclassified['train'].append(misclass_train)
        
        misclass_val = get_misclassified(
            val_with_seq, val_fold_pd, predictor, fold, 'val', misclassified_output_dir
        )
        all_misclassified['val'].append(misclass_val)
        
        misclass_test = get_misclassified(
            final_test_with_seq.reset_index(drop=True), 
            final_test_data, 
            predictor, fold, 'test', misclassified_output_dir
        )
        all_misclassified['test'].append(misclass_test)
        
        misclass_final = get_misclassified(
            final_final_test_with_seq.reset_index(drop=True), 
            final_final_test_data, 
            predictor, fold, 'final_test', misclassified_output_dir
        )
        all_misclassified['final_test'].append(misclass_final)
    
    # Cross-fold metric means, appended to the same results file as the per-fold lines.
    write_metric_means(results_output_path, fold_results)

    # Aggregate results across folds
    print(f"\n{'='*60}")
    print("AGGREGATING MISCLASSIFIED SAMPLES ACROSS FOLDS")
    print(f"{'='*60}")
    
    aggregated_results = aggregate_misclassified(misclassified_output_dir, n_folds=n_folds)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for dataset_name, agg_df in aggregated_results.items():
        if len(agg_df) > 0:
            print(f"\n{dataset_name.upper()}:")
            print(f"  Total unique misclassified sequences: {len(agg_df)}")
            print(f"  Most frequently misclassified (top 5):")
            for _, row in agg_df.head(5).iterrows():
                print(f"    - {row['chimeric_sequence'][:50]}... "
                      f"({row['times_misclassified']} times, "
                      f"avg prob: {row['mean_pred_proba']:.3f}, "
                      f"type: {row['dominant_error_type']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='K-fold AutoGluon training on the selected-feature CSVs produced by '
                    'src/feature_extraction.py.')
    parser.add_argument('--input', type=str, required=True,
                        help='Training CSV (selected features)')
    parser.add_argument('--test', type=str, required=True,
                        help='Held-out test CSV, scored every fold')
    parser.add_argument('--leftout', type=str, required=True,
                        help='Leftout CSV, scored every fold')
    parser.add_argument('--modelpath', type=str, default='models/gluon_fold',
                        help='Model path prefix; the fold number is appended')
    parser.add_argument('--label', type=str, default='label',
                        help='Name of the column label')
    parser.add_argument('--metric', type=str, default='f1',
                        help='Metric to use for evaluation')
    parser.add_argument('--time', type=int, help='Time the gluon runs in seconds')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--misclassified_dir', type=str,
                        default='results/misclassified_analysis/',
                        help='Directory to save misclassified samples')
    parser.add_argument('--results_path', type=str,
                        default='results/gluon_kfold_results.txt',
                        help='Path to save evaluation results')
    parser.add_argument('--feature-set', type=str, default='all',
                        choices=sorted(FEATURE_SETS),
                        help="Which columns to train on. 'all' (default) keeps every "
                             "column in the CSV except cols2drop; 'baseline' is the "
                             "featurewiz selection alone; 'baseline+new' adds the "
                             "candidates featurewiz has not yet judged. With a "
                             "default-mode CSV 'all' and 'baseline+new' coincide. "
                             "Defined in src/feature_extraction.FEATURE_SETS.")
    parser.add_argument('--features-json', type=str, default=None,
                        help="Cross-validate on exactly the list in this JSON file (a "
                             "feature_selection.py --output). Overrides --feature-set, so "
                             "you can A/B a fresh selection without editing "
                             "SELECTED_FEATURES first.")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.results_path) or '.', exist_ok=True)

    main(
        train_df_path=args.input,
        test_df_path=args.test,
        leftout_df_path=args.leftout,
        model_path=args.modelpath,
        label_col=args.label,
        eval_metric=args.metric,
        time_limit=args.time,
        misclassified_output_dir=args.misclassified_dir,
        results_output_path=args.results_path,
        n_folds=args.folds,
        feature_set_name=args.feature_set,
        features_json=args.features_json,
    )
