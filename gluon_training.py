import argparse
import ast
import pandas as pd
import polars as pl
import numpy as np
from autogluon.tabular import TabularPredictor
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from autogluon.core.metrics import make_scorer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold
from typing import List, Optional, Dict, Tuple


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
) -> None:
    """Evaluate predictor on all datasets and write results to file."""
    if decision_threshold:
        predictor.set_decision_threshold(decision_threshold)
    else:
        calibrated_threshold = predictor.calibrate_decision_threshold(metric="f1")
        predictor.set_decision_threshold(calibrated_threshold)
    train_eval, ap_score_train = evaluate_df(train_data, predictor)
    fold_eval, ap_score_eval = evaluate_df(val_data, predictor)
    test_eval, ap_score_test = evaluate_df(test_data, predictor)
    
    with open(outf_path, 'a') as outfile:
        outfile.write(f"Train_results_fold{fold}: {train_eval}, APS: {ap_score_train}\n")
        outfile.write(f"Val_results_fold{fold}: {fold_eval}, APS: {ap_score_eval}\n")
        outfile.write(f"Small_test_results_fold{fold}: {test_eval}, APS: {ap_score_test}\n")
        
        if additional_test_data:
            for i, add_test in enumerate(additional_test_data):
                add_test_eval, ap_score_add = evaluate_df(add_test, predictor)
                outfile.write(f"Final_test_{i}_results_fold{fold}: {add_test_eval}, APS: {ap_score_add}\n")


def preprocess_dataframe(
    df: pd.DataFrame, 
    cols2drop: List[str], 
    sequence_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess dataframe: drop duplicates, drop columns, preserve sequences.
    
    Returns:
        Tuple of (processed_df, df_with_sequences)
    """
    df = df.drop_duplicates(subset=['chimeric_sequence'], keep=False)
    # Preserve sequences BEFORE dropping columns
    available_seq_cols = [c for c in sequence_cols if c in df.columns]
    df_with_sequences = df[available_seq_cols + ['label']].copy()
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
    model_path: str,
    label_col: str, 
    eval_metric: str,
    time_limit: int,
    misclassified_output_dir: str = '/home/adam/adam/data/misclassified_analysis/',
    results_output_path: str = '/home/adam/adam/data/manakov_results_seqfs.txt',
):
    print("Starting Training...")
   
    cols2drop = [
        'conservation_range','target_id', 'query_id', 
        'contrafold_struct', 'hybrid_dp', 'subseq_dp', 
        'mre_sequence', 'mirna_sequence', 'chimeric_sequence', 
        'gene' , 'noncodingRNA' , 'noncodingRNA_name', 
        'noncodingRNA_fam', 'feature', 'label_right', 'chr', 
        'start', 'end', 'strand', 'gene_cluster_ID', 'gene_phyloP', 'gene_phastCons'
        ]
    
    # Sequence columns to preserve for misclassification analysis
    sequence_cols = [
        'chimeric_sequence', 'mre_sequence', 'mirna_sequence', 
        'target_id', 'query_id', 'mir_fam'
    ]
    
    # Load and preprocess training data
    df_raw = pd.read_csv(train_df_path)
    df, df_with_sequences = preprocess_dataframe(df_raw, cols2drop, sequence_cols)
    df_pl = pl.from_pandas(df)
    
    # Load and preprocess test data
    final_test_raw = pd.read_csv('/home/adam/adam/data/AGO2eCLIPManakov2022testclassifiedsitesunfiltSeedConsintarnaensembleseqft.csv')
    final_test_data, final_test_with_seq = preprocess_dataframe(
        final_test_raw, cols2drop, sequence_cols
    )
    
    final_final_test_raw = pd.read_csv('/home/adam/adam/data/AGO2eCLIPManakov2022leftoutclassifiedsitesunfiltSeedConsintarnaensembleseqft.csv')
    final_final_test_data, final_final_test_with_seq = preprocess_dataframe(
        final_final_test_raw, cols2drop, sequence_cols
    )
    # Setup cross-validation
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
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
        evaluate_gluon(
            train_data=train_fold_pd, 
            val_data=val_fold_pd, 
            test_data=final_test_data, 
            additional_test_data=[final_final_test_data], 
            outf_path=results_output_path, 
            predictor=predictor,
            fold=fold
        )
        
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
    
    # Aggregate results across folds
    print(f"\n{'='*60}")
    print("AGGREGATING MISCLASSIFIED SAMPLES ACROSS FOLDS")
    print(f"{'='*60}")
    
    aggregated_results = aggregate_misclassified(misclassified_output_dir, n_folds=5)
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--modelpath', type=str, default='/home/adam/eli-adam/models/', 
                        help='Where to store the models')
    parser.add_argument('--label', type=str, default='label', 
                        help='Name of the column label')
    parser.add_argument('--metric', type=str, default='f1', 
                        help='Metric to use for evaluation')
    parser.add_argument('--time', type=int, help='Time the gluon runs in seconds')
    parser.add_argument('--misclassified_dir', type=str, 
                        default='/home/adam/adam/data/misclassified_analysis/',
                        help='Directory to save misclassified samples')
    parser.add_argument('--results_path', type=str,
                        default='/home/adam/adam/data/manakov_results_seqft_improved_weighted.txt',
                        help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    main(
        train_df_path=args.input,
        model_path=args.modelpath,
        label_col=args.label,
        eval_metric=args.metric,
        time_limit=args.time,
        misclassified_output_dir=args.misclassified_dir,
        results_output_path=args.results_path,
    )
