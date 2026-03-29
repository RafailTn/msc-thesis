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
from sklearn.model_selection import GroupShuffleSplit
from typing import List, Optional, Dict, Tuple

def evaluate_df(df: pd.DataFrame, predictor: TabularPredictor) -> Tuple[dict, float]:
    """Evaluate a dataframe and return metrics including average precision score."""
    eval_metrics = predictor.evaluate(df)
    y_pred_proba = predictor.predict_proba(df)[1]
    y_true = df['label']
    ap_score = average_precision_score(y_true, y_pred_proba)
    return eval_metrics, ap_score

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
    family_counts = df['noncodingRNA_fam'].value_counts().clip(lower=100)
    total_samples = len(df) 
    # Weight = Total / (n_families * count_of_this_family)
    # This scales weights so they sum up to roughly len(df)
    n_families = len(family_counts)
    df['weights'] = df['noncodingRNA_fam'].map(lambda x: total_samples / (n_families * family_counts[x])) 
    df['label'] = df['label'].astype(int)
    return df, df_with_sequences

def get_misclassified(
    df_with_sequences: pd.DataFrame, 
    df_for_prediction: pd.DataFrame, 
    predictor: TabularPredictor, 
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
        dataset_name: Name for the output file (e.g., 'train', 'val', 'test')
        output_dir: Directory to save results
        save: Whether to save to CSV
        
    Returns:
        DataFrame of misclassified samples with predictions
    """
    y_true = df_for_prediction['label'].values
    y_pred = predictor.predict(df_for_prediction).values
    y_pred_proba = predictor.predict_proba(df_for_prediction)[1].values
    
    misclassified_mask = y_true != y_pred
    
    misclassified_df = df_with_sequences.loc[misclassified_mask].copy()
    misclassified_df['predicted_label'] = y_pred[misclassified_mask]
    misclassified_df['predicted_proba'] = y_pred_proba[misclassified_mask]
    misclassified_df['dataset'] = dataset_name
    
    misclassified_df['error_type'] = np.where(
        misclassified_df['label'] == 1, 
        'false_negative',
        'false_positive'
    )
    
    if save:
        os.makedirs(output_dir, exist_ok=True)
        outpath = os.path.join(output_dir, f'misclassified_{dataset_name}.csv')
        misclassified_df.to_csv(outpath, index=False)
        print(f"Saved {len(misclassified_df)} misclassified samples to {outpath}")
    
    n_fp = (misclassified_df['error_type'] == 'false_positive').sum()
    n_fn = (misclassified_df['error_type'] == 'false_negative').sum()
    print(f"  {dataset_name}: {len(misclassified_df)} errors ({n_fp} FP, {n_fn} FN)")
    
    return misclassified_df

def evaluate_gluon(
    train_data: pd.DataFrame, 
    tune_data: pd.DataFrame,
    test_data: pd.DataFrame, 
    additional_test_data: Optional[List[pd.DataFrame]], 
    outf_path: str, 
    predictor: TabularPredictor,
    decision_threshold: float | None = None 
) -> None:
    """Evaluate predictor on all datasets and write results to file."""
    if decision_threshold:
        predictor.set_decision_threshold(decision_threshold)
    else:
        calibrated_threshold = predictor.calibrate_decision_threshold(metric="f1")
        predictor.set_decision_threshold(calibrated_threshold)

    train_eval, ap_score_train = evaluate_df(train_data, predictor)
    tune_eval, ap_score_tune   = evaluate_df(tune_data, predictor)
    test_eval, ap_score_test   = evaluate_df(test_data, predictor)
    
    with open(outf_path, 'a') as outfile:
        outfile.write(f"Train_results: {train_eval}, APS: {ap_score_train}\n")
        outfile.write(f"Tune_results: {tune_eval}, APS: {ap_score_tune}\n")
        outfile.write(f"Small_test_results: {test_eval}, APS: {ap_score_test}\n")
        
        if additional_test_data:
            for i, add_test in enumerate(additional_test_data):
                add_test_eval, ap_score_add = evaluate_df(add_test, predictor)
                outfile.write(f"Final_test_{i}_results: {add_test_eval}, APS: {ap_score_add}\n")

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
        'target_id', 'query_id', 
        'contrafold_struct', 'hybrid_dp', 'subseq_dp', 
        'mre_sequence', 'mirna_sequence', 'chimeric_sequence', 
        'gene' , 'noncodingRNA' , 'noncodingRNA_name', 
        'feature', 'label_right', 'chr', 
        'start', 'end', 'strand', 'gene_cluster_ID', 'gene_phyloP', 'gene_phastCons'
    ]
    
    sequence_cols = [
        'chimeric_sequence', 'mre_sequence', 'mirna_sequence', 
        'target_id', 'query_id', 'noncodingRNA_fam'
    ]
    
    # Load and preprocess training data
    df_raw = pd.read_csv(train_df_path)
    df, df_with_sequences = preprocess_dataframe(df_raw, cols2drop, sequence_cols)

    # GroupShuffleSplit: 10% tuning set, groups by noncodingRNA_fam
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    groups = df['noncodingRNA_fam'].to_numpy()
    y = df['label'].to_numpy()

    train_idx, tune_idx = next(gss.split(df, y, groups=groups))

    # Drop noncodingRNA_fam before passing to AutoGluon
    train_fold = df.iloc[train_idx].drop(columns=['noncodingRNA_fam']).reset_index(drop=True)
    tune_fold  = df.iloc[tune_idx].drop(columns=['noncodingRNA_fam']).reset_index(drop=True)

    train_with_seq = df_with_sequences.iloc[train_idx].reset_index(drop=True)
    tune_with_seq  = df_with_sequences.iloc[tune_idx].reset_index(drop=True)

    print(f"Train size: {len(train_fold)} | Tune size: {len(tune_fold)}")
    print(f"Tune group overlap check — unique noncodingRNA_fam in tune not in train: "
          f"{set(df.iloc[tune_idx]['noncodingRNA_fam']) - set(df.iloc[train_idx]['noncodingRNA_fam'])}")

    # Load and preprocess test data
    final_test_raw = pd.read_csv('/home/adam/adam/Final_fs_featurewiz_test_withids.csv')
    final_test_data, final_test_with_seq = preprocess_dataframe(
        final_test_raw, cols2drop, sequence_cols
    )
    final_test_data = final_test_data.drop(columns=['noncodingRNA_fam'], errors='ignore')

    final_final_test_raw = pd.read_csv('/home/adam/adam/Final_fs_featurewiz_leftout_withids.csv')
    final_final_test_data, final_final_test_with_seq = preprocess_dataframe(
        final_final_test_raw, cols2drop, sequence_cols
    )
    final_final_test_data = final_final_test_data.drop(columns=['noncodingRNA_fam'], errors='ignore')

    os.makedirs(misclassified_output_dir, exist_ok=True)

    # Train final model on full training data with tuning split
    predictor = TabularPredictor(
        label=label_col,
        eval_metric=eval_metric,
        sample_weight='weights',
        weight_evaluation=True,
        path=model_path,
    )

    predictor.fit(
        train_data=train_fold,
        tuning_data=tune_fold,
        presets='best_quality',
        num_bag_folds=8,
        num_stack_levels=1,
        use_bag_holdout=True,
        time_limit=time_limit,
    )

    # Evaluate on all datasets
    evaluate_gluon(
        train_data=train_fold,
        tune_data=tune_fold,
        test_data=final_test_data,
        additional_test_data=[final_final_test_data],
        outf_path=results_output_path,
        predictor=predictor,
    )

    # Misclassification analysis
    print("\nMisclassification Analysis:")

    get_misclassified(train_with_seq, train_fold, predictor, 'train', misclassified_output_dir)
    get_misclassified(tune_with_seq,  tune_fold,  predictor, 'tune',  misclassified_output_dir)
    get_misclassified(
        final_test_with_seq.reset_index(drop=True),
        final_test_data,
        predictor, 'test', misclassified_output_dir
    )
    get_misclassified(
        final_final_test_with_seq.reset_index(drop=True),
        final_final_test_data,
        predictor, 'final_test', misclassified_output_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--modelpath', type=str, default='/home/adam/eli-adam/models/', 
                        help='Where to store the model')
    parser.add_argument('--label', type=str, default='label', 
                        help='Name of the column label')
    parser.add_argument('--metric', type=str, default='f1', 
                        help='Metric to use for evaluation')
    parser.add_argument('--time', type=int, help='Time the gluon runs in seconds')
    parser.add_argument('--misclassified_dir', type=str, 
                        default='/home/adam/adam/data/misclassified_analysis/',
                        help='Directory to save misclassified samples')
    parser.add_argument('--results_path', type=str,
                        default='/home/adam/adam/data/manakov_results_seqft_final.txt',
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

