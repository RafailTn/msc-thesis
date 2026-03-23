from featurewiz import FeatureWiz
import polars as pl
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

data = pl.read_csv('/home/adam/adam/data/AGO2eCLIPManakov2022trainimprovedwfeaturesaddconsplusseq.csv')
data = data.drop(['target_id', 'query_id', 'hybrid_dp', 'subseq_dp', 'mre_sequence', 'mirna_sequence',
                  'chimeric_sequence', 'gene', 'noncodingRNA', 'noncodingRNA_name', 'noncodingRNA_fam',
                  'feature', 'label_right', 'chr', 'start', 'end', 'strand', 'gene_cluster_ID',
                  'gene_phyloP', 'gene_phastCons'])
data_test = pl.read_csv('/home/adam/adam/data/AGO2eCLIPManakov2022testimprovedwfeaturesaddconsplusseq.csv')
data_test = data_test.drop(['target_id', 'query_id', 'hybrid_dp', 'subseq_dp', 'mre_sequence', 'mirna_sequence',
                             'chimeric_sequence', 'gene', 'noncodingRNA', 'noncodingRNA_name', 'noncodingRNA_fam',
                             'feature', 'label_right', 'chr', 'start', 'end', 'strand', 'gene_cluster_ID',
                             'gene_phyloP', 'gene_phastCons'])
data_leftout = pl.read_csv('/home/adam/adam/data/AGO2eCLIPManakov2022leftoutimprovedwfeaturesaddconsplusseq.csv')
data_leftout = data_leftout.drop(['target_id', 'query_id', 'hybrid_dp', 'subseq_dp', 'mre_sequence', 'mirna_sequence',
                                   'chimeric_sequence', 'gene', 'noncodingRNA', 'noncodingRNA_name', 'noncodingRNA_fam',
                                   'feature', 'label_right', 'chr', 'start', 'end', 'strand', 'gene_cluster_ID',
                                   'gene_phyloP', 'gene_phastCons'])

# Convert to pandas — featurewiz requires it
data_pd       = data.to_dummies('binding_type').to_pandas()
data_test_pd  = data_test.to_dummies('binding_type').to_pandas()
data_leftout_pd = data_leftout.to_dummies('binding_type').to_pandas()

# Align test/leftout columns to training — fills any missing dummies with 0
data_test_pd    = data_test_pd.reindex(columns=data_pd.columns, fill_value=0)
data_leftout_pd = data_leftout_pd.reindex(columns=data_pd.columns, fill_value=0)

# Now safe to cast everything
X_all        = data_pd.drop(columns=['label', 'mir_fam']).astype(np.float32)
X_test_pd    = data_test_pd.drop(columns=['label', 'mir_fam']).astype(np.float32)
X_leftout_pd = data_leftout_pd.drop(columns=['label', 'mir_fam']).astype(np.float32)

y_all    = data_pd['label'].values
groups   = data_pd['mir_fam'].values
y_test        = data_test_pd['label'].values
y_leftout     = data_leftout_pd['label'].values

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
selected_features = []
fwiz_all = None  # will hold last fold's lazy transformer

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_all, y_all, groups)):
    X_train_fold = X_all.iloc[train_idx]
    y_train_fold = y_all[train_idx]
    X_val_fold   = X_all.iloc[val_idx]
    y_val_fold   = y_all[val_idx]

    fwiz = FeatureWiz(feature_engg='', nrows=None, transform_target=True, scalers="std",
                      category_encoders="auto", add_missing=False, verbose=0, imbalanced=False,
                      ae_options={})

    # fit_transform expects a DataFrame for X and a Series/array for y
    X_train_selected, y_train_selected = fwiz.fit_transform(
        X_train_fold, pd.Series(y_train_fold, name='label'))
    X_val_selected = fwiz.transform(X_val_fold)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_selected, y_train_selected)
    y_proba = model.predict_proba(X_val_selected)[:, 1]
    ap = average_precision_score(y_val_fold, y_proba)
    print(f'Fold {fold + 1}: Validation APS = {ap:.4f}')

    selected_features.append(fwiz.features)
    fwiz_all = fwiz  # keep last fold's transformer

# Most stable features: intersection across all folds
common_features = list(set(selected_features[0]).intersection(*selected_features[1:]))
print(f'\nCommon stable features: {len(common_features)}\n', common_features)

# Transform full datasets with last fold's lazy transformer
X_train_transformed  = fwiz_all.transform(X_all)
X_test_transformed   = fwiz_all.transform(X_test_pd)
X_leftout_transformed = fwiz_all.transform(X_leftout_pd)

# Per-round evaluation
aps_test, aps_leftout = [], []
for i in range(5):
    feats = selected_features[i]
    model_round = LogisticRegression(max_iter=1000)
    model_round.fit(X_train_transformed[feats], y_all)  # train on full train set
    y_proba          = model_round.predict_proba(X_test_transformed[feats])[:, 1]
    y_leftout_proba  = model_round.predict_proba(X_leftout_transformed[feats])[:, 1]
    aps_test.append(average_precision_score(y_test, y_proba))
    aps_leftout.append(average_precision_score(y_leftout, y_leftout_proba))

# Final model on common features
model_final = LogisticRegression(max_iter=1000)
model_final.fit(X_train_transformed[common_features], y_all)
ap_test_final    = average_precision_score(y_test,    model_final.predict_proba(X_test_transformed[common_features])[:, 1])
ap_leftout_final = average_precision_score(y_leftout, model_final.predict_proba(X_leftout_transformed[common_features])[:, 1])

print(f'\nAverage APS test (5 rounds):     {np.mean(aps_test):.4f}')
print(f'Final APS test (common features): {ap_test_final:.4f}')
print(f'\nAverage APS leftout (5 rounds):     {np.mean(aps_leftout):.4f}')
print(f'Final APS leftout (common features): {ap_leftout_final:.4f}')
