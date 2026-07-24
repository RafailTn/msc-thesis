"""Feature-set resolution shared by the training entry points.

Both gluon_training_kfold.py and gluon_train_total.py need to reduce an extraction CSV
to a named feature set, and they must do it identically - a model compared under one
definition and retrained under another is not the same experiment. So the logic lives
here once rather than being copied into each, and the set *definitions* live further
upstream still, in src/feature_extraction.py, beside the code that produces the columns.
"""

import os
import sys
from typing import List

import pandas as pd

# The set definitions live with the extractor that computes the columns, so the names
# cannot drift from what is actually produced.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from feature_extraction import FEATURE_SETS, feature_set  # noqa: E402,F401


def select_feature_columns(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Reduce `df` to exactly `features` plus the columns training needs to run.

    One wrinkle is worth knowing about: an extraction CSV carries `binding_type` as a raw
    categorical string whenever the requested list did not name the one-hots, whereas the
    featurewiz baseline names one-hot columns like `binding_type_6mer.mismatch.3prime`.
    So the indicators are rebuilt here, by the same rule feature_extraction uses when it
    writes them, rather than being demanded from the file - otherwise
    `--feature-set baseline` could fail on a perfectly good CSV.

    `mir_fam` is kept regardless: it is the CV/tuning grouping key and the basis of the
    sample weights, and is dropped by the caller once the split is made.
    """
    out = pd.DataFrame(index=df.index)

    missing = []
    for feature in features:
        if feature in df.columns:
            out[feature] = df[feature]
        elif feature.startswith('binding_type_') and 'binding_type' in df.columns:
            value = feature[len('binding_type_'):]
            out[feature] = (df['binding_type'] == value).astype(int)
        else:
            missing.append(feature)

    if missing:
        raise SystemExit(
            f"ERROR: {len(missing)} requested feature(s) are not in the CSV and cannot "
            f"be derived: {missing[:10]}{' ...' if len(missing) > 10 else ''}\n"
            f"Regenerate the CSV with `feature_extraction.py` (and, for the shuffle "
            f"z-scores, --mirna-background); `--list-features` shows what is available."
        )

    for keep in ('mir_fam', 'label'):
        if keep in df.columns:
            out[keep] = df[keep]
    return out


# Identifier / passthrough columns that must never reach the model. Shared so the two
# entry points cannot disagree about what counts as a feature - gluon_train_total was
# missing `energy_source`, which would have been fed to AutoGluon as a categorical.
# Only consulted for `--feature-set all`; the named sets select columns positively and
# so drop everything else by construction.
COLS2DROP = [
    'target_id', 'query_id', 'binding_type', 'noncodingRNA_fam',
    'contrafold_struct', 'hybrid_dp', 'subseq_dp',
    'mre_sequence', 'mirna_sequence', 'chimeric_sequence', 'energy_source',
    'gene', 'noncodingRNA', 'noncodingRNA_name', 'feature', 'label_right',
    'chr', 'start', 'end', 'strand', 'gene_cluster_ID',
    'gene_phyloP', 'gene_phastCons',
]

SEQUENCE_COLS = [
    'chimeric_sequence', 'mre_sequence', 'mirna_sequence',
    'target_id', 'query_id', 'mir_fam',
]
