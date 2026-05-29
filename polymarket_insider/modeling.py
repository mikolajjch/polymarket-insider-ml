"""Modeling pipeline helpers.

Centralises the preprocessing, cross-validation and persistence plumbing
that every training script in Phase 4 uses, so each script can stay short.

Design notes:
  * Features = behavior features only (features.behavior_feature_columns).
    Outcome columns are NEVER fed to the model -- the label already
    encodes the outcome, so feeding it back as a feature would be leakage.
  * Cross-validation = GroupKFold by wallet (proxyWallet). The same wallet
    must not appear in both train and test, otherwise we measure memorisation
    of the wallet, not real generalisation.
  * Class balance = SMOTE on the training fold only. SMOTE oversamples the
    minority class so models actually learn it (positives are ~6% of rows).
    Test folds keep the real-world distribution.
"""

import joblib
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.preprocessing import RobustScaler

from polymarket_insider import features


LABEL_COLUMN = "suspicious"
GROUP_COLUMN = "proxyWallet"
RANDOM_SEED = 42
CV_SPLITS = 5

# ROC-AUC is the primary metric; PR-AUC (average_precision) matters more for
# the imbalanced positive class; F1/precision/recall use the default 0.5
# decision threshold and are reported for completeness.
METRICS = ["roc_auc", "average_precision", "f1", "precision", "recall"]


def build_xy_groups(df):
    """Return (X, y, groups, feature_cols) ready for sklearn.

    X uses ONLY the behavior features (no outcome leakage).
    """
    feature_cols = features.behavior_feature_columns(df)
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df[LABEL_COLUMN].astype(int)
    groups = df[GROUP_COLUMN]
    return X, y, groups, feature_cols


def build_pipeline(classifier, use_smote=True):
    """Wrap a classifier in Pipeline with RobustScaler [+ SMOTE] + clf.

    Using imblearn's Pipeline so SMOTE is applied only at fit time -- test
    folds keep the original class distribution.
    """
    steps = [("scaler", RobustScaler())]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_SEED)))
    steps.append(("clf", classifier))
    return ImbPipeline(steps)


def cross_validate_grouped(pipe, X, y, groups, n_splits=CV_SPLITS):
    """GroupKFold cross-validation -- same wallet never in train and test."""
    cv = GroupKFold(n_splits=n_splits)
    return cross_validate(
        pipe, X, y,
        groups=groups,
        cv=cv,
        scoring=METRICS,
        return_train_score=False,
        n_jobs=1,
    )


def summarize_cv(cv_results):
    """Reduce a cross_validate dict to mean+std per metric."""
    out = {}
    for key, vals in cv_results.items():
        if key.startswith("test_"):
            metric = key[len("test_"):]
            out[metric + "_mean"] = float(vals.mean())
            out[metric + "_std"] = float(vals.std())
    return out


def save_model(model, path):
    """Persist a fitted pipeline to disk with joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    """Load a fitted pipeline previously saved with save_model."""
    return joblib.load(path)
