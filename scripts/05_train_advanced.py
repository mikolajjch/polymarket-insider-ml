"""Train the advanced classifiers and compare them against the baselines.

Loads the newest labeled dataset, builds the same leak-free X/y/groups as
in 04, then trains four more models using the identical
modeling.build_pipeline (RobustScaler + SMOTE + clf):

    Random Forest           ensemble of decorrelated trees
    XGBoost                 gradient-boosted trees (usual SOTA on tabular)
    MLP                     small fully-connected neural net
    Logistic Regression     linear, interpretable, fastest baseline

Each model is cross-validated with GroupKFold-by-wallet, persisted to
data/processed/models/ and its metrics appended to
reports/tables/advanced_metrics.csv. If reports/tables/baseline_metrics.csv
also exists, the script writes a combined sorted table to
reports/tables/all_models_metrics.csv so we have a single ranking for the
report.

Hyperparameters here are intentionally modest so the script finishes fast.
Feel free to crank up n_estimators / hidden layer sizes once you're sure
the pipeline works end-to-end.

Run from the project root:  python scripts/05_train_advanced.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polymarket_insider import modeling


PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = PROCESSED_DIR / "models"
TABLE_DIR = ROOT / "reports" / "tables"


ADVANCED = [
    (
        "random_forest",
        RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            n_jobs=-1,
            random_state=modeling.RANDOM_SEED,
        ),
    ),
    (
        "xgboost",
        XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="aucpr",
            n_jobs=-1,
            tree_method="hist",
            random_state=modeling.RANDOM_SEED,
        ),
    ),
    (
        "mlp",
        MLPClassifier(
            hidden_layer_sizes=(16, 8),
            max_iter=300,
            early_stopping=True,
            random_state=modeling.RANDOM_SEED,
        ),
    ),
    (
        "logreg",
        LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=modeling.RANDOM_SEED,
        ),
    ),
]


def latest_labeled_csv():
    files = sorted(PROCESSED_DIR.glob("labeled_positions_*.csv"))
    if not files:
        raise FileNotFoundError(
            "No labeled_positions_*.csv -- run scripts/02_label_dataset.py first."
        )
    return files[-1]


def write_combined_table():
    """If baseline + advanced metrics both exist, save a combined ranking."""
    base_path = TABLE_DIR / "baseline_metrics.csv"
    adv_path = TABLE_DIR / "advanced_metrics.csv"
    if not (base_path.exists() and adv_path.exists()):
        return None
    base = pd.read_csv(base_path)
    adv = pd.read_csv(adv_path)
    combined = pd.concat([base, adv], ignore_index=True)
    combined = combined.sort_values("roc_auc_mean", ascending=False)
    combined.set_index("model").to_csv(TABLE_DIR / "all_models_metrics.csv")
    return TABLE_DIR / "all_models_metrics.csv"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                        help="train only this model name (e.g. --only logreg)")
    parser.add_argument("--append", action="store_true",
                        help="append to advanced_metrics.csv instead of overwriting")
    args = parser.parse_args()

    path = latest_labeled_csv()
    print("Reading " + path.name + " ...")
    df = pd.read_csv(path)
    X, y, groups, feature_cols = modeling.build_xy_groups(df)
    print(
        "  rows: {:,}  positives: {:,}  unique wallets: {:,}  features: {}".format(
            len(X), int(y.sum()), groups.nunique(), len(feature_cols)
        )
    )
    print(
        "  GroupKFold by wallet ({} folds), SMOTE on the minority class".format(
            modeling.CV_SPLITS
        )
    )
    print()

    chosen = [(n, c) for n, c in ADVANCED if args.only is None or n == args.only]
    if args.only is not None and not chosen:
        raise ValueError("Unknown model name: " + args.only)

    rows = []
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, clf in chosen:
        print("Training " + name + " ...")
        pipe = modeling.build_pipeline(clf, use_smote=True)
        cv = modeling.cross_validate_grouped(pipe, X, y, groups)
        summary = modeling.summarize_cv(cv)
        summary["model"] = name
        rows.append(summary)
        print(
            "  ROC-AUC {:.3f} +- {:.3f}   PR-AUC {:.3f}   F1 {:.3f}   "
            "precision {:.3f}   recall {:.3f}".format(
                summary["roc_auc_mean"], summary["roc_auc_std"],
                summary["average_precision_mean"],
                summary["f1_mean"],
                summary["precision_mean"], summary["recall_mean"],
            )
        )

        pipe.fit(X, y)
        modeling.save_model(pipe, MODELS_DIR / (name + ".joblib"))

    metrics_df = pd.DataFrame(rows).set_index("model")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    adv_path = TABLE_DIR / "advanced_metrics.csv"
    if args.append and adv_path.exists():
        existing = pd.read_csv(adv_path).set_index("model")
        # Drop rows we just re-trained so we don't double-count.
        existing = existing.drop(index=[r for r in metrics_df.index if r in existing.index], errors="ignore")
        metrics_df = pd.concat([existing, metrics_df])
    metrics_df.to_csv(adv_path)
    print()
    print("Saved metrics: " + str(adv_path.relative_to(ROOT)))
    combined_path = write_combined_table()
    if combined_path is not None:
        print("Saved ranking: " + str(combined_path.relative_to(ROOT)))
    print("Saved models : " + str(MODELS_DIR.relative_to(ROOT)) + "/*.joblib")


if __name__ == "__main__":
    main()
