"""Train the three baseline classifiers and report cross-validated metrics.

Loads the newest labeled dataset, builds the leak-free X/y/groups, then
trains:
    Naive Bayes (GaussianNB)
    k-Nearest Neighbours
    Decision Tree

Each baseline uses modeling.build_pipeline (RobustScaler + SMOTE + clf)
inside a GroupKFold-by-wallet cross-validation. Metrics go to
reports/tables/baseline_metrics.csv; fitted models are saved under
data/processed/models/.

Run from the project root:  python scripts/04_train_baselines.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polymarket_insider import modeling


PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = PROCESSED_DIR / "models"
TABLE_DIR = ROOT / "reports" / "tables"


BASELINES = [
    ("naive_bayes", GaussianNB()),
    ("knn", KNeighborsClassifier(n_neighbors=15, n_jobs=1)),
    (
        "decision_tree",
        DecisionTreeClassifier(max_depth=8, random_state=modeling.RANDOM_SEED),
    ),
]


def latest_labeled_csv():
    files = sorted(PROCESSED_DIR.glob("labeled_positions_*.csv"))
    if not files:
        raise FileNotFoundError(
            "No labeled_positions_*.csv -- run scripts/02_label_dataset.py first."
        )
    return files[-1]


def main():
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

    rows = []
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, clf in BASELINES:
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

        # Fit a final model on all data and persist it.
        pipe.fit(X, y)
        modeling.save_model(pipe, MODELS_DIR / (name + ".joblib"))

    metrics_df = pd.DataFrame(rows).set_index("model")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = TABLE_DIR / "baseline_metrics.csv"
    metrics_df.to_csv(metrics_path)
    print()
    print("Saved metrics: " + str(metrics_path.relative_to(ROOT)))
    print("Saved models : " + str(MODELS_DIR.relative_to(ROOT)) + "/*.joblib")


if __name__ == "__main__":
    main()
