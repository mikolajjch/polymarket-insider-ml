"""Evaluate the heuristic and every trained model against the gold set.

The gold set is the small, hand-labeled (or AI-assisted) sample at
data/processed/gold_set_labeled.csv. Each row has `gold_label` in [0, 1].
We binarise at 0.5 and treat the result as ground truth.

For each predictor (the heuristic itself + all 7 .joblib models) we:
  1. find the matching rows in the full labeled dataset by (proxyWallet,
     market_question, side) -- same composite key used to build the gold
     set -- so we can recover the behavior-feature columns.
  2. produce predictions.
  3. compute precision / recall / F1 / accuracy / agreement vs gold_label.

Output:
    reports/tables/gold_evaluation.csv  -- one row per predictor

This is the real "did the model match human judgement?" metric for the
report. Every other model metric is computed against the heuristic itself
and therefore partly tautological.

Run from the project root:  python scripts/09_evaluate_vs_gold.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polymarket_insider import features, modeling


PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = PROCESSED_DIR / "models"
TABLE_DIR = ROOT / "reports" / "tables"

GOLD_THRESHOLD = 0.5
JOIN_KEYS = ["proxyWallet", "market_question", "side"]


def latest_labeled_csv():
    files = sorted(PROCESSED_DIR.glob("labeled_positions_*.csv"))
    if not files:
        raise FileNotFoundError("No labeled_positions_*.csv")
    return files[-1]


def metrics_row(name, y_true, y_pred):
    """Compute the metrics dict for one predictor."""
    return {
        "predictor":     name,
        "accuracy":      accuracy_score(y_true, y_pred),
        "precision":     precision_score(y_true, y_pred, zero_division=0),
        "recall":        recall_score(y_true, y_pred, zero_division=0),
        "f1":            f1_score(y_true, y_pred, zero_division=0),
        "agreement":     (y_true == y_pred).mean(),
        "n_pred_pos":    int(y_pred.sum()),
        "n_true_pos":    int(y_true.sum()),
        "n":             int(len(y_true)),
    }


def main():
    gold_path = PROCESSED_DIR / "gold_set_labeled.csv"
    if not gold_path.exists():
        raise FileNotFoundError(
            "No gold_set_labeled.csv -- hand-label gold_set_template.csv first."
        )
    gold = pd.read_csv(gold_path)
    print("Gold set: {} rows".format(len(gold)))
    gold["gold_binary"] = (gold["gold_label"] >= GOLD_THRESHOLD).astype(int)
    print("  positives (gold_label >= {}): {}".format(
        GOLD_THRESHOLD, int(gold["gold_binary"].sum())
    ))

    full = pd.read_csv(latest_labeled_csv())
    full = features.add_engineered_features(full)  # ensure log_totalBought etc. exist
    print("Full labeled dataset: {:,} rows".format(len(full)))

    # Rejoin gold with full to recover the behavior-feature columns.
    merged = gold.merge(
        full, on=JOIN_KEYS, how="left", suffixes=("_gold", "")
    )
    # Some gold rows may match multiple full rows (same wallet bought twice
    # the same side of the same market). De-duplicate by keeping the first.
    merged = merged.drop_duplicates(subset=JOIN_KEYS, keep="first").reset_index(drop=True)
    print("After rejoin: {} rows".format(len(merged)))

    feature_cols = features.behavior_feature_columns(merged)
    X_gold = merged[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_gold = merged["gold_binary"].astype(int)

    rows = []

    # Heuristic baseline: the `suspicious` column from labeling.
    heur_pred = merged["suspicious"].astype(int)
    rows.append(metrics_row("heuristic", y_gold, heur_pred))

    # Each trained model.
    model_files = sorted(MODELS_DIR.glob("*.joblib"))
    for path in model_files:
        name = path.stem
        pipe = modeling.load_model(path)
        pred = pipe.predict(X_gold)
        rows.append(metrics_row(name, y_gold, pd.Series(pred, index=X_gold.index)))
        print("  evaluated {}".format(name))

    out = pd.DataFrame(rows).set_index("predictor").sort_values("f1", ascending=False)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLE_DIR / "gold_evaluation.csv"
    out.to_csv(out_path)
    print()
    print("Saved " + str(out_path.relative_to(ROOT)))
    print()
    print("Evaluation against the {}-row gold set:".format(len(merged)))
    with pd.option_context("display.width", 140, "display.max_colwidth", 30):
        print(out.round(3).to_string())


if __name__ == "__main__":
    main()
