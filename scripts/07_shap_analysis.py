"""SHAP analysis of the best model (XGBoost).

SHAP (SHapley Additive exPlanations) breaks each prediction into the
contribution of every feature, based on cooperative-game theory (Shapley
values from Lloyd Shapley, 1953; adapted to ML by Lundberg & Lee, 2017).

For a tree ensemble like XGBoost, SHAP can be computed exactly and fast
(TreeExplainer). For each row we get a vector of "this feature pushed the
prediction up/down by X". Averaged across rows -> global feature importance
that is more honest than the built-in `feature_importances_`, which only
counts splits and ignores their effect size.

This script loads the trained XGBoost model, computes SHAP values on a
sample of the data, and writes:
    reports/tables/shap_importance.csv  -- mean |SHAP| per feature, ranked
    reports/figures/shap_bar.png        -- bar chart of mean |SHAP|
    reports/figures/shap_beeswarm.png   -- per-row contribution beeswarm

Run from the project root:  python scripts/07_shap_analysis.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polymarket_insider import modeling


PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = PROCESSED_DIR / "models"
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"

SAMPLE_N = 500          # rows used for SHAP -- 500 is plenty for stable means
RANDOM_SEED = 42
MODEL_NAME = "xgboost"


def latest_labeled_csv():
    files = sorted(PROCESSED_DIR.glob("labeled_positions_*.csv"))
    if not files:
        raise FileNotFoundError("No labeled_positions_*.csv")
    return files[-1]


def main():
    df = pd.read_csv(latest_labeled_csv())
    X, y, _, feature_cols = modeling.build_xy_groups(df)
    print("Loaded {:,} rows, {} features".format(len(X), len(feature_cols)))

    pipe = modeling.load_model(MODELS_DIR / (MODEL_NAME + ".joblib"))
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]
    print("Loaded model: " + MODEL_NAME)

    sample = X.sample(n=min(SAMPLE_N, len(X)), random_state=RANDOM_SEED)
    sample_scaled = pd.DataFrame(
        scaler.transform(sample), columns=feature_cols, index=sample.index
    )

    # Use function-based Explainer (works around a shap+xgboost 3.x bug where
    # TreeExplainer crashes on the new base_score string serialisation).
    # A small background sample defines the masker for permutation-style SHAP.
    background = sample_scaled.sample(n=min(100, len(sample_scaled)), random_state=RANDOM_SEED)
    print("Computing SHAP values on {} rows (background={}) ...".format(
        len(sample_scaled), len(background)
    ))
    explainer = shap.Explainer(clf.predict, background)
    shap_values = explainer(sample_scaled)
    print("  shape: {}".format(shap_values.values.shape))

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    imp = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    imp_path = TABLE_DIR / "shap_importance.csv"
    imp.to_csv(imp_path, index=False)
    print("Saved " + str(imp_path.relative_to(ROOT)))

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp["feature"][::-1], imp["mean_abs_shap"][::-1])
    ax.set_xlabel("mean |SHAP value|  (avg magnitude of contribution)")
    ax.set_title("XGBoost behavior-feature importance (SHAP)")
    fig.tight_layout()
    bar_path = FIG_DIR / "shap_bar.png"
    fig.savefig(bar_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved " + str(bar_path.relative_to(ROOT)))

    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values, sample_scaled,
        feature_names=feature_cols, show=False,
    )
    beeswarm_path = FIG_DIR / "shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved " + str(beeswarm_path.relative_to(ROOT)))

    print()
    print("Top features by mean |SHAP|:")
    print(imp.to_string(index=False))


if __name__ == "__main__":
    main()
