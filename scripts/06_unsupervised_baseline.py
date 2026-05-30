"""Isolation Forest as an unsupervised sanity check.

Isolation Forest (Liu, Ting & Zhou, 2008) is an anomaly detector that
isolates points by recursively splitting on random features. Anomalies
need fewer splits to isolate, so a short average path length -> high
anomaly score. It is fast, scale-invariant, and crucially trained WITHOUT
labels.

We feed it the same behavior features the supervised models see, set
contamination to roughly the heuristic's positive rate (~6%), and compare
its "anomaly" predictions against our heuristic `suspicious` label. High
agreement means the heuristic flags positions that an independent,
label-blind detector also finds odd -- evidence the heuristic isn't
inventing structure that isn't there.

Outputs:
    reports/tables/iso_forest_vs_heuristic.csv  -- crosstab + summary stats
    reports/figures/iso_forest_score_hist.png   -- distribution of scores

Run from the project root:  python scripts/06_unsupervised_baseline.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polymarket_insider import modeling


PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"

CONTAMINATION = 0.064   # match the ~6.4% positive rate of the heuristic
N_ESTIMATORS = 200
RANDOM_SEED = 42


def latest_labeled_csv():
    files = sorted(PROCESSED_DIR.glob("labeled_positions_*.csv"))
    if not files:
        raise FileNotFoundError("No labeled_positions_*.csv")
    return files[-1]


def main():
    df = pd.read_csv(latest_labeled_csv())
    X, y_heur, _, feature_cols = modeling.build_xy_groups(df)
    print("Loaded {:,} rows, {} features".format(len(X), len(feature_cols)))
    print(
        "Heuristic positive rate: {:.1%}  ->  using contamination={}".format(
            float(y_heur.mean()), CONTAMINATION
        )
    )

    # Scale first (Isolation Forest is scale-invariant in theory but the same
    # RobustScaler keeps us consistent with the supervised pipeline).
    X_scaled = pd.DataFrame(
        RobustScaler().fit_transform(X), columns=feature_cols, index=X.index
    )

    iso = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    print("Training Isolation Forest (n_estimators={}) ...".format(N_ESTIMATORS))
    iso.fit(X_scaled)

    # predict: -1 = anomaly, 1 = normal  ->  recode to 0/1 (1 = anomaly)
    iso_anom = (iso.predict(X_scaled) == -1).astype(int)
    # higher score = more anomalous; sklearn returns negative outlier factor,
    # negate so larger = more anomalous (matches `suspicious_score` direction).
    anom_score = -iso.score_samples(X_scaled)

    crosstab = pd.crosstab(
        iso_anom, y_heur,
        rownames=["iso_anomaly"], colnames=["heuristic_suspicious"],
    )
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    crosstab_path = TABLE_DIR / "iso_forest_vs_heuristic.csv"
    crosstab.to_csv(crosstab_path)
    print("Saved " + str(crosstab_path.relative_to(ROOT)))

    n = len(X)
    tp = int(((iso_anom == 1) & (y_heur == 1)).sum())
    fp = int(((iso_anom == 1) & (y_heur == 0)).sum())
    fn = int(((iso_anom == 0) & (y_heur == 1)).sum())
    tn = int(((iso_anom == 0) & (y_heur == 0)).sum())
    agreement = (tp + tn) / n
    iso_pos = (iso_anom == 1).sum()
    heur_pos = (y_heur == 1).sum()
    overlap = tp
    overlap_pct = overlap / max(1, min(iso_pos, heur_pos))

    print()
    print("Agreement summary (Isolation Forest vs heuristic):")
    print("  total agreement     : {}/{} = {:.1%}".format(tp + tn, n, agreement))
    print("  iso flagged         : {:,}".format(iso_pos))
    print("  heuristic flagged   : {:,}".format(heur_pos))
    print(
        "  both flagged (overlap): {:,}  ({:.1%} of the smaller set)".format(
            overlap, overlap_pct
        )
    )

    # Save the anomaly-score histogram, splitting by heuristic label.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(anom_score[y_heur == 0], bins=40, alpha=0.6, label="heur=0")
    ax.hist(anom_score[y_heur == 1], bins=40, alpha=0.6, label="heur=1")
    ax.set_xlabel("Isolation Forest anomaly score (higher = more anomalous)")
    ax.set_ylabel("positions")
    ax.set_title("Isolation Forest score by heuristic label")
    ax.legend()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / "iso_forest_score_hist.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved " + str(fig_path.relative_to(ROOT)))


if __name__ == "__main__":
    main()
