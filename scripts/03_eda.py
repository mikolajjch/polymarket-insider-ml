"""Exploratory data analysis on the labeled dataset.

Loads the newest data/processed/labeled_positions_*.csv and writes:
    reports/figures/*.png   plots
    reports/tables/*.csv    summary tables

Covers class balance, the suspicious_score distribution, behavior-feature
distributions, their correlations, and how often the rule flags co-occur.
The behavior features here are exactly the ones the models will be trained
on -- the outcome columns are deliberately left out.

Run from the project root:  python scripts/03_eda.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render straight to files, no display needed

import matplotlib.pyplot as plt
import pandas as pd

# Make the `polymarket_insider` package importable when run as a plain script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polymarket_insider import features


PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"

RULE_COLUMNS = [
    "rule_big_position",
    "rule_contrarian_entry",
    "rule_concentration",
    "rule_small_wallet",
]


def latest_labeled_csv():
    """Return the most recent data/processed/labeled_positions_*.csv."""
    files = sorted(PROCESSED_DIR.glob("labeled_positions_*.csv"))
    if not files:
        raise FileNotFoundError(
            "No labeled_positions_*.csv in "
            + str(PROCESSED_DIR)
            + " -- run scripts/02_label_dataset.py first"
        )
    return files[-1]


def save_fig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  figure  " + str(path.relative_to(ROOT)))


def save_table(df, name):
    path = TABLE_DIR / name
    df.to_csv(path)
    print("  table   " + str(path.relative_to(ROOT)))


def plot_class_balance(df):
    """Bar chart + table: behavior_flag vs the final suspicious label."""
    counts = pd.DataFrame(
        {
            "behavior_flag": df["behavior_flag"].value_counts().sort_index(),
            "suspicious": df["suspicious"].value_counts().sort_index(),
        }
    ).fillna(0).astype(int)
    save_table(counts, "class_balance.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("label value (0 / 1)")
    ax.set_ylabel("positions")
    ax.set_title("Class balance: behavior_flag vs final suspicious label")
    save_fig(fig, "class_balance.png")


def plot_score_distribution(df):
    """Histogram of the continuous behavior suspicious_score."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["suspicious_score"], bins=40)
    ax.axvline(0.50, color="red", linestyle="--", label="threshold 0.50")
    ax.set_xlabel("suspicious_score")
    ax.set_ylabel("positions")
    ax.set_title("Distribution of the behavior suspicious_score")
    ax.legend()
    save_fig(fig, "suspicious_score_hist.png")


def plot_feature_distributions(df, feature_cols):
    """One histogram per behavior feature (log y-axis to show the tails)."""
    ncols = 3
    nrows = (len(feature_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()
    for ax, col in zip(axes, feature_cols):
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(values, bins=40)
        ax.set_yscale("log")
        ax.set_title(col)
    for ax in axes[len(feature_cols):]:
        ax.set_visible(False)
    fig.suptitle("Behavior feature distributions (log count)")
    fig.tight_layout()
    save_fig(fig, "feature_distributions.png")


def plot_correlation(df, feature_cols):
    """Correlation heatmap + table for the behavior features."""
    corr = df[feature_cols].apply(pd.to_numeric, errors="coerce").corr()
    save_table(corr.round(3), "feature_correlation.csv")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=90)
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels(feature_cols)
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            ax.text(
                j, i, "{:.2f}".format(corr.iloc[i, j]),
                ha="center", va="center", fontsize=7,
            )
    fig.colorbar(im, ax=ax)
    ax.set_title("Behavior feature correlation")
    save_fig(fig, "feature_correlation.png")


def plot_rule_cooccurrence(df):
    """How often do pairs of rule flags fire on the same position?"""
    rules = [c for c in RULE_COLUMNS if c in df.columns]
    flags = df[rules].astype(int)
    cooc = flags.T.dot(flags)  # diagonal = per-rule count, off-diagonal = both
    save_table(cooc, "rule_cooccurrence.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cooc, cmap="Blues")
    ax.set_xticks(range(len(rules)))
    ax.set_xticklabels(rules, rotation=90)
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels(rules)
    for i in range(len(rules)):
        for j in range(len(rules)):
            ax.text(
                j, i, int(cooc.iloc[i, j]),
                ha="center", va="center", fontsize=8,
            )
    fig.colorbar(im, ax=ax)
    ax.set_title("Rule-flag co-occurrence (positions where both fire)")
    save_fig(fig, "rule_cooccurrence.png")


def main():
    path = latest_labeled_csv()
    print("Reading " + path.name + " ...")
    df = pd.read_csv(path)
    print("  {:,} rows".format(len(df)))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = features.behavior_feature_columns(df)
    print("Behavior features (" + str(len(feature_cols)) + "): " + ", ".join(feature_cols))

    # Plain summary statistics of the behavior features.
    save_table(
        df[feature_cols].apply(pd.to_numeric, errors="coerce").describe().round(3),
        "feature_summary.csv",
    )

    print()
    print("Writing EDA outputs:")
    plot_class_balance(df)
    plot_score_distribution(df)
    plot_feature_distributions(df, feature_cols)
    plot_correlation(df, feature_cols)
    plot_rule_cooccurrence(df)

    print()
    print("Done. Figures in reports/figures/, tables in reports/tables/.")


if __name__ == "__main__":
    main()
