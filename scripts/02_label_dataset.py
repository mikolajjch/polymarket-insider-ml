"""Apply engineered features and the heuristic label to the latest scrape.

Reads the newest data/raw/positions_*.csv, adds the engineered features and
the heuristic 'suspicious' label, and writes two files into data/processed/:

    labeled_positions_<timestamp>.csv   the full labeled dataset
    gold_set_template.csv               ~100 rows to hand-check

The gold-set template is a balanced sample (roughly half flagged suspicious
by the heuristic, half not). Fill in its `gold_label` column by hand to get
a small trusted set for measuring how good the heuristic and models really
are -- the heuristic is only a weak label, so we need something to check it
against.

Run from the project root:  python scripts/02_label_dataset.py
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Make the `polymarket_insider` package importable when run as a plain script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polymarket_insider import features, labeling


RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
GOLD_SET_SIZE = 100
RANDOM_SEED = 42


def latest_raw_csv():
    """Return the most recent data/raw/positions_*.csv."""
    files = sorted(RAW_DIR.glob("positions_*.csv"))
    if not files:
        raise FileNotFoundError("No positions_*.csv found in " + str(RAW_DIR))
    return files[-1]


def build_gold_set(df, size=GOLD_SET_SIZE, seed=RANDOM_SEED):
    """Return a balanced sample of rows to label by hand.

    Half the rows are taken from the heuristic-suspicious group and half
    from the rest, so the sample has enough positives to be worth checking.
    A human labeling these rows may freely look at the outcome columns --
    only the *machine* label is required to ignore the outcome.
    """
    suspicious = df[df["suspicious"] == 1]
    rest = df[df["suspicious"] == 0]

    n_pos = min(len(suspicious), size // 2)
    n_neg = min(len(rest), size - n_pos)

    picked = pd.concat(
        [
            suspicious.sample(n=n_pos, random_state=seed),
            rest.sample(n=n_neg, random_state=seed),
        ]
    )
    picked = picked.sample(frac=1.0, random_state=seed)  # shuffle the order

    # Columns a human needs to judge a row.
    keep = [
        "event_title", "market_question", "side", "proxyWallet",
        "totalBought", "avgPrice", "account_total_traded_volume",
        "position_concentration",
        "score_big_position", "score_contrarian_entry",
        "score_concentration", "score_small_wallet",
        "suspicious_score", "behavior_flag",
        "outcome_won", "outcome_return_ratio", "suspicious",
    ]
    gold = picked[[c for c in keep if c in picked.columns]].copy()
    gold["gold_label"] = ""   # fill in by hand: 1 = insider-like, 0 = normal
    gold["gold_notes"] = ""
    return gold


def print_summary(summary):
    """Pretty-print the dict returned by labeling.label_summary."""
    print()
    print("Label summary")
    print("  positions             : {:,}".format(summary["n"]))
    print("  behavior-flagged      : {:,}".format(summary["n_behavior_flag"]))
    print(
        "  suspicious label (=1) : {:,}  ({:.1%} of all positions)".format(
            summary["n_suspicious"], summary["frac_suspicious"]
        )
    )
    print("  mean suspicious_score : {:.3f}".format(summary["score_mean"]))
    q = summary["score_quantiles"]
    print(
        "  score p50/p90/p99     : {:.3f} / {:.3f} / {:.3f}".format(
            q[0.5], q[0.9], q[0.99]
        )
    )
    if summary.get("behavior_flag_win_rate") is not None:
        print(
            "  of behavior-flagged, {:.1%} won -> these become positive labels".format(
                summary["behavior_flag_win_rate"]
            )
        )


def main():
    raw_path = latest_raw_csv()
    print("Reading " + raw_path.name + " ...")
    df = pd.read_csv(raw_path)
    print("  {:,} rows, {:,} wallets".format(len(df), df["proxyWallet"].nunique()))

    df = features.add_engineered_features(df)
    df = labeling.add_suspicious_label(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    labeled_path = PROCESSED_DIR / ("labeled_positions_" + timestamp + ".csv")
    df.to_csv(labeled_path, index=False)
    print("Wrote " + str(labeled_path.relative_to(ROOT)))

    gold = build_gold_set(df)
    gold_path = PROCESSED_DIR / "gold_set_template.csv"
    gold.to_csv(gold_path, index=False)
    print(
        "Wrote {}  ({} rows to hand-label)".format(
            gold_path.relative_to(ROOT), len(gold)
        )
    )

    print_summary(labeling.label_summary(df))

    # Top 10 most suspicious positions, for a quick eyeball check.
    print()
    print("Top 10 by suspicious_score:")
    cols = [
        "suspicious_score", "totalBought", "avgPrice",
        "account_total_traded_volume", "market_question",
    ]
    cols = [c for c in cols if c in df.columns]
    top = df.sort_values("suspicious_score", ascending=False).head(10)
    with pd.option_context("display.max_colwidth", 38, "display.width", 120):
        print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
