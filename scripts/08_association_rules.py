"""Mine association rules over the binary flag columns + the label.

Uses mlxtend's apriori + association_rules. The "items" are:
    rule_big_position, rule_contrarian_entry, rule_concentration,
    rule_small_wallet      -- the four behavior flags
    outcome_won            -- whether the position realized a profit
    suspicious             -- the final label (flag AND won)

Apriori finds frequent itemsets (combinations that appear together often
enough). `association_rules` then turns each frequent itemset into rules of
the form  X -> Y  with three metrics:
    support     P(X and Y)  -- how often the rule applies overall
    confidence  P(Y | X)    -- of rows matching X, what fraction also match Y
    lift        confidence / P(Y)  -- how much more often Y co-occurs with X
                                       than chance; lift=1 means independent,
                                       lift>1 means X predicts Y

We export rules whose right-hand side is `suspicious=1`, ranked by lift.
That tells us in plain language WHICH flag combinations are the strongest
predictors of the label, complementing the SHAP analysis (07).

Run from the project root:  python scripts/08_association_rules.py
"""

import sys
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "reports" / "tables"

ITEM_COLUMNS = [
    "rule_big_position",
    "rule_contrarian_entry",
    "rule_concentration",
    "rule_small_wallet",
    "outcome_won",
    "suspicious",
]

MIN_SUPPORT = 0.02     # itemset must appear in at least 2% of rows
MIN_CONFIDENCE = 0.30  # rule keeps if P(Y|X) >= 30%
TOP_N = 25             # how many rules to print to stdout


def latest_labeled_csv():
    files = sorted(PROCESSED_DIR.glob("labeled_positions_*.csv"))
    if not files:
        raise FileNotFoundError(
            "No labeled_positions_*.csv -- run scripts/02_label_dataset.py first."
        )
    return files[-1]


def to_transactions(df):
    """Coerce all item columns to bool (apriori expects a boolean DataFrame)."""
    cols = [c for c in ITEM_COLUMNS if c in df.columns]
    out = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).astype(bool)
    return out


def fmt_set(s):
    """frozenset -> human-readable 'a + b'."""
    return " + ".join(sorted(s))


def main():
    path = latest_labeled_csv()
    print("Reading " + path.name + " ...")
    df = pd.read_csv(path)
    transactions = to_transactions(df)
    print(
        "  {:,} transactions over {} items: {}".format(
            len(transactions), len(transactions.columns), list(transactions.columns)
        )
    )
    print()

    print(
        "Mining frequent itemsets (min_support={}) ...".format(MIN_SUPPORT)
    )
    itemsets = apriori(
        transactions, min_support=MIN_SUPPORT, use_colnames=True
    )
    print("  found {:,} frequent itemsets".format(len(itemsets)))

    rules = association_rules(
        itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE
    )
    print(
        "  generated {:,} rules with confidence >= {}".format(
            len(rules), MIN_CONFIDENCE
        )
    )

    # Pretty columns for export.
    rules["antecedents_str"] = rules["antecedents"].apply(fmt_set)
    rules["consequents_str"] = rules["consequents"].apply(fmt_set)

    export_cols = [
        "antecedents_str", "consequents_str",
        "support", "confidence", "lift",
    ]
    export = rules[export_cols].rename(
        columns={"antecedents_str": "antecedents", "consequents_str": "consequents"}
    )

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLE_DIR / "association_rules.csv"
    export.sort_values("lift", ascending=False).to_csv(out_path, index=False)
    print()
    print("Saved all rules: " + str(out_path.relative_to(ROOT)))

    # Highlight rules that PREDICT the label (consequents == {suspicious}).
    predicts_label = export[
        rules["consequents"].apply(lambda s: s == frozenset({"suspicious"}))
    ].copy()
    predicts_label = predicts_label.sort_values("lift", ascending=False)

    label_path = TABLE_DIR / "association_rules_predict_suspicious.csv"
    predicts_label.to_csv(label_path, index=False)
    print(
        "Saved label-predicting rules ({} rows): {}".format(
            len(predicts_label), label_path.relative_to(ROOT)
        )
    )

    print()
    print("Top {} rules that predict suspicious=True (sorted by lift):".format(TOP_N))
    print("=" * 80)
    if predicts_label.empty:
        print("  (none found at this threshold; try lowering MIN_CONFIDENCE)")
    else:
        with pd.option_context("display.width", 120, "display.max_colwidth", 60):
            print(predicts_label.head(TOP_N).to_string(index=False))


if __name__ == "__main__":
    main()
