"""Heuristic 'suspicious' labeling.

Polymarket does not label trades as insider/legitimate, so we manufacture
a weak label from a transparent rule set. Each rule that a position
satisfies contributes +1 to a `suspicious_score`. A position is labeled
suspicious=1 once its score reaches SUSPICIOUS_THRESHOLD.
"""

import pandas as pd


# -- Rule thresholds (tunable from here) ------------------------------------

BIG_POSITION_USD = 5_000
LOW_ENTRY_PRICE = 0.30
HIGH_PNL_RATIO = 2.0           # realizedPnl >= 2 * totalBought  (>=200% return)
SMALL_WALLET_USD = 50_000      # wallet lifetime volume below this = small fish
HIGH_CONCENTRATION = 0.50      # this position was >=50% of wallet lifetime vol

SUSPICIOUS_THRESHOLD = 4       # how many rules must fire to label suspicious=1


# (rule_name, human-readable description). Used for printing / report;
# the actual flag computation is hard-coded in compute_rule_flags below.
RULES = [
    ("big_position", f"totalBought >= ${BIG_POSITION_USD:,}"),
    ("low_entry_price", f"avgPrice < {LOW_ENTRY_PRICE} (contrarian entry)"),
    ("high_pnl_ratio", f"realizedPnl / totalBought >= {HIGH_PNL_RATIO}"),
    ("winning_position", "cashPnl > 0 (already realized profit)"),
    ("small_wallet", f"account_total_traded_volume < ${SMALL_WALLET_USD:,}"),
    ("high_concentration", f"position_concentration >= {HIGH_CONCENTRATION}"),
]


# -- Labeling ---------------------------------------------------------------


def compute_rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of 0/1 flag columns, one per rule."""
    if df.empty:
        return pd.DataFrame(index=df.index)

    flags = pd.DataFrame(index=df.index)
    flags["rule_big_position"] = (df["totalBought"] >= BIG_POSITION_USD).astype(int)
    flags["rule_low_entry_price"] = (df["avgPrice"] < LOW_ENTRY_PRICE).astype(int)

    # totalBought may be 0 — replace with NaN before dividing to avoid inf.
    pnl_ratio = df["realizedPnl"] / df["totalBought"].replace(0, pd.NA)
    flags["rule_high_pnl_ratio"] = (pnl_ratio.fillna(0) >= HIGH_PNL_RATIO).astype(int)

    flags["rule_winning_position"] = (df["cashPnl"] > 0).astype(int)

    if "account_total_traded_volume" in df.columns:
        flags["rule_small_wallet"] = (
            df["account_total_traded_volume"] < SMALL_WALLET_USD
        ).astype(int)
    else:
        flags["rule_small_wallet"] = 0

    if "position_concentration" in df.columns:
        flags["rule_high_concentration"] = (
            df["position_concentration"] >= HIGH_CONCENTRATION
        ).astype(int)
    else:
        flags["rule_high_concentration"] = 0

    return flags


def add_suspicious_label(df: pd.DataFrame, threshold: int = SUSPICIOUS_THRESHOLD) -> pd.DataFrame:
    """Add suspicious_score (int) and suspicious (0/1) columns to df.

    Also attaches the individual rule_* flag columns so we can later see
    which rule pushed each row over the threshold.
    """
    if df.empty:
        out = df.copy()
        out["suspicious_score"] = 0
        out["suspicious"] = 0
        return out

    flags = compute_rule_flags(df)
    out = df.copy()
    out["suspicious_score"] = flags.sum(axis=1).astype(int)
    out["suspicious"] = (out["suspicious_score"] >= threshold).astype(int)
    for col in flags.columns:
        out[col] = flags[col]
    return out


def label_summary(df: pd.DataFrame) -> dict:
    """Quick stats for sanity-checking the labels."""
    if df.empty or "suspicious" not in df.columns:
        return {"n": 0}
    return {
        "n": int(len(df)),
        "n_suspicious": int(df["suspicious"].sum()),
        "frac_suspicious": float(df["suspicious"].mean()),
        "score_distribution": df["suspicious_score"].value_counts().sort_index().to_dict(),
    }
