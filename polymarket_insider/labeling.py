"""Heuristic labeling of 'suspicious' positions.

Polymarket does not label trades as insider/legitimate, so we manufacture a
weak label from a transparent rule set. Each rule that fires contributes +1
to a `suspicious_score`. A row is labeled `suspicious=1` iff
`suspicious_score >= SUSPICIOUS_THRESHOLD`.

The score itself is also useful (continuous risk signal) and is kept on the
DataFrame so we can train a regressor on it instead of, or in addition to,
the binary classifier.

Rules are intentionally simple, readable, and tunable. Document any change
to thresholds in the project report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


# -- Rule thresholds --------------------------------------------------------

BIG_POSITION_USD = 5_000
LOW_ENTRY_PRICE = 0.30
HIGH_PNL_RATIO = 2.0           # realizedPnl > totalBought * 2 -> 200%+ return
SMALL_WALLET_USD = 50_000      # wallet lifetime volume below this = "small fish"
HIGH_CONCENTRATION = 0.50      # this position was >=50% of wallet lifetime volume

SUSPICIOUS_THRESHOLD = 4       # >= rules fired = label 1


@dataclass(frozen=True)
class Rule:
    name: str
    description: str


RULES: List[Rule] = [
    Rule("big_position", f"totalBought >= ${BIG_POSITION_USD:,}"),
    Rule("low_entry_price", f"avgPrice < {LOW_ENTRY_PRICE} (contrarian entry)"),
    Rule("high_pnl_ratio", f"realizedPnl / totalBought >= {HIGH_PNL_RATIO}"),
    Rule("winning_position", "cashPnl > 0 (already realized profit)"),
    Rule("small_wallet", f"account_total_traded_volume < ${SMALL_WALLET_USD:,}"),
    Rule("high_concentration", f"position_concentration >= {HIGH_CONCENTRATION}"),
]


# -- Labeling ---------------------------------------------------------------


def compute_rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of 0/1 rule-flag columns, one per rule."""
    if df.empty:
        return pd.DataFrame(index=df.index, columns=[f"rule_{r.name}" for r in RULES])

    out = pd.DataFrame(index=df.index)
    out["rule_big_position"] = (df["totalBought"] >= BIG_POSITION_USD).astype(int)
    out["rule_low_entry_price"] = (df["avgPrice"] < LOW_ENTRY_PRICE).astype(int)

    # Guard against div-by-zero rows where totalBought == 0.
    pnl_ratio = df["realizedPnl"] / df["totalBought"].replace(0, pd.NA)
    out["rule_high_pnl_ratio"] = (pnl_ratio.fillna(0) >= HIGH_PNL_RATIO).astype(int)

    out["rule_winning_position"] = (df["cashPnl"] > 0).astype(int)

    if "account_total_traded_volume" in df.columns:
        out["rule_small_wallet"] = (
            df["account_total_traded_volume"] < SMALL_WALLET_USD
        ).astype(int)
    else:
        out["rule_small_wallet"] = 0

    if "position_concentration" in df.columns:
        out["rule_high_concentration"] = (
            df["position_concentration"] >= HIGH_CONCENTRATION
        ).astype(int)
    else:
        out["rule_high_concentration"] = 0

    return out


def add_suspicious_label(
    df: pd.DataFrame,
    threshold: int = SUSPICIOUS_THRESHOLD,
) -> pd.DataFrame:
    """Add `suspicious_score` (int) and `suspicious` (0/1) columns."""
    if df.empty:
        out = df.copy()
        out["suspicious_score"] = 0
        out["suspicious"] = 0
        return out

    flags = compute_rule_flags(df)
    out = df.copy()
    out["suspicious_score"] = flags.sum(axis=1).astype(int)
    out["suspicious"] = (out["suspicious_score"] >= threshold).astype(int)

    # Also attach individual rule flags - useful for the report and for SHAP-like
    # sanity checks (e.g. "which rule drives suspicious_score most?").
    for col in flags.columns:
        out[col] = flags[col]

    return out


def label_summary(df: pd.DataFrame) -> dict:
    """Quick stats for sanity checks after labeling."""
    if df.empty or "suspicious" not in df.columns:
        return {"n": 0}
    return {
        "n": int(len(df)),
        "n_suspicious": int(df["suspicious"].sum()),
        "frac_suspicious": float(df["suspicious"].mean()),
        "score_distribution": df["suspicious_score"].value_counts().sort_index().to_dict(),
    }
