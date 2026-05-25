"""Feature engineering on the raw positions DataFrame.

`add_engineered_features` computes every derived column we might want --
both behavior features (size, entry price, concentration ...) and
outcome-derived ones (roi, pnl_ratio ...). The outcome-derived columns are
fine for EDA and for building the label, but they must NOT be fed to the
models, because the label already encodes the outcome. Use
`behavior_feature_columns()` to get the leak-free subset the models may see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-9  # avoid div-by-zero in ratios


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived feature columns. Returns a new DataFrame (does not mutate)."""
    if df.empty:
        return df.copy()

    out = df.copy()

    # ROI from total spent vs current value of remaining position.
    out["roi"] = (out["currentValue"] - out["totalBought"]) / (out["totalBought"] + EPS)

    # Realized PnL relative to what they put in -> "how big was the win/loss".
    out["pnl_ratio"] = out["realizedPnl"] / (out["totalBought"] + EPS)

    # Total PnL relative to total spent.
    out["total_pnl_ratio"] = out["totalPnl"] / (out["totalBought"] + EPS)

    # How concentrated is this position relative to the wallet's lifetime
    # volume? > 1 means the wallet bet more on this one market than it ever
    # traded overall -- a strong "account exists for one bet" signal. For
    # wallets with no recorded volume we cannot compute this, so we set it to
    # 0 rather than dividing by ~0 and getting an astronomically large number.
    if "account_total_traded_volume" in out.columns:
        volume = out["account_total_traded_volume"]
        concentration = out["totalBought"] / volume.where(volume > 0)
        out["position_concentration"] = concentration.fillna(0.0)

    # Did they enter at a very low price? (Contrarian / "knew something" signal.)
    out["low_price_entry"] = (out["avgPrice"] < 0.30).astype(int)
    out["very_low_price_entry"] = (out["avgPrice"] < 0.15).astype(int)

    # Big absolute spend (binary cutoff used in labeling too).
    out["big_position"] = (out["totalBought"] >= 5_000).astype(int)

    # Log-transforms for the heavy-tailed money columns.
    for col in ("totalBought", "currentValue", "realizedPnl", "totalPnl"):
        if col in out.columns:
            out[f"log_{col}"] = np.log1p(np.maximum(out[col], 0.0))

    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Every numeric column, including outcome-derived ones.

    Useful for EDA. Do NOT use this for model inputs -- it includes columns
    like realizedPnl and roi that leak the outcome. Use
    `behavior_feature_columns` for training instead.
    """
    candidates = [
        "size",
        "avgPrice",
        "currPrice",
        "currentValue",
        "totalBought",
        "realizedPnl",
        "totalPnl",
        "cashPnl",
        "account_total_traded_volume",
        "account_total_gain_loss",
        "roi",
        "pnl_ratio",
        "total_pnl_ratio",
        "position_concentration",
        "low_price_entry",
        "very_low_price_entry",
        "big_position",
        "log_totalBought",
        "log_currentValue",
        "log_realizedPnl",
        "log_totalPnl",
    ]
    return [c for c in candidates if c in df.columns]


def behavior_feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns the models are allowed to see.

    These describe HOW a bet was placed -- stake size, entry price, wallet
    concentration and wallet size -- and are all known at entry time.
    Outcome columns (realizedPnl, currPrice, currentValue, roi, pnl_ratio,
    ...) are deliberately excluded: the `suspicious` label already encodes
    the outcome, so feeding it back as a feature would be leakage.
    """
    candidates = [
        "size",
        "avgPrice",
        "totalBought",
        "account_total_traded_volume",
        "position_concentration",
        "low_price_entry",
        "very_low_price_entry",
        "big_position",
        "log_totalBought",
    ]
    return [c for c in candidates if c in df.columns]
