"""Feature engineering on the raw positions DataFrame.

These features are the inputs the ML models will see. Keep them:
    * deterministic (no leakage from labels or future info),
    * stable (same name -> same definition),
    * safe (no inf/NaN explosions on edge cases).
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

    # How concentrated is this position relative to the wallet's lifetime volume?
    # > 1 would mean the wallet bet more on this one market than it ever traded
    # overall — strong signal of an account that exists for one bet.
    if "account_total_traded_volume" in out.columns:
        out["position_concentration"] = out["totalBought"] / (
            out["account_total_traded_volume"] + EPS
        )

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
    """The canonical list of numeric columns models should see.

    Intentionally excludes IDs, wallet hashes, and free-text columns.
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
