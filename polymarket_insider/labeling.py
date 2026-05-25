"""Heuristic 'suspicious' labeling (weak supervision).

Polymarket does not tell us which trades are insider trades, so we build a
weak label ourselves from a transparent set of signals. Two design ideas:

  1. A two-part label. Behavior signals -- how a bet was placed (size,
     entry price, wallet concentration, wallet size) -- are combined into a
     0..1 `suspicious_score`. The final `suspicious` label then requires
     BOTH a high behavior score AND a winning outcome: an insider places an
     insider-style bet *and* it pays off, because they knew something. A big
     contrarian bet that lost is just a gambler, not an insider.

     The label is allowed to use the outcome because the models are trained
     on behavior features only -- the outcome columns are dropped before
     training. The label encodes what we want to predict; the model never
     sees the answer, so there is no leakage.

  2. Graded signals. Each behavior signal is scored on a 0..1 ramp instead
     of a hard yes/no, so a $5k stake and a $500k stake are not treated as
     the same thing. The weighted sum of the sub-scores is `suspicious_score`
     in [0, 1].

Everything tunable lives in LABELING_CONFIG below. Change the numbers there,
re-run scripts/02_label_dataset.py, and the whole dataset is re-labeled.
"""

import pandas as pd


EPS = 1e-9  # guards divisions


# -- Configuration ----------------------------------------------------------
# Each behavior signal has a linear ramp (low, high) and a weight.
#     value at `low`  -> sub-score 0.0
#     value at `high` -> sub-score 1.0
#     between them    -> linear ;  outside -> clamped to [0, 1]
# `low` may be greater than `high` for "smaller is more suspicious" signals.
# The weights sum to 1.0, so `suspicious_score` always lands in [0, 1].

LABELING_CONFIG = {
    "signals": {
        # Large absolute stake -- insiders bet big when they have an edge.
        "big_position":     {"low": 1_000.0,   "high": 50_000.0, "weight": 0.30},
        # Contrarian entry -- bought an outcome the market thought unlikely.
        "contrarian_entry": {"low": 0.50,      "high": 0.05,     "weight": 0.25},
        # Concentration -- stake as a fraction of the wallet's lifetime volume.
        "concentration":    {"low": 0.05,      "high": 0.60,     "weight": 0.25},
        # Small wallet -- weak proxy for a fresh / throwaway account.
        "small_wallet":     {"low": 200_000.0, "high": 2_000.0,  "weight": 0.20},
    },
    # `behavior_flag` fires once the weighted score reaches this. The final
    # `suspicious` label also requires the position to have won.
    "threshold": 0.50,
    # A rule_* flag fires when its sub-score is at least this high. The flags
    # are the crisp 0/1 view of the signals, used by the association-rules
    # analysis (which cannot work with continuous values).
    "flag_cutoff": 0.50,
}


# (signal_name, human-readable meaning) -- used for printing and the report.
RULES = [
    ("big_position",     "large absolute stake (totalBought)"),
    ("contrarian_entry", "bought a long shot (low avgPrice)"),
    ("concentration",    "stake is a large share of the wallet's lifetime volume"),
    ("small_wallet",     "small wallet by lifetime volume (fresh-account proxy)"),
]


# -- Helpers ----------------------------------------------------------------


def _num(df, col, default):
    """Return df[col] as a numeric Series, with missing values filled."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _ramp(values, low, high):
    """Map a numeric Series onto a 0..1 linear ramp between `low` and `high`.

    Linear inside the span, clamped to 0/1 outside it. Works in both
    directions: when low > high, smaller values score higher.
    """
    if low == high:
        return (values >= high).astype(float)
    return ((values - low) / (high - low)).clip(lower=0.0, upper=1.0)


# -- Signal scoring ---------------------------------------------------------


def _signal_inputs(df):
    """Return the raw numeric value each behavior signal is computed from."""
    total_bought = _num(df, "totalBought", 0.0)
    avg_price = _num(df, "avgPrice", 1.0)            # missing -> not contrarian
    wallet_volume = _num(df, "account_total_traded_volume", 0.0)
    concentration = total_bought / (wallet_volume + EPS)
    return pd.DataFrame(
        {
            "big_position": total_bought,
            "contrarian_entry": avg_price,
            "concentration": concentration,
            "small_wallet": wallet_volume,
            "_wallet_volume": wallet_volume,
        },
        index=df.index,
    )


def compute_signal_scores(df):
    """Return a DataFrame of graded 0..1 sub-scores, one column per signal."""
    if df.empty:
        return pd.DataFrame(index=df.index)

    inputs = _signal_inputs(df)
    scores = pd.DataFrame(index=df.index)
    for name, params in LABELING_CONFIG["signals"].items():
        scores[name] = _ramp(inputs[name], params["low"], params["high"])

    # Without lifetime volume we cannot honestly judge wallet size or
    # concentration, so for those wallets we drop both sub-scores to 0
    # instead of letting missing data look maximally suspicious.
    no_volume = inputs["_wallet_volume"] <= 0
    scores.loc[no_volume, "small_wallet"] = 0.0
    scores.loc[no_volume, "concentration"] = 0.0
    return scores


def compute_rule_flags(df):
    """Return 0/1 flag columns -- the binary view of each behavior signal.

    A flag fires when its sub-score reaches `flag_cutoff`. These feed the
    association-rules analysis, which needs crisp 0/1 columns.
    """
    scores = compute_signal_scores(df)
    if scores.empty:
        return pd.DataFrame(index=df.index)

    cutoff = LABELING_CONFIG["flag_cutoff"]
    flags = pd.DataFrame(index=df.index)
    for name in scores.columns:
        flags["rule_" + name] = (scores[name] >= cutoff).astype(int)
    return flags


def compute_outcome_columns(df):
    """Outcome signals. Used to finalize the label and to validate it -- but
    never given to the models as features."""
    if df.empty:
        return pd.DataFrame(index=df.index)

    realized = _num(df, "realizedPnl", 0.0)
    total_bought = _num(df, "totalBought", 0.0)
    out = pd.DataFrame(index=df.index)
    out["outcome_won"] = (realized > 0).astype(int)
    out["outcome_return_ratio"] = realized / (total_bought + EPS)
    return out


# -- Labeling ---------------------------------------------------------------


def add_suspicious_label(df, threshold=None):
    """Add the heuristic label and all supporting columns to df.

    Returns a new DataFrame (the input is not modified) with these extras:
        score_<signal>    graded 0..1 sub-score per behavior signal
        rule_<signal>     0/1 flag per behavior signal
        suspicious_score  weighted sum of the behavior sub-scores, in [0, 1]
        behavior_flag     1 if suspicious_score >= threshold
        outcome_won           1 if the position realized a profit
        outcome_return_ratio  realizedPnl / totalBought
        suspicious        the training label: 1 if behavior_flag AND outcome_won

    `suspicious` deliberately combines a behavior signal with the outcome.
    That is safe because models are trained on the behavior features only --
    see the module docstring.
    """
    if threshold is None:
        threshold = LABELING_CONFIG["threshold"]

    out = df.copy()
    if df.empty:
        out["suspicious_score"] = pd.Series(dtype=float)
        out["behavior_flag"] = pd.Series(dtype=int)
        out["suspicious"] = pd.Series(dtype=int)
        return out

    scores = compute_signal_scores(df)
    flags = compute_rule_flags(df)
    outcomes = compute_outcome_columns(df)

    # Weighted sum of the behavior sub-scores.
    weighted = pd.Series(0.0, index=df.index)
    for name, params in LABELING_CONFIG["signals"].items():
        weighted = weighted + scores[name] * params["weight"]

    for name in scores.columns:
        out["score_" + name] = scores[name]
    for col in flags.columns:
        out[col] = flags[col]
    for col in outcomes.columns:
        out[col] = outcomes[col]

    out["suspicious_score"] = weighted.clip(lower=0.0, upper=1.0)
    out["behavior_flag"] = (out["suspicious_score"] >= threshold).astype(int)
    # Final label: an insider-style bet that also paid off.
    out["suspicious"] = (out["behavior_flag"] & out["outcome_won"]).astype(int)
    return out


def label_summary(df):
    """Quick stats for sanity-checking the labels."""
    if df.empty or "suspicious" not in df.columns:
        return {"n": 0}

    summary = {
        "n": int(len(df)),
        "n_behavior_flag": int(df["behavior_flag"].sum()),
        "n_suspicious": int(df["suspicious"].sum()),
        "frac_suspicious": float(df["suspicious"].mean()),
        "score_mean": float(df["suspicious_score"].mean()),
        "score_quantiles": {
            q: float(df["suspicious_score"].quantile(q)) for q in (0.5, 0.9, 0.99)
        },
    }

    # Of the positions flagged on behavior, how many actually won? That
    # winning share is exactly what becomes a positive label.
    if "behavior_flag" in df.columns and "outcome_won" in df.columns:
        flagged = df["behavior_flag"] == 1
        summary["behavior_flag_win_rate"] = (
            float(df.loc[flagged, "outcome_won"].mean()) if flagged.any() else None
        )

    return summary
