"""Parsing & normalization helpers shared across scraping/feature code."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ---- Type-safe parsing ----------------------------------------------------


def parse_float(value: Any) -> Optional[float]:
    """Return float(value) or None if it's not parseable."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def to_float_or_zero(value: Any) -> float:
    """Like `parse_float` but returns 0.0 instead of None — useful for sums."""
    parsed = parse_float(value)
    return parsed if parsed is not None else 0.0


def safe_json_loads(value: Any) -> Any:
    """Try to JSON-parse a string; otherwise return the value unchanged."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


# ---- Event / market shape navigation --------------------------------------


def find_markets_in_event(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Locate the markets list inside an event payload.

    Gamma API has used different keys across versions (`markets`, `seriesMarkets`,
    `children`) — we try them all.
    """
    for key in ("markets", "seriesMarkets", "children"):
        value = event.get(key)
        if isinstance(value, list):
            return value
    return []


def get_condition_id(market: Dict[str, Any]) -> Optional[str]:
    """Extract the on-chain condition ID from a market record."""
    for key in ("conditionId", "condition_id", "condition"):
        value = market.get(key)
        if isinstance(value, str) and value:
            return value
    return None


# ---- Position normalization -----------------------------------------------


# Canonical keys we keep on every position row. Anything else from the API
# response is dropped at this point.
POSITION_FIELDS = (
    "proxyWallet",
    "name",
    "asset",
    "size",
    "avgPrice",
    "currPrice",
    "currentValue",
    "cashPnl",
    "totalBought",
    "realizedPnl",
    "totalPnl",
    "outcome",
    "outcomeIndex",
    "verified",
    "profileImage",
    "conditionId",
)


def normalize_position(position: Dict[str, Any]) -> Dict[str, Any]:
    """Project a raw position dict to the canonical schema."""
    return {field: position.get(field) for field in POSITION_FIELDS}
