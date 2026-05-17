"""HTTP client for Polymarket's public APIs.

Two endpoints are used:
    * Gamma API  (https://gamma-api.polymarket.com)  -> event/market metadata
    * Data API   (https://data-api.polymarket.com)   -> positions, leaderboard

Both are public read-only endpoints and do not require auth.

This module wraps `requests` with:
    * a single `get_json` helper that raises on non-2xx,
    * tenacity-based retry with exponential backoff for transient errors,
    * a configurable polite sleep between calls (rate limiting).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
REQUEST_TIMEOUT = 20

# Polite-by-default rate limit. Override via `set_rate_limit_sleep(...)`.
_RATE_LIMIT_SLEEP = 0.4

logger = logging.getLogger(__name__)


def set_rate_limit_sleep(seconds: float) -> None:
    """Set the global sleep between successful HTTP calls."""
    global _RATE_LIMIT_SLEEP
    _RATE_LIMIT_SLEEP = max(0.0, float(seconds))


class PolymarketAPIError(RuntimeError):
    """Raised when Polymarket API returns an unexpected response."""


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(
        (requests.ConnectionError, requests.Timeout, requests.HTTPError)
    ),
)
def get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """GET `url` and return parsed JSON. Retries on transient errors."""
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    # Retry on 5xx and 429 — surface 4xx other than 429 immediately.
    if response.status_code == 429 or response.status_code >= 500:
        response.raise_for_status()
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        raise PolymarketAPIError(f"Non-JSON response from {url}") from exc

    if _RATE_LIMIT_SLEEP:
        time.sleep(_RATE_LIMIT_SLEEP)
    return payload


# ---- Gamma API: events / markets ------------------------------------------


def get_event_by_slug(slug: str) -> Dict[str, Any]:
    """Fetch event metadata by human slug (e.g. 'will-trump-win-2024')."""
    slug = slug.strip()
    if not slug:
        raise ValueError("Slug cannot be empty.")
    return get_json(f"{GAMMA_API}/events/slug/{slug}")


def get_event_by_id(event_id: str) -> Dict[str, Any]:
    """Fetch event metadata by numeric ID."""
    event_id = str(event_id).strip()
    if not event_id:
        raise ValueError("Event ID cannot be empty.")
    return get_json(f"{GAMMA_API}/events/{event_id}")


def get_event_id_from_slug(slug: str) -> str:
    """Convenience: slug -> event ID."""
    event = get_event_by_slug(slug)
    event_id = event.get("id")
    if event_id is None:
        raise PolymarketAPIError(f"No event ID found for slug: {slug}")
    return str(event_id)


# ---- Data API: positions / leaderboard ------------------------------------


def get_market_positions(
    condition_id: str,
    limit: int = 500,
    offset: int = 0,
    status: str = "ALL",
    sort_by: str = "TOTAL_PNL",
    sort_direction: str = "DESC",
) -> List[Dict[str, Any]]:
    """Fetch positions for a single market identified by `condition_id`.

    `status="ALL"` returns both open and closed/cashed-out positions —
    important so we don't miss winners who already redeemed.
    """
    return get_json(
        f"{DATA_API}/v1/market-positions",
        params={
            "market": condition_id,
            "status": status,
            "sortBy": sort_by,
            "sortDirection": sort_direction,
            "limit": limit,
            "offset": offset,
        },
    )


def get_user_leaderboard_stats(user: str) -> Dict[str, float]:
    """Return aggregate volume + PnL for a single wallet from the leaderboard."""
    rows = get_json(
        f"{DATA_API}/v1/leaderboard",
        params={
            "user": user,
            "timePeriod": "ALL",
            "orderBy": "VOL",
            "limit": 1,
            "offset": 0,
        },
    )

    from polymarket_insider.normalize import to_float_or_zero

    if isinstance(rows, list) and rows:
        row = rows[0]
        return {
            "account_total_traded_volume": to_float_or_zero(row.get("vol")),
            "account_total_gain_loss": to_float_or_zero(row.get("pnl")),
        }

    return {
        "account_total_traded_volume": 0.0,
        "account_total_gain_loss": 0.0,
    }
