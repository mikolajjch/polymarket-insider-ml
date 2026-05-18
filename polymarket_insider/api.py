"""HTTP client for Polymarket's public APIs.

Two endpoints are used, both public, no auth needed:
    Gamma API  (https://gamma-api.polymarket.com)  -> events / markets
    Data API   (https://data-api.polymarket.com)   -> positions / leaderboard
"""

import time
from typing import Any, Dict, List, Optional

import requests

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
REQUEST_TIMEOUT = 20

# Sleep after every successful request to be a polite API client.
RATE_LIMIT_SLEEP = 0.4
MAX_RETRIES = 4


def get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """GET url and return parsed JSON.

    Retries up to MAX_RETRIES times on connection errors and on 5xx/429
    responses, with backoff 1s, 2s, 4s, ... 4xx errors (other than 429)
    are raised immediately because retrying won't help.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
            continue

        status = response.status_code
        if status == 429 or status >= 500:
            if attempt == MAX_RETRIES - 1:
                response.raise_for_status()
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()

        payload = response.json()
        if RATE_LIMIT_SLEEP:
            time.sleep(RATE_LIMIT_SLEEP)
        return payload

    # We should always return or raise inside the loop; this is a safety net.
    raise RuntimeError("get_json: exhausted all retries")


# ---- Gamma API: events / markets ------------------------------------------


def get_event_by_slug(slug: str) -> Dict[str, Any]:
    """Fetch an event by its slug (the part of the URL after /event/)."""
    slug = slug.strip()
    if not slug:
        raise ValueError("Slug cannot be empty.")
    return get_json(f"{GAMMA_API}/events/slug/{slug}")


def get_event_by_id(event_id: str) -> Dict[str, Any]:
    """Fetch an event by numeric ID."""
    event_id = str(event_id).strip()
    if not event_id:
        raise ValueError("Event ID cannot be empty.")
    return get_json(f"{GAMMA_API}/events/{event_id}")


def get_event_id_from_slug(slug: str) -> str:
    """Slug -> numeric event ID."""
    event = get_event_by_slug(slug)
    event_id = event.get("id")
    if event_id is None:
        raise RuntimeError(f"No event ID found for slug: {slug}")
    return str(event_id)


def list_events(
    closed: Optional[bool] = True,
    archived: Optional[bool] = False,
    limit: int = 100,
    offset: int = 0,
    order: str = "volume",
    ascending: bool = False,
) -> List[Dict[str, Any]]:
    """List events from the Gamma API.

    Defaults pull the top resolved events ordered by volume — useful when
    you want to seed events_to_scrape.txt without browsing the website.
    """
    params: Dict[str, Any] = {
        "limit": int(limit),
        "offset": int(offset),
        "order": order,
        "ascending": "true" if ascending else "false",
    }
    if closed is not None:
        params["closed"] = "true" if closed else "false"
    if archived is not None:
        params["archived"] = "true" if archived else "false"

    data = get_json(f"{GAMMA_API}/events", params=params)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]
    raise RuntimeError(f"Unexpected /events response shape: {type(data).__name__}")


# ---- Data API: positions / leaderboard ------------------------------------


def get_market_positions(
    condition_id: str,
    limit: int = 500,
    offset: int = 0,
    status: str = "ALL",
    sort_by: str = "TOTAL_PNL",
    sort_direction: str = "DESC",
) -> List[Dict[str, Any]]:
    """Fetch positions for a market identified by its on-chain conditionId.

    status='ALL' returns open + closed/cashed-out positions, so we don't
    miss winners who already redeemed.
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
    """Aggregate lifetime volume + PnL for a wallet from the leaderboard."""
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

    if isinstance(rows, list) and rows:
        row = rows[0]
        vol = row.get("vol") or 0
        pnl = row.get("pnl") or 0
        return {
            "account_total_traded_volume": float(vol),
            "account_total_gain_loss": float(pnl),
        }

    return {
        "account_total_traded_volume": 0.0,
        "account_total_gain_loss": 0.0,
    }
