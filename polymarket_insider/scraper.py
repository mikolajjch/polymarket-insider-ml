"""High-level scraping: events -> positions DataFrame.

The functions here glue together the low-level `api` and `normalize`
helpers into task-shaped operations you call from notebooks/scripts:

    extract_top_positions_for_event   -- one event -> ranked YES/NO holders
    build_positions_dataframe         -- nested dict -> flat DataFrame
    scrape_events                     -- batch many events into one DataFrame
    enrich_with_account_totals        -- add wallet-level volume/PnL columns,
                                         with on-disk cache so re-runs are fast
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from tqdm import tqdm

from polymarket_insider import api
from polymarket_insider.normalize import (
    find_markets_in_event,
    get_condition_id,
    normalize_position,
    to_float_or_zero,
)

logger = logging.getLogger(__name__)


# Path(__file__) = .../polymarket_insider/scraper.py
# .parent = polymarket_insider/    .parent.parent = project root
# This works no matter where Python is launched from (notebook, script, ...).
DEFAULT_WALLET_CACHE = (
    Path(__file__).resolve().parent.parent / "data" / "external" / "wallet_cache.json"
)


# ---- Ranking primitives ---------------------------------------------------


def rank_positions(
    positions: List[Dict[str, Any]],
    rank_by: str = "totalBought",
    top_n: int = 50,
) -> List[Dict[str, Any]]:
    """Sort positions desc by `rank_by` numeric column and keep top N."""
    ranked = sorted(
        positions,
        key=lambda x: to_float_or_zero(x.get(rank_by)),
        reverse=True,
    )
    return ranked[:top_n]


def split_yes_no(
    position_groups: List[Dict[str, Any]],
    rank_by: str = "totalBought",
    top_n: int = 50,
) -> Dict[str, List[Dict[str, Any]]]:
    """Flatten API position-groups and split by YES/NO outcome."""
    yes_positions: List[Dict[str, Any]] = []
    no_positions: List[Dict[str, Any]] = []

    for group in position_groups:
        positions = group.get("positions", []) or []
        for pos in positions:
            normalized = normalize_position(pos)
            outcome = str(normalized.get("outcome", "")).strip().upper()

            if outcome == "YES":
                yes_positions.append(normalized)
            elif outcome == "NO":
                no_positions.append(normalized)

    return {
        "YES": rank_positions(yes_positions, rank_by=rank_by, top_n=top_n),
        "NO": rank_positions(no_positions, rank_by=rank_by, top_n=top_n),
    }


# ---- Per-event extraction --------------------------------------------------


def extract_top_positions_for_event(
    event_id: str,
    rank_by: str = "totalBought",
    top_n: int = 50,
) -> Dict[str, Any]:
    """For an event ID, return a nested dict with top YES/NO holders per market."""
    event = api.get_event_by_id(event_id)
    event_title = event.get("title") or event.get("question") or f"Event {event_id}"
    event_slug = event.get("slug")
    markets = find_markets_in_event(event)

    if not markets:
        raise ValueError(f"No markets found in event {event_id}.")

    results: Dict[str, Any] = {
        "event_id": str(event_id),
        "event_slug": event_slug,
        "event_title": event_title,
        "rank_by": rank_by,
        "markets": [],
    }

    for idx, market in enumerate(markets, start=1):
        condition_id = get_condition_id(market)
        if not condition_id:
            logger.warning("Market %s has no conditionId, skipping.", idx)
            continue

        question = market.get("question") or market.get("title") or f"Market {idx}"
        market_id = market.get("id")
        market_slug = market.get("slug")

        try:
            groups = api.get_market_positions(
                condition_id=condition_id,
                limit=500,
                offset=0,
                status="ALL",
                sort_by="TOTAL_PNL",
                sort_direction="DESC",
            )
        except Exception as exc:
            logger.warning("Skipping market %s: %s", condition_id, exc)
            continue

        split = split_yes_no(groups, rank_by=rank_by, top_n=top_n)

        results["markets"].append(
            {
                "market_id": market_id,
                "market_slug": market_slug,
                "market_question": question,
                "condition_id": condition_id,
                "top_yes": split["YES"],
                "top_no": split["NO"],
            }
        )

    return results


# ---- DataFrame conversion -------------------------------------------------


def build_positions_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """Flatten the nested `extract_top_positions_for_event` output to a DataFrame.

    Each row is one (market, side, ranked holder).
    """
    rows: List[Dict[str, Any]] = []

    for market in data.get("markets", []):
        market_question = market.get("market_question", "")
        market_id = market.get("market_id", "")
        market_slug = market.get("market_slug", "")
        condition_id = market.get("condition_id", "")

        for side_key, side_name in (("top_yes", "YES"), ("top_no", "NO")):
            holders = market.get(side_key, []) or []
            for rank, holder in enumerate(holders, start=1):
                rows.append(
                    {
                        "event_id": data.get("event_id", ""),
                        "event_slug": data.get("event_slug", ""),
                        "event_title": data.get("event_title", ""),
                        "market_id": market_id,
                        "market_slug": market_slug,
                        "market_question": market_question,
                        "condition_id": condition_id,
                        "side": side_name,
                        "rank": rank,
                        "name": holder.get("name") or "",
                        "proxyWallet": holder.get("proxyWallet") or "",
                        "verified": holder.get("verified"),
                        "asset": holder.get("asset") or "",
                        "size": to_float_or_zero(holder.get("size")),
                        "totalBought": to_float_or_zero(holder.get("totalBought")),
                        "avgPrice": to_float_or_zero(holder.get("avgPrice")),
                        "currPrice": to_float_or_zero(holder.get("currPrice")),
                        "currentValue": to_float_or_zero(holder.get("currentValue")),
                        "cashPnl": to_float_or_zero(holder.get("cashPnl")),
                        "realizedPnl": to_float_or_zero(holder.get("realizedPnl")),
                        "totalPnl": to_float_or_zero(holder.get("totalPnl")),
                        "outcomeIndex": holder.get("outcomeIndex"),
                    }
                )

    return pd.DataFrame(rows)


# ---- Batch scraping --------------------------------------------------------


def scrape_events(
    slugs_or_ids: Iterable[str],
    rank_by: str = "totalBought",
    top_n: int = 50,
    resolve_slugs: bool = True,
) -> pd.DataFrame:
    """Scrape many events into one DataFrame.

    Parameters
    ----------
    slugs_or_ids:
        Iterable of event slugs or numeric IDs (strings).
    resolve_slugs:
        If True, treat non-numeric inputs as slugs and resolve them to IDs.

    Failures on individual events are logged and skipped — the function
    never raises on a single bad event.
    """
    frames: List[pd.DataFrame] = []

    for raw in slugs_or_ids:
        token = str(raw).strip()
        if not token:
            continue

        try:
            if resolve_slugs and not token.isdigit():
                event_id = api.get_event_id_from_slug(token)
            else:
                event_id = token

            data = extract_top_positions_for_event(
                event_id=event_id, rank_by=rank_by, top_n=top_n
            )
            df = build_positions_dataframe(data)
            frames.append(df)
            logger.info(
                "Scraped %s (%s): %d rows.", token, event_id, len(df)
            )
        except Exception as exc:
            logger.warning("Failed to scrape %s: %s", token, exc)
            continue

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ---- Wallet-level enrichment ----------------------------------------------


def _load_wallet_cache(path: Path) -> Dict[str, Dict[str, float]]:
    """Read wallet -> stats dict from disk, or {} if file is missing/broken."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # File got corrupted somehow — start over rather than crash.
        return {}


def _save_wallet_cache(cache: Dict[str, Dict[str, float]], path: Path) -> None:
    """Write cache to disk. Uses a temp file + rename so Ctrl+C can't leave
    a half-written JSON on disk (which would crash the next load)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    tmp.replace(path)


def enrich_with_account_totals(
    df: pd.DataFrame,
    cache_path: Optional[Path] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Add `account_total_traded_volume` and `account_total_gain_loss` columns.

    Each unique wallet's leaderboard stats are cached on disk at
    `cache_path` (default: data/external/wallet_cache.json). Re-running
    the notebook only fetches wallets that aren't already in the cache,
    which makes the second run almost instant.

    The cache is saved every 25 wallets and at the end, so even if you
    Ctrl+C halfway through, the wallets we already fetched are preserved.
    """
    if df.empty or "proxyWallet" not in df.columns:
        return df

    cache_path = Path(cache_path) if cache_path else DEFAULT_WALLET_CACHE
    cache = _load_wallet_cache(cache_path)

    unique_wallets = [w for w in df["proxyWallet"].dropna().astype(str).unique() if w]
    todo = [w for w in unique_wallets if w not in cache]

    if verbose:
        print(
            f"{len(unique_wallets)} unique wallets in df, "
            f"{len(unique_wallets) - len(todo)} already cached, "
            f"{len(todo)} to fetch."
        )

    save_every = 25
    for i, wallet in enumerate(tqdm(todo, desc="Fetching wallets", unit="wallet")):
        try:
            cache[wallet] = api.get_user_leaderboard_stats(wallet)
        except Exception as exc:
            logger.warning("Leaderboard fetch failed for %s: %s", wallet, exc)
            cache[wallet] = {
                "account_total_traded_volume": 0.0,
                "account_total_gain_loss": 0.0,
            }
        if (i + 1) % save_every == 0:
            _save_wallet_cache(cache, cache_path)

    # Final save in case the last batch was smaller than save_every.
    if todo:
        _save_wallet_cache(cache, cache_path)

    enriched = df.copy()
    enriched["account_total_traded_volume"] = enriched["proxyWallet"].map(
        lambda w: cache.get(str(w), {}).get("account_total_traded_volume", 0.0)
    )
    enriched["account_total_gain_loss"] = enriched["proxyWallet"].map(
        lambda w: cache.get(str(w), {}).get("account_total_gain_loss", 0.0)
    )
    return enriched


def filter_winners(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with positive realizedPnl, sorted by PnL desc."""
    if df.empty:
        return df.copy()
    winners = df[df["realizedPnl"] > 0].copy()
    if winners.empty:
        return winners
    return winners.sort_values(
        ["realizedPnl", "totalPnl", "totalBought"], ascending=False
    ).reset_index(drop=True)
