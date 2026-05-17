"""Online smoke test — hits the real Polymarket API.

Run this locally (your machine, not a sandbox) to confirm:
    * API client can resolve a slug,
    * scraper returns rows,
    * features + labeling apply without error.

Usage:
    python scripts/smoke_test_online.py
    python scripts/smoke_test_online.py <event-slug>
"""

from __future__ import annotations

import sys

from polymarket_insider import api, features, labeling, scraper


DEFAULT_SLUG = "presidential-election-winner-2024"


def main() -> int:
    slug = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SLUG
    print(f"[1/4] Resolving slug -> event id ({slug!r}) ...")
    event_id = api.get_event_id_from_slug(slug)
    print(f"      OK, event_id = {event_id}")

    print("[2/4] Scraping top 10 positions per side ...")
    df = scraper.scrape_events([slug], top_n=10)
    print(f"      OK, got {len(df)} rows")
    if df.empty:
        print("      WARNING: scraper returned 0 rows — try a different slug.")
        return 1

    print("[3/4] Enriching wallets with leaderboard stats ...")
    df = scraper.enrich_with_account_totals(df)
    print(f"      OK, {df['proxyWallet'].nunique()} unique wallets enriched")

    print("[4/4] Feature engineering + labeling ...")
    df = features.add_engineered_features(df)
    df = labeling.add_suspicious_label(df)
    summary = labeling.label_summary(df)
    print(f"      OK, label summary: {summary}")

    print("\nTop 5 suspicious rows:")
    cols = [
        "side",
        "name",
        "totalBought",
        "avgPrice",
        "realizedPnl",
        "account_total_traded_volume",
        "suspicious_score",
        "suspicious",
    ]
    print(
        df.sort_values("suspicious_score", ascending=False)[cols].head(5).to_string(
            index=False
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
