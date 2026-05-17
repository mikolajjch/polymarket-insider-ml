"""Offline smoke tests — no network required.

Validates that parsing, feature engineering and labeling work end-to-end
on a synthetic mini-payload that mirrors the real Polymarket API shape.
"""

from __future__ import annotations

import pandas as pd

from polymarket_insider import features, labeling, normalize, scraper


def _fake_position(**overrides):
    base = {
        "proxyWallet": "0xWHALE",
        "name": "whale_1",
        "asset": "asset-1",
        "size": "1000",
        "avgPrice": "0.20",
        "currPrice": "0.95",
        "currentValue": "9500",
        "cashPnl": "8500",
        "totalBought": "1000",
        "realizedPnl": "8500",
        "totalPnl": "8500",
        "outcome": "YES",
        "outcomeIndex": 0,
        "verified": False,
        "profileImage": None,
        "conditionId": "0xCOND",
    }
    base.update(overrides)
    return base


def test_normalize_position_keeps_canonical_keys():
    raw = _fake_position(extraneous_field="drop-me")
    out = normalize.normalize_position(raw)
    assert "extraneous_field" not in out
    assert set(out.keys()) == set(normalize.POSITION_FIELDS)


def test_parse_float_handles_garbage():
    assert normalize.parse_float("1.5") == 1.5
    assert normalize.parse_float(None) is None
    assert normalize.parse_float("not a number") is None
    assert normalize.to_float_or_zero("not a number") == 0.0


def test_build_positions_dataframe_flattens_event():
    fake_event = {
        "event_id": "1",
        "event_slug": "fake-event",
        "event_title": "Fake Event",
        "rank_by": "totalBought",
        "markets": [
            {
                "market_id": "m1",
                "market_slug": "fake-market",
                "market_question": "Will X happen?",
                "condition_id": "0xCOND",
                "top_yes": [_fake_position(), _fake_position(proxyWallet="0xBOB")],
                "top_no": [_fake_position(outcome="NO")],
            }
        ],
    }
    df = scraper.build_positions_dataframe(fake_event)
    assert len(df) == 3
    assert {"YES", "NO"}.issubset(set(df["side"]))
    # money columns should be parsed to float
    assert df["totalBought"].dtype.kind == "f"


def test_features_and_labels_end_to_end():
    df = pd.DataFrame(
        [
            {  # textbook 'suspicious': small wallet, low entry, big win
                "proxyWallet": "0xWHALE",
                "size": 1000.0,
                "avgPrice": 0.10,
                "currPrice": 0.99,
                "currentValue": 9900.0,
                "cashPnl": 9000.0,
                "totalBought": 9000.0,
                "realizedPnl": 27000.0,
                "totalPnl": 27000.0,
                "account_total_traded_volume": 10000.0,
                "account_total_gain_loss": 27000.0,
            },
            {  # benign whale: huge lifetime volume, modest concentration
                "proxyWallet": "0xPRO",
                "size": 1000.0,
                "avgPrice": 0.55,
                "currPrice": 0.60,
                "currentValue": 600.0,
                "cashPnl": 50.0,
                "totalBought": 550.0,
                "realizedPnl": 50.0,
                "totalPnl": 50.0,
                "account_total_traded_volume": 5_000_000.0,
                "account_total_gain_loss": 800_000.0,
            },
        ]
    )

    df = features.add_engineered_features(df)
    df = labeling.add_suspicious_label(df)

    # Engineered columns exist.
    for col in (
        "roi",
        "pnl_ratio",
        "position_concentration",
        "low_price_entry",
        "log_totalBought",
    ):
        assert col in df.columns

    # The contrived whale row should hit at least the suspicious threshold.
    assert df.loc[0, "suspicious"] == 1
    assert df.loc[1, "suspicious"] == 0

    summary = labeling.label_summary(df)
    assert summary["n"] == 2
    assert summary["n_suspicious"] == 1
