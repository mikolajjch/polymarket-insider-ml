# Problem definition — Polymarket Insider ML

## Goal

Given a snapshot of trading positions on a Polymarket event, score every
position (and the underlying wallet) for how *suspicious* — i.e. how
consistent with insider-style behavior — it looks.

We frame this as **binary classification with a continuous risk score**:

* Labels are produced by a transparent heuristic rule set (see below).
* Models are trained on those labels and output `predict_proba(suspicious=1)`.
* In the app and in the report, that probability is presented as a 0.0–1.0
  **risk score**, not as a hard verdict.

## Why heuristic labels?

Polymarket does not label trades as insider/legitimate. There is no public
ground truth. We have two choices:

1. **Pure unsupervised** (Isolation Forest, LOF). Honest about lack of
   labels, but doesn't match the course topic ("klasyfikacja").
2. **Heuristic-labeled supervised** with explicit rules. Lets us do
   classification, train + compare many models, *and* compare against
   unsupervised baselines.

We choose option 2 as the main approach, and use option 1 as an additional
baseline for the report's "aspekt badawczy" section.

The rules are noisy by design — we will say so clearly in the report.

## Suspicious rules (v1)

A position scores +1 for each of the following rules it satisfies. A
position is labeled `suspicious=1` iff its total score is **≥ 4**.

| Rule                | Condition                                                |
|---------------------|----------------------------------------------------------|
| `big_position`      | `totalBought >= $5,000`                                  |
| `low_entry_price`   | `avgPrice < 0.30` (contrarian entry)                     |
| `high_pnl_ratio`    | `realizedPnl / totalBought >= 2.0`                       |
| `winning_position`  | `cashPnl > 0`                                            |
| `small_wallet`      | `account_total_traded_volume < $50,000`                  |
| `high_concentration`| `position_concentration >= 0.50`                         |

`position_concentration = totalBought / account_total_traded_volume`.

These rules are encoded in `polymarket_insider.labeling`. Threshold and
constants live there too, so they're easy to tune from one place.

## Sanity checks for the labels

Before training anything we will:

1. Print `frac_suspicious` per event and overall. Expected: 5–15% globally.
2. Hand-inspect ~50 top-scored positions — do they look like plausible
   insider-style bets (right-side, low entry, late, big win)?
3. Hand-inspect ~50 bottom-scored positions — are they obviously not
   insider-style?
4. Check that the rules don't all fire together trivially (correlations
   between rule flags should not be ~1.0).

## Out of scope

* On-chain forensics (graph-of-wallets, mixers, funding sources).
* Real-time alerting.
* Time-series modeling of trades within a market.
* Claims about specific individuals — we will not name wallets in the report.
