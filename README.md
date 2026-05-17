# polymarket-insider-ml

ML classification of suspicious trades on Polymarket using scikit-learn.

Polymarket provides public APIs for event metadata, market positions and a
wallet-level leaderboard. We scrape positions from a chosen set of events,
attach wallet-level aggregates, label positions with a transparent rule
heuristic, and train classifiers that output a 0.0–1.0 **risk score** for
each (wallet, market) pair.

## Project layout

```
polymarket-insider-ml/
├── polymarket_insider/      # importable package
│   ├── api.py               # Gamma + Data API client (retry, rate limit)
│   ├── normalize.py         # parsing + position shape normalization
│   ├── scraper.py           # event -> positions DataFrame (batch supported)
│   ├── features.py          # engineered feature columns
│   ├── labeling.py          # heuristic 'suspicious' rules + threshold
│   └── io.py                # CSV/JSON/HTML output helpers
├── notebooks/
│   ├── 00_problem_definition.md
│   ├── polymarket_python.ipynb              # legacy / scratch
│   ├── polymartket_insider_analysis.ipynb   # legacy / scratch
│   ├── top_holders.ipynb                    # legacy / scratch
│   └── top_positions.ipynb                  # legacy / scratch
├── data/
│   ├── raw/                 # scraped CSVs (gitignored content)
│   ├── processed/           # cleaned, labeled, feature-engineered
│   └── external/            # any external lookups (e.g. known wallets)
├── app/                     # Streamlit demo (later)
├── reports/                 # figures + final report
├── tests/                   # pytest (optional)
├── events_to_scrape.txt     # list of event slugs to pull
├── requirements.txt
└── pyproject.toml           # `pip install -e .` makes the package importable
```

## Setup

Requires **Python 3.9+** and **pip 21.3+** (for PEP 660 editable installs).

```bash
python -m venv .venv
source .venv/bin/activate                   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip         # one-time; pip 20.x can't do editable pyproject
pip install -r requirements.txt
pip install -e .                            # makes `import polymarket_insider` work
```

Verify:

```bash
pytest tests/ -v                            # 4 offline tests, no network
python scripts/smoke_test_online.py         # hits the real Polymarket API
```

## Quick start

```python
from polymarket_insider import scraper, features, labeling

# 1. Pull a batch of events.
slugs = open("events_to_scrape.txt").read().splitlines()
slugs = [s for s in slugs if s and not s.startswith("#")]
df = scraper.scrape_events(slugs, top_n=50)

# 2. Add wallet-level totals (1 call per unique wallet).
df = scraper.enrich_with_account_totals(df, verbose=True)

# 3. Engineer features and label.
df = features.add_engineered_features(df)
df = labeling.add_suspicious_label(df)

print(labeling.label_summary(df))
```
