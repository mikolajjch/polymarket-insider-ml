"""Polymarket insider trade detection - reusable building blocks.

Submodules:
    api        - HTTP client for Gamma + Data Polymarket APIs.
    normalize  - parsing helpers and position normalization.
    scraper    - high-level event/positions scraping into DataFrames.
    features   - feature engineering on top of the raw positions DataFrame.
    labeling   - heuristic rules that produce the `suspicious` label.
    io         - file I/O helpers (HTML tables, CSV/JSON dumps).
"""

from polymarket_insider import api, normalize, scraper, features, labeling, io  # noqa: F401

__version__ = "0.1.0"
