"""Pytest bootstrap.

Adds the project root to sys.path so `from polymarket_insider import ...`
works without requiring `pip install -e .` first. This is a safety net —
once you've done an editable install, this file is harmless and a no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
