"""File I/O helpers — CSV/JSON dumps and the highlighted HTML table.

The HTML table renderer was useful in the original notebooks for quickly
eyeballing winners; it's preserved here so we can keep using it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def save_json(data: Any, path: Path | str) -> Path:
    """Pretty-print JSON to disk. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def save_csv(df: pd.DataFrame, path: Path | str) -> Path:
    """Write a DataFrame to CSV (utf-8, no index)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


DEFAULT_HIGHLIGHT_COLUMNS = [
    "totalBought",
    "currentValue",
    "realizedPnl",
    "totalPnl",
    "account_total_traded_volume",
    "account_total_gain_loss",
    "suspicious_score",
]

DEFAULT_FORMAT_MAP: Dict[str, str] = {
    "size": "{:,.2f}",
    "totalBought": "{:,.2f}",
    "avgPrice": "{:,.4f}",
    "currPrice": "{:,.4f}",
    "currentValue": "{:,.2f}",
    "cashPnl": "{:,.2f}",
    "realizedPnl": "{:,.2f}",
    "totalPnl": "{:,.2f}",
    "account_total_traded_volume": "{:,.2f}",
    "account_total_gain_loss": "{:,.2f}",
}


def save_highlighted_table_html(
    df: pd.DataFrame,
    output_file: Path | str,
    title: str,
    subtitle: str = "",
    highlight_columns: Optional[List[str]] = None,
) -> Path:
    """Save a DataFrame to a styled HTML table (gradient highlights on chosen cols)."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        output_file.write_text(
            f"<html><body><h2>{title}</h2><p>No data available</p></body></html>",
            encoding="utf-8",
        )
        return output_file

    cols = highlight_columns or DEFAULT_HIGHLIGHT_COLUMNS
    format_map = {k: v for k, v in DEFAULT_FORMAT_MAP.items() if k in df.columns}

    styler = df.style.format(format_map)
    for col in cols:
        if col in df.columns:
            styler = styler.background_gradient(cmap="Reds", subset=[col])

    styler = styler.set_properties(
        **{
            "text-align": "left",
            "border": "1px solid #d0d0d0",
            "font-size": "13px",
            "padding": "6px",
        }
    ).set_table_styles(
        [
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("width", "100%"),
                    ("font-family", "Arial, sans-serif"),
                ],
            },
            {
                "selector": "th",
                "props": [
                    ("background-color", "#f5f5f5"),
                    ("border", "1px solid #d0d0d0"),
                    ("padding", "6px"),
                    ("position", "sticky"),
                    ("top", "0"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border", "1px solid #d0d0d0"),
                    ("padding", "6px"),
                ],
            },
        ]
    )

    html = f"""<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
</head>
<body>
  <h2>{title}</h2>
  {f"<p>{subtitle}</p>" if subtitle else ""}
  {styler.to_html()}
</body>
</html>"""

    output_file.write_text(html, encoding="utf-8")
    return output_file
