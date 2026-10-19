import io

import pandas as pd
import requests

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _clean_number(series):
    """Turn strings like '25,845,900' or '$264.50' into floats."""
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_marketwatch_csv(raw):
    """
    Parse a CSV exported from MarketWatch (or any CSV with the same columns).

    Returns a DataFrame indexed by Date, sorted oldest first.
    """
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise ValueError("Could not read the file as CSV: {}".format(exc))

    df.columns = [str(c).strip().title() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing columns {}. Found columns: {}".format(missing, list(df.columns))
        )

    df = df[REQUIRED_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), errors="coerce")

    for col in NUMERIC_COLUMNS:
        df[col] = _clean_number(df[col])

    df = df.dropna(subset=["Date", "Close"])
    df = df.drop_duplicates(subset="Date")
    df = df.sort_values("Date").set_index("Date")

    if len(df) < 60:
        raise ValueError("At least 60 valid rows are needed; found {}.".format(len(df)))

    return df


def fetch_csv_bytes(url, timeout=30):
    """
    Download a CSV from a direct link you are allowed to use.

    This downloads one file. It does not scrape web pages.
    """
    headers = {"User-Agent": "Mozilla/5.0 (rl-trading-api)"}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.content