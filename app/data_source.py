import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def _unix_midnight_utc(day):
    """Convert a date into the Unix timestamp (seconds since 1970) that Yahoo expects."""
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def _request_chart(ticker, params, timeout=30, attempts=3):
    """Call Yahoo's chart endpoint, retrying a few times on network errors or rate limits."""
    last_problem = None
    for attempt in range(attempts):
        try:
            res = requests.get(
                CHART_URL.format(ticker=ticker),
                params=params,
                headers=HEADERS,
                timeout=timeout,
            )
            if res.status_code == 404:
                raise ValueError("Yahoo Finance does not recognise the ticker '{}'.".format(ticker))
            if res.status_code == 429:
                last_problem = "Yahoo Finance is rate-limiting requests (HTTP 429)."
                time.sleep(3 * (attempt + 1))
                continue
            res.raise_for_status()
            return res.json()
        except requests.RequestException as exc:
            last_problem = str(exc)
            time.sleep(2 * (attempt + 1))

    raise RuntimeError("Could not reach Yahoo Finance after {} tries. {}".format(attempts, last_problem))


def download_ticker(ticker, start, end):
    """
    Download daily prices for one ticker from Yahoo Finance's chart endpoint.

    The end date is included: Yahoo's period2 is exclusive, so one day is added.
    Returns a DataFrame with columns Date, Open, High, Low, Close, Volume.
    Raises ValueError for problems the user can fix (bad ticker, no data).
    Raises RuntimeError when Yahoo cannot be reached.
    """
    params = {
        "period1": _unix_midnight_utc(start),
        "period2": _unix_midnight_utc(end + timedelta(days=1)),
        "interval": "1d",
        "events": "history",
    }

    payload = _request_chart(ticker, params)

    chart = payload.get("chart", {})
    if chart.get("error"):
        description = chart["error"].get("description", "unknown error")
        raise ValueError("Yahoo Finance reported an error for {}: {}".format(ticker, description))

    results = chart.get("result") or []
    if not results or not results[0].get("timestamp"):
        raise ValueError(
            "No data found for {} between {} and {}. Check the ticker symbol and the dates.".format(
                ticker, start, end
            )
        )

    result = results[0]
    quote = result["indicators"]["quote"][0]

    # Yahoo gives one Unix timestamp per trading day. Convert it to a calendar date in UTC.
    dates = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_localize(None).normalize()

    df = pd.DataFrame({
        "Date": dates,
        "Open": quote.get("open"),
        "High": quote.get("high"),
        "Low": quote.get("low"),
        "Close": quote.get("close"),
        "Volume": quote.get("volume"),
    })

    # Some days have missing values (for example, trading halts). Convert to numbers, then drop them.
    for col in PRICE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0)

    df = df[(df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))]
    if df.empty:
        raise ValueError(
            "No trading days for {} between {} and {}.".format(ticker, start, end)
        )

    return df[REQUIRED_COLUMNS].reset_index(drop=True)