import numpy as np

FEATURE_COLUMNS = ["ret1", "ret5", "ma_ratio", "vol20", "log_volume_z"]


def add_features(df):
    """
    Add scale-free features that describe recent market behaviour.

    Every feature uses only past data (trailing windows), so there is no look-ahead.
    """
    out = df.copy()
    close = out["Close"]

    # Daily return: (P_t - P_{t-1}) / P_{t-1}
    out["ret1"] = close.pct_change()

    # Five-day return
    out["ret5"] = close.pct_change(5)

    # Distance from the 10-day moving average: P_t / MA10_t - 1
    ma10 = close.rolling(10).mean()
    out["ma_ratio"] = close / ma10 - 1.0

    # 20-day volatility: standard deviation of daily returns
    out["vol20"] = out["ret1"].rolling(20).std()

    # Volume, log-transformed then standardized over 20 days
    log_volume = np.log1p(out["Volume"])
    out["log_volume_z"] = (log_volume - log_volume.rolling(20).mean()) / (
        log_volume.rolling(20).std() + 1e-9
    )

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out