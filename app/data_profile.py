import numpy as np


def profile_data(train_df):
    """
    Describe the training prices in numbers the selection logic can use.

    Returns:
        n_days                         number of trading days
        annual_volatility              standard deviation of daily returns times sqrt(252)
        excess_kurtosis                how heavy the tails are (0 for a normal bell curve)
        lag1_autocorrelation           how much today's return predicts tomorrow's
        trend_slope_per_day            average daily growth rate of log price
        trend_t_stat                   how strong that trend is (above 3 in size is strong)
        buy_and_hold_max_drawdown_pct  worst fall from a peak, in percent
    """
    prices = train_df["Close"].values.astype(float)
    n_days = len(prices)

    returns = np.diff(prices) / prices[:-1]
    mean_r = returns.mean()
    sd_r = returns.std(ddof=1)

    annual_volatility = float(sd_r * np.sqrt(252))
    excess_kurtosis = float(((returns - mean_r) ** 4).mean() / sd_r ** 4 - 3.0)
    autocorr = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])

    t = np.arange(n_days, dtype=float)
    log_price = np.log(prices)
    slope, intercept = np.polyfit(t, log_price, 1)
    residuals = log_price - (slope * t + intercept)
    std_error = np.sqrt(
        (residuals ** 2).sum() / (n_days - 2) / ((t - t.mean()) ** 2).sum()
    )
    trend_t_stat = float(slope / std_error)

    running_peak = np.maximum.accumulate(prices)
    max_drawdown_pct = float((prices / running_peak - 1.0).min() * 100.0)

    return {
        "n_days": int(n_days),
        "annual_volatility": annual_volatility,
        "excess_kurtosis": excess_kurtosis,
        "lag1_autocorrelation": autocorr,
        "trend_slope_per_day": float(slope),
        "trend_t_stat": trend_t_stat,
        "buy_and_hold_max_drawdown_pct": max_drawdown_pct,
    }