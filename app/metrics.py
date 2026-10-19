import numpy as np

TRADING_DAYS = 252


def max_drawdown(equity):
    """
    Largest fall from a previous peak.

    MDD = min_t ( E_t / max_{s<=t} E_s - 1 )
    """
    equity = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def sharpe_ratio(equity):
    """
    Annualized Sharpe ratio, assuming a zero risk-free rate.

    S = sqrt(252) * mean(r) / std(r)

    r_t = E_t / E_{t-1} - 1  (daily returns)
    """
    equity = np.asarray(equity, dtype=float)
    returns = equity[1:] / equity[:-1] - 1.0
    sd = returns.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * returns.mean() / sd)


def summarize(equity, n_trades):
    equity = np.asarray(equity, dtype=float)
    return {
        "final_equity": float(equity[-1]),
        "total_return_pct": float((equity[-1] / equity[0] - 1.0) * 100.0),
        "sharpe_ratio": sharpe_ratio(equity),
        "max_drawdown_pct": float(max_drawdown(equity) * 100.0),
        "n_trades": int(n_trades),
    }


def strategy_equity(prices, positions, starting_capital, fee_rate):
    """
    Simulate a benchmark strategy from a list of target positions.

    positions[t] is in {-1, 0, +1} (or a fraction). It is decided at day t and
    applied to the move from day t to day t+1, so there is no look-ahead.

    E_{t+1} = E_t * (1 + pos_t * r_t - fee * |pos_t - pos_{t-1}|)
    """
    prices = np.asarray(prices, dtype=float)
    positions = np.asarray(positions, dtype=float)

    equity = np.empty(len(prices))
    equity[0] = starting_capital
    previous = 0.0
    n_trades = 0

    for t in range(len(prices) - 1):
        position = positions[t]
        daily_return = prices[t + 1] / prices[t] - 1.0
        change = abs(position - previous)
        if change > 0:
            n_trades += 1
        cost = fee_rate * change
        equity[t + 1] = equity[t] * (1.0 + position * daily_return - cost)
        previous = position

    return equity, n_trades