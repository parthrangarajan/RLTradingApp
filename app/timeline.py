import numpy as np


def _position_label(shares):
    if shares > 0:
        return "LONG"
    if shares < 0:
        return "SHORT"
    return "FLAT"


def _explain(kind, delta, price, fee, realized, new_shares):
    qty = abs(delta)
    verb = "Bought" if delta > 0 else "Sold"
    text = "{} {} shares at ${:.2f} (value ${:,.2f}, fee ${:.2f}).".format(
        verb, qty, price, qty * price, fee
    )
    if realized != 0:
        text += " Locked in a {} of ${:,.2f}.".format(
            "profit" if realized > 0 else "loss", abs(realized)
        )
    if new_shares > 0:
        text += " You now own {} shares.".format(new_shares)
    elif new_shares < 0:
        text += " You now owe {} borrowed shares.".format(abs(new_shares))
    else:
        text += " You now hold no shares."
    return text


def build_timeline(dates, prices, shares, starting_capital, fee_rate, benchmarks):
    """
    Build a day-by-day account record.

    dates:      DatetimeIndex, one entry per evaluation day
    prices:     closing prices, same length
    shares:     shares held on each day (positive = long, negative = short)
    benchmarks: dict of name -> equity array, same length

    Trade timing matches the environment: a position change shown on day k
    was executed at the closing price of day k-1.

    Cost basis and profit/loss:
        Entry price E is the average price paid for the open shares.
        Unrealized P&L = shares * (P_t - E)
        Realized P&L on a closing trade = (price - E) * sign(prev) * shares closed
    """
    prices = np.asarray(prices, dtype=float)
    shares = np.asarray(shares, dtype=int)
    n = len(prices)

    cash = float(starting_capital)
    entry = 0.0
    realized_total = 0.0
    fees_total = 0.0
    prev = 0
    days, trades = [], []

    for k in range(n):
        if k > 0 and shares[k] != prev:
            new = int(shares[k])
            delta = new - prev
            p = float(prices[k - 1])
            trade_date = dates[k - 1].strftime("%Y-%m-%d")

            fee = fee_rate * abs(delta) * p
            cash -= delta * p + fee
            fees_total += fee

            realized = 0.0
            if prev == 0:
                kind = "OPEN LONG" if new > 0 else "OPEN SHORT"
                entry = p
            elif new == 0:
                kind = "CLOSE LONG" if prev > 0 else "CLOSE SHORT"
                realized = (p - entry) * np.sign(prev) * abs(delta)
                entry = 0.0
            elif np.sign(new) == np.sign(prev):
                if abs(new) > abs(prev):
                    kind = "ADD LONG" if new > 0 else "ADD SHORT"
                    entry = (abs(prev) * entry + abs(delta) * p) / float(abs(new))
                else:
                    kind = "REDUCE LONG" if new > 0 else "REDUCE SHORT"
                    realized = (p - entry) * np.sign(prev) * abs(delta)
            else:
                # Switching from long to short, or short to long, in one trade
                kind = "FLIP TO SHORT" if new < 0 else "FLIP TO LONG"
                realized = (p - entry) * np.sign(prev) * abs(prev)
                entry = p

            realized = float(realized)
            realized_total += realized

            trades.append({
                "date": trade_date,
                "kind": kind,
                "direction": "BUY" if delta > 0 else "SELL",
                "shares_traded": int(abs(delta)),
                "price": round(p, 4),
                "notional": float(abs(delta) * p),
                "fee": float(fee),
                "realized_pnl": realized,
                "shares_after": new,
                "position_after": _position_label(new),
                "cash_after": float(cash),
                "entry_after": float(entry),
                "explanation": _explain(kind, delta, p, fee, realized, new),
            })

        pos = int(shares[k])
        price = float(prices[k])
        equity = cash + pos * price
        unrealized = pos * (price - entry) if pos != 0 else 0.0

        days.append({
            "date": dates[k].strftime("%Y-%m-%d"),
            "price": price,
            "shares": pos,
            "position": _position_label(pos),
            "entry_price": float(entry) if pos != 0 else None,
            "position_value": float(pos * price),
            "unrealized_pnl": float(unrealized),
            "realized_pnl_total": float(realized_total),
            "cash": float(cash),
            "equity": float(equity),
            "total_pnl": float(equity - starting_capital),
            "return_pct": float((equity / starting_capital - 1.0) * 100.0),
            "close_action": ("SELL" if pos > 0 else "BUY BACK") if pos != 0 else "NONE",
            "close_shares": int(abs(pos)),
            "benchmarks": {name: float(arr[k]) for name, arr in benchmarks.items()},
        })

        prev = pos

    return {
        "days": days,
        "trades": trades,
        "fees_total": float(fees_total),
        "realized_pnl_total": float(realized_total),
    }