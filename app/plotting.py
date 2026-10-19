import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def render_long_short_chart(dates, prices, held, trade_points, rl_equity, bh_equity, title):
    """
    Build a two-panel PNG.

    Top:    price, with green shading for long periods and red for short periods.
            Triangles mark each trade: green up for buys, red down for sells.
    Bottom: portfolio value of the RL agent versus buy and hold.

    held:         array, position during each day's interval (shares)
    trade_points: list of (index, trade_size) tuples
    """
    x = dates.to_pydatetime()
    prices = np.asarray(prices, dtype=float)
    held = np.asarray(held)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    low, high = prices.min(), prices.max()
    pad = (high - low) * 0.05
    ax1.plot(x, prices, color="black", linewidth=1.2, label="Close price")
    ax1.fill_between(
        x, low - pad, high + pad, where=held > 0,
        color="green", alpha=0.15, label="Long position",
    )
    ax1.fill_between(
        x, low - pad, high + pad, where=held < 0,
        color="red", alpha=0.15, label="Short position",
    )

    for idx, size in trade_points:
        if size > 0:
            ax1.scatter(x[idx], prices[idx], color="green", marker="^", s=60, zorder=5)
        else:
            ax1.scatter(x[idx], prices[idx], color="red", marker="v", s=60, zorder=5)

    ax1.set_ylabel("Price")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left")

    ax2.plot(x, rl_equity, color="blue", label="RL agent")
    ax2.plot(x, bh_equity, color="gray", linestyle="--", label="Buy and hold")
    ax2.set_ylabel("Portfolio value")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left")

    fig.autofmt_xdate()
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    return buffer.getvalue()