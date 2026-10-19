import gym
import numpy as np
from gym import spaces

SHORT, FLAT, LONG = 0, 1, 2


class TradingEnv(gym.Env):
    """
    A single-stock trading environment.

    Actions:
        0 = target a SHORT position (becomes FLAT if shorting is not allowed)
        1 = target a FLAT position (no shares)
        2 = target a LONG position

    Position size:
        units = min(max_shares, floor(max_fraction * equity / price))

    Reward modes:
        "pnl":          r_t = 100 * (E_{t+1} - E_t) / C0
        "risk_penalty": r_t = pnl_t - lambda * max(0, DD_t - DD_{t-1})

        E_t    = account value (cash plus shares times price) on day t
        C0     = starting capital
        DD_t   = drawdown in percent: how far the account sits below its highest value so far
        lambda = drawdown_penalty, how strongly growing drawdowns are punished
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        prices,
        features,
        window_size=10,
        starting_capital=10000.0,
        max_fraction_per_trade=0.25,
        max_shares_per_trade=500,
        fee_rate=0.001,
        allow_short=True,
        reward_mode="pnl",
        drawdown_penalty=1.0,
    ):
        super(TradingEnv, self).__init__()

        self.prices = np.asarray(prices, dtype=np.float64)
        self.features = np.asarray(features, dtype=np.float32)

        if len(self.prices) != len(self.features):
            raise ValueError("prices and features must have the same number of rows")
        if len(self.prices) <= window_size + 1:
            raise ValueError("Not enough rows for window_size={}".format(window_size))
        if reward_mode not in ("pnl", "risk_penalty"):
            raise ValueError("reward_mode must be 'pnl' or 'risk_penalty'")

        self.window_size = int(window_size)
        self.starting_capital = float(starting_capital)
        self.max_fraction = float(max_fraction_per_trade)
        self.max_shares = int(max_shares_per_trade)
        self.fee_rate = float(fee_rate)
        self.allow_short = bool(allow_short)
        self.reward_mode = reward_mode
        self.drawdown_penalty = float(drawdown_penalty)

        self.n_features = self.features.shape[1]
        obs_dim = self.window_size * self.n_features + 1

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.reset()

    # ----- helpers -------------------------------------------------------

    def _equity(self):
        return self.cash + self.shares * self.prices[self.t]

    def _obs(self):
        window = self.features[self.t - self.window_size + 1 : self.t + 1].reshape(-1)
        position_ratio = np.array([self.shares / float(self.max_shares)], dtype=np.float32)
        return np.concatenate([window, position_ratio]).astype(np.float32)

    def _record(self, action):
        price = self.prices[self.t]
        return {
            "index": int(self.t),
            "price": float(price),
            "shares": int(self.shares),
            "equity": float(self.cash + self.shares * price),
            "action": action,
        }

    # ----- gym API -------------------------------------------------------

    def reset(self):
        self.t = self.window_size - 1
        self.cash = self.starting_capital
        self.shares = 0
        self.n_trades = 0
        self.peak = self.starting_capital
        self.dd_prev = 0.0
        self.history = [self._record(action=None)]
        return self._obs()

    def step(self, action):
        action = int(action)
        if not self.allow_short and action == SHORT:
            action = FLAT

        price = self.prices[self.t]
        equity_before = self._equity()

        units = int(
            min(
                self.max_shares,
                np.floor(self.max_fraction * max(equity_before, 0.0) / price),
            )
        )

        if action == LONG:
            target = units
        elif action == SHORT:
            target = -units
        else:
            target = 0

        delta = target - self.shares
        if delta != 0:
            trade_value = delta * price
            fee = self.fee_rate * abs(trade_value)
            self.cash -= trade_value + fee
            self.shares = target
            self.n_trades += 1

        self.t += 1
        done = self.t >= len(self.prices) - 1

        equity_after = self._equity()
        pnl_pct = 100.0 * (equity_after - equity_before) / self.starting_capital

        if self.reward_mode == "pnl":
            reward = pnl_pct
        else:
            self.peak = max(self.peak, equity_after)
            dd_now = 100.0 * (self.peak - equity_after) / self.starting_capital
            reward = pnl_pct - self.drawdown_penalty * max(0.0, dd_now - self.dd_prev)
            self.dd_prev = dd_now

        self.history.append(self._record(action=action))
        info = {"equity": equity_after, "shares": self.shares, "trades": self.n_trades}

        return self._obs(), float(reward), done, info