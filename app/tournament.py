import numpy as np
import pandas as pd
from stable_baselines.common.vec_env import DummyVecEnv

from .env import TradingEnv
from .features import FEATURE_COLUMNS
from .metrics import sharpe_ratio, strategy_equity, summarize
from .models import act, make_model


# ----------------------------------------------------------------
# Environment and evaluation helpers
# ----------------------------------------------------------------
def build_env(df, cfg):
    return TradingEnv(
        prices=df["Close"].values,
        features=df[FEATURE_COLUMNS].values,
        window_size=cfg["window_size"],
        starting_capital=cfg["starting_capital"],
        max_fraction_per_trade=cfg["max_fraction_per_trade"],
        max_shares_per_trade=cfg["max_shares_per_trade"],
        fee_rate=cfg["fee_rate"],
        allow_short=cfg["allow_short"],
        reward_mode=cfg.get("reward_mode", "pnl"),
        drawdown_penalty=cfg.get("drawdown_penalty", 1.0),
    )


def split_validation(train_df, window, fraction=0.7):
    """
    Split the training period in time order.

    fit_df: the first 70 percent, used to learn.
    val_df: the last 30 percent, plus window-1 rows of history so the first
            validation decision has a full window of past data.
    """
    cut = int(len(train_df) * fraction)
    fit_df = train_df.iloc[:cut]
    val_df = train_df.iloc[cut - window + 1:]
    if len(fit_df) < window + 20:
        raise ValueError("Training period is too short to split. Choose a longer training period.")
    return fit_df, val_df


def validation_benchmark_sharpe(val_df, window, starting_capital, fee_rate):
    """Sharpe ratio of buy and hold on the validation days. This is the bar to beat."""
    prices = val_df["Close"].values[window - 1:]
    equity, _ = strategy_equity(prices, np.ones(len(prices)), starting_capital, fee_rate)
    return sharpe_ratio(equity)


def run_policy(model, frame, cfg):
    """Run a trained model through one period, with no learning, and return the account history."""
    env = build_env(frame, cfg)
    obs = env.reset()
    done = False
    steps_left = len(frame) + 10

    while not done and steps_left > 0:
        obs, _, done, _ = env.step(act(model, obs))
        steps_left -= 1

    history = pd.DataFrame(env.history)
    expected = len(frame) - cfg["window_size"] + 1
    if len(history) != expected:
        raise RuntimeError(
            "History length {} does not match expected {}".format(len(history), expected)
        )

    return {
        "equity": history["equity"].values.astype(float),
        "shares": history["shares"].values.astype(int),
        "trades": int(env.n_trades),
    }


# ----------------------------------------------------------------
# Stage 0: the switch on data profile
# ----------------------------------------------------------------
def choose_candidates(profile, allow_short):
    reasons = []

    if profile["n_days"] < 250:
        reasons.append(
            "Fewer than about one trading year of training days. Deep models memorize small "
            "datasets, so the sample-efficient DQN is tested alone."
        )
        return ["dqn"], reasons

    if profile["excess_kurtosis"] > 5 or profile["annual_volatility"] > 0.8:
        reasons.append(
            "Returns have extreme jumps or very high volatility. Proximal Policy Optimization "
            "limits how far each update can move, so it is the safer choice, with A2C as a comparison."
        )
        return ["ppo", "a2c"], reasons

    if not allow_short:
        reasons.append(
            "Long-or-cash only, which gives a small set of choices. A value-based method such as "
            "DQN handles this well, with PPO as a comparison."
        )
        return ["dqn", "ppo"], reasons

    reasons.append(
        "Typical data with short selling allowed. All three algorithms are tested, and "
        "validation results decide."
    )
    return ["ppo", "a2c", "dqn"], reasons


# ----------------------------------------------------------------
# Stage 1: the tournament
# ----------------------------------------------------------------
def train_candidate(algo, fit_df, cfg, seed):
    run_cfg = dict(cfg, seed=seed)
    vec_env = DummyVecEnv([lambda: build_env(fit_df, run_cfg)])
    model = make_model(algo, vec_env, run_cfg)
    model.learn(total_timesteps=cfg.get("tournament_steps", cfg["training_steps"]))
    return model


def search_best(fit_df, val_df, algos, cfg, progress, base, span, tag):
    """
    Train every algorithm with three seeds, and score each one on validation data.

    score = mean Sharpe ratio across seeds - 0.5 * standard deviation across seeds
    """
    seeds = [cfg["seed"], cfg["seed"] + 1, cfg["seed"] + 2]
    total = len(algos) * len(seeds)
    done = 0
    summary = {}

    for algo in algos:
        rows, best_model, best_sharpe = [], None, -np.inf

        for seed in seeds:
            progress(base + span * done / total, "Training {} with seed {}".format(algo, seed))
            model = train_candidate(algo, fit_df, cfg, seed)
            out = run_policy(model, val_df, cfg)
            metrics = summarize(out["equity"], out["trades"])
            rows.append(metrics)

            if metrics["sharpe_ratio"] > best_sharpe:
                best_sharpe, best_model = metrics["sharpe_ratio"], model
            done += 1

        sharpes = np.array([r["sharpe_ratio"] for r in rows])
        mean_s, std_s = float(sharpes.mean()), float(sharpes.std())

        summary[algo + "|" + tag] = {
            "algo": algo,
            "tag": tag,
            "mean_sharpe": mean_s,
            "std_sharpe": std_s,
            "score": mean_s - 0.5 * std_s,
            "mean_return_pct": float(np.mean([r["total_return_pct"] for r in rows])),
            "trades_mean": float(np.mean([r["n_trades"] for r in rows])),
            "cfg": cfg,
            "best_model": best_model,
        }

    return summary


# ----------------------------------------------------------------
# Stage 2: diagnose the outputs and decide whether to switch
# ----------------------------------------------------------------
def diagnose(summary, bench_sharpe, n_val_days, cfg):
    best = max(summary.values(), key=lambda row: row["score"])
    notes = []
    switch = None

    if best["mean_sharpe"] < bench_sharpe:
        notes.append(
            "On validation data, the best candidate's Sharpe ratio ({:.2f}) is below "
            "buy and hold ({:.2f}). The agent has not yet shown skill.".format(
                best["mean_sharpe"], bench_sharpe
            )
        )

    if best["std_sharpe"] > max(abs(best["mean_sharpe"]), 0.5):
        notes.append("Results change a lot between random seeds, so learning is unstable.")
        switch = {
            "algos": ["ppo"],
            "cfg": {"learning_rate": cfg["learning_rate"] / 3.0},
            "tag": "stable",
            "reason": "unstable learning, so PPO is tried with a smaller learning rate",
        }

    elif best["trades_mean"] / max(n_val_days, 1) > 0.2:
        notes.append(
            "The agent trades more than once every five days, so fees and noise dominate."
        )
        switch = {
            "algos": [best["algo"]],
            "cfg": {"reward_mode": "risk_penalty", "drawdown_penalty": 2.0},
            "tag": "risk",
            "reason": "overtrading, so a drawdown penalty is added to the reward",
        }

    else:
        notes.append("No instability or overtrading was detected, so no retry was needed.")

    return notes, switch


def apply_switch(cfg, switch):
    new_cfg = dict(cfg)
    new_cfg.update(switch["cfg"])
    return new_cfg