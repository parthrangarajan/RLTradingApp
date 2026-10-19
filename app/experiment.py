import json
import os

import numpy as np
import pandas as pd


from .features import add_features
from .metrics import strategy_equity, summarize
from .paths import MODELS_DIR, RESULTS_DIR, ensure_dirs
from .plotting import render_long_short_chart
from .data_profile import profile_data
from .timeline import build_timeline
from .tournament import (
    apply_switch,
    choose_candidates,
    diagnose,
    run_policy,
    search_best,
    split_validation,
    validation_benchmark_sharpe,
)

MA_WINDOW = 20  # moving-average benchmark window


def slice_period(feats, start, end, window):
    """
    Return the rows covering [start, end], plus (window - 1) rows of history
    before start, so the agent's first observation is complete.
    """
    i0 = feats.index.searchsorted(pd.Timestamp(start), side="left")
    i1 = feats.index.searchsorted(pd.Timestamp(end), side="right")

    if i0 < window - 1:
        raise ValueError(
            "Not enough history before {} for window_size={}. "
            "Move the start date later.".format(start, window)
        )
    if i1 - i0 < window + 5:
        raise ValueError(
            "Period {} to {} has only {} trading days. Choose a longer period.".format(
                start, end, i1 - i0
            )
        )
    return feats.iloc[i0 - window + 1 : i1]


def run_experiment(df, cfg, job_id, progress):
    """
    Full pipeline:
      0. Profile the training data and choose candidate algorithms.
      1. Train each candidate with three seeds and score it on validation data.
      2. Diagnose the outputs, and retry once with a switched setting if needed.
      3. Test the chosen candidate once on the test period.
      4. Compute benchmarks, metrics, trade log, timeline, and the chart.
    """
    ensure_dirs()
    np.random.seed(cfg["seed"])
    window = cfg["window_size"]
    fee = cfg["fee_rate"]
    start = cfg["starting_capital"]

    feats = add_features(df)
    train_df = slice_period(feats, cfg["train_start"], cfg["train_end"], window)
    test_df = slice_period(feats, cfg["test_start"], cfg["test_end"], window)

    # ----- 0. Profile the TRAINING data only, and choose candidates -------
    progress(0.02, "Profiling the training data")
    profile = profile_data(train_df)
    algos, reasons = choose_candidates(profile, cfg["allow_short"])

    # ----- 1. Tournament on VALIDATION data (the test period is not touched)
    fit_df, val_df = split_validation(train_df, window)
    n_val_days = len(val_df) - window + 1
    bench_sharpe = validation_benchmark_sharpe(val_df, window, start, fee)

    progress(0.05, "Comparing candidates: " + ", ".join(algos))
    summary = search_best(fit_df, val_df, algos, cfg, progress, 0.05, 0.35, "base")

    # ----- 2. Diagnose, then retry once if the outputs ask for a change ---
    notes, switch = diagnose(summary, bench_sharpe, n_val_days, cfg)
    if switch:
        progress(0.40, "Switching: " + switch["reason"])
        switched_cfg = apply_switch(cfg, switch)
        summary.update(
            search_best(
                fit_df, val_df, switch["algos"], switched_cfg,
                progress, 0.40, 0.25, switch["tag"],
            )
        )

    # Pull the trained models out of the summary, so the summary can be saved as JSON
    models = {label: row.pop("best_model") for label, row in summary.items()}
    best_label = max(summary, key=lambda label: summary[label]["score"])
    best = summary[best_label]
    model = models[best_label]
    best_cfg = best["cfg"]

    # ----- 3. Test ONCE, with the chosen candidate's settings --------------
    progress(0.70, "Running the chosen agent on the test period")
    test_out = run_policy(model, test_df, best_cfg)
    rl_equity = test_out["equity"]
    shares = test_out["shares"]
    test_trades = test_out["trades"]

    eval_dates = test_df.index[window - 1:]
    eval_prices = test_df["Close"].values[window - 1:]

    # ----- 4. Benchmarks ----------------------------------------------------
    progress(0.80, "Computing benchmarks")
    n = len(eval_prices)

    bh_positions = np.ones(n)
    bh_equity, bh_trades = strategy_equity(eval_prices, bh_positions, start, fee)

    ma_full = test_df["Close"].rolling(MA_WINDOW).mean().values[window - 1:]
    ma_positions = np.where(np.isnan(ma_full), 0.0, (eval_prices > ma_full).astype(float))
    ma_equity, ma_trades = strategy_equity(eval_prices, ma_positions, start, fee)

    rng = np.random.RandomState(cfg["seed"])
    choices = [-1.0, 0.0, 1.0] if cfg["allow_short"] else [0.0, 1.0]
    random_positions = rng.choice(choices, size=n)
    random_equity, random_trades = strategy_equity(eval_prices, random_positions, start, fee)

    metrics = {
        "rl_agent": summarize(rl_equity, test_trades),
        "buy_and_hold": summarize(bh_equity, bh_trades),
        "moving_average_20": summarize(ma_equity, ma_trades),
        "random": summarize(random_equity, random_trades),
    }

    # ----- 5. Trade log ----------------------------------------------------
    trades = []
    trade_points = []
    for j in range(1, len(shares)):
        if shares[j] != shares[j - 1]:
            i = j - 1
            before, after = int(shares[j - 1]), int(shares[j])
            trade_points.append((i, after - before))
            trades.append({
                "date": eval_dates[i].strftime("%Y-%m-%d"),
                "price": round(float(eval_prices[i]), 4),
                "shares_before": before,
                "shares_after": after,
                "trade_size": after - before,
                "position_after": "LONG" if after > 0 else ("SHORT" if after < 0 else "FLAT"),
            })

    # ----- 6. Day-by-day timeline for the interactive chart ----------------
    timeline = build_timeline(
        dates=eval_dates,
        prices=eval_prices,
        shares=shares,
        starting_capital=start,
        fee_rate=fee,
        benchmarks={
            "buy_and_hold": bh_equity,
            "moving_average_20": ma_equity,
            "random": random_equity,
        },
    )

    # ----- 7. Research summary: what the tournament found ------------------
    tournament = []
    for label, row in summary.items():
        tournament.append({
            "label": label,
            "algo": row["algo"],
            "tag": row["tag"],
            "mean_sharpe": row["mean_sharpe"],
            "std_sharpe": row["std_sharpe"],
            "score": row["score"],
            "mean_return_pct": row["mean_return_pct"],
            "trades_mean": row["trades_mean"],
        })
    tournament.sort(key=lambda row: -row["score"])

    research = {
        "profile": profile,
        "candidate_reasons": reasons,
        "diagnosis": notes,
        "validation_benchmark_sharpe": bench_sharpe,
        "chosen": best_label,
        "chosen_settings": {
            "algo": best["algo"],
            "reward_mode": best_cfg.get("reward_mode", "pnl"),
            "learning_rate": best_cfg["learning_rate"],
        },
        "validated_skill": bool(best["mean_sharpe"] > bench_sharpe),
        "tournament": tournament,
    }

    # ----- 8. Chart --------------------------------------------------------
    progress(0.92, "Rendering chart")
    held = np.append(shares[1:], shares[-1])
    title = "{} | RL agent vs buy and hold ({} to {})".format(
        cfg["file_id"][:8],
        eval_dates[0].strftime("%Y-%m-%d"),
        eval_dates[-1].strftime("%Y-%m-%d"),
    )
    png = render_long_short_chart(
        dates=eval_dates,
        prices=eval_prices,
        held=held,
        trade_points=trade_points,
        rl_equity=rl_equity,
        bh_equity=bh_equity,
        title=title,
    )

    # ----- 9. Save everything ----------------------------------------------
    out_dir = os.path.join(RESULTS_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)
    chart_path = os.path.join(out_dir, "chart.png")
    with open(chart_path, "wb") as fh:
        fh.write(png)

    result = {
        "job_id": job_id,
        "config": cfg,
        "test_period": {
            "start": eval_dates[0].strftime("%Y-%m-%d"),
            "end": eval_dates[-1].strftime("%Y-%m-%d"),
            "trading_days": int(n),
        },
        "metrics": metrics,
        "trades": trades,
        "timeline": timeline,
        "research": research,
        "chart_path": chart_path,
    }

    with open(os.path.join(out_dir, "result.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    progress(1.0, "Finished")
    return result