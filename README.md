# Reinforcement Learning Trading Simulator

A web application that trains a reinforcement learning agent to trade a single stock, then tests it on a period it never saw during training. The agent's results are compared against simple benchmark strategies, so you can judge whether the agent adds any real value.

The project is built with **FastAPI** (the backend), **Stable-Baselines 2.10** on **TensorFlow 1.15** (the learning algorithms), and plain **HTML, CSS, and JavaScript** with **Plotly** (the interactive frontend).

---

## Important disclaimer

This is an **educational project**. It is not investment advice, and it does not place real trades. Simulated past performance does not predict future results. Market data is taken from third-party sources and is used here only for learning and research.

---

## Table of contents

1. [Key features](#key-features)
2. [How the system works](#how-the-system-works)
3. [The reinforcement learning design](#the-reinforcement-learning-design)
4. [Metrics explained](#metrics-explained)
5. [Project structure](#project-structure)
6. [Getting started](#getting-started)
7. [Using the web application](#using-the-web-application)
8. [Configuration reference](#configuration-reference)
9. [API reference](#api-reference)
10. [Output files](#output-files)
11. [Data sources](#data-sources)
12. [Running with Docker](#running-with-docker)
13. [Limitations](#limitations)
14. [Troubleshooting](#troubleshooting)
15. [Roadmap](#roadmap)
16. [Author and links](#author-and-links)

---

## Key features

- **Two ways to load data:** fetch daily prices for any ticker (for example GME, AAPL, MSFT, TSLA), or upload a CSV file you downloaded yourself.
- **Automatic algorithm selection.** The system profiles the training data, chooses candidate algorithms, compares them on held-out validation data, and retries once with an adjusted setting if the results show a specific problem.
- **Three learning algorithms:** A2C (Advantage Actor-Critic), PPO (Proximal Policy Optimization), and DQN (Deep Q-Network).
- **Position sizing.** Each trade has a size limit based on a share of account value and a maximum number of shares.
- **Trading costs.** Every trade pays a fee, which discourages excessive trading.
- **Honest evaluation.** Models are chosen on validation data. The test period is used once, at the end.
- **Benchmarks on every run:** buy and hold, a 20-day moving-average rule, and a random policy.
- **Interactive results.** Hover for details, drag the day slider, click trades to see their explanations, and read a glossary of every term.
- **Downloadable outputs:** a chart image, and a full JSON report of every setting, metric, and trade.

---

## How the system works

```
 ┌────────────┐     ┌──────────────────┐     ┌──────────────────────────┐
 │ 1. Data    │ ──▶ │ 2. Features      │ ──▶ │ 3. Split by date         │
 │ Yahoo or   │     │ returns, moving  │     │ training │ test          │
 │ CSV upload │     │ averages, volume │     │ (training = fit + valid.)│
 └────────────┘     └──────────────────┘     └────────────┬─────────────┘
                                                         │
 ┌────────────────────────────────────────────────────────▼─────────────┐
 │ 4. Profile the training data  →  choose candidate algorithms          │
 ├───────────────────────────────────────────────────────────────────────┤
 │ 5. Tournament on validation data (3 seeds per algorithm)              │
 ├───────────────────────────────────────────────────────────────────────┤
 │ 6. Diagnose the outputs  →  optional single retry with a switch       │
 ├───────────────────────────────────────────────────────────────────────┤
 │ 7. Test the chosen agent once, on the unseen test period              │
 ├───────────────────────────────────────────────────────────────────────┤
 │ 8. Compare with benchmarks, build the chart, save the report          │
 └───────────────────────────────────────────────────────────────────────┘
```

Training runs in a background job. The web page polls the server every two seconds and shows progress.

---

## The reinforcement learning design

### The environment

The agent interacts with a simulated trading account at each trading day.

**Actions.** The agent chooses one of three target positions each day:

| Action | Meaning |
|---|---|
| `0` (SHORT) | Owe borrowed shares. Becomes FLAT if short selling is switched off. |
| `1` (FLAT) | Hold no shares. All money stays in cash. |
| `2` (LONG) | Own shares. |

**Position size.** When the agent goes long or short, the number of shares is:

$$\text{units} = \min\left(\text{max\_shares},\ \left\lfloor \frac{\text{max\_fraction} \times E_t}{P_t} \right\rfloor\right)$$

- $\text{units}$ is the number of shares in the position.
- $\min(\cdot,\cdot)$ takes the smaller of the two values.
- $\text{max\_shares}$ is the hard cap on shares per trade.
- $\lfloor \cdot \rfloor$ rounds down to a whole number, because you cannot buy part of a share.
- $\text{max\_fraction}$ is the largest share of account value one position may use, for example 0.25.
- $E_t$ is the account value on day $t$.
- $P_t$ is the closing price on day $t$.

**Trading costs.** Each trade pays a fee:

$$\text{fee} = c \times |V|$$

- $c$ is the fee rate, for example 0.001 (0.1 percent).
- $|V|$ is the value of the trade, the number of shares traded times the price. The vertical bars mean "absolute value", so the sign is ignored.

**What the agent sees (the observation).** The last `window_size` days of features, flattened into one list, plus the current position as a fraction of the share cap. The features are:

- `ret1`: today's return
- `ret5`: the five-day return
- `ma_ratio`: how far today's price sits above or below its 10-day average
- `vol20`: 20-day volatility (how much returns jump around)
- `log_volume_z`: trading volume, standardized over 20 days

All features use only past data, so the agent never sees the future.

### Reward functions

The reward is the signal the agent learns from. The project supports two reward modes.

**Profit reward (`"pnl"`):**

$$r_t = 100 \times \frac{E_{t+1} - E_t}{C_0}$$

- $r_t$ is the reward for day $t$, in percent of starting capital.
- $E_{t+1}$ is the account value on the next day.
- $E_t$ is the account value today.
- $C_0$ is the starting capital.

**Risk-penalty reward (`"risk_penalty"`):**

$$r_t = \text{pnl}_t - \lambda \times \max\left(0,\ DD_t - DD_{t-1}\right)$$

- $\text{pnl}_t$ is the profit reward above.
- $\lambda$ (lambda) is the drawdown penalty, which controls how strongly losses below a previous high are punished.
- $DD_t$ is the current drawdown in percent: how far the account sits below its highest value so far.
- $\max(0, \cdot)$ takes zero if the drawdown is shrinking. The penalty applies only when losses grow.

### Candidate algorithms

| Algorithm | Type | Actions | Strength | Weakness |
|---|---|---|---|---|
| **A2C** (Advantage Actor-Critic) | On-policy, actor and critic | Discrete | Fast and simple | Can be unstable |
| **PPO** (Proximal Policy Optimization) | On-policy, clipped updates | Discrete | Steadier learning | More computation |
| **DQN** (Deep Q-Network) | Off-policy, replay buffer | Discrete | Efficient with small data | Discrete actions only |

An **actor-critic** method has two parts: the *actor* chooses actions, and the *critic* estimates how good each situation is. The *clipped updates* in PPO limit how far the strategy can change in one step, which prevents sudden jumps.

### The selection logic

The system chooses algorithms using the data and the results, not by guesswork.

**Stage 1: profile the training data.** The system measures annual volatility, excess kurtosis (how heavy the extreme moves are), the length of the training period, and more. It then chooses candidates:

- Fewer than about 250 training days → DQN only, because deep models memorize small datasets.
- Extreme jumps (excess kurtosis above 5) or annual volatility above 80 percent → PPO and A2C.
- Long-or-cash only (short selling off) → DQN and PPO.
- Otherwise → PPO, A2C, and DQN.

**Stage 2: tournament on validation data.** The training period is split in time order: the first 70 percent is used to learn, and the last 30 percent is used to score each candidate. Each algorithm is trained with three random seeds and scored:

$$\text{score} = \bar{S} - 0.5 \times \sigma_S$$

- $\text{score}$ is the ranking value for an algorithm.
- $\bar{S}$ (S-bar) is the average validation Sharpe ratio across the three seeds.
- $\sigma_S$ (sigma) is the standard deviation of the Sharpe ratio across the seeds.

The penalty term $0.5 \times \sigma_S$ favors algorithms whose results are consistent. An algorithm that wins only with one lucky seed is ranked lower.

**Stage 3: diagnosis and one retry.** The system reads the outputs and decides whether to change something:

- **Results vary a lot across seeds** → retry with PPO and a learning rate one third as large, for more stable learning.
- **The agent trades more than once every five days** → retry with the risk-penalty reward, to discourage churning.
- **No candidate beats buy and hold on validation data** → the report says so, and the result is marked as showing no validated skill.

**Stage 4: final test.** The chosen agent runs once on the test period, which it never saw during learning.

---

## Metrics explained

Every strategy in the comparison table is measured with the same four metrics.

**Total return.** How much the account grew or shrank as a percentage of starting capital:

$$R = \frac{E_{\text{end}} - E_{\text{start}}}{E_{\text{start}}} \times 100$$

**Sharpe ratio.** Return per unit of risk, annualized:

$$S = \sqrt{252} \times \frac{\bar{r}}{\sigma_r}$$

- $S$ is the Sharpe ratio. Above 1 is usually considered good. Below 0 means the strategy lost money on average.
- $\sqrt{252}$ converts daily figures to yearly figures, since a year has about 252 trading days.
- $\bar{r}$ is the average daily return.
- $\sigma_r$ is the standard deviation of daily returns, which measures how much they vary.
- This project assumes a risk-free rate of zero.

**Maximum drawdown.** The largest fall from a previous high before recovery:

$$\text{MDD} = \min_t \left( \frac{E_t}{\max_{s \le t} E_s} - 1 \right)$$

- $\text{MDD}$ is the maximum drawdown, shown as a negative percentage. Closer to zero is better.
- $\max_{s \le t} E_s$ is the highest account value reached up to day $t$.

**Number of trades.** How many times the position changed.

---

## Project structure

```
RLTradingApp/
├── app/
│   ├── __init__.py        Makes "app" a Python package
│   ├── main.py            FastAPI routes and the web page server
│   ├── schemas.py         Input validation (Pydantic models)
│   ├── jobs.py            Background job queue, one worker
│   ├── experiment.py      Full pipeline: profile, tournament, test, report
│   ├── tournament.py      Candidate selection, diagnosis, validation split
│   ├── models.py          Builds A2C, PPO, and DQN with settings for this problem
│   ├── env.py             The trading environment (actions, costs, rewards)
│   ├── data_profile.py    Describes the training data (volatility, trend, and so on)
│   ├── data_source.py     Downloads daily prices from Yahoo Finance
│   ├── marketwatch.py     Reads and cleans CSV files (MarketWatch format)
│   ├── features.py        Calculates the features the agent sees
│   ├── metrics.py         Return, Sharpe ratio, drawdown, benchmark strategies
│   ├── timeline.py        Day-by-day account record for the interactive chart
│   ├── plotting.py        Builds the PNG chart
│   └── paths.py           Folder locations
├── frontend/
│   ├── index.html         The web page
│   ├── style.css          Styling
│   ├── app.js             Page behavior and interactive chart
│   ├── config.js          Backend address, used when hosting separately
│   ├── favicon.ico        Browser tab icon
│   └── assets/
│       └── rl-logo.png    Logo
├── data/                  Uploaded and downloaded price files (created automatically, not in Git)
├── models/                Saved trained models (created automatically, not in Git)
├── results/               Charts and JSON reports (created automatically, not in Git)
├── requirements.txt       Pinned Python packages
├── Dockerfile             Container build instructions (optional)
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Getting started

### Prerequisites

- **Windows, macOS, or Linux.** The steps below use Windows PowerShell.
- **Python 3.7.** TensorFlow 1.15, used by Stable-Baselines 2.10, supports only Python 3.7. Newer Python versions will not install it.
- **An internet connection** for downloading prices from Yahoo Finance and for loading the chart library.

Check your Python version:

```powershell
py -0
```

If `-3.7` appears in the list, you can use it. Otherwise, install Python 3.7.9 from python.org.

### Installation

**1. Clone the repository.**

```powershell
git clone https://github.com/<your-username>/rl-trading-simulator.git
cd rl-trading-simulator
```

**2. Create a virtual environment with Python 3.7.** A virtual environment keeps this project's packages separate from everything else on your computer.

```powershell
py -3.7 -m venv rl37
rl37\Scripts\Activate.ps1
python --version
```

The last command must print `Python 3.7.x`. If the activation script is blocked, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**3. Install the packages.** Install `gym` first, using a build workaround that avoids a metadata error in its older packaging:

```powershell
python -m pip install --upgrade pip
pip install "setuptools==57.5.0" "wheel==0.37.1"
pip install --no-build-isolation gym==0.18.0
pip install -r requirements.txt
```

**4. Check the imports.**

```powershell
python -c "import tensorflow, gym, stable_baselines, fastapi; print('All imports OK')"
```

The TensorFlow messages about `contrib` are normal and can be ignored.

### Running the application

From the project root, with the virtual environment active:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000/ui/** in your browser. Press **Ctrl+F5** the first time, to clear any old cached files.

Do not use `--reload` while training. Reloading restarts the TensorFlow session in the middle of a run.

The interactive API documentation is at **http://127.0.0.1:8000/docs**.

---

## Using the web application

**Step 1: Load market data.** Choose one option:

- **Option A:** enter a ticker symbol and a date range, then click **Fetch prices**.
- **Option B:** upload a CSV file with the columns `Date, Open, High, Low, Close, Volume`, then click **Upload CSV**.

**Step 2: Configure the run.** Every field has a default value and a plain-English explanation. Hover over any label, or read the grey text beneath it.

- **Dates** define four periods: training start and end, then test start and end. The agent learns during training and is judged on the test period.
- **Money and position size** set the starting capital and the limits on each position.
- **Trading costs and learning** set the fee rate, memory window, training steps, learning rate, random seed, and whether short selling is allowed.

**Step 3: Progress.** The page shows the current stage, for example "Training ppo with seed 43", and a progress bar.

**Step 4: Results.**

- **Verdict:** a sentence summarizing the agent's test return against buy and hold.
- **Interactive chart:** the price line with green shading for long periods, red for short periods, and triangles for trades. The lower panel compares the agent's account value with the benchmarks.
- **Day explorer:** drag the slider or click the chart to see the account on any day, including what closing the position would involve.
- **Strategy comparison:** the four strategies side by side. The best value in each column is shown in bold.
- **Trade log:** every position change, with a plain-English explanation.

**Recent runs.** Runs you start are listed in your browser, so you can reopen them while the server keeps them in memory.

---

## Configuration reference

These are the settings the `/runs` endpoint accepts. The web page sets them for you.

| Setting | Default | Allowed range | What it does |
|---|---|---|---|
| `train_start`, `train_end` | 2020-01-02, 2020-10-30 | Within loaded data | Period the agent learns from |
| `test_start`, `test_end` | 2020-11-02, 2021-03-12 | After training end | Period used for the final, single test |
| `starting_capital` | 10000 | Above 0 | Starting cash in the simulated account |
| `max_fraction_per_trade` | 0.25 | Above 0, up to 1 | Largest share of account value in one position |
| `max_shares_per_trade` | 500 | 1 or more | Hard cap on shares per position |
| `fee_rate` | 0.001 | 0 to 0.05 | Fee as a fraction of the value traded |
| `window_size` | 10 | 2 to 60 | Number of past days the agent observes |
| `training_steps` | 100000 | 1,000 to 2,000,000 | Practice decisions per model |
| `learning_rate` | 0.0007 | Above 0, up to 0.1 | Size of each learning correction |
| `allow_short` | true | true or false | Whether the agent may owe borrowed shares |
| `seed` | 42 | Any whole number | Fixes the randomness, so results are repeatable |
| `file_id` | (required) | 32-character ID | Identifies the uploaded or downloaded dataset |

Internal settings, chosen automatically and not yet exposed in the interface:

| Setting | Default | What it does |
|---|---|---|
| `reward_mode` | `"pnl"` | Set to `"risk_penalty"` by the retry logic when overtrading is detected |
| `drawdown_penalty` | 1.0 | Strength of the drawdown penalty, `λ` |
| `tournament_steps` | same as `training_steps` | Training steps for each tournament model |

---

## API reference

All endpoints are served at `http://127.0.0.1:8000`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Checks that the server is running |
| `POST` | `/data/fetch-ticker` | Downloads daily prices for a ticker |
| `POST` | `/data/upload` | Uploads a CSV file (multipart form, field name `file`) |
| `POST` | `/runs` | Starts a background run. Returns a `job_id` |
| `GET` | `/runs/{job_id}` | Returns progress, and the full result when complete |
| `GET` | `/runs/{job_id}/chart` | Returns the chart as a PNG image |
| `GET` | `/ui/` | The web application |

### Example: fetch data, start a run, and check progress

```powershell
# 1. Fetch GME prices
curl -X POST http://127.0.0.1:8000/data/fetch-ticker `
  -H "Content-Type: application/json" `
  -d '{"ticker":"GME","start":"2019-11-01","end":"2021-03-12"}'

# 2. Start a run, using the file_id returned in step 1
curl -X POST http://127.0.0.1:8000/runs `
  -H "Content-Type: application/json" `
  -d '{"file_id":"PASTE_FILE_ID","train_start":"2020-01-02","train_end":"2020-10-30","test_start":"2020-11-02","test_end":"2021-03-12","starting_capital":10000,"max_fraction_per_trade":0.25,"max_shares_per_trade":500,"fee_rate":0.001,"window_size":10,"training_steps":20000,"learning_rate":0.0007,"allow_short":true,"seed":42}'

# 3. Check progress, using the job_id returned in step 2
curl http://127.0.0.1:8000/runs/PASTE_JOB_ID
```

---

## Output files

For each completed run, the server writes to `results/<job_id>/`:

- **`chart.png`:** the two-panel chart (price with trades, and account value against benchmarks).
- **`result.json`:** the complete report, including:
  - `config`: every setting used for the run
  - `test_period`: the first and last test dates, and the number of trading days
  - `metrics`: total return, Sharpe ratio, maximum drawdown, and trade count for each strategy
  - `trades`: every position change
  - `timeline`: the day-by-day account record used by the interactive chart
  - `research`: the data profile, candidate reasons, diagnosis notes, the full tournament table, and `validated_skill`

Check `research.validated_skill` before drawing any conclusion. It is `true` only when the chosen agent beat buy and hold on validation data.

---

## Data sources

**Yahoo Finance (ticker option).** The application calls Yahoo Finance's public chart endpoint for daily prices. This endpoint is not an official, documented API, so it can change or become unavailable without notice. Yahoo's terms apply to the data. Use it for personal and educational purposes only.

**CSV upload (file option).** Any CSV file with the columns `Date, Open, High, Low, Close, Volume` works. This is how you can use data from MarketWatch's download page. MarketWatch does not offer a public API, so the project does not scrape it. Download the CSV manually and upload it.

Dates are read in the order they appear, then sorted oldest first. Duplicate dates are removed, and rows without a valid closing price are discarded.

---

## Running with Docker

A `Dockerfile` is included for running the application in a container. It uses Python 3.7 because TensorFlow 1.15 requires it.

```bash
docker build -t rl-trading .
docker run -p 7860:7860 rl-trading
```

Then open **http://127.0.0.1:7860/ui/**. The container reads a `PORT` environment variable, so you can change the port on hosting platforms.

To allow a separately hosted frontend to call the backend, set the `ALLOWED_ORIGINS` environment variable to a comma-separated list of allowed website addresses.

---

## Limitations

- **One stock at a time.** The agent trades a single ticker. It is not yet trained across several stocks.
- **Simplified trading.** Short selling is modeled as a cash-only position. Borrowing costs, margin calls, and dividends are not included. Open positions are valued at the final price, and no closing fee is charged at the end of the test period.
- **Small test periods are noisy.** A test period of a few months can produce results driven by luck. Repeat runs with several seeds and other date ranges before drawing conclusions.
- **Results are stored in memory.** Run status and results are lost when the server restarts. A database would fix this.
- **Training time.** Training on a normal computer takes minutes, and the tournament trains each candidate three times. Lower `training_steps` for quick experiments.
- **Single worker.** Only one training job runs at a time, because TensorFlow 1.x does not handle parallel training well inside one process.
- **Dependence on old libraries.** TensorFlow 1.15, Stable-Baselines 2.10, and Python 3.7 are old and no longer maintained. A future version could move to Stable-Baselines3 with Gymnasium and Python 3.10 or newer.

---

## Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `No module named 'tensorflow.contrib'` | You are not using Python 3.7 with TensorFlow 1.15. Check `python --version` inside the environment. |
| `ImportError: cannot import name 'TypedDict'` | Python 3.7 cannot use a newer library. Reinstall the pinned versions from `requirements.txt`. |
| `gym` fails to build with an opencv-related error | Use the `--no-build-isolation` command in Installation, step 3. |
| "Could not reach Yahoo Finance" | Network blocking, or Yahoo rate limiting. Wait a few minutes, or use CSV upload. |
| "No data found for ..." | Check the ticker spelling and the date range. Some international stocks need a suffix, for example `SHOP.TO`. |
| The page appears without styling | Open `/ui/style.css` directly. If it returns "Not Found", check that the file sits in `frontend/`. |
| The chart area says Plotly could not load | The chart library loads from the internet. Check your connection and refresh. |
| "Unknown file_id" when starting a run | The server restarted, or the dataset was never loaded. Load the data again. |

---

## Roadmap

- Train across several tickers, and test on a stock the agent never saw.
- Replace Stable-Baselines 2.10 with Stable-Baselines3, and add continuous position sizing using SAC or TD3.
- Add observation normalization, saved with each model.
- Store runs and results in a database such as SQLite.
- Add automated tests for the environment, metrics, and API endpoints.

---

## Author and links

**Parth Rangarajan**
Master of Financial Insurance, Statistical Sciences, University of Toronto

- Website: https://parthrangarajan.github.io/
- LinkedIn: https://www.linkedin.com/in/parthrangarajan/

Built as a personal learning project in reinforcement learning and quantitative finance.

---

## License

No license has been chosen yet. Until one is added, all rights are reserved by the author. To make the project open for reuse, add a `LICENSE` file. The MIT License is a common, permissive choice for projects like this.