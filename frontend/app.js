// ============================================================
// Setup
// ============================================================
// Safety net: a form must never reload the page. Our own handlers still run.
document.addEventListener("submit", (e) => e.preventDefault());

// Opened from disk, talk to the local server. Served by FastAPI, use the same origin.
const API_BASE =
  window.API_BASE_OVERRIDE ||
  (window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "");
const POLL_MS = 2000;
const HISTORY_KEY = "rl_trading_history_v2";
const PLOTLY_OK = typeof window.Plotly !== "undefined";

// Values the form starts with. Dates are the GameStop example period.
const DEFAULTS = {
  train_start: "2020-01-02",
  train_end: "2020-10-30",
  test_start: "2020-11-02",
  test_end: "2021-03-12",
  starting_capital: 10000,
  max_fraction_per_trade: 0.25,
  max_shares_per_trade: 500,
  fee_rate: 0.001,
  window_size: 10,
  training_steps: 100000,
  learning_rate: 0.0007,
  seed: 42,
  allow_short: true,
};

const TICKER_DEFAULTS = { ticker: "GME", start: "2019-11-01", end: "2021-03-12" };

// Every explanation in the app lives here. Tooltips, field help and the glossary all read from this.
const GLOSSARY = {
  // ----- Step 1: data -----
  ticker: {
    label: "Ticker symbol",
    text: "A short code that identifies a company's stock on an exchange. For example, GME is GameStop, AAPL is Apple, MSFT is Microsoft, and TSLA is Tesla.",
  },
  start_date: {
    label: "Data start date",
    text: "The first day of price history to download. Earlier dates give the agent more history, but they also make the download longer.",
  },
  end_date: {
    label: "Data end date",
    text: "The last day of price history to download. It must come after the start date.",
  },

  // ----- Step 2: period settings -----
  train_start: {
    label: "Training start",
    text: "The first day the agent may learn from. Pick a date a few weeks after the start of your data, so the indicators have enough past prices to be calculated.",
  },
  train_end: {
    label: "Training end",
    text: "The last day of the learning period. The agent practices on prices between training start and training end, the way a student practices with old exam papers.",
  },
  test_start: {
    label: "Test start",
    text: "The first day of the test period. The agent does not learn from these prices, so this is an honest exam. Keep it after the training end.",
  },
  test_end: {
    label: "Test end",
    text: "The last day of the test period. All results, the chart, and the trade log cover the days from test start to test end.",
  },

  // ----- Step 2: money and position size -----
  starting_capital: {
    label: "Starting capital",
    text: "The amount of money the simulated account begins with, in your own currency. Returns are also shown as percentages, so the exact amount does not change the percentage results.",
  },
  max_fraction_per_trade: {
    label: "Largest share of account per position",
    text: "The biggest portion of the total account value that one position may use. A value of 0.25 means no more than one quarter of the account goes into a single position. This controls how large each trade can be.",
  },
  max_shares_per_trade: {
    label: "Maximum shares per trade",
    text: "A hard upper limit on the number of shares in one position, whatever the account size. Think of it as a safety cap that stops very large trades.",
  },

  // ----- Step 2: costs and learning -----
  fee_rate: {
    label: "Trading fee rate",
    text: "The cost of each trade, as a fraction of the money that moves. 0.001 means 0.1 percent. Trading 1,000 dollars of shares costs 1 dollar in fees. Brokers charge commissions, so higher fees make frequent trading less profitable.",
  },
  window_size: {
    label: "Memory window (days)",
    text: "How many recent trading days the agent looks at before it decides. A window of 10 means the agent sees the last 10 days of price information each time it chooses an action.",
  },
  training_steps: {
    label: "Training steps",
    text: "How many practice decisions the agent makes while it learns. More steps usually means better learning, but also a longer wait. A step is one day of practice trading.",
  },
  learning_rate: {
    label: "Learning rate",
    text: "How big a correction the agent makes after each lesson. If it is too large, learning becomes unstable and jumps around. If it is too small, learning is very slow. The default of 0.0007 is a common starting point.",
  },
  seed: {
    label: "Random seed",
    text: "A number that fixes the random starting point, so the same settings always produce the same result. Change it and run again to check whether a good result was luck.",
  },
  allow_short: {
    label: "Allow short positions",
    text: "A short position means you borrow shares, sell them now, and hope to buy them back more cheaply later. If this box is unticked, the agent can only own shares or hold cash.",
  },

  // ----- Step 4: results -----
  price: {
    label: "Price",
    text: "The closing price of the stock on that day, in the currency of the exchange where it trades.",
  },
  position: {
    label: "Position",
    text: "Your holding on that day. LONG means you own shares. SHORT means you owe borrowed shares. FLAT means you hold only cash.",
  },
  shares: {
    label: "Shares held",
    text: "How many shares you hold. A positive number means you own them. A negative number means you owe borrowed shares.",
  },
  entry_price: {
    label: "Average entry price",
    text: "The average price you paid for the shares you hold right now. For a short position, it is the average price you received when you sold the borrowed shares. This is also called the cost basis.",
  },
  position_value: {
    label: "Position value",
    text: "What the current holding is worth today: the number of shares multiplied by today's price. For a short position, this value is negative, because you owe the shares.",
  },
  unrealized: {
    label: "Open profit or loss",
    text: "The gain or loss you would make if you closed the position at today's price. It is not final until you actually close the position, so it can still change.",
  },
  realized: {
    label: "Locked-in profit or loss",
    text: "Gains or losses already fixed by trades you have closed. This money is final and will not change with the market.",
  },
  cash: {
    label: "Cash",
    text: "Money not invested in shares. In this simulation, cash earns no interest.",
  },
  equity: {
    label: "Account value",
    text: "Cash plus the current value of the shares you hold. For a short position, the value of the borrowed shares is subtracted. This is the line the chart follows.",
  },
  total_gain: {
    label: "Total gain or loss",
    text: "Today's account value minus the starting capital. The percentage shows the same change relative to the starting capital.",
  },
  fee: {
    label: "Fee",
    text: "The commission paid for a trade. It equals the fee rate multiplied by the value traded.",
  },
  trade_value: {
    label: "Value traded",
    text: "The number of shares traded multiplied by the price. This is the amount of money that moved in the trade, before the fee.",
  },
  final_equity: {
    label: "Final account value",
    text: "The account value on the last day of the test period.",
  },
  total_return: {
    label: "Total return",
    text: "How much the account grew or shrank, as a percentage of the starting capital. A return of 10 percent turns 10,000 into 11,000.",
  },
  sharpe: {
    label: "Sharpe ratio",
    text: "Gain per unit of risk. It is the average daily return divided by how much the daily returns jump around (their standard deviation), then multiplied by the square root of 252, the number of trading days in a year. Higher is better. Above 1 is usually considered good. Below 0 means the strategy lost money on average.",
  },
  max_drawdown: {
    label: "Maximum drawdown",
    text: "The largest fall from a previous high in account value before it recovered. A drawdown of minus 20 percent means that at one point the account lost a fifth of its peak value. Closer to zero is better.",
  },
  trades: {
    label: "Number of trades",
    text: "How many times the position changed. This includes opening, adding to, reducing, closing, and switching between long and short. More trades usually means more fees.",
  },
  rl_agent: {
    label: "Reinforcement learning agent",
    text: "The agent that learned by trial and error during the training period and was then tested on the test period, which it had never seen.",
  },
  buy_and_hold: {
    label: "Buy and hold",
    text: "Buy on the first day and keep the shares until the end. This is the simplest benchmark. The agent is only useful if it beats it.",
  },
  ma20: {
    label: "20-day moving average rule",
    text: "Own shares when today's price is above its average over the last 20 days, and hold cash otherwise. A popular rule of thumb among traders.",
  },
  random: {
    label: "Random policy",
    text: "Picks long, short, or flat at random each day. It shows what pure chance can achieve, so a useful agent should clearly beat it.",
  },
};

const STRATEGIES = [
  { key: "rl_agent", gloss: "rl_agent" },
  { key: "buy_and_hold", gloss: "buy_and_hold" },
  { key: "moving_average_20", gloss: "ma20" },
  { key: "random", gloss: "random" },
];

const METRIC_COLUMNS = [
  { key: "final_equity", gloss: "final_equity", fmt: (v) => money(v), best: null },
  { key: "total_return_pct", gloss: "total_return", fmt: (v) => fmtPct(v), best: "max" },
  { key: "sharpe_ratio", gloss: "sharpe", fmt: (v) => Number(v).toFixed(2), best: "max" },
  { key: "max_drawdown_pct", gloss: "max_drawdown", fmt: (v) => fmtPct(v), best: "max" },
  { key: "n_trades", gloss: "trades", fmt: (v) => String(v), best: null },
];

const state = {
  dataset: null,
  jobId: null,
  pollTimer: null,
  lastResult: null,
  explorerDays: [],
  baseShapes: [],
};

// ============================================================
// Helpers
// ============================================================
const $ = (id) => document.getElementById(id);

// Attach an event handler. If the element is missing, log it instead of stopping the whole script.
function on(id, type, handler) {
  const el = $(id);
  if (!el) {
    console.error("Missing page element #" + id + ". Check that index.html and app.js are the same version.");
    return;
  }
  el.addEventListener(type, handler);
}

const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
  );

const money = (v) =>
  "$" + Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

const fmtSigned = (v) =>
  (Number(v) >= 0 ? "+" : "-") +
  "$" + Math.abs(Number(v)).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

const fmtNum = (v, digits = 2) =>
  Number(v).toLocaleString(undefined, { maximumFractionDigits: digits });

const fmtPct = (v) => Number(v).toFixed(2) + "%";

const show = (el) => el.classList.remove("hidden");
const hide = (el) => el.classList.add("hidden");

// A term with a hover explanation. Pass a label to override the glossary label.
function tip(key, label) {
  const entry = GLOSSARY[key] || { label: key, text: "" };
  const shown = label !== undefined ? label : entry.label;
  return `<span class="term" data-tip="${esc(entry.text)}">${esc(shown)} <b class="info">ⓘ</b></span>`;
}

function notify(message, kind = "info") {
  const el = $("global-status");
  el.textContent = message;
  el.className = "status " + kind;
}

function formError(message) {
  const el = $("config-error");
  el.textContent = message;
  el.className = "status error";
}

function clearFormError() {
  hide($("config-error"));
}

function setBusy(button, busy) {
  if (!button) return;
  button.disabled = busy;
  if (busy) {
    button.dataset.label = button.textContent;
    button.textContent = "Working...";
  } else if (button.dataset.label) {
    button.textContent = button.dataset.label;
  }
}

// Floating tooltip box for elements with a data-tip attribute.
const tipBox = document.createElement("div");
tipBox.id = "tip-box";
tipBox.className = "tip-box hidden";
document.body.appendChild(tipBox);

document.addEventListener("mouseover", (e) => {
  const el = e.target.closest && e.target.closest("[data-tip]");
  if (!el) return;
  tipBox.textContent = el.dataset.tip;
  tipBox.classList.remove("hidden");
});

document.addEventListener("mousemove", (e) => {
  if (tipBox.classList.contains("hidden")) return;
  const left = Math.min(e.clientX + 16, window.innerWidth - 340);
  tipBox.style.left = Math.max(8, left) + "px";
  tipBox.style.top = e.clientY + 18 + "px";
});

document.addEventListener("mouseout", (e) => {
  if (e.target.closest && e.target.closest("[data-tip]")) {
    tipBox.classList.add("hidden");
  }
});

// ============================================================
// API wrapper
// ============================================================
async function api(path, options = {}) {
  let res;
  try {
    res = await fetch(API_BASE + path, options);
  } catch (err) {
    throw new Error("Cannot reach the server. Is uvicorn running on port 8000?");
  }

  const text = await res.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch (err) {
    body = text;
  }

  if (!res.ok) {
    let message = res.statusText || "Request failed";
    if (body && body.detail) {
      if (typeof body.detail === "string") {
        message = body.detail;
      } else if (Array.isArray(body.detail)) {
        message = body.detail
          .map((d) => (d.loc ? d.loc.slice(1).join(" ") + ": " : "") + d.msg)
          .join("; ");
      }
    }
    throw new Error(message);
  }
  return body;
}

async function checkApi() {
  const el = $("api-indicator");
  try {
    await api("/health");
    el.textContent = "Server status: online";
    el.className = "api-indicator online";
  } catch (err) {
    el.textContent = "Server status: offline. Start the server with uvicorn, then refresh this page.";
    el.className = "api-indicator offline";
  }
}

// ============================================================
// Step 1: Data
// ============================================================
function renderDataset(summary) {
  const first = summary.preview[0] || {};
  const headers = Object.keys(first).map((k) => `<th>${esc(k)}</th>`).join("");
  const rows = summary.preview
    .map((r) => {
      const cells = Object.keys(r)
        .map((k) => `<td>${esc(k === "Date" ? r[k] : fmtNum(r[k], 4))}</td>`)
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");

  $("dataset-summary").innerHTML = `
    <div class="kpis">
      <div class="kpi"><span>Where the data came from</span><strong>${esc(summary.source)}</strong></div>
      <div class="kpi"><span>Trading days</span><strong>${summary.rows}</strong></div>
      <div class="kpi"><span>First date</span><strong>${esc(summary.start)}</strong></div>
      <div class="kpi"><span>Last date</span><strong>${esc(summary.end)}</strong></div>
    </div>
    <p class="muted small">The first five rows of your data:</p>
    <div class="table-wrap">
      <table>
        <thead><tr>${headers}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
  show($("dataset-summary"));
}

// Fill the four dates from the loaded data. The first 45 days are skipped so the indicators have history.
function prefillDates(start, end) {
  const day = 86400000;
  const toIso = (d) => d.toISOString().slice(0, 10);

  const s = new Date(start + "T00:00:00Z");
  const e = new Date(end + "T00:00:00Z");

  const warmupEnd = new Date(s.getTime() + 45 * day);
  const span = e.getTime() - warmupEnd.getTime();
  const trainEnd = new Date(warmupEnd.getTime() + span * 0.7);
  const testStart = new Date(trainEnd.getTime() + day);

  const f = $("run-form").elements;
  f.train_start.value = toIso(warmupEnd);
  f.train_end.value = toIso(trainEnd);
  f.test_start.value = toIso(testStart);
  f.test_end.value = end;
}

function setDataset(summary) {
  state.dataset = summary;
  renderDataset(summary);
  prefillDates(summary.start, summary.end);
  notify(
    `Loaded ${summary.rows} trading days of ${summary.source} from ${summary.start} to ${summary.end}. The dates in step 2 were filled in for you.`,
    "ok"
  );
  $("step-config").scrollIntoView({ behavior: "smooth", block: "start" });
}

on("ticker-form", "submit", async (e) => {
  const ticker = $("ticker").value.trim().toUpperCase();
  const start = $("ticker-start").value;
  const end = $("ticker-end").value;

  if (!ticker) {
    notify("Enter a ticker symbol, for example GME.", "error");
    return;
  }
  if (!start || !end || start >= end) {
    notify("The data end date must come after the data start date.", "error");
    return;
  }

  const button = e.target.querySelector("button");
  setBusy(button, true);
  notify("Downloading prices for " + ticker + ". This can take a few seconds...", "info");

  try {
    const summary = await api("/data/fetch-ticker", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker, start, end }),
    });
    setDataset(summary);
  } catch (err) {
    notify(err.message, "error");
  } finally {
    setBusy(button, false);
  }
});

on("upload-form", "submit", async (e) => {
  const file = $("csv-file").files[0];
  if (!file) {
    notify("Choose a CSV file first.", "error");
    return;
  }

  const button = e.target.querySelector("button");
  setBusy(button, true);
  const body = new FormData();
  body.append("file", file);

  try {
    setDataset(await api("/data/upload", { method: "POST", body }));
  } catch (err) {
    notify(err.message, "error");
  } finally {
    setBusy(button, false);
  }
});

// ============================================================
// Step 2: Configuration
// ============================================================
function validateRun(form) {
  const e = form.elements;
  const dates = ["train_start", "train_end", "test_start", "test_end"].map((n) => e[n].value);

  if (dates.some((d) => !d)) return "Fill in all four dates.";
  if (!(dates[0] < dates[1] && dates[1] < dates[2] && dates[2] < dates[3])) {
    return "The dates must be in order: training start, then training end, then test start, then test end.";
  }

  if (state.dataset) {
    if (dates[0] < state.dataset.start || dates[3] > state.dataset.end) {
      return `All dates must fall between ${state.dataset.start} and ${state.dataset.end}, the range of your loaded data.`;
    }
  }

  if (!(parseFloat(e.starting_capital.value) > 0)) return "Starting capital must be a positive number.";

  const frac = parseFloat(e.max_fraction_per_trade.value);
  if (!(frac > 0 && frac <= 1)) return "The largest share of account per position must be above 0 and at most 1.";

  if (!(parseInt(e.max_shares_per_trade.value, 10) >= 1)) return "Maximum shares per trade must be at least 1.";

  const fee = parseFloat(e.fee_rate.value);
  if (!(fee >= 0 && fee <= 0.05)) return "The trading fee rate must be between 0 and 0.05.";

  const win = parseInt(e.window_size.value, 10);
  if (!(win >= 2 && win <= 60)) return "The memory window must be between 2 and 60 days.";

  const steps = parseInt(e.training_steps.value, 10);
  if (!(steps >= 1000 && steps <= 2000000)) return "Training steps must be between 1,000 and 2,000,000.";

  if (!(parseFloat(e.learning_rate.value) > 0)) return "The learning rate must be a positive number.";

  if (isNaN(parseInt(e.seed.value, 10))) return "The random seed must be a whole number.";

  return null;
}

function buildRunBody(form) {
  const e = form.elements;
  return {
    file_id: state.dataset.file_id,
    train_start: e.train_start.value,
    train_end: e.train_end.value,
    test_start: e.test_start.value,
    test_end: e.test_end.value,
    starting_capital: parseFloat(e.starting_capital.value),
    max_fraction_per_trade: parseFloat(e.max_fraction_per_trade.value),
    max_shares_per_trade: parseInt(e.max_shares_per_trade.value, 10),
    fee_rate: parseFloat(e.fee_rate.value),
    window_size: parseInt(e.window_size.value, 10),
    training_steps: parseInt(e.training_steps.value, 10),
    learning_rate: parseFloat(e.learning_rate.value),
    allow_short: e.allow_short.checked,
    seed: parseInt(e.seed.value, 10),
  };
}

// Put every default into its field. Dates follow the loaded data when there is some.
function fillDefaults() {
  const e = $("run-form").elements;
  Object.entries(DEFAULTS).forEach(([key, value]) => {
    const field = e[key];
    if (!field) return;
    if (field.type === "checkbox") {
      field.checked = value;
    } else {
      field.value = value;
    }
  });
  if (state.dataset) {
    prefillDates(state.dataset.start, state.dataset.end);
  }
}

// Add the plain-English title and help text to every field, from the glossary.
function decorateFields() {
  document.querySelectorAll("[data-field]").forEach((el) => {
    const key = el.dataset.field;
    const entry = GLOSSARY[key];
    if (!entry) return;

    const title = el.querySelector(".field-title");
    if (title) title.innerHTML = tip(key);

    const help = el.querySelector(".field-help");
    if (help) help.textContent = entry.text;
  });
}

on("reset-defaults", "click", () => {
  clearFormError();
  fillDefaults();
  notify("Every field was reset to its default value.", "info");
});

on("run-form", "submit", async (e) => {
  const form = e.target;
  clearFormError();

  if (!state.dataset) {
    formError("Step 1 comes first. Fetch a ticker or upload a CSV file so the agent has prices to work with.");
    $("step-data").scrollIntoView({ behavior: "smooth", block: "start" });
    return;
  }

  const problem = validateRun(form);
  if (problem) {
    formError(problem);
    return;
  }

  const button = form.querySelector('button[type="submit"]');
  setBusy(button, true);

  try {
    const job = await api("/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildRunBody(form)),
    });

    track(job.job_id, {
      job_id: job.job_id,
      label: state.dataset.source,
      created: new Date().toISOString(),
      status: "running",
    });
  } catch (err) {
    formError(err.message);
  } finally {
    setBusy(button, false);
  }
});

// ============================================================
// Step 3: Progress
// ============================================================
function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

function renderProgress(job) {
  const pct = Math.round((job.progress || 0) * 100);
  $("progress-bar").style.width = pct + "%";
  $("progress-pct").textContent = pct + "%";
  $("progress-text").textContent = job.message || "";

  const badge = $("progress-status");
  badge.textContent = job.status;
  badge.className = "badge " + job.status;
}

function renderFailure(job) {
  $("progress-text").textContent = job.error || "The run failed.";
  $("error-trace").textContent = job.traceback || job.error || "No details available.";
  show($("error-details"));
}

function track(jobId, meta) {
  stopPolling();
  state.jobId = jobId;

  if (meta) addHistory(meta);

  hide($("step-results"));
  hide($("error-details"));
  show($("step-progress"));
  $("step-progress").scrollIntoView({ behavior: "smooth", block: "start" });

  renderProgress({ status: "queued", progress: 0, message: "Waiting to start" });
  poll();
  state.pollTimer = setInterval(poll, POLL_MS);
}

async function poll() {
  const jobId = state.jobId;
  if (!jobId) return;

  let job;
  try {
    job = await api("/runs/" + jobId);
  } catch (err) {
    stopPolling();
    notify(
      "This run is no longer on the server. The server may have restarted, which clears saved results. Start a new run.",
      "error"
    );
    markHistory(jobId, "failed");
    return;
  }

  if (jobId !== state.jobId) return; // the user opened another run meanwhile

  renderProgress(job);

  if (job.status === "completed") {
    stopPolling();
    renderResults(job);
    markHistory(jobId, "completed");
    notify("Run completed. Scroll down to see the results.", "ok");
  } else if (job.status === "failed") {
    stopPolling();
    renderFailure(job);
    markHistory(jobId, "failed");
    notify("The run failed. Open the error details in step 3.", "error");
  }
}

// ============================================================
// Step 4: Results
// ============================================================
function renderResults(job) {
  const r = job.result;
  state.lastResult = r;

  if (!r.timeline) {
    notify("This run was created before the interactive chart was added. Start a new run to see the chart.", "error");
    return;
  }

  const m = r.metrics;
  const diff = m.rl_agent.total_return_pct - m.buy_and_hold.total_return_pct;
  const verb = diff >= 0 ? "outperformed" : "underperformed";

  $("verdict").innerHTML =
    `Over ${r.test_period.trading_days} trading days (${esc(r.test_period.start)} to ${esc(r.test_period.end)}), ` +
    `the reinforcement learning agent returned <strong>${fmtPct(m.rl_agent.total_return_pct)}</strong>. ` +
    `Buying and holding returned <strong>${fmtPct(m.buy_and_hold.total_return_pct)}</strong>, ` +
    `so the agent ${verb} by <strong>${Math.abs(diff).toFixed(2)} percentage points</strong>. ` +
    `One test period and one seed are not proof of skill. Repeat with other seeds and dates before drawing conclusions.`;

  $("download-chart").href = API_BASE + "/runs/" + job.job_id + "/chart";

  show($("step-results"));

  buildChart(r);
  renderExplorer(r);
  renderMetrics(r);
  renderTrades(r);

  $("step-results").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------- Interactive chart ----------
function positionWords(d) {
  if (d.shares > 0) return `LONG, you own ${d.shares} shares`;
  if (d.shares < 0) return `SHORT, you owe ${Math.abs(d.shares)} borrowed shares`;
  return "FLAT, you hold only cash";
}

function priceHover(d) {
  const lines = [
    `<b>${d.date}</b>`,
    `Price: ${money(d.price)}`,
    `Position: ${positionWords(d)}`,
  ];
  if (d.shares !== 0) {
    lines.push(`Average entry price: ${money(d.entry_price)}`);
    lines.push(`Position value: ${money(d.position_value)}`);
    lines.push(`Open profit or loss if closed now: ${fmtSigned(d.unrealized_pnl)}`);
    lines.push(
      `To close now you would ${d.shares > 0 ? "SELL" : "BUY BACK"} ${Math.abs(d.shares)} shares`
    );
  }
  lines.push(`Account value (cash plus shares): ${money(d.equity)}`);
  lines.push(`Total gain or loss so far: ${fmtSigned(d.total_pnl)} (${d.return_pct.toFixed(2)}%)`);
  return lines.join("<br>");
}

function tradeHover(t) {
  return [
    `<b>${t.kind}</b> on ${t.date}`,
    `${t.direction === "BUY" ? "Bought" : "Sold"} ${t.shares_traded} shares at ${money(t.price)}`,
    `Value traded: ${money(t.notional)}`,
    `Fee paid: ${money(t.fee)}`,
    `Locked-in profit or loss from this trade: ${fmtSigned(t.realized_pnl)}`,
    `Shares held after this trade: ${t.shares_after} (${t.position_after})`,
  ].join("<br>");
}

function positionShapes(days) {
  const shapes = [];
  if (!days.length) return shapes;

  const colorOf = (p) =>
    p === "LONG" ? "rgba(27,127,77,0.14)" : p === "SHORT" ? "rgba(179,38,30,0.14)" : null;

  let start = 0;
  for (let i = 1; i <= days.length; i++) {
    const current = i < days.length ? days[i].position : null;
    if (current !== days[start].position) {
      const color = colorOf(days[start].position);
      if (color) {
        shapes.push({
          type: "rect",
          xref: "x",
          yref: "paper",
          x0: days[start].date,
          x1: days[i - 1].date,
          y0: 0.38,
          y1: 1,
          fillcolor: color,
          line: { width: 0 },
          layer: "below",
        });
      }
      start = i;
    }
  }
  return shapes;
}

function buildChart(r) {
  const chartDiv = $("chart");

  if (!PLOTLY_OK) {
    chartDiv.innerHTML =
      `<p class="status error">The chart library (Plotly) could not load. ` +
      `Check your internet connection and refresh the page. The image download below still works.</p>`;
    return;
  }

  const days = r.timeline.days;
  const trades = r.timeline.trades;
  const dates = days.map((d) => d.date);

  const traces = [
    {
      x: dates,
      y: days.map((d) => d.price),
      type: "scatter",
      mode: "lines",
      name: "Price",
      line: { color: "#002855", width: 1.6 },
      hoverinfo: "text",
      text: days.map(priceHover),
      xaxis: "x",
      yaxis: "y",
    },
  ];

  if (trades.length) {
    traces.push({
      x: trades.map((t) => t.date),
      y: trades.map((t) => t.price),
      type: "scatter",
      mode: "markers",
      name: "Trades",
      marker: {
        size: 12,
        symbol: trades.map((t) => (t.direction === "BUY" ? "triangle-up" : "triangle-down")),
        color: trades.map((t) => (t.direction === "BUY" ? "#1b7f4d" : "#b3261e")),
        line: { color: "#ffffff", width: 1 },
      },
      hoverinfo: "text",
      text: trades.map(tradeHover),
      xaxis: "x",
      yaxis: "y",
    });
  }

  const equitySeries = [
    { name: "Reinforcement learning agent", values: days.map((d) => d.equity), color: "#0b3d7a", dash: "solid" },
    { name: "Buy and hold", values: days.map((d) => d.benchmarks.buy_and_hold), color: "#7a8699", dash: "dash" },
    { name: "20-day moving average rule", values: days.map((d) => d.benchmarks.moving_average_20), color: "#b26a00", dash: "dot" },
    { name: "Random policy", values: days.map((d) => d.benchmarks.random), color: "#8e44ad", dash: "dashdot" },
  ];

  equitySeries.forEach((s) => {
    traces.push({
      x: dates,
      y: s.values,
      type: "scatter",
      mode: "lines",
      name: s.name,
      line: { color: s.color, width: 1.8, dash: s.dash },
      hovertemplate: s.name + ": %{y:$,.2f}<extra></extra>",
      xaxis: "x",
      yaxis: "y2",
    });
  });

  state.baseShapes = positionShapes(days);

  const layout = {
    height: 680,
    margin: { t: 30, r: 20, b: 50, l: 70 },
    hovermode: "closest",
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    shapes: state.baseShapes,
    xaxis: {
      type: "date",
      anchor: "y2",
      rangeselector: {
        buttons: [
          { count: 1, label: "1M", step: "month", stepmode: "backward" },
          { count: 3, label: "3M", step: "month", stepmode: "backward" },
          { step: "all", label: "All" },
        ],
      },
    },
    yaxis: { domain: [0.38, 1], title: "Price ($)" },
    yaxis2: { domain: [0, 0.28], title: "Account value ($)", anchor: "x" },
    legend: { orientation: "h", y: -0.12 },
  };

  Plotly.purge(chartDiv);
  Plotly.newPlot(chartDiv, traces, layout, { responsive: true, displaylogo: false });

  // Clicking a point on the price line jumps the day explorer to that day.
  chartDiv.on("plotly_click", (ev) => {
    const p = ev.points[0];
    if (p.curveNumber === 0) setDay(p.pointIndex);
  });
}

// ---------- Day explorer ----------
function renderExplorer(r) {
  state.explorerDays = r.timeline.days;
  const slider = $("day-slider");
  slider.max = Math.max(0, state.explorerDays.length - 1);
  setDay(state.explorerDays.length - 1);
}

function setDay(index) {
  const days = state.explorerDays;
  if (!days.length) return;
  const i = Math.max(0, Math.min(days.length - 1, index));
  $("day-slider").value = i;
  renderDayCard(i);
  moveCursor(days[i].date);
}

function setDayByDate(date) {
  const i = state.explorerDays.findIndex((d) => d.date === date);
  if (i >= 0) setDay(i);
}

function moveCursor(date) {
  if (!PLOTLY_OK || !$("chart").data) return;
  const cursor = {
    type: "line",
    xref: "x",
    yref: "paper",
    x0: date,
    x1: date,
    y0: 0,
    y1: 1,
    line: { color: "#14213d", width: 1.5, dash: "dot" },
  };
  Plotly.relayout("chart", { shapes: state.baseShapes.concat([cursor]) });
}

function renderDayCard(i) {
  const d = state.explorerDays[i];
  const cfg = state.lastResult.config;

  const closeFee = cfg.fee_rate * Math.abs(d.shares) * d.price;
  const netIfClosed = d.unrealized_pnl - closeFee;

  $("day-label").innerHTML = `<strong>Day ${i + 1} of ${state.explorerDays.length}: ${esc(d.date)}</strong>`;

  const items = [
    ["Price", money(d.price), "price"],
    ["Position", d.position, "position"],
    ["Shares held", String(d.shares), "shares"],
    ["Average entry price", d.shares ? money(d.entry_price) : "Not applicable", "entry_price"],
    ["Position value", money(d.position_value), "position_value"],
    ["Open profit or loss", fmtSigned(d.unrealized_pnl), "unrealized"],
    ["Locked-in profit or loss", fmtSigned(d.realized_pnl_total), "realized"],
    ["Cash", money(d.cash), "cash"],
    ["Account value", money(d.equity), "equity"],
    ["Total gain or loss", `${fmtSigned(d.total_pnl)} (${d.return_pct.toFixed(2)}%)`, "total_gain"],
  ];

  $("day-card").innerHTML = items
    .map(
      ([label, value, key]) =>
        `<div class="kpi"><span>${tip(key, label)}</span><strong>${esc(value)}</strong></div>`
    )
    .join("");

  let sentence;
  if (d.shares === 0) {
    sentence = "On this day you hold no shares, so there is nothing to close. All of your money is in cash.";
  } else {
    sentence =
      `On this day you are ${d.position === "LONG" ? "long" : "short"} ${Math.abs(d.shares)} shares. ` +
      `To close the position at today's price you would ${d.close_action === "SELL" ? "SELL" : "BUY BACK"} ` +
      `${d.close_shares} shares, a value of about ${money(Math.abs(d.position_value))}. ` +
      `After the closing fee of ${money(closeFee)}, the result would be ${fmtSigned(netIfClosed)}.`;
  }
  $("day-sentence").textContent = sentence;
}

// ---------- Comparison table ----------
function renderMetrics(r) {
  const m = r.metrics;

  const best = {};
  METRIC_COLUMNS.forEach((c) => {
    if (c.best) {
      best[c.key] = Math.max(...STRATEGIES.map((s) => m[s.key][c.key]));
    }
  });

  const head =
    "<tr><th>Strategy</th>" +
    METRIC_COLUMNS.map((c) => `<th>${tip(c.gloss)}</th>`).join("") +
    "</tr>";

  const body = STRATEGIES.map((s) => {
    const cells = METRIC_COLUMNS.map((c) => {
      const v = m[s.key][c.key];
      const isBest = c.best && v === best[c.key];
      return `<td class="${isBest ? "best" : ""}">${c.fmt(v)}</td>`;
    }).join("");
    const rowClass = s.key === "rl_agent" ? "highlight" : "";
    return `<tr class="${rowClass}"><td><strong>${tip(s.gloss)}</strong></td>${cells}</tr>`;
  }).join("");

  $("metrics-table").innerHTML = `<thead>${head}</thead><tbody>${body}</tbody>`;
}

// ---------- Trade log ----------
function renderTrades(r) {
  const trades = r.timeline.trades;
  const table = $("trade-table");

  if (trades.length === 0) {
    table.innerHTML = `<tbody><tr><td class="muted">The agent made no trades in the test period. It stayed in its starting position the whole time.</td></tr></tbody>`;
    return;
  }

  const rows = trades
    .map((t) => {
      const sign = t.direction === "BUY" ? "+" : "-";
      const cls = t.direction === "BUY" ? "buy" : "sell";
      const pill = t.position_after.toLowerCase();
      return `<tr class="clickable" data-date="${esc(t.date)}">
        <td>${esc(t.date)}</td>
        <td><span class="pill ${pill}">${esc(t.kind)}</span></td>
        <td class="${cls}">${sign}${t.shares_traded}</td>
        <td>${money(t.price)}</td>
        <td>${money(t.notional)}</td>
        <td>${money(t.fee)}</td>
        <td>${fmtSigned(t.realized_pnl)}</td>
        <td>${t.shares_after}</td>
        <td class="wrap">${esc(t.explanation)}</td>
      </tr>`;
    })
    .join("");

  table.innerHTML = `
    <thead>
      <tr>
        <th>Date</th>
        <th>${tip("position", "Action")}</th>
        <th>${tip("shares", "Shares traded")}</th>
        <th>${tip("price", "Price")}</th>
        <th>${tip("trade_value", "Value traded")}</th>
        <th>${tip("fee", "Fee")}</th>
        <th>${tip("realized", "Locked-in P&L")}</th>
        <th>${tip("shares", "Shares after")}</th>
        <th>Plain-English explanation</th>
      </tr>
    </thead>
    <tbody>${rows}</tbody>`;
}

on("trade-table", "click", (e) => {
  const row = e.target.closest("tr[data-date]");
  if (!row) return;
  setDayByDate(row.dataset.date);
  $("day-card").scrollIntoView({ behavior: "smooth", block: "center" });
});

on("day-slider", "input", (e) => setDay(parseInt(e.target.value, 10)));

// ---------- Downloads ----------
on("download-json", "click", () => {
  if (!state.lastResult) return;
  const blob = new Blob([JSON.stringify(state.lastResult, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "result_" + state.lastResult.job_id.slice(0, 8) + ".json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

// ============================================================
// Glossary
// ============================================================
function renderGlossary() {
  $("glossary").innerHTML = Object.values(GLOSSARY)
    .map((g) => `<dt>${esc(g.label)}</dt><dd>${esc(g.text)}</dd>`)
    .join("");
}

// ============================================================
// Recent runs (stored in this browser)
// ============================================================
function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
  } catch (err) {
    return [];
  }
}

function saveHistory(list) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(list.slice(0, 20)));
  } catch (err) {
    // Storage can be blocked, for example in private windows. The page still works without it.
  }
}

function addHistory(entry) {
  const list = loadHistory().filter((h) => h.job_id !== entry.job_id);
  list.unshift(entry);
  saveHistory(list);
  renderHistory();
}

function markHistory(jobId, status) {
  const list = loadHistory().map((h) => (h.job_id === jobId ? { ...h, status: status } : h));
  saveHistory(list);
  renderHistory();
}

function renderHistory() {
  const list = loadHistory();
  const el = $("history-list");

  if (list.length === 0) {
    el.innerHTML = `<li class="muted">No runs yet. Your completed and failed runs will appear here.</li>`;
    return;
  }

  el.innerHTML = list
    .map(
      (h) => `
      <li>
        <div>
          <strong>${esc(h.label)}</strong>
          <span class="muted small">${esc(new Date(h.created).toLocaleString())}</span>
          <div class="mono small muted">${esc(h.job_id)}</div>
        </div>
        <div class="history-actions">
          <span class="badge ${esc(h.status)}">${esc(h.status)}</span>
          <button type="button" class="btn btn-small btn-ghost" data-open="${esc(h.job_id)}">Open</button>
        </div>
      </li>`
    )
    .join("");
}

on("history-list", "click", (e) => {
  const jobId = e.target.dataset.open;
  if (!jobId) return;
  track(jobId, null);
});

// ============================================================
// Start
// ============================================================
function init() {
  decorateFields();
  renderGlossary();

  $("ticker").value = TICKER_DEFAULTS.ticker;
  $("ticker-start").value = TICKER_DEFAULTS.start;
  $("ticker-end").value = TICKER_DEFAULTS.end;

  fillDefaults();
  renderHistory();
  checkApi();
}

init();