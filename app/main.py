import os
import re
import uuid

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from .data_source import download_ticker  # noqa: E402
from .experiment import run_experiment  # noqa: E402
from .jobs import create_job, get_job  # noqa: E402
from .marketwatch import parse_marketwatch_csv  # noqa: E402
from .paths import DATA_DIR, RESULTS_DIR, ensure_dirs  # noqa: E402
from .schemas import RunRequest, TickerRequest  # noqa: E402

app = FastAPI(
    title="RL Trading Simulator",
    description="Train a reinforcement learning agent on stock data and compare it with benchmarks.",
    version="1.1.0",
)

from fastapi.middleware.cors import CORSMiddleware

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://127.0.0.1:8000,http://localhost:8000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

JOB_ID = re.compile(r"^[0-9a-f]{32}$")

FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend"
)
app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(os.path.join(FRONTEND_DIR, "favicon.ico"), media_type="image/x-icon")

@app.on_event("startup")
def _startup():
    ensure_dirs()


def _check_job_id(job_id):
    if not JOB_ID.match(job_id):
        raise HTTPException(status_code=404, detail="Unknown job id")


def _parse_or_422(raw):
    try:
        return parse_marketwatch_csv(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


def _dataset_summary(file_id, df, source):
    preview = []
    for ts, row in df.head(5).iterrows():
        record = {"Date": ts.strftime("%Y-%m-%d")}
        record.update({key: float(value) for key, value in row.items()})
        preview.append(record)

    return {
        "file_id": file_id,
        "source": source,
        "rows": int(len(df)),
        "start": df.index.min().strftime("%Y-%m-%d"),
        "end": df.index.max().strftime("%Y-%m-%d"),
        "columns": list(df.columns),
        "preview": preview,
    }


def _save_raw(file_id, raw):
    path = os.path.join(DATA_DIR, file_id + ".csv")
    with open(path, "wb") as fh:
        fh.write(raw)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/data/upload", status_code=201)
async def upload_data(file: UploadFile = File(...)):
    """Upload a CSV with columns Date, Open, High, Low, Close, Volume."""
    raw = await file.read()
    df = _parse_or_422(raw)
    file_id = uuid.uuid4().hex
    _save_raw(file_id, raw)
    return _dataset_summary(file_id, df, source="Uploaded file: " + file.filename)


@app.post("/data/fetch-ticker", status_code=201)
def fetch_ticker(body: TickerRequest):
    """Download daily prices for a ticker from Yahoo Finance."""
    try:
        frame = download_ticker(body.ticker, body.start, body.end)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not download data from Yahoo Finance: {}".format(exc),
        )

    raw = frame.to_csv(index=False).encode("utf-8")
    df = _parse_or_422(raw)
    file_id = uuid.uuid4().hex
    _save_raw(file_id, raw)
    return _dataset_summary(file_id, df, source="Yahoo Finance: " + body.ticker)


def _run_job(job_id, payload, progress):
    with open(payload["data_path"], "rb") as fh:
        raw = fh.read()
    df = parse_marketwatch_csv(raw)
    return run_experiment(df, payload["cfg"], job_id, progress)


@app.post("/runs", status_code=202)
def start_run(req: RunRequest):
    """Start training and evaluation in the background."""
    data_path = os.path.join(DATA_DIR, req.file_id + ".csv")
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Unknown file_id. Load data first.")

    if not (req.train_start < req.train_end < req.test_start < req.test_end):
        raise HTTPException(
            status_code=422,
            detail="Dates must satisfy train_start < train_end < test_start < test_end",
        )

    cfg = req.dict()
    for key in ("train_start", "train_end", "test_start", "test_end"):
        cfg[key] = cfg[key].isoformat()

    job_id = create_job(_run_job, {"data_path": data_path, "cfg": cfg})
    return {
        "job_id": job_id,
        "status_url": "/runs/" + job_id,
        "chart_url": "/runs/" + job_id + "/chart",
    }


@app.get("/runs/{job_id}")
def run_status(job_id: str):
    _check_job_id(job_id)
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/runs/{job_id}/chart")
def run_chart(job_id: str):
    _check_job_id(job_id)
    path = os.path.join(RESULTS_DIR, job_id, "chart.png")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="Chart not ready. Check /runs/" + job_id + " until status is completed.",
        )
    return FileResponse(path, media_type="image/png")