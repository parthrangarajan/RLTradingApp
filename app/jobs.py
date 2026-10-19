import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

# One worker only: TensorFlow 1.x training should not run in parallel threads.
_executor = ThreadPoolExecutor(max_workers=1)
_jobs = {}
_lock = threading.Lock()


def _make_progress(job_id):
    def progress(fraction, message):
        with _lock:
            _jobs[job_id]["progress"] = round(float(fraction), 4)
            _jobs[job_id]["message"] = message
    return progress


def _worker(job_id, fn, payload):
    with _lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["message"] = "Started"

    try:
        result = fn(job_id, payload, _make_progress(job_id))
        with _lock:
            _jobs[job_id].update(
                status="completed", progress=1.0, message="Done", result=result
            )
    except Exception as exc:
        with _lock:
            _jobs[job_id].update(
                status="failed",
                message="Failed",
                error="{}: {}".format(type(exc).__name__, exc),
                traceback=traceback.format_exc(),
            )


def create_job(fn, payload):
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Queued",
            "result": None,
            "error": None,
            "created_at": time.time(),
        }
    _executor.submit(_worker, job_id, fn, payload)
    return job_id


def get_job(job_id):
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None