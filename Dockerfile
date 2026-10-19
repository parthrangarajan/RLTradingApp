FROM python:3.7-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /code

# libgomp1 is a runtime library that TensorFlow needs
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install gym first, using the build workaround that worked on your machine.
# requirements.txt then sees gym as already installed and does not rebuild it.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "setuptools==57.5.0" "wheel==0.37.1" \
    && pip install --no-cache-dir --no-build-isolation gym==0.18.0 \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY frontend ./frontend

# Run as a non-root user, which is a basic security practice
RUN useradd --create-home appuser \
    && mkdir -p /code/data /code/models /code/results \
    && chown -R appuser /code
USER appuser

EXPOSE 7860

# Hugging Face expects port 7860. PORT can override it on other platforms.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]