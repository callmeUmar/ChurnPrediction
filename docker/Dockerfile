# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.9-slim

# Keeps Python from buffering stdout/stderr (important for Docker logs)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first (Docker layer caching — only reinstalls if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        xgboost \
        optuna \
        gradio \
        -r requirements.txt

# ── Copy project files ────────────────────────────────────────────────────────
COPY app/        ./app/
COPY src/        ./src/
COPY artifacts/  ./artifacts/

# ── Expose FastAPI port ───────────────────────────────────────────────────────
EXPOSE 8000

# ── Start the server ──────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
