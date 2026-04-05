# 1. Use the official lightweight Python base image
FROM python:3.11-slim

# 2. Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# 3. Install system dependencies (required for XGBoost + OpenMP on Linux)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory inside the container
WORKDIR /app

# 5. Copy only requirements first (Docker layer caching)
COPY requirements.txt .

# 6. Install all Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        xgboost \
        optuna \
        gradio \
        joblib \
        -r requirements.txt

# 7. Copy the project files
COPY app/       ./app/
COPY src/        ./src/
COPY artifacts/  ./artifacts/

# 8. Expose FastAPI port
EXPOSE 8000

# 9. Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
