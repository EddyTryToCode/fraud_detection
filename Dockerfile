# ==============================================================================
# Dockerfile: Production Container for Bank Account Fraud (BAF) Detection System
# ==============================================================================
FROM python:3.10-slim

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho C++/OpenMP (XGBoost, LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code & artifacts
COPY src/ /app/src/
COPY models/ /app/models/
COPY scripts/ /app/scripts/

# Healthcheck để kiểm tra trạng thái container
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000 8501

# Lệnh khởi chạy mặc định: FastAPI Serving
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
