# Sử dụng Python 3.10 slim image làm base
FROM python:3.10-slim

# Thiết lập working directory
WORKDIR /app

# Cài đặt system dependencies (nếu cần)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements_api.txt .

# Cài đặt Python dependencies
RUN pip install --no-cache-dir -r requirements_api.txt

# Cài đặt thêm torch, numpy, scikit-learn cho model
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    numpy==1.24.3 \
    scikit-learn==1.3.2 \
    pandas==2.1.3

# Copy source code
COPY src/ ./src/
COPY api.py .

# Copy model file (nếu có)
COPY dqn_product_recommendation.pth . 

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Command để chạy API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
