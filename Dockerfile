# ── Stage 1: base image ───────────────────────────────────────────────────
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# ── Install system dependencies ────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy source code ───────────────────────────────────────────────────────
COPY . .

# ── Train model at build time (artifacts saved inside container) ───────────
# If you already have pre-trained artifacts, comment out the next line
# and COPY your artifacts/ folder instead.
RUN python model.py

# ── Expose port & run Flask ────────────────────────────────────────────────
EXPOSE 5000

# Liveness / readiness check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "app.py"]
