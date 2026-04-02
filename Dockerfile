# ==============================
# AutoMind OpenEnv - Dockerfile
# FINAL (LIGHTWEIGHT + SAFE)
# ==============================

FROM python:3.10-slim

# ------------------------------
# SYSTEM SETUP
# ------------------------------
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------
# COPY FILES
# ------------------------------
COPY . .

# ------------------------------
# INSTALL PYTHON DEPENDENCIES
# ------------------------------
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------
# EXPOSE PORT
# ------------------------------
EXPOSE 8000

# ------------------------------
# START SERVER
# ------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]