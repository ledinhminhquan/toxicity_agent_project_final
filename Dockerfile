FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first (better layer caching)
COPY requirements.txt pyproject.toml README.md /app/

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Copy source
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY docs/ /app/docs/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["toxicity-agent", "serve", "--config", "configs/infer.yaml", "--host", "0.0.0.0", "--port", "8000"]
