ARG BUILD_FROM=python:3.11-slim
FROM ${BUILD_FROM}

ENV PYTHONUNBUFFERED=1

# Install required system dependencies for opencv-python-headless
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libglib2.0-0 \
       libsm6 \
       libxext6 \
       libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (for docker layer caching)
COPY pyproject.toml uv.lock* ./
RUN uv pip install --system .

# Copy application files
COPY app /app/app
COPY models /app/models
COPY scripts /app/scripts
COPY run.sh /app/run.sh

RUN chmod a+x /app/run.sh

ENTRYPOINT [ "/app/run.sh" ]
