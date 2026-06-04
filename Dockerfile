# a2z_hunter API — agentic vector search
FROM python:3.12-slim

WORKDIR /app

# System deps: build tools for any wheels that need compiling
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better layer caching).
# Copy only what's needed to resolve & install the package.
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 8000

# API_PORT is honored by a2z_hunter.api.run()
ENV API_PORT=8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -fs http://localhost:8000/health || exit 1

CMD ["python", "-m", "a2z_hunter.api"]
