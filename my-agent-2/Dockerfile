# Dockerfile — Track 2 submission image
#
# Build:  docker build -t my-agent-2 .
# Run:    docker run --network=none -v ./cards/F1:/input -v ./out:/output my-agent-2
#
# The entrypoint reads /input/card.toml and writes three deliverables to /output/.
# Network is restricted (only House model endpoint reachable via MODEL_ENDPOINT).

FROM python:3.11-slim

# System deps for scipy/statsmodels/arch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libopenblas-dev liblapack-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 agent
WORKDIR /home/agent

# Create output directory with correct ownership
RUN mkdir -p /output && chown agent:agent /output

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY --chown=agent:agent . .

# Switch to non-root
USER agent

# Entrypoint: expects /input/card.toml, writes to /output
ENV PYTHONPATH=/home/agent
ENTRYPOINT ["sh", "-c", "python forecast.py --panels /input --text /input/text --asof \"$ASOF\" --out /output/forecast.parquet"]