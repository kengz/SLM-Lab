FROM ubuntu:22.04

LABEL org.opencontainers.image.source=https://github.com/kengz/SLM-Lab
LABEL org.opencontainers.image.description="Modular Deep Reinforcement Learning framework in PyTorch"

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"

# System dependencies for gymnasium (box2d, mujoco, atari)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git swig \
    python3-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Install dependencies first (cache layer)
# README.md needed for hatchling build metadata
COPY pyproject.toml uv.lock README.md ./
RUN uv python install 3.12 && uv sync --frozen

# Copy remaining source
COPY . .

CMD ["uv", "run", "slm-lab", "--help"]
