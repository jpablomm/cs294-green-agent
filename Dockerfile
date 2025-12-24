# CS294 Green Agent - OSWorld Assessment
# A2A-compliant agent for desktop automation benchmarks

FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Install system dependencies required by some Python packages
USER root
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN adduser agent
USER agent
WORKDIR /home/agent

# Copy dependency files first for better caching
COPY --chown=agent:agent pyproject.toml README.md ./

# Copy application code
COPY --chown=agent:agent src src
COPY --chown=agent:agent green_agent green_agent
COPY --chown=agent:agent white_agent white_agent
COPY --chown=agent:agent vendor vendor

# Install dependencies (without lock file since we're adding new deps)
RUN uv sync

# Set PYTHONPATH to include vendor code
ENV PYTHONPATH=/home/agent:/home/agent/vendor/OSWorld
ENV HOST=0.0.0.0
ENV PORT=9009

EXPOSE 9009

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the server
ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0", "--port", "9009"]
