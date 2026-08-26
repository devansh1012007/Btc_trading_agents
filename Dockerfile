# =============================================================================
# Multi-stage build for Btc_trading_agents
# =============================================================================
# Stage 1: builder — install dependencies into a virtualenv
# Stage 2: runtime — slim image with only the venv + app code
# =============================================================================

FROM python:3.12-slim AS builder

WORKDIR /build

# Install build deps (gcc for ccxt native extensions, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        && rm -rf /var/lib/apt/lists/*

# Create virtualenv so we can copy it into the final image
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip first (faster builds, security fixes)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python deps (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Runtime stage
# =============================================================================
FROM python:3.12-slim AS runtime

# Non-root user for safety
RUN groupadd -r trader && useradd -r -g trader -m trader

WORKDIR /opt/trading-agent

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only runtime deps that don't need compilation
RUN pip install --no-cache-dir --upgrade pip

# Create data directories with correct ownership
RUN mkdir -p /opt/trading-agent/data/{raw,processed,models,logs,control} \
    && chown -R trader:trader /opt/trading-agent

# Copy application code
COPY --chown=trader:trader src/ ./src/
COPY --chown=trader:trader config/ ./config/
COPY --chown=trader:trader scripts/entrypoint.sh ./scripts/entrypoint.sh
COPY --chown=trader:trader scripts/run.py ./scripts/run.py
COPY --chown=trader:trader scripts/retrain_all.py ./scripts/retrain_all.py
COPY --chown=trader:trader scripts/approve_model.py ./scripts/approve_model.py
COPY --chown=trader:trader scripts/rollback_model.py ./scripts/rollback_model.py
COPY --chown=trader:trader scripts/status.py ./scripts/status.py
COPY --chown=trader:trader scripts/train_*.py ./scripts/
COPY --chown=trader:trader scripts/backtest_*.py ./scripts/
COPY --chown=trader:trader scripts/calibrate_synthetic_liquidity.py ./scripts/
COPY --chown=trader:trader scripts/label_zone_events.py ./scripts/
COPY --chown=trader:trader scripts/reset_breaker.py ./scripts/

# Switch to non-root user
USER trader

# Expose no ports (this is a trading agent, not a web server)

# Health check: verify Python and key imports work
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import pandas; import lightgbm; import ccxt; print('OK')" || exit 1

# Default: run historical backtest (override with docker-compose or `docker run`)
CMD ["python", "scripts/run.py", "--mode", "historical", "--env", "paper"]
