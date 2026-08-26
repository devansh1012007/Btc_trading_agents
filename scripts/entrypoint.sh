#!/usr/bin/env bash
# =============================================================================
# Entrypoint wrapper for Btc_trading_agents
# =============================================================================
# Provides:
#   1. Network connectivity check with exponential backoff
#   2. Graceful degradation when exchange is unreachable
#   3. Signal handling (SIGTERM, SIGINT) for clean shutdown
#   4. Health reporting to stdout
#
# Usage:
#   docker-compose up              # uses env vars from .env or compose defaults
#   docker run --env-file .env     # pass env vars directly
#
# Environment variables (all optional, see docker-compose.yml for defaults):
#   TRADING_MODE          - "historical" or "live" (default: historical)
#   TRADING_ENV           - "paper" or "live" (default: paper)
#   INPUT_DATA            - path to OHLCV file (required for historical mode)
#   SYMBOL                - trading pair (default: BTC/USDT)
#   MAX_BARS              - max bars to process (historical mode)
#   INITIAL_EQUITY        - starting equity for backtest (default: 10000)
#   NETWORK_RETRY_MAX     - max network retries (default: 5)
#   NETWORK_RETRY_BASE    - base delay in seconds (default: 10)
#   NETWORK_RETRY_BACKOFF - backoff multiplier (default: 2)
#   SHUTDOWN_TIMEOUT      - seconds to wait for graceful shutdown (default: 30)
# =============================================================================

set -euo pipefail

# ── Configuration from environment ────────────────────────────────────────
TRADING_MODE="${TRADING_MODE:-historical}"
TRADING_ENV="${TRADING_ENV:-paper}"
INPUT_DATA="${INPUT_DATA:-}"
SYMBOL="${SYMBOL:-BTC/USDT}"
MAX_BARS="${MAX_BARS:-}"
INITIAL_EQUITY="${INITIAL_EQUITY:-10000}"
NETWORK_RETRY_MAX="${NETWORK_RETRY_MAX:-5}"
NETWORK_RETRY_BASE="${NETWORK_RETRY_BASE_DELAY:-10}"
NETWORK_RETRY_BACKOFF="${NETWORK_RETRY_BACKOFF:-2}"
SHUTDOWN_TIMEOUT="${SHUTDOWN_TIMEOUT:-30}"

# ── Logging helpers ───────────────────────────────────────────────────────
log_info()  { echo "[INFO]  $(date -u '+%Y-%m-%d %H:%M:%S UTC') $*"; }
log_warn()  { echo "[WARN]  $(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" >&2; }
log_error() { echo "[ERROR] $(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" >&2; }

# ── State ─────────────────────────────────────────────────────────────────
SHUTTING_DOWN=0
NETWORK_AVAILABLE=0

# ── Signal handlers ───────────────────────────────────────────────────────
handle_shutdown() {
    if [ "$SHUTTING_DOWN" -eq 1 ]; then
        log_warn "Received second signal during shutdown — forcing exit"
        exit 1
    fi
    SHUTTING_DOWN=1
    log_info "Shutdown signal received — finishing current bar and exiting gracefully..."
    if [ -n "${MAIN_PID:-}" ] && kill -0 "$MAIN_PID" 2>/dev/null; then
        # Send SIGTERM to the Python process (it handles SIGTERM for graceful exit)
        kill -TERM "$MAIN_PID" 2>/dev/null || true
        # Wait up to SHUTDOWN_TIMEOUT seconds
        local waited=0
        while kill -0 "$MAIN_PID" 2>/dev/null && [ "$waited" -lt "$SHUTDOWN_TIMEOUT" ]; do
            sleep 1
            waited=$((waited + 1))
        done
        # Force kill if still running
        if kill -0 "$MAIN_PID" 2>/dev/null; then
            log_warn "Shutdown timeout reached — force killing"
            kill -9 "$MAIN_PID" 2>/dev/null || true
        else
            log_info "Process exited cleanly"
        fi
    fi
    exit 0
}

trap handle_shutdown SIGTERM SIGINT

# ── Network connectivity check ────────────────────────────────────────────
check_network() {
    # Try to resolve a known host (Binance API)
    python -c "
import socket
import sys
try:
    # Try resolving api.binance.com (primary)
    socket.getaddrinfo('api.binance.com', 443, socket.AF_INET)
    sys.exit(0)
except socket.gaierror:
    pass
try:
    # Fallback: try sandbox
    socket.getaddrinfo('sapi.binance.com', 443, socket.AF_INET)
    sys.exit(0)
except socket.gaierror:
    pass
sys.exit(1)
" 2>/dev/null
}

# ── Exchange reachability check ───────────────────────────────────────────
check_exchange() {
    # Attempt a lightweight CCXT exchange info call
    python -c "
import ccxt
import sys
exchange = ccxt.binance({'enableRateLimit': True, 'sandbox': True})
try:
    exchange.load_markets()
    sys.exit(0)
except Exception as e:
    print(f'Exchange check failed: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null
}

# ── Network retry with exponential backoff ────────────────────────────────
retry_network() {
    local max_retries="$1"
    local base_delay="$2"
    local backoff_factor="$3"
    local attempt=0
    local delay="$base_delay"

    log_info "Checking network connectivity and exchange reachability..."

    while [ "$attempt" -lt "$max_retries" ]; do
        if check_network && check_exchange; then
            log_info "Network and exchange are reachable"
            return 0
        fi

        attempt=$((attempt + 1))
        if [ "$attempt" -lt "$max_retries" ]; then
            log_warn "Exchange not reachable (attempt $attempt/$max_retries), retrying in ${delay}s..."
            sleep "$delay"
            delay=$(echo "$delay * $backoff_factor" | bc 2>/dev/null || echo "$((delay * backoff_factor))")
        fi
    done

    log_error "Exchange unreachable after $max_retries attempts"
    return 1
}

# ── Build command arguments ───────────────────────────────────────────────
build_args() {
    local args=("--env" "$TRADING_ENV" "--mode" "$TRADING_MODE")

    if [ "$TRADING_MODE" = "historical" ]; then
        if [ -z "$INPUT_DATA" ]; then
            log_error "INPUT_DATA is required for historical mode"
            log_error "Usage: docker run -e INPUT_DATA=/data/BTCUSDT.csv ..."
            exit 1
        fi
        args+=("--input" "$INPUT_DATA")
    fi

    if [ -n "$SYMBOL" ]; then
        args+=("--symbol" "$SYMBOL")
    fi

    if [ -n "$MAX_BARS" ]; then
        args+=("--max-bars" "$MAX_BARS")
    fi

    if [ -n "$INITIAL_EQUITY" ]; then
        args+=("--initial-equity" "$INITIAL_EQUITY")
    fi

    echo "${args[@]}"
}

# ── Main ──────────────────────────────────────────────────────────────────
main() {
    log_info "=== Btc_trading_agents Docker Entrypoint ==="
    log_info "Mode: $TRADING_MODE | Env: $TRADING_ENV | Symbol: $SYMBOL"

    # ── Network check (skip for historical mode with local data) ──────────
    if [ "$TRADING_MODE" = "live" ]; then
        if ! retry_network "$NETWORK_RETRY_MAX" "$NETWORK_RETRY_BASE" "$NETWORK_RETRY_BACKOFF"; then
            log_error "Cannot reach exchange after retries. Aborting live mode."
            log_error "Options:"
            log_error "  1. Check your network connection"
            log_error "  2. Verify Binance API is accessible from this container"
            log_error "  3. Switch to historical mode: TRADING_MODE=historical"
            log_error "  4. Increase retry limits: NETWORK_RETRY_MAX=10 NETWORK_RETRY_BASE=15"
            exit 1
        fi
    else
        log_info "Historical mode — skipping live exchange check (uses local data)"
    fi

    # ── Validate input data for historical mode ──────────────────────────
    if [ "$TRADING_MODE" = "historical" ] && [ -n "$INPUT_DATA" ]; then
        if [ ! -f "$INPUT_DATA" ]; then
            log_error "Input data file not found: $INPUT_DATA"
            log_error "Mount your data with: -v /host/path:/opt/trading-agent/data/raw"
            exit 1
        fi
        log_info "Input data found: $INPUT_DATA"
    fi

    # ── Build run command ────────────────────────────────────────────────
    local cmd_args
    cmd_args=$(build_args)

    log_info "Starting trading agent: python scripts/run.py $cmd_args"
    log_info "-------------------------------------------"

    # ── Run the trading agent ────────────────────────────────────────────
    python scripts/run.py $cmd_args &
    MAIN_PID=$!

    # Wait for the process to exit
    wait "$MAIN_PID"
    local exit_code=$?

    log_info "Trading agent exited with code $exit_code"
    exit "$exit_code"
}

main "$@"
