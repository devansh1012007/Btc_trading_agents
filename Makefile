# Btc_trading_agents Makefile
# ============================================================================
# Development, testing, and deployment targets for the BTC trading agent.
# ============================================================================
SHELL := /bin/bash

# Directories
SRC_DIR     := src
TESTS_DIR   := tests
SCRIPTS_DIR := scripts
DATA_DIR    := data
REPORTS_DIR := $(DATA_DIR)/reports

# Tools (project venv; created via `make install-dev`)
PYTHON   := python3
RUFF     := .venv/bin/ruff
MYPY     := .venv/bin/mypy
PYTEST_V := .venv/bin/python -m pytest

BENCH_DATE := $(shell date +%Y_%m_%d)

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make test              Run the full test suite"
	@echo "  make unit              Run unit tests only (tests/unit)"
	@echo "  make integration       Run integration tests only (tests/integration)"
	@echo "  make e2e               Run e2e tests only (tests/e2e)"
	@echo "  make coverage          Run tests with coverage report"
	@echo "  make lint              Lint with ruff"
	@echo "  make typecheck         Type-check strict modules with mypy"
	@echo "  make format            Auto-format with ruff"
	@echo "  make backtest          Full-stack walk-forward backtest on real data"
	@echo "  make benchmark-smoke   Synthetic smoke-run of the benchmark ladder"
	@echo "  make preflight         Pre-flight config validation (paper env)"
	@echo "  make pipeline          Pipeline dry-run (fetches 100 sandbox bars)"
	@echo "  make dashboard         Launch TUI dashboard (requires TTY)"
	@echo "  make dashboard-headless One-shot headless dashboard render"
	@echo "  make latency           Software performance benchmark (latency/RAM)"
	@echo "  make figures           Generate docs/images figures (synthetic)"
	@echo "  make demo              Launch the Streamlit decision demo"
	@echo "  make build-image       Build the Docker image"
	@echo "  make install-dev       Create venv + install runtime and dev deps"
	@echo "  make clean             Remove caches and build artifacts"

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
.PHONY: test unit integration e2e coverage fast-test
test:
	$(PYTHON) -m pytest $(TESTS_DIR)/ -v

unit:
	$(PYTEST_V) $(TESTS_DIR)/unit -v

integration:
	$(PYTEST_V) $(TESTS_DIR)/integration -v

e2e:
	$(PYTEST_V) $(TESTS_DIR)/e2e -v

coverage:
	$(PYTEST_V) $(TESTS_DIR)/ --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html

fast-test:
	$(PYTHON) -m pytest $(TESTS_DIR)/ -x -q --tb=short

# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------
.PHONY: lint format typecheck check
lint:
	$(RUFF) check $(SRC_DIR) $(SCRIPTS_DIR) $(TESTS_DIR)

format:
	$(RUFF) check $(SRC_DIR) $(SCRIPTS_DIR) $(TESTS_DIR) --fix
	$(RUFF) format $(SRC_DIR) $(SCRIPTS_DIR) $(TESTS_DIR)

typecheck:
	$(MYPY) src/state/ src/reporting/

# Everything CI runs, in one target
check: lint typecheck test

# ---------------------------------------------------------------------------
# Backtest / benchmark / validation
# ---------------------------------------------------------------------------
.PHONY: backtest benchmark-smoke preflight pipeline
backtest:
	$(PYTHON) $(SCRIPTS_DIR)/full_stack_walk_forward_backtest.py \
		--input $(DATA_DIR)/raw/BTCUSDT.csv --env paper

benchmark-smoke:
	$(PYTHON) $(SCRIPTS_DIR)/full_stack_walk_forward_backtest.py \
		--synthetic 900 --relax-liquidity --window-bars 300 \
		--report-dir $(REPORTS_DIR)/benchmark_$(BENCH_DATE)
	$(PYTHON) $(SCRIPTS_DIR)/full_stack_walk_forward_backtest.py \
		--synthetic 900 --relax-liquidity --window-bars 300 --single-expert trend_expert \
		--report-dir $(REPORTS_DIR)/benchmark_$(BENCH_DATE)
	$(PYTHON) $(SCRIPTS_DIR)/full_stack_walk_forward_backtest.py \
		--synthetic 900 --relax-liquidity --window-bars 300 --no-risk-layer \
		--report-dir $(REPORTS_DIR)/benchmark_$(BENCH_DATE)
	$(PYTHON) $(SCRIPTS_DIR)/full_stack_walk_forward_backtest.py \
		--synthetic 900 --relax-liquidity --window-bars 300 --no-regime-gating \
		--report-dir $(REPORTS_DIR)/benchmark_$(BENCH_DATE)
	@echo "Ablation ladder written to $(REPORTS_DIR)/benchmark_$(BENCH_DATE)/"

preflight:
	$(PYTHON) $(SCRIPTS_DIR)/preflight.py --env paper

pipeline:
	$(PYTHON) -m $(SRC_DIR).pipeline

# ---------------------------------------------------------------------------
# Observability tooling
# ---------------------------------------------------------------------------
.PHONY: dashboard dashboard-headless latency figures
dashboard:
	$(PYTHON) $(SCRIPTS_DIR)/dashboard.py

dashboard-headless:
	$(PYTHON) $(SCRIPTS_DIR)/dashboard.py --once

latency:
	$(PYTHON) $(SCRIPTS_DIR)/benchmark_latency.py --synthetic 300

figures:
	$(PYTHON) $(SCRIPTS_DIR)/generate_figures.py --synthetic 900 --relax-liquidity

demo:
	streamlit run $(SCRIPTS_DIR)/demo_app.py

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
.PHONY: build-image
build-image:
	docker compose build

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
.PHONY: install-dev
install-dev:
	python3 -m venv .venv || true
	.venv/bin/python -m ensurepip -q || true
	.venv/bin/python -m pip install -q --upgrade pip
	.venv/bin/python -m pip install -q -r requirements.txt
	.venv/bin/python -m pip install -q -e ".[dev]"
	@echo "Done. Use .venv/bin/python, or: source .venv/bin/activate"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
.PHONY: clean
clean:
	@find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ruff_cache" \
		-o -name ".mypy_cache" -o -name "htmlcov" \) -not -path "./.venv/*" \
		-exec rm -rf {} + 2>/dev/null || true
	@find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name ".coverage" \) \
		-not -path "./.venv/*" -delete 2>/dev/null || true
	@echo "Cleaned."
