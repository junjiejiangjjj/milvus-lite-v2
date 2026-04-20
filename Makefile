PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin

.PHONY: venv install dev test test-all benchmark lint clean

# Create virtual environment
venv:
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created at $(VENV)"
	@echo "Run: source $(VENV)/bin/activate"

# Install package with all dependencies
install: venv
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e .

# Install with dev dependencies
dev: venv
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

# Run tests (excludes benchmark, real-time output)
test:
	$(BIN)/pytest --tb=short -v

# Run all tests including benchmark
test-all:
	$(BIN)/pytest --tb=short -q tests/
	$(BIN)/pytest tests/benchmark/ -v -s

# Run benchmark only
benchmark:
	$(BIN)/pytest tests/benchmark/ -v -s

# Run tests with coverage
coverage:
	$(BIN)/pytest --cov=milvus_lite --cov-report=term-missing -q

# Start gRPC server
serve:
	$(BIN)/milvus-lite-server --data-dir ./data --port 19530

# Clean up
clean:
	rm -rf $(VENV) .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
