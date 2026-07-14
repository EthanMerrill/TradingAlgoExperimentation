.PHONY: help install lint test test-integration run clean docker-build docker-run

help:
	@echo "Common Makefile targets:"
	@echo "  install          Install Python dependencies"
	@echo "  lint             Run flake8 linter"
	@echo "  test             Run all tests (pytest or custom runner)"
	@echo "  test-integration Run live Alpaca order integration tests"
	@echo "  run              Run the main trading algorithm"
	@echo "  clean            Remove Python cache and logs"
	@echo "  docker-build     Build the Docker image"
	@echo "  docker-run       Run the app in Docker"

install:
	pip install -r requirements.txt

lint:
	flake8 app/ --max-line-length=120

test:
	# Prefer pytest, fallback to custom runner if needed
	@if [ -f tests/run_tests.py ]; then \
		python tests/run_tests.py; \
	else \
		pytest; \
	fi

test-integration:
	# Live Alpaca paper-trading integration tests (requires .env with ALPACA_DEV_PAPER_KEY/SECRET)
	@if [ -f .env ]; then \
		export $$(grep -v '^#' .env | grep ALPACA_DEV_PAPER | xargs); \
	fi; \
	.venv/bin/python -m pytest tests/test_order_integration.py -v

run:
	python -m app.main

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf app/logs/* logs/* 2>/dev/null || true

# Docker targets

docker-build:
	docker build -t trading-algo .

docker-run:
	docker run --rm -it -e PAPER_TRADE=True trading-algo
