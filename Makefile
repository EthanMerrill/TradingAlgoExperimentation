.PHONY: help install lint test run clean docker-build docker-run

help:
	@echo "Common Makefile targets:"
	@echo "  install      Install Python dependencies"
	@echo "  lint         Run flake8 linter"
	@echo "  test         Run all tests (pytest or custom runner)"
	@echo "  run          Run the main trading algorithm"
	@echo "  clean        Remove Python cache and logs"
	@echo "  docker-build Build the Docker image"
	@echo "  docker-run   Run the app in Docker"

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

run:
	.venv/bin/python app/main.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf app/logs/*

# Docker targets

docker-build:
	docker build -t trading-algo .

docker-run:
	docker run --rm -it -e PAPER_TRADE=True trading-algo
