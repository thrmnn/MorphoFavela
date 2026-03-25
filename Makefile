.DEFAULT_GOAL := help
AREA ?= vidigal_tls

.PHONY: help test test-fast lint format svf morphology pipeline report cross-cluster clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

test: ## Run pytest (-x -q)
	python -m pytest -x -q

test-fast: ## Run only fast (synthetic-geometry) tests
	python -m pytest -x -q -m fast

lint: ## Lint with ruff
	ruff check src/ tests/

format: ## Format with ruff
	ruff format src/ tests/

svf: ## Run SVF computation (AREA=rocinha)
	python scripts/run_svf_v2.py --area $(AREA) --mode all

morphology: ## Run morphology metrics (AREA=rocinha)
	python scripts/calculate_morphology_metrics.py --area $(AREA)

pipeline: ## Run full pipeline for an area (AREA=rocinha)
	python scripts/run_svf_v2.py --area $(AREA) --mode all
	python scripts/calculate_morphology_metrics.py --area $(AREA)

report: ## Generate report for an area (AREA=rocinha)
	python scripts/generate_report.py --area $(AREA) --format both

cross-cluster: ## Run cross-area clustering
	python scripts/run_cross_area_clustering.py

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name '*.egg-info' -exec rm -rf {} +
