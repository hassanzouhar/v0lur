.PHONY: help setup download-models run test lint security-audit eval calibrate clean clean-models

# Default target
help:
	@echo "Available targets:"
	@echo "  setup          - Set up development environment with pyenv and dependencies"
	@echo "  download-models - Pre-download ML models to avoid runtime delays"
	@echo "  run            - Run analysis with default sample data"
	@echo "  test           - Run test suite with coverage"
	@echo "  lint           - Run linting and formatting (ruff, black)"
	@echo "  security-audit - Run security audits (pip-audit, bandit)"
	@echo "  eval           - Run evaluation on gold standard dataset"
	@echo "  calibrate      - Generate calibration report"
	@echo "  clean          - Remove outputs and temporary files"
	@echo "  clean-models   - Remove downloaded model cache"

# Environment setup
setup:
	@echo "Setting up development environment..."
	pyenv install 3.11.9 -s
	pyenv local 3.11.9
	pyenv exec python -m venv .venv
	.venv/bin/pip install -U pip pip-tools pip-audit
	@if [ -f requirements.txt ]; then \
		.venv/bin/pip install -r requirements.txt; \
	else \
		echo "Warning: requirements.txt not found. Run 'pip-compile requirements.in' to generate it."; \
	fi
	.venv/bin/python -m spacy download en_core_web_sm
	@echo "Environment setup complete. Activate with: source .venv/bin/activate"

# Download models
download-models:
	@echo "Pre-downloading ML models..."
	.venv/bin/python -c "from transformers import pipeline; pipeline('ner', 'dslim/bert-base-NER')"
	.venv/bin/python -c "from transformers import pipeline; pipeline('sentiment-analysis', 'cardiffnlp/twitter-roberta-base-sentiment-latest')"
	.venv/bin/python -c "from transformers import pipeline; pipeline('text-classification', 'unitary/toxic-bert')"
	.venv/bin/python -c "from transformers import pipeline; pipeline('zero-shot-classification', 'facebook/bart-large-mnli')"
	@echo "Models downloaded successfully"

# Run analysis
run:
	@export CLANG=1; unset CUDA; unset GPU; \
	.venv/bin/python telegram_analyzer.py \
		--config config/config.yaml \
		--input $(or $(INPUT),data/sample_channel.json) \
		--out out/run-$$(date +%Y%m%d-%H%M)

# Testing
test:
	.venv/bin/pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Linting
lint:
	.venv/bin/ruff check src/ tests/
	.venv/bin/ruff format src/ tests/
	.venv/bin/black src/ tests/

# Security audit
security-audit:
	.venv/bin/pip-audit --req requirements.txt --format json --output security-audit.json
	@if command -v bandit >/dev/null 2>&1; then \
		.venv/bin/bandit -r src/ -f json -o bandit-report.json; \
	fi

# Evaluation
eval:
	@if [ -f data/gold.csv ]; then \
		.venv/bin/python scripts/eval.py \
			--gold data/gold.csv \
			--pred out/run-*/posts_enriched.parquet \
			--metrics stance entity attribution; \
	else \
		echo "Error: Gold standard dataset not found at data/gold.csv"; \
		echo "Create it first with: python scripts/create_gold_set.py"; \
	fi

# Calibration
calibrate:
	@if [ -f eval_output.json ]; then \
		.venv/bin/python scripts/calibrate.py \
			--eval-results eval_output.json \
			--config config/config.yaml; \
	else \
		echo "Error: Run 'make eval' first to generate evaluation results"; \
	fi

# Cleanup
clean:
	rm -rf out/run-*
	rm -rf __pycache__ .pytest_cache
	rm -f *.log security-audit.json bandit-report.json
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

clean-models:
	@echo "Removing model cache..."
	rm -rf ~/.cache/huggingface/transformers/
	rm -rf ~/.cache/huggingface/datasets/
	@echo "Model cache cleared"

# Development helpers
requirements:
	.venv/bin/pip-compile --upgrade requirements.in
	.venv/bin/pip-audit --req requirements.txt

install-dev:
	.venv/bin/pip install pre-commit ruff black mypy bandit pytest pytest-cov
	.venv/bin/pre-commit install

# CI targets
ci-setup: setup install-dev
ci-test: lint security-audit test