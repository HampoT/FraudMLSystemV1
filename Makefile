.PHONY: install install-dev lint test coverage train api dashboard batch-predict evaluate pipeline clean dev

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
NC = \033[0m # No Color

# Default Python executable
PYTHON = python3

# Install dependencies
install:
	$(PYTHON) -m pip install -r requirements.txt

# Install development dependencies
install-dev:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install black ruff pytest-cov

# Run linter
lint:
	@echo "$(YELLOW)Running linter...$(NC)"
	ruff check src/ tests/
	black --check src/ tests/

# Run formatter
format:
	@echo "$(YELLOW)Running formatter...$(NC)"
	black src/ tests/
	ruff check --fix src/ tests/

# Run tests
test:
	@echo "$(YELLOW)Running tests...$(NC)"
	pytest tests/ -v

# Run tests with coverage
coverage:
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	pytest tests/ --cov=src/fraudml --cov-report=term-missing --cov-report=html

# Run tests with integration flag
test-integration:
	@echo "$(YELLOW)Running integration tests...$(NC)"
	pytest tests/ -v --integration

# Train model
train:
	@echo "$(GREEN)Training model...$(NC)"
	$(PYTHON) src/fraudml/data/download.py
	$(PYTHON) src/fraudml/data/preprocess.py
	$(PYTHON) src/fraudml/models/train.py

# Train with specific model
train-lr:
	$(PYTHON) src/fraudml/models/train.py --model LogisticRegression

train-rf:
	$(PYTHON) src/fraudml/models/train.py --model RandomForest

train-xgb:
	$(PYTHON) src/fraudml/models/train.py --model XGBoost

# Train all models and compare
train-all:
	@echo "$(GREEN)Training all models...$(NC)"
	$(PYTHON) src/fraudml/models/train.py --model LogisticRegression
	$(PYTHON) src/fraudml/models/train.py --model RandomForest
	$(PYTHON) src/fraudml/models/train.py --model XGBoost
	$(PYTHON) src/fraudml/models/compare.py

# Evaluate model
evaluate:
	@echo "$(GREEN)Evaluating model...$(NC)"
	$(PYTHON) src/fraudml/models/evaluate.py

# Run batch predictions
batch-predict:
	@echo "$(GREEN)Running batch predictions...$(NC)"
	$(PYTHON) src/fraudml/scripts/batch_predict.py

# Start API server
api:
	@echo "$(GREEN)Starting API server...$(NC)"
	uvicorn src.fraudml.api.app:app --reload --host 0.0.0.0 --port 8000

# Start dashboard
dashboard:
	@echo "$(GREEN)Starting dashboard...$(NC)"
	streamlit run src/fraudml/ui/dashboard.py

# Start all services (dev mode)
dev: api dashboard
	@echo "$(GREEN)Started API and Dashboard$(NC)"

# Full ML pipeline
pipeline: train evaluate
	@echo "$(GREEN)Full pipeline complete!$(NC)"

# Data drift detection
drift-check:
	@echo "$(YELLOW)Checking for data drift...$(NC)"
	$(PYTHON) src/fraudml/monitoring/drift.py

# SHAP analysis
shap-analysis:
	@echo "$(YELLOW)Running SHAP analysis...$(NC)"
	$(PYTHON) src/fraudml/scripts/shap_analysis.py

# Model comparison report
compare-models:
	@echo "$(YELLOW)Comparing models...$(NC)"
	$(PYTHON) src/fraudml/models/compare.py

# Clean artifacts
clean:
	@echo "$(RED)Cleaning artifacts...$(NC)"
	rm -rf artifacts/*
	rm -rf data/*
	rm -rf reports/*
	rm -rf tests/data/*
	rm -rf tests/artifacts/*

# Setup development environment
setup: install-dev
	@echo "$(GREEN)Development environment ready!$(NC)"

# Show help
help:
	@echo "Fraud Detection ML System - Available Commands:"
	@echo ""
	@echo "  make install-dev    - Install all dependencies"
	@echo "  make lint           - Run linter and formatter check"
	@echo "  make format         - Auto-format code"
	@echo "  make test           - Run all tests"
	@echo "  make coverage       - Run tests with coverage report"
	@echo "  make train          - Train model (full pipeline)"
	@echo "  make train-lr       - Train Logistic Regression"
	@echo "  make train-rf       - Train Random Forest"
	@echo "  make train-xgb      - Train XGBoost"
	@echo "  make train-all      - Train all models and compare"
	@echo "  make evaluate       - Evaluate model"
	@echo "  make api            - Start API server"
	@echo "  make dashboard      - Start Streamlit dashboard"
	@echo "  make dev            - Start API and dashboard"
	@echo "  make pipeline       - Full ML pipeline"
	@echo "  make drift-check    - Check for data drift"
	@echo "  make shap-analysis  - Run SHAP analysis"
	@echo "  make compare-models - Compare model performance"
	@echo "  make clean          - Remove all artifacts"
	@echo "  make setup          - Setup development environment"
	@echo "  make help           - Show this help message"
