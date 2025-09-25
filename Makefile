# Makefile for Customer Churn Prediction & CLV Project (Ubuntu Linux)

.PHONY: install check test data clv train interpret all app clean help setup-dirs download-data clean-all
.DEFAULT_GOAL := help

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# Project structure
DATA_DIR = data
RAW_DATA_DIR = $(DATA_DIR)/raw
PROCESSED_DATA_DIR = $(DATA_DIR)/processed
SRC_DIR = src
MODELS_DIR = models

install: ## Create venv and install dependencies
	@echo "Creating virtual environment and installing dependencies..."
	@if [ -d "$(VENV)" ]; then rm -rf $(VENV); fi
	@python3 -m venv $(VENV)
	@echo "Installing packages from requirements.txt..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt

check: ## Run code quality checks
	@echo "Linting with Ruff..."
	@$(PYTHON) -m ruff check .

test: ## Run unit tests
	@echo "Running tests with pytest..."
	@$(PYTHON) -m pytest tests/ -v

download-data: ## Download the dataset
	@echo "Downloading dataset..."
	@mkdir -p "$(RAW_DATA_DIR)"
	@wget -O "$(RAW_DATA_DIR)/Telco-Customer-Churn.csv" "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" 2>/dev/null || curl -o "$(RAW_DATA_DIR)/Telco-Customer-Churn.csv" "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
	@echo "Dataset downloaded to $(RAW_DATA_DIR)/"

data: download-data ## Run data preparation
	@echo "Running data preparation..."
	@$(PYTHON) $(SRC_DIR)/data_prep.py

clv: ## Run CLV analysis
	@echo "Running CLV analysis..."
	@$(PYTHON) $(SRC_DIR)/clv_analysis.py

train: ## Train models
	@echo "Training models..."
	@$(PYTHON) $(SRC_DIR)/train_models.py

interpret: ## Run interpretability
	@echo "Running interpretability..."
	@$(PYTHON) $(SRC_DIR)/interpretability.py

all: data clv train interpret ## Run full pipeline (data to interpretability)

app: ## Start Streamlit app
	@echo "Launching Streamlit app..."
	@export NUMBA_SVML_DEFAULT=intelsvml_b && export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false && export STREAMLIT_SERVER_HEADLESS=true && $(PYTHON) -m streamlit run app.py

setup-dirs: ## Create necessary directories
	@echo "Creating project directories..."
	@mkdir -p "$(RAW_DATA_DIR)"
	@mkdir -p "$(PROCESSED_DATA_DIR)"
	@mkdir -p "$(MODELS_DIR)"
	@mkdir -p "$(SRC_DIR)"
	@mkdir -p "tests"
	@echo "Directories created!"

clean: ## Remove temp files and artifacts
	@echo "Cleaning up artifacts..."
	-@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	-@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	-@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	-@find $(PROCESSED_DATA_DIR) -name "*.csv" -type f -delete 2>/dev/null || true
	-@find $(MODELS_DIR) -name "*.pkl" -type f -delete 2>/dev/null || true
	@echo "Cleanup completed!"

clean-all: clean ## Remove everything including virtual environment
	@echo "Removing virtual environment..."
	@if [ -d "$(VENV)" ]; then rm -rf $(VENV); fi
	@echo "Virtual environment removed!"

help: ## Show available Make targets
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m                    \033[0m \n", $$1, $$2}' $(MAKEFILE_LIST)
