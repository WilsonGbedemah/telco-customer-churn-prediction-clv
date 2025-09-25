# Telco Customer Churn Prediction & CLV Analysis

This project provides an end-to-end solution for predicting customer churn and analyzing customer lifetime value (CLV) for a fictional telecommunications company. The final output is an interactive Streamlit web application that allows users to get on-demand churn predictions, understand the key drivers of churn, and view analysis on customer value.

## Business Problem

Customer churn is a critical issue for subscription-based businesses, leading to significant revenue loss. This project addresses two key business questions:
1.  **Who is likely to churn?** By building a predictive machine learning model.
2.  **Which customers are most valuable to retain?** By calculating and analyzing Customer Lifetime Value.

The goal is to empower the business to proactively identify high-value customers who are at risk of churning and to target them with effective retention strategies.

## Solution Overview

The solution is an interactive Streamlit application with three main functionalities:

*   **Churn Prediction:** Get a real-time churn probability for any customer profile using our trained machine learning model.
*   **Model Performance:** Review the performance metrics (Precision, Recall, AUC-ROC) of the different models and explore the key features that influence their predictions.
*   **CLV Overview:** View an analysis of how customer value relates to churn rate, providing insights into which customer segments are most critical to the business.

## CLV Assumptions

To calculate Customer Lifetime Value, we made a key simplifying assumption:

*   **`ExpectedTenure`**: This was calculated as the average tenure of all non-churning customers in our training dataset. This provides a stable estimate of how long a loyal customer is expected to stay with the company.

## Final Model Performance

The final, recommended model is a **Logistic Regression** classifier, which was optimized to maximize **Recall**—our primary business metric for ensuring we identify the highest number of at-risk customers.

The model was tuned to a specific decision threshold to meet the business requirements, achieving the following final performance on the unseen test data:

*   **Optimal Decision Threshold:** 0.580
*   **Precision:** 54.8%
*   **Recall:** 74.9%
*   **AUC-ROC:** 83.7%

## System Requirements

- **Operating System:** Ubuntu Linux (or other Linux distributions)
- **Python:** 3.12+ (automatically managed by virtual environment)
- **Dependencies:** All Python packages are listed in `requirements.txt`
- **Tools:** git, wget/curl (for data download), make

## Quick Start

Get the project running in 3 simple commands:

```bash
git clone https://github.com/WilsonGbedemah/telco-customer-churn-prediction-clv.git
cd telco-customer-churn-prediction-clv
make install && make all && make app
```

The application will be available at `http://localhost:8501`

## Detailed Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/WilsonGbedemah/telco-customer-churn-prediction-clv.git
cd telco-customer-churn-prediction-clv
```

### 2. Set Up Environment and Install Dependencies
This project uses `make` to streamline setup. The following command will create a virtual environment and install all required packages from `requirements.txt`:
```bash
make install
```

### 3. Run the Complete Data and Modeling Pipeline
The `Makefile` includes a command to run all the necessary scripts in order, from data preparation to model training:
```bash
make all
```
This command will:
- Download the dataset automatically
- Prepare and clean the data
- Run the CLV analysis and generate visualizations
- Train and evaluate all machine learning models
- Generate feature importance analysis

### 4. Launch the Interactive Streamlit Application
Once the pipeline has been run, you can launch the interactive web application:
```bash
make app
```
The Streamlit server will start and display a local URL (usually `http://localhost:8501`) where you can access the application.

## Available Make Commands

The project includes a comprehensive Makefile with the following commands:

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands with descriptions |
| `make install` | Create virtual environment and install dependencies |
| `make check` | Run code quality checks (Ruff linting) |
| `make test` | Run unit tests with pytest |
| `make download-data` | Download the Telco Customer Churn dataset |
| `make data` | Run data preparation (includes download) |
| `make clv` | Run CLV analysis and generate plots |
| `make train` | Train and evaluate all machine learning models |
| `make interpret` | Generate feature importance analysis |
| `make all` | Run complete pipeline (data → clv → train → interpret) |
| `make app` | Launch Streamlit web application |
| `make clean` | Remove temporary files and artifacts |
| `make clean-all` | Remove everything including virtual environment |
| `make setup-dirs` | Create necessary project directories |

## Project Structure

```
telco-customer-churn-prediction-clv/
├── Makefile                    # Build automation and task runner
├── requirements.txt            # Python dependencies
├── pytest.ini                 # Test configuration
├── app.py                      # Streamlit web application
├── README.md                   # This file
├── AI_USAGE.md                # Documentation of AI assistance
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned and split data
├── src/                        # Source code modules
│   ├── config.py              # Project configuration and parameters
│   ├── data_prep.py           # Data cleaning and feature engineering
│   ├── clv_analysis.py        # Customer Lifetime Value analysis
│   ├── train_models.py        # Model training and evaluation
│   ├── predict.py             # Prediction utilities
│   ├── interpretability.py    # Feature importance analysis
│   └── importance_utils.py    # Utility functions for interpretability
├── models/                     # Trained models and artifacts
│   ├── *.pkl                  # Serialized ML models
│   ├── *_feature_importance.csv # Feature importance scores
│   └── clv_avg_tenure.txt     # CLV calculation parameters
├── tests/                      # Unit tests
│   ├── test_data_prep.py      # Tests for data preparation
│   ├── test_clv_analysis.py   # Tests for CLV analysis
│   └── test_predict.py        # Tests for prediction functions
└── *.png                       # Generated visualizations
```

## Development and Quality Assurance

This project maintains high code quality through:

- **Linting:** Ruff for code style and quality checks
- **Testing:** Comprehensive unit tests with pytest
- **Type Safety:** Python type hints throughout the codebase
- **Documentation:** Extensive inline documentation and README
- **Automation:** Make-based workflow for reproducible builds

## Troubleshooting

### Common Issues and Solutions

**Issue:** `make: command not found`
- **Solution:** Install make: `sudo apt-get install make` (Ubuntu/Debian)

**Issue:** `python3: command not found`
- **Solution:** Install Python 3.12+: `sudo apt-get install python3 python3-pip python3-venv`

**Issue:** Data download fails
- **Solution:** Ensure wget or curl is installed: `sudo apt-get install wget curl`

**Issue:** Virtual environment creation fails
- **Solution:** Install python3-venv: `sudo apt-get install python3-venv`

**Issue:** Tests fail with import errors
- **Solution:** Ensure you've run `make install` first, then `make test`

**Issue:** Streamlit app won't start
- **Solution:** Ensure the full pipeline has run: `make all` then `make app`

### Getting Help

If you encounter issues:
1. Check that all system requirements are met
2. Run `make clean-all && make install` to reset the environment  
3. Check the error messages - they often contain helpful information
4. Ensure you're in the project root directory when running make commands

## Contributing

This project was developed with significant AI assistance. See `AI_USAGE.md` for details on how AI tools were used in the development process.

## License

See `LICENSE` file for details.