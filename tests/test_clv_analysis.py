
import pytest
import pandas as pd
import os
from src.clv_analysis import analyze_clv

@pytest.fixture
def create_dummy_data(tmp_path):
    """Create dummy data files required for the CLV analysis script."""
    # Create directories
    raw_path = tmp_path / "raw"
    processed_path = tmp_path / "processed"
    raw_path.mkdir()
    processed_path.mkdir()

    # Create dummy raw data
    raw_data = {
        'tenure': [10, 20],
        'MonthlyCharges': [50, 80],
        'TotalCharges': [500, 1600]
    }
    pd.DataFrame(raw_data).to_csv(raw_path / "Telco-Customer-Churn.csv", index=False)

    # Create dummy processed data
    x_train_data = {'feature1': [1, 2]}
    y_train_data = {'Churn': [0, 1]}
    pd.DataFrame(x_train_data).to_csv(processed_path / "X_train.csv", index=False)
    pd.DataFrame(y_train_data).to_csv(processed_path / "y_train.csv", index=False)
    
    return str(raw_path.parent)

def test_clv_analysis_smoke(monkeypatch, create_dummy_data):
    """Smoke test to ensure the clv_analysis script runs without errors."""
    
    # Use the temporary directory created by the fixture
    temp_dir = create_dummy_data
    
    # Monkeypatch the path variables in the config module to use the temp data
    monkeypatch.setattr('src.config.RAW_DATA_PATH', os.path.join(temp_dir, 'raw', 'Telco-Customer-Churn.csv'))
    monkeypatch.setattr('src.config.PROCESSED_DATA_PATH', os.path.join(temp_dir, 'processed'))
    monkeypatch.setattr('src.config.MODELS_PATH', temp_dir)  # Use temp_dir as models path
    
    # Monkeypatch the savefig function to avoid saving a plot during tests
    monkeypatch.setattr('matplotlib.pyplot.savefig', lambda x: None)

    try:
        analyze_clv()
    except Exception as e:
        pytest.fail(f"analyze_clv() raised an exception: {e}")
