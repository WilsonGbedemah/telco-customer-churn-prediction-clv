
import pytest
from src.predict import make_prediction

import numpy as np

# Mock model class for testing
class MockModel:
    def predict_proba(self, df):
        # Return a fixed probability for testing as a numpy array
        return np.array([[0.2, 0.8]])

@pytest.fixture
def sample_input_data():
    """Fixture for sample single-customer input data."""
    return {
        'tenure': 12,
        'MonthlyCharges': 50.0,
        'TotalCharges': 600.0,
        'Contract': 'One year',
        'InternetService': 'DSL'
    }

@pytest.fixture
def train_cols():
    """Fixture for sample training columns."""
    # A subset of columns for simplicity
    return [
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'Contract_One year', 'Contract_Two year',
        'InternetService_DSL', 'InternetService_Fiber optic'
    ]

def test_make_prediction(sample_input_data, train_cols):
    """Test the make_prediction function."""
    mock_model = MockModel()
    
    churn_prob, risk_label = make_prediction(sample_input_data, mock_model, train_cols)
    
    # Check that the probability is a float between 0 and 1
    assert isinstance(churn_prob, float)
    assert 0.0 <= churn_prob <= 1.0
    
    # Check that the risk label is one of the expected values
    assert risk_label in ["Low Risk", "Medium Risk", "High Risk"]
    
    # Check the logic for the risk label based on the mock probability
    assert risk_label == "High Risk" # Since mock prob is 0.8
