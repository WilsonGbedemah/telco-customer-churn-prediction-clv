
import pandas as pd
import numpy as np
import pytest
from src.data_prep import (
    clean_data,
    engineer_features,
    encode_data
)

@pytest.fixture
def raw_df():
    """Fixture to create a sample raw DataFrame for testing."""
    data = {
        'customerID': ['1234-ABCDE', '5678-FGHIJ'],
        'gender': ['Female', 'Male'],
        'Partner': ['Yes', 'No'],
        'Dependents': ['No', 'No'],
        'tenure': [1, 10],
        'PhoneService': ['No', 'Yes'],
        'MultipleLines': ['No phone service', 'No'],
        'InternetService': ['DSL', 'Fiber optic'],
        'OnlineSecurity': ['No', 'Yes'],
        'OnlineBackup': ['Yes', 'No'],
        'DeviceProtection': ['No', 'No'],
        'TechSupport': ['No', 'No'],
        'StreamingTV': ['No', 'No'],
        'StreamingMovies': ['No', 'No'],
        'Contract': ['Month-to-month', 'One year'],
        'PaperlessBilling': ['Yes', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check'],
        'MonthlyCharges': [29.85, 70.70],
        'TotalCharges': ['29.85', '700.70'],
        'Churn': ['No', 'Yes']
    }
    return pd.DataFrame(data)

def test_clean_data(raw_df):
    """Test the clean_data function."""
    cleaned_df = clean_data(raw_df)
    # Check that TotalCharges is numeric and missing values are handled
    assert pd.api.types.is_numeric_dtype(cleaned_df['TotalCharges'])
    assert cleaned_df['TotalCharges'].isnull().sum() == 0

def test_engineer_features(raw_df):
    """Test the engineer_features function."""
    # First, clean the data as it's a prerequisite
    df = clean_data(raw_df)
    featured_df = engineer_features(df)
    
    # Check if new columns are created
    assert 'tenure_bucket' in featured_df.columns
    assert 'services_count' in featured_df.columns
    assert 'monthly_to_total_ratio' in featured_df.columns
    assert 'internet_but_no_tech_support' in featured_df.columns
    
    # Check a value for one of the engineered features
    assert featured_df.loc[0, 'services_count'] == 3  # MultipleLines='No phone service', InternetService='DSL', OnlineBackup='Yes'
    assert featured_df.loc[1, 'internet_but_no_tech_support'] == 1  # InternetService='Fiber optic' != 'No' AND TechSupport='No'

def test_encode_data(raw_df):
    """Test the encode_data function."""
    # Clean and feature engineer the data first
    df = clean_data(raw_df)
    df = engineer_features(df)
    
    X_encoded, y_encoded = encode_data(df)
    
    # Check that the output is a pandas DataFrame and a numpy array
    assert isinstance(X_encoded, pd.DataFrame)
    assert isinstance(y_encoded, np.ndarray)
    
    # Check that 'Churn' is not in the features DataFrame
    assert 'Churn' not in X_encoded.columns
    
    # Check that categorical columns have been one-hot encoded
    assert 'gender_Male' in X_encoded.columns
    assert 'Contract_One year' in X_encoded.columns
    
    # Check that the target variable is label encoded (0s and 1s)
    assert all(y in [0, 1] for y in y_encoded)

