import joblib
import pandas as pd
import os

# Define paths
MODELS_PATH = "models"
PROCESSED_DATA_PATH = "data/processed"

# --- Load Model and Columns ---
@pd.api.extensions.register_dataframe_accessor("safe_access")
class SafeAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __getitem__(self, key):
        try:
            return self._obj[key]
        except KeyError:
            return None

def load_model_and_cols(model_name='logisticregression'):
    """Loads a model and the columns used for training.""" 
    model = joblib.load(os.path.join(MODELS_PATH, f'{model_name}.pkl'))
    train_cols = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'X_train.csv')).columns.tolist()
    return model, train_cols

# --- Prediction Function (Corrected) ---
def make_prediction(input_data, model, train_cols):
    """Makes a churn prediction on a single data instance with robust preprocessing."""
    
    # Create a single-row DataFrame with all training columns, filled with zeros
    processed_df = pd.DataFrame(0, columns=train_cols, index=[0])

    # Update numerical features directly from input_data
    for feature in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        if feature in input_data:
            processed_df[feature] = input_data[feature]

    # Update one-hot encoded features based on input
    for key, value in input_data.items():
        # Construct the column name, e.g., 'Contract_One year'
        col_name = f"{key}_{value}"
        if col_name in processed_df.columns:
            processed_df[col_name] = 1

    # Ensure all columns are in the correct order
    processed_df = processed_df[train_cols]

    # Make prediction
    churn_prob = model.predict_proba(processed_df)[0, 1]

    # Get risk label
    if churn_prob < 0.3:
        risk_label = "Low Risk"
    elif churn_prob < 0.6:
        risk_label = "Medium Risk"
    else:
        risk_label = "High Risk"

    return churn_prob, risk_label