import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- Project Paths ---
# Using os.path.join for cross-platform compatibility
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Telco-Customer-Churn.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")

# --- Data Preparation ---
RANDOM_STATE = 42
TEST_SIZE_1 = 0.4  # For the first split (train vs. temp)
TEST_SIZE_2 = 0.5  # For the second split (val vs. test)

# --- Feature Engineering ---
TENURE_BINS = [0, 6, 12, 24, 72]
TENURE_LABELS = ['0-6m', '6-12m', '12-24m', '24m+']
SERVICE_COLUMNS = [
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies'
]

# --- Model Training ---
# Define models and hyperparameters with class weighting
MODELS = {
    'LogisticRegression': LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced', random_state=RANDOM_STATE),
    'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
    'XGBoost': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
}

PARAMS = {
    'LogisticRegression': {'C': [0.1, 1, 10]},
    'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, 20]},
    'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]}
}
