import joblib
import pandas as pd
import numpy as np
import os

# Define paths
MODELS_PATH = "models"
PROCESSED_DATA_PATH = "data/processed"

# Load models
print("---" + "-" * 10 + " Loading Models " + "-" * 10 + "---")
lr_model = joblib.load(os.path.join(MODELS_PATH, 'logisticregression.pkl'))
rf_model = joblib.load(os.path.join(MODELS_PATH, 'randomforest.pkl'))
xgb_model = joblib.load(os.path.join(MODELS_PATH, 'xgboost.pkl'))
print("Models loaded.")

# Load training data for feature names and std dev
X_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'X_train.csv'))

# --- Logistic Regression Feature Importance ---
print("\n" + "---" + "-" * 10 + " Calculating Logistic Regression Feature Importance " + "-" * 10 + "---")
# Get coefficients
coefficients = lr_model.coef_[0]

# Calculate standard deviation of features
std_devs = X_train.std()

# Calculate importance
feature_importance_lr = pd.DataFrame({
    'feature': X_train.columns,
    'importance': np.abs(coefficients * std_devs)
}).sort_values(by='importance', ascending=False)

# Save importance
feature_importance_lr.to_csv(os.path.join(MODELS_PATH, 'lr_feature_importance.csv'), index=False)
print("Saved Logistic Regression feature importance to models/lr_feature_importance.csv")

# --- Random Forest Feature Importance (Fallback) ---
print("\n" + "---" + "-" * 10 + " Calculating Random Forest Feature Importance " + "-" * 10 + "---")
feature_importance_rf = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values(by='importance', ascending=False)

# Save importance
feature_importance_rf.to_csv(os.path.join(MODELS_PATH, 'rf_feature_importance.csv'), index=False)
print("Saved Random Forest feature importance to models/rf_feature_importance.csv")

# --- XGBoost Feature Importance (Fallback) ---
print("\n" + "---" + "-" * 10 + " Calculating XGBoost Feature Importance " + "-" * 10 + "---")
feature_importance_xgb = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values(by='importance', ascending=False)

# Save importance
feature_importance_xgb.to_csv(os.path.join(MODELS_PATH, 'xgb_feature_importance.csv'), index=False)
print("Saved XGBoost feature importance to models/xgb_feature_importance.csv")

print("\nInterpretability analysis complete.")