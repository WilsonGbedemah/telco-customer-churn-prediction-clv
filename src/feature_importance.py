import pandas as pd
import joblib
import os
import numpy as np
import sys

# Add src directory to path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import src.config as config

# Load data
print("--- Loading Data for Feature Importance ---")
X_train = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'X_train.csv'))
X_val = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'X_val.csv'))
X_train_full = pd.concat([X_train, X_val], axis=0)

# Load models
print("--- Loading Models ---")
lr_model = joblib.load(os.path.join(config.MODELS_PATH, 'logisticregression.pkl'))
rf_model = joblib.load(os.path.join(config.MODELS_PATH, 'randomforest.pkl'))
xgb_model = joblib.load(os.path.join(config.MODELS_PATH, 'xgboost.pkl'))

# --- Logistic Regression Feature Importance ---
print("\n--- Calculating Logistic Regression Feature Importance ---")
std_devs = X_train_full.std()
std_devs[std_devs == 0] = 1
abs_coeffs = np.abs(lr_model.coef_[0])
feature_importance_lr = abs_coeffs * std_devs
lr_importance_df = pd.DataFrame({
    'feature': X_train_full.columns,
    'importance': feature_importance_lr
}).sort_values(by='importance', ascending=False)
lr_importance_df.to_csv(os.path.join(config.MODELS_PATH, 'lr_feature_importance.csv'), index=False)
print("Top 10 Features for Logistic Regression:")
print(lr_importance_df.head(10))

# --- Random Forest Feature Importance ---
print("\n--- Calculating Random Forest Feature Importance ---")
feature_importance_rf = rf_model.feature_importances_
rf_importance_df = pd.DataFrame({
    'feature': X_train_full.columns,
    'importance': feature_importance_rf
}).sort_values(by='importance', ascending=False)
rf_importance_df.to_csv(os.path.join(config.MODELS_PATH, 'rf_feature_importance.csv'), index=False)
print("\nTop 10 Features for Random Forest:")
print(rf_importance_df.head(10))

# --- XGBoost Feature Importance ---
print("\n--- Calculating XGBoost Feature Importance ---")
feature_importance_xgb = xgb_model.feature_importances_
xgb_importance_df = pd.DataFrame({
    'feature': X_train_full.columns,
    'importance': feature_importance_xgb
}).sort_values(by='importance', ascending=False)
xgb_importance_df.to_csv(os.path.join(config.MODELS_PATH, 'xgb_feature_importance.csv'), index=False)
print("\nTop 10 Features for XGBoost:")
print(xgb_importance_df.head(10))