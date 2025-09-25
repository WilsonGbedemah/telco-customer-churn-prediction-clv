import pandas as pd
import joblib
import os
import sys
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Add src directory to path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import src.config as config

# Create models directory if it doesn't exist
os.makedirs(config.MODELS_PATH, exist_ok=True)

# Load data
print("--- Loading Data ---")
X_train = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'X_train.csv'))
y_train = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'y_train.csv')).values.ravel()
X_val = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'X_val.csv'))
y_val = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'y_val.csv')).values.ravel()
X_test = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'y_test.csv')).values.ravel()

# Combine train and validation sets for final training
X_train_full = pd.concat([X_train, X_val], axis=0)
y_train_full = pd.concat([pd.Series(y_train), pd.Series(y_val)], axis=0)
print("Data loaded and prepared for training.")

# Calculate scale_pos_weight for XGBoost to handle class imbalance
counts = Counter(y_train)
scale_pos_weight = counts[0] / counts[1]
print(f"Calculated scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")

# Get models and params from config
models = config.MODELS
params = config.PARAMS
# Update XGBoost with the calculated scale_pos_weight
models['XGBoost'].set_params(scale_pos_weight=scale_pos_weight)

# Train, tune, and evaluate models
print("\n--- Training and Evaluating Final Models ---")
for name, model in models.items():
    print(f"\nTraining {name}...")
    grid = GridSearchCV(model, params[name], cv=3, scoring='roc_auc')
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"Best parameters for {name}: {grid.best_params_}")

    best_model.fit(X_train_full, y_train_full)

    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_pred_proba >= 0.5).astype(int)

    precision = precision_score(y_test, y_pred_default)
    recall = recall_score(y_test, y_pred_default)
    f1 = f1_score(y_test, y_pred_default)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n{name} Evaluation on Test Set (Default 0.5 Threshold):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")

    joblib.dump(best_model, os.path.join(config.MODELS_PATH, f'{name.lower()}.pkl'))
    print(f"Saved {name} model to {config.MODELS_PATH}/{name.lower()}.pkl")
