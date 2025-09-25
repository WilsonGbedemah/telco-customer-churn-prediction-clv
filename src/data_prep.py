import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import sys

# Add src directory to path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import src.config as config

def clean_data(df):
    """Handles missing values and corrects data types."""
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    print("Filled missing TotalCharges values.")
    return df

def engineer_features(df):
    """Creates new features to improve model performance."""
    df['tenure_bucket'] = pd.cut(df['tenure'], bins=config.TENURE_BINS, labels=config.TENURE_LABELS, right=False)
    df['services_count'] = df[config.SERVICE_COLUMNS].apply(lambda row: sum(row != 'No'), axis=1)
    df['monthly_to_total_ratio'] = df['TotalCharges'] / np.maximum(1, df['tenure'] * df['MonthlyCharges'])
    df['internet_but_no_tech_support'] = ((df['InternetService'] != 'No') & (df['TechSupport'] == 'No')).astype(int)
    print("Engineered new features.")
    return df

def encode_data(df):
    """Encodes categorical variables and separates features from target."""
    df.drop('customerID', axis=1, inplace=True)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols.remove('Churn')
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print("Encoded categorical variables.")
    return X_encoded, y_encoded

def main():
    """Main function to run the full data preparation pipeline."""
    os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)
    df = pd.read_csv(config.RAW_DATA_PATH)
    df = clean_data(df.copy())
    df = engineer_features(df)
    X_encoded, y_encoded = encode_data(df)

    print("\n--- Splitting Data ---")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_encoded, y_encoded, 
        test_size=config.TEST_SIZE_1, 
        random_state=config.RANDOM_STATE, 
        stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=config.TEST_SIZE_2, 
        random_state=config.RANDOM_STATE, 
        stratify=y_temp
    )
    print("Data split into 60/20/20 train/val/test sets.")

    print("\n--- Saving Processed Data ---")
    pd.DataFrame(X_train).to_csv(os.path.join(config.PROCESSED_DATA_PATH, 'X_train.csv'), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(config.PROCESSED_DATA_PATH, 'y_train.csv'), index=False)
    pd.DataFrame(X_val).to_csv(os.path.join(config.PROCESSED_DATA_PATH, 'X_val.csv'), index=False)
    pd.DataFrame(y_val).to_csv(os.path.join(config.PROCESSED_DATA_PATH, 'y_val.csv'), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(config.PROCESSED_DATA_PATH, 'X_test.csv'), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(config.PROCESSED_DATA_PATH, 'y_test.csv'), index=False)
    print("Data preparation complete.")

if __name__ == "__main__":
    main()