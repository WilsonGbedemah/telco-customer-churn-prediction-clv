import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add src directory to path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import src.config as config

def analyze_clv():
    """Performs CLV analysis and generates plots and insights."""
    try:
        # We need the original, non-encoded data to get tenure and monthly charges
        df_raw = pd.read_csv(config.RAW_DATA_PATH)
        y_train = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'y_train.csv'))
        # Use the index from the processed training data to select the correct raw rows
        df_train_raw = df_raw.loc[y_train.index]
        df_train_raw['Churn'] = y_train.values
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # Define Expected Tenure
    average_tenure_non_churn = df_train_raw[df_train_raw['Churn'] == 0]['tenure'].mean()
    print(f"Assumption: Expected Tenure for all customers is {average_tenure_non_churn:.2f} months.")
    
    # Save the average tenure to be used by the app
    with open(os.path.join(config.MODELS_PATH, 'clv_avg_tenure.txt'), 'w') as f:
        f.write(str(average_tenure_non_churn))

    df_train_raw['ExpectedTenure'] = average_tenure_non_churn

    # Calculate CLV
    # Ensure TotalCharges is numeric before calculating CLV
    df_train_raw['TotalCharges'] = pd.to_numeric(df_train_raw['TotalCharges'], errors='coerce').fillna(0)
    df_train_raw['CLV'] = df_train_raw['MonthlyCharges'] * df_train_raw['ExpectedTenure']

    # --- Generate Churn by Quartile Plot ---
    df_train_raw['CLV_Quartile'] = pd.qcut(df_train_raw['CLV'], q=4, labels=['Low', 'Med', 'High', 'Premium'])
    clv_churn_rate = df_train_raw.groupby('CLV_Quartile', observed=False)['Churn'].value_counts(normalize=True).unstack().fillna(0)
    clv_churn_rate['churn_rate'] = clv_churn_rate.get(1, 0)

    plt.figure(figsize=(10, 6))
    clv_churn_rate['churn_rate'].plot(kind='bar', color='skyblue')
    plt.title('Churn Rate by CLV Quartile')
    plt.ylabel('Churn Rate')
    plt.xlabel('CLV Quartile')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--')
    plt.savefig('clv_churn_rate.png')
    print("Saved churn rate by CLV quartile plot to clv_churn_rate.png")
    plt.close()

    # --- Generate CLV Distribution Plot ---
    plt.figure(figsize=(10, 6))
    df_train_raw['CLV'].hist(bins=50, color='lightgreen')
    plt.title('Distribution of Customer Lifetime Value (CLV)')
    plt.xlabel('CLV ($)')
    plt.ylabel('Number of Customers')
    plt.grid(axis='y', linestyle='--')
    plt.savefig('clv_distribution.png')
    print("Saved CLV distribution plot to clv_distribution.png")
    plt.close()

if __name__ == "__main__":
    analyze_clv()