"""
Feature Importance Utilities for Streamlit App
Provides fallback interpretability when SHAP is unavailable.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path for config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import src.config as config

def load_feature_importance(model_name):
    """Load pre-computed feature importance for a given model."""
    filename_map = {
        'logisticregression': 'lr_feature_importance.csv',
        'randomforest': 'rf_feature_importance.csv',
        'xgboost': 'xgb_feature_importance.csv'
    }
    
    if model_name not in filename_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    filepath = os.path.join(config.MODELS_PATH, filename_map[model_name])
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Feature importance file not found: {filepath}")
    
    return pd.read_csv(filepath)

def create_global_importance_plot(model_name, top_n=10):
    """Create a horizontal bar plot of global feature importance."""
    importance_df = load_feature_importance(model_name)
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar plot (reversed for top-to-bottom order)
    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_features['importance'], color='steelblue', alpha=0.7)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()  # Top feature at top
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Feature Importances - {model_name.replace("_", " ").title()}')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def get_local_importance_explanation(model, input_data, model_name, top_n=10):
    """
    Generate local feature importance explanation for a single prediction.
    This simulates SHAP waterfall plot functionality using feature values and importance.
    """
    # Get global feature importance
    importance_df = load_feature_importance(model_name)
    
    # Get feature values for this prediction
    feature_values = input_data.iloc[0] if hasattr(input_data, 'iloc') else input_data
    
    # Calculate local contribution approximation:
    # contribution = global_importance * normalized_feature_value
    local_contributions = []
    
    for _, row in importance_df.head(top_n).iterrows():
        feature = row['feature']
        global_imp = row['importance']
        
        if feature in feature_values.index:
            feature_val = feature_values[feature]
            # Normalize contribution by feature value (simple approximation)
            local_contrib = global_imp * abs(feature_val)
            local_contributions.append({
                'feature': feature,
                'value': feature_val,
                'contribution': local_contrib,
                'global_importance': global_imp
            })
    
    return pd.DataFrame(local_contributions).sort_values('contribution', ascending=False)

def create_local_importance_plot(local_contributions, prediction_prob):
    """Create a waterfall-style plot for local feature contributions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = local_contributions['feature']
    contributions = local_contributions['contribution']
    values = local_contributions['value']
    
    # Create horizontal bar plot
    y_pos = np.arange(len(features))
    colors = ['red' if contrib > contributions.median() else 'blue' for contrib in contributions]
    bars = ax.barh(y_pos, contributions, color=colors, alpha=0.7)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{feat}\n(val: {val:.2f})" for feat, val in zip(features, values)])
    ax.invert_yaxis()
    ax.set_xlabel('Feature Contribution to Prediction')
    ax.set_title(f'Local Feature Importance\nPredicted Churn Probability: {prediction_prob:.1%}')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left' if width >= 0 else 'right', 
                va='center', fontsize=9)
    
    # Add legend
    ax.text(0.02, 0.98, 'Red: Above median contribution\nBlue: Below median contribution', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig

def get_model_type_explanation(model_name):
    """Get explanation text for different model types."""
    explanations = {
        'logisticregression': {
            'method': 'Coefficient Analysis',
            'description': 'Importance calculated as |coefficient × feature_std_dev|. Higher values indicate features that have larger impact on the prediction.',
            'interpretation': 'Linear relationships - each unit increase in feature affects prediction by the coefficient amount.'
        },
        'randomforest': {
            'method': 'Mean Decrease Impurity',
            'description': 'Based on how much each feature decreases impurity when used for splits across all trees.',
            'interpretation': 'Non-linear relationships captured. Higher importance means feature is more useful for making accurate splits.'
        },
        'xgboost': {
            'method': 'Gain-based Importance',
            'description': 'Based on the improvement in accuracy brought by each feature to the branches it is on.',
            'interpretation': 'Gradient boosting importance. Features that improve model performance the most get higher scores.'
        }
    }
    
    return explanations.get(model_name, {
        'method': 'Unknown',
        'description': 'Feature importance calculation method not defined.',
        'interpretation': 'Interpretation guidelines not available.'
    })