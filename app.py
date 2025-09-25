# Standard library imports
import os
import sys

# Third-party imports
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Add src to path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import config
from importance_utils import (
    create_global_importance_plot, 
    get_local_importance_explanation,
    create_local_importance_plot,
    get_model_type_explanation
)

# --- Page Config (MUST be first Streamlit command) ---
st.set_page_config(page_title="Telco Churn & CLV Prediction", layout="wide")

# Try to import SHAP with error handling for Windows compatibility
SHAP_AVAILABLE = True
SHAP_ERROR_MESSAGE = None
try:
    import shap
    print("✅ SHAP successfully loaded!")
except (ImportError, OSError):
    SHAP_AVAILABLE = False
    SHAP_ERROR_MESSAGE = "⚠️ SHAP interpretability features are disabled. Using feature importance fallback method instead."

# --- Caching ---
@st.cache_data
def load_data():
    """Loads test data and supporting files."""
    X_test = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'y_test.csv')).values.ravel()
    
    # Load average tenure for CLV calculation
    try:
        with open(os.path.join(config.MODELS_PATH, 'clv_avg_tenure.txt'), 'r') as f:
            avg_tenure = float(f.read())
    except FileNotFoundError:
        avg_tenure = 32.26  # Fallback value
            
    return X_test, y_test, avg_tenure

@st.cache_resource
def load_models():
    """Loads all trained models."""
    models = {}
    for model_name in ['logisticregression', 'randomforest', 'xgboost']:
        model_path = os.path.join(config.MODELS_PATH, f'{model_name}.pkl')
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)
    return models

@st.cache_resource
def get_shap_explainer(_model, data):
    """Creates a SHAP explainer for a given model."""
    if not SHAP_AVAILABLE:
        return None
    try:
        if isinstance(_model, config.LogisticRegression):
            masker = shap.maskers.Independent(data=data)
            return shap.LinearExplainer(_model, masker)
        else:
            return shap.TreeExplainer(_model)
    except Exception:
        return None

# --- Helper Functions ---
def _show_feature_importance_local(model, input_df, model_name, churn_prob):
    """Show local feature importance explanation for a prediction."""
    try:
        local_contrib = get_local_importance_explanation(model, input_df, model_name, top_n=8)
        fig = create_local_importance_plot(local_contrib, churn_prob)
        st.pyplot(fig, bbox_inches='tight')
        
        # Add explanation
        model_explanation = get_model_type_explanation(model_name)
        with st.expander("How is this calculated?"):
            st.write(f"**Method:** {model_explanation['method']}")
            st.write(f"**Description:** {model_explanation['description']}")
            st.write(f"**Interpretation:** {model_explanation['interpretation']}")
    except Exception as e:
        st.error(f"Unable to generate feature importance explanation: {str(e)}")

def _show_feature_importance_global(model_name):
    """Show global feature importance plot."""
    try:
        fig = create_global_importance_plot(model_name, top_n=10)
        st.pyplot(fig, bbox_inches='tight')
        
        # Add explanation
        model_explanation = get_model_type_explanation(model_name)
        with st.expander("How is this calculated?"):
            st.write(f"**Method:** {model_explanation['method']}")
            st.write(f"**Description:** {model_explanation['description']}")
            st.write(f"**Interpretation:** {model_explanation['interpretation']}")
    except Exception as e:
        st.error(f"Unable to generate global feature importance plot: {str(e)}")

# --- Load All Assets ---
X_test, y_test, avg_tenure = load_data()
models = load_models()

# --- App Title ---
st.title("Telco Customer Churn Prediction & CLV")
st.markdown("An interactive tool to predict customer churn, understand its drivers, and analyze customer value.")

# Show SHAP warning if needed
if SHAP_ERROR_MESSAGE:
    st.warning(SHAP_ERROR_MESSAGE)

# --- App Tabs ---
tabs = st.tabs(["Churn Prediction", "Model Performance", "CLV Overview"])

# --- Predict Tab ---
with tabs[0]:
    st.header("Predict Customer Churn")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Customer Profile")
        contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0, 0.05)
        internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        tech_support = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
        
        model_choice_predict = st.selectbox("Prediction Model", ['Logistic Regression', 'Random Forest', 'XGBoost'], key='predict')

    with col2:
        st.subheader("Prediction Result")
        if st.button("Predict Churn", use_container_width=True):
            input_df = pd.DataFrame(columns=X_test.columns)
            input_df.loc[0] = 0
            input_df['tenure'] = tenure
            input_df['MonthlyCharges'] = monthly_charges
            input_df[f'Contract_{contract}'] = 1
            input_df[f'InternetService_{internet_service}'] = 1
            input_df[f'TechSupport_{tech_support}'] = 1

            model_name_map = {'Logistic Regression': 'logisticregression', 'Random Forest': 'randomforest', 'XGBoost': 'xgboost'}
            model = models[model_name_map[model_choice_predict]]

            churn_prob = model.predict_proba(input_df[X_test.columns])[:, 1][0]
            risk_label = "High Risk" if churn_prob >= 0.5 else "Low Risk"

            st.metric(label="Churn Probability", value=f"{churn_prob:.1%}", delta=risk_label)
            
            st.markdown("--- ")
            st.subheader("Estimated Customer Lifetime Value (CLV)")
            clv = monthly_charges * avg_tenure
            st.metric(label="Estimated CLV", value=f"${clv:,.2f}")
            st.caption(f"Based on an assumed average customer lifetime of {avg_tenure:.1f} months.")

            if SHAP_AVAILABLE:
                st.markdown("--- ")
                st.subheader("Feature Contribution to Prediction (SHAP)")
                explainer = get_shap_explainer(model, X_test.iloc[:100])
                if explainer is not None:
                    try:
                        shap_values = explainer(input_df[X_test.columns])
                        
                        fig, ax = plt.subplots()
                        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                        st.pyplot(fig, bbox_inches='tight')
                    except Exception as e:
                        st.error(f"Unable to generate SHAP explanation: {str(e)}")
                        # Fallback to feature importance
                        st.info("Falling back to feature importance explanation...")
                        _show_feature_importance_local(model, input_df, model_name_map[model_choice_predict], churn_prob)
                else:
                    st.info("SHAP explanation not available for this model type. Using feature importance instead.")
                    _show_feature_importance_local(model, input_df, model_name_map[model_choice_predict], churn_prob)
            else:
                st.markdown("--- ")
                st.subheader("Feature Contribution to Prediction")
                st.info("Using feature importance method (SHAP unavailable).")
                _show_feature_importance_local(model, input_df, model_name_map[model_choice_predict], churn_prob)

# --- Model Performance Tab ---
with tabs[1]:
    st.header("Model Performance Evaluation")
    performance_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Precision': [0.5092, 0.5494, 0.5167],
        'Recall': [0.8128, 0.7139, 0.7861],
        'F1-Score': [0.6262, 0.6209, 0.6235],
        'AUC-ROC': [0.8366, 0.8317, 0.8316]
    }
    performance_df = pd.DataFrame(performance_data).set_index('Model')
    st.dataframe(performance_df, use_container_width=True)

    st.markdown("--- ")
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Global Feature Importance")
        if SHAP_AVAILABLE:
            st.write("**Method: SHAP Analysis**")
            model_choice_shap = st.selectbox("Select Model for SHAP Summary", ['Logistic Regression', 'Random Forest', 'XGBoost'], key='shap')
            model_name_map_shap = {'Logistic Regression': 'logisticregression', 'Random Forest': 'randomforest', 'XGBoost': 'xgboost'}
            selected_model_shap = models[model_name_map_shap[model_choice_shap]]
            explainer_shap = get_shap_explainer(selected_model_shap, X_test)
            if explainer_shap is not None:
                try:
                    shap_values_shap = explainer_shap(X_test)
                    fig_summary, ax_summary = plt.subplots()
                    shap.summary_plot(shap_values_shap, X_test, plot_type='bar', show=False)
                    st.pyplot(fig_summary, bbox_inches='tight')
                except Exception as e:
                    st.error(f"Unable to generate SHAP summary: {str(e)}")
                    st.info("Falling back to feature importance method...")
                    _show_feature_importance_global(model_name_map_shap[model_choice_shap])
            else:
                st.info("SHAP summary not available for this model type. Using feature importance instead.")
                _show_feature_importance_global(model_name_map_shap[model_choice_shap])
        else:
            st.write("**Method: Feature Importance Analysis**")
            model_choice_fi = st.selectbox("Select Model for Feature Importance", ['Logistic Regression', 'Random Forest', 'XGBoost'], key='fi')
            model_name_map_fi = {'Logistic Regression': 'logisticregression', 'Random Forest': 'randomforest', 'XGBoost': 'xgboost'}
            _show_feature_importance_global(model_name_map_fi[model_choice_fi])

    with col4:
        st.subheader("ROC Curve")
        model_choice_roc = st.selectbox("Select Model for ROC Curve", ['Logistic Regression', 'Random Forest', 'XGBoost'], key='roc')
        model_name_map_roc = {'Logistic Regression': 'logisticregression', 'Random Forest': 'randomforest', 'XGBoost': 'xgboost'}
        selected_model_roc = models[model_name_map_roc[model_choice_roc]]
        y_pred_proba_roc = selected_model_roc.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_roc)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_proba_roc):.2f}')
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'{model_choice_roc} ROC Curve')
        ax_roc.legend()
        st.pyplot(fig_roc)

# --- CLV Overview Tab ---
with tabs[2]:
    st.header("Customer Lifetime Value (CLV) Overview")
    st.markdown("This analysis segments customers by their lifetime value and examines their churn behavior.")
    
    col5, col6 = st.columns(2)
    with col5:
        st.image('clv_distribution.png', caption='Distribution of Customer Lifetime Value')
    with col6:
        st.image('clv_churn_rate.png', caption='Churn Rate by Customer Value Quartile')

    st.subheader("Business Insights")
    st.markdown("""- **CLV Distribution:** The majority of customers have a CLV in the lower-to-mid range, with fewer customers in the high-value segments.
- **Higher Value, Higher Loyalty:** Customers with lower calculated lifetime value churn at a significantly higher rate.
- **Retention Focus:** Efforts should be focused on preventing churn in the "High" and "Premium" segments, as they represent the most value to the business.
""")