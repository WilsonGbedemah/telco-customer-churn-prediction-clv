# Standard library imports
import os
import sys
import io
import time

# Third-party imports
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import time
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
def get_shap_explainer(model_name, _model, data):
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
def load_feature_importance(model_name):
    """Load pre-calculated feature importance for a given model."""
    model_file_map = {
        'Logistic Regression': 'lr_feature_importance.csv',
        'Random Forest': 'rf_feature_importance.csv', 
        'XGBoost': 'xgb_feature_importance.csv'
    }
    
    file_name = model_file_map.get(model_name)
    if not file_name:
        return None
    
    file_path = os.path.join(config.MODELS_PATH, file_name)
    
    try:
        importance_df = pd.read_csv(file_path)
        return importance_df.head(15)  # Top 15 features for better visualization
    except FileNotFoundError:
        return None

def create_feature_importance_plot(importance_df, model_name):
    """Create a feature importance plot from pre-calculated values."""
    if importance_df is None or importance_df.empty:
        return None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort by importance for better visualization
    importance_df_sorted = importance_df.sort_values('importance', ascending=True)
    
    # Create horizontal bar plot (similar to SHAP style)
    bars = ax.barh(range(len(importance_df_sorted)), 
                   importance_df_sorted['importance'],
                   color=['#FF6B6B' if imp > importance_df_sorted['importance'].median() 
                         else '#4ECDC4' for imp in importance_df_sorted['importance']])
    
    # Customize the plot
    ax.set_yticks(range(len(importance_df_sorted)))
    ax.set_yticklabels(importance_df_sorted['feature'], fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Feature Importance - {model_name}', fontsize=14, pad=20)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, importance_df_sorted['importance'])):
        width = bar.get_width()
        ax.text(width + max(importance_df_sorted['importance']) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', ha='left', va='center', fontsize=9)
    
    # Style improvements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def get_customer_segment(churn_prob, clv):
    """Classify customer into business segments based on churn risk and CLV."""
    if churn_prob >= 0.5 and clv >= 2000:
        return "🚨 Critical Risk - High Value", "#FF4B4B"
    elif churn_prob >= 0.5 and clv < 2000:
        return "⚠️ High Risk - Low Value", "#FF8C00" 
    elif churn_prob >= 0.3 and clv >= 2000:
        return "💰 Monitor - High Value", "#FFA500"
    elif churn_prob >= 0.3 and clv < 2000:
        return "📊 Standard Risk", "#32CD32"
    elif churn_prob < 0.3 and clv >= 2000:
        return "⭐ Champions - Retain", "#1E90FF"
    else:
        return "✅ Low Risk - Stable", "#228B22"

def get_retention_strategy(churn_prob, clv, customer_data):
    """Generate personalized retention strategy recommendations."""
    strategies = []
    
    if churn_prob >= 0.5:
        if clv >= 2000:
            strategies.append("🎯 **Executive Intervention**: Personal call from account manager")
            strategies.append("💳 **Premium Offers**: 20-30% discount or service upgrades")
        else:
            strategies.append("📞 **Retention Call**: Automated or junior staff outreach")
            strategies.append("💰 **Cost-Effective Offers**: 10-15% discount or loyalty rewards")
    
    # Check if month-to-month (neither one year nor two year contract)
    if customer_data.get('Contract_One year', 0) == 0 and customer_data.get('Contract_Two year', 0) == 0:
        strategies.append("📋 **Contract Upgrade**: Offer annual contract with incentives")
    
    if customer_data.get('TechSupport_No', 0) == 1:
        strategies.append("🛠️ **Support Enhancement**: Free tech support for 3-6 months")
    
    if customer_data.get('InternetService_Fiber optic', 0) == 1 and churn_prob > 0.5:
        strategies.append("🌐 **Service Optimization**: Network quality improvements")
    
    return strategies

def create_risk_value_plot(predictions_df):
    """Create interactive scatter plot of churn risk vs CLV."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(predictions_df['CLV'], predictions_df['Churn_Probability'], 
                        c=predictions_df['Churn_Probability'], cmap='RdYlGn_r',
                        alpha=0.6, s=60)
    
    ax.set_xlabel('Customer Lifetime Value ($)')
    ax.set_ylabel('Churn Probability')
    ax.set_title('Customer Risk-Value Matrix')
    
    # Add quadrant labels
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=predictions_df['CLV'].median(), color='gray', linestyle='--', alpha=0.5)
    
    plt.colorbar(scatter, label='Churn Risk')
    return fig

def process_batch_predictions(df, model, avg_tenure):
    """Process batch predictions for uploaded CSV."""
    results = []
    
    for idx, row in df.iterrows():
        # Create input dataframe with proper encoding
        input_df = pd.DataFrame(columns=X_test.columns)
        input_df.loc[0] = 0
        
        # Map the uploaded data to the model features
        for col in df.columns:
            if col in input_df.columns:
                input_df[col] = row[col]
            elif f'Contract_{row.get("Contract", "")}' in input_df.columns:
                input_df[f'Contract_{row["Contract"]}'] = 1
            elif f'InternetService_{row.get("InternetService", "")}' in input_df.columns:
                input_df[f'InternetService_{row["InternetService"]}'] = 1
        
        churn_prob = model.predict_proba(input_df[X_test.columns])[:, 1][0]
        clv = row.get('MonthlyCharges', 70) * avg_tenure
        segment, _ = get_customer_segment(churn_prob, clv)
        
        results.append({
            'Customer_ID': row.get('customerID', f'Customer_{idx+1}'),
            'Churn_Probability': churn_prob,
            'Risk_Level': 'High' if churn_prob >= 0.5 else 'Medium' if churn_prob >= 0.3 else 'Low',
            'CLV': clv,
            'Segment': segment,
            'Monthly_Charges': row.get('MonthlyCharges', 70),
            'Tenure': row.get('tenure', 12)
        })
    
    return pd.DataFrame(results)

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

# --- Easy Interactive Features ---
def generate_retention_email(customer_data, churn_prob, clv):
    """Generate personalized retention email."""
    risk_level = "high" if churn_prob >= 0.5 else "medium" if churn_prob >= 0.3 else "low"
    
    if risk_level == "high":
        return f"""
Subject: Special Retention Offer - We Value Your Business!

Dear Valued Customer,

We've noticed some changes in your account and want to ensure you're getting the best value from our services.

As a customer with ${clv:.0f} lifetime value, you're important to us. We'd like to offer:

• 20% discount on your next 6 months (${customer_data.get('monthly_charges', 0) * 0.2:.0f}/month savings)
• Free upgrade to premium support services
• Flexible payment options to better suit your needs

Please call us at 1-800-TELCO or reply to schedule a consultation.

Best regards,
Customer Retention Team
        """
    elif risk_level == "medium":
        return f"""
Subject: Exclusive Offer Just For You!

Hello,

We appreciate your loyalty as a ${clv:.0f} lifetime value customer. 

To show our appreciation, we're offering:
• 10% discount on your next 3 months
• Free service upgrade consultation
• Priority customer support

Contact us to learn more about optimizing your plan.

Best regards,
Customer Success Team
        """
    else:
        return f"""
Subject: Thank You for Your Loyalty!

Dear Loyal Customer,

Thank you for being a valued ${clv:.0f} lifetime value customer.

We'd love to help you get even more value:
• Explore our latest services
• Consider bundling for additional savings
• Learn about loyalty rewards program

We're here to help optimize your experience.

Best regards,
Account Management Team
        """

def calculate_model_confidence(model, input_df, n_samples=50):
    """Calculate prediction confidence using bootstrap sampling."""
    predictions = []
    
    # Add small random noise to inputs to simulate uncertainty
    for _ in range(n_samples):
        noisy_input = input_df.copy()
        
        # Add small noise to numerical features
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numerical_cols:
            if col in noisy_input.columns:
                noise = np.random.normal(0, 0.02)  # 2% noise
                noisy_input[col] = noisy_input[col] * (1 + noise)
        
        try:
            pred = model.predict_proba(noisy_input)[0][1]
            predictions.append(pred)
        except:
            predictions.append(input_df.iloc[0]['tenure'] / 100)  # fallback
    
    predictions = np.array(predictions)
    confidence_interval = np.percentile(predictions, [2.5, 97.5])
    uncertainty = confidence_interval[1] - confidence_interval[0]
    
    return {
        'mean': np.mean(predictions),
        'std': np.std(predictions), 
        'confidence_interval': confidence_interval,
        'uncertainty': uncertainty,
        'certainty_level': 'High' if uncertainty < 0.1 else 'Medium' if uncertainty < 0.2 else 'Low'
    }

def calculate_optimal_offer(clv, churn_prob):
    """Calculate optimal retention offer based on CLV and churn risk."""
    if churn_prob >= 0.5:
        discount_pct = min(25, int(churn_prob * 30))
        return f"{discount_pct}% discount for 6 months + free premium support"
    elif churn_prob >= 0.3:
        discount_pct = min(15, int(churn_prob * 20))
        return f"{discount_pct}% discount for 3 months + service consultation"
    else:
        return "Loyalty rewards program + optional service upgrades"

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
# Custom CSS for centered and colored tabs
st.markdown("""
<style>
/* Center the tabs container */
.stTabs [data-baseweb="tab-list"] {
    justify-content: center;
    max-width: 900px;
    margin: 0 auto;
}

/* Style individual tabs with different colors and borders */
.stTabs [data-baseweb="tab-list"] button {
    margin: 0 5px;
    border-radius: 10px;
    border: 2px solid transparent;
    padding: 10px 20px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Tab 1 - Churn Prediction (Blue) */
.stTabs [data-baseweb="tab-list"] button:nth-child(1) {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-color: #5a67d8;
}

.stTabs [data-baseweb="tab-list"] button:nth-child(1):hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
}

/* Tab 2 - Batch Analysis (Green) */
.stTabs [data-baseweb="tab-list"] button:nth-child(2) {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: white;
    border-color: #0f8a7e;
}

.stTabs [data-baseweb="tab-list"] button:nth-child(2):hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(17, 153, 142, 0.3);
}

/* Tab 3 - Model Performance (Purple) */
.stTabs [data-baseweb="tab-list"] button:nth-child(3) {
    background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
    color: white;
    border-color: #9575cd;
}

.stTabs [data-baseweb="tab-list"] button:nth-child(3):hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(161, 140, 209, 0.3);
}

/* Tab 4 - CLV Overview (Teal) */
.stTabs [data-baseweb="tab-list"] button:nth-child(4) {
    background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    color: white;
    border-color: #4db6ac;
}

.stTabs [data-baseweb="tab-list"] button:nth-child(4):hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(132, 250, 176, 0.3);
}

/* Active tab styling */
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    border-width: 3px;
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

tabs = st.tabs(["🎯 Churn Prediction", "📊 Batch Analysis", " Model Performance", "💰 CLV Overview"])

# --- Predict Tab ---
with tabs[0]:
    # Animated header for churn prediction
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h1 style="color: white; margin: 0; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🎯 Customer Churn Prediction Engine
        </h1>
        <p style="color: #f8f9fa; margin: 10px 0; font-size: 1.2em;">
            🚀 AI-Powered Risk Assessment • 🔍 Intelligent Insights • 💡 Actionable Recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Animated loading message
    prediction_intro = st.empty()
    for i in range(4):
        dots = "." * (i + 1)
        prediction_intro.markdown(f"🔮 **Initializing prediction engine{dots}**")
        time.sleep(0.2)
    
    prediction_intro.markdown("""
    📊 **Ready to analyze customer churn risk!**  
    Enter customer details below to get instant predictions with AI-powered explanations.
    """)
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📋 Customer Profile")
        
        # Demographic Information
        st.markdown("**👤 Demographics**")
        st.caption("Basic customer information")
        
        # Use populated values if available, otherwise use defaults
        gender_default = st.session_state.get('populate_gender', "Male")
        gender_index = 0 if gender_default == "Male" else 1
        gender = st.selectbox("Gender", ["Male", "Female"], index=gender_index)
        
        senior_citizen_default = st.session_state.get('populate_senior_citizen', "No")
        senior_citizen_index = 0 if senior_citizen_default == "No" else 1
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], index=senior_citizen_index)
        
        partner_default = st.session_state.get('populate_partner', "No")
        partner_index = 0 if partner_default == "No" else 1
        partner = st.selectbox("Has Partner", ["No", "Yes"], index=partner_index)
        
        dependents_default = st.session_state.get('populate_dependents', "No")
        dependents_index = 0 if dependents_default == "No" else 1
        dependents = st.selectbox("Has Dependents", ["No", "Yes"], index=dependents_index)
        
        st.markdown("---")
        
        # Account Information
        st.markdown("**💳 Account Details**")
        st.caption("Contract and billing information")
        
        contract_default = st.session_state.get('populate_contract', 'Month-to-month')
        contract_options = ['Month-to-month', 'One year', 'Two year']
        contract_index = contract_options.index(contract_default) if contract_default in contract_options else 0
        contract = st.selectbox("Contract Type", contract_options, index=contract_index,
                               help="Contract duration affects customer commitment and churn risk")
        
        # Replace sliders with number inputs
        tenure_default = st.session_state.get('populate_tenure', 12)
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=tenure_default,
                               help="Number of months the customer has been with the company. Longer tenure typically means lower churn risk.")
        
        monthly_charges_default = st.session_state.get('populate_monthly_charges', 70.0)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=float(monthly_charges_default), step=0.50,
                                        help="Monthly amount charged to customer. Higher charges may increase churn risk.")
        
        total_charges_default = st.session_state.get('populate_total_charges', tenure_default * monthly_charges_default)
        total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=8700.0, value=float(total_charges_default), step=0.50,
                                      help="Total amount charged to customer over their tenure.")
        
        paperless_billing_default = st.session_state.get('populate_paperless_billing', "No")
        paperless_billing_index = 0 if paperless_billing_default == "No" else 1
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], index=paperless_billing_index)
        
        payment_method_default = st.session_state.get('populate_payment_method', "Electronic check")
        payment_options = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        payment_index = payment_options.index(payment_method_default) if payment_method_default in payment_options else 0
        payment_method = st.selectbox("Payment Method", payment_options, index=payment_index)
        
        st.markdown("---")
        
        # Services Information  
        st.markdown("**📞 Services**")
        st.caption("Communication and internet services")
        
        phone_service_default = st.session_state.get('populate_phone_service', "No")
        phone_service_index = 0 if phone_service_default == "No" else 1
        phone_service = st.selectbox("Phone Service", ["No", "Yes"], index=phone_service_index)
        
        multiple_lines_default = st.session_state.get('populate_multiple_lines', "No")
        multiple_lines_options = ["No", "Yes", "No phone service"]
        multiple_lines_index = multiple_lines_options.index(multiple_lines_default) if multiple_lines_default in multiple_lines_options else 0
        multiple_lines = st.selectbox("Multiple Lines", multiple_lines_options, index=multiple_lines_index)
        
        internet_service_default = st.session_state.get('populate_internet_service', 'DSL')
        internet_options = ['DSL', 'Fiber optic', 'No']
        internet_index = internet_options.index(internet_service_default) if internet_service_default in internet_options else 0
        internet_service = st.selectbox("Internet Service", internet_options, index=internet_index,
                                      help="Internet service type affects customer satisfaction and churn")
        
        if internet_service != 'No':
            online_security_default = st.session_state.get('populate_online_security', "No")
            online_security_index = 0 if online_security_default == "No" else 1
            online_security = st.selectbox("Online Security", ["No", "Yes"], index=online_security_index)
            
            online_backup_default = st.session_state.get('populate_online_backup', "No")
            online_backup_index = 0 if online_backup_default == "No" else 1
            online_backup = st.selectbox("Online Backup", ["No", "Yes"], index=online_backup_index)
            
            device_protection_default = st.session_state.get('populate_device_protection', "No")
            device_protection_index = 0 if device_protection_default == "No" else 1
            device_protection = st.selectbox("Device Protection", ["No", "Yes"], index=device_protection_index)
            
            tech_support_default = st.session_state.get('populate_tech_support', "No")
            tech_support_index = 0 if tech_support_default == "No" else 1
            tech_support = st.selectbox("Tech Support", ["No", "Yes"], index=tech_support_index,
                                      help="Technical support reduces churn risk significantly")
            
            streaming_tv_default = st.session_state.get('populate_streaming_tv', "No")
            streaming_tv_index = 0 if streaming_tv_default == "No" else 1
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"], index=streaming_tv_index)
            
            streaming_movies_default = st.session_state.get('populate_streaming_movies', "No")
            streaming_movies_index = 0 if streaming_movies_default == "No" else 1
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"], index=streaming_movies_index)
        else:
            online_security = online_backup = device_protection = tech_support = streaming_tv = streaming_movies = "No internet service"
        
        model_choice_predict = st.selectbox("🤖 Prediction Model", 
                                          ['Logistic Regression', 'Random Forest', 'XGBoost'], 
                                          key='predict',
                                          help="Choose the machine learning model for prediction")

    with col2:
        st.subheader("🎯 Prediction Results")
        
        if st.button("🔮 Predict Churn Risk", width='stretch', type="primary"):
            # Create input dataframe with all features
            input_df = pd.DataFrame(columns=X_test.columns)
            input_df.loc[0] = 0
            
            # Basic features
            input_df['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0
            input_df['tenure'] = tenure
            input_df['MonthlyCharges'] = monthly_charges
            input_df['TotalCharges'] = total_charges
            
            # Categorical features (one-hot encoded)
            input_df[f'gender_{gender}'] = 1
            input_df[f'Partner_{partner}'] = 1
            input_df[f'Dependents_{dependents}'] = 1
            input_df[f'PhoneService_{phone_service}'] = 1
            input_df[f'MultipleLines_{multiple_lines}'] = 1
            input_df[f'Contract_{contract}'] = 1
            input_df[f'PaperlessBilling_{paperless_billing}'] = 1
            input_df[f'PaymentMethod_{payment_method}'] = 1
            input_df[f'InternetService_{internet_service}'] = 1
            
            if internet_service != 'No':
                input_df[f'OnlineSecurity_{online_security}'] = 1
                input_df[f'OnlineBackup_{online_backup}'] = 1
                input_df[f'DeviceProtection_{device_protection}'] = 1
                input_df[f'TechSupport_{tech_support}'] = 1
                input_df[f'StreamingTV_{streaming_tv}'] = 1
                input_df[f'StreamingMovies_{streaming_movies}'] = 1
            
            # Count services properly (only when customer has them)
            services = []
            if phone_service == "Yes":
                services.append(1)
            if internet_service != "No":
                if online_security == "Yes":
                    services.append(1)
                if online_backup == "Yes":
                    services.append(1)
                if device_protection == "Yes":
                    services.append(1)
                if tech_support == "Yes":
                    services.append(1)
                if streaming_tv == "Yes":
                    services.append(1)
                if streaming_movies == "Yes":
                    services.append(1)
            
            input_df['services_count'] = len(services)
            
            # Engineered features
            input_df['monthly_to_total_ratio'] = monthly_charges / max(total_charges, 1)  # Avoid division by zero
            input_df['internet_but_no_tech_support'] = 1 if (internet_service != "No" and tech_support == "No") else 0
            
            # Tenure buckets
            if tenure <= 6:
                pass  # 0-6m is the reference category (all zeros)
            elif tenure <= 12:
                input_df['tenure_bucket_6-12m'] = 1
            elif tenure <= 24:
                input_df['tenure_bucket_12-24m'] = 1
            else:
                input_df['tenure_bucket_24m+'] = 1
            
            model_name_map = {'Logistic Regression': 'logisticregression', 'Random Forest': 'randomforest', 'XGBoost': 'xgboost'}
            model = models[model_name_map[model_choice_predict]]

            churn_prob = model.predict_proba(input_df[X_test.columns])[:, 1][0]
            clv = monthly_charges * avg_tenure
            segment, segment_color = get_customer_segment(churn_prob, clv)
            
            # Display main results
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                if churn_prob >= 0.5:
                    risk_level = "HIGH RISK"
                    risk_color = "#ff4444"
                elif churn_prob >= 0.3:
                    risk_level = "MEDIUM RISK" 
                    risk_color = "#ffcc00"
                else:
                    risk_level = "LOW RISK"
                    risk_color = "#44aa44"
                
                # Custom colored metric
                st.markdown(f"""
                <div style="background: {risk_color}; color: white; padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0;">
                    <p style="margin: 0; font-size: 14px; opacity: 0.9;">Churn Probability</p>
                    <h2 style="margin: 5px 0; font-size: 2.5em; font-weight: bold;">{churn_prob:.1%}</h2>
                    <p style="margin: 0; font-size: 16px; font-weight: bold;">{risk_level}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_metric2:
                st.metric(label="Customer Lifetime Value", value=f"${clv:,.2f}")
            
            # Customer segment
            st.markdown(f"**Customer Segment:** <span style='color:{segment_color}'>{segment}</span>", 
                       unsafe_allow_html=True)
            
            # Retention strategies
            st.markdown("### 🎯 Recommended Retention Strategies")
            customer_dict = input_df.to_dict('records')[0] if len(input_df) > 0 else {}
            strategies = get_retention_strategy(churn_prob, clv, customer_dict)
            
            if strategies:
                for strategy in strategies:
                    st.markdown(f"• {strategy}")
            elif churn_prob >= 0.5:
                st.error("⚠️ High churn risk detected - immediate retention action recommended!")
            elif churn_prob >= 0.3:
                st.warning("📊 Medium churn risk - monitor and consider proactive retention")
            else:
                st.success("✅ Customer is low risk - maintain current service level")

            # SHAP Explanation with Simple Insights
            st.markdown("---")
            st.subheader("🔍 Why This Prediction?")
            st.markdown("Understanding which factors drive the churn prediction for this customer:")
            
            # Show SHAP chart first (technical analysis)
            if SHAP_AVAILABLE:
                explainer = get_shap_explainer(model_name_map[model_choice_predict], model, X_test.iloc[:100])
                if explainer is not None:
                    try:
                        shap_values = explainer(input_df[X_test.columns])
                        
                        st.markdown("#### 📊 Technical Analysis (SHAP)")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                        st.pyplot(fig, bbox_inches='tight')
                        
                        with st.expander("📖 How to Read This Chart"):
                            st.markdown("""
                            - **Blue bars**: Features that DECREASE churn probability
                            - **Red bars**: Features that INCREASE churn probability  
                            - **Bar length**: How much each feature contributes to the prediction
                            - **Base value**: Average churn rate for all customers
                            - **Final prediction**: The actual probability for this customer
                            """)
                    except Exception as e:
                        st.error(f"Unable to generate SHAP explanation: {str(e)}")
                        _show_feature_importance_local(model, input_df, model_name_map[model_choice_predict], churn_prob)
                else:
                    _show_feature_importance_local(model, input_df, model_name_map[model_choice_predict], churn_prob)
            else:
                _show_feature_importance_local(model, input_df, model_name_map[model_choice_predict], churn_prob)
            
            # Add separator
            st.markdown("---")
            
            # Simple Business Insights (underneath the technical chart)
            st.markdown("#### 💼 Business Insights")
            st.markdown("Key factors that influenced this prediction in simple terms:")
            
            # Create a simple, understandable explanation
            if churn_prob >= 0.5:
                risk_level = "High Risk"
                risk_color = "#ff4444"
            elif churn_prob >= 0.3:
                risk_level = "Medium Risk"
                risk_color = "#ffcc00"
            else:
                risk_level = "Low Risk"
                risk_color = "#44aa44"
            
            # Risk indicator
            st.markdown(f"""
            <div style="background: {risk_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0;">
                <h3 style="margin: 0;">Customer Risk Level: {risk_level}</h3>
                <p style="margin: 5px 0; font-size: 1.1em;">Churn Probability: {churn_prob:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Simple explanation based on input values
            explanations = []
            
            # Tenure analysis
            if input_df['tenure'].iloc[0] <= 12:
                explanations.append("⚠️ **New Customer**: Short tenure (≤12 months) increases churn risk")
            elif input_df['tenure'].iloc[0] >= 60:
                explanations.append("✅ **Loyal Customer**: Long tenure (≥5 years) reduces churn risk")
            
            # Contract analysis
            if input_df['Contract_One year'].iloc[0] == 0 and input_df['Contract_Two year'].iloc[0] == 0:
                explanations.append("⚠️ **Month-to-Month Contract**: No long-term commitment increases risk")
            elif input_df['Contract_Two year'].iloc[0] == 1:
                explanations.append("✅ **Two-Year Contract**: Long-term commitment reduces churn risk")
            
            # Payment method analysis
            if input_df['PaymentMethod_Electronic check'].iloc[0] == 1:
                explanations.append("⚠️ **Electronic Check Payment**: This payment method shows higher churn rates")
            elif input_df['PaymentMethod_Credit card (automatic)'].iloc[0] == 1:
                explanations.append("✅ **Automatic Credit Card**: Convenient payment reduces churn risk")
            
            # Monthly charges analysis
            monthly_charges = input_df['MonthlyCharges'].iloc[0]
            if monthly_charges > 80:
                explanations.append("⚠️ **High Monthly Charges**: Premium pricing may increase churn risk")
            elif monthly_charges < 35:
                explanations.append("✅ **Affordable Pricing**: Lower charges reduce churn likelihood")
            
            # Internet service analysis
            if input_df['InternetService_Fiber optic'].iloc[0] == 1:
                explanations.append("⚠️ **Fiber Optic Service**: Higher churn rates observed with this service")
            elif input_df['InternetService_No'].iloc[0] == 1:
                explanations.append("ℹ️ **No Internet Service**: Different risk profile than internet customers")
            
            # Senior citizen analysis
            if input_df['SeniorCitizen'].iloc[0] == 1:
                explanations.append("ℹ️ **Senior Citizen**: Age demographics may influence loyalty patterns")
            
            # Display explanations
            if explanations:
                for explanation in explanations:
                    st.markdown(f"• {explanation}")
            else:
                st.markdown("• **Balanced Profile**: Customer shows mixed indicators")
            
            # Action recommendations
            st.markdown("#### 🎯 Recommended Actions")
            
            if churn_prob >= 0.5:
                st.error("""
                **High Risk - Immediate Action Required:**
                • Contact customer within 24 hours
                • Offer retention incentives or discounts
                • Consider contract upgrade options
                • Review service satisfaction
                """)
            elif churn_prob >= 0.3:
                st.warning("""
                **Medium Risk - Proactive Engagement:**
                • Send personalized retention offer
                • Survey customer satisfaction
                • Highlight service benefits
                • Consider loyalty program enrollment
                """)
            else:
                st.success("""
                **Low Risk - Maintain Relationship:**
                • Continue excellent service
                • Consider upselling opportunities
                • Recognize loyalty with rewards
                • Monitor for any changes
                """)

            # --- NEW: 4 Easy Interactive Features ---
            st.markdown("---")
            
            # Feature 1: Model Confidence Indicator
            st.subheader("🎯 Model Confidence Analysis")
            
            confidence_data = calculate_model_confidence(model, input_df)
            
            conf_col1, conf_col2, conf_col3 = st.columns(3)
            with conf_col1:
                st.metric("Prediction", f"{churn_prob:.1%}")
            with conf_col2:
                st.metric("Confidence Level", confidence_data['certainty_level'])
            with conf_col3:
                st.metric("Uncertainty Range", f"±{confidence_data['uncertainty']:.1%}")
            
            # Confidence explanation
            with st.expander("Understanding Model Confidence"):
                st.markdown(f"""
                **Confidence Analysis:**
                - **Main Prediction**: {churn_prob:.1%}
                - **Confidence Range**: {confidence_data['confidence_interval'][0]:.1%} - {confidence_data['confidence_interval'][1]:.1%}
                - **Model Certainty**: {confidence_data['certainty_level']}
                
                **What this means:**
                - High certainty: Model is very confident in this prediction
                - Medium certainty: Prediction is reliable but consider additional factors
                - Low certainty: Gather more information before making decisions
                """)

    # Feature 2: Customer Comparison Tool  
    st.markdown("---")
    st.subheader("⚖️ Customer Comparison Tool")
    st.markdown("Compare two different customer profiles side-by-side.")
    
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.markdown("**Customer A**")
        a_tenure = st.number_input("Tenure A (months)", 0, 72, 12, key="comp_a_tenure")
        a_contract = st.selectbox("Contract A", ['Month-to-month', 'One year', 'Two year'], key="comp_a_contract")
        a_charges = st.number_input("Monthly Charges A ($)", 18.0, 120.0, 70.0, key="comp_a_charges")
        a_payment = st.selectbox("Payment Method A", 
                                ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 
                                key="comp_a_payment")
    
    with comp_col2:
        st.markdown("**Customer B**")
        b_tenure = st.number_input("Tenure B (months)", 0, 72, 24, key="comp_b_tenure") 
        b_contract = st.selectbox("Contract B", ['Month-to-month', 'One year', 'Two year'], index=1, key="comp_b_contract")
        b_charges = st.number_input("Monthly Charges B ($)", 18.0, 120.0, 50.0, key="comp_b_charges")
        b_payment = st.selectbox("Payment Method B",
                                ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
                                index=2, key="comp_b_payment")
    
    if st.button("🔍 Compare Customers", key="compare_btn"):
        # Create input DataFrames for both customers (simplified)
        # This is a basic comparison - you'd need to create full feature vectors
        
        comparison_results = {
            'Customer A': {
                'Tenure': f"{a_tenure} months",
                'Contract Risk': "High" if a_contract == "Month-to-month" else "Low",
                'Price Risk': "High" if a_charges > 75 else "Medium" if a_charges > 50 else "Low",
                'Payment Risk': "High" if a_payment == "Electronic check" else "Low",
                'Estimated Risk': "High" if (a_contract == "Month-to-month" and a_charges > 75) else "Medium"
            },
            'Customer B': {
                'Tenure': f"{b_tenure} months", 
                'Contract Risk': "High" if b_contract == "Month-to-month" else "Low",
                'Price Risk': "High" if b_charges > 75 else "Medium" if b_charges > 50 else "Low",
                'Payment Risk': "High" if b_payment == "Electronic check" else "Low",
                'Estimated Risk': "High" if (b_contract == "Month-to-month" and b_charges > 75) else "Low"
            }
        }
        
        # Display comparison
        st.markdown("### Comparison Results")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.markdown("**Customer A Analysis**")
            for key, value in comparison_results['Customer A'].items():
                color = "🔴" if "High" in str(value) else "🟡" if "Medium" in str(value) else "🟢"
                st.markdown(f"{color} {key}: {value}")
        
        with result_col2:
            st.markdown("**Customer B Analysis**") 
            for key, value in comparison_results['Customer B'].items():
                color = "🔴" if "High" in str(value) else "🟡" if "Medium" in str(value) else "🟢"
                st.markdown(f"{color} {key}: {value}")
        
        # Winner determination
        a_risk_score = sum(1 for v in comparison_results['Customer A'].values() if "High" in str(v))
        b_risk_score = sum(1 for v in comparison_results['Customer B'].values() if "High" in str(v))
        
        if a_risk_score > b_risk_score:
            st.error("🚨 Customer A has higher churn risk - prioritize retention efforts!")
        elif b_risk_score > a_risk_score:
            st.error("🚨 Customer B has higher churn risk - prioritize retention efforts!")
        else:
            st.warning("⚖️ Both customers have similar risk levels - monitor both closely.")

# --- Batch Analysis Tab ---
with tabs[1]:
    st.header("📊 Batch Customer Analysis")
    st.markdown("Upload a CSV file to analyze multiple customers at once and get comprehensive insights.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📁 Upload Data")
        
        # Sample CSV download
        sample_data = {
            'customerID': ['7590-VHVEG', '5575-GNVDE', '3668-QPYBK'],
            'gender': ['Female', 'Male', 'Male'], 
            'SeniorCitizen': [0, 0, 0],
            'Partner': ['Yes', 'No', 'No'],
            'Dependents': ['No', 'No', 'No'],
            'tenure': [1, 34, 2],
            'PhoneService': ['No', 'Yes', 'Yes'],
            'MultipleLines': ['No phone service', 'No', 'No'],
            'InternetService': ['DSL', 'DSL', 'DSL'],
            'OnlineSecurity': ['No', 'Yes', 'Yes'],
            'OnlineBackup': ['Yes', 'No', 'Yes'],
            'DeviceProtection': ['No', 'Yes', 'No'],
            'TechSupport': ['No', 'No', 'No'],
            'StreamingTV': ['No', 'No', 'No'],
            'StreamingMovies': ['No', 'No', 'No'],
            'Contract': ['Month-to-month', 'One year', 'Month-to-month'],
            'PaperlessBilling': ['Yes', 'No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Mailed check'],
            'MonthlyCharges': [29.85, 56.95, 53.85],
            'TotalCharges': [29.85, 1889.5, 108.15]
        }
        sample_df = pd.DataFrame(sample_data)
        
        # Convert to CSV for download
        csv_buffer = io.StringIO()
        sample_df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=csv_buffer.getvalue(),
            file_name="customer_template.csv",
            mime="text/csv",
            help="Download this template to see the required format"
        )
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'], 
                                       help="Upload a CSV file with customer data")
        
        batch_model = st.selectbox("Select Model for Batch Analysis", 
                                 ['Logistic Regression', 'Random Forest', 'XGBoost'], 
                                 key='batch')
    
    with col2:
        st.subheader("📈 Analysis Results")
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                df_upload = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {len(df_upload)} customers")
                
                # Process predictions
                model_name_map = {'Logistic Regression': 'logisticregression', 
                                'Random Forest': 'randomforest', 'XGBoost': 'xgboost'}
                model = models[model_name_map[batch_model]]
                
                results_df = process_batch_predictions(df_upload, model, avg_tenure)
                
                # Summary statistics
                st.markdown("### 📊 Summary Statistics")
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                
                with col_stat1:
                    high_risk_count = len(results_df[results_df['Risk_Level'] == 'High'])
                    st.metric("High Risk Customers", high_risk_count, 
                            f"{high_risk_count/len(results_df)*100:.1f}%")
                
                with col_stat2:
                    avg_clv = results_df['CLV'].mean()
                    st.metric("Average CLV", f"${avg_clv:,.2f}")
                
                with col_stat3:
                    total_risk_value = results_df[results_df['Risk_Level'] == 'High']['CLV'].sum()
                    st.metric("At-Risk Revenue", f"${total_risk_value:,.2f}")
                
                # Risk distribution chart
                st.markdown("### 📈 Risk Distribution")
                fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
                risk_counts = results_df['Risk_Level'].value_counts()
                colors = ['#FF4B4B' if x == 'High' else '#00D4AA' for x in risk_counts.index]
                ax_dist.bar(risk_counts.index, risk_counts.values, color=colors)
                ax_dist.set_title('Customer Risk Distribution')
                ax_dist.set_ylabel('Number of Customers')
                st.pyplot(fig_dist)
                
                # Risk-Value Matrix
                st.markdown("### 🎯 Customer Risk-Value Matrix")
                risk_value_fig = create_risk_value_plot(results_df)
                st.pyplot(risk_value_fig)
                
                # Detailed results table
                st.markdown("### 📋 Detailed Results")
                st.dataframe(results_df.round(3), width='stretch')
                
                # Download results
                results_csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=results_csv,
                    file_name=f"churn_analysis_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please ensure your CSV file matches the required format. Download the sample template for reference.")

# --- Model Performance Tab ---
with tabs[2]:
    st.header("Model Performance Evaluation")
    st.markdown("Compare model performance and analyze feature importance and discrimination ability.")
    
    # Performance metrics table
    performance_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Precision': [0.5092, 0.5494, 0.5167],
        'Recall': [0.8128, 0.7139, 0.7861],
        'F1-Score': [0.6262, 0.6209, 0.6235],
        'AUC-ROC': [0.8366, 0.8317, 0.8316]
    }
    performance_df = pd.DataFrame(performance_data).set_index('Model')
    
    st.subheader("Performance Metrics")
    st.dataframe(performance_df.round(4), width='stretch')
    
    # Explanation of metrics
    with st.expander("Understanding Performance Metrics"):
        st.markdown("""
        - **Precision**: Of all customers predicted to churn, how many actually churned?
        - **Recall**: Of all customers who actually churned, how many did we catch?  
        - **F1-Score**: Balanced metric combining precision and recall
        - **AUC-ROC**: Overall model discriminative ability (0.5 = random, 1.0 = perfect)
        """)

    st.markdown("---")
    
    # Two-column layout for Feature Importance and ROC Curve
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Importance Analysis")
        
        model_choice_features = st.selectbox(
            "Select Model for Feature Analysis", 
            ["Logistic Regression", "Random Forest", "XGBoost"], 
            key="feature_analysis"
        )
        
        # Load feature importance data
        importance_df = load_feature_importance(model_choice_features)
        
        if importance_df is not None:
            # Get top 15 features for better display
            top_features = importance_df.head(15)
            
            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Create color gradient
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
            
            # Create horizontal bars
            y_positions = np.arange(len(top_features))
            bars = ax.barh(y_positions, top_features['importance'], 
                          color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Customize the plot
            ax.set_yticks(y_positions)
            ax.set_yticklabels(top_features['feature'], fontsize=10)
            ax.set_xlabel('Feature Importance Score', fontsize=12)
            ax.set_title(f'{model_choice_features} - Feature Importance', 
                       fontsize=14, fontweight='bold', pad=15)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
                width = bar.get_width()
                ax.text(width + max(top_features['importance']) * 0.01, 
                       bar.get_y() + bar.get_height()/2, 
                       f'{value:.3f}', 
                       ha='left', va='center', fontsize=9)
            
            # Style the plot
            ax.grid(axis='x', alpha=0.3)
            ax.set_facecolor('#fafafa')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Invert y-axis (most important at top)
            ax.invert_yaxis()
            
            plt.tight_layout()
            st.pyplot(fig, bbox_inches='tight')
            plt.close()
            
            # Feature details
            with st.expander("Feature Details"):
                st.dataframe(top_features.round(4), height=200)
                
        else:
            st.error(f"Feature importance data not found for {model_choice_features}")
            st.info("Run `make interpret` to generate feature importance files")
    
    with col2:
        st.subheader("ROC Curve Analysis")
        st.markdown("Model discrimination ability visualization")
        
        model_choice_roc = st.selectbox(
            "Select Model for ROC Curve", 
            ['Logistic Regression', 'Random Forest', 'XGBoost'], 
            key='roc_analysis'
        )
        
        # Model name mapping
        model_name_map_roc = {
            'Logistic Regression': 'logisticregression', 
            'Random Forest': 'randomforest', 
            'XGBoost': 'xgboost'
        }
        
        selected_model_roc = models[model_name_map_roc[model_choice_roc]]
        y_pred_proba_roc = selected_model_roc.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_roc)
        auc_score = roc_auc_score(y_test, y_pred_proba_roc)
        
        # Create ROC curve plot
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        ax_roc.plot(fpr, tpr, linewidth=2, color='#2E86AB',
                   label=f'{model_choice_roc} (AUC = {auc_score:.3f})')
        ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7,
                   label='Random Classifier (AUC = 0.5)')
        
        ax_roc.set_xlabel('False Positive Rate', fontsize=12)
        ax_roc.set_ylabel('True Positive Rate', fontsize=12)
        ax_roc.set_title(f'{model_choice_roc} ROC Curve', fontsize=14, fontweight='bold')
        ax_roc.legend(loc='lower right')
        ax_roc.grid(True, alpha=0.3)
        ax_roc.set_facecolor('#fafafa')
        
        # Add AUC shading
        ax_roc.fill_between(fpr, tpr, alpha=0.2, color='#2E86AB')
        
        plt.tight_layout()
        st.pyplot(fig_roc, bbox_inches='tight')
        plt.close()
        
        # ROC interpretation
        with st.expander("ROC Curve Interpretation"):
            st.markdown(f"""
            **Model Performance:** {model_choice_roc}
            - **AUC Score:** {auc_score:.3f}
            - **Interpretation:** {'Excellent' if auc_score > 0.8 else 'Good' if auc_score > 0.7 else 'Fair'}
            
            **How to read:**
            - **Closer to top-left corner:** Better performance
            - **Diagonal line:** Random classifier performance
            - **Area under curve:** Overall discriminative ability
            """)
        
        # Performance summary
        st.markdown("**Performance Summary**")
        perf_metrics = {
            'Logistic Regression': {'AUC': 0.8366, 'Precision': 0.5092, 'Recall': 0.8128},
            'Random Forest': {'AUC': 0.8317, 'Precision': 0.5494, 'Recall': 0.7139},
            'XGBoost': {'AUC': 0.8316, 'Precision': 0.5167, 'Recall': 0.7861}
        }
        
        metrics = perf_metrics[model_choice_roc]
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("AUC", f"{metrics['AUC']:.3f}")
        with metric_col2:
            st.metric("Precision", f"{metrics['Precision']:.3f}")
        with metric_col3:
            st.metric("Recall", f"{metrics['Recall']:.3f}")

# --- CLV Overview Tab ---
with tabs[3]:
    st.header("💰 Customer Lifetime Value (CLV) Analysis")
    st.markdown("Understand customer segments based on their lifetime value and churn behavior.")
    
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("📊 CLV Distribution")
        if os.path.exists('clv_distribution.png'):
            st.image('clv_distribution.png', caption='Distribution of Customer Lifetime Value')
        else:
            st.warning("CLV distribution chart not found. Please run 'make clv' to generate.")
    
    with col6:
        st.subheader("🎯 Churn Rate by Value Segment")
        if os.path.exists('clv_churn_rate.png'):
            st.image('clv_churn_rate.png', caption='Churn Rate by Customer Value Quartile')
        else:
            st.warning("Churn rate chart not found. Please run 'make clv' to generate.")

    st.markdown("---")
    st.subheader("🧠 Business Insights & Strategic Recommendations")
    
    col7, col8 = st.columns(2)
    
    with col7:
        st.markdown("""
        #### 📈 Key Findings
        - **CLV Distribution**: The majority of customers have a CLV in the lower-to-mid range, with fewer customers in the high-value segments
        - **Value-Loyalty Correlation**: Customers with lower calculated lifetime value churn at a significantly higher rate
        - **Revenue at Risk**: High-value customers show better retention patterns
        """)
    
    with col8:
        st.markdown("""
        #### 🎯 Strategic Recommendations
        - **Premium Segment**: Focus executive-level retention on "Premium" CLV customers
        - **High-Value Monitoring**: Implement proactive monitoring for "High" CLV segment
        - **Cost-Effective Programs**: Use automated/digital retention for lower CLV segments
        - **Early Intervention**: Target new customers in first 6 months to improve tenure
        """)
    
    # CLV Calculator
    st.markdown("---")
    st.subheader("🧮 CLV Calculator")
    st.markdown("Calculate customer lifetime value for different scenarios:")
    
    calc_col1, calc_col2, calc_col3 = st.columns(3)
    
    with calc_col1:
        calc_monthly = st.number_input("Monthly Revenue ($)", 20.0, 150.0, 70.0, key="calc_monthly")
        calc_tenure = st.number_input("Expected Tenure (months)", 1, 120, int(avg_tenure), key="calc_tenure")
    
    with calc_col2:
        calc_discount_rate = st.slider("Annual Discount Rate (%)", 0.0, 20.0, 8.0, key="calc_discount") / 100
        calc_churn_rate = st.slider("Monthly Churn Rate (%)", 0.5, 10.0, 2.5, key="calc_churn") / 100
    
    with calc_col3:
        # Simple CLV calculation
        simple_clv = calc_monthly * calc_tenure
        
        # Advanced CLV with discount and churn rate
        if calc_churn_rate > 0:
            # Expected customer lifespan in months
            expected_lifespan = 1 / calc_churn_rate
            # Discounted CLV
            monthly_discount = calc_discount_rate / 12
            if monthly_discount > 0:
                periods = min(expected_lifespan, 120)  # Cap at 10 years
                discounted_clv = calc_monthly * (1 - (1 + monthly_discount)**(-periods)) / monthly_discount
            else:
                discounted_clv = calc_monthly * expected_lifespan
        else:
            discounted_clv = simple_clv
        
        st.metric("Simple CLV", f"${simple_clv:,.2f}")
        st.metric("Advanced CLV", f"${discounted_clv:,.2f}")
        st.caption(f"Expected lifespan: {(1/calc_churn_rate if calc_churn_rate > 0 else calc_tenure):.1f} months")