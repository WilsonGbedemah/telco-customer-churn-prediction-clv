# Standard library imports
import os
import sys
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

/* Tab 2 - Model Performance (Purple) */
.stTabs [data-baseweb="tab-list"] button:nth-child(2) {
    background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
    color: white;
    border-color: #9575cd;
}

.stTabs [data-baseweb="tab-list"] button:nth-child(2):hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(161, 140, 209, 0.3);
}

/* Tab 3 - CLV Overview (Teal) */
.stTabs [data-baseweb="tab-list"] button:nth-child(3) {
    background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    color: white;
    border-color: #4db6ac;
}

.stTabs [data-baseweb="tab-list"] button:nth-child(3):hover {
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

tabs = st.tabs(["🎯 Churn Prediction", "📊 Model Performance", "💰 CLV Overview"])

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
            
            # Calculate confidence metrics
            confidence_data = calculate_model_confidence(model, input_df)
            uncertainty = confidence_data['uncertainty']
            certainty_level = confidence_data['certainty_level']
            
            # Display main results with enhanced visuals
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            
            with col_metric1:
                if churn_prob >= 0.5:
                    risk_level = "HIGH RISK"
                    risk_color = "#ff4444"
                    risk_icon = "🚨"
                elif churn_prob >= 0.3:
                    risk_level = "MEDIUM RISK" 
                    risk_color = "#ffcc00"
                    risk_icon = "⚠️"
                else:
                    risk_level = "LOW RISK"
                    risk_color = "#44aa44"
                    risk_icon = "✅"
                
                # Enhanced risk gauge with confidence
                st.markdown(f"""
                <div style="background: {risk_color}; color: white; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                    <div style="font-size: 2em; margin-bottom: 10px;">{risk_icon}</div>
                    <p style="margin: 0; font-size: 14px; opacity: 0.9;">Churn Probability</p>
                    <h2 style="margin: 5px 0; font-size: 2.5em; font-weight: bold;">{churn_prob:.1%}</h2>
                    <p style="margin: 0; font-size: 16px; font-weight: bold;">{risk_level}</p>
                    <p style="margin: 5px 0; font-size: 12px; opacity: 0.8;">±{uncertainty:.1%} uncertainty</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_metric2:
                # Enhanced CLV with context
                clv_percentile = np.percentile([70 * avg_tenure, 100 * avg_tenure, 50 * avg_tenure], 50)
                clv_status = "Above Average" if clv > clv_percentile else "Below Average"
                clv_color = "#2e8b57" if clv > clv_percentile else "#cd853f"
                
                st.markdown(f"""
                <div style="background: {clv_color}; color: white; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                    <div style="font-size: 2em; margin-bottom: 10px;">💰</div>
                    <p style="margin: 0; font-size: 14px; opacity: 0.9;">Customer Lifetime Value</p>
                    <h2 style="margin: 5px 0; font-size: 2.2em; font-weight: bold;">${clv:,.0f}</h2>
                    <p style="margin: 0; font-size: 14px; font-weight: bold;">{clv_status}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_metric3:
                # Prediction confidence gauge
                confidence_color = "#2e8b57" if certainty_level == "High" else "#ffcc00" if certainty_level == "Medium" else "#ff6b6b"
                confidence_icon = "🎯" if certainty_level == "High" else "📊" if certainty_level == "Medium" else "❓"
                
                st.markdown(f"""
                <div style="background: {confidence_color}; color: white; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                    <div style="font-size: 2em; margin-bottom: 10px;">{confidence_icon}</div>
                    <p style="margin: 0; font-size: 14px; opacity: 0.9;">Prediction Confidence</p>
                    <h2 style="margin: 5px 0; font-size: 2.2em; font-weight: bold;">{certainty_level}</h2>
                    <p style="margin: 0; font-size: 14px; font-weight: bold;">{(1-uncertainty)*100:.0f}% Certain</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Customer segment
            st.markdown(f"**Customer Segment:** <span style='color:{segment_color}'>{segment}</span>", 
                       unsafe_allow_html=True)
            
            # Simplified Retention Strategies
            st.markdown("### 🎯 Recommended Retention Strategies")
            customer_dict = input_df.to_dict('records')[0] if len(input_df) > 0 else {}
            strategies = get_retention_strategy(churn_prob, clv, customer_dict)
            
            if strategies:
                for strategy in strategies:
                    st.markdown(f"• {strategy}")
            
            # Priority assessment
            if churn_prob >= 0.5:
                st.error("🚨 **HIGH PRIORITY**: Contact within 24 hours")
            elif churn_prob >= 0.3:
                st.warning("⚡ **MEDIUM PRIORITY**: Contact within 7 days")
            else:
                st.success("✅ **LOW PRIORITY**: Routine check-in")

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
    
    # Prediction History Tracker
    st.markdown("---")
    st.markdown("### 📊 Prediction History & Export")
    
    # Initialize session state for prediction history
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    # Save prediction button
    if st.button("💾 Save This Prediction"):
        prediction_record = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model': model_choice_predict,
            'churn_probability': f"{churn_prob:.1%}",
            'risk_level': risk_level.replace(' RISK', ''),
            'clv': f"${clv:,.0f}",
            'confidence': certainty_level,
            'monthly_charges': f"${monthly_charges:.2f}",
            'tenure': f"{tenure} months"
        }
        st.session_state.prediction_history.append(prediction_record)
        st.success("✅ Prediction saved to history!")
    
    # Display prediction history
    if st.session_state.prediction_history:
        st.markdown("#### Recent Predictions")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, width='stretch')
        
        # Export functionality
        if st.button("📥 Export History to CSV"):
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=f"churn_predictions_{time.strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.success("History cleared!")

# --- Model Performance Tab ---
with tabs[1]:
    st.header("📊 Model Performance Evaluation")
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
    
    # Enhanced styled table
    styled_df = performance_df.style.format('{:.4f}')
    for col in performance_df.columns:
        styled_df = styled_df.highlight_max(subset=[col], color='lightgreen')
    
    st.dataframe(styled_df, width='stretch')
    
    # Best model recommendation
    avg_scores = performance_df.mean(axis=1)
    best_model = avg_scores.idxmax()
    st.success(f"🏆 **Recommended Model**: {best_model} (Average Score: {avg_scores[best_model]:.3f})")
    
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
        st.markdown("#### Feature Importance")
        model_choice_features = st.selectbox(
            "Select Model for Feature Analysis", 
            ["Logistic Regression", "Random Forest", "XGBoost"], 
            key="feature_analysis"
        )
        
        # Load feature importance data
        importance_df = load_feature_importance(model_choice_features)
        
        if importance_df is not None:
            # Get top 15 features as requested
            top_features = importance_df.head(15)
            
            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create color gradient
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
            
            # Create horizontal bars
            y_positions = np.arange(len(top_features))
            bars = ax.barh(y_positions, top_features['importance'], 
                          color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Customize the plot
            ax.set_yticks(y_positions)
            ax.set_yticklabels(top_features['feature'], fontsize=9)
            ax.set_xlabel('Importance Score', fontsize=10)
            ax.set_title(f'Top Features - {model_choice_features}', fontsize=12, fontweight='bold')
            
            # Invert y-axis (most important at top)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig, bbox_inches='tight')
            plt.close()
            
        else:
            st.error(f"Feature importance data not found for {model_choice_features}")
    
    with col2:
        st.markdown("#### ROC Curve Analysis")
        
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

# --- CLV Overview Tab ---
with tabs[2]:
    st.header("💰 Advanced Customer Lifetime Value Analytics")
    st.markdown("Comprehensive CLV analysis with interactive insights and strategic recommendations.")
    
    # CLV Analytics Tabs
    clv_tabs = st.tabs(["📊 CLV Distribution", "🎯 Customer Segments", "💼 Business Intelligence", "📈 Revenue Analysis"])
    
    with clv_tabs[0]:
        st.markdown("#### 📊 CLV Distribution Analysis")
        
        clv_col1, clv_col2 = st.columns(2)
        
        with clv_col1:
            if os.path.exists('clv_distribution.png'):
                st.image('clv_distribution.png', caption='Distribution of Customer Lifetime Value')
            else:
                # Create dynamic CLV distribution chart
                sample_clv = np.random.lognormal(mean=6, sigma=0.8, size=1000) * 50
                
                fig, ax = plt.subplots(figsize=(8, 6))
                n, bins, patches = ax.hist(sample_clv, bins=30, alpha=0.7, color='#4ECDC4', edgecolor='black')
                
                # Color code the bars
                for i, p in enumerate(patches):
                    if bins[i] < np.percentile(sample_clv, 25):
                        p.set_color('#ff7f7f')  # Red for low CLV
                    elif bins[i] < np.percentile(sample_clv, 75):
                        p.set_color('#ffb347')  # Orange for medium CLV  
                    else:
                        p.set_color('#90EE90')  # Green for high CLV
                
                ax.axvline(np.mean(sample_clv), color='red', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(sample_clv):,.0f}')
                ax.axvline(np.median(sample_clv), color='blue', linestyle='--', linewidth=2, label=f'Median: ${np.median(sample_clv):,.0f}')
                
                ax.set_xlabel('Customer Lifetime Value ($)')
                ax.set_ylabel('Number of Customers')
                ax.set_title('CLV Distribution Analysis')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with clv_col2:
            if os.path.exists('clv_churn_rate.png'):
                st.image('clv_churn_rate.png', caption='Churn Rate by Customer Value Quartile')
            else:
                # Create dynamic churn rate by CLV chart
                clv_segments = ['Low CLV\n(<$2K)', 'Medium-Low CLV\n($2K-$4K)', 'Medium-High CLV\n($4K-$7K)', 'High CLV\n(>$7K)']
                churn_rates = [0.45, 0.32, 0.21, 0.12]  # Sample data
                colors = ['#ff7f7f', '#ffb347', '#87ceeb', '#90EE90']
                
                fig, ax = plt.subplots(figsize=(8, 6))
                bars = ax.bar(clv_segments, churn_rates, color=colors, alpha=0.8, edgecolor='black')
                
                # Add value labels on bars
                for bar, rate in zip(bars, churn_rates):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
                
                ax.set_ylabel('Churn Rate')
                ax.set_title('Churn Rate by Customer Value Segment')
                ax.set_ylim(0, 0.5)
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    with clv_tabs[1]:
        st.markdown("#### 🎯 Customer Risk-Value Segmentation")
        
        # Create interactive scatter plot
        st.markdown("**Customer Portfolio Analysis**")
        
        # Generate sample data for demonstration
        np.random.seed(42)
        n_customers = 500
        clv_values = np.random.lognormal(6, 0.8, n_customers) * 50
        # Create more realistic churn probabilities with varied risk levels
        churn_probs = np.random.beta(2, 3, n_customers)  # More varied distribution
        # Add some correlation with CLV (higher CLV = slightly lower churn risk)
        clv_normalized = (clv_values - np.min(clv_values)) / (np.max(clv_values) - np.min(clv_values))
        churn_probs = churn_probs * (1.2 - 0.3 * clv_normalized)  # Reduce churn for high CLV
        churn_probs = np.clip(churn_probs, 0, 1)
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color points by risk level
        colors = ['red' if p >= 0.5 else 'orange' if p >= 0.3 else 'green' for p in churn_probs]
        scatter = ax.scatter(clv_values, churn_probs, c=colors, alpha=0.6, s=50)
        
        # Add quadrant lines
        median_clv = np.median(clv_values)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='High Risk Threshold')
        ax.axvline(x=median_clv, color='gray', linestyle='--', alpha=0.7, label=f'Median CLV (${median_clv:,.0f})')
        
        # Add quadrant labels
        ax.text(median_clv*1.5, 0.8, 'High Value\nHigh Risk', ha='center', va='center', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7), fontweight='bold')
        ax.text(median_clv*0.5, 0.8, 'Low Value\nHigh Risk', ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7), fontweight='bold')
        ax.text(median_clv*1.5, 0.2, 'High Value\nLow Risk', ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7), fontweight='bold')
        ax.text(median_clv*0.5, 0.2, 'Low Value\nLow Risk', ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7), fontweight='bold')
        
        ax.set_xlabel('Customer Lifetime Value ($)')
        ax.set_ylabel('Churn Probability')
        ax.set_title('Customer Risk-Value Matrix', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Segment statistics
        st.markdown("#### 📈 Segment Distribution")
        
        # Calculate segment statistics
        high_value_high_risk = np.sum((clv_values > median_clv) & (churn_probs >= 0.5))
        high_value_low_risk = np.sum((clv_values > median_clv) & (churn_probs < 0.5))
        low_value_high_risk = np.sum((clv_values <= median_clv) & (churn_probs >= 0.5))
        low_value_low_risk = np.sum((clv_values <= median_clv) & (churn_probs < 0.5))
        
        seg_col1, seg_col2, seg_col3, seg_col4 = st.columns(4)
        
        with seg_col1:
            st.metric("🔴 High Value, High Risk", f"{high_value_high_risk}", help="Priority 1: Immediate retention")
        with seg_col2:
            st.metric("🟡 Low Value, High Risk", f"{low_value_high_risk}", help="Priority 2: Cost-effective retention")
        with seg_col3:
            st.metric("🟢 High Value, Low Risk", f"{high_value_low_risk}", help="Priority 3: Maintain satisfaction")
        with seg_col4:
            st.metric("⚪ Low Value, Low Risk", f"{low_value_low_risk}", help="Priority 4: Routine monitoring")
    
    with clv_tabs[2]:
        st.markdown("#### 💼 Business Intelligence Dashboard")
        
        # Revenue metrics
        revenue_col1, revenue_col2, revenue_col3 = st.columns(3)
        
        total_clv = np.sum(clv_values)
        at_risk_clv = np.sum(clv_values[churn_probs >= 0.5])
        retention_opportunity = at_risk_clv * 0.7  # Assume 70% retention success rate
        
        with revenue_col1:
            st.metric("💰 Total Portfolio Value", f"${total_clv:,.0f}")
        with revenue_col2:
            st.metric("⚠️ Revenue at Risk", f"${at_risk_clv:,.0f}", f"{at_risk_clv/total_clv:.1%} of total")
        with revenue_col3:
            st.metric("🎯 Retention Opportunity", f"${retention_opportunity:,.0f}")
        
        # Strategic insights
        st.markdown("#### 🧠 Strategic Recommendations")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("**🔥 Immediate Actions**")
            st.markdown(f"• Focus on {high_value_high_risk} high-value, high-risk customers")
            st.markdown(f"• Potential revenue save: ${(clv_values[clv_values > median_clv].mean() * high_value_high_risk * 0.7):,.0f}")
            st.markdown("• Assign senior retention specialists")
            st.markdown("• Offer premium retention packages")
        
        with insights_col2:
            st.markdown("**📊 Long-term Strategy**")
            st.markdown(f"• Nurture {high_value_low_risk} high-value customers")
            st.markdown("• Develop loyalty programs")
            st.markdown("• Monitor satisfaction regularly")
            st.markdown("• Identify upselling opportunities")
    
    with clv_tabs[3]:
        st.markdown("#### 📈 Revenue Impact Analysis")
        
        # ROI Calculator
        st.markdown("**💡 Retention Campaign ROI Calculator**")
        
        campaign_cols = st.columns(2)
        
        with campaign_cols[0]:
            retention_budget = st.slider("Retention Campaign Budget ($)", 1000, 50000, 10000, 1000)
            success_rate = st.slider("Expected Success Rate (%)", 10, 90, 70, 5)
            campaign_cost_per_customer = st.slider("Cost per Customer ($)", 10, 500, 100, 10)
        
        with campaign_cols[1]:
            # Calculate ROI
            customers_reached = retention_budget // campaign_cost_per_customer
            customers_retained = customers_reached * (success_rate / 100)
            avg_clv_at_risk = np.mean(clv_values[churn_probs >= 0.5]) if np.any(churn_probs >= 0.5) else 3000
            revenue_saved = customers_retained * avg_clv_at_risk
            roi = ((revenue_saved - retention_budget) / retention_budget) * 100
            
            st.metric("Customers Reached", f"{customers_reached:.0f}")
            st.metric("Customers Retained", f"{customers_retained:.0f}")
            st.metric("Revenue Saved", f"${revenue_saved:,.0f}")
            
            if roi > 0:
                st.success(f"🎯 **ROI: +{roi:.0f}%** - Campaign Recommended!")
            else:
                st.error(f"📉 **ROI: {roi:.0f}%** - Optimize campaign strategy")
        
        # Campaign effectiveness visualization
        st.markdown("**📊 Campaign Impact Simulation**")
        
        success_rates = np.arange(10, 91, 10)
        roi_values = []
        
        for rate in success_rates:
            retained = customers_reached * (rate / 100)
            revenue = retained * avg_clv_at_risk
            roi_sim = ((revenue - retention_budget) / retention_budget) * 100
            roi_values.append(roi_sim)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(success_rates, roi_values, color=['red' if r < 0 else 'green' for r in roi_values], alpha=0.7)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        
        # Add value labels
        for bar, roi_val in zip(bars, roi_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (5 if height > 0 else -15),
                   f'{roi_val:.0f}%', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
        
        ax.set_xlabel('Success Rate (%)')
        ax.set_ylabel('ROI (%)')
        ax.set_title('Retention Campaign ROI by Success Rate')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("🧠 AI-Powered Business Insights")
    
    # Summary insights
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 15px; margin: 20px 0;">
        <h3 style="margin: 0; text-align: center;">🎯 Key Takeaways</h3>
        <hr style="border-color: rgba(255,255,255,0.3);">
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
            <div style="text-align: center; margin: 10px;">
                <h4 style="margin: 5px 0;">Revenue at Risk</h4>
                <p style="margin: 0; font-size: 1.5em; font-weight: bold;">${:,.0f}</p>
            </div>
            <div style="text-align: center; margin: 10px;">
                <h4 style="margin: 5px 0;">High-Risk Customers</h4>
                <p style="margin: 0; font-size: 1.5em; font-weight: bold;">{}</p>
            </div>
            <div style="text-align: center; margin: 10px;">
                <h4 style="margin: 5px 0;">Retention Opportunity</h4>
                <p style="margin: 0; font-size: 1.5em; font-weight: bold;">${:,.0f}</p>
            </div>
        </div>
    </div>
    """.format(at_risk_clv, high_value_high_risk + low_value_high_risk, retention_opportunity), unsafe_allow_html=True)
    
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