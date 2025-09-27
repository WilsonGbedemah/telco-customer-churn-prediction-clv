# Standard library imports
import os
import sys
import io

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
def get_customer_segment(churn_prob, clv):
    """Classify customer into business segments based on churn risk and CLV."""
    if churn_prob >= 0.7 and clv >= 2000:
        return "🚨 Critical Risk - High Value", "#FF4B4B"
    elif churn_prob >= 0.7 and clv < 2000:
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
    
    if churn_prob >= 0.7:
        if clv >= 2000:
            strategies.append("🎯 **Executive Intervention**: Personal call from account manager")
            strategies.append("💳 **Premium Offers**: 20-30% discount or service upgrades")
        else:
            strategies.append("📞 **Retention Call**: Automated or junior staff outreach")
            strategies.append("💰 **Cost-Effective Offers**: 10-15% discount or loyalty rewards")
    
    if customer_data.get('Contract_Month-to-month', 0) == 1:
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
            'Risk_Level': 'High' if churn_prob >= 0.5 else 'Low',
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
tabs = st.tabs(["🎯 Churn Prediction", "📊 Batch Analysis", "🔍 What-If Analysis", "📈 Model Performance", "💰 CLV Overview"])

# --- Predict Tab ---
with tabs[0]:
    st.header("🎯 Individual Customer Churn Prediction")
    st.markdown("Enter customer details below to predict churn risk and receive personalized retention recommendations.")
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📋 Customer Profile")
        
        # Demographic Information
        st.markdown("**👤 Demographics**")
        st.caption("Basic customer information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        
        st.markdown("---")
        
        # Account Information
        st.markdown("**💳 Account Details**")
        st.caption("Contract and billing information")
        contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'], 
                               help="Contract duration affects customer commitment and churn risk")
        
        # Replace sliders with number inputs
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12,
                               help="Number of months the customer has been with the company. Longer tenure typically means lower churn risk.")
        
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0, step=0.50,
                                        help="Monthly amount charged to customer. Higher charges may increase churn risk.")
        
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
        st.markdown("---")
        
        # Services Information  
        st.markdown("**📞 Services**")
        st.caption("Communication and internet services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'],
                                      help="Internet service type affects customer satisfaction and churn")
        
        if internet_service != 'No':
            online_security = st.selectbox("Online Security", ["No", "Yes"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes"],
                                      help="Technical support reduces churn risk significantly")
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
        else:
            online_security = online_backup = device_protection = tech_support = streaming_tv = streaming_movies = "No internet service"
        
        model_choice_predict = st.selectbox("🤖 Prediction Model", 
                                          ['Logistic Regression', 'Random Forest', 'XGBoost'], 
                                          key='predict',
                                          help="Choose the machine learning model for prediction")

    with col2:
        st.subheader("🎯 Prediction Results")
        
        if st.button("🔮 Predict Churn Risk", use_container_width=True, type="primary"):
            # Create input dataframe with all features
            input_df = pd.DataFrame(columns=X_test.columns)
            input_df.loc[0] = 0
            
            # Map all inputs to encoded features
            input_df['tenure'] = tenure
            input_df['MonthlyCharges'] = monthly_charges
            input_df[f'gender_{gender}'] = 1
            input_df[f'SeniorCitizen_{1 if senior_citizen == "Yes" else 0}'] = 1
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
            
            # Feature engineering
            input_df['services_count'] = sum([
                1 for service in [phone_service, online_security, online_backup, 
                                device_protection, tech_support, streaming_tv, streaming_movies]
                if service == "Yes"
            ])
            
            model_name_map = {'Logistic Regression': 'logisticregression', 'Random Forest': 'randomforest', 'XGBoost': 'xgboost'}
            model = models[model_name_map[model_choice_predict]]

            churn_prob = model.predict_proba(input_df[X_test.columns])[:, 1][0]
            clv = monthly_charges * avg_tenure
            segment, segment_color = get_customer_segment(churn_prob, clv)
            
            # Display main results
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                risk_level = "🔴 HIGH RISK" if churn_prob >= 0.5 else "🟢 LOW RISK"
                st.metric(label="Churn Probability", value=f"{churn_prob:.1%}", 
                         delta=risk_level)
            
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
            else:
                st.success("✅ Customer is low risk - maintain current service level")

            # SHAP Explanation - Always show after prediction
            st.markdown("---")
            st.subheader("🔍 Why This Prediction?")
            st.markdown("Understanding which factors drive the churn prediction for this customer:")
            
            if SHAP_AVAILABLE:
                explainer = get_shap_explainer(model, X_test.iloc[:100])
                if explainer is not None:
                    try:
                        shap_values = explainer(input_df[X_test.columns])
                        
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
                st.dataframe(results_df.round(3), use_container_width=True)
                
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

# --- What-If Analysis Tab ---
with tabs[2]:
    st.header("🔍 What-If Scenario Analysis")
    st.markdown("Explore how different changes affect customer churn probability and test retention strategies.")
    
    # Base customer profile
    st.subheader("👤 Base Customer Profile")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        base_tenure = st.number_input("Base Tenure (months)", 0, 72, 12, key="base_tenure")
        base_charges = st.number_input("Base Monthly Charges ($)", 18.0, 120.0, 70.0, key="base_charges")
    with col2:
        base_contract = st.selectbox("Base Contract", ['Month-to-month', 'One year', 'Two year'], key="base_contract")
        base_internet = st.selectbox("Base Internet Service", ['DSL', 'Fiber optic', 'No'], key="base_internet")
    with col3:
        base_tech_support = st.selectbox("Base Tech Support", ['No', 'Yes', 'No internet service'], key="base_tech")
        scenario_model = st.selectbox("Model", ['Logistic Regression', 'Random Forest', 'XGBoost'], key="scenario")
    
    # Calculate base prediction
    if st.button("🔮 Calculate Base Scenario", key="base_calc"):
        # Create base input
        base_input = pd.DataFrame(columns=X_test.columns)
        base_input.loc[0] = 0
        base_input['tenure'] = base_tenure
        base_input['MonthlyCharges'] = base_charges
        base_input[f'Contract_{base_contract}'] = 1
        base_input[f'InternetService_{base_internet}'] = 1
        base_input[f'TechSupport_{base_tech_support}'] = 1
        
        model_name_map = {'Logistic Regression': 'logisticregression', 'Random Forest': 'randomforest', 'XGBoost': 'xgboost'}
        model = models[model_name_map[scenario_model]]
        
        base_prob = model.predict_proba(base_input[X_test.columns])[:, 1][0]
        base_clv = base_charges * avg_tenure
        
        st.session_state['base_prob'] = base_prob
        st.session_state['base_clv'] = base_clv
        st.session_state['base_input'] = base_input
    
    if 'base_prob' in st.session_state:
        st.markdown("### 📊 Base Scenario Results")
        col_base1, col_base2 = st.columns(2)
        with col_base1:
            st.metric("Base Churn Probability", f"{st.session_state['base_prob']:.1%}")
        with col_base2:
            st.metric("Base CLV", f"${st.session_state['base_clv']:,.2f}")
        
        st.markdown("---")
        st.subheader("🎯 Scenario Testing")
        st.markdown("Test different interventions and see their impact:")
        
        scenarios = []
        scenario_names = []
        
        # Scenario 1: Contract Upgrade
        if st.checkbox("📋 Scenario: Upgrade to Annual Contract"):
            scenario1_input = st.session_state['base_input'].copy()
            scenario1_input[f'Contract_{base_contract}'] = 0
            scenario1_input['Contract_One year'] = 1
            
            scenario1_prob = model.predict_proba(scenario1_input[X_test.columns])[:, 1][0]
            scenarios.append(scenario1_prob)
            scenario_names.append("Annual Contract")
        
        # Scenario 2: Add Tech Support  
        if st.checkbox("🛠️ Scenario: Add Tech Support"):
            scenario2_input = st.session_state['base_input'].copy()
            scenario2_input[f'TechSupport_{base_tech_support}'] = 0
            scenario2_input['TechSupport_Yes'] = 1
            
            scenario2_prob = model.predict_proba(scenario2_input[X_test.columns])[:, 1][0]
            scenarios.append(scenario2_prob)
            scenario_names.append("With Tech Support")
        
        # Scenario 3: Discount
        discount_pct = st.slider("💰 Scenario: Apply Discount (%)", 0, 50, 15)
        if discount_pct > 0:
            scenario3_input = st.session_state['base_input'].copy()
            discounted_charges = base_charges * (1 - discount_pct/100)
            scenario3_input['MonthlyCharges'] = discounted_charges
            
            scenario3_prob = model.predict_proba(scenario3_input[X_test.columns])[:, 1][0]
            scenarios.append(scenario3_prob)
            scenario_names.append(f"{discount_pct}% Discount")
        
        # Display comparison
        if scenarios:
            st.markdown("### 📈 Scenario Comparison")
            comparison_data = {
                'Scenario': ['Base'] + scenario_names,
                'Churn Probability': [st.session_state['base_prob']] + scenarios,
                'Risk Reduction': [0] + [st.session_state['base_prob'] - prob for prob in scenarios]
            }
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df['Risk Reduction %'] = (comparison_df['Risk Reduction'] / st.session_state['base_prob'] * 100).round(1)
            
            st.dataframe(comparison_df.round(3), use_container_width=True)
            
            # Visualization
            fig_scenarios, ax_scenarios = plt.subplots(figsize=(10, 5))
            colors = ['red'] + ['green' if prob < st.session_state['base_prob'] else 'orange' for prob in scenarios]
            bars = ax_scenarios.bar(comparison_df['Scenario'], comparison_df['Churn Probability'], color=colors, alpha=0.7)
            ax_scenarios.set_ylabel('Churn Probability')
            ax_scenarios.set_title('Scenario Impact Analysis')
            ax_scenarios.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, prob in zip(bars, comparison_df['Churn Probability']):
                ax_scenarios.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                f'{prob:.1%}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig_scenarios)
# --- Model Performance Tab ---
with tabs[3]:
    st.header("📈 Model Performance Evaluation")
    st.markdown("Compare the performance of different machine learning models on the test dataset.")
    
    # Performance metrics table
    performance_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Precision': [0.5092, 0.5494, 0.5167],
        'Recall': [0.8128, 0.7139, 0.7861],
        'F1-Score': [0.6262, 0.6209, 0.6235],
        'AUC-ROC': [0.8366, 0.8317, 0.8316]
    }
    performance_df = pd.DataFrame(performance_data).set_index('Model')
    
    st.subheader("📊 Performance Metrics")
    st.dataframe(performance_df.round(4), use_container_width=True)
    
    # Explanation of metrics
    with st.expander("📖 Understanding the Metrics"):
        st.markdown("""
        - **Precision**: Of all customers predicted to churn, how many actually churned? (Higher = fewer false alarms)
        - **Recall**: Of all customers who actually churned, how many did we catch? (Higher = fewer missed churners)  
        - **F1-Score**: Balanced metric combining precision and recall
        - **AUC-ROC**: Overall model discriminative ability (0.5 = random, 1.0 = perfect)
        """)

    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🎯 Global Feature Importance")
        st.markdown("Most important features across the entire dataset")
        
        if SHAP_AVAILABLE:
            st.write("**Method: SHAP Analysis**")
            model_choice_shap = st.selectbox("Select Model for SHAP Summary", 
                                           ['Logistic Regression', 'Random Forest', 'XGBoost'], 
                                           key='shap')
            model_name_map_shap = {'Logistic Regression': 'logisticregression', 
                                 'Random Forest': 'randomforest', 'XGBoost': 'xgboost'}
            selected_model_shap = models[model_name_map_shap[model_choice_shap]]
            explainer_shap = get_shap_explainer(selected_model_shap, X_test)
            
            if explainer_shap is not None:
                try:
                    shap_values_shap = explainer_shap(X_test)
                    fig_summary, ax_summary = plt.subplots(figsize=(8, 6))
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
            model_choice_fi = st.selectbox("Select Model for Feature Importance", 
                                         ['Logistic Regression', 'Random Forest', 'XGBoost'], 
                                         key='fi')
            model_name_map_fi = {'Logistic Regression': 'logisticregression', 
                               'Random Forest': 'randomforest', 'XGBoost': 'xgboost'}
            _show_feature_importance_global(model_name_map_fi[model_choice_fi])

    with col4:
        st.subheader("📈 ROC Curve Analysis")  
        st.markdown("Model discrimination ability visualization")
        
        model_choice_roc = st.selectbox("Select Model for ROC Curve", 
                                      ['Logistic Regression', 'Random Forest', 'XGBoost'], 
                                      key='roc')
        model_name_map_roc = {'Logistic Regression': 'logisticregression', 
                            'Random Forest': 'randomforest', 'XGBoost': 'xgboost'}
        selected_model_roc = models[model_name_map_roc[model_choice_roc]]
        y_pred_proba_roc = selected_model_roc.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_roc)
        
        fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
        ax_roc.plot(fpr, tpr, linewidth=2, 
                   label=f'AUC = {roc_auc_score(y_test, y_pred_proba_roc):.3f}')
        ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Guess (AUC = 0.5)')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'{model_choice_roc} ROC Curve')
        ax_roc.legend()
        ax_roc.grid(True, alpha=0.3)
        st.pyplot(fig_roc)

# --- CLV Overview Tab ---
with tabs[4]:
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