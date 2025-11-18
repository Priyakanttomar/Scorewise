# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import base64
from datetime import datetime

# --- Image Path Definition ---
IMAGE_PATH = 'images/loan.jpg' 

# -------------------------
# Utility functions (kept for completeness)
# -------------------------
def calculate_composite_score_with_prob(row, probability, features=None):
    """Calculate composite credit score (0-1000) using pre-calculated probability"""
    base_score = probability * 800
    
    # Bonus points for good behavior
    bonus = 0
    if row.get('PreviousLoanDefaults', 0) == 0:
        bonus += 50
    if row.get('PaymentHistory', 0) > 20:
        bonus += 50
    if row.get('UtilityBillsPaymentHistory', 0) > 0.8:
        bonus += 50
    if row.get('DebtToIncomeRatio', 1) < 0.3:
        bonus += 50

    # Penalty for high risk factors
    penalty = 0
    if row.get('PreviousLoanDefaults', 0) > 0:
        penalty += 100
    if row.get('DebtToIncomeRatio', 0) > 0.6:
        penalty += 50

    composite_score = base_score + bonus - penalty
    return max(0, min(1000, composite_score))


def assign_risk_band(row):
    score = row.get('CompositeScore', 0)
    income = row.get('AnnualIncome', 0)

    if score >= 700:
        return 'Low Risk-High Need' if income < 50000 else 'Low Risk-Low Need'
    elif score >= 500:
        return 'Medium Risk-High Need' if income < 50000 else 'Medium Risk-Low Need'
    else:
        return 'High Risk-High Need' if income < 50000 else 'High Risk-Low Need'

# --- Initialization for Navigation ---
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# -------------------------
# App styling
# -------------------------
st.set_page_config(page_title="ScoreWise — Loan Predictor", page_icon="🏦", layout="wide")
st.markdown(
    """
    <style>
    /* General Styles */
    .stApp { background: linear-gradient(180deg,#f6f9ff 0%, #ffffff 100%); }
    .card { background: white; padding: 1rem; border-radius: 12px; box-shadow: 0 6px 18px rgba(32,33,36,0.06); }
    .big-title { font-size:34px; font-weight:700; color:#1f77b4; }
    .sub { color:#2e86ab; font-weight:600; margin-bottom:0.5rem; }
    .metric { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white; padding:10px; border-radius:10px; text-align:center; }
    .small { color:#6b7280; font-size:13px }

    /* --- HOME PAGE STYLES --- */
    .home-container { padding: 10px 0; max-width: 100%; margin: 0; }
    .home-title { font-size: 4.5em; font-weight: 800; line-height: 1.1; margin-bottom: 0.05em; color: #000000; text-align: left; }
    .home-subtitle { font-size: 4.5em; font-weight: 800; line-height: 1.1; color: #4A90E2; margin-bottom: 0.5em; text-align: left; }
    .home-description { font-size: 1.1em; color: #616161; margin-bottom: 1.5em; line-height: 1.6; max-width: 700px; text-align: left; }
    .metric-card-home { background-color: #F0F8FF; color: #1a2a3a; padding: 15px; border-radius: 10px; text-align: center; margin: 5px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.3); border: 1px solid #E0EBF5; }
    .metric-card-home h4 { color: #4A90E2; font-size: 1em; margin-bottom: 3px; }
    .metric-card-home h3 { color: #1a2a3a; font-size: 1.8em; font-weight: 700; margin: 0; }
    .image-container { display: flex; justify-content: flex-end; align-items: center; height: 100%; padding-top: 20px; }
    
    /* ------------------------- UPDATED SIDEBAR STYLES ------------------------- */
    
    /* Logo Section (White Box Header) */
    .logo-container {
        background: white; 
        border-radius: 8px;
        padding: 15px; 
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1); 
        display: flex;
        align-items: center;
        justify-content: flex-start;
        width: 100%; /* Ensure it spans the full sidebar width */
    }
    .logo-icon { font-size: 28px; color: #1f77b4; margin-right: 10px; }
    .logo-text { font-size: 24px; font-weight: 700; color: #1f77b4; }
    
    /* Navigation Container */
    .nav-container { padding: 0; margin-bottom: 20px; }
    
    /* Active Item (Pure HTML/Markdown block) */
    .nav-item.active {
        background-color: #E8F4FD; /* Light blue background */
        color: #1f77b4; /* Dark blue text */
        font-weight: 600;
        box-shadow: 0 2px 6px rgba(31,119,180,0.1);
        border-left: 4px solid #1f77b4; 
        margin: 4px 0;
        padding: 12px 16px; /* Consistent padding */
        border-radius: 8px;
        display: flex;
        align-items: center;
        width: 100%;
    }
    
    /* Inactive Item (Targeting Streamlit's button structure) */
    /* This targets the button element inside the custom st.markdown style wrapper */
    .stButton button {
        background-color: white; /* Force white background for inactive state */
        color: #333; 
        padding: 12px 16px; /* Consistent padding with active state */
        border-radius: 8px;
        border: none;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05); /* Very light shadow for block effect */
        width: 100%;
        text-align: left; /* Align text left */
        display: flex; /* Use flex to align icon and text */
        align-items: center;
        font-weight: 500;
        font-size: 14px;
        margin: 4px 0;
        line-height: 1.2; /* Fix vertical alignment */
    }
    .stButton button:hover {
        background-color: #f0f4f8 !important; /* Slight hover effect */
        color: #1f77b4 !important;
    }
    
    /* Icon styling for both states */
    .nav-icon {
        margin-right: 12px;
        width: 20px;
        text-align: center;
        font-size: 16px;
    }
    
    /* User Section */
    .user-section { background: white; border-radius: 8px; padding: 15px; margin-top: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .user-name { font-weight: 600; color: #333; margin: 0; }
    .user-role { color: #666; font-size: 12px; margin: 0; }
    
    /* Fix for button labels containing custom HTML - this is the core fix! */
    /* We hide Streamlit's internal wrapper for the button content that displays the raw HTML tags */
    .stButton button > div:first-child { 
        display: flex;
        align-items: center;
        justify-content: flex-start;
        width: 100%; /* Crucial for full-width layout */
    }
    
    </style>
    """, unsafe_allow_html=True
)

# -------------------------
# Try loading model artifacts
# -------------------------
@st.cache_resource
def load_model_artifacts():
    try:
        with open('model/model.pkl', 'rb') as f:
            package = pickle.load(f)
        
        # Load components, using the consistent key 'categorical_cols'
        model = package.get('model')
        scaler = package.get('scaler')
        label_encoders = package.get('label_encoders', {})
        features = package.get('features')
        categorical_cols = package.get('categorical_cols', [])
        categorical_mappings = package.get('categorical_mappings', {})
        
        return {
            'model': model,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'features': features,
            'categorical_cols': categorical_cols,
            'categorical_mappings': categorical_mappings
        }
    except Exception as e:
        # st.error(f"Error loading model artifacts: {e}") 
        return None

artifacts = load_model_artifacts()
model_loaded = artifacts is not None

# -------------------------
# Layout / Sidebar
# -------------------------
with st.sidebar:
    # Logo section with text (inside white block)
    st.markdown("""
    <div class="logo-container">
        <div class="logo-icon">🏦</div>
        <div class="logo-text">ScoreWise</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation section styled like the image
    st.markdown("<div class='nav-container'>", unsafe_allow_html=True)
    
    # Define navigation items exactly as requested
    nav_items = [
        ("Home", "📊", "Home"),
        ("Loan Predictor", "📈", "Loan Predictor"), 
        ("Bulk Upload", "📁", "Bulk Upload"),
        ("Application History", "📋", "Application History"),
        ("About", "ℹ️", "About")
    ]
    
    # Create navigation items
    for item_name, icon, page_key in nav_items:
        is_active = st.session_state.page == page_key
        
        # 1. Use the clean HTML block for the ACTIVE state
        if is_active:
            st.markdown(f"<div class='nav-item active'><span class='nav-icon'>{icon}</span>{item_name}</div>", unsafe_allow_html=True)
        # 2. Use st.button for the INACTIVE state
        else:
            button_label_with_icon = f"{icon} {item_name}"
            if st.button(button_label_with_icon, key=page_key, use_container_width=True):
                st.session_state.page = page_key
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # User section at the bottom
    st.markdown("""
    <div class="user-section">
        <p class="user-name">Priyakant Tomar</p>
        <p class="user-role">Admin</p>
    </div>
    """, unsafe_allow_html=True)

page = st.session_state.page

# -------------------------
# Home
# -------------------------
if page == "Home":
    st.markdown("<div class='home-container'>", unsafe_allow_html=True)
    
    # Create two columns for the hero section: text on left, image on right
    text_col, image_col = st.columns([2, 1]) # 2/3 width for text, 1/3 for image

    with text_col:
        st.markdown("<div class='home-title'>ScoreWise Loan</div>", unsafe_allow_html=True)
        st.markdown("<div class='home-subtitle'>Predictor</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='home-description'>", unsafe_allow_html=True)
        st.write(
            "ScoreWise provides instant loan eligibility predictions, composite credit scores and an interpreted risk band. "
            "This demo shows integration between the trained Random Forest model and a clean Streamlit UI."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # --- Button for navigation ---
        if st.button("Prediction", key="home_prediction_button"):
            st.session_state.page = "Loan Predictor"
            st.rerun() 
            
    with image_col:
        # Use a custom div to control image alignment within its column
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        try:
            st.image(IMAGE_PATH, width=350) # Set a fixed width for the image
        except FileNotFoundError:
            st.error(f"Image not found. Please ensure '{IMAGE_PATH}' exists.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)

    # --- Metrics Section with light blue cards ---
    st.markdown("<h4>Our Performance</h4>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    col1.markdown("<div class='metric-card-home'><h4>Total Predictions</h4><h3>25,234</h3></div>", unsafe_allow_html=True)
    col2.markdown("<div class='metric-card-home'><h4>Accuracy (Reported)</h4><h3>91.2%</h3></div>", unsafe_allow_html=True)
    col3.markdown("<div class='metric-card-home'><h4>Avg. Latency</h4><h3>2.3s</h3></div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True) # Close .home-container

# -------------------------
# Loan Predictor
# -------------------------
elif page == "Loan Predictor":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin:0'>📈 Loan Eligibility — Single Applicant</h3>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Fill the form below and click Predict. If model artifacts are not available the app will run a demo prediction.</div>", unsafe_allow_html=True)
    st.write("")

    with st.form("single_form"):
        left, right = st.columns(2)

        with left:
            annual_income = st.number_input("Annual Income (USD)", min_value=0, value=50000, step=1000)
            monthly_income = st.number_input("Monthly Income (USD)", min_value=0, value=4166, step=100)
            credit_score = st.slider("Credit Score", 300, 850, 650)
            debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.4, 0.01)
            payment_history = st.slider("Payment History Score (0-30)", 0, 30, 18)
            previous_defaults = st.number_input("Previous Loan Defaults", min_value=0, value=0, step=1)

        with right:
            loan_amount = st.number_input("Requested Loan Amount (USD)", min_value=0, value=10000, step=500)
            loan_duration = st.slider("Loan Duration (months)", 6, 120, 36)
            interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=8.0, step=0.1)
            monthly_loan_payment = st.number_input("Monthly Loan Payment (USD)", min_value=0, value=420, step=10)
            job_tenure = st.number_input("Job Tenure (months)", min_value=0, value=24, step=1)
            # FIX 1: ADD MISSING FEATURE INPUT
            length_of_credit_history = st.number_input("Length of Credit History (months)", min_value=0, value=60, step=6) 
            utility_payment_history = st.slider("Utility Bills Payment History (0-1)", 0.0, 1.0, 0.85, 0.01)

        st.markdown("#### Employment & Home")
        employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-Employed"])
        home_ownership = st.selectbox("Home Ownership", ["Owned", "Rented", "Mortgaged"]) 

        submitted = st.form_submit_button("🎯 Predict Loan Eligibility")

    if submitted:
        # prepare input dict with keys that the model expects
        input_dict = {
            'AnnualIncome': annual_income,
            'MonthlyIncome': monthly_income,
            'CreditScore': credit_score,
            'DebtToIncomeRatio': debt_to_income,
            'PaymentHistory': payment_history,
            'PreviousLoanDefaults': previous_defaults,
            'LoanAmount': loan_amount,
            'LoanDuration': loan_duration,
            'InterestRate': interest_rate / 100.0 if interest_rate > 1 else interest_rate,   
            'MonthlyLoanPayment': monthly_loan_payment,
            'JobTenure': job_tenure,
            # FIX 1: ADD MISSING FEATURE TO DICT
            'LengthOfCreditHistory': length_of_credit_history, 
            'UtilityBillsPaymentHistory': utility_payment_history,
            # FIX 2: NORMALIZE CATEGORICAL INPUTS TO LOWERCASE 
            # (Assumes training data was mostly/all lowercase)
            'EmploymentStatus': employment_status.lower(),
            'HomeOwnershipStatus': home_ownership.lower()
        }

        st.info("Preparing data and running prediction...")
        # If model artifacts loaded, use them. Otherwise simulate.
        if model_loaded:
            pkg = artifacts
            model = pkg['model']
            scaler = pkg['scaler']
            label_encoders = pkg.get('label_encoders', {})
            features = pkg['features'] or []
            categorical_cols = pkg.get('categorical_cols', [])
            categorical_mappings = pkg.get('categorical_mappings', {})

            # 1. Start with a copy of input_dict
            row = input_dict.copy()
            
            # 2. Encode categoricals using label encoders (FIXED LOGIC)
            for cat in categorical_cols:
                if cat in row:
                    le = label_encoders.get(cat)
                    user_input = row[cat] 
                    
                    if le is not None:
                        try:
                            # Attempt to transform the user input string into the learned integer value
                            row[cat] = int(le.transform([user_input])[0])
                        except Exception:
                            # If transform fails (unseen category/casing issue), set to 0 
                            st.warning(f"Category '{user_input}' not found in training data for {cat}. Defaulting to 0.")
                            row[cat] = 0
                    else:
                        # Safety net: If no encoder was saved, default to a number
                        row[cat] = 0

            # 3. Make DataFrame using only the features the model expects, now all numerical
            # This should now work as LengthOfCreditHistory is in 'row'
            X = pd.DataFrame([row])[features]
            
            try:
                # This should now work as X only contains numbers
                X_scaled = scaler.transform(X) 
                proba = model.predict_proba(X_scaled)[0][1]
                pred_label = model.predict(X_scaled)[0]
                prediction_text = "APPROVED" if int(pred_label) == 1 else "REJECTED"
            except Exception as e:
                st.error(f"Model prediction failed: {e}. Please check your model features and scaling.")
                X_scaled = None
                proba = 0.0
                prediction_text = "ERROR"

            # The rest of the scoring logic uses the correctly encoded 'row' dictionary
            composite_score = calculate_composite_score_with_prob(row, proba, features)
            row['CompositeScore'] = composite_score
            risk_band = assign_risk_band(row)

            # Results display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                status_color = "🟢" if prediction_text == 'APPROVED' else "🔴"
                st.markdown(f"<div class='metric'><h4 style='margin:4px'>Decision</h4><h3 style='margin:4px'>{status_color} {prediction_text}</h3></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric'><h4 style='margin:4px'>Probability</h4><h3 style='margin:4px'>{proba:.1%}</h3></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='metric'><h4 style='margin:4px'>Score</h4><h3 style='margin:4px'>{composite_score:.0f}/1000</h3></div>", unsafe_allow_html=True)
            with col4:
                risk_color = "🟢" if "Low" in risk_band else "🟡" if "Medium" in risk_band else "🔴"
                st.markdown(f"<div class='metric'><h4 style='margin:4px'>Risk Band</h4><h3 style='margin:4px'>{risk_color} {risk_band}</h3></div>", unsafe_allow_html=True)

            st.markdown("---")
            st.write("#### 📊 Key Factors & Recommendations")
            st.write(f"- **Credit Score:** {credit_score} — {'Good' if credit_score >= 650 else 'Needs improvement'}")
            st.write(f"- **Income:** ${annual_income:,.0f} (DTI: {debt_to_income:.2f})")
            st.write(f"- **Payment History:** {payment_history}/30")
            if prediction_text == "APPROVED":
                st.success("✅ Strong candidate for approval. Consider standard interest rates and finalize underwriting checks.")
            else:
                st.error("❌ Candidate likely to be rejected. Recommend improving credit metrics or reducing debt.")

            # Downloadable mini-report CSV
            report_df = pd.DataFrame([row])
            csv = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📄 Download Report (CSV)", data=csv, file_name=f"scorewise_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime='text/csv')

        else:
            # Demo/simulated path (no model artifacts)
            st.warning("Model artifacts not found — running demo/simulated prediction. To use real predictions, place `model/model.pkl` and `model/scaler.pkl` in `./model/`.")
            # simple heuristic demo
            score_base = (credit_score - 300) / (850 - 300)  # 0-1
            income_boost = min(annual_income / 100000, 1.0)
            dti_penalty = max(0, (debt_to_income - 0.4)) * -1
            proba = max(0.01, min(0.99, 0.5 * score_base + 0.4 * income_boost + 0.2 * (1 - debt_to_income)))
            composite_score = calculate_composite_score_with_prob(input_dict, proba)
            prediction_text = "APPROVED" if proba >= 0.5 else "REJECTED"
            risk_band = assign_risk_band({'CompositeScore': composite_score, 'AnnualIncome': annual_income})

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Decision", prediction_text)
            col2.metric("Probability", f"{proba:.1%}")
            col3.metric("Composite Score", f"{composite_score:.0f}/1000")
            col4.metric("Risk Band", risk_band)

            st.markdown("---")
            st.write("Recommendations (demo):")
            if prediction_text == "APPROVED":
                st.success("✅ Demo: Looks good — proceed to underwriting.")
            else:
                st.error("❌ Demo: Improve credit score and reduce debt-to-income ratio.")
    st.markdown("</div>", unsafe_allow_html=True) 

# -------------------------
# Bulk Upload
# -------------------------
elif page == "Bulk Upload":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>📁 Bulk Upload (CSV)</h3>", unsafe_allow_html=True)
    st.write("Upload a CSV with columns matching your training `features` and `LoanApproved` (optional).")
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} rows.")
            st.dataframe(df.head())

            if st.button("Process all rows"):
                if not model_loaded:
                    st.error("Model artifacts are not available. Bulk processing requires model/model.pkl and model/scaler.pkl present.")
                else:
                    pkg = artifacts
                    model = pkg['model']
                    scaler = pkg['scaler']
                    label_encoders = pkg.get('label_encoders', {})
                    features = pkg['features']

                    # ensure features present
                    missing = [f for f  in features if f not in df.columns]
                    if missing:
                        st.error(f"Missing required features: {missing}")
                    else:
                        X = df[features].copy()
                        # encode categoricals
                        for cat, le in (pkg.get('label_encoders') or {}).items():
                            if cat in X.columns:
                                try:
                                    X[cat] = le.transform(X[cat].str.lower()) # Normalizing bulk upload data as well
                                except:
                                    # unknown categories -> map to mode or 0
                                    X[cat] = X[cat].map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

                        X_scaled = scaler.transform(X)
                        probs = model.predict_proba(X_scaled)[:, 1]
                        df['CompositeScore'] = [calculate_composite_score_with_prob(df.iloc[i].to_dict(), probs[i]) for i in range(len(df))]
                        df['RiskBand'] = df.apply(assign_risk_band, axis=1)
                        df['ApprovalProbability'] = probs
                        st.success("Processed successfully.")
                        st.dataframe(df.head())

                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download results CSV", csv, "scorewise_bulk_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Application History
# -------------------------
elif page == "Application History":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>📋 Application History</h3>", unsafe_allow_html=True)
    sample = pd.DataFrame({
        'Date': ['2024-01-15', '2024-01-10', '2024-01-05', '2023-12-28', '2023-12-20'],
        'Applicant': ['John Smith', 'Sarah Johnson', 'Mike Brown', 'Emily Davis', 'David Wilson'],
        'Loan Amount': [15000, 25000, 10000, 30000, 18000],
        'Status': ['APPROVED', 'REJECTED', 'APPROVED', 'APPROVED', 'REJECTED'],
        'Score': [720, 450, 680, 710, 380],
        'Risk Band': ['Low Risk', 'High Risk', 'Medium Risk', 'Low Risk', 'High Risk']
    })
    st.dataframe(sample)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# About
# -------------------------
elif page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>ℹ️ About ScoreWise</h3>", unsafe_allow_html=True)
    st.write("ScoreWise leverages Random Forest-based models to provide quick loan eligibility decisions and composite credit scores.")
    st.write("This app demonstrates a production-like Flow: input -> preprocess -> predict -> score -> risk band -> downloadable report.")
    if model_loaded:
        st.success("Model artifacts loaded successfully from `model/model.pkl`.")
    else:
        st.warning("Model artifacts are not available. Place `model/model.pkl` and `model/scaler.pkl` in `./model/` for real predictions.")
    st.markdown("</div>", unsafe_allow_html=True)
