import streamlit as st
import pandas as pd
import joblib
import os
import warnings 

# ─────────────────────────────────────────────────────────────────────────────
# PAGE SETUP & STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

# Custom CSS for Amber, Purple, and Green theme
st.markdown(f"""
    <style>
    /* 1. Deepest Purple Background */
    .stAppViewContainer {{
        background-color: #300049;
    }}

    /* 2. Patient Profile Container: Indigo */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: #4A0072 !important;
        border: 1px solid #D5B4E7 !important; /* Wisteria border */
        padding: 30px !important;
        border-radius: 15px !important;
    }}

    /* 3. Typography: Wisteria & White */
    h1, h2, h3 {{
        color: #D5B4E7 !important; /* Wisteria */
    }}
    
    label {{
        color: #FFFFFF !important; /* White labels for contrast */
        font-weight: 500 !important;
    }}

    /* 4. Inputs: Muted Teal Tint */
    input, div[data-baseweb="select"] {{
        background-color: #F8F4FF !important; /* Very soft lilac tint */
        border-radius: 8px !important;
    }}

    /* 5. The "Predict" Button: Teal */
    .stButton>button {{
        background-color: #00838F !important; /* Teal from palette */
        color: white !important;
        border: none !important;
        font-weight: bold !important;
        width: 100%;
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        background-color: #00695C !important; /* Darker Pine Green on hover */
        box-shadow: 0px 4px 15px rgba(0, 131, 143, 0.4);
    }}

    /* 6. Metrics: Wisteria labels and Teal values */
    [data-testid="stMetricLabel"] {{ color: #D5B4E7 !important; }}
    [data-testid="stMetricValue"] {{ color: #FFFFFF !important; }}

    /* 7. Success/Error Boxes: Muted Pine Green */
    div[data-testid="stNotificationContentSuccess"] {{
        background-color: #00695C !important;
        color: white !important;
        border: none !important;
    }}
    </style>
    """, unsafe_allow_html=True)




@st.cache_resource
def load_model():
    if not os.path.exists("patients_pipeline.pkl"):
        return None
    return joblib.load("patients_pipeline.pkl")

model = load_model()

if model is None:
    st.error("⚠️ Model file 'patients_pipeline.pkl' not found.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Predictor")
st.markdown('<p style="color: #FAFAFA; font-size: 1.1rem; opacity: 0.9;">Fill in the patient details below to assess metabolic risk.</p>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────────────────────────────────────
with st.container():

    st.subheader("📋 Patient Profile")
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 3)
        glucose     = st.slider("Glucose (mg/dL)", 40, 300, 120)
        bp          = st.slider("Blood Pressure (mm Hg)", 20, 130, 72)
        skin        = st.number_input("Skin Thickness (mm)", 5, 100, 23)

    with col2:
        insulin     = st.number_input("Insulin (uU/mL)", 0, 900, 79)
        bmi         = st.number_input("BMI (kg/m2)", 15.0, 70.0, 32.0, 0.1)
        dpf         = st.number_input("Pedigree Function", 0.0, 3.0, 0.47, 0.001)
        age         = st.number_input("Age (years)", 21, 90, 35)

    c1, c2 = st.columns(2)
    with c1:
        bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"], index=2)
    with c2:
        blood_type = st.selectbox("Blood Type", ["A", "B", "AB", "O"], index=3)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def build_input_row():
    bmi_cat_enc = ["Underweight", "Normal", "Overweight", "Obese"].index(bmi_category)
    insulin_glucose_ratio = insulin / (glucose + 1)
    return pd.DataFrame([{
        "pregnancies": pregnancies, "glucose": glucose, "blood_pressure": bp,
        "skin_thickness": skin, "insulin": insulin, "bmi": bmi,
        "diabetes_pedigree_function": dpf, "age": age,
        "bmi_category_enc": bmi_cat_enc, "insulin_glucose_ratio": insulin_glucose_ratio,
        "blood_B": int(blood_type == "B"), "blood_AB": int(blood_type == "AB"), "blood_O": int(blood_type == "O")
    }])

if st.button("Generate Risk Assessment", use_container_width=True):
    row = build_input_row()
    
    # This block silences that specific warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob = model.predict_proba(row)
    
    p_dm = prob[0][1]
    
    st.divider()

    # Dynamic result display using your Green and Amber colors
    if p_dm < 0.3:
        st.success(f"### Low Risk Identified")
        color = "green"
    elif p_dm < 0.6:
        st.warning(f"### Moderate Risk Identified")
        color = "orange" # Amber
    else:
        st.error(f"### High Risk Identified")
        color = "red"

    m1, m2, m3 = st.columns(3)
    m1.metric("Diabetes Prob.", f"{p_dm:.1%}")
    m2.metric("Health Score", f"{100-(p_dm*100):.0f}/100")
    m3.metric("Status", "Stable" if p_dm < 0.5 else "Action Req.")

    st.markdown(f"**Risk Confidence**")
    st.progress(float(p_dm))
    
    st.info("💡 **Recommendation:** " + 
            ("Maintain current lifestyle." if p_dm < 0.3 else "Consult a specialist for further testing."))

st.caption("Educational tool only. Data is not stored.")

