import streamlit as st
import joblib
import os
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Premium Cardio Risk AI",
    layout="centered"
)

# ---------------- GLOBAL STYLE ----------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #ffe9e9 0%, #f5e9ff 100%);
}

/* Glass Card */
.glass-card {
    background: rgba(255, 255, 255, 0.80);
    border-radius: 22px;
    padding: 26px;
    box-shadow: 0 10px 26px rgba(0,0,0,0.15);
    backdrop-filter: blur(8px);
    margin-bottom: 22px;
}

/* Premium Header */
h1{
    color:#8b0000;
    font-weight:900;
    text-align:center;
}

/* Section Headings */
h3{
    color:#b71c1c;
    font-weight:800;
}

/* Button */
div.stButton > button {
    background: linear-gradient(135deg, #d32f2f 0%, #8e0000 100%);
    color:white;
    border-radius:14px;
    border:none;
    height:3em;
    width:100%;
    font-size:18px;
}
div.stButton > button:hover{
    background: linear-gradient(135deg, #ff1744 0%, #c62828 100%);
}

/* Risk box */
.risk-box{
    text-align:center;
    font-size:20px;
    padding:14px;
    border-radius:14px;
    font-weight:700;
}
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown("""
<h1>🫀 Premium AI Cardio Risk Predictor</h1>
<p style="text-align:center">
AI-assisted Cardiovascular Disease Risk Assessment
</p>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
MODEL_PATH = "model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except:
        st.error("⚠️ Could not load model.pkl.")
else:
    st.error("⚠️ model.pkl not found.")


# ---------------- PATIENT DETAILS ----------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("👤 Patient Details")

c1,c2 = st.columns(2)
with c1:
    gender_ui = st.selectbox("Gender",["Female","Male"])
with c2:
    age = st.number_input("Age (years)",1,120,45)

gender = 1 if gender_ui=="Male" else 0
st.markdown('</div>', unsafe_allow_html=True)


# ---------------- BODY ----------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📏 Body Measurements")

c1,c2 = st.columns(2)
with c1:
    height = st.number_input("Height (cm)",100,250,170)
with c2:
    weight = st.number_input("Weight (kg)",30.0,300.0,70.0)

bmi = weight / ((height/100)**2)
st.info(f"Calculated BMI: {bmi:.2f}")
st.markdown('</div>', unsafe_allow_html=True)


# ---------------- BLOOD PRESSURE ----------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("❤️ Blood Pressure")

c1,c2 = st.columns(2)
with c1:
    ap_hi = st.number_input("Systolic BP",80,250,120)
with c2:
    ap_lo = st.number_input("Diastolic BP",40,150,80)

pulse_pressure = ap_hi - ap_lo
st.info(f"Pulse Pressure: {pulse_pressure} mmHg")
st.markdown('</div>', unsafe_allow_html=True)


# ---------------- BLOOD TEST ----------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("🧪 Blood Test Results")

c1,c2 = st.columns(2)
with c1:
    chol_ui = st.selectbox("Cholesterol",["Normal","Above Normal","Well Above Normal"])
with c2:
    gluc_ui = st.selectbox("Glucose",["Normal","Above Normal","Well Above Normal"])

cholesterol = {"Normal":1,"Above Normal":2,"Well Above Normal":3}[chol_ui]
gluc = {"Normal":1,"Above Normal":2,"Well Above Normal":3}[gluc_ui]
st.markdown('</div>', unsafe_allow_html=True)


# ---------------- LIFESTYLE ----------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("🚶 Lifestyle")

c1,c2,c3 = st.columns(3)
with c1:
    smoke = 1 if st.radio("Smoking",["No","Yes"])=="Yes" else 0
with c2:
    alco = 1 if st.radio("Alcohol",["No","Yes"])=="Yes" else 0
with c3:
    active = 1 if st.radio("Physically Active",["No","Yes"])=="Yes" else 0

st.markdown('</div>', unsafe_allow_html=True)


# ---------------- BUTTON ----------------
pressed = st.button("🔍 Check My Cardio Risk")


# ---------------- PREDICTION ----------------
if pressed:

    if model is None:
        st.error("⚠️ Model not loaded.")

    elif ap_lo > ap_hi:
        st.error("⚠️ Diastolic BP must be less than Systolic BP.")

    else:
        data = {
            "gender": gender,
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": cholesterol,
            "gluc": gluc,
            "smoke": smoke,
            "alco": alco,
            "active": active,
            "age_years": age,
            "bmi": bmi,
            "pulse_pressure": pulse_pressure
        }

        df = pd.DataFrame([data])

        if hasattr(model,"feature_names_in_"):
            df = df.reindex(columns=model.feature_names_in_, fill_value=0)

        pred = model.predict(df)
        risk = model.predict_proba(df)[0][1]

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📊 AI Risk Assessment")

        if risk < 0.30:
            risk_text = "LOW RISK"
            color = "#2e7d32"
        elif risk < 0.60:
            risk_text = "MODERATE RISK"
            color = "#f9a825"
        else:
            risk_text = "HIGH RISK"
            color = "#c62828"

        st.markdown(
            f"<div class='risk-box' style='background:{color};color:white'>{risk_text}</div>",
            unsafe_allow_html=True
        )

        st.metric("Risk Probability", f"{risk*100:.2f}%")
        st.progress(float(risk))

        st.markdown('</div>', unsafe_allow_html=True)


        # ---------- DOCTOR STYLE RECOMMENDATIONS ----------
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("👨‍⚕️ Doctor-Style Recommendations")

        # General recommendation based on risk
        if risk < 0.30:
            st.success("Your risk is low. Continue maintaining a healthy lifestyle.")
        elif risk < 0.60:
            st.warning("Your risk is moderate. Consider lifestyle improvements and regular check-ups.")
        else:
            st.error("Your risk is high. Medical consultation is strongly recommended.")

        # --- BMI advice ---
        if bmi < 18.5:
            st.info("• Your BMI is low. A balanced calorie-rich diet may be helpful.")
        elif bmi <= 24.9:
            st.success("• Your BMI is in the healthy range. Keep it up.")
        elif bmi <= 29.9:
            st.warning("• Your BMI indicates overweight. Weight reduction may reduce heart risk.")
        else:
            st.error("• Your BMI indicates obesity. Weight management is highly recommended.")

        # --- Blood pressure advice ---
        if ap_hi >= 140 or ap_lo >= 90:
            st.error("• Your blood pressure is in the hypertension range. Medical advice is recommended.")
        elif ap_hi >= 130 or ap_lo >= 85:
            st.warning("• Your blood pressure is slightly elevated. Monitor regularly and improve lifestyle.")
        else:
            st.success("• Your blood pressure is within the normal range.")

        # --- Pulse pressure ---
        if pulse_pressure > 60:
            st.warning("• Pulse pressure is elevated — may indicate arterial stiffness.")
        elif pulse_pressure < 30:
            st.warning("• Pulse pressure is low — discuss with your physician if persistent.")
        else:
            st.success("• Pulse pressure is in a normal range.")

        # --- Cholesterol advice ---
        if cholesterol == 3:
            st.error("• Cholesterol is significantly above normal — consult a doctor.")
        elif cholesterol == 2:
            st.warning("• Cholesterol is slightly elevated — dietary changes advised.")
        else:
            st.success("• Cholesterol level is normal.")

        # --- Glucose advice ---
        if gluc == 3:
            st.error("• Glucose is high — screening for diabetes is suggested.")
        elif gluc == 2:
            st.warning("• Glucose is slightly high — reduce sugar intake and monitor.")
        else:
            st.success("• Glucose level is normal.")

        # --- Lifestyle ---
        if smoke:
            st.error("• Smoking increases heart risk — quitting strongly recommended.")
        if alco:
            st.warning("• Limit alcohol to reduce heart and liver risk.")
        if not active:
            st.warning("• Increase daily physical activity for cardiovascular benefit.")

        st.info("These suggestions are educational and do not replace clinical medical advice.")
        st.markdown('</div>', unsafe_allow_html=True)

        # ---------------- MODEL ACCURACIES ----------------
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎯 Model Accuracies")
        
        ACCURACIES_PATH = "model_accuracies.json"
        if os.path.exists(ACCURACIES_PATH):
            try:
                accuracies_df = pd.read_json(ACCURACIES_PATH)
                st.table(accuracies_df)
            except Exception as e:
                st.error(f"⚠️ Could not load model_accuracies.json: {e}")
        else:
            st.warning("⚠️ model_accuracies.json not found. Please run Project.ipynb to generate it.")
        
        st.markdown('</div>', unsafe_allow_html=True)


# ---------------- FOOTER ----------------
st.caption("This AI tool is for academic use only — always consult a licensed physician.")
