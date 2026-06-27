import streamlit as st
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Placement Predictor", page_icon="🎯")

BASE = Path(__file__).parent

@st.cache_resource
def load_models():
    clf       = joblib.load(BASE / "clf_model.pkl")
    reg       = joblib.load(BASE / "reg_model.pkl")
    le_branch = joblib.load(BASE / "le_branch.pkl")
    le_tier   = joblib.load(BASE / "le_tier.pkl")
    return clf, reg, le_branch, le_tier

clf, reg, le_branch, le_tier = load_models()

# ── Title ────────────────────────────────────────────────────────────────────
st.title("🎯 Student Placement Predictor")
st.write("Fill in your details below and click **Predict** to see your placement chances.")
st.divider()

# ── Inputs ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    branch   = st.selectbox("Branch", ["CSE","IT","ECE","EE","ME","CE","Chemical"])
    tier     = st.selectbox("College Tier", ["Tier-1","Tier-2","Tier-3"])
    cgpa     = st.number_input("CGPA", min_value=4.0, max_value=10.0, value=7.5, step=0.1)
    backlogs = st.number_input("Number of Backlogs", min_value=0, max_value=3, value=0)
    coding   = st.slider("Coding Skills (1–10)", 1, 10, 6)
    dsa      = st.slider("DSA Score (1–10)", 1, 10, 5)
    apt      = st.slider("Aptitude Score (%)", 20, 100, 65)
    ml       = st.slider("ML Knowledge (1–10)", 1, 10, 4)

with col2:
    sys_d  = st.slider("System Design (1–10)", 1, 10, 4)
    comm   = st.slider("Communication (1–10)", 1, 10, 6)
    intern = st.number_input("Internships Done", min_value=0, max_value=4, value=1)
    proj   = st.number_input("Projects Count", min_value=0, max_value=8, value=2)
    cert   = st.number_input("Certifications", min_value=0, max_value=6, value=1)
    hack   = st.number_input("Hackathons", min_value=0, max_value=5, value=0)
    oss    = st.number_input("Open Source Contributions", min_value=0, max_value=4, value=0)
    extra  = st.number_input("Extracurriculars", min_value=0, max_value=4, value=1)

st.divider()

# ── Predict button ───────────────────────────────────────────────────────────
if st.button("Predict", type="primary", use_container_width=True):

    X = np.array([[
        le_branch.transform([branch])[0],
        le_tier.transform([tier])[0],
        cgpa, backlogs, coding, dsa, apt, comm, ml, sys_d,
        intern, proj, cert, hack, oss, extra
    ]])

    prob    = clf.predict_proba(X)[0][1]
    sal_est = max(2.5, reg.predict(X)[0])
    pct     = int(round(prob * 100))

    st.subheader("Results")

    # Placement probability
    if pct >= 60:
        st.success(f"✅ Placement Probability: **{pct}%** — High chance of getting placed!")
    elif pct >= 35:
        st.warning(f"⚠️ Placement Probability: **{pct}%** — Moderate chance, room to improve.")
    else:
        st.error(f"❌ Placement Probability: **{pct}%** — Low chance, needs significant improvement.")

    # Salary estimate
    st.info(f"💰 Expected Salary: **₹{sal_est:.1f} LPA** &nbsp;(Range: ₹{max(2.5, sal_est-2.5):.1f}L – ₹{sal_est+3:.1f}L)")

    st.divider()

    # Tip
    st.subheader("💡 Suggestion")
    if pct >= 65:
        st.write("Your profile looks strong! Focus on practising DSA and mock interviews to convert offers.")
    elif pct >= 40:
        st.write("Good start! Try to complete 1–2 more internships and improve your DSA score — these have the biggest impact on placement chances.")
    else:
        st.write("First priority: clear all backlogs — most companies filter on this. Then build 2–3 projects and earn certifications to strengthen your profile.")
