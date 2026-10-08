import streamlit as st
import pickle
import numpy as np

# Load Model AI
with open('stunting_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Tampilan Antarmuka (UI) Web App
st.set_page_config(page_title="NutriPredict-AI", layout="centered")

st.title("🩺 NutriPredict-AI")
st.subheader("Early Stunting Risk Prediction & Pediatric Biomarker Analytics")
st.write("An interdisciplinary biomedical & data-driven application for early pediatric health screening.")

st.markdown("---")

# Form Input Data
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (Months)", min_value=6, max_value=60, value=24)
    gender = st.selectbox("Gender", ["Female", "Male"])
    height = st.number_input("Height (cm)", min_value=40.0, max_value=130.0, value=82.0)

with col2:
    weight = st.number_input("Weight (kg)", min_value=3.0, max_value=30.0, value=11.5)
    birth_weight = st.number_input("Birth Weight (kg)", min_value=1.5, max_value=5.0, value=3.0)
    asi = st.selectbox("Exclusive Breastfeeding (6 Months)", ["Yes", "No"])

# Konversi Data Input
gender_val = 1 if gender == "Male" else 0
asi_val = 1 if asi == "Yes" else 0

# Tombol Prediksi
if st.button("Analyze Stunting Risk"):
    features = np.array([[age, gender_val, height, weight, birth_weight, asi_val]])
    prediction = model.predict(features)
    
    st.markdown("### **Clinical Risk Assessment Output:**")
    if prediction[0] == 1:
        st.error("⚠️ **HIGH RISK OF STUNTING DETECTED**")
        st.write("Recommendation: Immediate clinical nutrition assessment and pediatric growth consultation required.")
    else:
        st.success("✅ **LOW RISK / NORMAL GROWTH PATTERN**")
        st.write("Recommendation: Continue standard dietary intake and routine Posyandu monitoring.")
