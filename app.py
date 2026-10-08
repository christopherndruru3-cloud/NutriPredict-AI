import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Set Judul Halaman
st.set_page_config(page_title="NutriPredict-AI", page_icon="🥗", layout="centered")

st.title("🥗 NutriPredict-AI")
st.subheader("Sistem Prediksi Risiko Stunting Balita Berbasis AI")

# Train Model Otomatis di Latar Belakang (cached agar cepat)
@st.cache_resource
def load_trained_model():
    np.random.seed(42)
    n_samples = 1000
    data = {
        'Age_Months': np.random.randint(6, 60, n_samples),
        'Gender': np.random.choice([0, 1], n_samples),
        'Height_cm': np.random.uniform(60, 110, n_samples),
        'Weight_kg': np.random.uniform(5, 20, n_samples),
        'Birth_Weight_kg': np.random.uniform(2.0, 4.0, n_samples),
        'Exclusive_Breastfeeding': np.random.choice([0, 1], n_samples),
    }
    df = pd.DataFrame(data)
    df['Stunting_Risk'] = np.where(df['Height_cm'] / df['Age_Months'] < 1.4, 1, 0)
    
    X = df.drop('Stunting_Risk', axis=1)
    y = df['Stunting_Risk']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = load_trained_model()

st.write("---")
st.subheader("Masukkan Data Antropometri Balita:")

# Form Input
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia (Bulan)", min_value=6, max_value=60, value=24)
    gender = st.selectbox("Jenis Kelamin", options=["Perempuan", "Laki-laki"])
    height = st.number_input("Tinggi Badan (cm)", min_value=40.0, max_value=120.0, value=75.0)

with col2:
    weight = st.number_input("Berat Badan (kg)", min_value=2.0, max_value=30.0, value=10.0)
    birth_weight = st.number_input("Berat Badan Lahir (kg)", min_value=1.0, max_value=5.0, value=3.0)
    asi = st.selectbox("ASI Eksklusif", options=["Ya", "Tidak"])

# Konversi Input
gender_val = 1 if gender == "Laki-laki" else 0
asi_val = 1 if asi == "Ya" else 0

if st.button("🔮 Prediksi Risiko Stunting", type="primary"):
    input_data = np.array([[age, gender_val, height, weight, birth_weight, asi_val]])
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][prediction] * 100

    st.write("---")
    if prediction == 1:
        st.error(f"⚠️ **Risiko Stunting Tinggi** (Tingkat Keyakinan: {proba:.1f}%)")
        st.info("💡 **Saran:** Segera konsultasikan tumbuh kembang anak ke fasilitas kesehatan atau posyandu terdekat.")
    else:
        st.success(f"✅ **Risiko Stunting Rendah / Normal** (Tingkat Keyakinan: {proba:.1f}%)")
        st.info("💡 **Saran:** Pertahankan asupan gizi seimbang dan pola asuh yang baik.")
