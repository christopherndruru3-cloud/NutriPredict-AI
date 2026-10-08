import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# 1. Konfigurasi Halaman & Logo Bayi Imut 👶
st.set_page_config(
    page_title="NutriPredict-AI | AI Pemantau Stunting Balita",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Biar Tampilannya Berwarna, Animatif & Super Aesthetic! ✨
st.markdown("""
<style>
    /* Background & Font Global */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    }
    
    /* Custom Card Header */
    .main-header {
        text-align: center;
        padding: 30px 20px;
        background: linear-gradient(135deg, #FF758C 0%, #FF7EB3 100%);
        border-radius: 24px;
        color: white;
        box-shadow: 0 10px 25px rgba(255, 117, 140, 0.3);
        margin-bottom: 30px;
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 8px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.15);
    }
    
    /* Custom Card Box untuk Edukasi */
    .edu-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
        color: #F8FAFC;
    }
    .edu-card h3 {
        color: #FF758C;
        margin-bottom: 12px;
    }
    
    /* Tombol Super Animatif */
    div.stButton > button:first-child {
        background: linear-gradient(45deg, #FF512F, #DD2476);
        color: white;
        font-weight: 800;
        font-size: 18px;
        border-radius: 50px;
        padding: 14px 30px;
        border: none;
        box-shadow: 0 4px 15px rgba(221, 36, 118, 0.4);
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 12px 25px rgba(221, 36, 118, 0.7);
        background: linear-gradient(45deg, #DD2476, #FF512F);
    }
</style>
""", unsafe_allow_html=True)

# 3. Model Machine Learning
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

# Header Utama Banner
st.markdown("""
<div class="main-header">
    <h1>👶 NutriPredict-AI</h1>
    <p style="font-size: 1.2rem; font-weight: 500; opacity: 0.95;">
        Sistem Deteksi Dini & Edukasi Cegah Stunting Berbasis Artificial Intelligence
    </p>
</div>
""", unsafe_allow_html=True)

# 4. TAB NAVIGASI LENGKAP
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prediksi Risiko AI", 
    "📈 Grafik Kurva Pertumbuhan", 
    "📚 Edukasi Stunting", 
    "💡 Solusi & Panduan Gizi"
])

# ================= TAB 1: PREDIKSI RISIKO =================
with tab1:
    st.markdown("### 📝 Form Antropometri Balita")
    st.caption("Masukkan data fisik dan riwayat balita di bawah ini:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Usia Anak (Bulan)", min_value=6, max_value=60, value=24, step=1)
        gender = st.selectbox("Jenis Kelamin", options=["Perempuan", "Laki-laki"])
        height = st.number_input("Tinggi Badan (cm)", min_value=40.0, max_value=120.0, value=70.0, step=0.5)

    with col2:
        weight = st.number_input("Berat Badan (kg)", min_value=2.0, max_value=30.0, value=10.0, step=0.1)
        birth_weight = st.number_input("Berat Badan Lahir (kg)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        asi = st.selectbox("ASI Eksklusif (6 Bulan)", options=["Ya", "Tidak"])

    gender_val = 1 if gender == "Laki-laki" else 0
    asi_val = 1 if asi == "Ya" else 0

    st
