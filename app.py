import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# 1. Konfigurasi Halaman & Judul
st.set_page_config(
    page_title="NutriPredict-AI",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Styling CSS Modern dengan Background Animasi Bayi Colorful
st.markdown("""
<style>
    /* Background Animasi Bayi dengan Overlap Gelap Transparan */
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.85)), 
                    url('https://img.freepik.com/free-vector/cute-baby-pattern-background_23-2148154101.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    .main-header {
        text-align: center;
        padding: 25px 20px;
        background: linear-gradient(135deg, #FF758C 0%, #FF7EB3 100%);
        border-radius: 20px;
        color: white;
        box-shadow: 0 10px 25px rgba(255, 117, 140, 0.4);
        margin-bottom: 25px;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .edu-card {
        background: rgba(30, 41, 59, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        margin-bottom: 15px;
        color: #F8FAFC;
    }
    .edu-card h3 {
        color: #FF758C;
        margin-bottom: 10px;
    }
    .report-box {
        background-color: #FFFFFF;
        color: #1E293B;
        padding: 25px;
        border-radius: 15px;
        border-left: 8px solid #FF512F;
        margin-top: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    div.stButton > button:first-child {
        background: linear-gradient(45deg, #FF512F, #DD2476);
        color: white;
        font-weight: 800;
        font-size: 16px;
        border-radius: 50px;
        padding: 12px 25px;
        border: none;
        box-shadow: 0 4px 15px rgba(221, 36, 118, 0.4);
        width: 100%;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# 3. Model Machine Learning Sintetis (Cached)
@st.cache_resource
def load_model():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'Age_Months': np.random.randint(6, 60, n),
        'Gender': np.random.choice([0, 1], n),
        'Height_cm': np.random.uniform(60, 110, n),
        'Weight_kg': np.random.uniform(5, 20, n),
        'Birth_Weight_kg': np.random.uniform(2.0, 4.0, n),
        'Exclusive_Breastfeeding': np.random.choice([0, 1], n)
    })
    df['Stunting'] = np.where(df['Height_cm'] / df['Age_Months'] < 1.4, 1, 0)
    
    X = df.drop('Stunting', axis=1)
    y = df['Stunting']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = load_model()

# Header Utama
st.markdown("""
<div class="main-header">
    <h1>👶 NutriPredict-AI</h1>
    <p style="font-size: 1.1rem; opacity: 0.95;">
        Sistem Deteksi Dini & Edukasi Cegah Stunting Berbasis Artificial Intelligence
    </p>
</div>
""", unsafe_allow_html=True)

# Session State untuk menyimpan data input pengguna antar-tab
if 'age_val' not in st.session_state:
    st.session_state.age_val = 24
if 'height_val' not in st.session_state:
    st.session_state.height_val = 70.0

# 4. Tab Navigasi
tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Prediksi Risiko AI", 
    "📈 Grafik Kurva Pertumbuhan", 
    "📚 Edukasi Stunting", 
    "💡 Solusi & Panduan Gizi"
])

# ================= TAB 1: PREDIKSI RISIKO AI =================
with tab1:
    st.markdown("### 📝 Form Antropometri Balita")
    st.caption("Masukkan data anak untuk melakukan analisis pertumbuhan:")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Usia Anak (Bulan)", min_value=6, max_value=60, value=24, step=1)
        gender = st.selectbox("Jenis Kelamin", options=["Perempuan", "Laki-laki"])
        height = st.number_input("Tinggi Badan (cm)", min_value=40.0, max_value=120.0, value=70.0, step=0.5)

    with col2:
        weight = st.number_input("Berat Badan (kg)", min_value=2.0, max_value=30.0, value=10.0, step=0.1)
        birth_weight = st.number_input("Berat Badan Lahir (kg)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        asi = st.selectbox("ASI Eksklusif (6 Bulan)", options=["Ya", "Tidak"])

    st.session_state.age_val = age
    st.session_state.height_val = height

    gender_val = 1 if gender == "Laki-laki" else 0
    asi_val = 1 if asi == "Ya" else 0

    st.write("")
    if st.button("✨ JALANKAN DIAGNOSIS SEKARANG"):
        input_data = np.array([[age, gender_val, height, weight, birth_weight, asi_val]])
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][prediction] * 100

        ideal_height = round(age * 1.65, 1)
        defisit = round(ideal_height - height, 1)

        st.write("---")
        st.markdown("### 📊 Hasil Evaluasi Diagnosa")
        
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            if prediction == 1:
                st.error(f"⚠️ **STATUS: TERINDIKASI RISIKO STUNTING**\nTingkat Akurasi Evaluasi: **{proba:.1f}%**")
                if defisit > 0:
                    st.warning(f"🔍 **Kalkulasi Defisit:** Tinggi anak Anda saat ini **{height} cm**. Rata-rata ideal usia {age} bulan adalah **{ideal_height} cm** (Selisih **-{defisit} cm**).")
            else:
                st.success(f"✅ **STATUS: PERTUMBUHAN NORMAL / IDEAL**\nTingkat Akurasi Evaluasi: **{proba:.1f}%**")
                st.info(f"🎉 **Kalkulasi:** Tinggi anak Anda (**{height} cm**) sudah sesuai standar rata-rata ideal usia {age} bulan ({ideal_height} cm).")

        with res_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba if prediction == 1 else (100 - proba),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Tingkat Risiko Stunting (%)", 'font': {'size': 15, 'color': "white"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickcolor': "white"},
                    'bar': {'color': "#FF512F" if prediction == 1 else "#00FFAB"},
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(0, 255, 171, 0.2)"},
                        {'range': [40, 70], 'color': "rgba(255, 206, 86, 0.2)"},
                        {'range': [70, 100], 'color': "rgba(255, 81, 47, 0.2)"}
                    ],
                }
            ))
            fig_gauge.update_layout(height=240, paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div class="report-box">
            <h3 style="color: #DD2476; margin-top:0;">📋 Kartu Hasil Analisis & Rujukan Orang Tua</h3>
            <p><strong>Subjek Evaluasi:</strong> Balita Usia {age} Bulan ({gender})</p>
            <p><strong>Status Ringkas:</strong> {"Perlu Penanganan & Intervensi Gizi Intensif" if prediction == 1 else "Pertumbuhan Sesuai Usia, Pertahankan Nutrisi"}</p>
            <hr>
            <h4>📌 Tindakan Rekomendasi:</h4>
            <ul>
                <li><strong>Gizi Hewani:</strong> Berikan minimal 2 porsi protein hewani/hari (telur, ikan, hati ayam, atau daging).</li>
                <li><strong>Pemantauan Rutin:</strong> Ukur tinggi & berat badan di Posyandu/Puskesmas terdekat setiap bulan.</li>
                <li><strong>Kebersihan:</strong> Jaga kebersihan peralatan makan dan sanitasi air di rumah.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ================= TAB 2: GRAFIK KURVA PERTUMBUHAN =================
with tab2:
    st.markdown("### 📈 Grafik Posisi Anak vs Standard WHO")
    st.caption("Grafik ini membandingkan tinggi badan anak Anda dengan kurva standar pertumbuhan balita:")

    ages = np.arange(6, 61, 1)
    stunting_line = ages * 1.4
    normal_line = ages * 1.65

    df_chart = pd.DataFrame({
        'Usia (Bulan)': ages,
        'Batas Bawah Stunting': stunting_line,
        'Rata-Rata Normal (WHO)': normal_line
    })

    fig = px.line(
        df_chart, 
        x='Usia (Bulan)', 
        y=['Batas Bawah Stunting', 'Rata-Rata Normal (WHO)'],
        color_discrete_sequence=['#FF512F', '#00FFAB'],
        labels={'value': 'Tinggi Badan (cm)'}
    )
    
    current_age = st.session_state.age_val
    current_height = st.session_state.height_val

    fig.add_trace(go.Scatter(
        x=[current_age], y=[current_height],
        mode='markers+text',
        name='Posisi Anak Anda',
        text=[f'Anak Anda ({current_height} cm)'],
        textposition="top center",
        marker=dict(size=16, color='#FFD700', symbol='star')
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=480
    )
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 3: EDUKASI STUNTING =================
with tab3:
    st.markdown("### 📚 Pemahaman Penting Mengenai Stunting")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="edu-card">
            <h3>❓ Apa Itu Stunting?</h3>
            <p>Stunting adalah kondisi gangguan pertumbuhan pada anak akibat kekurangan gizi kronis dan infeksi berulang dalam <strong>1.000 Hari Pertama Kehidupan (0-24 bulan)</strong>.</p>
        </div>
        <div class="edu-card">
            <h3>⚠️ Dampak Panjang Stunting</h3>
            <ul>
                <li>Perkembangan kognitif dan kapasitas otak terhambat.</li>
                <li>Daya tahan tubuh anak lebih lemah dan rentan sakit.</li>
                <li>Risiko tinggi terkena penyakit metabolisme saat dewasa.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="edu-card">
            <h3>🚨 Penyebab Utama</h3>
            <ol>
                <li><strong>Kurang Asupan Protein Hewani:</strong> Terlalu banyak karbohidrat tanpa lauk pauk gizi tinggi.</li>
                <li><strong>Infeksi Berulang:</strong> Diare dan ISPA akibat lingkungan yang kurang bersih.</li>
                <li><strong>Pola Asuh MPASI Kurang Tepat:</strong> Pemberian makanan tidak sesuai usia dan tekstur.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# ================= TAB 4: SOLUSI & PANDUAN GIZI =================
with tab4:
    st.markdown("### 💡 Panduan Aksi & Pencegahan Stunting")
    
    s1, s2, s3 = st.columns(3)
    
    with s1:
        st.markdown("""
        <div class="edu-card">
            <h3>🍼 0 - 6 Bulan</h3>
            <p>✔️ Berikan <strong>ASI Eksklusif</strong> penuh tanpa tambahan air/makanan lain.</p>
            <p>✔️ Timbang berat dan tinggi anak tiap bulan di Posyandu.</p>
            <p>✔️ Pastikan bayi mendapat imunisasi dasar lengkap.</p>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown("""
        <div class="edu-card">
            <h3>🥣 6 - 24 Bulan</h3>
            <p>✔️ Berikan MPASI padat nutrisi kaya <strong>Protein Hewani</strong> (Telur, Ikan, Daging).</p>
            <p>✔️ Lanjutkan pemberian ASI hingga usia 2 tahun.</p>
            <p>✔️ Berikan suplemen vitamin A dan obat cacing sesuai petunjuk medis.</p>
        </div>
        """, unsafe_allow_html=True)

    with s3:
        st.markdown("""
        <div class="edu-card">
            <h3>🧼 Sanitasi & Kesehatan</h3>
            <p>✔️ Cuci tangan pakai sabun sebelum menyiapkan makanan.</p>
            <p>✔️ Gunakan air minum bersih dan higienis.</p>
            <p>✔️ Konsumsi Tablet Tambah Darah (TTD) untuk ibu hamil.</p>
        </div>
        """, unsafe_allow_html=True)
