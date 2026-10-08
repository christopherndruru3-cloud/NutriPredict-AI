import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN & LAYOUT
# ---------------------------------------------------------
st.set_page_config(
    page_title="NutriPredict-AI Pro",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. DICTIONARY MULTI-LANGUAGE (ID / EN)
# ---------------------------------------------------------
LANG = {
    'ID': {
        'title': "👶 NutriPredict-AI Pro",
        'subtitle': "Sistem Deteksi Dini, Simulasi Pertumbuhan & Edukasi Stunting Berbasis AI",
        'tab1': "🩺 Prediksi & Z-Score AI",
        'tab2': "📈 Grafik WHO & Simulator",
        'tab3': "🥣 Resep & Menu MPASI",
        'tab4': "📚 Edukasi & Panduan",
        'form_title': "📝 Form Antropometri Balita",
        'age_label': "Usia Anak (Bulan)",
        'gender_label': "Jenis Kelamin",
        'height_label': "Tinggi Badan (cm)",
        'weight_label': "Berat Badan (kg)",
        'birth_weight_label': "Berat Badan Lahir (kg)",
        'asi_label': "ASI Eksklusif (6 Bulan)",
        'btn_predict': "✨ JALANKAN DIAGNOSIS SEKARANG",
        'male': "Laki-laki",
        'female': "Perempuan",
        'yes': "Ya",
        'no': "Tidak",
        'xai_title': "💡 Transparansi AI (Explainable AI - Feature Importance)",
        'sim_title': "🔮 Simulasi Target Pertumbuhan (What-If Analysis)",
        'sim_months': "Simulasi Usia Muka (Bulan Ke Depan)",
        'print_btn': "🖨️ Cetak / Simpan Kartu Laporan (PDF)"
    },
    'EN': {
        'title': "👶 NutriPredict-AI Pro",
        'subtitle': "Early Detection, Growth Simulation & AI-Powered Stunting Prevention System",
        'tab1': "🩺 AI Diagnosis & Z-Score",
        'tab2': "📈 WHO Curves & Simulator",
        'tab3': "🥣 MPASI Recipes & Menu",
        'tab4': "📚 Education & Guide",
        'form_title': "📝 Child Anthropometry Form",
        'age_label': "Child Age (Months)",
        'gender_label': "Gender",
        'height_label': "Height (cm)",
        'weight_label': "Weight (kg)",
        'birth_weight_label': "Birth Weight (kg)",
        'asi_label': "Exclusive Breastfeeding (6 Months)",
        'btn_predict': "✨ RUN DIAGNOSIS NOW",
        'male': "Male",
        'female': "Female",
        'yes': "Yes",
        'no': "No",
        'xai_title': "💡 AI Transparency (Explainable AI - Feature Importance)",
        'sim_title': "🔮 Growth Target Simulation (What-If Analysis)",
        'sim_months': "Months Ahead to Simulate",
        'print_btn': "🖨️ Print / Save Report Card (PDF)"
    }
}

# Sidebar Language Switcher
st.sidebar.title("🌐 Language / Bahasa")
lang_choice = st.sidebar.radio("Select Language / Pilih Bahasa:", ["Bahasa Indonesia", "English"])
curr_lang = 'ID' if lang_choice == "Bahasa Indonesia" else 'EN'
txt = LANG[curr_lang]

# ---------------------------------------------------------
# 3. STYLING CSS MODERN & PRINT STYLESHEET
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Background Utama */
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.8), rgba(15, 23, 42, 0.8)), 
                    url('https://i.pinimg.com/736x/e9/67/8d/e9678dd9f3233a7528d3e9e3310bbed8.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    /* Header Glassmorphism Biru Muda */
    .main-header {
        text-align: center;
        padding: 25px 20px;
        background: rgba(56, 189, 248, 0.22);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-radius: 20px;
        color: #F0F9FF;
        box-shadow: 0 10px 30px rgba(14, 165, 233, 0.25);
        margin-bottom: 25px;
    }
    .main-header h1 {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 5px;
        color: #38BDF8;
    }
    
    /* Kartu Edukasi & Fitur */
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
        color: #38BDF8;
        margin-bottom: 10px;
    }
    
    /* Box Laporan Hasil */
    .report-box {
        background-color: #FFFFFF;
        color: #1E293B;
        padding: 25px;
        border-radius: 15px;
        border-left: 8px solid #0284C7;
        margin-top: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    /* Tombol Utama */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0EA5E9, #38BDF8);
        color: white;
        font-weight: 800;
        font-size: 16px;
        border-radius: 50px;
        padding: 12px 25px;
        border: none;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.4);
        width: 100%;
        text-transform: uppercase;
    }
    
    /* Styling Cetak / Print PDF */
    @media print {
        body * {
            visibility: hidden;
        }
        .report-box, .report-box * {
            visibility: visible;
        }
        .report-box {
            position: absolute;
            left: 0;
            top: 0;
            width: 100%;
            border: 2px solid #000;
            box-shadow: none;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. MODEL ML & ENGINE WHO Z-SCORE
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    np.random.seed(42)
    n = 1200
    df = pd.DataFrame({
        'Age_Months': np.random.randint(6, 60, n),
        'Gender': np.random.choice([0, 1], n),
        'Height_cm': np.random.uniform(55, 115, n),
        'Weight_kg': np.random.uniform(4, 22, n),
        'Birth_Weight_kg': np.random.uniform(2.0, 4.2, n),
        'Exclusive_Breastfeeding': np.random.choice([0, 1], n)
    })
    # Target sintesis dengan pendekatan medis HAZ
    df['Stunting'] = np.where(df['Height_cm'] / df['Age_Months'] < 1.45, 1, 0)
    
    X = df.drop('Stunting', axis=1)
    y = df['Stunting']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns

model, feature_names = load_model()

# Fungsi perhitungan Z-score sederhana (HAZ WHO)
def calculate_who_zscore(age, height, gender):
    # Estimasi median & SD WHO HAZ
    base_median = 48.0 + (age * 1.25)
    sd = 3.2
    z_score = (height - base_median) / sd
    
    if z_score < -3:
        status = "Sangat Pendek (Severely Stunted)" if curr_lang == 'ID' else "Severely Stunted"
        color = "red"
    elif -3 <= z_score < -2:
        status = "Pendek (Stunted)" if curr_lang == 'ID' else "Stunted"
        color = "orange"
    elif -2 <= z_score <= 2:
        status = "Normal / Ideal" if curr_lang == 'ID' else "Normal"
        color = "green"
    else:
        status = "Tinggi (Tall)" if curr_lang == 'ID' else "Tall"
        color = "blue"
        
    return round(z_score, 2), status, color, round(base_median, 1)

# Header Utama
st.markdown(f"""
<div class="main-header">
    <h1>{txt['title']}</h1>
    <p style="font-size: 1.1rem; opacity: 0.95;">
        {txt['subtitle']}
    </p>
</div>
""", unsafe_allow_html=True)

# Session state initialization
if 'age_val' not in st.session_state:
    st.session_state.age_val = 24
if 'height_val' not in st.session_state:
    st.session_state.height_val = 75.0
if 'weight_val' not in st.session_state:
    st.session_state.weight_val = 11.0

# ---------------------------------------------------------
# 5. TAB NAVIGASI UTAMA
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    txt['tab1'], 
    txt['tab2'], 
    txt['tab3'], 
    txt['tab4']
])

# ================= TAB 1: PREDIKSI & Z-SCORE =================
with tab1:
    st.markdown(f"### {txt['form_title']}")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input(txt['age_label'], min_value=6, max_value=60, value=24, step=1)
        gender_str = st.selectbox(txt['gender_label'], options=[txt['female'], txt['male']])
        height = st.number_input(txt['height_label'], min_value=40.0, max_value=120.0, value=75.0, step=0.5)

    with col2:
        weight = st.number_input(txt['weight_label'], min_value=2.0, max_value=30.0, value=11.0, step=0.1)
        birth_weight = st.number_input(txt['birth_weight_label'], min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        asi_str = st.selectbox(txt['asi_label'], options=[txt['yes'], txt['no']])

    st.session_state.age_val = age
    st.session_state.height_val = height
    st.session_state.weight_val = weight

    gender_val = 1 if gender_str == txt['male'] else 0
    asi_val = 1 if asi_str == txt['yes'] else 0

    st.write("")
    if st.button(txt['btn_predict']):
        # Inference AI
        input_data = np.array([[age, gender_val, height, weight, birth_weight, asi_val]])
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][prediction] * 100

        # WHO Z-Score Calculation
        z_score, who_status, z_color, median_h = calculate_who_zscore(age, height, gender_str)

        st.write("---")
        st.markdown("### 📊 Hasil Evaluasi Diagnosa & Z-Score WHO")
        
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.metric("Z-Score Tinggi/Umur (HAZ WHO)", f"{z_score} SD", delta=who_status, delta_color="normal" if z_color=="green" else "inverse")
            if prediction == 1 or z_score < -2:
                st.error(f"⚠️ **STATUS: {who_status.upper()}**\nTingkat Kepastian AI: **{proba:.1f}%**")
                st.warning(f"🔍 Median standar WHO usia {age} bulan adalah **{median_h} cm** (Selisih **{round(height - median_h, 1)} cm**).")
            else:
                st.success(f"✅ **STATUS: {who_status.upper()}**\nTingkat Kepastian AI: **{proba:.1f}%**")
                st.info(f"🎉 Tinggi anak Anda (**{height} cm**) berada di kisaran normal WHO (Median: {median_h} cm).")

        with res_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba if prediction == 1 else (100 - proba),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Tingkat Indikasi Stunting (%)", 'font': {'size': 15, 'color': "white"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickcolor': "white"},
                    'bar': {'color': "#FF512F" if (prediction == 1 or z_score < -2) else "#00FFAB"},
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(0, 255, 171, 0.2)"},
                        {'range': [40, 70], 'color': "rgba(255, 206, 86, 0.2)"},
                        {'range': [70, 100], 'color': "rgba(255, 81, 47, 0.2)"}
                    ],
                }
            ))
            fig_gauge.update_layout(height=220, paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)

        # FITUR PRINT / LAPORAN PDF
        st.markdown(f"""
        <div class="report-box" id="printable-report">
            <h3 style="color: #0284C7; margin-top:0;">📋 Kartu Laporan Antropometri & Evaluasi Balita</h3>
            <p><strong>Subjek Evaluasi:</strong> Balita Usia {age} Bulan ({gender_str})</p>
            <p><strong>Hasil Z-Score HAZ WHO:</strong> {z_score} SD ({who_status})</p>
            <p><strong>Status AI:</strong> {"Perlu Intervensi Intensif" if prediction == 1 else "Pertumbuhan Normal"}</p>
            <hr>
            <h4>📌 Rencana Tindakan Lanjutan:</h4>
            <ul>
                <li><strong>Protein Hewani:</strong> Berikan minimal 2 porsi/hari (telur, hati ayam, atau ikan kembung).</li>
                <li><strong>Pantau Rutin:</strong> Bawa anak ke Posyandu setiap bulan.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.button(txt['print_btn'], on_click=lambda: st.components.v1.html("<script>window.print();</script>"))

        # FITUR TRANSPARANSI AI (EXPLAINABLE AI)
        st.write("---")
        st.markdown(f"### {txt['xai_title']}")
        importances = model.feature_importances_
        df_imp = pd.DataFrame({
            'Faktor/Fitur': ['Usia (Bulan)', 'Jenis Kelamin', 'Tinggi Badan', 'Berat Badan', 'Berat Lahir', 'ASI Eksklusif'],
            'Tingkat Pengaruh (%)': importances * 100
        }).sort_values(by='Tingkat Pengaruh (%)', ascending=True)

        fig_xai = px.bar(df_imp, x='Tingkat Pengaruh (%)', y='Faktor/Fitur', orientation='h',
                         color='Tingkat Pengaruh (%)', color_continuous_scale='Blues')
        fig_xai.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=280)
        st.plotly_chart(fig_xai, use_container_width=True)

# ================= TAB 2: GRAFIK WHO & SIMULATOR =================
with tab2:
    st.markdown("### 📈 Kurva Pertumbuhan WHO & Simulator")
    
    ages = np.arange(6, 61, 1)
    df_chart = pd.DataFrame({
        'Usia (Bulan)': ages,
        'Sangat Pendek (-3 SD)': 48.0 + (ages * 1.25) - 9.6,
        'Batas Stunted (-2 SD)': 48.0 + (ages * 1.25) - 6.4,
        'Median WHO (0 SD)': 48.0 + (ages * 1.25)
    })

    fig = px.line(df_chart, x='Usia (Bulan)', 
                  y=['Sangat Pendek (-3 SD)', 'Batas Stunted (-2 SD)', 'Median WHO (0 SD)'],
                  color_discrete_sequence=['#EF4444', '#F59E0B', '#10B981'])
    
    curr_a = st.session_state.age_val
    curr_h = st.session_state.height_val

    fig.add_trace(go.Scatter(
        x=[curr_a], y=[curr_h], mode='markers+text',
        name='Posisi Saat Ini', text=[f'Anak Anda ({curr_h} cm)'],
        textposition="top center", marker=dict(size=14, color='#38BDF8', symbol='star')
    ))

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=420)
    st.plotly_chart(fig, use_container_width=True)

    # FITUR SIMULATOR TARGET PERTUMBUHAN ("WHAT-IF")
    st.write("---")
    st.markdown(f"### {txt['sim_title']}")
    sim_months = st.slider(txt['sim_months'], min_value=1, max_value=12, value=6)
    
    future_age = curr_a + sim_months
    target_h_normal = round(48.0 + (future_age * 1.25), 1)
    needed_growth = round(target_h_normal - curr_h, 1)

    c_sim1, c_sim2 = st.columns(2)
    with c_sim1:
        st.info(f"🗓️ **Target Usia:** {future_age} Bulan ({sim_months} bulan lagi)")
        st.success(f"🎯 **Target Tinggi Badan Ideal:** {target_h_normal} cm")
    with c_sim2:
        st.metric("Kebutuhan Pertambahan Tinggi", f"+{needed_growth} cm", delta=f"{round(needed_growth/sim_months, 1)} cm/bulan")

# ================= TAB 3: RESEP & MENU MPASI =================
with tab3:
    st.markdown("### 🥣 Rekomendasi Menu MPASI Protein Hewani Lokal")
    
    age_group = st.radio("Pilih Kelompok Usia Balita:", ["6 - 8 Bulan", "9 - 11 Bulan", "12 - 23 Bulan"], horizontal=True)
    
    if age_group == "6 - 8 Bulan":
        st.markdown("""
        <div class="edu-card">
            <h3>🥣 Tekstur: Bubur Kental (Saring/Lumat)</h3>
            <p><strong>Frekuensi:</strong> 2-3 kali makan utama + 1-2 kali selingan per hari.</p>
            <ul>
                <li><strong>Menu 1:</strong> Puree Hati Ayam + Nasi + Wortel (Sangat kaya zat besi).</li>
                <li><strong>Menu 2:</strong> Bubur Tim Telur Puyuh & Santan.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif age_group == "9 - 11 Bulan":
        st.markdown("""
        <div class="edu-card">
            <h3>🥣 Tekstur: Cincang Halus / Nasi Tim Lumat</h3>
            <p><strong>Frekuensi:</strong> 3-4 kali makan utama + 1-2 kali selingan per hari.</p>
            <ul>
                <li><strong>Menu 1:</strong> Tim Ikan Kembung Suwir + Bayam & Minyak Kelapa (Tinggi Omega-3).</li>
                <li><strong>Menu 2:</strong> Nasi Tim Daging Cincang & Buncis.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="edu-card">
            <h3>🍽️ Tekstur: Makanan Keluarga</h3>
            <p><strong>Frekuensi:</strong> 3-4 kali makan utama + 2 kali selingan keluarga.</p>
            <ul>
                <li><strong>Menu 1:</strong> Sup Bola-Bola Ayam Udang + Buncis & Wortel.</li>
                <li><strong>Menu 2:</strong> Nasi + Pepes Ikan Kembung / Telur Dadar Daun Kelor.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ================= TAB 4: EDUKASI & PANDUAN =================
with tab4:
    st.markdown("### 📚 Panduan Lengkap Cegah Stunting")
    
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("""
        <div class="edu-card">
            <h3>📌 Apa itu 1.000 HPK?</h3>
            <p>1.000 Hari Pertama Kehidupan (sejak dalam kandungan hingga usia 2 tahun) adalah masa keemasan perkembangan otak dan fisik anak yang tidak dapat terulang.</p>
        </div>
        """, unsafe_allow_html=True)
    with e2:
        st.markdown("""
        <div class="edu-card">
            <h3>🛡️ Pilar Utama Pencegahan</h3>
            <ul>
                <li>Asupan cukup protein hewani harian.</li>
                <li>Kebersihan air minum & sanitasi lingkungan.</li>
                <li>Imunisasi dasar lengkap & suplementasi vitamin.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
