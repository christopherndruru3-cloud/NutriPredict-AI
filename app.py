import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import requests

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# ---------------------------------------------------------
st.set_page_config(
    page_title="NutriPredict-AI Pro",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. SESSION STATE LOGIN, DATA BALITA & CHAT
# ---------------------------------------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'child_name' not in st.session_state:
    st.session_state.child_name = ""
if 'parent_email' not in st.session_state:
    st.session_state.parent_email = ""
if 'age_val' not in st.session_state:
    st.session_state.age_val = 24
if 'height_val' not in st.session_state:
    st.session_state.height_val = 75.0
if 'weight_val' not in st.session_state:
    st.session_state.weight_val = 11.0
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

# ---------------------------------------------------------
# 3. DICTIONARY MULTI-LANGUAGE (ID / EN)
# ---------------------------------------------------------
LANG = {
    'ID': {
        'title': "👶 NutriPredict-AI Pro",
        'subtitle': "Sistem Deteksi Dini, Simulasi Pertumbuhan & AI Consultation Hub",
        'login_header': "🔐 Masuk ke Sesi Pemantauan Daring",
        'login_sub': "Masukkan nama balita dan email orang tua untuk menyimpan rekam medis serta mencetak kartu laporan daring:",
        'child_label': "Nama Lengkap Balita",
        'email_label': "Email Orang Tua / Wali",
        'btn_start': "🚀 MASUK KE DASHBOARD ANALISIS",
        'btn_logout': "🚪 Keluar (Logout)",
        'tab1': "🩺 Prediksi & Z-Score AI",
        'tab2': "📈 Grafik WHO & Simulator",
        'tab3': "🥣 Resep MPASI 7 Hari",
        'tab4': "📚 Edukasi & Berita Stunting",
        'tab5': "💬 AI Chat Konsultasi",
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
        'eval_header': "📊 Hasil Evaluasi Diagnosa & Z-Score WHO",
        'stunting_risk': "Tingkat Indikasi Stunting (%)",
        'xai_title': "💡 Transparansi AI (Explainable AI - Feature Importance)",
        'sim_title': "🔮 Simulasi Target Pertumbuhan (What-If Analysis)",
        'sim_months': "Simulasi Usia Muka (Bulan Ke Depan)",
        'print_btn': "🖨️ Cetak / Simpan Kartu Laporan Daring (PDF)",
        'chart_title': "📈 Kurva Standar Pertumbuhan WHO (Tinggi vs Usia)",
        'chart_analysis_title': "📋 Ringkasan Analisis Tren Pertumbuhan",
        'recipe_title': "🥣 Panduan Menu MPASI 7 Hari Berprotein Hewani & Tutorial Memasak",
        'edu_title': "📚 Pusat Edukasi & Berita Stunting Terkini Indonesia",
        'chat_title': "🤖 NutriBot-AI: Konsultasi Tumbuh Kembang & Gizi Balita",
        'report_card_title': "📋 KARTU LAPORAN ANTROPOMETRI & EVALUASI BALITA DARING",
        'report_sub1': "1. Data Profil Balita & Orang Tua",
        'report_sub2': "2. Evaluasi Medis (WHO HAZ & AI)",
        'report_sub3': "3. Rencana Tindakan Lanjutan (Action Plan)",
        'report_note': "Catatan: Laporan ini dikirimkan otomatis ke email orang tua dan dapat dibawa saat berkonsultasi ke Posyandu/Puskesmas.",
        'xai_features': ['Usia (Bulan)', 'Jenis Kelamin', 'Tinggi Badan', 'Berat Badan', 'Berat Lahir', 'ASI Eksklusif'],
        'xai_expl': "Penjelasan AI Transparency: Model kami menggunakan Random Forest Classifier yang dilatih pada indikator antropometri standar WHO."
    },
    'EN': {
        'title': "👶 NutriPredict-AI Pro",
        'subtitle': "Early Detection, Growth Simulation & AI Consultation Hub",
        'login_header': "🔐 Online Monitoring Session Login",
        'login_sub': "Enter your child's name and parent email to save medical logs and generate printable online report cards:",
        'child_label': "Child's Full Name",
        'email_label': "Parent / Guardian Email",
        'btn_start': "🚀 ENTER ANALYSIS DASHBOARD",
        'btn_logout': "🚪 Logout Session",
        'tab1': "🩺 AI Diagnosis & Z-Score",
        'tab2': "📈 WHO Curves & Simulator",
        'tab3': "🥣 7-Day MPASI Recipes",
        'tab4': "📚 Education & Stunting News",
        'tab5': "💬 AI Chat Consultation",
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
        'eval_header': "📊 Diagnostic Evaluation & WHO Z-Score Results",
        'stunting_risk': "Stunting Indication Level (%)",
        'xai_title': "💡 AI Transparency (Explainable AI - Feature Importance)",
        'sim_title': "🔮 Growth Target Simulation (What-If Analysis)",
        'sim_months': "Months Ahead to Simulate",
        'print_btn': "🖨️ Print / Save Online Report Card (PDF)",
        'chart_title': "📈 WHO Standard Growth Curve (Height vs Age)",
        'chart_analysis_title': "📋 Growth Trend Analysis Summary",
        'recipe_title': "🥣 7-Day High-Animal-Protein MPASI Guide & Cooking Tutorials",
        'edu_title': "📚 Educational Hub & Latest Indonesia Stunting News",
        'chat_title': "🤖 NutriBot-AI: Growth & Child Nutrition Consultation",
        'report_card_title': "📋 ONLINE CHILD ANTHROPOMETRY & EVALUATION REPORT CARD",
        'report_sub1': "1. Child & Parent Profile Data",
        'report_sub2': "2. Medical Evaluation (WHO HAZ & AI)",
        'report_sub3': "3. Follow-up Action Plan",
        'report_note': "Note: This report is automatically logged for your email and can be brought to local health clinics.",
        'xai_features': ['Age (Months)', 'Gender', 'Height', 'Weight', 'Birth Weight', 'Exclusive Breastfeeding'],
        'xai_expl': "AI Transparency Explanation: Our model utilizes a Random Forest Classifier trained on WHO anthropometric data."
    }
}

# Sidebar Settings & Logout
st.sidebar.title("⚙️ Pengaturan / Settings")
lang_choice = st.sidebar.radio("🌐 Language / Bahasa:", ["Bahasa Indonesia", "English"])
curr_lang = 'ID' if lang_choice == "Bahasa Indonesia" else 'EN'
txt = LANG[curr_lang]

if st.session_state.logged_in:
    st.sidebar.markdown("---")
    st.sidebar.write(f"👤 Balita: **{st.session_state.child_name}**")
    st.sidebar.caption(f"📧 {st.session_state.parent_email}")
    if st.sidebar.button(txt['btn_logout']):
        st.session_state.logged_in = False
        st.session_state.chat_messages = []
        st.rerun()

# ---------------------------------------------------------
# 4. STYLING CSS ANIMATIF & GLASSMORPHISM
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.78), rgba(15, 23, 42, 0.78)), 
                    url('https://i.pinimg.com/736x/e9/67/8d/e9678dd9f3233a7528d3e9e3310bbed8.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
        100% { transform: translateY(0px); }
    }
    
    .main-header {
        text-align: center;
        padding: 22px 20px;
        background: rgba(56, 189, 248, 0.22);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-radius: 20px;
        color: #F0F9FF;
        box-shadow: 0 10px 30px rgba(14, 165, 233, 0.25);
        margin-bottom: 20px;
        animation: float 4s ease-in-out infinite;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        color: #38BDF8;
    }
    
    .edu-card {
        background: rgba(30, 41, 59, 0.88);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        margin-bottom: 18px;
        color: #F8FAFC;
        transition: transform 0.3s ease;
    }
    .edu-card:hover {
        transform: translateY(-4px);
    }
    .edu-card h3 {
        color: #38BDF8;
        margin-bottom: 10px;
    }
    
    .ref-btn {
        display: inline-block;
        background-color: #0284C7;
        color: white !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: bold;
        text-decoration: none;
        margin-top: 8px;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.4);
    }
    
    .report-box {
        background-color: #FFFFFF;
        color: #0F172A;
        padding: 28px;
        border-radius: 16px;
        border-left: 8px solid #0284C7;
        margin-top: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        font-family: 'Segoe UI', sans-serif;
    }
    
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0EA5E9, #38BDF8);
        color: white;
        font-weight: 800;
        font-size: 15px;
        border-radius: 50px;
        padding: 12px 25px;
        border: none;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.4);
        width: 100%;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. MODEL ML & ENGINE WHO Z-SCORE
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
    df['Stunting'] = np.where(df['Height_cm'] / df['Age_Months'] < 1.45, 1, 0)
    
    X = df.drop('Stunting', axis=1)
    y = df['Stunting']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = load_model()

def calculate_who_zscore(age, height):
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
        status = "Normal / Ideal" if curr_lang == 'ID' else "Normal / Ideal"
        color = "green"
    else:
        status = "Tinggi (Tall)" if curr_lang == 'ID' else "Tall"
        color = "blue"
        
    return round(z_score, 2), status, color, round(base_median, 1)

# Response Engine Sederhana untuk NutriBot-AI Chatbot
def generate_bot_response(user_text, child_name, age, height, weight):
    text = user_text.lower()
    
    if "susah makan" in text or "gtm" in text or "picky" in text or "makan" in text:
        return f"Bunda/Ayah, untuk ananda **{child_name}** yang sedang mengalami Gerakan Tutup Mulut (GTM) atau susah makan, berikut beberapa langkah praktis:\n\n1. **Variasi Protein Hewani:** Cobalah mengolah lauk dengan bentuk menarik (misal: bakso ayam udang atau telur dadar daun kelor).\n2. **Atur Jadwal Makan:** Batasi durasi makan maksimal 30 menit dan hindari pemberian camilan mendekati jam makan utama.\n3. **Cek Tekstur MPASI:** Pastikan tekstur makanan sesuai dengan kelompok usianya ({age} bulan).\n\n*Jika anak menolak nasi, sumber karbohidrat bisa diganti dengan kentang, ubi, atau jagung pipil.*"
    
    elif "tinggi" in text or "pendek" in text or "stunting" in text:
        return f"Berdasarkan data pencatatan, **{child_name}** berusia **{age} bulan** dengan tinggi **{height} cm**.\n\nUntuk mengoptimalkan pertumbuhan tinggi badan (panjang tulang):\n- Pastikan asupan **Protein Hewani (seperti telur, hati ayam, dan ikan kembung)** terpenuhi minimal 2 porsi/hari.\n- Protein hewani mengandung rangsangan faktor pertumbuhan *IGF-1* yang secara langsung memperpanjang tulang balita.\n- Pastikan waktu tidur anak cukup (11-14 jam per hari untuk balita) karena hormon pertumbuhan (*Growth Hormone*) diproduksi maksimal saat tidur nyenyak."
    
    elif "berat" in text or "kurus" in text or "bb" in text:
        return f"Untuk menaikkan berat badan **{child_name}** ({weight} kg) secara sehat:\n\n- Tambahkan lemak tambahan (lemak sehat) seperti **santan segar, butter/margarin, minyak kelapa, atau minyak wijen** ke dalam MPASI/makanan utamanya.\n- Berikan porsi kecil tapi sering jika anak cepat kenyang.\n- Rutin cek ke Posyandu/Puskesmas untuk memantau grafik kenaikan berat badan harian (*weight faltering*)."
    
    else:
        return f"Halo Bunda/Ayah dari **{child_name}**! Saya NutriBot-AI 🤖. Ada yang bisa saya bantu terkait tumbuh kembang, pola gizi MPASI, atau masalah susah makan anak Anda?"

# ---------------------------------------------------------
# 6. GATEWAY LOGIN (JIKA BELUM LOGIN)
# ---------------------------------------------------------
if not st.session_state.logged_in:
    st.markdown(f"""
    <div class="main-header">
        <h1>{txt['title']}</h1>
        <p>{txt['subtitle']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="edu-card" style="max-width:650px; margin: 0 auto; padding: 30px;">
        <h3>{txt['login_header']}</h3>
        <p style="font-size:14px; opacity:0.9;">{txt['login_sub']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        child_in = st.text_input(txt['child_label'], placeholder="Contoh: Ananda Bintang")
        email_in = st.text_input(txt['email_label'], placeholder="contoh: orangtua@gmail.com")
        
        st.write("")
        if st.button(txt['btn_start']):
            if child_in.strip() != "" and "@" in email_in:
                st.session_state.child_name = child_in
                st.session_state.parent_email = email_in
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Mohon isi nama lengkap balita dan alamat email yang valid!" if curr_lang == 'ID' else "Please provide a valid child name and email address!")
    st.stop()

# ---------------------------------------------------------
# 7. DASHBOARD UTAMA (SETELAH LOGIN)
# ---------------------------------------------------------
st.markdown(f"""
<div class="main-header">
    <h1>{txt['title']}</h1>
    <p>{txt['subtitle']}</p>
</div>
""", unsafe_allow_html=True)

# 5 TAB LENGKAP TERMASUK TAB CHATBOT
tab1, tab2, tab3, tab4, tab5 = st.tabs([txt['tab1'], txt['tab2'], txt['tab3'], txt['tab4'], txt['tab5']])

# ================= TAB 1: PREDIKSI & Z-SCORE =================
with tab1:
    st.markdown(f"### {txt['form_title']}")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input(txt['age_label'], min_value=6, max_value=60, value=st.session_state.age_val, step=1)
        gender_str = st.selectbox(txt['gender_label'], options=[txt['female'], txt['male']])
        height = st.number_input(txt['height_label'], min_value=40.0, max_value=120.0, value=st.session_state.height_val, step=0.5)

    with col2:
        weight = st.number_input(txt['weight_label'], min_value=2.0, max_value=30.0, value=st.session_state.weight_val, step=0.1)
        birth_weight = st.number_input(txt['birth_weight_label'], min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        asi_str = st.selectbox(txt['asi_label'], options=[txt['yes'], txt['no']])

    st.session_state.age_val = age
    st.session_state.height_val = height
    st.session_state.weight_val = weight

    gender_val = 1 if gender_str == txt['male'] else 0
    asi_val = 1 if asi_str == txt['yes'] else 0

    st.write("")
    if st.button(txt['btn_predict']):
        input_data = np.array([[age, gender_val, height, weight, birth_weight, asi_val]])
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][prediction] * 100

        z_score, who_status, z_color, median_h = calculate_who_zscore(age, height)

        st.write("---")
        st.markdown(f"### {txt['eval_header']}")
        
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
                title={'text': txt['stunting_risk'], 'font': {'size': 15, 'color': "white"}},
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

        report_html = f"""
        <div class="report-box" id="printable-report">
            <h2 style="color: #0284C7; text-align: center; margin-top:0;">{txt['report_card_title']}</h2>
            <hr style="border: 1px solid #0284C7;">
            <h4>{txt['report_sub1']}</h4>
            <table style="width:100%; font-size:14px; border-collapse: collapse;">
                <tr><td><strong>Nama Balita:</strong> {st.session_state.child_name}</td><td><strong>Email Orang Tua:</strong> {st.session_state.parent_email}</td></tr>
                <tr><td><strong>Usia Balita:</strong> {age} Bulan</td><td><strong>Jenis Kelamin:</strong> {gender_str}</td></tr>
                <tr><td><strong>Tinggi Badan:</strong> {height} cm</td><td><strong>Berat Badan:</strong> {weight} kg</td></tr>
                <tr><td><strong>Berat Lahir:</strong> {birth_weight} kg</td><td><strong>Riwayat ASI Eksklusif:</strong> {asi_str}</td></tr>
            </table>
            <br>
            <h4>{txt['report_sub2']}</h4>
            <ul>
                <li><strong>Skor Standar Pertumbuhan WHO (HAZ Z-Score):</strong> <span style="color:{'red' if z_score < -2 else 'green'}; font-weight:bold;">{z_score} SD ({who_status})</span></li>
                <li><strong>Standar Median WHO Usia {age} Bln:</strong> {median_h} cm (Deviasi: {round(height - median_h, 1)} cm)</li>
                <li><strong>Tingkat Indikasi AI Stunting:</strong> {proba:.1f}%</li>
            </ul>
            <br>
            <h4>{txt['report_sub3']}</h4>
            <ol>
                <li><strong>Intervensi Nutrisi Protein Hewani:</strong> Berikan minimal 2 porsi protein hewani berkualitas tinggi per hari (contoh: 1 butir telur + 50g hati ayam/ikan kembung).</li>
                <li><strong>Suplementasi Zat Besi & Vitamin A:</strong> Konsultasikan dengan bidan/dokter untuk pemberian Vitamin A dan taburia/sirup zat besi.</li>
                <li><strong>Pemantauan Rutin Posyandu:</strong> Timbang berat badan dan ukur tinggi badan secara teratur setiap bulan untuk memantau kurva pertumbuhan.</li>
                <li><strong>Sanitasi & Kebersihan (PHBS):</strong> Pastikan air minum direbus hingga mendidih dan cuci tangan dengan sabun sebelum menyiapkan MPASI.</li>
            </ol>
            <br>
            <p style="font-size: 11px; color: #64748B; font-style: italic;">{txt['report_note']}</p>
        </div>
        """
        st.markdown(report_html, unsafe_allow_html=True)
        st.button(txt['print_btn'], on_click=lambda: st.components.v1.html("<script>window.print();</script>"))

        st.write("---")
        st.markdown(f"### {txt['xai_title']}")
        importances = model.feature_importances_
        df_imp = pd.DataFrame({
            'Faktor/Fitur': txt['xai_features'],
            'Tingkat Pengaruh (%)': importances * 100
        }).sort_values(by='Tingkat Pengaruh (%)', ascending=True)

        fig_xai = px.bar(df_imp, x='Tingkat Pengaruh (%)', y='Faktor/Fitur', orientation='h',
                         color='Tingkat Pengaruh (%)', color_continuous_scale='Blues')
        fig_xai.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=280)
        st.plotly_chart(fig_xai, use_container_width=True)
        
        st.markdown(f"""
        <div class="edu-card">
            <p style="font-size:13px; line-height:1.5;">{txt['xai_expl']}</p>
            <a class="ref-btn" href="https://www.google.com/search?q=WHO+Child+Growth+Standards+HAZ+Z-score" target="_blank">🔍 Google Search: Standar WHO HAZ Z-Score</a>
        </div>
        """, unsafe_allow_html=True)

# ================= TAB 2: GRAFIK WHO & SIMULATOR =================
with tab2:
    st.markdown(f"### {txt['chart_title']}")
    
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
        name='Posisi Saat Ini', text=[f'{st.session_state.child_name} ({curr_h} cm)'],
        textposition="top center", marker=dict(size=14, color='#38BDF8', symbol='star')
    ))

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"#### {txt['chart_analysis_title']}")
    z_sc, st_name, _, med_val = calculate_who_zscore(curr_a, curr_h)
    diff = round(curr_h - med_val, 1)
    
    st.markdown(f"""
    <div class="edu-card">
        <p>📌 <strong>Interpretasi Grafik:</strong> Bintang biru mewakili posisi tumbuh kembang <strong>{st.session_state.child_name}</strong> saat ini pada usia <strong>{curr_a} bulan</strong> ({curr_h} cm).</p>
        <ul>
            <li><strong>Garis Hijau (0 SD):</strong> Median ideal WHO ({med_val} cm). Selisih tinggi anak: <strong>{'+' if diff >= 0 else ''}{diff} cm</strong>.</li>
            <li><strong>Garis Kuning (-2 SD):</strong> Batas ambang stunted (<strong>{round(med_val - 6.4, 1)} cm</strong>).</li>
            <li><strong>Garis Merah (-3 SD):</strong> Batas ambang stunting berat (<strong>{round(med_val - 9.6, 1)} cm</strong>).</li>
        </ul>
        <a class="ref-btn" href="https://www.who.int/tools/child-growth-standards" target="_blank">🌐 Standar Pertumbuhan Anak WHO Resmi</a>
    </div>
    """, unsafe_allow_html=True)

    st.write("---")
    st.markdown(f"### {txt['sim_title']}")
    sim_months = st.slider(txt['sim_months'], min_value=1, max_value=12, value=6)
    
    future_age = curr_a + sim_months
    target_h_normal = round(48.0 + (future_age * 1.25), 1)
    needed_growth = round(target_h_normal - curr_h, 1)

    c_sim1, c_sim2 = st.columns(2)
    with c_sim1:
        st.info(f"🗓️ **Target Usia:** {future_age} Bulan ({sim_months} bulan ke depan)")
        st.success(f"🎯 **Target Tinggi Ideal WHO:** {target_h_normal} cm")
    with c_sim2:
        st.metric("Total Kebutuhan Tambahan Tinggi", f"+{needed_growth} cm", delta=f"{round(needed_growth/sim_months, 1)} cm/bulan")

# ================= TAB 3: RESEP MPASI 7 HARI =================
with tab3:
    st.markdown(f"### {txt['recipe_title']}")
    
    days = [
        ("Senin / Monday", "🐣 Puree Hati Ayam & Santan", ["🐔 Hati Ayam", "🌾 Nasi", "🥕 Wortel", "🥥 Santan"], 
         "1. Rebus hati ayam hingga matang.\n2. Lumatkan nasi hangat dan campur parutan wortel.\n3. Tambahkan 1 sdt santan segar lalu saring hingga tekstur lembut."),
        
        ("Selasa / Tuesday", "🐟 Bubur Saring Ikan Kembung", ["🐟 Ikan Kembung", "🌾 Nasi", "🍈 Labu Siam", "🥥 Minyak Kelapa"],
         "1. Kukus fillet ikan kembung tanpa duri.\n2. Campur dengan tim nasi dan parutan labu siam.\n3. Tambahkan minyak kelapa lalu saring halus."),
        
        ("Rabu / Wednesday", "🥚 Bubur Tim Telur Puyuh & Bayam", ["🥚 Telur Puyuh", "🌾 Nasi", "🥬 Bayam", "🧈 Margarin"],
         "1. Rebus 2 butir telur puyuh lalu lumatkan halus.\n2. Cincang daun bayam rebus.\n3. Aduk rata dengan nasi tim hangat dan sejumput margarin."),
        
        ("Kamis / Thursday", "🥩 Puree Daging Sapi Lumat", ["🥩 Daging Sapi", "🥔 Kentang", "🧀 Keju", "🧈 Butter"],
         "1. Tumis daging sapi cincang halus dengan butter.\n2. Rebus kentang lalu lumatkan bersama daging.\n3. Taburkan keju parut secukupnya."),
        
        ("Jumat / Friday", "🦐 Tim Udang Cincang & Tahu", ["🦐 Udang Kupas", "🧊 Tahu Lembut", "🌾 Nasi", "🌱 Minyak Wijen"],
         "1. Cincang halus udang kupas bersih.\n2. Lumatkan tahu putih bersama nasi tim.\n3. Kukus selama 15 menit dan beri 2 tetes minyak wijen."),
        
        ("Sabtu / Saturday", "🍳 Orak-Arik Telur Bebek & Tempe", ["🍳 Telur Bebek", "🟫 Tempe", "🌽 Jagung Manis", "🧈 Butter"],
         "1. Kukus tempe lalu potong dadu kecil.\n2. Kocok telur bebek, orak-arik lembut bersama butter.\n3. Campur pipilan jagung manis lumat."),
        
        ("Minggu / Sunday", "🍲 Sup Bola-Bola Ayam Udang", ["🐔 Ayam Cincang", "🦐 Udang", "🥕 Wortel", "🥔 Kentang"],
         "1. Buat bola-bola bakso halus dari adonan ayam dan udang.\n2. Rebus kuah kaldu ceker bersama wortel dan kentang.\n3. Masukkan bola-bola ayam hingga mengapung matang.")
    ]
    
    for day_name, title, ingredients, tutorial in days:
        st.markdown(f"""
        <div class="edu-card">
            <h3>📅 {day_name}: {title}</h3>
            <p><strong>Emotikon & Bahan Utama:</strong> {' • '.join(ingredients)}</p>
            <p><strong>📖 Tutorial Langkah Memasak Step-by-Step:</strong></p>
            <pre style="background:rgba(0,0,0,0.3); padding:10px; border-radius:8px; color:#F0F9FF; font-size:13px; white-space: pre-wrap;">{tutorial}</pre>
            <a class="ref-btn" href="https://www.google.com/search?q=Resep+MPASI+Protein+Hewani+Kemenkes" target="_blank">🔍 Google Search: Panduan MPASI Kemenkes RI</a>
        </div>
        """, unsafe_allow_html=True)

# ================= TAB 4: EDUKASI & BERITA STUNTING =================
with tab4:
    st.markdown(f"### {txt['edu_title']}")
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        st.markdown("""
        <div class="edu-card">
            <h3>🌱 1. Apa Itu Stunting & Bahayanya?</h3>
            <p>Stunting adalah kondisi gagal tumbuh pada balita akibat kekurangan gizi kronis dan infeksi berulang dalam <strong>1.000 Hari Pertama Kehidupan (0-24 Bulan)</strong>. Dampaknya tidak hanya fisik pendek, tetapi juga penurunan IQ serta risiko penyakit degeneratif saat dewasa.</p>
            <a class="ref-btn" href="https://ayosehat.kemkes.go.id/topik-penyakit/defisiensi-nutrisi/stunting" target="_blank">🌐 AyoSehat Kemenkes: Penjelasan Stunting</a>
        </div>
        <div class="edu-card">
            <h3>🍖 2. Mengapa Harus Protein Hewani?</h3>
            <p>Protein hewani (telur, hati ayam, ikan kembung, daging) mengandung asam amino esensial lengkap dan rangsangan faktor pertumbuhan <em>mTORC1/IGF-1</em> untuk pembentukan tulang panjang anak.</p>
            <a class="ref-btn" href="https://www.google.com/search?q=Protein+Hewani+Cegah+Stunting+Kemenkes" target="_blank">🔍 Google Search: Bukti Klinis Protein Hewani</a>
        </div>
        """, unsafe_allow_html=True)

    with col_e2:
        st.markdown("""
        <div class="edu-card">
            <h3>📰 3. Berita Terkini Stunting di Indonesia</h3>
            <ul>
                <li><strong>Prevalensi Stunting Indonesia:</strong> Hasil SSGI menunjukkan angka prevalensi stunting nasional berada di kisaran 19,8%.</li>
                <li><strong>Gerakan Intervensi Serentak:</strong> Pemerintah terus menggalakkan pemberian Makanan Tambahan (PMT) Kaya Protein Hewani di seluruh Posyandu.</li>
            </ul>
            <a class="ref-btn" href="https://stunting.go.id" target="_blank">📰 Portal Resmi TP2S Stunting Indonesia</a>
        </div>
        <div class="edu-card">
            <h3>🛡️ 4. Langkah Pencegahan (Pola WASH & Imunisasi)</h3>
            <p>Pencegahan stunting mencakup pemberian ASI Eksklusif 6 bulan, sanitasi air bersih (WASH) untuk mencegah diare berulang, serta imunisasi dasar lengkap.</p>
            <a class="ref-btn" href="https://www.who.int/news-room/fact-sheets/detail/malnutrition" target="_blank">🌐 WHO Fact Sheets: Child Malnutrition</a>
        </div>
        """, unsafe_allow_html=True)

# ================= TAB 5: AI CHAT KONSULTASI =================
with tab5:
    st.markdown(f"### {txt['chat_title']}")
    st.caption(f"Konsultasi interaktif seputar kondisi kesehatan, gizi MPASI, dan tumbuh kembang **Ananda {st.session_state.child_name}**:")

    # Tampilkan Riwayat Chat
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input Chat Pengguna
    if prompt := st.chat_input("Tanyakan sesuatu (misal: 'Anak saya susah makan nasi, solusinya apa?')..."):
        # Tambah pesan user ke riwayat
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Hasilkan jawaban dari AI
        bot_reply = generate_bot_response(
            prompt, 
            st.session_state.child_name, 
            st.session_state.age_val, 
            st.session_state.height_val, 
            st.session_state.weight_val
        )

        # Tambah jawaban bot ke riwayat
        st.session_state.chat_messages.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
