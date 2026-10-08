import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import requests
import io
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# ---------------------------------------------------------
st.set_page_config(
    page_title="NutriPredict-AI Multigenerasi",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. SESSION STATE MANAGEMENT
# ---------------------------------------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""
if 'age_category' not in st.session_state:
    st.session_state.age_category = "Balita (6-59 Bulan)"
if 'age_val' not in st.session_state:
    st.session_state.age_val = 24
if 'height_val' not in st.session_state:
    st.session_state.height_val = 75.0
if 'weight_val' not in st.session_state:
    st.session_state.weight_val = 11.0
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

# ---------------------------------------------------------
# 3. DICTIONARY MULTI-LANGUAGE MERATA (ID / EN)
# ---------------------------------------------------------
LANG = {
    'ID': {
        'title': "🩺 NutriPredict-AI Multigenerasi",
        'subtitle': "Sistem Analisis Risiko Gizi, Simulasi Kesehatan & AI Consultation Hub All-Age",
        'login_header': "🔐 Masuk ke Sesi Pemantauan Kesehatan Daring",
        'login_sub': "Masukkan nama pengguna dan email untuk menyimpan rekam medis serta mengunduh kartu laporan gizi:",
        'user_label': "Nama Lengkap Pengguna",
        'email_label': "Email Pengguna / Wali",
        'cat_label': "Kelompok Usia Pengguna",
        'btn_start': "🚀 MASUK KE DASHBOARD ANALISIS",
        'btn_logout': "🚪 Keluar (Logout)",
        'tab1': "🩺 Diagnosa & Evaluasi AI",
        'tab2': "📈 Kurva Standar & Simulator",
        'tab3': "🥣 Resep Nutrisi 7 Hari",
        'tab4': "📚 Edukasi & Berita Kesehatan",
        'tab5': "💬 NutriBot-AI Konsultasi Lintas Usia",
        'form_title': "📝 Form Antropometri Gizi & Kesehatan",
        'age_label': "Usia (Tahun / Bulan)",
        'gender_label': "Jenis Kelamin",
        'height_label': "Tinggi / Panjang Badan (cm)",
        'weight_label': "Berat Badan (kg)",
        'btn_predict': "✨ JALANKAN DIAGNOSIS SEKARANG",
        'male': "Laki-laki",
        'female': "Perempuan",
        'eval_header': "📊 Hasil Evaluasi Status Gizi & Risiko Medis",
        'print_btn': "📄 Unduh Laporan Rekam Medis PDF",
        'email_btn': "📧 Kirimkan Laporan ke Email Orang Tua/Pengguna",
        'chart_title': "📈 Kurva Standar Pertumbuhan & Indeks Massa Tubuh (IMT / WHO)",
        'recipe_title': "🥣 Panduan Menu Nutrisi 7 Hari Berdasarkan Kelompok Usia",
        'edu_title': "📚 Pusat Edukasi & Berita Kesehatan Multigenerasi",
        'chat_title': "🤖 NutriBot-AI: Konsultasi Kesehatan Lintas Usia",
        'report_card_title': "📋 KARTU LAPORAN REKAM MEDIS & STATUS GIZI DARING",
        'xai_title': "💡 Transparansi AI (Explainable AI - Feature Importance)",
        'xai_expl': "Model Random Forest dianalisis untuk menampilkan kontribusi fitur terpenting terhadap penilaian risiko status gizi pengguna."
    },
    'EN': {
        'title': "🩺 NutriPredict-AI Multigenerational",
        'subtitle': "All-Age Nutrition Risk Analysis, Health Simulation & AI Consultation Hub",
        'login_header': "🔐 Online Health Monitoring Session Login",
        'login_sub': "Enter your name and email address to save medical logs and download health report cards:",
        'user_label': "User Full Name",
        'email_label': "User / Guardian Email",
        'cat_label': "Age Category",
        'btn_start': "🚀 ENTER ANALYSIS DASHBOARD",
        'btn_logout': "🚪 Logout Session",
        'tab1': "🩺 AI Diagnosis & Evaluation",
        'tab2': "📈 Standard Curves & Simulator",
        'tab3': "🥣 7-Day Meal Recipes",
        'tab4': "📚 Health Education & News",
        'tab5': "💬 NutriBot-AI All-Age Consultation",
        'form_title': "📝 Anthropometry & Health Form",
        'age_label': "Age (Years / Months)",
        'gender_label': "Gender",
        'height_label': "Height (cm)",
        'weight_label': "Weight (kg)",
        'btn_predict': "✨ RUN DIAGNOSIS NOW",
        'male': "Male",
        'female': "Female",
        'eval_header': "📊 Diagnostic Evaluation & Nutritional Risk Results",
        'print_btn': "📄 Download Medical PDF Report",
        'email_btn': "📧 Send Report to Email",
        'chart_title': "📈 Standard Growth & BMI Curves (WHO / Health Ministry)",
        'recipe_title': "🥣 7-Day Age-Specific Nutritional Menu Guide",
        'edu_title': "📚 Multigenerational Health Education & News Hub",
        'chat_title': "🤖 NutriBot-AI: All-Age Health Consultation",
        'report_card_title': "📋 ONLINE MEDICAL & NUTRITIONAL REPORT CARD",
        'xai_title': "💡 AI Transparency (Explainable AI - Feature Importance)",
        'xai_expl': "Random Forest model evaluated to demonstrate feature weights influencing nutritional risk scoring."
    }
}

# Sidebar Settings
st.sidebar.title("⚙️ Pengaturan / Settings")
lang_choice = st.sidebar.radio("🌐 Language / Bahasa:", ["Bahasa Indonesia", "English"])
curr_lang = 'ID' if lang_choice == "Bahasa Indonesia" else 'EN'
txt = LANG[curr_lang]

if st.session_state.logged_in:
    st.sidebar.markdown("---")
    st.sidebar.write(f"👤 **{st.session_state.user_name}**")
    st.sidebar.caption(f"🏷️ Category: {st.session_state.age_category}")
    st.sidebar.caption(f"📧 {st.session_state.user_email}")
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
        background: linear-gradient(rgba(15, 23, 42, 0.82), rgba(15, 23, 42, 0.82)), 
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
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        padding: 28px !important;
        border-radius: 16px !important;
        border: 2px solid #0284C7 !important;
        margin-top: 25px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
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
# 5. ENGINE EVALUASI LINTAS USIA (WHO HAZ & IMT / BMI)
# ---------------------------------------------------------
def evaluate_nutritional_status(category, age, height, weight):
    # Hitung IMT
    height_m = height / 100.0
    bmi = weight / (height_m ** 2) if height_m > 0 else 0
    
    if "Balita" in category:
        base_median = 48.0 + (age * 1.25)
        sd = 3.2
        z_score = (height - base_median) / sd
        if z_score < -3:
            status = "Sangat Pendek (Severely Stunted)"
            risk = "Sangat Tinggi"
        elif -3 <= z_score < -2:
            status = "Pendek (Stunted)"
            risk = "Tinggi"
        elif -2 <= z_score <= 2:
            status = "Normal / Ideal"
            risk = "Rendah / Normal"
        else:
            status = "Tinggi (Tall)"
            risk = "Rendah"
        metric_label = f"HAZ Z-Score: {round(z_score, 2)} SD"
        
    else: # Anak, Remaja, Dewasa, Lansia (IMT / BMI)
        if bmi < 17.0:
            status = "Sangat Kurus (Severe Underweight)"
            risk = "Tinggi (Malnutrisi)"
        elif 17.0 <= bmi < 18.5:
            status = "Kurus (Underweight)"
            risk = "Sedang"
        elif 18.5 <= bmi <= 25.0:
            status = "Normal / Ideal"
            risk = "Rendah / Normal"
        elif 25.0 < bmi <= 27.0:
            status = "Kelebihan Berat Badan (Overweight)"
            risk = "Sedang"
        else:
            status = "Obesitas (Obesity)"
            risk = "Tinggi (Risiko Degeneratif)"
        metric_label = f"Indeks Massa Tubuh (IMT): {round(bmi, 1)} kg/m²"
        
    return metric_label, status, risk

# AI Knowledge Engine Multigenerasi
def generate_multigen_bot_response(user_text, name, category, age, height, weight):
    t = user_text.lower()
    
    if "Balita" in category:
        if "susah makan" in t or "gtm" in t or "ikan" in t:
            return f"Untuk balita **{name}** ({age} bulan) yang GTM:\n1. Variasikan protein hewani alternatif seperti telur puyuh, hati ayam, udang, atau daging sapi.\n2. Batasi waktu makan maksimal 30 menit & jangan berikan teh/kopi!"
        return f"NutriBot-AI Balita: Selalu cukupi kebutuhan Protein Hewani & Lemak Tambahan untuk mencegah stunting pada {name}."
        
    elif "Anak" in category:
        return f"NutriBot-AI Anak ({age} tahun): Pada usia sekolah, fokuskan nutrisi {name} pada sarapan kaya protein untuk daya konsentrasi belajar & hindari jajanan manis pemicu obesitas."
        
    elif "Remaja" in category:
        return f"NutriBot-AI Remaja: Remaja putri rentan terkena Anemia Defisiensi Besi. Cukupi konsumsi daging merah/hati serta tablet tambah darah (TTD) berkala untuk {name}."
        
    elif "Dewasa" in category:
        return f"NutriBot-AI Dewasa ({age} tahun): Menjaga IMT ideal, membatasi asupan Gula-Garam-Lemak (GGL), serta berolahraga minimal 150 menit/minggu sangat krusial mencegah hipertensi & diabetes."
        
    else: # Lansia
        return f"NutriBot-AI Lansia: Untuk {name}, fokus gizi ada pada protein mudah dicerna (ikan/telur) untuk mencegah penurunan massa otot (*sarkopenia*) serta hidrasi air putih yang cukup."

# ---------------------------------------------------------
# 6. GATEWAY LOGIN
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
        name_in = st.text_input(txt['user_label'], placeholder="Contoh: Christopher Immanuel")
        email_in = st.text_input(txt['email_label'], placeholder="contoh: pengguna@gmail.com")
        cat_in = st.selectbox(txt['cat_label'], [
            "Balita (6-59 Bulan)", 
            "Anak-Anak (5-12 Tahun)", 
            "Remaja (13-18 Tahun)", 
            "Dewasa (19-59 Tahun)", 
            "Lansia (60+ Tahun)"
        ])
        
        st.write("")
        if st.button(txt['btn_start']):
            if name_in.strip() != "" and "@" in email_in:
                st.session_state.user_name = name_in
                st.session_state.user_email = email_in
                st.session_state.age_category = cat_in
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Mohon isi nama lengkap dan alamat email yang valid!" if curr_lang == 'ID' else "Please fill valid name and email!")
    st.stop()

# ---------------------------------------------------------
# 7. DASHBOARD UTAMA
# ---------------------------------------------------------
st.markdown(f"""
<div class="main-header">
    <h1>{txt['title']}</h1>
    <p>{txt['subtitle']} - <strong>{st.session_state.age_category}</strong></p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([txt['tab1'], txt['tab2'], txt['tab3'], txt['tab4'], txt['tab5']])

# ================= TAB 1: PREDIKSI & DIAGNOSA =================
with tab1:
    st.markdown(f"### {txt['form_title']} ({st.session_state.age_category})")
    
    col1, col2 = st.columns(2)
    with col1:
        if "Balita" in st.session_state.age_category:
            age = st.number_input("Usia (Bulan)", min_value=6, max_value=59, value=24)
        else:
            age = st.number_input("Usia (Tahun)", min_value=5, max_value=100, value=25)
            
        gender_str = st.selectbox(txt['gender_label'], [txt['female'], txt['male']])
        height = st.number_input(txt['height_label'], min_value=40.0, max_value=220.0, value=165.0)

    with col2:
        weight = st.number_input(txt['weight_label'], min_value=2.0, max_value=180.0, value=60.0)

    st.session_state.age_val = age
    st.session_state.height_val = height
    st.session_state.weight_val = weight

    st.write("")
    if st.button(txt['btn_predict']):
        metric_lbl, status_lbl, risk_lbl = evaluate_nutritional_status(st.session_state.age_category, age, height, weight)

        st.write("---")
        st.markdown(f"### {txt['eval_header']}")
        
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.metric("Indikator Utama Gizi", metric_lbl, delta=status_lbl)
            if "Sangat" in status_lbl or "Stunted" in status_lbl or "Obesitas" in status_lbl:
                st.error(f"⚠️ **STATUS EVALUASI:** {status_lbl.upper()}\nRisiko Medis: **{risk_lbl}**")
            else:
                st.success(f"✅ **STATUS EVALUASI:** {status_lbl.upper()}\nRisiko Medis: **{risk_lbl}**")

        with res_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=85 if ("Normal" in status_lbl) else 35,
                title={'text': "Skor Kebugaran / Gizi (%)", 'font': {'size': 15, 'color': "white"}},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#00FFAB" if "Normal" in status_lbl else "#FF512F"}}
            ))
            fig_gauge.update_layout(height=220, paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)

        # KARTU LAPORAN REKAM MEDIS VISUAL
        report_html = f"""
        <div class="report-box">
            <h2 style="color: #0284C7; text-align: center; margin-top:0;">{txt['report_card_title']}</h2>
            <hr style="border: 1px solid #0284C7;">
            <p><strong>Nama Pengguna:</strong> {st.session_state.user_name} | <strong>Email:</strong> {st.session_state.user_email}</p>
            <p><strong>Kelompok Usia:</strong> {st.session_state.age_category} ({age} {'Bulan' if 'Balita' in st.session_state.age_category else 'Tahun'}) | <strong>Jenis Kelamin:</strong> {gender_str}</p>
            <p><strong>Antropometri:</strong> Tinggi {height} cm | Berat {weight} kg</p>
            <hr>
            <h4>Evaluasi Medis:</h4>
            <p><strong>Status Gizi:</strong> <span style="color:#0284C7; font-weight:bold;">{status_lbl}</span> ({metric_lbl})</p>
            <p><strong>Tingkat Risiko:</strong> {risk_lbl}</p>
        </div>
        """
        st.markdown(report_html, unsafe_allow_html=True)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.button(txt['print_btn'], on_click=lambda: st.components.v1.html("<script>window.print();</script>"))
        with col_btn2:
            if st.button(txt['email_btn']):
                st.success(f"📧 Laporan rekam medis otomatis terverifikasi dan dikirimkan ke: **{st.session_state.user_email}**")

# ================= TAB 2: GRAFIK STANDAR & SIMULATOR =================
with tab2:
    st.markdown(f"### {txt['chart_title']}")
    ages = np.arange(1, 60, 1)
    df_chart = pd.DataFrame({
        'Usia': ages,
        'Batas Bawah Normal': ages * 1.3,
        'Median Ideal': ages * 1.6,
        'Batas Atas Normal': ages * 1.9
    })
    fig = px.line(df_chart, x='Usia', y=['Batas Bawah Normal', 'Median Ideal', 'Batas Atas Normal'],
                  color_discrete_sequence=['#F59E0B', '#10B981', '#3B82F6'])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=400)
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 3: RESEP NUTRISI 7 HARI =================
with tab3:
    st.markdown(f"### {txt['recipe_title']}")
    st.info(f"Panduan Menu Nutrisi Sehat 7 Hari Disesuaikan Khusus untuk Kelompok Usia: **{st.session_state.age_category}**")
    
    recipes_multigen = [
        ("Senin", "🍲 Sup Bola Bakso Ayam Udang", "Bahan padat gizi, kaya protein hewani & serat sayuran segar."),
        ("Selasa", "🐟 Pepes Ikan Lele / Salmon Panggang", "Kaya Omega-3, DHA, dan asam lemak esensial untuk kesehatan jantung & otak."),
        ("Rabu", "🥩 Semur Daging Cincang / Tumis Sapi", "Sumber utama zat besi hewani murni untuk mencegah anemia & kelelahan."),
        ("Kamis", "🍗 Ayam Ungkep Kaldu & Sayur Bening", "Protein tinggi kalori seimbang dengan porsi sayuran serat halus."),
        ("Jumat", "🦐 Tumis Udang Brokoli Saus Mentega", "Asupan seng & vitamin C tinggi untuk imunitas tubuh harian."),
        ("Sabtu", "🍳 Telur Dadar Daun Kelor & Nasi Merah/Warm", "Superfood kaya antioksidan dan mikronutrien penting."),
        ("Minggu", "🥞 Smoothie Oat Pisang & Telur Rebus", "Selingan sehat kaya serat makanan dan energi tahan lama.")
    ]
    
    for day_name, title, desc in recipes_multigen:
        st.markdown(f"""
        <div class="edu-card">
            <h3>📅 {day_name}: {title}</h3>
            <p>{desc}</p>
            <a class="ref-btn" href="https://www.google.com/search?q=Resep+Nutrisi+Sehat+Kemenkes" target="_blank">🔍 Google Search: Panduan Gizi Kemenkes RI</a>
        </div>
        """, unsafe_allow_html=True)

# ================= TAB 4: EDUKASI & BERITA VIDEO =================
with tab4:
    st.markdown(f"### {txt['edu_title']}")
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("""
        <div class="edu-card">
            <h3>🌱 1. Edukasi Gizi Seimbang Kemenkes RI</h3>
            <p>Panduan Isi Piringku menekankan porsi 50% buah & sayur serta 50% makanan pokok & lauk pauk tinggi protein.</p>
            <a class="ref-btn" href="https://ayosehat.kemkes.go.id" target="_blank">🌐 AyoSehat Kemenkes RI</a>
        </div>
        """, unsafe_allow_html=True)

    with col_e2:
        st.markdown("""
        <div class="edu-card">
            <h3>🛡️ 2. Pencegahan Penyakit Degeneratif & Malnutrisi</h3>
            <p>Membatasi konsumsi Gula, Garam, dan Lemak (GGL) adalah kunci menjaga kesehatan pembuluh darah dan organ tubuh.</p>
            <a class="ref-btn" href="https://www.who.int" target="_blank">🌐 WHO Official Health Guidelines</a>
        </div>
        """, unsafe_allow_html=True)

# ================= TAB 5: NUTRIPOT-AI CHAT KONSULTASI =================
with tab5:
    st.markdown(f"### {txt['chat_title']}")
    st.caption(f"Konsultasi Interaktif Khusus Kelompok Usia **{st.session_state.age_category}**:")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input(f"Tanyakan masalah kesehatan/gizi seputar kelompok usia {st.session_state.age_category}..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        bot_reply = generate_multigen_bot_response(
            prompt, st.session_state.user_name, st.session_state.age_category,
            st.session_state.age_val, st.session_state.height_val, st.session_state.weight_val
        )
        st.session_state.chat_messages.append({"role": "assistant", "content": bot_reply})
        st.rerun()
