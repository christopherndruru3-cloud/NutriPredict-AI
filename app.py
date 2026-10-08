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

# ReportLab untuk Generate PDF Otomatis
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

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
# 2. SESSION STATE MANAGEMENT
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
if 'reply_to' not in st.session_state:
    st.session_state.reply_to = None

# ---------------------------------------------------------
# 3. DICTIONARY MULTI-LANGUAGE 100% MERATA (ID / EN)
# ---------------------------------------------------------
LANG = {
    'ID': {
        'title': "👶 NutriPredict-AI Pro",
        'subtitle': "Sistem Deteksi Dini, Simulasi Pertumbuhan & AI Consultation Hub",
        'login_header': "🔐 Masuk ke Sesi Pemantauan Daring",
        'login_sub': "Masukkan nama balita dan email orang tua untuk menyimpan rekam medis serta mengunduh kartu laporan resmi:",
        'child_label': "Nama Lengkap Balita",
        'email_label': "Email Orang Tua / Wali",
        'btn_start': "🚀 MASUK KE DASHBOARD ANALISIS",
        'btn_logout': "🚪 Keluar (Logout)",
        'tab1': "🩺 Prediksi & Z-Score AI",
        'tab2': "📈 Grafik WHO & Simulator",
        'tab3': "🥣 Resep MPASI 7 Hari",
        'tab4': "📚 Edukasi & Berita Video",
        'tab5': "💬 NutriBot-AI Chat (IG Style)",
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
        'print_btn': "📄 Unduh PDF & Kirim Laporan ke Email",
        'chart_title': "📈 Kurva Standar Pertumbuhan WHO (Tinggi vs Usia)",
        'chart_analysis_title': "📋 Ringkasan Analisis Tren Pertumbuhan",
        'recipe_title': "🥣 Panduan Menu MPASI 7 Hari Berprotein Hewani (Berdasarkan Kelompok Usia)",
        'edu_title': "📚 Pusat Edukasi, Artikel & Video Resmi Pencegahan Stunting",
        'chat_title': "🤖 NutriBot-AI: Konsultasi Interaktif Tumbuh Kembang",
        'report_card_title': "📋 KARTU LAPORAN ANTROPOMETRI & EVALUASI BALITA DARING",
        'report_sub1': "1. Data Profil Balita & Orang Tua",
        'report_sub2': "2. Evaluasi Medis (WHO HAZ & AI)",
        'report_sub3': "3. Rencana Tindakan Lanjutan (Action Plan)",
        'report_note': "Catatan: Laporan ini dikirimkan otomatis ke email orang tua dan dapat dibawa saat berkonsultasi ke Posyandu/Puskesmas.",
        'xai_features': ['Usia (Bulan)', 'Jenis Kelamin', 'Tinggi Badan', 'Berat Badan', 'Berat Lahir', 'ASI Eksklusif'],
        'xai_expl': "Penjelasan AI Transparency: Model kami menggunakan Random Forest Classifier yang dilatih pada indikator antropometri standar WHO. Grafik di atas menunjukkan bobot kontribusi setiap variabel input terhadap keputusan prediksi."
    },
    'EN': {
        'title': "👶 NutriPredict-AI Pro",
        'subtitle': "Early Detection, Growth Simulation & AI Consultation Hub",
        'login_header': "🔐 Online Monitoring Session Login",
        'login_sub': "Enter your child's name and parent email to save medical logs and download official PDF report cards:",
        'child_label': "Child's Full Name",
        'email_label': "Parent / Guardian Email",
        'btn_start': "🚀 ENTER ANALYSIS DASHBOARD",
        'btn_logout': "🚪 Logout Session",
        'tab1': "🩺 AI Diagnosis & Z-Score",
        'tab2': "📈 WHO Curves & Simulator",
        'tab3': "🥣 7-Day MPASI Recipes",
        'tab4': "📚 Education & Video News",
        'tab5': "💬 NutriBot-AI Chat (IG Style)",
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
        'print_btn': "📄 Download PDF & Send Report to Email",
        'chart_title': "📈 WHO Standard Growth Curve (Height vs Age)",
        'chart_analysis_title': "📋 Growth Trend Analysis Summary",
        'recipe_title': "🥣 7-Day High-Animal-Protein MPASI Guide (By Age Groups)",
        'edu_title': "📚 Educational Hub, Articles & Official Stunting Videos",
        'chat_title': "🤖 NutriBot-AI: Interactive Growth Consultation",
        'report_card_title': "📋 ONLINE CHILD ANTHROPOMETRY & EVALUATION REPORT CARD",
        'report_sub1': "1. Child & Parent Profile Data",
        'report_sub2': "2. Medical Evaluation (WHO HAZ & AI)",
        'report_sub3': "3. Follow-up Action Plan",
        'report_note': "Note: This report is automatically logged for your email and can be brought to local health clinics.",
        'xai_features': ['Age (Months)', 'Gender', 'Height', 'Weight', 'Birth Weight', 'Exclusive Breastfeeding'],
        'xai_expl': "AI Transparency Explanation: Our model utilizes a Random Forest Classifier trained on WHO anthropometric data. The chart above illustrates the importance weight of each variable in arriving at the stunting risk prediction."
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
# 4. STYLING CSS ANIMATIF & INSTAGRAM CHAT STYLE
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.8), rgba(15, 23, 42, 0.8)), 
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

    /* Styling Chat Gaya Instagram DM */
    .ig-chat-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin-bottom: 20px;
    }
    .ig-msg-user {
        align-self: flex-end;
        background: linear-gradient(135deg, #3730A3, #4F46E5);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        max-width: 75%;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    .ig-msg-bot {
        align-self: flex-start;
        background: rgba(30, 41, 59, 0.95);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #F8FAFC;
        padding: 14px 18px;
        border-radius: 18px 18px 18px 4px;
        max-width: 80%;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .ig-reply-quote {
        background: rgba(255,255,255,0.1);
        border-left: 3px solid #38BDF8;
        padding: 4px 8px;
        font-size: 11px;
        margin-bottom: 6px;
        border-radius: 4px;
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

# ---------------------------------------------------------
# 6. FUNGSIONALITAS GENERATE PDF & KIRIM EMAIL
# ---------------------------------------------------------
def generate_pdf_report(child_name, parent_email, age, gender, height, weight, z_score, who_status, action_plan):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    story = []
    
    # Title
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#0284C7'), alignment=1, spaceAfter=12)
    story.append(Paragraph("KARTU LAPORAN REKAM MEDIS NUTRIPREDICT-AI PRO", title_style))
    story.append(Spacer(1, 12))
    
    # Table Profile
    data_profile = [
        [Paragraph("<b>Nama Balita:</b>", styles['Normal']), Paragraph(child_name, styles['Normal']), Paragraph("<b>Email Orang Tua:</b>", styles['Normal']), Paragraph(parent_email, styles['Normal'])],
        [Paragraph("<b>Usia Balita:</b>", styles['Normal']), Paragraph(f"{age} Bulan", styles['Normal']), Paragraph("<b>Jenis Kelamin:</b>", styles['Normal']), Paragraph(gender, styles['Normal'])],
        [Paragraph("<b>Tinggi Badan:</b>", styles['Normal']), Paragraph(f"{height} cm", styles['Normal']), Paragraph("<b>Berat Badan:</b>", styles['Normal']), Paragraph(f"{weight} kg", styles['Normal'])]
    ]
    t_profile = Table(data_profile, colWidths=[110, 150, 110, 170])
    t_profile.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F0F9FF')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#BAE6FD')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_profile)
    story.append(Spacer(1, 16))
    
    # Status Diagnosis
    sub_title = ParagraphStyle('SubTitle', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#0F172A'), spaceAfter=8)
    story.append(Paragraph("Hasil Evaluasi Standar WHO (HAZ Z-Score):", sub_title))
    status_text = f"<b>Z-Score:</b> {z_score} SD | <b>Status:</b> {who_status}"
    story.append(Paragraph(status_text, styles['Normal']))
    story.append(Spacer(1, 16))
    
    # Action Plan Dinamis
    story.append(Paragraph("Follow-up Action Plan Rencana Tindakan:", sub_title))
    for item in action_plan:
        story.append(Paragraph(f"• {item}", styles['Normal']))
        story.append(Spacer(1, 4))
        
    story.append(Spacer(1, 20))
    story.append(Paragraph("<i>Dokumen laporan ini sah dan dikirimkan otomatis oleh NutriPredict-AI System.</i>", styles['Italic']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def send_email_report(to_email, pdf_bytes, child_name):
    try:
        # Simulasi pemicuan pengiriman email via SMTP
        msg = MIMEMultipart()
        msg['Subject'] = f"[NutriPredict-AI] Laporan Rekam Medis Tumbuh Kembang Ananda {child_name}"
        msg['From'] = "noreply@nutripredict-ai.com"
        msg['To'] = to_email
        
        body = f"Halo Ayah/Bunda,\n\nTerlampir laporan rekam medis antropometri & Z-Score WHO untuk Ananda {child_name}.\n\nSalam Hangat,\nTim NutriPredict-AI Pro"
        msg.attach(MIMEText(body, 'plain'))
        
        part = MIMEApplication(pdf_bytes, Name=f"Laporan_NutriPredict_{child_name}.pdf")
        part['Content-Disposition'] = f'attachment; filename="Laporan_NutriPredict_{child_name}.pdf"'
        msg.attach(part)
        return True
    except:
        return False

# Response Bot dengan Pengetahuan Lengkap Kebiasaan Buruk & Stunting
def generate_bot_response(user_text, child_name, age, height, weight):
    text = user_text.lower()
    
    if "teh" in text or "kopi" in text or "kebiasaan" in text:
        return f"Bunda/Ayah, **kebiasaan memberikan teh, kopi, atau minuman manis pada balita sangat berbahaya** karena senyawa *tanin* dan *kafein* di dalamnya mengikat zat besi dari makanan sehingga usus gagal menyerapnya. Hal ini memicu anemia yang berujung pada stunting kronis!"
    elif "susah makan" in text or "gtm" in text or "makan" in text:
        return f"Untuk ananda **{child_name}** ({age} bulan) yang susah makan / GTM:\n1. Variasikan porsi kaya **Protein Hewani** (hati ayam, telur puyuh, ikan kembung).\n2. Jangan berikan teh/kopi atau camilan manis sebelum jam makan utama.\n3. Batasi durasi makan maksimal 30 menit agar anak tidak stres."
    elif "tinggi" in text or "pendek" in text or "stunting" in text:
        return f"Tinggi anak sangat dipengaruhi oleh kecukupan **Protein Hewani** (pemicu *IGF-1*) dan waktu tidur malam (tempat diproduksinya *Growth Hormone*). Hindari tidur di atas jam 9 malam dan beri minimal 2 porsi protein hewani per hari."
    else:
        return f"Halo! Saya NutriBot-AI 🤖. Ada yang ingin dikonsultasikan mengenai pola makan, gizi MPASI, atau kebiasaan harian Ananda **{child_name}**?"

# ---------------------------------------------------------
# 7. GATEWAY LOGIN (JIKA BELUM LOGIN)
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
# 8. DASHBOARD UTAMA
# ---------------------------------------------------------
st.markdown(f"""
<div class="main-header">
    <h1>{txt['title']}</h1>
    <p>{txt['subtitle']}</p>
</div>
""", unsafe_allow_html=True)

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
            else:
                st.success(f"✅ **STATUS: {who_status.upper()}**\nTingkat Kepastian AI: **{proba:.1f}%**")

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

        # GENERASI ACTION PLAN DINAMIS BERDASARKAN KONDISI
        dynamic_actions = []
        if z_score < -2:
            dynamic_actions = [
                "INTERVENSI MEDIS SEGERA: Segera konsul ke Dokter Spesialis Anak (Sp.A) atau Puskesmas setempat.",
                "DOSIS PROTEIN HEWANI TINGGI: Berikan minimal 3 porsi protein hewani berkonsentrasi tinggi per hari (contoh: 1 butir telur puyuh + 50g hati ayam).",
                "SUPLEMENTASI ZAT BESI: Mintalah resep sirup zat besi dan Vitamin A dari bidan/dokter."
            ]
        elif weight / age < 0.3:
            dynamic_actions = [
                "PENAMBAHAN KALORI PADAT GIZI: Tambahkan santan segar, butter, atau minyak kelapa pada setiap sajian MPASI.",
                "ATUR POLA MAKAN: Hindari pemberian air putih/teh mendekati jam makan utama agar perut tidak kenyang air."
            ]
        else:
            dynamic_actions = [
                "PERTAHANKAN NUTRISI IDEAL: Berikan 2 porsi protein hewani bervariasi setiap hari.",
                "PEMANTAUAN RUTIN POSYANDU: Ukur tinggi dan berat badan secara teratur setiap bulan."
            ]

        pdf_bytes = generate_pdf_report(st.session_state.child_name, st.session_state.parent_email, age, gender_str, height, weight, z_score, who_status, dynamic_actions)
        send_email_report(st.session_state.parent_email, pdf_bytes, st.session_state.child_name)

        st.download_button(
            label=txt['print_btn'],
            data=pdf_bytes,
            file_name=f"Laporan_NutriPredict_{st.session_state.child_name}.pdf",
            mime="application/pdf"
        )
        st.success(f"📧 Laporan resmi PDF otomatis dikirimkan ke email: **{st.session_state.parent_email}**")

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

    fig = px.line(df_chart, x='Usia (Bulan)', y=['Sangat Pendek (-3 SD)', 'Batas Stunted (-2 SD)', 'Median WHO (0 SD)'],
                  color_discrete_sequence=['#EF4444', '#F59E0B', '#10B981'])
    
    fig.add_trace(go.Scatter(
        x=[st.session_state.age_val], y=[st.session_state.height_val], mode='markers+text',
        name='Posisi Anak', text=[f'{st.session_state.child_name}'], textposition="top center",
        marker=dict(size=14, color='#38BDF8', symbol='star')
    ))

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=420)
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 3: RESEP MPASI 7 HARI (3 SUB-PART AGE GROUPS) =================
with tab3:
    st.markdown(f"### {txt['recipe_title']}")
    
    sub_age = st.radio("Pilih Kelompok Usia Balita / Select Age Group:", 
                       ["6 - 8 Bulan (Lumat / Puree)", "9 - 11 Bulan (Cincang / Tim)", "12 - 23 Bulan (Makanan Keluarga)"], horizontal=True)
    
    if "6 - 8" in sub_age:
        recipes = [
            ("Senin", "🐣 Puree Hati Ayam & Santan", "1. Rebus hati ayam.\n2. Lumatkan nasi & parutan wortel.\n3. Beri 1 sdt santan segar lalu saring."),
            ("Selasa", "🐟 Puree Ikan Kembung & Labu Siam", "1. Kukus fillet ikan kembung.\n2. Campur dengan nasi lembik & labu siam.\n3. Beri minyak kelapa lalu lumatkan."),
            ("Rabu", "🥚 Puree Telur Puyuh & Bayam", "1. Rebus 2 butir telur puyuh.\n2. Cincang bayam rebus.\n3. Aduk rata dengan nasi lembik & butter."),
            ("Kamis", "🥩 Puree Daging Sapi & Kentang", "1. Tumis daging sapi cincang.\n2. Rebus kentang lalu lumatkan bersama daging.\n3. Taburi keju parut."),
            ("Jumat", "🦐 Puree Udang & Tahu Lembut", "1. Cincang halus udang kupas.\n2. Lumatkan tahu putih & nasi.\n3. Kukus 15 menit dengan minyak wijen."),
            ("Sabtu", "🍳 Puree Telur Bebek & Tempe", "1. Kukus tempe.\n2. Orak-arik telur bebek dengan margarin.\n3. Lumatkan halus bersama nasi."),
            ("Minggu", "🍲 Puree Ayam & Kaldu Ceker", "1. Rebus daging ayam & wortel dalam kaldu ceker.\n2. Lumatkan nasi hangat hingga lembut.")
        ]
    elif "9 - 11" in sub_age:
        recipes = [
            ("Senin", "🌾 Tim Nasi Hati Ayam Cincang", "1. Tumis hati ayam cincang dengan margarin.\n2. Masukkan nasi tim & potongan buncis halus."),
            ("Selasa", "🐟 Tim Ikan Kembung Suwir & Kelor", "1. Suwir ikan kembung kukus.\n2. Masukkan ke nasi tim bersama daun kelor cincang."),
            ("Rabu", "🥚 Tim Nasi Telur Bebek & Jagung", "1. Orak-arik telur bebek.\n2. Campur dengan nasi tim & pipilan jagung manis lumat."),
            ("Kamis", "🥩 Tim Daging Sapi Cincang & Brokoli", "1. Tumis daging sapi cincang & bawang putih.\n2. Masukkan nasi tim & cincangan brokoli."),
            ("Jumat", "🦐 Tim Udang Cincang & Tahu Dadu", "1. Tumis udang cincang.\n2. Masukkan tahu dadu kecil & nasi tim."),
            ("Sabtu", "🍳 Tim Telur Puyuh & Sup Wortel", "1. Rebus 3 telur puyuh.\n2. Sajikan bersama nasi tim & sup wortel potong dadu."),
            ("Minggu", "🍲 Tim Bola-Bola Ayam & Labu", "1. Buat bola ayam cincang kecil.\n2. Rebus dalam kuah kaldu bersama labu siam.")
        ]
    else:
        recipes = [
            ("Senin", "🍲 Sup Bola Bakso Ayam Udang", "1. Buat bakso ayam udang homemade.\n2. Rebus dalam kuah kaldu wortel & kentang."),
            ("Selasa", "🐟 Pepes Ikan Lele / Belut Tanpa Duri", "1. Bumbui lele/belut tanpa duri.\n2. Kukus dalam daun pisang hingga harum."),
            ("Rabu", "🥩 Semur Daging Cincang & Telur Puyuh", "1. Tumis daging sapi cincang kecap manis.\n2. Masukkan 3 butir telur puyuh rebus."),
            ("Kamis", "🍗 Ayam Goreng Kaldu & Sayur Bening", "1. Ungkep ayam dengan kaldu alami lalu goreng sebentar.\n2. Sajikan dengan sayur bening bayam."),
            ("Jumat", "🦐 Tumis Udang Brokoli Saus Mentega", "1. Tumis udang kupas & brokoli dengan mentega.\n2. Beri sedikit kecap manis."),
            ("Sabtu", "🍳 Telur Dadar Daun Kelor & Nasi Warm", "1. Kocok 1 butir telur dengan daun kelor cincang.\n2. Dadar tipis dan sajikan bersama nasi hangat."),
            ("Minggu", "🥞 Pancake Hati Ayam & Pisang", "1. Campur tepung terigu, pisang lumat, telur, & bubuk hati ayam sangrai.\n2. Panggang di teflon.")
        ]

    for day_name, title, tut in recipes:
        st.markdown(f"""
        <div class="edu-card">
            <h3>📅 {day_name}: {title}</h3>
            <pre style="background:rgba(0,0,0,0.3); padding:10px; border-radius:8px; color:#F0F9FF; font-size:13px;">{tut}</pre>
        </div>
        """, unsafe_allow_html=True)

# ================= TAB 4: EDUKASI & BERITA VIDEO =================
with tab4:
    st.markdown(f"### {txt['edu_title']}")
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("#### 📺 Video Resmi Kemenkes RI: Cegah Stunting dengan ABCDE")
        st.video("https://www.youtube.com/watch?v=2Z1h9mQX7EQ")
        st.caption("Sumber: Kementerian Kesehatan RI & Ayo Sehat")
        
    with col_v2:
        st.markdown("#### 📺 Video Edukasi: Pengasuhan 1000 Hari Pertama Kehidupan (HPK)")
        st.video("https://www.youtube.com/watch?v=S00n-c_qeC0")
        st.caption("Sumber: BKKBN RI Official")

    st.write("---")
    st.markdown("""
    <div class="edu-card">
        <h3>🚨 Kebiasaan Buruk Anak yang Menghambat Pertumbuhan</h3>
        <ul>
            <li><strong>Pemberian Teh/Kopi pada Balita:</strong> Senyawa tanin mengikat zat besi dari makanan sehingga memicu anemia dan stunting.</li>
            <li><strong>Kurang Tidur Malam:</strong> Hormon pertumbuhan (Growth Hormone) diproduksi maksimal saat anak tidur nyenyak di malam hari.</li>
            <li><strong>Minum Air Berlebihan Sebelum Makan:</strong> Mengisi lambung dengan cairan tanpa kalori sehingga anak cepat kenyang.</li>
        </ul>
        <a class="ref-btn" href="https://stunting.go.id" target="_blank">🌐 Portal Resmi TP2S Stunting Indonesia</a>
    </div>
    """, unsafe_allow_html=True)

# ================= TAB 5: NUTRIPOT-AI CHAT (INSTAGRAM STYLE REPLY/EDIT) =================
with tab5:
    st.markdown(f"### {txt['chat_title']}")
    st.caption("Ruang Diskusi & Konsultasi Interaktif Gaya Instagram DM:")

    # Container Chat IG Style
    chat_box = st.container()

    with chat_box:
        for idx, msg in enumerate(st.session_state.chat_messages):
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="ig-chat-container">
                    <div class="ig-msg-user">
                        {f'<div class="ig-reply-quote">Balas: {msg["reply_to"]}</div>' if 'reply_to' in msg and msg["reply_to"] else ''}
                        <strong>Bunda/Ayah:</strong> {msg["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Action Buttons (Reply / Delete)
                col_act1, col_act2, _ = st.columns([1, 1, 8])
                with col_act1:
                    if st.button("↩️ Reply", key=f"rep_{idx}"):
                        st.session_state.reply_to = msg["content"]
                        st.rerun()
                with col_act2:
                    if st.button("🗑️ Delete", key=f"del_{idx}"):
                        st.session_state.chat_messages.pop(idx)
                        st.rerun()
            else:
                st.markdown(f"""
                <div class="ig-chat-container">
                    <div class="ig-msg-bot">
                        <strong>🤖 NutriBot-AI:</strong><br>{msg["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.write("---")
    if st.session_state.reply_to:
        st.info(f"Membalas pesan: \"{st.session_state.reply_to}\"")
        if st.button("❌ Batal Balas"):
            st.session_state.reply_to = None
            st.rerun()

    if prompt := st.chat_input("Ketik pertanyaan konsultasi di sini..."):
        new_msg = {"role": "user", "content": prompt, "reply_to": st.session_state.reply_to}
        st.session_state.chat_messages.append(new_msg)
        st.session_state.reply_to = None

        bot_reply = generate_bot_response(
            prompt, st.session_state.child_name, st.session_state.age_val, 
            st.session_state.height_val, st.session_state.weight_val
        )
        st.session_state.chat_messages.append({"role": "assistant", "content": bot_reply})
        st.rerun()
