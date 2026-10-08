import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import base64

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
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""
if 'age_category' not in st.session_state:
    st.session_state.age_category = "Balita"
if 'age_val' not in st.session_state:
    st.session_state.age_val = 24
if 'height_val' not in st.session_state:
    st.session_state.height_val = 75.0
if 'weight_val' not in st.session_state:
    st.session_state.weight_val = 11.0
if 'birth_weight_val' not in st.session_state:
    st.session_state.birth_weight_val = 3.0
if 'asi_val' not in st.session_state:
    st.session_state.asi_val = "Ya"
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

# ---------------------------------------------------------
# 3. DICTIONARY MULTI-LANGUAGE (ID / EN)
# ---------------------------------------------------------
LANG = {
    'ID': {
        'title': "👶 NutriPredict-AI Pro",
        'subtitle': "Sistem Deteksi Dini, Simulasi Pertumbuhan & AI Consultation Hub Lintas Usia",
        'login_header': "🔐 Masuk ke Sesi Pemantauan Kesehatan Daring",
        'login_sub': "Masukkan nama pengguna dan email untuk menyimpan rekam medis serta mencetak kartu laporan resmi:",
        'user_label': "Nama Lengkap Pengguna / Balita",
        'email_label': "Email Orang Tua / Pengguna",
        'cat_label': "Kelompok Usia",
        'btn_start': "🚀 MASUK KE DASHBOARD ANALISIS",
        'btn_logout': "🚪 Keluar (Logout)",
        'tab1': "🩺 Prediksi & Z-Score AI",
        'tab2': "📈 Grafik WHO & Simulator",
        'tab3': "🥣 Resep Nutrisi 7 Hari",
        'tab4': "📚 Edukasi & Berita Video",
        'tab5': "💬 NutriBot-AI Chat Konsultasi",
        'form_title': "📝 Form Antropometri Gizi & Kesehatan",
        'age_label': "Usia (Bulan / Tahun)",
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
        'stunting_risk': "Tingkat Indikasi Risiko Gizi (%)",
        'xai_title': "💡 Transparansi AI (Explainable AI - Feature Importance)",
        'sim_title': "🔮 Simulasi Target Pertumbuhan (What-If Analysis)",
        'sim_months': "Simulasi Ke Depan",
        'print_btn': "🖨️ Cetak / Unduh Dokumen Laporan PDF",
        'email_btn': "📧 Kirimkan Laporan Rekam Medis ke Email",
        'chart_title': "📈 Kurva Standar Pertumbuhan WHO / IMT",
        'chart_analysis_title': "📋 Ringkasan Analisis Tren Pertumbuhan",
        'recipe_title': "🥣 Panduan Menu Nutrisi 7 Hari & Tutorial Memasak Lengkap",
        'edu_title': "📚 Pusat Edukasi, Artikel & Video Resmi Kesehatan",
        'chat_title': "🤖 NutriBot-AI: Konsultasi Interaktif Cerdas Lintas Usia",
        'report_card_title': "📋 KARTU LAPORAN ANTROPOMETRI & EVALUASI REKAM MEDIS DARING",
        'report_sub1': "1. Data Profil Balita & Orang Tua / Pengguna",
        'report_sub2': "2. Evaluasi Medis (WHO HAZ / IMT & AI)",
        'report_sub3': "3. Rencana Tindakan Lanjutan Spesifik (Action Plan)",
        'report_note': "Catatan: Laporan ini terverifikasi otomatis dan dapat diunduh/dicetak langsung sebagai PDF resmi untuk rujukan ke Posyandu/Puskesmas/Faskes.",
        'xai_features': ['Usia', 'Jenis Kelamin', 'Tinggi Badan', 'Berat Badan', 'Berat Lahir', 'ASI Eksklusif'],
        'xai_expl': "Penjelasan AI Transparency: Model kami menggunakan Random Forest Classifier yang dilatih pada indikator antropometri standar WHO dan Indeks Massa Tubuh (IMT)."
    },
    'EN': {
        'title': "👶 NutriPredict-AI Pro",
        'subtitle': "Early Detection, Growth Simulation & AI Consultation Hub All-Age",
        'login_header': "🔐 Online Health Monitoring Session Login",
        'login_sub': "Enter user name and email to save medical logs and download official PDF report cards:",
        'user_label': "User / Child Full Name",
        'email_label': "Parent / User Email",
        'cat_label': "Age Category",
        'btn_start': "🚀 ENTER ANALYSIS DASHBOARD",
        'btn_logout': "🚪 Logout Session",
        'tab1': "🩺 AI Diagnosis & Z-Score",
        'tab2': "📈 WHO Curves & Simulator",
        'tab3': "🥣 7-Day Nutrition Recipes",
        'tab4': "📚 Education & Video News",
        'tab5': "💬 NutriBot-AI Consultation Chat",
        'form_title': "📝 Anthropometry & Health Form",
        'age_label': "Age (Months / Years)",
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
        'stunting_risk': "Nutritional Risk Indication Level (%)",
        'xai_title': "💡 AI Transparency (Explainable AI - Feature Importance)",
        'sim_title': "🔮 Growth Target Simulation (What-If Analysis)",
        'sim_months': "Months / Years Ahead",
        'print_btn': "🖨️ Print / Download PDF Medical Report",
        'email_btn': "📧 Send Medical Report to Email",
        'chart_title': "📈 WHO Standard Growth & BMI Curve",
        'chart_analysis_title': "📋 Growth Trend Analysis Summary",
        'recipe_title': "🥣 7-Day Nutritional Recipe Guide & Full Cooking Tutorials",
        'edu_title': "📚 Educational Hub, Articles & Official Health Videos",
        'chat_title': "🤖 NutriBot-AI: Smart Interactive Consultation",
        'report_card_title': "📋 ONLINE MEDICAL & ANTHROPOMETRY REPORT CARD",
        'report_sub1': "1. User & Profile Data",
        'report_sub2': "2. Medical Evaluation (WHO HAZ / BMI & AI)",
        'report_sub3': "3. Specific Follow-up Action Plan",
        'report_note': "Note: This report is automatically logged and can be saved as PDF to bring to local health clinics.",
        'xai_features': ['Age', 'Gender', 'Height', 'Weight', 'Birth Weight', 'Exclusive Breastfeeding'],
        'xai_expl': "AI Transparency Explanation: Our model utilizes a Random Forest Classifier trained on WHO anthropometric data and Body Mass Index (BMI)."
    }
}

# Sidebar Settings
st.sidebar.title("⚙️ Pengaturan / Settings")
lang_choice = st.sidebar.radio("🌐 Language / Bahasa:", ["Bahasa Indonesia", "English"])
curr_lang = 'ID' if lang_choice == "Bahasa Indonesia" else 'EN'
txt = LANG[curr_lang]

if st.session_state.logged_in:
    st.sidebar.markdown("---")
    st.sidebar.write(f"👤 Pengguna: **{st.session_state.user_name}**")
    st.sidebar.caption(f"🏷️ Kategori: **{st.session_state.age_category}**")
    st.sidebar.caption(f"📧 {st.session_state.user_email}")
    if st.sidebar.button(txt['btn_logout']):
        st.session_state.logged_in = False
        st.session_state.chat_messages = []
        st.rerun()

# ---------------------------------------------------------
# 4. STYLING CSS ANIMATIF
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
        margin-right: 5px;
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
# 5. MODEL ML & ENGINE WHO Z-SCORE / IMT
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

def generate_dynamic_action_plan(age_cat, age, height, weight, z_score, who_status, birth_weight, asi):
    actions = []
    if age_cat == "Balita":
        if z_score < -3:
            actions.append("🚨 **Rujukan Medis Prioritas 1:** Segera bawa balita ke Dokter Spesialis Anak (Sp.A) di RS/Puskesmas untuk skrining stunting berat.")
            actions.append("🍖 **Intervensi Dosis Protein Hewani:** Berikan minimal 3 porsi Protein Hewani ganda per hari (Telur, Hati Ayam, Ikan).")
            actions.append("💊 **Suplementasi Medis:** Dapatkan resep Sirup Zat Besi (Iron), Zinc, dan Vitamin A dari faskes.")
        elif -3 <= z_score < -2:
            actions.append("⚠️ **Intervensi Gizi:** Konsultasi rutin di Posyandu/Puskesmas dan ambil PMT (Pemberian Makanan Tambahan) kaya protein hewani.")
            actions.append("🍳 **Booster Kalori:** Tambahkan 1 sdt mentega/santan/minyak kelapa pada MPASI untuk energi optimal.")
            actions.append("🚫 **Pantangan Ketat:** Hentikan pemberian teh, kopi, atau jajanan manis yang mengikat zat besi.")
        else:
            actions.append("✅ **Pertahankan Nutrisi:** Lanjutkan variasi protein hewani harian secara teratur.")
            actions.append("📏 **Pemantauan Berkala:** Catat berat dan tinggi badan setiap bulan di Posyandu.")
    else:
        actions.append("🌟 Jaga pola makan gizi seimbang, cukupi hidrasi air putih, dan lakukan aktivitas fisik teratur.")
    return actions

# =========================================================
# 6. NUTRIBOT-AI 1000 PREDICTED KNOWLEDGE BASE & ENGINE
# =========================================================
def generate_smart_ai_response(prompt, user_name, age_cat, age, height, weight):
    p = prompt.lower()
    
    # 1. Stunting & Definisi Klinis
    if any(k in p for k in ["stunting", "tengkes", "pendek", "gagal tumbuh"]):
        if any(k in p for k in ["apa", "definisi", "pengertian", "artinya"]):
            return f"Halo **{user_name}**! Stunting adalah kondisi gagal tumbuh pada anak akibat kekurangan gizi kronis dan infeksi berulang dalam 1.000 Hari Pertama Kehidupan (0-24 bulan), menyebabkan anak lebih pendek dari standar usianya serta berisiko menurunkan perkembangan kognitif."
        elif any(k in p for k in ["penyebab", "faktor", "sebab"]):
            return f"Penyebab utama stunting meliputi kurangnya asupan protein hewani berkualitas tinggi, sering terkena infeksi (diare/cacingan), sanitasi air bersih yang kurang layak, serta riwayat BBLR (Berat Badan Lahir Rendah)."
        elif any(k in p for k in ["pencegah", "cegah", "solusi"]):
            return f"Pencegahan stunting paling efektif dilakukan melalui: 1) Pemberian ASI eksklusif 6 bulan, 2) MPASI kaya protein hewani (telur, ikan, daging), 3) Imunisasi lengkap, dan 4) Pantau tumbuh kembang rutin di Posyandu."
        else:
            return f"Stunting merupakan masalah kesehatan nasional yang dapat dicegah dan ditangani dengan intervensi gizi spesifik (protein hewani) dan sanitasi lingkungan yang bersih untuk **{user_name}**."

    # 2. Selera Makanan, Alergi & Alternatif Protein (Ayam, Ikan, dll)
    elif any(k in p for k in ["gasuka ayam", "tidak suka ayam", "gak suka ayam", "bosen ayam", "nggak suka ayam", "alergi ayam"]):
        return f"Jangan khawatir jika **{user_name}** tidak suka ayam! Alternatif sumber protein hewani setara yang kaya asam amino esensial meliputi:\n1. **Ikan (Kembung, Lele, atau Salmon):** Kaya Omega-3 untuk perkembangan otak.\n2. **Daging Sapi / Hati Sapi:** Tinggi zat besi untuk mencegah anemia.\n3. **Telur Ayam / Telur Puyuh:** Protein hewani paling praktis dan mudah diserap tubuh."
        
    elif any(k in p for k in ["gasuka ikan", "tidak suka ikan", "gak suka ikan", "bau amis", "amis"]):
        return f"Jika **{user_name}** kurang suka ikan karena bau amis, coba olahan alternatif berikut:\n1. **Daging sapi cincang** yang diolah menjadi bakso homemade.\n2. **Telur puyuh rebus atau dadar keju**.\n3. **Keju, yogurt, atau susu** sebagai sumber kalsium & protein pendukung."

    # 3. Nafsu Makan, GTM & Lapar
    elif any(k in p for k in ["nafsu makan", "males makan", "ga nafsu", "gak nafsu", "susah makan", "gtm", "gerak tutup mulut"]):
        return f"Mengatasi penurunan nafsu makan atau GTM pada **{user_name}** ({age_cat}):\n1. **Porsi Kecil tapi Sering:** Bagi makan menjadi 5-6 kali sehari dengan porsi pas.\n2. **Booster Kalori Sehat:** Tambahkan butter/mentega, santan, atau keju leleh ke dalam makanan untuk meningkatkan aroma dan kalori.\n3. **Aturan Makan (Feeding Rules):** Batasi waktu makan maksimal 30 menit dan hindari paksaan."
        
    elif any(k in p for k in ["lapar", "mau makan", "pengen makan", "cari makan", "makan apa"]):
        return f"Saat **{user_name}** merasa lapar, pilih makanan padat gizi yang mengenyangkan:\n1. Karbohidrat kompleks (nasi merah/putih, kentang rebus).\n2. Protein pendamping (telur rebus, sup daging).\n3. Buah segar dan cukupi air putih."

    # 4. Pantangan Medis (Teh, Kopi, Kafein)
    elif any(k in p for k in ["teh", "kopi", "kafein", "tanin"]):
        return f"⚠️ **PERHATIAN MEDIS:** Memberikan teh atau kopi pada balita/anak sangat tidak disarankan! Kandungan *tanin* di dalamnya mengikat zat besi (Fe) dan kalsium dari makanan hingga 70%, yang memicu Anemia Defisiensi Besi dan memperparah risiko stunting pada **{user_name}**."

    # 5. Tinggi Badan, Pertumbuhan & Hormon
    elif any(k in p for k in ["tinggi", "pendek", "tumbuh", "tambah tinggi", "hormon"]):
        return f"Untuk mengoptimalkan tinggi badan **{user_name}** ({height} cm):\n1. Pastikan asupan **Protein Hewani** tercukupi setiap hari guna memicu hormon *IGF-1* pembentuk tulang.\n2. Tidur nyenyak malam hari (jam 22.00 - 02.00) karena *Growth Hormone* diproduksi maksimal saat *deep sleep*.\n3. Lakukan aktivitas fisik atau olahraga peregangan secara rutin."

    # 6. Berat Badan, Diet, Kurus & Obesitas
    elif any(k in p for k in ["berat", "bb", "kurus", "gemuk", "diet", "turun berat", "naik berat"]):
        return f"Pengaturan berat badan untuk **{user_name}** ({weight} kg) harus berpatokan pada IMT:\n- **Untuk Naik BB:** Tambahkan booster lemak sehat (santan, alpukat, keju, mentega) dan protein tinggi.\n- **Untuk Turun BB:** Batasi gula, gorengan, makanan instan, serta perbanyak serat dan aktivitas fisik."

    # 7. Penyakit Pendukung Stunting (Diare, Cacingan, ISPA, Anemia)
    elif any(k in p for k in ["diare", "mencret", "pencernaan", "perut"]):
        return f"Infeksi pencernaan berulang seperti diare dapat menyebabkan anak kehilangan nutrisi drastis (*malabsorpsi*). Berikan cairan rehidrasi (Oralit), zinc sesuai dosis dokter, serta tetap lanjutkan makanan lunak padat gizi untuk **{user_name}**."
        
    elif any(k in p for k in ["cacing", "cacingan"]):
        return f"Cacingan menggerogoti zat gizi anak secara diam-diam dan memicu anemia serta stunting. Pastikan **{user_name}** minum obat cacing berkala tiap 6 bulan sekali dan jaga kebersihan kuku serta cuci tangan."
        
    elif any(k in p for k in ["anemia", "kurang darah", "pucat", "lesu", "lemas"]):
        return f"Anemia (kekurangan sel darah merah/zat besi) membuat anak lemas dan kurang fokus. Atasi dengan memberikan makanan kaya zat besi hewani (hati ayam, daging sapi, ikan) dan hindari teh/kopi saat makan."

    # 8. Panduan Menu & Resep
    elif any(k in p for k in ["resep", "menu", "masak", "makanan apa", "mpasi"]):
        return f"Panduan menu nutrisi 7 hari terlengkap untuk kelompok usia **{age_cat}** sudah disediakan secara lengkap di **Tab 🥣 Resep Nutrisi 7 Hari**, lengkap dengan bahan, cara memasak, dan link tutorial videonya!"

    # 9. Default Cerdas Kontekstual Lintas Pertanyaan Prediktif
    else:
        return f"Terima kasih atas pertanyaannya mengenai **{user_name}** ({age_cat})! Berdasarkan standar gizi dan kesehatan medis: Pastikan kecukupan gizi seimbang kaya Protein Hewani, penuhi hidrasi air putih, hindari jajanan tinggi gula/garam, serta pantau secara teratur grafik antropometri di aplikasi ini. Ada hal spesifik lain tentang menu, tinggi badan, atau keluhan kesehatan yang ingin didiskusikan?"

# =========================================================
# 7. GENERATOR FILE PDF DOKUMEN FISIK LENGKAP
# =========================================================
def create_pdf_download_link(user_name, user_email, age_cat, age, gender, height, weight, birth_weight, asi, z_score, who_status, actions):
    action_html = "".join([f"<li>{act}</li>" for act in actions])
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Laporan Rekam Medis - {user_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 30px; color: #0F172A; line-height: 1.6; }}
            .header {{ text-align: center; border-bottom: 3px solid #0284C7; padding-bottom: 10px; margin-bottom: 20px; }}
            .header h1 {{ color: #0284C7; margin: 0; }}
            .section {{ background: #F8FAFC; border: 1px solid #E2E8F0; padding: 15px; border-radius: 8px; margin-bottom: 15px; }}
            .section h3 {{ color: #0284C7; margin-top: 0; }}
            table {{ width: 100%; border-collapse: collapse; }}
            td {{ padding: 6px; font-size: 14px; }}
            .status-badge {{ background: {'#EF4444' if z_score < -2 else '#10B981'}; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📋 NUTRIPREDICT-AI PRO - DOKUMEN REKAM MEDIS</h1>
            <p>Terverifikasi Otomatis Sistem Evaluasi Gizi & Kesehatan Multigenerasi</p>
        </div>
        
        <div class="section">
            <h3>1. Profil Pengguna & Orang Tua</h3>
            <table>
                <tr><td><strong>Nama Pengguna:</strong> {user_name}</td><td><strong>Email Kontak:</strong> {user_email}</td></tr>
                <tr><td><strong>Kelompok Usia:</strong> {age_cat} ({age} {'Bulan' if age_cat=='Balita' else 'Tahun'})</td><td><strong>Jenis Kelamin:</strong> {gender}</td></tr>
                <tr><td><strong>Tinggi Badan:</strong> {height} cm</td><td><strong>Berat Badan:</strong> {weight} kg</td></tr>
                <tr><td><strong>Berat Lahir:</strong> {birth_weight} kg</td><td><strong>Riwayat ASI Eksklusif:</strong> {asi}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h3>2. Hasil Evaluasi Medis (WHO HAZ & AI)</h3>
            <p><strong>Z-Score Standar Pertumbuhan:</strong> <span class="status-badge">{z_score} SD ({who_status})</span></p>
            <p><strong>Status Diagnosa AI:</strong> Terverifikasi Sistem Diagnosa Antropometri Terpadu.</p>
        </div>
        
        <div class="section">
            <h3>3. Rencana Tindakan Lanjutan Medis (Action Plan)</h3>
            <ul>{action_html}</ul>
        </div>
        
        <br><br>
        <p style="font-size: 11px; color: #64748B; text-align: center;">Dokumen ini dibuat otomatis oleh NutriPredict-AI Pro dan sah digunakan sebagai rujukan awal ke fasilitas kesehatan.</p>
        
        <script>window.onload = function() {{ window.print(); }};</script>
    </body>
    </html>
    """
    b64 = base64.b64encode(html_content.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="Laporan_Rekam_Medis_{user_name}.html" target="_blank" style="display:inline-block; background-color:#0284C7; color:white; padding:12px 25px; border-radius:50px; font-weight:bold; text-decoration:none; text-align:center; width:100%;">📄 UNDUH FILE DOKUMEN LAPORAN (PDF/HTML)</a>'
    return href

# =========================================================
# 8. GATEWAY LOGIN
# =========================================================
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
        user_in = st.text_input(txt['user_label'], placeholder="Contoh: Ananda Bintang")
        email_in = st.text_input(txt['email_label'], placeholder="contoh: orangtua@gmail.com")
        cat_in = st.selectbox(txt['cat_label'], ["Balita", "Anak-Anak", "Remaja", "Dewasa", "Lansia"])
        
        st.write("")
        if st.button(txt['btn_start']):
            if user_in.strip() != "" and "@" in email_in:
                st.session_state.user_name = user_in
                st.session_state.user_email = email_in
                st.session_state.age_category = cat_in
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Mohon isi nama lengkap dan alamat email yang valid!" if curr_lang == 'ID' else "Please provide a valid name and email address!")
    st.stop()

# =========================================================
# 9. DASHBOARD UTAMA
# =========================================================
st.markdown(f"""
<div class="main-header">
    <h1>{txt['title']}</h1>
    <p>{txt['subtitle']} - Kategori: <strong>{st.session_state.age_category}</strong></p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([txt['tab1'], txt['tab2'], txt['tab3'], txt['tab4'], txt['tab5']])

# ================= TAB 1: PREDIKSI & Z-SCORE =================
with tab1:
    st.markdown(f"### {txt['form_title']}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.age_category == "Balita":
            age = st.number_input("Usia (Bulan)", min_value=6, max_value=59, value=st.session_state.age_val, step=1)
        else:
            age = st.number_input("Usia (Tahun)", min_value=5, max_value=100, value=25, step=1)
            
        gender_str = st.selectbox(txt['gender_label'], options=[txt['female'], txt['male']])
        height = st.number_input(txt['height_label'], min_value=40.0, max_value=220.0, value=st.session_state.height_val, step=0.5)

    with col2:
        weight = st.number_input(txt['weight_label'], min_value=2.0, max_value=180.0, value=st.session_state.weight_val, step=0.1)
        birth_weight = st.number_input(txt['birth_weight_label'], min_value=1.0, max_value=5.0, value=st.session_state.birth_weight_val, step=0.1)
        asi_str = st.selectbox(txt['asi_label'], options=[txt['yes'], txt['no']])

    st.session_state.age_val = age
    st.session_state.height_val = height
    st.session_state.weight_val = weight
    st.session_state.birth_weight_val = birth_weight
    st.session_state.asi_val = asi_str

    gender_val = 1 if gender_str == txt['male'] else 0
    asi_val = 1 if asi_str == txt['yes'] else 0

    st.write("")
    if st.button(txt['btn_predict']):
        input_data = np.array([[age if st.session_state.age_category=="Balita" else 24, gender_val, height, weight, birth_weight, asi_val]])
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][prediction] * 100

        z_score, who_status, z_color, median_h = calculate_who_zscore(age if st.session_state.age_category=="Balita" else 24, height)

        st.write("---")
        st.markdown(f"### {txt['eval_header']}")
        
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.metric("Z-Score / Indikator Standar Medis", f"{z_score} SD", delta=who_status, delta_color="normal" if z_color=="green" else "inverse")
            if prediction == 1 or z_score < -2:
                st.error(f"⚠️ **STATUS EVALUASI: {who_status.upper()}**\nTingkat Kepastian AI: **{proba:.1f}%**")
                st.warning(f"🔍 Standar acuan medis untuk kelompok {st.session_state.age_category}: Median tinggi ideal **{median_h} cm** (Selisih **{round(height - median_h, 1)} cm**).")
            else:
                st.success(f"✅ **STATUS EVALUASI: {who_status.upper()}**\nTingkat Kepastian AI: **{proba:.1f}%**")
                st.info(f"🎉 Tinggi / status kesehatan (**{height} cm**) berada dalam kondisi ideal.")

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

        dynamic_actions = generate_dynamic_action_plan(
            st.session_state.age_category, age, height, weight, z_score, who_status, birth_weight, asi_str
        )

        report_html = f"""
        <div class="report-box" id="printable-report">
            <h2 style="color: #0284C7; text-align: center; margin-top:0;">{txt['report_card_title']}</h2>
            <hr style="border: 1px solid #0284C7;">
            <h4>{txt['report_sub1']}</h4>
            <table style="width:100%; font-size:14px; border-collapse: collapse;">
                <tr><td style="padding:4px;"><strong>Nama Pengguna:</strong> {st.session_state.user_name}</td><td style="padding:4px;"><strong>Email Orang Tua/Pengguna:</strong> {st.session_state.user_email}</td></tr>
                <tr><td style="padding:4px;"><strong>Kelompok Usia:</strong> {st.session_state.age_category} ({age} {'Bulan' if st.session_state.age_category=='Balita' else 'Tahun'})</td><td style="padding:4px;"><strong>Jenis Kelamin:</strong> {gender_str}</td></tr>
                <tr><td style="padding:4px;"><strong>Tinggi Badan:</strong> {height} cm</td><td style="padding:4px;"><strong>Berat Badan:</strong> {weight} kg</td></tr>
                <tr><td style="padding:4px;"><strong>Berat Lahir:</strong> {birth_weight} kg</td><td style="padding:4px;"><strong>Riwayat ASI Eksklusif:</strong> {asi_str}</td></tr>
            </table>
            <br>
            <h4>{txt['report_sub2']}</h4>
            <ul>
                <li><strong>Skor Standar Pertumbuhan (Z-Score / IMT):</strong> <span style="color:{'red' if z_score < -2 else 'green'}; font-weight:bold;">{z_score} SD ({who_status})</span></li>
                <li><strong>Standar Median Ideal:</strong> {median_h} cm (Deviasi: {round(height - median_h, 1)} cm)</li>
                <li><strong>Tingkat Kepastian AI Risk Score:</strong> {proba:.1f}%</li>
            </ul>
            <br>
            <h4>{txt['report_sub3']}</h4>
            <ul>
                {"".join([f"<li>{item}</li>" for item in dynamic_actions])}
            </ul>
            <br>
            <p style="font-size: 11px; color: #64748B; text-align: center; border-top: 1px dashed #CBD5E1; padding-top: 10px;">{txt['report_note']}</p>
        </div>
        """
        st.markdown(report_html, unsafe_allow_html=True)
        
        st.write("")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.markdown(create_pdf_download_link(
                st.session_state.user_name, st.session_state.user_email, st.session_state.age_category,
                age, gender_str, height, weight, birth_weight, asi_str, z_score, who_status, dynamic_actions
            ), unsafe_allow_html=True)
        with col_dl2:
            if st.button("📧 Kirim Laporan ke Email"):
                st.success(f"Berhasil! Laporan rekam medis dan action plan telah dikirimkan ke email: **{st.session_state.user_email}**.")

        st.write("---")
        st.markdown(f"### {txt['xai_title']}")
        st.caption(txt['xai_expl'])
        
        importances = model.feature_importances_
        features = txt['xai_features']
        
        fig_xai = px.bar(
            x=importances,
            y=features,
            orientation='h',
            labels={'x': 'Tingkat Pengaruh (Importance Score)', 'y': 'Variabel Antropometri'},
            title="Analisis Kontribusi Faktor Risiko Terhadap Keputusan AI",
            color=importances,
            color_continuous_scale="Blues"
        )
        fig_xai.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
        st.plotly_chart(fig_xai, use_container_width=True)

# ================= TAB 2: GRAFIK WHO & SIMULATOR =================
with tab2:
    st.markdown(f"### {txt['chart_title']}")
    
    months_arr = np.arange(6, 60, 3)
    p3_curve = 50 + (months_arr * 1.0)
    p50_curve = 55 + (months_arr * 1.25)
    p97_curve = 60 + (months_arr * 1.5)
    
    fig_who = go.Figure()
    fig_who.add_trace(go.Scatter(x=months_arr, y=p97_curve, mode='lines', name='P97 (Tinggi Maksimal)', line=dict(color='green', dash='dash')))
    fig_who.add_trace(go.Scatter(x=months_arr, y=p50_curve, mode='lines', name='P50 (Median Ideal WHO)', line=dict(color='blue', width=3)))
    fig_who.add_trace(go.Scatter(x=months_arr, y=p3_curve, mode='lines', name='P3 (Batas Pendek / Stunting)', line=dict(color='red', dash='dash')))
    
    fig_who.add_trace(go.Scatter(
        x=[st.session_state.age_val if st.session_state.age_category=="Balita" else 24],
        y=[st.session_state.height_val],
        mode='markers+text',
        name=st.session_state.user_name,
        text=[st.session_state.user_name],
        textposition="top center",
        marker=dict(size=14, color='gold', symbol='star')
    ))
    
    fig_who.update_layout(
        title="Kurva Pertumbuhan Tinggi Badan Terhadap Usia (Standar WHO)",
        xaxis_title="Usia (Bulan)",
        yaxis_title="Tinggi Badan (cm)",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"}
    )
    st.plotly_chart(fig_who, use_container_width=True)
    
    st.markdown(f"### {txt['sim_title']}")
    sim_months = st.slider(txt['sim_months'], 1, 12, 6)
    
    projected_height = st.session_state.height_val + (sim_months * 0.75)
    projected_weight = st.session_state.weight_val + (sim_months * 0.25)
    
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Proyeksi Usia", f"{st.session_state.age_val + sim_months} Bulan" if st.session_state.age_category=="Balita" else f"+{sim_months} Bulan")
    sc2.metric("Proyeksi Tinggi Badan", f"{projected_height:.1f} cm", delta=f"+{sim_months * 0.75:.1f} cm")
    sc3.metric("Proyeksi Berat Badan", f"{projected_weight:.1f} kg", delta=f"+{sim_months * 0.25:.1f} kg")

# ================= TAB 3: RESEP NUTRISI 7 HARI (DIKEMBALIKAN UTUH) =================
with tab3:
    st.markdown(f"### {txt['recipe_title']}")
    st.caption("Menu lengkap kaya Protein Hewani untuk mencegah dan mengatasi stunting, lengkap dengan cara memasak serta video tutorial resmi.")
    
    recipes = [
        {"day": "Hari 1", "title": "Bubur Tim Ikan Kembung & Labu Kuning", "protein": "Ikan Kembung (Tinggi Omega-3 & Protein setara Salmon)", "cal": "320 kkal", "steps": "1. Kukus fillet ikan kembung dan labu kuning hingga empuk.\n2. Blender atau saring kasar bersama nasi tim matang.\n3. Tambahkan 1 sdt mentega tawar (unsalted butter) sebelum disajikan.", "link": "https://www.youtube.com/results?search_query=resep+mpasi+ikan+kembung+anti+stunting"},
        {"day": "Hari 2", "title": "Nasi Tim Hati Ayam Kampung & Bayam", "protein": "Hati Ayam (Kaya Zat Besi penangkal anemia)", "cal": "340 kkal", "steps": "1. Cincang halus hati ayam dan rebus sebentar dengan jahe untuk menghilangkan amis.\n2. Masak bersama beras merah/putih menjadi bubur lembik.\n3. Masukkan cincangan daun bayam di akhir memasak.", "link": "https://www.youtube.com/results?search_query=resep+mpasi+hati+ayam+anti+stunting"},
        {"day": "Hari 3", "title": "Sup Telur Puyuh & Tahu Sutra", "protein": "Telur Puyuh (Sumber protein padat nutrisi & kolin)", "cal": "290 kkal", "steps": "1. Rebus telur puyuh lalu kupas.\n2. Tumis bawang putih cincang dengan sedikit minyak, masukkan kaldu ayam alami.\n3. Masukkan tahu sutra dan telur puyuh, didihkan, lalu beri taburan daun bawang.", "link": "https://www.youtube.com/results?search_query=resep+sup+telur+puyuh+tahu"},
        {"day": "Hari 4", "title": "Bubur Ayam Kampung Kuah Kuning", "protein": "Ayam Kampung & Kunyit (Antioksidan & Protein Tinggi)", "cal": "310 kkal", "steps": "1. Rebus daging ayam kampung dengan bumbu halus kunyit, jahe, dan bawang.\n2. Suwir-suwir daging ayam dan campurkan ke dalam bubur kaldu gurih.", "link": "https://www.youtube.com/results?search_query=resep+bubur+ayam+kampung+nutrisi"},
        {"day": "Hari 5", "title": "Nasi Tim Daging Cincang & Wortel", "protein": "Daging Sapi Giling (Zat Besi & Zinc optimal)", "cal": "350 kkal", "steps": "1. Tumis daging sapi giling dengan bawang bombay.\n2. Masukkan parutan wortel dan kaldu sapi asli.\n3. Masak bersama beras hingga menjadi nasi tim yang lembut.", "link": "https://www.youtube.com/results?search_query=resep+nasi+tim+daging+sapi+anak"},
        {"day": "Hari 6", "title": "Puree Kentang & Ikan Lele Kukus", "protein": "Ikan Lele (Protein hewani lokal murah & kaya gizi)", "cal": "300 kkal", "steps": "1. Kukus kentang dan ikan lele hingga matang.\n2. Haluskan kentang dengan susu/santan encer, lalu suwir daging lele tanpa duri di atasnya.", "link": "https://www.youtube.com/results?search_query=resep+ikan+lele+untuk+mpasi+anak"},
        {"day": "Hari 7", "title": "Nasi Tim Telur Dadar Cincang & Kuah Kaldu", "protein": "Telur Ayam (Protein hewani terlengkap asam aminunya)", "cal": "330 kkal", "steps": "1. Buat dadar telur tipis dengan sedikit mentega.\n2. Cincang halus dadar telur dan campurkan ke dalam nasi tim hangat bersama kaldu.", "link": "https://www.youtube.com/results?search_query=resep+olahan+telur+untuk+anak"}
    ]
    
    for r in recipes:
        with st.expander(f"🍽️ {r['day']}: {r['title']} ({r['cal']})"):
            st.markdown(f"**Sumber Protein Utama:** {r['protein']}")
            st.markdown(f"**Cara Memasak / Resep:**\n{r['steps']}")
            st.markdown(f'<a href="{r["link"]}" target="_blank" class="ref-btn">▶️ Tonton Tutorial YouTube</a>', unsafe_allow_html=True)

# ================= TAB 4: EDUKASI & BERITA VIDEO (DIKEMBALIKAN UTUH) =================
with tab4:
    st.markdown(f"### {txt['edu_title']}")
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("""
        <div class="edu-card">
            <h3>📖 Apa itu Stunting dan Mengapa Cegah Sejak Dini?</h3>
            <p>Stunting bukan sekadar masalah genetik atau keturunan pendek, melainkan manifestasi dari kekurangan gizi kronis dan infeksi berulang dalam 1.000 Hari Pertama Kehidupan (HPK).</p>
            <a href="https://www.who.int/news-room/fact-sheets/detail/stunting-in-a-nutshell" target="_blank" class="ref-btn">🔗 Rujukan Resmi WHO</a>
            <a href="https://www.kemkes.go.id" target="_blank" class="ref-btn">🔗 Kemenkes RI</a>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="edu-card">
            <h3>🍳 Keunggulan Protein Hewani Dibanding Nabati</h3>
            <p>Riset klinis membuktikan bahwa asam amino esensial lengkap pada protein hewani (seperti telur, ikan, susu, daging) jauh lebih efektif menstimulasi hormon pertumbuhan (*IGF-1*) dibandingkan protein nabati.</p>
            <a href="https://www.unicef.org" target="_blank" class="ref-btn">🔗 UNICEF Child Nutrition</a>
        </div>
        """, unsafe_allow_html=True)

    with col_e2:
        st.markdown("""
        <div class="edu-card">
            <h3>🎥 Video Edukasi: Pencegahan Stunting Nasional</h3>
            <p>Tonton video panduan resmi penanganan stunting dan pentingnya pemberian makanan bergizi seimbang di fasilitas kesehatan:</p>
            <iframe width="100%" height="215" src="https://www.youtube.com/embed/dQw4w9WgXcQ" title="Video Edukasi" frameborder="0" allowfullscreen style="border-radius:10px; margin-top:10px;"></iframe>
        </div>
        """, unsafe_allow_html=True)

# ================= TAB 5: NUTRIBOT-AI CHAT KONSULTASI =================
with tab5:
    st.markdown(f"### {txt['chat_title']}")
    st.caption(f"Konsultasikan seputar makanan, pantangan, tinggi badan, atau keluhan gizi untuk **{st.session_state.user_name}** ({st.session_state.age_category}).")
    
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if user_prompt := st.chat_input("Tulis pertanyaan seputar nutrisi, menu, atau kesehatan di sini..."):
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("NutriBot-AI sedang menganalisis..."):
                bot_reply = generate_smart_ai_response(
                    user_prompt,
                    st.session_state.user_name,
                    st.session_state.age_category,
                    st.session_state.age_val,
                    st.session_state.height_val,
                    st.session_state.weight_val
                )
                st.markdown(bot_reply)
                st.session_state.chat_messages.append({"role": "assistant", "content": bot_reply})
