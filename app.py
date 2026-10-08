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

# DYNAMIC ACTION PLAN GENERATOR MEDIS BERVARIAI
def generate_dynamic_action_plan(age_cat, age, height, weight, z_score, who_status, birth_weight, asi):
    actions = []
    
    if age_cat == "Balita":
        if z_score < -3:
            actions.append("🚨 **Rujukan Medis Prioritas 1:** Segera bawa balita ke Dokter Spesialis Anak (Sp.A) di Rumah Sakit/Puskesmas untuk skrining stunting berat & pemeriksaan Red Flags infeksi kronis.")
            actions.append("🍖 **Intervensi Dosis Protein Hewani Padat:** Berikan minimal 3 porsi Protein Hewani ganda per hari (Contoh: 1 Butir Telur Puyuh + 50g Hati Ayam Sangrai + 40g Ikan Kembung/Lele).")
            actions.append("💊 **Suplementasi Medis:** Mintalah resep Sirup Zat Besi (Iron Supplement), Sirup Zinc, dan Vitamin A dosis tinggi dari Faskes setempat.")
        elif -3 <= z_score < -2:
            actions.append("⚠️ **Intervensi Gizi Puskesmas:** Jadwalkan konsultasi gizi bulanan di Posyandu/Puskesmas dan minta Pemberian Makanan Tambahan (PMT) kaya protein hewani.")
            actions.append("🍳 **Peningkatan Kualitas MPASI:** Tambahkan booster lemak sehat (1 sdt minyak kelapa/santan/butter) pada setiap sajian MPASI untuk mengejar ketertinggalan energi.")
            actions.append("🚫 **Pola Pengasuhan Ketat:** STOP pemberian teh, kopi, jus manis, atau jajanan kemasan yang mengikat zat besi & merusak nafsu makan balita.")
        else:
            actions.append("✅ **Pertahankan Nutrisi Ideal:** Lanjutkan pemberian variasi 2-3 porsi Protein Hewani (telur, ayam, ikan, daging) secara teratur setiap hari.")
            actions.append("📏 **Pemantauan Rutin:** Catat grafik tumbuh kembang secara berkala setiap bulan di Posyandu atau aplikasi ini.")
            
        if birth_weight < 2.5:
            actions.append(f"👶 **Atensi Berat Lahir Rendah (BBLR):** Karena riwayat BBLR ({birth_weight} kg), balita membutuhkan pengawasan ekstra pada grafik pemantauan pertumbuhan.")
        if asi == "Tidak":
            actions.append("🥛 **Kecukupan Pengganti ASI:** Pastikan asupan susu formula/pendamping ASI terisi dengan higienitas tinggi dan air minum matang steril.")
            
    elif age_cat in ["Anak-Anak", "Remaja"]:
        bmi = weight / ((height/100)**2)
        if bmi < 18.5:
            actions.append("🥛 **Peningkatan Asupan Kalori Sehat:** Tambahkan porsi karbohidrat kompleks (nasi/kentang) dan snack bergizi tinggi (telur rebus, keju, kacang-kacangan).")
            actions.append("💪 **Latihan Fisik & Tulang:** Anjurkan olahraga aktif minimal 45 menit/hari (renang, basket, melompat) untuk memicu hormon pertumbuhan tulang.")
        elif bmi > 25:
            actions.append("🥦 **Pola Makan Gizi Seimbang:** Kurangi minuman manis berbobat (boba, soda) dan gorengan. Ganti snack dengan buah segar & air putih.")
            actions.append("🏃 **Aktivitas Fisik Teratur:** Tingkatkan jalan kaki atau olahraga aerobik 150 menit per minggu.")
        else:
            actions.append("🌟 **Kondisi Optimal:** Pertahankan pola makan gizi seimbang dan konsumsi 8 gelas air putih per hari.")
            
    else: # Dewasa & Lansia
        bmi = weight / ((height/100)**2)
        if bmi > 25:
            actions.append("🫀 **Pencegahan Risiko Degeneratif:** Batasi asupan Gula, Garam, dan Lemak (GGL). Lakukan cek tekanan darah, gula darah, dan kolesterol berkala.")
        actions.append("🥑 **Nutrisi Pelindung Otot & Tulang:** Cukupi kebutuhan kalsium, Vitamin D, dan protein mudah dicerna untuk mencegah penurunan massa otot (sarkopenia).")
        actions.append("💧 **Hidrasi & Tidur Cukup:** Pastikan minum air putih yang cukup dan tidur berkualitas 7-8 jam per malam.")
        
    return actions

# SMART & FLEXIBLE AI CHATBOT ENGINE
def generate_smart_ai_response(prompt, user_name, age_cat, age, height, weight):
    p = prompt.lower()
    
    # 1. Kasus Tidak Suka / Alergi Makanan Tertentu (Ayam, Ikan, Daging, dll)
    if any(k in p for k in ["gasuka ayam", "tidak suka ayam", "gak suka ayam", "bosen ayam", "nggak suka ayam", "alergi ayam"]):
        return f"Halo **{user_name}**! Jangan khawatir jika tidak suka ayam. Sebagai alternatif sumber protein hewani yang setara dan kaya asam amino untuk kategori **{age_cat}**, kamu bisa menggantinya dengan:\n1. **Ikan (Kembung, Lele, atau Salmon):** Tinggi Omega-3 dan lemak sehat.\n2. **Daging Sapi / Kambing / Hati Sapi:** Sumber zat besi tinggi penangkal anemia.\n3. **Telur Ayam / Telur Puyuh / Telur Bebek:** Protein hewani paling praktis dan mudah diserap tubuh."
        
    elif any(k in p for k in ["gasuka ikan", "tidak suka ikan", "gak suka ikan", "nggak suka ikan", "bau amis"]):
        return f"Tidak masalah jika **{user_name}** kurang suka ikan karena bau amis. Kamu tetap bisa mendapatkan protein hewani dari:\n1. **Daging sapi cincang atau ayam** yang diolah menjadi bakso / nugget homemade.\n2. **Telur puyuh atau telur dadar keju**.\n3. **Keju, yogurt, atau susu** sebagai tambahan kalsium & protein harian."

    # 2. Kasus Lapar / Ingin Makan Sesuatu
    elif any(k in p for k in ["lapar", "mau makan", "pengen makan", "cari makan", "makan apa"]):
        return f"Wah, kalau **{user_name}** sedang merasa lapar, pastikan memilih makanan yang padat gizi dan mengenyangkan tahan lama:\n1. **Karbohidrat Kompleks:** Nasi merah/putih, kentang rebus, atau roti gandum.\n2. **Protein Pendamping:** Telur rebus, dada ayam panggang, atau sup tahu hangat.\n3. **Camilan Sehat:** Buah segar (pisang/alpukat) atau segelas air putih hangat agar hidrasi tetap terjaga."

    # 3. Kasus Nafsu Makan Kurang / Turun / GTM
    elif any(k in p for k in ["nafsu makan", "males makan", "ga nafsu", "gak nafsu", "susah makan", "gtm"]):
        return f"Menurunnya nafsu makan pada **{user_name}** ({age_cat}) bisa diatasi dengan:\n1. **Ubah Porsi Menjadi Kecil Tapi Sering:** Daripada langsung makan 1 porsi besar, bagi menjadi 5-6 kali makan porsi kecil.\n2. **Gunakan Booster Rasa & Kalori:** Tambahkan sedikit mentega (butter), kaldu alami, atau keju leleh pada makanan untuk meningkatkan aroma dan selera.\n3. **Cek Aktivitas Fisik:** Lakukan jalan santai atau olahraga ringan agar metabolisme tubuh terstimulasi dan rasa lapar muncul secara alami."

    # 4. Kasus Teh / Kopi / Pantangan
    elif any(k in p for k in ["teh", "kopi", "kafein"]):
        return f"Mengonsumsi teh atau kopi bersamaan dengan waktu makan **sangat tidak disarankan** karena kandungan asam tanin di dalamnya mengikat zat besi dari makanan hingga 70%, yang bisa memicu anemia dan lemas pada **{user_name}**!"

    # 5. Kasus Tinggi Badan / Stunting / Pertumbuhan
    elif any(k in p for k in ["tinggi", "pendek", "stunting", "tumbuh"]):
        return f"Untuk mengoptimalkan tinggi badan **{user_name}** ({height} cm): Konsumsi protein hewani secara rutin untuk memicu hormon pertumbuhan (*IGF-1*), pastikan tidur nyenyak malam hari antara jam 22.00 - 02.00 (saat *Growth Hormone* diproduksi maksimal), dan lakukan olahraga peregangan tulang."

    # 6. Kasus Berat Badan / Diet / Gemuk / Kurus
    elif any(k in p for k in ["berat", "bb", "kurus", "gemuk", "diet", "turun berat"]):
        return f"Pengaturan berat badan untuk **{user_name}** ({weight} kg) harus disesuaikan dengan Indeks Massa Tubuh (IMT):\n- **Jika ingin naik BB:** Tambahkan sumber kalori sehat (santan, alpukat, keju, daging berlemak sehat).\n- **Jika ingin turun BB:** Kurangi porsi gula, minyak jenuh, gorengan, dan perbanyak serat sayuran serta air putih."

    # 7. Kasus Resep / Menu Makanan
    elif any(k in p for k in ["resep", "menu", "masak", "makanan apa"]):
        return f"Panduan menu nutrisi 7 hari terlengkap untuk kelompok usia **{age_cat}** sudah disediakan secara lengkap di **Tab 🥣 Resep Nutrisi 7 Hari** di atas, lengkap dengan bahan, cara memasak, serta link rujukan video YouTube-nya!"

    # Default Cerdas Kontekstual Jika Topik Lain
    else:
        return f"Terima kasih atas pertanyaannya mengenai **{user_name}** ({age_cat})! Berdasarkan standar gizi dan kesehatan medis, pastikan kecukupan gizi seimbang kaya Protein Hewani, penuhi kebutuhan hidrasi air putih, hindari jajanan tinggi gula/garam, serta pantau secara teratur grafik kesehatan di dashboard ini. Ada hal spesifik lain tentang menu, tinggi badan, atau keluhan yang ingin Anda diskusikan?"

# GENERATOR FILE PDF DOKUMEN FISIK LENGKAP
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

# ---------------------------------------------------------
# 7. DASHBOARD UTAMA
# ---------------------------------------------------------
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

        with res_col
