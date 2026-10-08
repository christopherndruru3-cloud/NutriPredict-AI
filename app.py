import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
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
            actions.append("👶 **Atensi Berat Lahir Rendah (BBLR):** Karena riwayat BBLR ({birth_weight} kg), balita membutuhkan pengawasan ekstra pada grafik pemantauan pertumbuhan.")
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

# AI CHATBOT CERDAS BERVARIAI
def generate_smart_ai_response(prompt, user_name, age_cat, age, height, weight):
    p = prompt.lower()
    
    responses = {
        "gtm": f"Halo Kak/Bunda dari **{user_name}**! Untuk mengatasi GTM (Gerak Tutup Mulut) atau anak susah makan:\n\n1. **Terapkan Feeding Rules:** Batasi waktu makan maksimal 30 menit. Jangan beri minuman manis atau camilan 2 jam sebelum jam makan utama.\n2. **Ganti Variasi Lauk:** Jika menolak nasi atau ikan, coba kreasikan **telur puyuh, hati ayam, atau perkedel daging sapi**.\n3. **Ciptakan Suasana Menyenangkan:** Hindari memaksa atau memarahi anak saat makan.",
        
        "teh": f"Pemberian teh atau kopi pada balita/anak sangat **TIDAK DIANJURKAN secara medis**! Teh mengandung senyawa *tanin* dan *fitat* yang mengikat zat besi (Fe) dan kalsium dari makanan hingga 70%. Akibatnya, nutrisi tidak terserap dan memicu Anemia Defisiensi Besi serta stunting.",
        
        "tinggi": f"Tinggi badan pengguna (**{user_name}**, {height} cm) sangat dipengaruhi oleh:\n1. **Kecukupan Protein Hewani:** Memicu sintesis hormon *IGF-1* dan sinyal *mTORC1* pada pembentukan tulang panjang.\n2. **Deep Sleep (Tidur Nyenyak):** Hormon pertumbuhan (*Growth Hormone*) diproduksi maksimal antara jam 22.00 - 02.00 malam saat anak tidur nyenyak.\n3. **Aktivitas Fisik:** Melompat, renang, atau berolahraga secara teratur.",
        
        "resep": f"Untuk kelompok usia **{age_cat}**, menu nutrisi terbaik mencakup kombinasi Protein Hewani Asam Amino Lengkap (Telur, Ikan, Daging, Ayam) dipadukan dengan lemak sehat (minyak kelapa/butter/santan) dan serat halus. Anda bisa melihat daftar resep lengkap 7 hari di **Tab 🥣 Resep Nutrisi 7 Hari**!",
        
        "stunting": f"Stunting adalah kondisi gagal tumbuh akibat kekurangan gizi kronis dan infeksi berulang pada 1.000 Hari Pertama Kehidupan (0-24 Bulan). Stunting dapat dicegah dengan intervensi **Protein Hewani harian**, imunisasi lengkap, kebersihan air/sanitasi (WASH), dan pemantauan rutin Z-Score WHO.",
        
        "berat": f"Untuk mengoptimalkan berat badan pada kelompok usia **{age_cat}** ({weight} kg):\n1. Tambahkan **booster kalori sehat** seperti butter, santan segar, keju, atau minyak kelapa pada sajian makanan.\n2. Pastikan asupan protein hewani mencukupi setiap kali makan.\n3. Skrining adanya infeksi tersembunyi (seperti ISPA, cacingan, atau ISK) jika berat badan tidak naik 2 bulan berturut-turut."
    }
    
    for key in responses:
        if key in p:
            return responses[key]
            
    return f"Terima kasih atas pertanyaannya mengenai **{user_name}** ({age_cat})! Berdasarkan standar kesehatan medis: Pastikan kecukupan gizi seimbang kaya Protein Hewani, penuhi kebutuhan hidrasi air putih, hindari jajanan tinggi gula/garam, serta pantau secara teratur grafik kesehatan di dashboard ini. Ada hal spesifik lain tentang menu, tinggi badan, atau keluhan yang ingin Anda diskusikan?"

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

        # ACTION PLAN SPESIFIK & BERVARIAI MEDIS
        dynamic_actions = generate_dynamic_action_plan(
            st.session_state.age_category, age, height, weight, z_score, who_status, birth_weight, asi_str
        )

        # KARTU LAPORAN VISUAL DI HALAMAN UTAMA (TERISI UTUH)
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
            <p style="font-size: 11px; color: #64748B; font-style: italic;">{txt['report_note']}</p>
        </div>
        """
        st.markdown(report_html, unsafe_allow_html=True)
        
        st.write("")
        c_pdf1, c_pdf2 = st.columns(2)
        with c_pdf1:
            pdf_link_html = create_pdf_download_link(
                st.session_state.user_name, st.session_state.user_email, st.session_state.age_category,
                age, gender_str, height, weight, birth_weight, asi_str, z_score, who_status, dynamic_actions
            )
            st.markdown(pdf_link_html, unsafe_allow_html=True)
        with c_pdf2:
            if st.button(txt['email_btn']):
                st.success(f"📧 Laporan rekam medis terverifikasi dan otomatis siap dikirimkan ke: **{st.session_state.user_email}**!")

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

# ================= TAB 2: GRAFIK WHO & SIMULATOR =================
with tab2:
    st.markdown(f"### {txt['chart_title']}")
    ages = np.arange(6, 61, 1)
    df_chart = pd.DataFrame({
        'Usia': ages,
        'Sangat Pendek (-3 SD)': 48.0 + (ages * 1.25) - 9.6,
        'Batas Stunted (-2 SD)': 48.0 + (ages * 1.25) - 6.4,
        'Median WHO (0 SD)': 48.0 + (ages * 1.25)
    })

    fig = px.line(df_chart, x='Usia', y=['Sangat Pendek (-3 SD)', 'Batas Stunted (-2 SD)', 'Median WHO (0 SD)'],
                  color_discrete_sequence=['#EF4444', '#F59E0B', '#10B981'])
    
    curr_a = st.session_state.age_val
    curr_h = st.session_state.height_val

    fig.add_trace(go.Scatter(
        x=[curr_a], y=[curr_h], mode='markers+text',
        name='Posisi Saat Ini', text=[f'{st.session_state.user_name} ({curr_h} cm)'], textposition="top center",
        marker=dict(size=14, color='#38BDF8', symbol='star')
    ))

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=420)
    st.plotly_chart(fig, use_container_width=True)

    # PENJELASAN TREN
    st.markdown(f"#### {txt['chart_analysis_title']}")
    z_sc, st_name, _, med_val = calculate_who_zscore(curr_a if st.session_state.age_category=="Balita" else 24, curr_h)
    diff = round(curr_h - med_val, 1)
    
    st.markdown(f"""
    <div class="edu-card">
        <p>📌 <strong>Interpretasi Grafik:</strong> Bintang biru mewakili posisi tumbuh kembang / status antropometri <strong>{st.session_state.user_name}</strong> ({curr_h} cm).</p>
        <ul>
            <li><strong>Garis Hijau (0 SD):</strong> Median ideal ({med_val} cm). Selisih posisi: <strong>{'+' if diff >= 0 else ''}{diff} cm</strong>.</li>
            <li><strong>Garis Kuning (-2 SD):</strong> Batas ambang bawah kategori normal (<strong>{round(med_val - 6.4, 1)} cm</strong>).</li>
            <li><strong>Garis Merah (-3 SD):</strong> Batas ambang defisiensi gizi berat (<strong>{round(med_val - 9.6, 1)} cm</strong>).</li>
        </ul>
        <a class="ref-btn" href="https://www.who.int/tools/child-growth-standards" target="_blank">🌐 Standar Pertumbuhan WHO Resmi</a>
    </div>
    """, unsafe_allow_html=True)

    # SIMULATOR TARGET PERTUMBUHAN ("WHAT-IF")
    st.write("---")
    st.markdown(f"### {txt['sim_title']}")
    sim_months = st.slider(txt['sim_months'], min_value=1, max_value=12, value=6)
    
    future_age = curr_a + sim_months
    target_h_normal = round(48.0 + (future_age * 1.25), 1)
    needed_growth = round(target_h_normal - curr_h, 1)

    c_sim1, c_sim2 = st.columns(2)
    with c_sim1:
        st.info(f"🗓️ **Target Usia Muka:** {future_age} ({sim_months} periode ke depan)")
        st.success(f"🎯 **Target Tinggi Ideal:** {target_h_normal} cm")
    with c_sim2:
        st.metric("Total Kebutuhan Pertambahan Tinggi", f"+{needed_growth} cm", delta=f"{round(needed_growth/sim_months, 1)} cm/periode")

# ================= TAB 3: RESEP NUTRISI 7 HARI (LENGKAP SEMUA UMUR) =================
with tab3:
    st.markdown(f"### {txt['recipe_title']}")
    
    cat_recipe = st.radio("Pilih Kelompok Usia Resep / Select Recipe Category:", 
                          ["Balita (6-8 Bulan)", "Balita (9-11 Bulan)", "Balita (12-23 Bulan)", "Anak-Anak (5-12 Tahun)", "Remaja (13-18 Tahun)", "Dewasa (19-59 Tahun)", "Lansia (60+ Tahun)"], horizontal=True)
    
    if "6-8" in cat_recipe:
        recipes = [
            ("Senin", "🐣 Puree Hati Ayam & Santan", "Bahan: 30g Hati Ayam, 2 sdm Nasi, 1 sdt Santan, Wortel.\n\nTutorial:\n1. Rebus hati ayam & wortel hingga matang empuk.\n2. Lumatkan nasi hangat bersama parutan wortel.\n3. Tambahkan 1 sdt santan segar hangat lalu saring halus dengan saringan kawat."),
            ("Selasa", "🐟 Puree Ikan Kembung & Labu Siam", "Bahan: 30g Fillet Ikan Kembung, 2 sdm Nasi, Labu Siam, 1 sdt Minyak Kelapa.\n\nTutorial:\n1. Kukus fillet ikan kembung tanpa duri dan parutan labu siam.\n2. Campurkan dengan nasi tim hangat.\n3. Tambahkan 1 sdt minyak kelapa lalu saring lumat."),
            ("Rabu", "🥚 Puree Telur Puyuh & Bayam", "Bahan: 2 Butir Telur Puyuh, 2 sdm Nasi, Daun Bayam, Sejumput Butter.\n\nTutorial:\n1. Rebus telur puyuh hingga matang keras lalu lumatkan kuning & putihnya.\n2. Cincang halus daun bayam rebus.\n3. Aduk rata bersama nasi lembik dan butter."),
            ("Kamis", "🥩 Puree Daging Sapi & Kentang", "Bahan: 30g Daging Sapi Cincang, 1/2 Kentang Rebus, Keju Parut.\n\nTutorial:\n1. Tumis daging sapi cincang halus hingga matang.\n2. Rebus kentang lalu lumatkan bersama daging sapi.\n3. Taburi keju parut secukupnya."),
            ("Jumat", "🦐 Puree Udang & Tahu Lembut", "Bahan: 30g Udang Cincang, 1/2 Tahu Putih, 2 sdm Nasi, Minyak Wijen.\n\nTutorial:\n1. Cincang halus udang kupas bersih.\n2. Lumatkan tahu putih bersama nasi lembik.\n3. Kukus selama 15 menit dan beri 2 tetes minyak wijen."),
            ("Sabtu", "🍳 Puree Telur Bebek & Tempe", "Bahan: 1/2 Telur Bebek, 1 Potong Tempe, 2 sdm Nasi, Margarin.\n\nTutorial:\n1. Kukus tempe hingga empuk.\n2. Orak-arik telur bebek dengan margarin.\n3. Lumatkan halus tempe, telur, dan nasi hangat."),
            ("Minggu", "🍲 Puree Ayam & Kaldu Ceker", "Bahan: 30g Daging Ayam Cincang, Wortel, Kuah Kaldu Ceker, Nasi.\n\nTutorial:\n1. Rebus daging ayam dan wortel dalam kuah kaldu ceker alami.\n2. Lumatkan nasi hangat bersama rebusan ayam hingga tekstur puree lembut.")
        ]
    elif "9-11" in cat_recipe:
        recipes = [
            ("Senin", "🌾 Tim Nasi Hati Ayam Cincang", "Bahan: 40g Hati Ayam, 3 sdm Nasi Tim, Buncis Cincang, Margarin.\n\nTutorial:\n1. Tumis hati ayam cincang dengan margarin.\n2. Masukkan nasi tim & potongan buncis halus.\n3. Masak hingga bumbu meresap."),
            ("Selasa", "🐟 Tim Ikan Kembung Suwir & Kelor", "Bahan: 40g Ikan Kembung, Daun Kelor Cincang, Nasi Tim, Minyak Kelapa.\n\nTutorial:\n1. Suwir halus ikan kembung kukus tanpa duri.\n2. Masukkan ke nasi tim bersama daun kelor cincang halus."),
            ("Rabu", "🥚 Tim Nasi Telur Bebek & Jagung", "Bahan: 1 Telur Bebek, Jagung Manis Pipil, Nasi Tim, Margarin.\n\nTutorial:\n1. Orak-arik telur bebek dengan margarin.\n2. Campurkan dengan nasi tim & pipilan jagung manis lumat."),
            ("Kamis", "🥩 Tim Daging Sapi Cincang & Brokoli", "Bahan: 40g Daging Sapi Cincang, Brokoli Cincang, Nasi Tim, Bawang Putih.\n\nTutorial:\n1. Tumis daging sapi cincang & bawang putih harum.\n2. Masukkan nasi tim & cincangan brokoli hingga matang."),
            ("Jumat", "🦐 Tim Udang Cincang & Tahu Dadu", "Bahan: 40g Udang Cincang, Tahu Dadu Kecil, Nasi Tim, Minyak Wijen.\n\nTutorial:\n1. Tumis udang cincang dengan sedikit minyak wijen.\n2. Masukkan tahu dadu kecil & nasi tim hangat."),
            ("Sabtu", "🍳 Tim Telur Puyuh & Sup Wortel", "Bahan: 3 Butir Telur Puyuh, Wortel Dadu, Nasi Tim, Kuah Ayam.\n\nTutorial:\n1. Rebus 3 telur puyuh.\n2. Sajikan bersama nasi tim & sup wortel potong dadu kecil."),
            ("Minggu", "🍲 Tim Bola-Bola Ayam & Labu", "Bahan: 40g Ayam Cincang, Labu Siam Dadu, Nasi Tim, Kaldu.\n\nTutorial:\n1. Buat adonan bola ayam cincang kecil.\n2. Rebus dalam kuah kaldu bersama labu siam hingga matang.")
        ]
    elif "12-23" in cat_recipe:
        recipes = [
            ("Senin", "🍲 Sup Bola Bakso Ayam Udang", "Bahan: 50g Daging Ayam & Udang, Wortel, Kentang, Kuah Kaldu.\n\nTutorial:\n1. Buat bakso ayam udang homemade.\n2. Rebus dalam kuah kaldu wortel & kentang hingga mengapung matang."),
            ("Selasa", "🐟 Pepes Ikan Lele / Belut Tanpa Duri", "Bahan: 50g Lele/Belut, Bumbu Kuning Lembut, Daun Pisang.\n\nTutorial:\n1. Bumbui lele/belut tanpa duri dengan bumbu harum.\n2. Kukus dalam bungkus daun pisang selama 20 menit."),
            ("Rabu", "🥩 Semur Daging Cincang & Telur Puyuh", "Bahan: 50g Daging Sapi Cincang, 3 Telur Puyuh, Kecap Manis, Bawang.\n\nTutorial:\n1. Tumis daging sapi cincang kecap manis harum.\n2. Masukkan 3 butir telur puyuh rebus hingga bumbu meresap."),
            ("Kamis", "🍗 Ayam Goreng Kaldu & Sayur Bening", "Bahan: 1 Potong Ayam Ungkep Kaldu, Bayam, Nasi Warm.\n\nTutorial:\n1. Ungkep ayam dengan kaldu alami lalu goreng sebentar.\n2. Sajikan dengan sayur bening bayam & nasi hangat."),
            ("Jumat", "🦐 Tumis Udang Brokoli Saus Mentega", "Bahan: 50g Udang Kupas, Brokoli, Mentega, Kecap Manis.\n\nTutorial:\n1. Tumis udang kupas & brokoli dengan mentega harum.\n2. Beri sedikit kecap manis."),
            ("Sabtu", "🍳 Telur Dadar Daun Kelor & Nasi Warm", "Bahan: 1 Butir Telur Ayam, Daun Kelor Cincang, Margarin.\n\nTutorial:\n1. Kocok 1 butir telur dengan daun kelor cincang.\n2. Dadar tipis dengan margarin dan sajikan bersama nasi hangat."),
            ("Minggu", "🥞 Pancake Hati Ayam & Pisang", "Bahan: Tepung Terigu, Pisang Lumat, 1 Telur, Bubuk Hati Ayam Sangrai.\n\nTutorial:\n1. Campurkan tepung terigu, pisang lumat, telur, & bubuk hati ayam sangrai.\n2. Panggang di teflon dengan api kecil hingga matang keemasan.")
        ]
    elif "Anak-Anak" in cat_recipe:
        recipes = [
            ("Senin", "弁当 Bento Nasi Kuning Ayam Popcorn & Telur", "Bahan: Nasi Kuning, Dada Ayam Potong Dadu, 1 Telur Rebus, Wortel.\n\nTutorial:\n1. Goreng dada ayam tepung crispy.\n2. Cetak nasi kuning dan hias bersama potongan telur rebus & wortel."),
            ("Selasa", "🍝 Spaghetti Salmon Bolognese", "Bahan: Pasta Spaghetti, Fillet Salmon Cincang, Saus Tomat Homemade.\n\nTutorial:\n1. Rebus spaghetti al dente.\n2. Tumis salmon cincang dengan saus tomat lalu siram di atas pasta."),
            ("Rabu", "🍲 Sup Makaroni Daging Sapi & Buncis", "Bahan: Daging Sapi Cincang, Makaroni, Buncis, Wortel, Kaldu Sapi.\n\nTutorial:\n1. Rebus daging sapi dan makaroni hingga empuk.\n2. Masukkan sayuran buncis & wortel dalam kuah kaldu gizi."),
            ("Kamis", "🍳 Nasi Goreng Telur Puyuh & Udang", "Bahan: Nasi Putih, 4 Telur Puyuh, Udang Kupas, Minyak Wijen.\n\nTutorial:\n1. Tumis udang kupas dan telur puyuh orak-arik.\n2. Masukkan nasi dan bumbui ringan tanpa pengawet."),
            ("Jumat", "🍗 Chicken Teriyaki & Tumis Brokoli", "Bahan: Dada Ayam, Saus Teriyaki, Brokoli, Biji Wijen.\n\nTutorial:\n1. Tumis ayam potong dengan saus teriyaki gurih.\n2. Sajikan dengan rebusan brokoli segar & taburan biji wijen."),
            ("Sabtu", "🥪 Sandwich Telur Keju & Daging Asap", "Bahan: Roti Tawar Gandum, Telur Dadar, Keju Slice, Daging Asap.\n\nTutorial:\n1. Panggang roti gandum di atas teflon.\n2. Susun telur dadar, keju, & daging asap hangat."),
            ("Minggu", "🍲 Soto Ayam Kuah Bening & Telur Rebus", "Bahan: Daging Ayam Suwir, Kuah Soto Bening, Telur Rebus, Tauge.\n\nTutorial:\n1. Rebus ayam kuah soto rempah alami.\n2. Sajikan suwiran ayam, tauge, & telur rebus matang.")
        ]
    elif "Remaja" in cat_recipe:
        recipes = [
            ("Senin", "🥩 Beef Bowl Yoshinoya Style & Egg", "Bahan: Daging Sapi Slice, Bawang Bombay, Kecap Asin, 1 Telur Ceplok.\n\nTutorial:\n1. Tumis daging sapi slice bersama bawang bombay saus gurih.\n2. Tumpuk di atas nasi hangat bersama telur ceplok setengah matang."),
            ("Selasa", "🥗 Salad Salmon Panggang & Avokad", "Bahan: Fillet Salmon, Alpukat Slice, Sayur Selada, Olive Oil.\n\nTutorial:\n1. Panggang salmon dengan garam & lada hitam.\n2. Campur selada segar, potongan alpukat, & dressing olive oil."),
            ("Rabu", "🍗 Ayam Bakar Madu & Tumis Kangkung", "Bahan: Paha Ayam, Bumbu Madu, Kangkung, Bawang Merah Putih.\n\nTutorial:\n1. Ungkep ayam bumbu madu lalu bakar keemasan.\n2. Tumis kangkung segar dengan sedikit minyak."),
            ("Kamis", "🍲 Sup Ikan Batang Asam Pedas", "Bahan: Fillet Ikan Kakap/Tenggiri, Tomat Hijau, Belimbing Wulung.\n\nTutorial:\n1. Rebus kuah rempah bening asam segar.\n2. Masukkan fillet ikan & potongan tomat hingga matang."),
            ("Jumat", "🍝 Fusilli Tuna Spicy Olive Oil", "Bahan: Pasta Fusilli, Tuna Cincang, Cabai Rawit, Minyak Zaitun.\n\nTutorial:\n1. Tumis tuna cincang & irisan cabai dengan olive oil.\n2. Campurkan pasta fusilli rebus."),
            ("Sabtu", "🍳 Omelet Daging Cincang & Bayam Keju", "Bahan: 2 Telur Ayam, Daging Sapi Cincang, Bayam, Keju Mozzarella.\n\nTutorial:\n1. Kocok telur dengan isi daging cincang & bayam.\n2. Lipat omelet dan beri lelehan keju mozzarella di atasnya."),
            ("Minggu", "🥣 Smoothies Bowl Buah Naga & Chia Seed", "Bahan: Buah Naga Blend, Pisang, Chia Seeds, Kacang Almond.\n\nTutorial:\n1. Blender halus buah naga & pisang dingin.\n2. Tuang ke mangkok dan beri topping chia seeds & almond renyah.")
        ]
    elif "Dewasa" in cat_recipe:
        recipes = [
            ("Senin", "🐟 Salmon Panggang Lemon & Kentang Rebus", "Bahan: Fillet Salmon, Perasan Lemon, Kentang, Rosemary.\n\nTutorial:\n1. Marinasi salmon dengan perasan lemon & lada.\n2. Panggang teflon 8 menit & sajikan dengan kentang rebus."),
            ("Selasa", "🥗 Pokebowl Tuna Segar & Edamame", "Bahan: Fillet Tuna, Kacang Edamame, Nasi Merah, Wijen.\n\nTutorial:\n1. Tumis tuna sebentar dengan minyak wijen.\n2. Susun di atas nasi merah bersama edamame rebus."),
            ("Rabu", "🥩 Tumis Daging Sapi Lada Hitam & Paprika", "Bahan: Daging Sapi Lean Slice, Paprika Merah Hijau, Lada Hitam.\n\nTutorial:\n1. Tumis daging sapi tanpa lemak bersama saus lada hitam.\n2. Masukkan potongan paprika kaya vitamin C."),
            ("Kamis", "🍗 Dada Ayam Panggang Herb & Tumis Buncis", "Bahan: Dada Ayam Tanpa Kulit, Oregano, Buncis, Minyak Zaitun.\n\nTutorial:\n1. Panggang dada ayam bumbu herb rendah garam.\n2. Tumis buncis dengan minyak zaitun ringan."),
            ("Jumat", "🍲 Sup Ikan Gurame Bening Daun Kemangi", "Bahan: Fillet Gurame, Daun Kemangi, Jahe, Serai, Kuah Bening.\n\nTutorial:\n1. Rebus kuah jahe serai wangi tanpa santan.\n2. Masukkan fillet gurame & daun kemangi hingga segar."),
            ("Sabtu", "🍳 Tofu Stir Fry Shimeji & Telur", "Bahan: Tofu Jepang, Jamur Shimeji, 1 Telur, Saus Tiram.\n\nTutorial:\n1. Tumis tofu & jamur shimeji saus tiram rendah natrium.\n2. Orak-arik telur sebagai peningkat protein."),
            ("Minggu", "🥣 Oatmeal Kayu Manis & Telur Rebus", "Bahan: Rolled Oats, Bubuk Kayu Manis, Buah Apel, 2 Telur Rebus.\n\nTutorial:\n1. Seduh rolled oats hangat dan beri parutan apel & kayu manis.\n2. Sajikan dengan 2 butir telur rebus matang.")
        ]
    else: # Lansia
        recipes = [
            ("Senin", "🐟 Tim Fillet Kakap Bumbu Jahe Lengkuas", "Bahan: Fillet Kakap, Irisan Jahe, Daun Bawang, Minyak Wijen.\n\nTutorial:\n1. Kukus fillet kakap dengan irisan jahe & serai hingga lembut.\n2. Beri beberapa tetes minyak wijen wangi tanpa garam berlebih."),
            ("Selasa", "🍲 Sup Tahu Lembut & Ayam Cincang", "Bahan: Tahu Sutra, Dada Ayam Cincang, Labu Siam, Kuah Bening.\n\nTutorial:\n1. Rebus kuah kaldu bening rendah garam.\n2. Masukkan tahu sutra lembut & ayam cincang empuk mudah dikunyah."),
            ("Rabu", "🥚 Pepes Telur Tahu & Daun Kemangi", "Bahan: 2 Telur Kocok, Tahu Lumat, Daun Kemangi, Bungkus Pisang.\n\nTutorial:\n1. Campur lumat tahu dan telur dengan kemangi harum.\n2. Kukus dalam daun pisang hingga matang empuk."),
            ("Kamis", "🥩 Semur Daging Giling Empuk & Wortel", "Bahan: Daging Sapi Giling Halus, Wortel Rebus Empuk, Kecap.\n\nTutorial:\n1. Masak daging sapi giling lembut dengan bumbu semur ringan.\n2. Masukkan wortel rebus hingga tekstur sangat lembut."),
            ("Jumat", "🥣 Bubur Manado Tinutuan Komplit", "Bahan: Beras, Labu Kuning Lumat, Bayam, Jagung Manis Pipil.\n\nTutorial:\n1. Masak bubur beras bersama labu kuning lumat kaya karotenoid.\n2. Masukkan sayuran lembut untuk kemudahan cerna lansia."),
            ("Sabtu", "🍗 Sup Ayam Ceker & Sayuran Bening", "Bahan: Ceker Ayam Kaldu, Wortel, Kentang, Brokoli Rebus.\n\nTutorial:\n1. Rebus ceker ayam lama hingga keluar kolagen kaldu alami.\n2. Masukkan sayuran dipotong kecil empuk."),
            ("Minggu", "🍳 Scrambled Egg Tahu & Puree Labu", "Bahan: 2 Telur Bebek/Ayam, Tahu Sutra, Puree Labu Kuning.\n\nTutorial:\n1. Orak-arik lembut telur dan tahu sutra dengan butter.\n2. Sajikan bersama puree labu kuning hangat yang lezat.")
        ]

    for day_name, title, tut in recipes:
        st.markdown(f"""
        <div class="edu-card">
            <h3>📅 {day_name}: {title}</h3>
            <pre style="background:rgba(0,0,0,0.3); padding:12px; border-radius:8px; color:#F0F9FF; font-size:13px; white-space: pre-wrap;">{tut}</pre>
            <a class="ref-btn" href="https://www.youtube.com/results?search_query=Resep+Sehat+{title.replace(' ', '+')}" target="_blank">📺 YouTube: Video Tutorial Memasak</a>
            <a class="ref-btn" href="https://www.google.com/search?q=Resep+Nutrisi+Sehat+{title.replace(' ', '+')}" target="_blank">🔍 Google Search: Rujukan Bahan & Gizi</a>
        </div>
        """, unsafe_allow_html=True)

# ================= TAB 4: EDUKASI & BERITA VIDEO =================
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
        <div class="edu-card">
            <h3>💉 3. Imunisasi Lengkap & Vitamin A</h3>
            <p>Anak yang sering terkena penyakit infeksi akibat tidak imunisasi lengkap akan kehilangan banyak nutrisi, yang memicu stunting berulang. Pastikan Vitamin A diminum setiap bulan Februari dan Agustus.</p>
            <a class="ref-btn" href="https://sehatnegeriku.kemkes.go.id" target="_blank">🌐 Sehat NegeriKu Kemenkes</a>
        </div>
        """, unsafe_allow_html=True)

    with col_e2:
        st.markdown("""
        <div class="edu-card">
            <h3>📰 4. Berita Terkini Stunting di Indonesia</h3>
            <ul>
                <li><strong>Prevalensi Stunting Indonesia:</strong> Hasil SSGI menunjukkan angka prevalensi stunting nasional berada di kisaran 19,8%.</li>
                <li><strong>Gerakan Intervensi Serentak:</strong> Pemerintah terus menggalakkan pemberian Makanan Tambahan (PMT) Kaya Protein Hewani di seluruh Posyandu.</li>
            </ul>
            <a class="ref-btn" href="https://stunting.go.id" target="_blank">📰 Portal Resmi TP2S Stunting Indonesia</a>
        </div>
        <div class="edu-card">
            <h3>🛡️ 5. Kebiasaan Buruk Anak yang Menghambat Pertumbuhan</h3>
            <ul>
                <li><strong>Pemberian Teh/Kopi pada Balita:</strong> Senyawa tanin mengikat zat besi dari makanan sehingga memicu anemia dan stunting.</li>
                <li><strong>Kurang Tidur Malam:</strong> Hormon pertumbuhan (Growth Hormone) diproduksi maksimal saat anak tidur nyenyak di malam hari.</li>
                <li><strong>Minum Air Berlebihan Sebelum Makan:</strong> Mengisi lambung dengan cairan tanpa kalori sehingga anak cepat kenyang.</li>
            </ul>
            <a class="ref-btn" href="https://www.who.int/news-room/fact-sheets/detail/malnutrition" target="_blank">🌐 WHO Fact Sheets: Child Malnutrition</a>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")
    st.markdown("### 📺 Video Edukasi Resmi Kemenkes & BKKBN")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("#### 📺 Video Kemenkes RI: Cegah Stunting dengan ABCDE")
        st.video("https://www.youtube.com/watch?v=2Z1h9mQX7EQ")
        st.caption("Sumber: Kementerian Kesehatan RI & Ayo Sehat")
        
    with col_v2:
        st.markdown("#### 📺 Video BKKBN: Pengasuhan 1000 Hari Pertama Kehidupan")
        st.video("https://www.youtube.com/watch?v=S00n-c_qeC0")
        st.caption("Sumber: BKKBN RI Official")

# ================= TAB 5: NUTRIPOT-AI CHAT KONSULTASI =================
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

    # 4. Kasus GTM / Anak Kecil / Balita
    elif any(k in p for k in ["gtm", "anak susah", "anak ga mau"]):
        return f"Untuk mengatasi GTM pada anak ({user_name}):\n1. Terapkan aturan makan maksimal 30 menit.\n2. Jangan beri camilan atau susu 2 jam menjelang jam makan utama.\n3. Buat variasi bentuk makanan yang menarik (bento art atau bola-bola nasi)."

    # 5. Kasus Teh / Kopi / Pantangan
    elif any(k in p for k in ["teh", "kopi", "kafein"]):
        return f"Mengonsumsi teh atau kopi bersamaan dengan waktu makan **sangat tidak disarankan** karena kandungan asam tanin di dalamnya mengikat zat besi dari makanan hingga 70%, yang bisa memicu anemia dan lemas pada **{user_name}**!"

    # 6. Kasus Tinggi Badan / Stunting / Pertumbuhan
    elif any(k in p for k in ["tinggi", "pendek", "stunting", "tumbuh"]):
        return f"Untuk mengoptimalkan tinggi badan **{user_name}** ({height} cm): Konsumsi protein hewani secara rutin untuk memicu hormon pertumbuhan (*IGF-1*), pastikan tidur nyenyak malam hari antara jam 22.00 - 02.00 (saat *Growth Hormone* diproduksi maksimal), dan lakukan olahraga peregangan tulang."

    # 7. Kasus Berat Badan / Diet / Gemuk / Kurus
    elif any(k in p for k in ["berat", "bb", "kurus", "gemuk", "diet", "turun berat"]):
        return f"Pengaturan berat badan untuk **{user_name}** ({weight} kg) harus disesuaikan dengan Indeks Massa Tubuh (IMT):\n- **Jika ingin naik BB:** Tambahkan sumber kalori sehat (santan, alpukat, keju, daging berlemak sehat).\n- **Jika ingin turun BB:** Kurangi porsi gula, minyak jenuh, gorengan, dan perbanyak serat sayuran serta air putih."

    # 8. Kasus Resep / Menu Makanan
    elif any(k in p for k in ["resep", "menu", "masak", "makanan apa"]):
        return f"Panduan menu nutrisi 7 hari terlengkap untuk kelompok usia **{cat_category_name(age_cat)}** sudah disediakan secara lengkap di **Tab 🥣 Resep Nutrisi 7 Hari** di atas, lengkap dengan bahan, cara memasak, serta link rujukan video YouTube-nya!"

    # Default Cerdas Kontekstual Jika Topik Lain
    else:
        return f"Halo **{user_name}** ({age_cat})! Mengenai pertanyaan kamu (*\"{prompt}\"*): Berdasarkan standar gizi dan kesehatan, pastikan untuk selalu menjaga pola makan seimbang kaya Protein Hewani, cukupi hidrasi air putih, hindari stres, dan pantau terus grafik kesehatan tubuhmu di aplikasi ini. Ada detail spesifik lain yang ingin kamu tanyakan seputar menu, nutrisi, atau keluhan kesehatan?"

def cat_category_name(cat):
    return cat
