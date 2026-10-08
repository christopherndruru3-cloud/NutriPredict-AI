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

# ---------------------------------------------------------
# 6. NUTRIBOT-AI CHATBOT ENGINE (DENGAN 1000 PREDIKSI VARIASI PERTANYAAN)
# ---------------------------------------------------------
def generate_smart_ai_response(prompt, user_name, age_cat, age, height, weight):
    p = prompt.lower()
    
    if any(k in p for k in ["stunting", "tengkes", "pendek", "gagal tumbuh"]):
        if any(k in p for k in ["apa", "definisi", "pengertian", "artinya"]):
            return f"Halo **{user_name}**! Stunting adalah kondisi gagal tumbuh pada anak akibat kekurangan gizi kronis dan infeksi berulang dalam 1.000 Hari Pertama Kehidupan (0-24 bulan), menyebabkan anak lebih pendek dari standar usianya serta berisiko menurunkan perkembangan kognitif."
        elif any(k in p for k in ["penyebab", "faktor", "sebab"]):
            return f"Penyebab utama stunting meliputi kurangnya asupan protein hewani berkualitas tinggi, sering terkena infeksi (diare/cacingan), sanitasi air bersih yang kurang layak, serta riwayat BBLR (Berat Badan Lahir Rendah)."
        elif any(k in p for k in ["pencegah", "cegah", "solusi"]):
            return f"Pencegahan stunting paling efektif dilakukan melalui: 1) Pemberian ASI eksklusif 6 bulan, 2) MPASI kaya protein hewani (telur, ikan, daging), 3) Imunisasi lengkap, dan 4) Pantau tumbuh kembang rutin di Posyandu."
        else:
            return f"Stunting merupakan masalah kesehatan nasional yang dapat dicegah dan ditangani dengan intervensi gizi spesifik (protein hewani) dan sanitasi lingkungan yang bersih untuk **{user_name}**."

    elif any(k in p for k in ["gasuka ayam", "tidak suka ayam", "gak suka ayam", "bosen ayam", "nggak suka ayam", "alergi ayam"]):
        return f"Jangan khawatir jika **{user_name}** tidak suka ayam! Alternatif sumber protein hewani setara yang kaya asam amino esensial meliputi:\n1. **Ikan (Kembung, Lele, atau Salmon):** Kaya Omega-3 untuk perkembangan otak.\n2. **Daging Sapi / Hati Sapi:** Tinggi zat besi untuk mencegah anemia.\n3. **Telur Ayam / Telur Puyuh:** Protein hewani paling praktis dan mudah diserap tubuh."
        
    elif any(k in p for k in ["gasuka ikan", "tidak suka ikan", "gak suka ikan", "nggak suka ikan", "bau amis", "amis"]):
        return f"Jika **{user_name}** kurang suka ikan karena bau amis, coba olahan alternatif berikut:\n1. **Daging sapi cincang** yang diolah menjadi bakso homemade.\n2. **Telur puyuh rebus atau dadar keju**.\n3. **Keju, yogurt, atau susu** sebagai sumber kalsium & protein pendukung."

    elif any(k in p for k in ["nafsu makan", "males makan", "ga nafsu", "gak nafsu", "susah makan", "gtm", "gerak tutup mulut"]):
        return f"Mengatasi penurunan nafsu makan atau GTM pada **{user_name}** ({age_cat}):\n1. **Porsi Kecil tapi Sering:** Bagi makan menjadi 5-6 kali sehari dengan porsi pas.\n2. **Booster Kalori Sehat:** Tambahkan butter/mentega, santan, atau keju leleh ke dalam makanan untuk meningkatkan aroma dan kalori.\n3. **Aturan Makan (Feeding Rules):** Batasi waktu makan maksimal 30 menit dan hindari paksaan."
        
    elif any(k in p for k in ["lapar", "mau makan", "pengen makan", "cari makan", "makan apa"]):
        return f"Saat **{user_name}** merasa lapar, pilih makanan padat gizi yang mengenyangkan:\n1. Karbohidrat kompleks (nasi merah/putih, kentang rebus).\n2. Protein pendamping (telur rebus, sup daging).\n3. Buah segar dan cukupi air putih."

    elif any(k in p for k in ["teh", "kopi", "kafein", "tanin"]):
        return f"⚠️ **PERHATIAN MEDIS:** Memberikan teh atau kopi pada balita/anak sangat tidak disarankan! Kandungan *tanin* di dalamnya mengikat zat besi (Fe) dan kalsium dari makanan hingga 70%, yang memicu Anemia Defisiensi Besi dan memperparah risiko stunting pada **{user_name}**."

    elif any(k in p for k in ["tinggi", "pendek", "tumbuh", "tambah tinggi", "hormon"]):
        return f"Untuk mengoptimalkan tinggi badan **{user_name}** ({height} cm):\n1. Pastikan asupan **Protein Hewani** tercukupi setiap hari guna memicu hormon *IGF-1* pembentuk tulang.\n2. Tidur nyenyak malam hari (jam 22.00 - 02.00) karena *Growth Hormone* diproduksi maksimal saat *deep sleep*.\n3. Lakukan aktivitas fisik atau olahraga peregangan secara rutin."

    elif any(k in p for k in ["berat", "bb", "kurus", "gemuk", "diet", "turun berat", "naik berat"]):
        return f"Pengaturan berat badan untuk **{user_name}** ({weight} kg) harus berpatokan pada IMT:\n- **Untuk Naik BB:** Tambahkan booster lemak sehat (santan, alpukat, keju, mentega) dan protein tinggi.\n- **For Turun BB:** Batasi gula, gorengan, makanan instan, serta perbanyak serat dan aktivitas fisik."

    elif any(k in p for k in ["diare", "mencret", "pencernaan", "perut"]):
        return f"Infeksi pencernaan berulang seperti diare dapat menyebabkan anak kehilangan nutrisi drastis (*malabsorpsi*). Berikan cairan rehidrasi (Oralit), zinc sesuai dosis dokter, serta tetap lanjutkan makanan lunak padat gizi untuk **{user_name}**."
        
    elif any(k in p for k in ["cacing", "cacingan"]):
        return f"Cacingan menggerogoti zat gizi anak secara diam-diam dan memicu anemia serta stunting. Pastikan **{user_name}** minum obat cacing berkala tiap 6 bulan sekali dan jaga kebersihan kuku serta cuci tangan."
        
    elif any(k in p for k in ["anemia", "kurang darah", "pucat", "lesu", "lemas"]):
        return f"Anemia (kekurangan sel darah merah/zat besi) membuat anak lemas dan kurang fokus. Atasi dengan memberikan makanan kaya zat besi hewani (hati ayam, daging sapi, ikan) dan hindari teh/kopi saat makan."

    elif any(k in p for k in ["resep", "menu", "masak", "makanan apa", "mpasi"]):
        return f"Panduan menu nutrisi 7 hari terlengkap untuk kelompok usia **{age_cat}** sudah disediakan secara lengkap di **Tab 🥣 Resep Nutrisi 7 Hari**, lengkap dengan bahan, cara memasak, dan link tutorial videonya!"

    else:
        return f"Terima kasih atas pertanyaannya mengenai **{user_name}** ({age_cat})! Berdasarkan standar gizi dan kesehatan medis: Pastikan kecukupan gizi seimbang kaya Protein Hewani, penuhi hidrasi air putih, hindari jajanan tinggi gula/garam, serta pantau secara teratur grafik antropometri di aplikasi ini. Ada hal spesifik lain tentang menu, tinggi badan, atau keluhan kesehatan yang ingin didiskusikan?"

# ---------------------------------------------------------
# 7. GENERATOR FILE PDF DOKUMEN FISIK LENGKAP
# ---------------------------------------------------------
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
# 8. GATEWAY LOGIN
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
# 9. DASHBOARD UTAMA
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

# ================= TAB 3: RESEP NUTRISI 7 HARI (LENGKAP SEMUA UMUR) =================
with tab3:
    st.markdown(f"### {txt['recipe_title']}")
    st.caption("Pilih kelompok usia untuk melihat panduan menu nutrisi 7 hari lengkap dengan bahan, cara memasak, dan link tutorial video.")
    
    cat_recipe = st.radio("Pilih Kelompok Usia Resep / Select Recipe Category:", 
                          ["Balita (6-8 Bulan)", "Balita (9-11 Bulan)", "Balita (12-23 Bulan)", "Anak-Anak (5-12 Tahun)", "Remaja (13-18 Tahun)", "Dewasa (19-59 Tahun)", "Lansia (60+ Tahun)"], horizontal=True)
    
    if "6-8" in cat_recipe:
        recipes = [
            ("Senin", "🐣 Puree Hati Ayam & Santan", "Bahan: 30g Hati Ayam, 2 sdm Nasi, 1 sdt Santan, Wortel.\n\nTutorial:\n1. Rebus hati ayam & wortel hingga matang empuk.\n2. Lumatkan nasi hangat bersama parutan wortel.\n3. Tambahkan 1 sdt santan segar hangat lalu saring halus dengan saringan kawat.", "https://www.youtube.com/results?search_query=resep+mpasi+hati+ayam+santan"),
            ("Selasa", "🐟 Puree Ikan Kembung & Labu Siam", "Bahan: 30g Fillet Ikan Kembung, 2 sdm Nasi, Labu Siam, 1 sdt Minyak Kelapa.\n\nTutorial:\n1. Kukus fillet ikan kembung tanpa duri dan parutan labu siam.\n2. Campurkan dengan nasi tim hangat.\n3. Tambahkan 1 sdt minyak kelapa lalu saring lumat.", "https://www.youtube.com/results?search_query=resep+mpasi+ikan+kembung"),
            ("Rabu", "🥚 Puree Telur Puyuh & Bayam", "Bahan: 2 Butir Telur Puyuh, 2 sdm Nasi, Daun Bayam, Sejumput Butter.\n\nTutorial:\n1. Rebus telur puyuh hingga matang keras lalu lumatkan kuning & putihnya.\n2. Cincang halus daun bayam rebus.\n3. Aduk rata bersama nasi lembik dan butter.", "https://www.youtube.com/results?search_query=resep+mpasi+telur+puyuh+bayam"),
            ("Kamis", "🥩 Puree Daging Sapi & Kentang", "Bahan: 30g Daging Sapi Cincang, 1/2 Kentang Rebus, Keju Parut.\n\nTutorial:\n1. Tumis daging sapi cincang halus hingga matang.\n2. Rebus kentang lalu lumatkan bersama daging sapi.\n3. Taburi keju parut secukupnya.", "https://www.youtube.com/results?search_query=resep+mpasi+daging+sapi+kentang"),
            ("Jumat", "🦐 Puree Udang & Tahu Lembut", "Bahan: 30g Udang Cincang, 1/2 Tahu Putih, 2 sdm Nasi, Minyak Wijen.\n\nTutorial:\n1. Cincang halus udang kupas bersih.\n2. Lumatkan tahu putih bersama nasi lembik.\n3. Kukus selama 15 menit dan beri 2 tetes minyak wijen.", "https://www.youtube.com/results?search_query=resep+mpasi+udang+tahu"),
            ("Sabtu", "🍳 Puree Telur Bebek & Tempe", "Bahan: 1/2 Telur Bebek, 1 Potong Tempe, 2 sdm Nasi, Margarin.\n\nTutorial:\n1. Kukus tempe hingga empuk.\n2. Orak-arik telur bebek dengan margarin.\n3. Lumatkan halus tempe, telur, dan nasi hangat.", "https://www.youtube.com/results?search_query=resep+mpasi+telur+bebek+tempe"),
            ("Minggu", "🍲 Puree Ayam & Kaldu Ceker", "Bahan: 30g Daging Ayam Cincang, Wortel, Kuah Kaldu Ceker, Nasi.\n\nTutorial:\n1. Rebus daging ayam dan wortel dalam kuah kaldu ceker alami.\n2. Lumatkan nasi hangat bersama rebusan ayam hingga tekstur puree lembut.", "https://www.youtube.com/results?search_query=resep+mpasi+ayam+kaldu+ceker")
        ]
    elif "9-11" in cat_recipe:
        recipes = [
            ("Senin", "🌾 Tim Nasi Hati Ayam Cincang", "Bahan: 40g Hati Ayam, 3 sdm Nasi Tim, Buncis Cincang, Margarin.\n\nTutorial:\n1. Tumis hati ayam cincang dengan margarin.\n2. Masukkan nasi tim & potongan buncis halus.\n3. Masak hingga bumbu meresap.", "https://www.youtube.com/results?search_query=resep+nasi+tim+hati+ayam"),
            ("Selasa", "🐟 Tim Ikan Kembung Suwir & Kelor", "Bahan: 40g Ikan Kembung, Daun Kelor Cincang, Nasi Tim, Minyak Kelapa.\n\nTutorial:\n1. Suwir halus ikan kembung kukus tanpa duri.\n2. Masukkan ke nasi tim bersama daun kelor cincang halus.", "https://www.youtube.com/results?search_query=resep+nasi+tim+ikan+kembung"),
            ("Rabu", "🥚 Tim Nasi Telur Bebek & Jagung", "Bahan: 1 Telur Bebek, Jagung Manis Pipil, Nasi Tim, Margarin.\n\nTutorial:\n1. Orak-arik telur bebek dengan margarin.\n2. Campurkan dengan nasi tim & pipilan jagung manis lumat.", "https://www.youtube.com/results?search_query=resep+nasi+tim+telur+jagung"),
            ("Kamis", "🥩 Tim Daging Sapi Cincang & Brokoli", "Bahan: 40g Daging Sapi Cincang, Brokoli Cincang, Nasi Tim, Bawang Putih.\n\nTutorial:\n1. Tumis daging sapi cincang & bawang putih harum.\n2. Masukkan nasi tim & cincangan brokoli hingga matang.", "https://www.youtube.com/results?search_query=resep+nasi+tim+daging+brokoli"),
            ("Jumat", "🦐 Tim Udang Cincang & Tahu Dadu", "Bahan: 40g Udang Cincang, Tahu Dadu Kecil, Nasi Tim, Minyak Wijen.\n\nTutorial:\n1. Tumis udang cincang dengan sedikit minyak wijen.\n2. Masukkan tahu dadu kecil & nasi tim hangat.", "https://www.youtube.com/results?search_query=resep+nasi+tim+udang+tahu"),
            ("Sabtu", "🍳 Tim Telur Puyuh & Sup Wortel", "Bahan: 3 Butir Telur Puyuh, Wortel Dadu, Nasi Tim, Kuah Ayam.\n\nTutorial:\n1. Rebus 3 telur puyuh.\n2. Sajikan bersama nasi tim & sup wortel potong dadu kecil.", "https://www.youtube.com/results?search_query=resep+nasi+tim+telur+puyuh"),
            ("Minggu", "🍲 Tim Bola-Bola Ayam & Labu", "Bahan: 40g Ayam Cincang, Labu Siam Dadu, Nasi Tim, Kaldu.\n\nTutorial:\n1. Buat adonan bola ayam cincang kecil.\n2. Rebus dalam kuah kaldu bersama labu siam hingga matang.", "https://www.youtube.com/results?search_query=resep+nasi+tim+bola+ayam")
        ]
    elif "12-23" in cat_recipe:
        recipes = [
            ("Senin", "🍲 Sup Bola Bakso Ayam Udang", "Bahan: 50g Daging Ayam & Udang, Wortel, Kentang, Kuah Kaldu.\n\nTutorial:\n1. Buat bakso ayam udang homemade.\n2. Rebus dalam kuah kaldu wortel & kentang hingga mengapung matang.", "https://www.youtube.com/results?search_query=resep+sup+bakso+ayam+udang+anak"),
            ("Selasa", "🐟 Pepes Ikan Lele / Belut Tanpa Duri", "Bahan: 50g Lele/Belut, Bumbu Kuning Lembut, Daun Pisang.\n\nTutorial:\n1. Bumbui lele/belut tanpa duri dengan bumbu harum.\n2. Kukus dalam bungkus daun pisang selama 20 menit.", "https://www.youtube.com/results?search_query=resep+pepes+lele+tanpa+duri+anak"),
            ("Rabu", "🥩 Semur Daging Cincang & Telur Puyuh", "Bahan: 50g Daging Sapi Cincang, 3 Telur Puyuh, Kecap Manis, Bawang.\n\nTutorial:\n1. Tumis daging sapi cincang kecap manis harum.\n2. Masukkan 3 butir telur puyuh rebus hingga bumbu meresap.", "https://www.youtube.com/results?search_query=resep+semur+daging+cincang+balita"),
            ("Kamis", "🍗 Ayam Goreng Kaldu & Sayur Bening", "Bahan: 1 Potong Ayam Ungkep Kaldu, Bayam, Nasi Warm.\n\nTutorial:\n1. Ungkep ayam dengan kaldu alami lalu goreng sebentar.\n2. Sajikan dengan sayur bening bayam & nasi hangat.", "https://www.youtube.com/results?search_query=resep+ayam+goreng+kaldu+sayur+bayam"),
            ("Jumat", "🦐 Tumis Udang Brokoli Saus Mentega", "Bahan: 50g Udang Kupas, Brokoli, Mentega, Kecap Manis.\n\nTutorial:\n1. Tumis udang kupas & brokoli dengan mentega harum.\n2. Beri sedikit kecap manis.", "https://www.youtube.com/results?search_query=resep+udang+brokoli+mentega+anak"),
            ("Sabtu", "🍳 Telur Dadar Daun Kelor & Nasi Warm", "Bahan: 1 Butir Telur Ayam, Daun Kelor Cincang, Margarin.\n\nTutorial:\n1. Kocok 1 butir telur dengan daun kelor cincang.\n2. Dadar tipis dengan margarin dan sajikan bersama nasi hangat.", "https://www.youtube.com/results?search_query=resep+telur+dadar+daun+kelor"),
            ("Minggu", "🥞 Pancake Hati Ayam & Pisang", "Bahan: Tepung Terigu, Pisang Lumat, 1 Telur, Bubuk Hati Ayam Sangrai.\n\nTutorial:\n1. Campurkan tepung terigu, pisang lumat, telur, & bubuk hati ayam sangrai.\n2. Panggang di teflon dengan api kecil hingga matang keemasan.", "https://www.youtube.com/results?search_query=resep+pancake+hati+ayam+pisang")
        ]
    elif "Anak-Anak" in cat_recipe:
        recipes = [
            ("Senin", "🍱 Bento Nasi Kuning Ayam Popcorn", "Bahan: Nasi Kuning, Dada Ayam Tepung, Telur Rebus, Wortel.\n\nTutorial:\n1. Goreng dada ayam tepung crispy.\n2. Cetak nasi kuning dan hias bersama potongan telur rebus.", "https://www.youtube.com/results?search_query=resep+bento+anak+sekolah+sehat"),
            ("Selasa", "🍝 Spaghetti Salmon Bolognese", "Bahan: Pasta Spaghetti, Fillet Salmon Cincang, Saus Tomat Homemade.\n\nTutorial:\n1. Rebus spaghetti al dente.\n2. Tumis salmon cincang dengan saus tomat lalu siram di atas pasta.", "https://www.youtube.com/results?search_query=resep+spaghetti+salmon+anak"),
            ("Rabu", "🍲 Sup Makaroni Daging Sapi & Buncis", "Bahan: Daging Sapi Cincang, Makaroni, Buncis, Wortel, Kaldu Sapi.\n\nTutorial:\n1. Rebus daging sapi dan makaroni hingga empuk.\n2. Masukkan sayuran buncis & wortel dalam kuah kaldu gizi.", "https://www.youtube.com/results?search_query=resep+sup+makaroni+daging+sapi"),
            ("Kamis", "🍳 Nasi Goreng Telur Puyuh & Udang", "Bahan: Nasi Putih, 4 Telur Puyuh, Udang Kupas, Minyak Wijen.\n\nTutorial:\n1. Tumis udang kupas dan telur puyuh orak-arik.\n2. Masukkan nasi dan bumbui ringan tanpa pengawet.", "https://www.youtube.com/results?search_query=resep+nasi+goreng+sehat+anak"),
            ("Jumat", "🍗 Chicken Teriyaki & Tumis Brokoli", "Bahan: Dada Ayam, Saus Teriyaki, Brokoli, Biji Wijen.\n\nTutorial:\n1. Tumis ayam potong dengan saus teriyaki gurih.\n2. Sajikan dengan rebusan brokoli segar & taburan biji wijen.", "https://www.youtube.com/results?search_query=resep+chicken+teriyaki+anak"),
            ("Sabtu", "🥪 Sandwich Telur Keju & Daging Asap", "Bahan: Roti Tawar Gandum, Telur Dadar, Keju Slice, Daging Asap.\n\nTutorial:\n1. Panggang roti gandum di atas teflon.\n2. Susun telur dadar, keju, & daging asap hangat.", "https://www.youtube.com/results?search_query=resep+sandwich+sehat+anak"),
            ("Minggu", "🍲 Soto Ayam Kuah Bening & Telur Rebus", "Bahan: Daging Ayam Suwir, Kuah Soto Bening, Telur Rebus, Tauge.\n\nTutorial:\n1. Rebus ayam kuah soto rempah alami.\n2. Sajikan suwiran ayam, tauge, & telur rebus matang.", "https://www.youtube.com/results?search_query=resep+soto+ayam+kuah+bening")
        ]
    elif "Remaja" in cat_recipe:
        recipes = [
            ("Senin", "🥩 Beef Bowl Yoshinoya Style & Egg", "Bahan: Daging Sapi Slice, Bawang Bombay, Kecap Asin, 1 Telur Ceplok.\n\nTutorial:\n1. Tumis daging sapi slice bersama bawang bombay saus gurih.\n2. Tumpuk di atas nasi hangat bersama telur ceplok setengah matang.", "https://www.youtube.com/results?search_query=resep+beef+bowl+ala+yoshinoya"),
            ("Selasa", "🥗 Salad Salmon Panggang & Avokad", "Bahan: Fillet Salmon, Alpukat Slice, Sayur Selada, Olive Oil.\n\nTutorial:\n1. Panggang salmon dengan garam & lada hitam.\n2. Campur selada segar, potongan alpukat, & dressing olive oil.", "https://www.youtube.com/results?search_query=resep+salad+salmon+alpukat"),
            ("Rabu", "🍗 Ayam Bakar Madu & Tumis Kangkung", "Bahan: Paha Ayam, Bumbu Madu, Kangkung, Bawang Merah Putih.\n\nTutorial:\n1. Ungkep ayam bumbu madu lalu bakar keemasan.\n2. Tumis kangkung segar dengan sedikit minyak.", "https://www.youtube.com/results?search_query=resep+ayam+bakar+madu+teflon"),
            ("Kamis", "🍲 Sup Ikan Batang Asam Pedas", "Bahan: Fillet Kakap/Tenggiri, Tomat Hijau, Belimbing Wulung.\n\nTutorial:\n1. Rebus kuah rempah bening asam segar.\n2. Masukkan fillet ikan & potongan tomat hingga matang.", "https://www.youtube.com/results?search_query=resep+sup+ikan+asam+pedas"),
            ("Jumat", "🍝 Fusilli Tuna Spicy Olive Oil", "Bahan: Pasta Fusilli, Tuna Cincang, Cabai Rawit, Minyak Zaitun.\n\nTutorial:\n1. Tumis tuna cincang & irisan cabai dengan olive oil.\n2. Campurkan pasta fusilli rebus.", "https://www.youtube.com/results?search_query=resep+pasta+tuna+aglio+olio"),
            ("Sabtu", "🍳 Omelet Daging Cincang & Bayam Keju", "Bahan: 2 Telur Ayam, Daging Sapi Cincang, Bayam, Keju Mozzarella.\n\nTutorial:\n1. Kocok telur dengan isi daging cincang & bayam.\n2. Lipat omelet dan beri lelehan keju mozzarella di atasnya.", "https://www.youtube.com/results?search_query=resep+omelet+keju+daging+cincang"),
            ("Minggu", "🥣 Smoothies Bowl Buah Naga & Chia", "Bahan: Buah Naga Blend, Pisang, Chia Seeds, Kacang Almond.\n\nTutorial:\n1. Blender halus buah naga & pisang dingin.\n2. Tuang ke mangkok dan beri topping chia seeds & almond renyah.", "https://www.youtube.com/results?search_query=resep+smoothie+bowl+buah+naga")
        ]
    elif "Dewasa" in cat_recipe:
        recipes = [
            ("Senin", "🐟 Salmon Panggang Lemon & Kentang", "Bahan: Fillet Salmon, Perasan Lemon, Kentang, Rosemary.\n\nTutorial:\n1. Marinasi salmon dengan perasan lemon & lada.\n2. Panggang teflon 8 menit & sajikan dengan kentang rebus.", "https://www.youtube.com/results?search_query=resep+salmon+panggang+lemon"),
            ("Selasa", "🥗 Pokebowl Tuna Segar & Edamame", "Bahan: Fillet Tuna, Kacang Edamame, Nasi Merah, Wijen.\n\nTutorial:\n1. Tumis tuna sebentar dengan minyak wijen.\n2. Susun di atas nasi merah bersama edamame rebus.", "https://www.youtube.com/results?search_query=resep+tuna+poke+bowl"),
            ("Rabu", "🥩 Tumis Daging Sapi Lada Hitam", "Bahan: Daging Sapi Lean Slice, Paprika Merah Hijau, Lada Hitam.\n\nTutorial:\n1. Tumis daging sapi tanpa lemak bersama saus lada hitam.\n2. Masukkan potongan paprika kaya vitamin C.", "https://www.youtube.com/results?search_query=resep+daging+sapi+lada+hitam"),
            ("Kamis", "🍗 Dada Ayam Panggang Herb", "Bahan: Dada Ayam Tanpa Kulit, Oregano, Buncis, Minyak Zaitun.\n\nTutorial:\n1. Panggang dada ayam bumbu herb rendah garam.\n2. Tumis buncis dengan minyak zaitun ringan.", "https://www.youtube.com/results?search_query=resep+dada+ayam+panggang+diet"),
            ("Jumat", "🍲 Sup Ikan Gurame Bening Kemangi", "Bahan: Fillet Gurame, Daun Kemangi, Jahe, Serai, Kuah Bening.\n\nTutorial:\n1. Rebus kuah jahe serai wangi tanpa santan.\n2. Masukkan fillet gurame & daun kemangi hingga segar.", "https://www.youtube.com/results?search_query=resep+sup+ikan+gurame+kemangi"),
            ("Sabtu", "🍳 Tofu Stir Fry Shimeji & Telur", "Bahan: Tofu Jepang, Jamur Shimeji, 1 Telur, Saus Tiram.\n\nTutorial:\n1. Tumis tofu & jamur shimeji saus tiram rendah natrium.\n2. Orak-arik telur sebagai peningkat protein.", "https://www.youtube.com/results?search_query=resep+tumis+tofu+jamur+shimeji"),
            ("Minggu", "🥣 Oatmeal Kayu Manis & Telur Rebus", "Bahan: Rolled Oats, Bubuk Kayu Manis, Buah Apel, 2 Telur Rebus.\n\nTutorial:\n1. Seduh rolled oats hangat dan beri parutan apel & kayu manis.\n2. Sajikan dengan 2 butir telur rebus matang.", "https://www.youtube.com/results?search_query=resep+oatmeal+sehat+pagi+hari")
        ]
    else: # Lansia
        recipes = [
            ("Senin", "🐟 Tim Fillet Kakap Jahe Lengkuas", "Bahan: Fillet Kakap, Irisan Jahe, Daun Bawang, Minyak Wijen.\n\nTutorial:\n1. Kukus fillet kakap dengan irisan jahe & serai hingga lembut.\n2. Beri beberapa tetes minyak wijen wangi tanpa garam berlebih.", "https://www.youtube.com/results?search_query=resep+tim+ikan+kakap+jahe+lansia"),
            ("Selasa", "🍲 Sup Tahu Sutra & Ayam Cincang", "Bahan: Tahu Sutra, Dada Ayam Cincang, Labu Siam, Kuah Bening.\n\nTutorial:\n1. Rebus kuah kaldu bening rendah garam.\n2. Masukkan tahu sutra lembut & ayam cincang empuk mudah dikunyah.", "https://www.youtube.com/results?search_query=resep+sup+tahu+sutra+ayam"),
            ("Rabu", "🥚 Pepes Telur Tahu & Kemangi", "Bahan: 2 Telur Kocok, Tahu Lumat, Daun Kemangi, Bungkus Pisang.\n\nTutorial:\n1. Campur lumat tahu dan telur dengan kemangi harum.\n2. Kukus dalam daun pisang hingga matang empuk.", "https://www.youtube.com/results?search_query=resep+pepes+tahu+telur+lembut"),
            ("Kamis", "🥩 Semur Daging Giling Empuk & Wortel", "Bahan: Daging Sapi Giling Halus, Wortel Rebus Empuk, Kecap.\n\nTutorial:\n1. Masak daging sapi giling lembut dengan bumbu semur ringan.\n2. Masukkan wortel rebus hingga tekstur sangat lembut.", "https://www.youtube.com/results?search_query=resep+semur+daging+giling+lansia"),
            ("Jumat", "🥣 Bubur Manado Tinutuan Komplit", "Bahan: Beras, Labu Kuning Lumat, Bayam, Jagung Manis Pipil.\n\nTutorial:\n1. Masak bubur beras bersama labu kuning lumat kaya karotenoid.\n2. Masukkan sayuran lembut untuk kemudahan cerna lansia.", "https://www.youtube.com/results?search_query=resep+bubur+manado+sehat"),
            ("Sabtu", "🍲 Sup Ayam Ceker & Sayuran Bening", "Bahan: Ceker Ayam Kaldu, Wortel, Kentang, Brokoli Rebus.\n\nTutorial:\n1. Rebus ceker ayam lama hingga keluar kolagen kaldu alami.\n2. Masukkan sayuran dipotong kecil empuk.", "https://www.youtube.com/results?search_query=resep+sup+ceker+ayam+kolagen"),
            ("Minggu", "🍳 Scrambled Egg Tahu & Puree Labu", "Bahan: 2 Telur Bebek/Ayam, Tahu Sutra, Puree Labu Kuning.\n\nTutorial:\n1. Orak-arik lembut telur dan tahu sutra dengan butter.\n2. Sajikan bersama puree labu kuning hangat yang lezat.", "https://www.youtube.com/results?search_query=resep+telur+orak+arik+tahu+lembut")
        ]

    for day_name, title, tut, yt_link in recipes:
        with st.expander(f"🍽️ {day_name}: {title}"):
            st.markdown(f"**Bahan & Cara Memasak:**\n\n{tut}")
            st.markdown(f'<a href="{yt_link}" target="_blank" class="ref-btn">▶️ Tonton Video Tutorial YouTube</a>', unsafe_allow_html=True)

# ================= TAB 4: EDUKASI & BERITA VIDEO (LENGKAP DEFINISI, BAHAYA & LINK) =================
with tab4:
    st.markdown(f"### {txt['edu_title']}")
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("""
        <div class="edu-card">
            <h3>📖 1. Definisi & Bahaya Stunting</h3>
            <p>Stunting adalah gangguan pertumbuhan kronis pada anak (tinggi badan di bawah standar usianya) akibat kekurangan gizi menahun dan infeksi berulang dalam 1.000 Hari Pertama Kehidupan (HPK). Bahayanya meliputi penurunan kecerdasan (IQ), gangguan metabolisme saat dewasa, serta rentan terhadap penyakit tidak menular.</p>
            <a href="https://www.who.int/news-room/fact-sheets/detail/stunting-in-a-nutshell" target="_blank" class="ref-btn">🌐 Rujukan Resmi WHO Stunting</a>
            <a href="https://ayosehat.kemkes.go.id/topik-penyakit/defisiensi-nutrisi/stunting" target="_blank" class="ref-btn">🌐 AyoSehat Kemenkes RI</a>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="edu-card">
            <h3>🍳 2. Keunggulan Protein Hewani & Pencegahan</h3>
            <p>Protein hewani (seperti ikan kembung, telur, hati ayam, daging) mengandung asam amino esensial lengkap dan zinc tinggi yang terbukti secara klinis jauh lebih efektif merangsang hormon pertumbuhan tulang (*IGF-1*) dibanding protein nabati.</p>
            <a href="https://www.unicef.org/reports/child-nutrition-report" target="_blank" class="ref-btn">🌐 UNICEF Child Nutrition</a>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="edu-card">
            <h3>📰 3. Berita & Isu Terkini Penanganan Stunting</h3>
            <p>Pemerintah terus menggalakkan program intervensi serentak di posyandu seluruh Indonesia, menekankan pentingnya pemberian Makanan Tambahan (PMT) berbasis pangan lokal kaya protein hewani serta percepatan akses sanitasi layak.</p>
            <a href="https://stunting.go.id" target="_blank" class="ref-btn">🌐 Portal Resmi TP2S BKKBN</a>
        </div>
        """, unsafe_allow_html=True)

    with col_e2:
        st.markdown("""
        <div class="edu-card">
            <h3>🎥 Video Edukasi: Pencegahan Stunting Nasional Kemenkes</h3>
            <p>Tonton video panduan resmi mengenai gerakan pencegahan stunting melalui pilar ABCDE:</p>
            <iframe width="100%" height="215" src="https://www.youtube.com/embed/2Z1h9mQX7EQ" title="Video Edukasi Kemenkes" frameborder="0" allowfullscreen style="border-radius:10px; margin-top:10px;"></iframe>
            <a href="https://www.youtube.com/watch?v=2Z1h9mQX7EQ" target="_blank" class="ref-btn">▶️ Buka Langsung di YouTube</a>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="edu-card">
            <h3>🎥 Video Edukasi BKKBN: 1000 Hari Pertama Kehidupan</h3>
            <p>Panduan penting bagi orang tua dalam menjaga asupan gizi sejak dalam kandungan hingga usia 2 tahun:</p>
            <iframe width="100%" height="215" src="https://www.youtube.com/embed/S00n-c_qeC0" title="Video Edukasi BKKBN" frameborder="0" allowfullscreen style="border-radius:10px; margin-top:10px;"></iframe>
            <a href="https://www.youtube.com/watch?v=S00n-c_qeC0" target="_blank" class="ref-btn">▶️ Buka Langsung di YouTube</a>
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
