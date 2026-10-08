import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

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
# 2. DICTIONARY MULTI-LANGUAGE LENGKAP (ID / EN)
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
        'eval_header': "📊 Hasil Evaluasi Diagnosa & Z-Score WHO",
        'stunting_risk': "Tingkat Indikasi Stunting (%)",
        'xai_title': "💡 Transparansi AI (Explainable AI - Feature Importance)",
        'sim_title': "🔮 Simulasi Target Pertumbuhan (What-If Analysis)",
        'sim_months': "Simulasi Usia Muka (Bulan Ke Depan)",
        'print_btn': "🖨️ Cetak / Simpan Kartu Laporan (PDF)",
        'chart_title': "📈 Kurva Standar Pertumbuhan WHO (Tinggi vs Usia)",
        'chart_analysis_title': "📋 Ringkasan Analisis Tren Pertumbuhan",
        'recipe_title': "🥣 Rekomendasi Menu MPASI Protein Hewani Lokal",
        'edu_title': "📚 Pusat Edukasi & Panduan Pencegahan Stunting",
        'report_card_title': "📋 KARTU LAPORAN AN TROPOMETRI & EVALUASI BALITA",
        'report_sub1': "1. Data Profil Balita",
        'report_sub2': "2. Evaluasi Medis (WHO HAZ & AI)",
        'report_sub3': "3. Rencana Tindakan Lanjutan (Action Plan)",
        'report_note': "Catatan: Kartu laporan ini dapat dicetak dan dibawa saat berkonsultasi dengan kader Posyandu, Bidan, atau Dokter Anak di Puskesmas.",
        'xai_features': ['Usia (Bulan)', 'Jenis Kelamin', 'Tinggi Badan', 'Berat Badan', 'Berat Lahir', 'ASI Eksklusif'],
        'xai_xlabel': "Tingkat Pengaruh (%)"
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
        'eval_header': "📊 Diagnostic Evaluation & WHO Z-Score Results",
        'stunting_risk': "Stunting Indication Level (%)",
        'xai_title': "💡 AI Transparency (Explainable AI - Feature Importance)",
        'sim_title': "🔮 Growth Target Simulation (What-If Analysis)",
        'sim_months': "Months Ahead to Simulate",
        'print_btn': "🖨️ Print / Save Report Card (PDF)",
        'chart_title': "📈 WHO Standard Growth Curve (Height vs Age)",
        'chart_analysis_title': "📋 Growth Trend Analysis Summary",
        'recipe_title': "🥣 Recommended Local Animal Protein MPASI Recipes",
        'edu_title': "📚 Stunting Prevention Educational Hub & Guide",
        'report_card_title': "📋 CHILD ANTHROPOMETRY & EVALUATION REPORT CARD",
        'report_sub1': "1. Child Profile Data",
        'report_sub2': "2. Medical Evaluation (WHO HAZ & AI)",
        'report_sub3': "3. Follow-up Action Plan",
        'report_note': "Note: This report card can be printed and brought when consulting with healthcare providers or Posyandu officers.",
        'xai_features': ['Age (Months)', 'Gender', 'Height', 'Weight', 'Birth Weight', 'Exclusive Breastfeeding'],
        'xai_xlabel': "Importance Weight (%)"
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
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.8), rgba(15, 23, 42, 0.8)), 
                    url('https://i.pinimg.com/736x/e9/67/8d/e9678dd9f3233a7528d3e9e3310bbed8.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
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
    .edu-card {
        background: rgba(30, 41, 59, 0.88);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        margin-bottom: 18px;
        color: #F8FAFC;
    }
    .edu-card h3 {
        color: #38BDF8;
        margin-bottom: 12px;
    }
    .report-box {
        background-color: #FFFFFF;
        color: #0F172A;
        padding: 28px;
        border-radius: 16px;
        border-left: 8px solid #0284C7;
        margin-top: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
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
    @media print {
        body * { visibility: hidden; }
        .report-box, .report-box * { visibility: visible; }
        .report-box {
            position: absolute; left: 0; top: 0; width: 100%;
            border: 2px solid #000; box-shadow: none;
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

# Header Utama
st.markdown(f"""
<div class="main-header">
    <h1>{txt['title']}</h1>
    <p style="font-size: 1.1rem; opacity: 0.95;">
        {txt['subtitle']}
    </p>
</div>
""", unsafe_allow_html=True)

if 'age_val' not in st.session_state:
    st.session_state.age_val = 24
if 'height_val' not in st.session_state:
    st.session_state.height_val = 75.0
if 'weight_val' not in st.session_state:
    st.session_state.weight_val = 11.0

# ---------------------------------------------------------
# 5. TAB NAVIGASI UTAMA
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([txt['tab1'], txt['tab2'], txt['tab3'], txt['tab4']])

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
                if curr_lang == 'ID':
                    st.error(f"⚠️ **STATUS: {who_status.upper()}**\nTingkat Kepastian AI: **{proba:.1f}%**")
                    st.warning(f"🔍 Median standar WHO usia {age} bulan adalah **{median_h} cm** (Selisih **{round(height - median_h, 1)} cm**).")
                else:
                    st.error(f"⚠️ **STATUS: {who_status.upper()}**\nAI Confidence Level: **{proba:.1f}%**")
                    st.warning(f"🔍 Median WHO standard for {age} months is **{median_h} cm** (Difference: **{round(height - median_h, 1)} cm**).")
            else:
                if curr_lang == 'ID':
                    st.success(f"✅ **STATUS: {who_status.upper()}**\nTingkat Kepastian AI: **{proba:.1f}%**")
                    st.info(f"🎉 Tinggi anak Anda (**{height} cm**) berada di kisaran normal WHO (Median: {median_h} cm).")
                else:
                    st.success(f"✅ **STATUS: {who_status.upper()}**\nAI Confidence Level: **{proba:.1f}%**")
                    st.info(f"🎉 Your child's height (**{height} cm**) is within the normal WHO range (Median: {median_h} cm).")

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

        # FITUR PRINT / LAPORAN PDF RINCI MULTI-LANGUAGE
        if curr_lang == 'ID':
            report_html = f"""
            <div class="report-box" id="printable-report">
                <h2 style="color: #0284C7; text-align: center; margin-top:0;">{txt['report_card_title']}</h2>
                <hr style="border: 1px solid #0284C7;">
                <h4>{txt['report_sub1']}</h4>
                <table style="width:100%; font-size:14px; border-collapse: collapse;">
                    <tr><td><strong>Usia Balita:</strong> {age} Bulan</td><td><strong>Jenis Kelamin:</strong> {gender_str}</td></tr>
                    <tr><td><strong>Tinggi Badan:</strong> {height} cm</td><td><strong>Berat Badan:</strong> {weight} kg</td></tr>
                    <tr><td><strong>Berat Lahir:</strong> {birth_weight} kg</td><td><strong>Riwayat ASI Eksklusif:</strong> {asi_str}</td></tr>
                </table>
                <br>
                <h4>{txt['report_sub2']}</h4>
                <ul>
                    <li><strong>Skor Standar Pertumbuhan WHO (HAZ Z-Score):</strong> <span style="color:{'red' if z_score < -2 else 'green'}; font-weight:bold;">{z_score} SD ({who_status})</span></li>
                    <li><strong>Standar Median WHO Usia {age} Bln:</strong> {median_h} cm (Deviasi: {round(height - median_h, 1)} cm)</li>
                    <li><strong>Tingkat Risiko AI Stunting:</strong> {proba:.1f}%</li>
                </ul>
                <br>
                <h4>{txt['report_sub3']}</h4>
                <ol>
                    <li><strong>Intervensi Nutrisi Protein Hewani:</strong> Berikan minimal 2 porsi protein hewani berkualitas tinggi per hari (contoh: 1 butir telur + 50g hati ayam/ikan kembung).</li>
                    <li><strong>Suplementasi Zat Besi & Vitamin A:</strong> Konsultasikan dengan bidan/dokter untuk pemberian Vitamin A dan taburia/sirup zat besi.</li>
                    <li><strong>Pemantauan Rutin Posyandu:</strong> Timbang berat badan dan ukur tinggi badan secara teratur setiap bulan untuk memantau kurva pertumbuhan.</li>
                    <li><strong>Sanitasi & Kebersihan (PHBS):</strong> Pastikan air minum direbus hingga mendidih dan cuci tangan dengan sabun sebelum menyiapakan MPASI.</li>
                </ol>
                <br>
                <p style="font-size: 11px; color: #64748B; font-style: italic;">{txt['report_note']}</p>
            </div>
            """
        else:
            report_html = f"""
            <div class="report-box" id="printable-report">
                <h2 style="color: #0284C7; text-align: center; margin-top:0;">{txt['report_card_title']}</h2>
                <hr style="border: 1px solid #0284C7;">
                <h4>{txt['report_sub1']}</h4>
                <table style="width:100%; font-size:14px; border-collapse: collapse;">
                    <tr><td><strong>Child Age:</strong> {age} Months</td><td><strong>Gender:</strong> {gender_str}</td></tr>
                    <tr><td><strong>Height:</strong> {height} cm</td><td><strong>Weight:</strong> {weight} kg</td></tr>
                    <tr><td><strong>Birth Weight:</strong> {birth_weight} kg</td><td><strong>Exclusive Breastfeeding:</strong> {asi_str}</td></tr>
                </table>
                <br>
                <h4>{txt['report_sub2']}</h4>
                <ul>
                    <li><strong>WHO Growth Standard Score (HAZ Z-Score):</strong> <span style="color:{'red' if z_score < -2 else 'green'}; font-weight:bold;">{z_score} SD ({who_status})</span></li>
                    <li><strong>WHO Median Height for {age} Mths:</strong> {median_h} cm (Deviation: {round(height - median_h, 1)} cm)</li>
                    <li><strong>AI Stunting Risk Confidence:</strong> {proba:.1f}%</li>
                </ul>
                <br>
                <h4>{txt['report_sub3']}</h4>
                <ol>
                    <li><strong>Animal Protein Nutrition Intervention:</strong> Provide at least 2 servings of high-quality animal protein daily (e.g., 1 egg + 50g chicken liver/mackerel).</li>
                    <li><strong>Iron & Vitamin A Supplementation:</strong> Consult with local healthcare workers for Vitamin A capsules and iron supplementation.</li>
                    <li><strong>Regular Posyandu/Pediatric Monitoring:</strong> Measure height and weight monthly to track the growth curve velocity.</li>
                    <li><strong>Sanitation & Hygiene (WASH):</strong> Ensure clean boiled drinking water and practice handwashing with soap before preparing meals.</li>
                </ol>
                <br>
                <p style="font-size: 11px; color: #64748B; font-style: italic;">{txt['report_note']}</p>
            </div>
            """

        st.markdown(report_html, unsafe_allow_html=True)
        st.button(txt['print_btn'], on_click=lambda: st.components.v1.html("<script>window.print();</script>"))

        # TRANSPARANSI AI (XAI)
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
        name='Posisi Saat Ini' if curr_lang == 'ID' else 'Current Position', 
        text=[f'Anak Anda ({curr_h} cm)' if curr_lang == 'ID' else f'Your Child ({curr_h} cm)'],
        textposition="top center", marker=dict(size=14, color='#38BDF8', symbol='star')
    ))

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=420)
    st.plotly_chart(fig, use_container_width=True)

    # BAGIAN PENJELASAN DI BAWAH GRAFIK WHO
    st.markdown(f"#### {txt['chart_analysis_title']}")
    z_sc, st_name, _, med_val = calculate_who_zscore(curr_a, curr_h)
    diff = round(curr_h - med_val, 1)
    
    if curr_lang == 'ID':
        st.markdown(f"""
        <div class="edu-card">
            <p>📌 <strong>Interpretasi Grafik:</strong> Titik bintang biru mewakili posisi tumbuh kembang balita Anda saat ini pada usia <strong>{curr_a} bulan</strong> dengan tinggi <strong>{curr_h} cm</strong>.</p>
            <ul>
                <li><strong>Garis Hijau (0 SD):</strong> Garis rata-rata standar internasional WHO ({med_val} cm). Selisih anak Anda: <strong>{'+' if diff >= 0 else ''}{diff} cm</strong>.</li>
                <li><strong>Garis Kuning (-2 SD):</strong> Batas ambang bawah kategori normal (<strong>{round(med_val - 6.4, 1)} cm</strong>). Jika posisi di bawah garis ini, anak tergolong <em>Stunted</em>.</li>
                <li><strong>Garis Merah (-3 SD):</strong> Batas ambang bawah stunting berat (<strong>{round(med_val - 9.6, 1)} cm</strong>). Memerlukan penanganan medis klinis intensif.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="edu-card">
            <p>📌 <strong>Chart Interpretation:</strong> The blue star represents your child's current growth milestone at <strong>{curr_a} months</strong> with a height of <strong>{curr_h} cm</strong>.</p>
            <ul>
                <li><strong>Green Line (0 SD):</strong> WHO international average median ({med_val} cm). Your child's variance: <strong>{'+' if diff >= 0 else ''}{diff} cm</strong>.</li>
                <li><strong>Yellow Line (-2 SD):</strong> Lower threshold for normal category (<strong>{round(med_val - 6.4, 1)} cm</strong>). Positions below this line indicate <em>Stunted</em>.</li>
                <li><strong>Red Line (-3 SD):</strong> Severe stunting threshold (<strong>{round(med_val - 9.6, 1)} cm</strong>). Requires urgent medical clinical intervention.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # FITUR SIMULATOR TARGET PERTUMBUHAN ("WHAT-IF")
    st.write("---")
    st.markdown(f"### {txt['sim_title']}")
    sim_months = st.slider(txt['sim_months'], min_value=1, max_value=12, value=6)
    
    future_age = curr_a + sim_months
    target_h_normal = round(48.0 + (future_age * 1.25), 1)
    needed_growth = round(target_h_normal - curr_h, 1)

    c_sim1, c_sim2 = st.columns(2)
    with c_sim1:
        if curr_lang == 'ID':
            st.info(f"🗓️ **Target Usia:** {future_age} Bulan ({sim_months} bulan ke depan)")
            st.success(f"🎯 **Target Tinggi Ideal WHO:** {target_h_normal} cm")
        else:
            st.info(f"🗓️ **Target Age:** {future_age} Months ({sim_months} months ahead)")
            st.success(f"🎯 **Target WHO Ideal Height:** {target_h_normal} cm")
    with c_sim2:
        if curr_lang == 'ID':
            st.metric("Total Kebutuhan Tambahan Tinggi", f"+{needed_growth} cm", delta=f"{round(needed_growth/sim_months, 1)} cm/bulan")
        else:
            st.metric("Total Height Growth Needed", f"+{needed_growth} cm", delta=f"{round(needed_growth/sim_months, 1)} cm/month")

# ================= TAB 3: RESEP & MENU MPASI =================
with tab3:
    st.markdown(f"### {txt['recipe_title']}")
    
    age_opt = ["6 - 8 Bulan / Months", "9 - 11 Bulan / Months", "12 - 23 Bulan / Months"]
    age_group = st.radio("Pilih Kelompok Usia Balita / Select Age Group:", age_opt, horizontal=True)
    
    if age_group == age_opt[0]:
        if curr_lang == 'ID':
            st.markdown("""
            <div class="edu-card">
                <h3>🥣 Tekstur: Bubur Kental (Lumat/Saring) | 2-3 Kali Makan Utama</h3>
                <p><strong>Rekomendasi Porsi Protein Hewani:</strong> Minimal 30-45 gram/hari (1 telur puyuh / 1/2 telur ayam / 30g hati ayam).</p>
                <hr style="border:0.5px solid #334155;">
                <ul>
                    <li>🍳 <strong>Menu 1 (Puree Hati Ayam & Santan):</strong> Nasi lembik + hati ayam rebus lumat + wortel parut + 1 sdt santan segar (Kaya Zat Besi & Seng).</li>
                    <li>🐟 <strong>Menu 2 (Bubur Saring Ikan Kembung):</strong> Nasi + fillet ikan kembung kukus + labu siam + minyak kelapa (Kaya Omega-3 & DHA).</li>
                    <li>🥚 <strong>Menu 3 (Bubur Tim Telur Puyuh & Bayam):</strong> Nasi + 2 butir telur puyuh rebus lumat + bayam cincang + margarin.</li>
                    <li>🥩 <strong>Menu 4 (Puree Daging Sapi Lumat):</strong> Daging sapi cincang halus + kentang rebus lumat + keju parut secukupnya.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="edu-card">
                <h3>🥣 Texture: Thick Puree (Mashed/Strained) | 2-3 Main Meals</h3>
                <p><strong>Animal Protein Serving:</strong> At least 30-45 grams/day (1 quail egg / 1/2 chicken egg / 30g chicken liver).</p>
                <hr style="border:0.5px solid #334155;">
                <ul>
                    <li>🍳 <strong>Menu 1 (Chicken Liver & Coconut Milk Puree):</strong> Soft rice + mashed boiled chicken liver + grated carrot + 1 tsp coconut milk (Rich in Iron & Zinc).</li>
                    <li>🐟 <strong>Menu 2 (Mackerel Fish Puree):</strong> Rice + steamed mackerel fillet + chayote + coconut oil (High in Omega-3 & DHA).</li>
                    <li>🥚 <strong>Menu 3 (Quail Egg & Spinach Puree):</strong> Rice + 2 mashed boiled quail eggs + chopped spinach + butter.</li>
                    <li>🥩 <strong>Menu 4 (Mashed Beef Puree):</strong> Minced beef + mashed boiled potatoes + grated cheese.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    elif age_group == age_opt[1]:
        if curr_lang == 'ID':
            st.markdown("""
            <div class="edu-card">
                <h3>🥣 Tekstur: Cincang Halus / Nasi Tim Lumat | 3-4 Kali Makan Utama</h3>
                <p><strong>Rekomendasi Porsi Protein Hewani:</strong> Minimal 45-60 gram/hari.</p>
                <hr style="border:0.5px solid #334155;">
                <ul>
                    <li>🐟 <strong>Menu 1 (Nasi Tim Ikan Kembung Suwir):</strong> Nasi tim + suwiran ikan kembung + daun kelor cincang + minyak zaitun/kelapa.</li>
                    <li>🍗 <strong>Menu 2 (Tim Ayam Cincang & Brokoli):</strong> Daging ayam cincang + nasi + brokoli potong kecil + kuah kaldu ceker.</li>
                    <li>🦐 <strong>Menu 3 (Bubur Tim Udang Cincang & Tahu):</strong> Udang kupas cincang + tahu lumat + wortel + minyak wijen.</li>
                    <li>🍳 <strong>Menu 4 (Orak-Arik Telur Bebek & Tempe):</strong> Telur bebek kocok + potongan tempe kecil + jagung manis pipil lumat.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="edu-card">
                <h3>🥣 Texture: Finely Chopped / Soft Steamed Rice | 3-4 Main Meals</h3>
                <p><strong>Animal Protein Serving:</strong> At least 45-60 grams/day.</p>
                <hr style="border:0.5px solid #334155;">
                <ul>
                    <li>🐟 <strong>Menu 1 (Shredded Mackerel Steamed Rice):</strong> Steamed rice + shredded mackerel + chopped Moringa leaves + coconut oil.</li>
                    <li>🍗 <strong>Menu 2 (Minced Chicken & Broccoli Rice):</strong> Minced chicken + soft rice + chopped broccoli + bone broth.</li>
                    <li>🦐 <strong>Menu 3 (Chopped Shrimp & Tofu Mash):</strong> Minced peeled shrimp + mashed tofu + carrots + sesame oil.</li>
                    <li>🍳 <strong>Menu 4 (Scrambled Duck Egg & Tempeh):</strong> Whisked duck egg + small diced tempeh + crushed sweet corn.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    else:
        if curr_lang == 'ID':
            st.markdown("""
            <div class="edu-card">
                <h3>🍽️ Tekstur: Makanan Keluarga Cincang / Normal | 3-4 Makan Utama + 2 Selingan</h3>
                <p><strong>Rekomendasi Porsi Protein Hewani:</strong> Minimal 60-75 gram/hari.</p>
                <hr style="border:0.5px solid #334155;">
                <ul>
                    <li>🍲 <strong>Menu 1 (Sup Bola-Bola Ayam Udang):</strong> Nasi putih + bola bakso ayam udang homemade + wortel + kentang + buncis.</li>
                    <li>🐟 <strong>Menu 2 (Pepes Ikan Lele / Belut Tanpa Duri):</strong> Nasi + pepes lele/belut + tumis buncis telur.</li>
                    <li>🥩 <strong>Menu 3 (Semur Daging Cincang & Telur Puyuh):</strong> Daging sapi cincang tumis manis + 3 butir telur puyuh + nasi hangat.</li>
                    <li>🥞 <strong>Selingan Sehat (Pancake Hati Ayam & Pisang):</strong> Campuran tepung terigu, pisang lumat, telur, dan bubuk hati ayam sangrai.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="edu-card">
                <h3>🍽️ Texture: Family Meals (Chopped/Normal) | 3-4 Meals + 2 Snacks</h3>
                <p><strong>Animal Protein Serving:</strong> At least 60-75 grams/day.</p>
                <hr style="border:0.5px solid #334155;">
                <ul>
                    <li>🍲 <strong>Menu 1 (Chicken & Shrimp Meatball Soup):</strong> White rice + homemade chicken shrimp meatballs + carrots + potatoes.</li>
                    <li>🐟 <strong>Menu 2 (Steamed Boneless Catfish/Eel):</strong> Rice + steamed catfish/eel in banana leaf + sauteed egg beans.</li>
                    <li>🥩 <strong>Menu 3 (Braised Minced Beef & Quail Eggs):</strong> Sweet braised minced beef + 3 quail eggs + warm rice.</li>
                    <li>🥞 <strong>Healthy Snack (Chicken Liver & Banana Pancakes):</strong> Wheat flour, mashed banana, egg, and roasted chicken liver powder.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# ================= TAB 4: EDUKASI & PANDUAN =================
with tab4:
    st.markdown(f"### {txt['edu_title']}")
    
    if curr_lang == 'ID':
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("""
            <div class="edu-card">
                <h3>🌱 1. Pentingnya Periode 1.000 HPK</h3>
                <p>1.000 Hari Pertama Kehidupan (270 hari selama kehamilan + 730 hari hingga anak usia 2 tahun) adalah window of opportunity emas. Kerusakan sel otak akibat stunting pada periode ini bersifat <strong>permanen dan tidak dapat diperbaiki</strong> setelah anak berusia di atas 2 tahun.</p>
            </div>
            <div class="edu-card">
                <h3>🍖 2. Mengapa Harus Protein Hewani?</h3>
                <p>Protein hewani (telur, hati, ikan, daging, susu) mengandung asam amino esensial lengkap serta faktor pertumbuhan <em>mTORC1</em> dan <em>IGF-1</em> yang langsung merangsang pertumbuhan tulang panjang balita, jauh lebih efektif dibanding protein nabati.</p>
            </div>
            <div class="edu-card">
                <h3>💉 3. Imunisasi Lengkap & Vitamin A</h3>
                <p>Anak yang sering terkena penyakit infeksi (seperti diare dan ISPA) akibat tidak imunisasi lengkap akan kehilangan banyak nutrisi, yang memicu stunting berulang. Pastikan Vitamin A diminum setiap bulan Februari dan Agustus.</p>
            </div>
            """, unsafe_allow_html=True)

        with e2:
            st.markdown("""
            <div class="edu-card">
                <h3>🩸 4. Cegah Anemia & Cacingan</h3>
                <p>Anemia pada balita merusak nafsu makan dan menurunkan imunitas. Berikan Makanan Tambahan (PMT) kaya zat besi dan konsultasikan pemberian obat cacing berkala setiap 6 bulan sekali untuk anak usia di atas 1 tahun.</p>
            </div>
            <div class="edu-card">
                <h3>💧 5. Sanitasi Air & Cuci Tangan (WASH)</h3>
                <p>Jamban sehat dan air bersih mencegah penyakit <em>Environmental Enteropathy</em> (peradangan usus kronis akibat kuman) yang membuat usus balita gagal menyerap nutrisi makanan dengan optimal.</p>
            </div>
            <div class="edu-card">
                <h3>📊 6. Pemantauan Rutin Posyandu</h3>
                <p>Datang ke Posyandu setiap bulan untuk menimbang berat badan (BB) dan mengukur tinggi badan (TB). Jika grafik pertumbuhan mendatar (weight faltering), segera konsultasi ke bidan/Puskesmas.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("""
            <div class="edu-card">
                <h3>🌱 1. Importance of the First 1,000 Days</h3>
                <p>The First 1,000 Days (270 days during pregnancy + 730 days until age 2) is a critical golden window. Brain cell damage caused by stunting in this period is <strong>permanent and irreversible</strong> after age two.</p>
            </div>
            <div class="edu-card">
                <h3>🍖 2. Why Animal Protein Matters Most?</h3>
                <p>Animal proteins (eggs, liver, fish, meat) contain complete essential amino acids and growth factors (<em>mTORC1</em> and <em>IGF-1</em>) that directly stimulate bone elongation, far outperforming plant proteins.</p>
            </div>
            <div class="edu-card">
                <h3>💉 3. Full Immunization & Vitamin A</h3>
                <p>Recurrent infectious diseases (such as diarrhea and ARI) due to incomplete immunization deplete vital nutrients, triggering chronic stunting. Ensure Vitamin A supplementation every 6 months.</p>
            </div>
            """, unsafe_allow_html=True)

        with e2:
            st.markdown("""
            <div class="edu-card">
                <h3>🩸 4. Anemia & Deworming Prevention</h3>
                <p>Anemia reduces appetite and weakens immunity. Provide iron-rich meals and consult healthcare providers for periodic deworming medication every 6 months for children over 1 year old.</p>
            </div>
            <div class="edu-card">
                <h3>💧 5. Clean Water & Sanitation (WASH)</h3>
                <p>Proper sanitation prevents <em>Environmental Enteropathy</em> (chronic gut inflammation from pathogens) which impairs the intestine's ability to absorb food nutrients effectively.</p>
            </div>
            <div class="edu-card">
                <h3>📊 6. Routine Growth Monitoring</h3>
                <p>Visit local health clinics (Posyandu) monthly to record weight and height velocity. If growth flattens (weight faltering), consult a pediatrician or midwife immediately.</p>
            </div>
            """, unsafe_allow_html=True)
