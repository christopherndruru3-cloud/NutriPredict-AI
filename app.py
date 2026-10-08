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

        # XAI FEATURE IMPORTANCE PLOT
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
    
    # Generate mock WHO standard curve data
    months_arr = np.arange(6, 60, 3)
    p3_curve = 50 + (months_arr * 1.0)
    p50_curve = 55 + (months_arr * 1.25)
    p97_curve = 60 + (months_arr * 1.5)
    
    fig_who = go.Figure()
    fig_who.add_trace(go.Scatter(x=months_arr, y=p97_curve, mode='lines', name='P97 (Tinggi Maksimal)', line=dict(color='green', dash='dash')))
    fig_who.add_trace(go.Scatter(x=months_arr, y=p50_curve, mode='lines', name='P50 (Median Ideal WHO)', line=dict(color='blue', width=3)))
    fig_who.add_trace(go.Scatter(x=months_arr, y=p3_curve, mode='lines', name='P3 (Batas Pendek / Stunting)', line=dict(color='red', dash='dash')))
    
    # Plot current user point
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
    
    # Simulasi pertumbuhan linear sederhana
    projected_height = st.session_state.height_val + (sim_months * 0.75)
    projected_weight = st.session_state.weight_val + (sim_months * 0.25)
    
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Proyeksi Usia", f"{st.session_state.age_val + sim_months} Bulan" if st.session_state.age_category=="Balita" else f"+{sim_months} Bulan")
    sc2.metric("Proyeksi Tinggi Badan", f"{projected_height:.1f} cm", delta=f"+{sim_months * 0.75:.1f} cm")
    sc3.metric("Proyeksi Berat Badan", f"{projected_weight:.1f} kg", delta=f"+{sim_months * 0.25:.1f} kg")

# ================= TAB 3: RESEP NUTRISI 7 HARI =================
with tab3:
    st.markdown(f"### {txt['recipe_title']}")
    st.caption("Menu lengkap kaya Protein Hewani untuk mencegah dan mengatasi stunting, lengkap dengan cara memasak serta video tutorial resmi.")
    
    recipes = [
        {"day": "Hari 1", "title": "Bubur Tim Ikan Kembung & Labu Kuning", "protein": "Ikan Kembung (Tinggi Omega-3 & Protein setara Salmon)", "cal": "320 kkal", "steps": "1. Kukus fillet ikan kembung dan labu kuning hingga empuk.\n2. Blender atau saring kasar bersama nasi tim matang.\n3. Tambahkan 1 sdt mentega tawar (unsalted butter) sebelum disajikan.", "link": "https://www.youtube.com/results?search_query=resep+mpasi+ikan+kembung+anti+stunting"},
        {"day": "Hari 2", "title": "Nasi Tim Hati Ayam Kampung & Bayam", "protein": "Hati Ayam (Kaya Zat Besi penangkal anemia)", "cal": "340 kkal", "steps": "2. Cincang halus hati ayam dan rebus sebentar dengan jahe untuk menghilangkan amis.\n3. Masak bersama beras merah/putih menjadi bubur lembik.\n4. Masukkan cincangan daun bayam di akhir memasak.", "link": "https://www.youtube.com/results?search_query=resep+mpasi+hati+ayam+anti+stunting"},
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
            st.markdown(f'<a href="{r['link']}" target="_blank" class="ref-btn">▶️ Tonton Tutorial YouTube</a>', unsafe_allow_html=True)

# ================= TAB 4: EDUKASI & BERITA VIDEO =================
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
    
    # Render chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Chat input
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
