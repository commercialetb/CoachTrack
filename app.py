import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sqlite3
import json
from groq import Groq # Nuova integrazione

# Configurazione Groq - Inserisci la tua API Key nei segreti di Streamlit o come variabile d'ambiente
client = None
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# =================================================================
# DATABASE - SCHEMA INTEGRATO (v4.5)
# =================================================================
def init_db_v45():
    conn = sqlite3.connect('coachtrack_v45.db')
    c = conn.cursor()
    # Tabella Biometria Completa (Tutti i tuoi dati originali + nuovi)
    c.execute('''CREATE TABLE IF NOT EXISTS biometrics_full
                 (id INTEGER PRIMARY KEY, player_id TEXT, player_name TEXT, 
                  timestamp TEXT, weight_kg REAL, body_fat_pct REAL, 
                  muscle_mass_kg REAL, water_pct REAL, bone_mass_kg REAL, 
                  bmr_kcal REAL, rpe INTEGER, hrv REAL, measurement_type TEXT, 
                  notes TEXT)''')
    conn.commit()
    conn.close()

init_db_v45()

# =================================================================
# AI AGENT - ANALISI PREDITTIVA (GROQ)
# =================================================================
def get_ai_advice(player_data, context_type="diet"):
    if not client:
        return "⚠️ Configura GROQ_API_KEY per ricevere consigli dall'AI."
    
    prompt = f"""
    Sei un esperto di performance nel basket NBA. Analizza questi dati biometrici:
    {player_data.to_json()}
    
    Fornisci un consiglio breve e diretto (massimo 100 parole) su:
    {"Dieta e Integrazione" if context_type == "diet" else "Gestione Carico e Infortuni"}.
    Sii specifico con numeri (es. grammi di acqua o proteine).
    """
    
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# =================================================================
# MODULO BIOMETRICO EVOLUTO
# =================================================================
def render_biometric_v45():
    st.header("⚖️ Biometria Avanzata & AI Coach")
    
    with st.expander("➕ Inserimento Dati Completi (Original App)"):
        with st.form("bio_full"):
            c1, c2, c3 = st.columns(3)
            with c1:
                name = st.text_input("Nome Giocatore *")
                weight = st.number_input("Peso (kg)", 40.0, 150.0, 80.0)
                bf = st.number_input("Grasso Corporeo (%)", 3.0, 40.0, 12.0)
            with c2:
                muscle = st.number_input("Massa Muscolare (kg)", 20.0, 90.0, 40.0)
                water = st.number_input("Acqua (%)", 40.0, 75.0, 60.0)
                bmr = st.number_input("BMR (kcal)", 1200, 3500, 2000)
            with c3:
                hrv = st.number_input("HRV (ms)", 20, 150, 65)
                rpe = st.slider("RPE Sforzo (1-10)", 1, 10, 5)
                mtype = st.selectbox("Momento", ["Mattina", "Post-Allenamento", "Pre-Match"])
            
            notes = st.text_area("Note e Dieta attuale")
            if st.form_submit_button("💾 Salva e Analizza con AI"):
                conn = sqlite3.connect('coachtrack_v45.db')
                c = conn.cursor()
                c.execute("INSERT INTO biometrics_full (player_id, player_name, timestamp, weight_kg, body_fat_pct, muscle_mass_kg, water_pct, bmr_kcal, rpe, hrv, measurement_type, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                          (name, name, str(datetime.now()), weight, bf, muscle, water, bmr, rpe, hrv, mtype, notes))
                conn.commit()
                conn.close()
                st.success("Dati archiviati!")

    # Analisi AI
    conn = sqlite3.connect('coachtrack_v45.db')
    df = pd.read_sql_query("SELECT * FROM biometrics_full", conn)
    conn.close()

    if not df.empty:
        players = df['player_name'].unique()
        sel_p = st.selectbox("Seleziona Giocatore per AI Advice", players)
        p_data = df[df['player_name'] == sel_p].tail(5)

        col_ai1, col_ai2 = st.columns(2)
        with col_ai1:
            st.subheader("🍎 Piano Alimentare AI")
            if st.button("Genera Dieta"):
                with st.spinner("L'AI sta calcolando i macronutrienti..."):
                    advice = get_ai_advice(p_data, "diet")
                    st.info(advice)
        
        with col_ai2:
            st.subheader("🛡️ Prevenzione Infortuni AI")
            if st.button("Valuta Rischio"):
                with st.spinner("Analisi bio-meccanica in corso..."):
                    risk = get_ai_advice(p_data, "injury")
                    st.warning(risk)

# =================================================================
# NUOVI GRAFICI - RADAR CHART PERFORMANCE
# =================================================================
def render_performance_charts(df):
    st.subheader("📊 Analisi Multidimensionale")
    
    # Esempio Radar Chart per l'ultimo inserimento
    if not df.empty:
        last_entry = df.iloc[-1]
        categories = ['HRV', 'Muscolo', 'Acqua', 'Peso (inv)', 'RPE (inv)']
        
        # Normalizzazione fake per il grafico
        values = [
            last_entry['hrv']/100 * 10, 
            last_entry['muscle_mass_kg']/60 * 10,
            last_entry['water_pct']/70 * 10,
            (1 - (last_entry['weight_kg']/120)) * 10,
            (11 - last_entry['rpe']) 
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name=last_entry['player_name']))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap Storica della fatica
        st.subheader("🔥 Timeline Fatica (RPE)")
        fig2 = px.density_heatmap(df, x="timestamp", y="player_name", z="rpe", colorscale="Reds")
        st.plotly_chart(fig2, use_container_width=True)
