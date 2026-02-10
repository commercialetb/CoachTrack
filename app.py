import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sqlite3
import os
from fpdf import FPDF
from groq import Groq

# =================================================================
# CONFIGURAZIONE GROQ & PDF
# =================================================================
# Inserisci qui la tua API KEY o usa st.secrets["GROQ_API_KEY"]
GROQ_API_KEY = st.sidebar.text_input("Groq API Key", type="password")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

class DietaPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'CoachTrack Elite - Piano Nutrizionale AI', 0, 1, 'C')
        self.ln(10)

# =================================================================
# DATABASE - SCHEMA FULL BIOMETRICS (v5.0)
# =================================================================
def init_db():
    conn = sqlite3.connect('coachtrack_v5.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS biometrics_full
                 (id INTEGER PRIMARY KEY, player_name TEXT, timestamp TEXT, 
                  weight REAL, fat REAL, muscle REAL, water REAL, bone REAL, 
                  bmr REAL, hrv REAL, rpe INTEGER, notes TEXT, ai_diet TEXT)''')
    conn.commit()
    conn.close()

init_db()

# =================================================================
# FUNZIONI AI (GROQ)
# =================================================================
def genera_consiglio_ai(data, mode="diet"):
    if not client: return "Inserisci la Groq API Key per attivare l'assistente."
    
    prompt = f"""
    Sei un nutrizionista e performance coach NBA. Analizza questi dati: {data}.
    Genera un piano alimentare {'personalizzato' if mode=='diet' else 'di recupero'} 
    specificando: calorie totali, grammi di proteine, carboidrati e grassi. 
    Includi suggerimenti sugli integratori (es. Creatina, Omega-3). 
    Sii tecnico e preciso.
    """
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
    )
    return completion.choices[0].message.content

# =================================================================
# INTERFACCIA STREAMLIT
# =================================================================
st.set_page_config(page_title="CoachTrack Elite v5", layout="wide")
st.title("🏀 CoachTrack Elite v5.0")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "⚖️ Bio & Dieta AI", "🎥 Video Analysis", "💬 AI Chat Coach"])

with tab2:
    st.header("⚖️ Full Biometrics & AI Nutrition")
    
    with st.expander("📝 Inserimento Dati Originali (v3.2)"):
        with st.form("form_bio"):
            col1, col2, col3 = st.columns(3)
            with col1:
                name = st.text_input("Atleta")
                w = st.number_input("Peso (kg)", 50.0, 150.0, 85.0)
                f = st.number_input("Grasso (%)", 5.0, 30.0, 12.0)
            with col2:
                m = st.number_input("Massa Muscolare (kg)", 30.0, 100.0, 45.0)
                h = st.number_input("HRV (ms)", 20, 150, 60)
                wat = st.number_input("Acqua (%)", 40.0, 80.0, 60.0)
            with col3:
                bmr = st.number_input("BMR (kcal)", 1500, 4000, 2200)
                rpe = st.slider("RPE (Fatica)", 1, 10, 5)
                note = st.text_area("Note")
            
            if st.form_submit_button("Analizza e Salva"):
                ai_resp = genera_consiglio_ai({"peso": w, "grasso": f, "muscolo": m, "hrv": h, "bmr": bmr}, "diet")
                conn = sqlite3.connect('coachtrack_v5.db')
                conn.cursor().execute("INSERT INTO biometrics_full (player_name, timestamp, weight, fat, muscle, water, bmr, hrv, rpe, notes, ai_diet) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                      (name, str(datetime.now()), w, f, m, wat, bmr, h, rpe, note, ai_resp))
                conn.commit()
                st.success("Dati e Dieta generati con successo!")

    # Visualizzazione Dati e Download PDF
    conn = sqlite3.connect('coachtrack_v5.db')
    df = pd.read_sql_query("SELECT * FROM biometrics_full ORDER BY id DESC", conn)
    
    if not df.empty:
        st.subheader("Dati Recenti")
        p_name = st.selectbox("Seleziona Giocatore", df['player_name'].unique())
        latest = df[df['player_name'] == p_name].iloc[0]
        
        c1, c2 = st.columns([1, 2])
        with c1:
            # Grafico Radar
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[latest['hrv']/10, latest['muscle']/10, latest['water']/10, 11-latest['rpe'], (150-latest['weight'])/10],
                theta=['HRV', 'Muscolo', 'Acqua', 'Recupero', 'Peso-Target'],
                fill='toself'
            ))
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.markdown("### 🥗 Piano Alimentare AI")
            st.write(latest['ai_diet'])
            
            # Generazione PDF
            if st.button("📥 Scarica Dieta PDF"):
                pdf = DietaPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(0, 10, f"Atleta: {latest['player_name']}", ln=1)
                pdf.cell(0, 10, f"Data: {latest['timestamp']}", ln=1)
                pdf.ln(5)
                pdf.multi_cell(0, 10, latest['ai_diet'].replace('•', '-'))
                pdf_output = f"dieta_{latest['player_name']}.pdf".replace(" ", "_")
                pdf.output(pdf_output)
                
                with open(pdf_output, "rb") as f:
                    st.download_button("Clicca qui per il PDF", f, file_name=pdf_output)

with tab4:
    st.header("💬 AI Tactical Chat")
    st.write("Chiedi all'AI consigli sulla rotazione dei giocatori o analisi del carico.")
    user_q = st.text_input("Esempio: Chi è il giocatore più stanco? Che allenamento fare per chi ha HRV basso?")
    if user_q and client:
        with st.spinner("L'AI sta analizzando il database..."):
            db_context = df.tail(10).to_string()
            ans = client.chat.completions.create(
                messages=[{"role": "system", "content": "Sei un assistente tattico. Usa questi dati: " + db_context},
                          {"role": "user", "content": user_q}],
                model="llama3-8b-8192"
            )
            st.chat_message("assistant").write(ans.choices[0].message.content)

# DASHBOARD GRAFICA IN TAB1
with tab1:
    if not df.empty:
        st.subheader("🔥 Heatmap Fatica Team")
        fig_heat = px.density_heatmap(df, x="timestamp", y="player_name", z="rpe", text_auto=True, colorscale="Reds")
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.subheader("📈 Trend Peso vs Massa Muscolare")
        fig_trend = px.line(df, x="timestamp", y=["weight", "muscle"], color="player_name")
        st.plotly_chart(fig_trend, use_container_width=True)
