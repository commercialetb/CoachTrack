# =================================================================
# COACHTRACK ELITE AI v4.0 - PRO EDITION
# =================================================================
import sys
import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from pathlib import Path
import sqlite3
import json
import os
from collections import deque
import cv2
from fpdf import FPDF  # Assicurati di installare: pip install fpdf2
from groq import Groq  # pip install groq

# =================================================================
# CONFIGURAZIONE GROQ & PDF [web:1]
# =================================================================
GROQ_API_KEY = st.sidebar.text_input("Groq API Key", type="password")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None [file:1]

class DietaPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'CoachTrack Elite - Piano Nutrizionale AI', 0, 1, 'C')
        self.ln(10) [file:1]

class PDFReport(FPDF):  # Classe esistente mantenuta per video reports
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'CoachTrack Elite - Match Report', 0, 1, 'C')
        self.ln(5)

# =================================================================
# DATABASE - SCHEMA FULL BIOMETRICS (v5.0) [file:1]
# =================================================================
def init_db():
    conn = sqlite3.connect('coachtrack_v5.db')
    c = conn.cursor()
    # Nuova tabella full biometrics
    c.execute('''CREATE TABLE IF NOT EXISTS biometrics_full
                 (id INTEGER PRIMARY KEY, player_name TEXT, timestamp TEXT, 
                  weight REAL, fat REAL, muscle REAL, water REAL, bone REAL, 
                  bmr REAL, hrv REAL, rpe INTEGER, notes TEXT, ai_diet TEXT)''')
    # Tabella esistente biometrics (per compatibilità ACWR/video)
    c.execute('''CREATE TABLE IF NOT EXISTS biometrics
                 (id INTEGER PRIMARY KEY, playerid TEXT, playername TEXT, timestamp TEXT, 
                  weightkg REAL, rpe INTEGER, hrv REAL, fatiguescore REAL, notes TEXT)''')
    # Tabella video clips
    c.execute('''CREATE TABLE IF NOT EXISTS videoclips
                 (id INTEGER PRIMARY KEY, filename TEXT, eventtype TEXT, timestamp TEXT, 
                  playerid TEXT, duration REAL)''')
    conn.commit()
    conn.close()

# =================================================================
# FUNZIONI AI (GROQ) [file:1]
# =================================================================
def genera_consiglio_ai(data, mode="diet"):
    if not client: 
        return "Inserisci la Groq API Key per attivare l'assistente."
    
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
    return completion.choices[0].message.content [file:1]

# =================================================================
# FUNZIONI DB ESISTENTI (adattate per v5)
# =================================================================
def savebiodbdata(data):
    conn = sqlite3.connect('coachtrack_v5.db')
    c = conn.cursor()
    c.execute("INSERT INTO biometrics (playerid, playername, timestamp, weightkg, rpe, hrv, fatiguescore, notes) VALUES (?,?,?,?,?,?,?,?)",
              (data['playerid'], data['playername'], str(datetime.now()), data['weightkg'], data['rpe'], data['hrv'], data['fatiguescore'], data['notes']))
    conn.commit()
    conn.close()

def loadbiodb():
    conn = sqlite3.connect('coachtrack_v5.db')
    df = pd.read_sql_query("SELECT * FROM biometrics ORDER BY id DESC", conn)
    conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Fallback/aggiungi da biometrics_full se vuoto
    df_full = pd.read_sql_query("SELECT player_name as playername, timestamp, weight as weightkg, hrv, rpe, notes FROM biometrics_full ORDER BY id DESC LIMIT 100", conn)
    if not df_full.empty and df.empty:
        df = df_full
    return df [file:1]

init_db()

# =================================================================
# FUNZIONI VIDEO & ALTRI (invariate da app-6.py)
# =================================================================
PDF_AVAILABLE = True  # Assumiamo installato
logging.basicConfig(level=logging.INFO)

def savevideoclip(frames, fps, filename):
    if not frames: return False
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    return True

# ... (altre funzioni come generatereport, CVAIPipeline mock, drawcourtfig, addanalyticstab, renderbiometricmodule - copiate da app-6.py originale, omesse per brevità)

# =================================================================
# INTERFACCIA STREAMLIT AGGIORNATA v5.0
# =================================================================
st.set_page_config(page_title="CoachTrack Elite v5", layout="wide")
st.title("🏀 CoachTrack Elite v5.0") [file:1]

# Login sidebar (esistente)
if 'loggedin' not in st.session_state:
    st.session_state.loggedin = False

with st.sidebar:
    st.title("CoachTrack PRO")
    if not st.session_state.loggedin:
        u = st.text_input("User")
        p = st.text_input("Pass", type="password")
        if st.button("Login"):
            if u == "admin" and p == "admin":
                st.session_state.loggedin = True
                st.rerun()
    else:
        st.success("Loggato come Admin")
        if st.button("Logout"):
            st.session_state.loggedin = False
            st.rerun()

if st.session_state.loggedin:
    # Nuovi tabs v5 [file:1]
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "⚖️ Bio & Dieta AI", "🎥 Video Analysis", "💬 AI Chat Coach"])

    df = loadbiodb()  # Carica dati condivisi

    with tab1:
        st.header("📊 Dashboard Team")
        if not df.empty:
            st.subheader("🔥 Heatmap Fatica Team")
            fig_heat = px.density_heatmap(df, x="timestamp", y="playername", z="rpe", text_auto=True, colorscale="Reds")
            st.plotly_chart(fig_heat, use_container_width=True)
            
            st.subheader("📈 Trend Peso vs HRV")
            fig_trend = px.line(df, x="timestamp", y=["weightkg", "hrv"], color="playername")
            st.plotly_chart(fig_trend, use_container_width=True) [file:1]

    with tab2:
        st.header("⚖️ Full Biometrics & AI Nutrition")
        
        with st.expander("📝 Inserimento Dati Originali (v5.0)"):
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
                    data = {"peso": w, "grasso": f, "muscolo": m, "hrv": h, "bmr": bmr}
                    ai_resp = genera_consiglio_ai(data, "diet")
                    conn = sqlite3.connect('coachtrack_v5.db')
                    c = conn.cursor()
                    c.execute("INSERT INTO biometrics_full (player_name, timestamp, weight, fat, muscle, water, bmr, hrv, rpe, notes, ai_diet) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                              (name, str(datetime.now()), w, f, m, wat, bmr, h, rpe, note, ai_resp))
                    # Salva anche in biometrics per compatibilità
                    import hashlib
                    pid = hashlib.md5(name.encode()).hexdigest()[:8]
                    savebiodbdata({'playerid': pid, 'playername': name, 'weightkg': w, 'rpe': rpe*60, 'hrv': h, 'fatiguescore': rpe, 'notes': note})  # Approx sRPE
                    conn.commit()
                    conn.close()
                    st.success("Dati e Dieta generati con successo!")
                    st.rerun()

        # Visualizzazione Dati e Download PDF
        conn = sqlite3.connect('coachtrack_v5.db')
        df_full = pd.read_sql_query("SELECT * FROM biometrics_full ORDER BY id DESC", conn)
        
        if not df_full.empty:
            st.subheader("Dati Recenti")
            p_name = st.selectbox("Seleziona Giocatore", df_full['player_name'].unique())
            latest = df_full[df_full['player_name'] == p_name].iloc[0]
            
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
                        st.download_button("Clicca qui per il PDF", f.read(), file_name=pdf_output)
                    os.remove(pdf_output)  # Pulizia [file:1]

    with tab3:
        # Video Analysis (ex tab2)
        # ... Inserisci qui addcomputervisiontab() da app-6.py

    with tab4:
        st.header("💬 AI Tactical Chat")
        st.write("Chiedi all'AI consigli sulla rotazione dei giocatori o analisi del carico.")
        user_q = st.text_input("Esempio: Chi è il giocatore più stanco? Che allenamento fare per chi ha HRV basso?")
        if user_q and client:
            with st.spinner("L'AI sta analizzando il database..."):
                db_context = df.tail(10).to_string()
                ans = client.chat.completions.create(
                    messages=[{"role": "system", "content": "Sei un assistente tattico per basket indoor. Usa questi dati: " + db_context},
                              {"role": "user", "content": user_q}],
                    model="llama3-8b-8192"
                )
                st.chat_message("assistant").write(ans.choices[0].message.content) [file:1]

    # Sposta Biometria ACWR e Tattica in espander o sub-tabs se serve, ma integrati in dashboard

else:
    st.info("Effettua il login dalla sidebar per accedere alla suite PRO.")

# Fine app [file:1]
