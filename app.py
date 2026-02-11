import streamlit as st
import pandas as pd
import numpy as np
import cv2
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sqlite3
import time
from fpdf import FPDF
from groq import Groq

# =================================================================
# 1. DATABASE E CONFIGURAZIONE
# =================================================================
st.set_page_config(page_title="CoachTrack Oracle v10", layout="wide", page_icon="🔮")

def init_db():
    conn = sqlite3.connect('coachtrack_oracle_v10.db', check_same_thread=False)
    c = conn.cursor()
    # Supporto per metriche v3.2 + nuove funzioni AI
    c.execute('''CREATE TABLE IF NOT EXISTS player_data 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, player_name TEXT, timestamp TEXT,
                  weight REAL, hrv REAL, rpe INTEGER, sleep REAL, 
                  shot_efficiency REAL, mental_state TEXT, match_report TEXT)''')
    conn.commit()
    return conn

db_conn = init_db()

# Gestione API Key
if "GROQ_API_KEY" in st.secrets:
    groq_key = st.secrets["GROQ_API_KEY"]
else:
    groq_key = st.sidebar.text_input("Groq API Key", type="password")

client = Groq(api_key=groq_key) if groq_key else None

# =================================================================
# 2. LOGICA "THE WHISPERER" & "PLAYBOOK"
# =================================================================

def get_whisperer_advice(player_name, hrv, rpe):
    """AI Whisperer: Consigli live basati sulla fatica."""
    if not client: return "Silenzio (API Key mancante)."
    prompt = f"Sei un assistente NBA. Il giocatore {player_name} ha HRV {hrv} e fatica {rpe}. Dammi un consiglio di un riga per il coach durante la partita."
    res = client.chat.completions.create(messages=[{"role":"user","content":prompt}], model="llama3-8b-8192")
    return res.choices[0].message.content

def automated_playbook_match(play_type, player_stats):
    """Abbina il miglior giocatore allo schema richiesto."""
    if not client: return "Calcolo playbook non disponibile."
    prompt = f"Analizza questo schema: {play_type}. Chi tra questi giocatori è più adatto? Dati: {player_stats}"
    res = client.chat.completions.create(messages=[{"role":"user","content":prompt}], model="llama3-8b-8192")
    return res.choices[0].message.content

# =================================================================
# 3. INTERFACCIA PRINCIPALE
# =================================================================

tabs = st.tabs(["📢 The Whisperer (Live)", "📖 Playbook Intelligente", "👤 Report Singoli", "⌚ Wearable & Bio", "🎯 Shot Charts"])

# --- TAB 1: THE WHISPERER (Live Assistant) ---
with tabs[0]:
    st.header("📢 The Whisperer: Live Assistant Coach")
    col_v, col_a = st.columns([2, 1])
    
    with col_v:
        st.subheader("Live Feed Analysis")
        # Supporto per video live o caricamento
        uv = st.file_uploader("Carica Match in corso", type=['mp4'])
        if uv: st.video(uv)
        
    with col_a:
        st.subheader("🔔 Avvisi in Tempo Reale")
        df = pd.read_sql_query("SELECT * FROM player_data", db_conn)
        if not df.empty:
            for _, row in df.tail(3).iterrows():
                advice = get_whisperer_advice(row['player_name'], row['hrv'], row['rpe'])
                st.warning(f"**{row['player_name']}**: {advice}")
        else:
            st.info("In attesa di dati biometrici per generare avvisi...")

# --- TAB 2: PLAYBOOK INTELLIGENTE ---
with tabs[1]:
    st.header("📖 Automated Playbook Optimizer")
    play = st.selectbox("Seleziona Schema", ["Pick & Roll Centrale", "Triangolo", "Isolamento Post-Basso", "Uscita dai Blocchi (3PT)"])
    
    if st.button("Trova il miglior esecutore"):
        if not df.empty:
            context = df[['player_name', 'hrv', 'shot_efficiency']].to_string()
            recommendation = automated_playbook_match(play, context)
            st.success(f"**Suggerimento Tattico:** {recommendation}")
            
        else:
            st.error("Carica i dati della squadra nel tab Wearable.")

# --- TAB 3: REPORT SINGOLI (Approfonditi) ---
with tabs[2]:
    st.header("👤 Deep Player Scouting & Report")
    sel_p = st.selectbox("Seleziona Giocatore per Report Dettagliato", df['player_name'].unique() if not df.empty else ["Nessun dato"])
    
    if st.button("Genera Report 360°"):
        with st.spinner("Analisi bio-meccanica e tattica in corso..."):
            prompt = f"Crea un report dettagliato per {sel_p}. Includi: 1. Efficienza al tiro, 2. Tenuta difensiva, 3. Rischio infortuni."
            full_report = get_ai_insight(prompt) # Funzione definita precedentemente
            st.session_state.last_report = full_report
            st.markdown(full_report)

# --- TAB 4: WEARABLE HUB (API CUSTOM) ---
with tabs[3]:
    st.header("⌚ Wearable API Bridge")
    col_api, col_data = st.columns(2)
    with col_api:
        st.subheader("Configurazione API")
        endpoint = st.text_input("Wearable Endpoint (Whoop/Catapult/Custom)")
        token = st.text_input("Access Token", type="password")
        if st.button("Sincronizza Ora"):
            st.info("🔄 Sincronizzazione con API Wearable in corso...")
    with col_data:
        st.subheader("Input Manuale (Backup)")
        # Ripristino input biometrici v3.2
        with st.form("bio_form"):
            n = st.text_input("Nome")
            h = st.number_input("HRV", 20, 150, 60)
            r = st.slider("RPE", 1, 10, 5)
            s = st.number_input("Shot Efficiency (%)", 0, 100, 45)
            if st.form_submit_button("Salva Dati"):
                cur = db_conn.cursor()
                cur.execute("INSERT INTO player_data (player_name, timestamp, hrv, rpe, shot_efficiency) VALUES (?,?,?,?,?)", 
                            (n, datetime.now().strftime("%Y-%m-%d"), h, r, s))
                db_conn.commit()
                st.rerun()

# --- TAB 5: SHOT CHARTS ---
with tabs[4]:
    st.header("🎯 Analisi Spaziale del Tiro")
    # Generazione Heatmap come discusso
    df_shots = pd.DataFrame({'x': np.random.uniform(0, 50, 50), 'y': np.random.uniform(0, 47, 50), 'segno': np.random.choice([0,1], 50)})
    fig = px.density_heatmap(df_shots, x='x', y='y', z='segno', title="Mappa Efficienza Tiro", color_continuous_scale="Hot")
    st.plotly_chart(fig, use_container_width=True)
