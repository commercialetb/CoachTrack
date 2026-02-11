import streamlit as st
import pandas as pd
import numpy as np
import cv2
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sqlite3
import hashlib
import os
import time
from fpdf import FPDF
from groq import Groq
from PIL import Image

# =================================================================
# 1. CONFIGURAZIONE E LIBRERIE AI
# =================================================================
st.set_page_config(page_title="CoachTrack Oracle v18.7", layout="wide", page_icon="🏀")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

if "GROQ_API_KEY" in st.secrets:
    groq_key = st.secrets["GROQ_API_KEY"]
else:
    groq_key = st.sidebar.text_input("Groq API Key", type="password", help="Inserisci la chiave per attivare THE ORACLE")

client = Groq(api_key=groq_key) if groq_key else None

# =================================================================
# 2. DATABASE & SICUREZZA (RIPRISTINO NOME ORIGINALE)
# =================================================================
def init_db():
    # Usiamo il nome database che avevi inizialmente per recuperare i tuoi dati
    conn = sqlite3.connect('coachtrack_v17.db', check_same_thread=False)
    c = conn.cursor()
    # Tabella Utenti (aggiunto team_name se mancante)
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, team_name TEXT)''')
    # Tabella Dati (assicuriamoci che esistano le nuove colonne biometriche)
    try:
        c.execute("ALTER TABLE player_data ADD COLUMN body_fat REAL")
        c.execute("ALTER TABLE player_data ADD COLUMN muscle_mass REAL")
        c.execute("ALTER TABLE player_data ADD COLUMN water_perc REAL")
    except:
        pass # Le colonne esistono già
    
    c.execute('''CREATE TABLE IF NOT EXISTS player_data 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT, player_name TEXT, 
                  timestamp TEXT, hrv REAL, rpe INTEGER, shot_efficiency REAL, 
                  weight REAL, sleep REAL, body_fat REAL, muscle_mass REAL, 
                  water_perc REAL, bone_mass REAL, video_notes TEXT)''')
    conn.commit()
    return conn

db_conn = init_db()

def make_hashes(password): return hashlib.sha256(str.encode(password)).hexdigest()
def check_hashes(password, hashed_text): return make_hashes(password) == hashed_text

# =================================================================
# 3. FUNZIONI PDF & AI
# =================================================================
def generate_branded_manual(team_name, logo_path=None):
    pdf = FPDF()
    pdf.add_page()
    if logo_path and os.path.exists(logo_path):
        try: pdf.image(logo_path, 10, 8, 30); pdf.ln(20)
        except: pass
    pdf.set_font("Arial", 'B', 22)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 15, f"{team_name.upper()} - ORACLE SYSTEM", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=11)
    content = "MANUALE TECNICO: Tracking Video, Biometria Avanzata e Prevenzione Infortuni."
    pdf.multi_cell(0, 10, content)
    return pdf.output(dest='S').encode('latin-1')

def oracle_chat(prompt, context=""):
    if not client: return "⚠️ THE ORACLE offline."
    full_p = f"Sei THE ORACLE (NBA Assist). Team: {st.session_state.get('team_name')}. Context: {context}. Rispondi breve: {prompt}"
    try:
        res = client.chat.completions.create(messages=[{"role":"user","content":full_p}], model="llama3-8b-8192")
        return res.choices[0].message.content
    except: return "Errore AI."

# =================================================================
# 4. GESTIONE AUTENTICAZIONE (CORRETTA)
# =================================================================
if 'logged_in' not in st.session_state: st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🏀 CoachTrack Oracle v18.7")
    auth_tabs = st.tabs(["Login Coach", "Registrazione Team"])
    
    with auth_tabs[0]:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Accedi"):
            c = db_conn.cursor()
            c.execute("SELECT password, team_name FROM users WHERE username = ?", (u,))
            data = c.fetchone()
            if data and check_hashes(p, data[0]):
                st.session_state.logged_in = True
                st.session_state.username = u
                st.session_state.team_name = data[1] if data[1] else "Mio Team"
                st.rerun()
            else: st.error("Dati non corretti o utente non trovato.")
            
    with auth_tabs[1]:
        nu = st.text_input("Scegli Username")
        nt = st.text_input("Nome Squadra (es. Lakers)")
        np = st.text_input("Scegli Password", type="password")
        if st.button("Registra Squadra"):
            if nu and nt and np:
                try:
                    c = db_conn.cursor()
                    c.execute("INSERT INTO users (username, password, team_name) VALUES (?,?,?)", 
                              (nu, make_hashes(np), nt))
                    db_conn.commit()
                    st.success("Registrazione completata! Ora puoi fare il Login.")
                except Exception as e:
                    st.error(f"Errore: L'utente esiste già o database occupato. {e}")
            else:
                st.warning("Riempi tutti i campi!")
    st.stop()

# =================================================================
# 5. DASHBOARD OPERATIVA
# =================================================================
curr_user = st.session_state.username
team_name = st.session_state.team_name
logo_path = f"logo_{curr_user}.png"

st.sidebar.title(f"🏟️ {team_name}")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)

if st.sidebar.button("🚪 Esci"):
    st.session_state.logged_in = False
    st.rerun()

# Recupero Dati
df_team = pd.read_sql_query(f"SELECT * FROM player_data WHERE owner = '{curr_user}'", db_conn)

tabs = st.tabs(["🎥 Video Tracking", "👤 Bio 360°", "⚔️ War Room", "💬 The Oracle", "⚙️ Sync & Logo"])

# --- TAB VIDEO ---
with tabs[0]:
    st.header("Analisi Video YOLO")
    if not YOLO_AVAILABLE: st.error("Libreria YOLO non caricata.")
    else:
        v_file = st.file_uploader("Carica Match", type=['mp4'])
        if v_file and st.checkbox("Esegui Tracking"):
            with open("temp.mp4", "wb") as f: f.write(v_file.read())
            model = YOLO("yolov8n.pt")
            cap = cv2.VideoCapture("temp.mp4")
            st_img = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                res = model.predict(frame, verbose=False, conf=0.3)
                st_img.image(res[0].plot(), channels="BGR", use_container_width=True)
            cap.release()

# --- TAB BIO ---
with tabs[1]:
    st.header("Injury Prevention")
    if not df_team.empty:
        sel_p = st.selectbox("Seleziona Atleta", df_team['player_name'].unique())
        p = df_team[df_team['player_name'] == sel_p].iloc[-1]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Massa Muscolare", f"{p.get('muscle_mass', 0)} kg")
        c2.metric("HRV", f"{p.get('hrv', 0)} ms")
        c3.metric("Body Fat", f"{p.get('body_fat', 0)}%")
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=[p['shot_efficiency'], p['hrv'], p.get('body_fat', 10)], 
                                      theta=['Tiro', 'Recupero', 'Grasso'], fill='toself'))
        st.plotly_chart(fig)
    else: st.info("Nessun dato trovato.")

# --- TAB THE ORACLE ---
with tabs[3]:
    st.header("💬 The Oracle AI")
    if "msgs" not in st.session_state: st.session_state.msgs = []
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]): st.write(m["content"])
    if prompt := st.chat_input("Chiedi a The Oracle..."):
        st.session_state.msgs.append({"role":"user", "content":prompt})
        with st.chat_message("user"): st.write(prompt)
        r = oracle_chat(prompt, df_team.to_string())
        st.session_state.msgs.append({"role":"assistant", "content":r})
        with st.chat_message("assistant"): st.write(r)

# --- TAB SETTINGS ---
with tabs[4]:
    st.header("⚙️ Settings")
    up_logo = st.file_uploader("Carica Logo Team", type=['png', 'jpg'])
    if up_logo:
        Image.open(up_logo).save(logo_path)
        st.success("Logo salvato! Ricarica per vedere le modifiche.")
    
    with st.form("manual"):
        nm = st.text_input("Nome Atleta")
        hr = st.number_input("HRV", 20, 150, 60)
        bf = st.number_input("Body Fat %", 5, 25, 10)
        if st.form_submit_button("Salva Dati"):
            c = db_conn.cursor()
            c.execute("INSERT INTO player_data (owner, player_name, timestamp, hrv, body_fat) VALUES (?,?,?,?,?)",
                      (curr_user, nm, datetime.now().strftime("%Y-%m-%d"), hr, bf))
            db_conn.commit()
            st.rerun()
