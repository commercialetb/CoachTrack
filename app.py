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
st.set_page_config(page_title="CoachTrack Oracle v18.6", layout="wide", page_icon="🏀")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Gestione API Groq
if "GROQ_API_KEY" in st.secrets:
    groq_key = st.secrets["GROQ_API_KEY"]
else:
    groq_key = st.sidebar.text_input("Groq API Key", type="password", help="Inserisci la chiave per attivare THE ORACLE")

client = Groq(api_key=groq_key) if groq_key else None

# =================================================================
# 2. DATABASE & SICUREZZA
# =================================================================
def init_db():
    conn = sqlite3.connect('coachtrack_v18_6.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, team_name TEXT)''')
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
# 3. LOGICA MANUALI E REPORT PDF
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
    pdf.set_font("Arial", 'I', 12)
    pdf.cell(0, 10, "Protocollo di Performance Intelligence & Prevenzione Infortuni", ln=True, align='C')
    pdf.ln(15)

    def add_sec(title, content):
        pdf.set_font("Arial", 'B', 14); pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 10, f"  {title}", ln=True, fill=True); pdf.ln(3)
        pdf.set_font("Arial", size=11); pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 7, content.encode('latin-1', 'ignore').decode('latin-1')); pdf.ln(8)

    add_sec("1. VIDEO TRACKING YOLO", "Analisi spaziale tramite Computer Vision (v8/v11). Monitorare lo spacing e il carico motorio.")
    add_sec("2. BIOMETRIA & COMPOSIZIONE", "Monitoraggio di Massa Muscolare, Body Fat, Idratazione e HRV per il calcolo del Power-to-Weight ratio.")
    add_sec("3. PROTOCOLLO INFORTUNI", "Semaforo Rosso: HRV basso + RPE alto = Stop immediato. Semaforo Giallo: Deidratazione = Carico ridotto.")
    add_sec("4. THE ORACLE AI", "Assistente per diete personalizzate e analisi tattiche avanzate basate su dati storici.")
    
    return pdf.output(dest='S').encode('latin-1')

def oracle_chat(prompt, context=""):
    if not client: return "⚠️ THE ORACLE offline. Inserisci API Key."
    full_p = f"Sei THE ORACLE (NBA Assist). Team: {st.session_state.get('team_name')}. Context: {context}. Rispondi: {prompt}"
    try:
        res = client.chat.completions.create(messages=[{"role":"user","content":full_p}], model="llama3-8b-8192")
        return res.choices[0].message.content
    except Exception as e: return f"Errore AI: {e}"

# =================================================================
# 4. GESTIONE AUTENTICAZIONE
# =================================================================
if 'logged_in' not in st.session_state: st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🏀 CoachTrack Oracle v18.6")
    auth_tabs = st.tabs(["Login", "Registra Squadra"])
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
                st.session_state.team_name = data[1]
                st.rerun()
            else: st.error("Credenziali errate")
    with auth_tabs[1]:
        nu = st.text_input("Nuovo Coach User")
        nt = st.text_input("Nome Team")
        np = st.text_input("Password Team", type="password")
        if st.button("Crea Franchise"):
            try:
                c = db_conn.cursor()
                c.execute("INSERT INTO users VALUES (?,?,?)", (nu, make_hashes(np), nt))
                db_conn.commit()
                st.success("Squadra registrata!")
            except: st.error("User già esistente.")
    st.stop()

# =================================================================
# 5. DASHBOARD & SIDEBAR BRANDING
# =================================================================
curr_user = st.session_state.username
team_name = st.session_state.team_name
logo_path = f"logo_{curr_user}.png"

st.sidebar.title(f"🏟️ {team_name}")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)

try:
    man_pdf = generate_branded_manual(team_name, logo_path if os.path.exists(logo_path) else None)
    st.sidebar.download_button("📘 Scarica Protocollo Ufficiale", man_pdf, f"Protocollo_{team_name}.pdf")
except: st.sidebar.warning("Errore PDF")

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

df_team = pd.read_sql_query(f"SELECT * FROM player_data WHERE owner = '{curr_user}'", db_conn)

tabs = st.tabs(["🎥 Video Tracking", "👤 Bio 360°", "⚔️ War Room", "📖 Playbook", "💬 The Oracle", "⚙️ Sync & Logo"])

# --- TAB 1: VIDEO ---
with tabs[0]:
    st.header("Analisi Video YOLO")
    if not YOLO_AVAILABLE: st.error("Libreria Ultralytics mancante.")
    else:
        v_file = st.file_uploader("Carica Match", type=['mp4'])
        if v_file and st.checkbox("Avvia Tracking"):
            with open("temp.mp4", "wb") as f: f.write(v_file.read())
            model = YOLO("yolov8n.pt")
            cap = cv2.VideoCapture("temp.mp4")
            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            bar = st.progress(0); fr_text = st.empty(); st_img = st.empty()
            curr_f = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                curr_f += 1
                if curr_f % 5 == 0:
                    bar.progress(min(curr_f/total_f, 1.0))
                    fr_text.text(f"Frame: {curr_f}/{total_f}")
                res = model.predict(frame, verbose=False, conf=0.3)
                st_img.image(res[0].plot(), channels="BGR", use_container_width=True)
            cap.release()

# --- TAB 2: BIO 360 & PREVENZIONE ---
with tabs[1]:
    st.header("Injury Prevention & Bio Analysis")
    if not df_team.empty:
        sel_p = st.selectbox("Seleziona Atleta", df_team['player_name'].unique())
        p = df_team[df_team['player_name'] == sel_p].iloc[-1]
        
        # Logica Semaforo
        col_s, col_m = st.columns([1, 3])
        if p['hrv'] < 45 or p['rpe'] > 8: 
            col_s.error("🔴 RISCHIO ELEVATO"); adv = "STOP E RECUPERO."
        elif p['hrv'] < 55 or p['water_perc'] < 58: 
            col_s.warning("🟡 ATTENZIONE"); adv = "CARICO RIDOTTO."
        else: 
            col_s.success("🟢 DISPONIBILE"); adv = "CARICO FULL."
        
        col_m.info(f"**Protocollo:** {adv}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Massa Muscolare", f"{p['muscle_mass']} kg")
        c2.metric("Body Fat", f"{p['body_fat']}%")
        c3.metric("HRV", f"{p['hrv']} ms")
        c4.metric("Acqua", f"{p['water_perc']}%")

        
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=[p['shot_efficiency'], p['hrv'], (11-p['rpe'])*10, p['water_perc']], 
                                      theta=['Tiro', 'Recupero', 'Freschezza', 'Idratazione'], fill='toself'))
        st.plotly_chart(fig)
    else: st.info("Inserisci dati nel Tab Sync.")

# --- TAB 3: WAR ROOM ---
with tabs[2]:
    if len(df_team['player_name'].unique()) >= 2:
        p1 = st.selectbox("Giocatore 1", df_team['player_name'].unique(), key="p1")
        p2 = st.selectbox("Giocatore 2", df_team['player_name'].unique(), key="p2")
        if st.button("Analizza Fit"):
            st.write(oracle_chat(f"Analizza sinergia tra {p1} e {p2}", df_team.to_string()))
        

# --- TAB 5: THE ORACLE ---
with tabs[4]:
    st.header("💬 The Oracle AI")
    with st.expander("💡 Prompt Suggeriti"):
        st.write("- 'Crea una dieta per [Atleta] per recuperare massa muscolare.'")
        st.write("- 'Chi è più a rischio infortunio oggi?'")
    if "msgs" not in st.session_state: st.session_state.msgs = []
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]): st.write(m["content"])
    if prompt := st.chat_input("Chiedi a The Oracle..."):
        st.session_state.msgs.append({"role":"user", "content":prompt})
        with st.chat_message("user"): st.write(prompt)
        r = oracle_chat(prompt, df_team.to_string())
        st.session_state.msgs.append({"role":"assistant", "content":r})
        with st.chat_message("assistant"): st.write(r)

# --- TAB 6: SYNC & LOGO SETTINGS ---
with tabs[5]:
    st.header("⚙️ Settings & Data Sync")
    c_l, c_d = st.columns(2)
    with c_l:
        st.subheader("Team Branding")
        up_logo = st.file_uploader("Carica Logo (PNG)", type=['png', 'jpg'])
        if up_logo:
            Image.open(up_logo).save(logo_path)
            st.success("Logo salvato! Ricarica l'app.")
            if st.button("Applica"): st.rerun()
    with c_d:
        st.subheader("Input Biometrico Manuale")
        with st.form("add_player"):
            name = st.text_input("Nome Atleta")
            w = st.number_input("Peso", 60.0, 150.0, 95.0)
            bf = st.number_input("Body Fat %", 3.0, 25.0, 10.0)
            mm = st.number_input("Massa Muscolare kg", 30.0, 100.0, 65.0)
            wa = st.number_input("Acqua %", 40.0, 80.0, 60.0)
            hr = st.number_input("HRV", 20, 150, 60)
            if st.form_submit_button("Salva Atleta"):
                c = db_conn.cursor()
                c.execute("INSERT INTO player_data (owner, player_name, timestamp, weight, body_fat, muscle_mass, water_perc, hrv) VALUES (?,?,?,?,?,?,?,?)",
                          (curr_user, name, datetime.now().strftime("%Y-%m-%d"), w, bf, mm, wa, hr))
                db_conn.commit()
                st.rerun()
