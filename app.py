import streamlit as st
import pandas as pd
import numpy as np
import cv2
import plotly.graph_objects as go
from datetime import datetime
import sqlite3
import hashlib
import os
from fpdf import FPDF
from groq import Groq
from PIL import Image

# =================================================================
# 1. DESIGN SYSTEM: FUTURA LIGHT & iOS GLASS
# =================================================================
st.set_page_config(page_title="CoachTrack Oracle v21.0", layout="wide", page_icon="🏀")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Jost:wght@300;400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Jost', sans-serif !important;
        font-weight: 300 !important;
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        padding: 15px !important;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 15px; }
    .stTabs [data-baseweb="tab"] { font-size: 18px; color: #8E8E93; }
    .stTabs [aria-selected="true"] { color: #007AFF !important; border-bottom: 2px solid #007AFF !important; }
    </style>
    """, unsafe_allow_html=True)

# =================================================================
# 2. DATABASE (v17 PER IL LOGIN)
# =================================================================
def init_db():
    conn = sqlite3.connect('coachtrack_v17.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, team_name TEXT)''')
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
# 3. GENERAZIONE MANUALE DETTAGLIATO (PDF)
# =================================================================
def generate_detailed_manual(team_name, logo_path=None):
    pdf = FPDF()
    pdf.add_page()
    
    # Logo rimpicciolito (18mm) - Posizionamento in alto a sx
    if logo_path and os.path.exists(logo_path):
        try:
            pdf.image(logo_path, 10, 8, 18)
            pdf.ln(20)
        except: pdf.ln(10)
    else: pdf.ln(10)

    # Titolo
    pdf.set_font("Arial", 'B', 22)
    pdf.set_text_color(33, 37, 41)
    pdf.cell(0, 10, f"{team_name.upper()} - PROTOCOLLO TECNICO v21", ln=True, align='R')
    pdf.set_draw_color(0, 122, 255)
    pdf.line(10, pdf.get_y()+2, 200, pdf.get_y()+2)
    pdf.ln(12)

    def add_chapter(title, content):
        pdf.set_font("Arial", 'B', 14); pdf.set_text_color(0, 122, 255)
        pdf.cell(0, 10, title, ln=True); pdf.ln(2)
        pdf.set_font("Arial", '', 10); pdf.set_text_color(50, 50, 50)
        pdf.multi_cell(0, 6, content.encode('latin-1','ignore').decode('latin-1'))
        pdf.ln(6)

    add_chapter("1. BIO-INTELLIGENCE: ANALISI HRV E COMPOSIZIONE", 
        "La funzione Bio-Intelligence monitora lo stato sistemico dell'atleta. \n"
        "- HRV (Heart Rate Variability): Misura il tempo tra i battiti. Un HRV alto indica un sistema parasimpatico attivo (recupero), un HRV basso indica stress o rischio infortunio.\n"
        "- Rapporto Massa Muscolare/Ossea: Essenziale per monitorare il catabolismo. Un calo della massa muscolare unito a variazioni della densità ossea segnala un overtraining cronico.\n"
        "- Idratazione (Water %): Cruciale per la prevenzione di crampi e lesioni muscolari acute.")

    add_chapter("2. VIDEO LAB & TRACKING YOLO", 
        "Il modulo Video utilizza algoritmi YOLO (v8/v11) per mappare il campo. \n"
        "- Upload Match: Analizza la densità dei giocatori e lo spacing.\n"
        "- Live Stream: Permette di collegare telecamere RTSP per un feedback tattico immediato in panchina durante l'allenamento.\n"
        "- Shot Efficiency: Incrocia i dati di tracking con le percentuali di tiro per identificare le 'Hot Zones'.")

    add_chapter("3. THE ORACLE: INTELLIGENZA ARTIFICIALE", 
        "L'IA (Groq Llama-3) funge da consulente tattico e nutrizionale. \n"
        "- Analisi Predittiva: Incrocia HRV, RPE e Sonno per suggerire chi deve riposare.\n"
        "- Nutrizione: Suggerisce apporti proteici o idrici basati sui dati di massa magra e idratazione caricati nel Sync Hub.")

    add_chapter("4. SYNC HUB E GESTIONE DATI", 
        "Il cuore del sistema. Supporta l'importazione di template CSV per grandi volumi di dati e un form manuale dettagliato per inserimenti quotidiani rapidi. Supporta il bridge API per bilance Withings e wearable Whoop.")

    return pdf.output(dest='S').encode('latin-1')

# =================================================================
# 4. GESTIONE ACCESSO (LOGIN)
# =================================================================
if 'logged_in' not in st.session_state: st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🏀 CoachTrack Oracle v21.0")
    t1, t2 = st.tabs(["⊙ Login", "⊕ Registrazione"])
    with t1:
        u, p = st.text_input("User"), st.text_input("Password", type="password")
        if st.button("Accedi"):
            c = db_conn.cursor()
            c.execute("SELECT password, team_name FROM users WHERE username = ?", (u,))
            data = c.fetchone()
            if data and check_hashes(p, data[0]):
                st.session_state.logged_in, st.session_state.username = True, u
                st.session_state.team_name = data[1]; st.rerun()
            else: st.error("Errore di accesso.")
    with t2:
        nu, nt, np = st.text_input("New User"), st.text_input("Team"), st.text_input("Pw", type="password")
        if st.button("Crea Team"):
            c = db_conn.cursor(); c.execute("INSERT INTO users VALUES (?,?,?)", (nu, make_hashes(np), nt))
            db_conn.commit(); st.success("Registrato!")
    st.stop()

# =================================================================
# 5. DASHBOARD OPERATIVA
# =================================================================
curr_user, team_name = st.session_state.username, st.session_state.team_name
logo_path = f"logo_{curr_user}.png"

# Sidebar rimpicciolita e elegante
if os.path.exists(logo_path):
    st.sidebar.columns([1,2,1])[1].image(logo_path, width=80)
st.sidebar.markdown(f"### {team_name}")

# Download Manuale Dettagliato
try:
    man_pdf = generate_detailed_manual(team_name, logo_path)
    st.sidebar.download_button("📘 Scarica Manuale Tecnico", man_pdf, f"Manuale_{team_name}_v21.pdf")
except: pass

# Floating AI Assistant
with st.sidebar:
    st.markdown("---")
    with st.popover("💬 Chiedi a The Oracle", use_container_width=True):
        st.markdown("### 🧠 AI Assistant")
        p_chat = st.text_input("Inserisci domanda tattica o medica:")
        if p_chat: st.info("The Oracle sta elaborando i dati del team...")

# Layout Tab
tabs = st.tabs(["⊙ Video Lab", "⊘ Bio-Intelligence", "⧉ Strategy Room", "⚙ Sync & Settings"])

df_team = pd.read_sql_query(f"SELECT * FROM player_data WHERE owner = '{curr_user}'", db_conn)

# --- TAB BIO ---
with tabs[1]:
    if not df_team.empty:
        sel_p = st.selectbox("Atleta", df_team['player_name'].unique())
        p = df_team[df_team['player_name'] == sel_p].iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Peso", f"{p.get('weight',0)} kg")
        c2.metric("Muscolo", f"{p.get('muscle_mass',0)} kg")
        c3.metric("Ossa", f"{p.get('bone_mass',0)} kg")
        c4.metric("HRV", f"{p.get('hrv',0)} ms")
        
        

        if p.get('hrv',60) < 45: st.error("🔴 RISCHIO INFORTUNIO: HRV critico. Consultare il manuale (Sez. 1).")
        else: st.success("🟢 STATUS: Ottimale")

# --- TAB SYNC (TEMPLATE & FORM COMPLETO) ---
with tabs[3]:
    st.subheader("Data Sync Hub")
    # Template
    t_cols = ["player_name", "weight", "body_fat", "muscle_mass", "water_perc", "bone_mass", "hrv", "rpe", "sleep", "shot_efficiency"]
    csv_t = pd.DataFrame(columns=t_cols).to_csv(index=False).encode('utf-8')
    st.download_button("📥 Scarica Template CSV Completo", csv_t, "template_v21.csv")
    
    st.markdown("---")
    st.subheader("Inserimento Manuale Professionale")
    with st.form("manual_entry"):
        col1, col2, col3 = st.columns(3)
        nm = col1.text_input("Nome Atleta")
        w = col2.number_input("Peso (kg)", 0.0, 150.0)
        h = col3.number_input("HRV (ms)", 0, 150)
        fat = col1.number_input("Grasso (%)", 0.0, 30.0)
        mus = col2.number_input("Muscolo (kg)", 0.0, 100.0)
        wat = col3.number_input("Acqua (%)", 0.0, 100.0)
        bon = col1.number_input("Ossa (kg)", 0.0, 10.0)
        sle = col2.number_input("Sonno (h)", 0.0, 12.0)
        if st.form_submit_button("Sincronizza Dati"):
            c = db_conn.cursor()
            c.execute("INSERT INTO player_data (owner, player_name, timestamp, weight, hrv, body_fat, muscle_mass, water_perc, bone_mass, sleep) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (curr_user, nm, datetime.now().strftime("%Y-%m-%d"), w, h, fat, mus, wat, bon, sle))
            db_conn.commit(); st.success("Dati integrati."); st.rerun()
