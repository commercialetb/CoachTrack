import streamlit as st
import pandas as pd
import numpy as np
import cv2
import plotly.graph_objects as go
from datetime import datetime
import sqlite3
import hashlib
import os
import time
from fpdf import FPDF
from groq import Groq
from PIL import Image

# =================================================================
# 1. CONFIGURAZIONE & DESIGN iOS GLASSMORPHISM
# =================================================================
st.set_page_config(page_title="CoachTrack Oracle v19.8", layout="wide", page_icon="🏀")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    [data-testid="stSidebar"] { background-color: rgba(255, 255, 255, 0.15); backdrop-filter: blur(20px); border-right: 1px solid rgba(255, 255, 255, 0.2); }
    [data-testid="stMetric"] { background: rgba(255, 255, 255, 0.35); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.4); border-radius: 20px; padding: 20px !important; }
    [data-testid="stMetricValue"] { color: #007AFF !important; font-size: 1.8rem !important; }
    [data-testid="stMetricLabel"] { color: #1d1d1f !important; font-weight: 600 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: rgba(0, 0, 0, 0.05); padding: 10px; border-radius: 15px; }
    .stTabs [aria-selected="true"] { background-color: rgba(255, 255, 255, 0.5) !important; color: #007AFF !important; }
    /* Chat Style Glass */
    .stChatMessage { background: rgba(255, 255, 255, 0.2); border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.3); }
    </style>
    """, unsafe_allow_html=True)

# =================================================================
# 2. CORE: DATABASE & IA (GROQ)
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

# Gestione Chiave API Groq
if "GROQ_API_KEY" in st.secrets:
    groq_key = st.secrets["GROQ_API_KEY"]
else:
    groq_key = st.sidebar.text_input("🔑 Groq API Key", type="password")

client = Groq(api_key=groq_key) if groq_key else None

def oracle_chat(prompt, context_data=""):
    if not client: return "⚠️ THE ORACLE è offline. Inserisci la Groq API Key per attivare l'intelligenza."
    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": f"Sei THE ORACLE, l'analista NBA del team {st.session_state.get('team_name')}. Analizza i dati biometrici e fornisci consigli da Head Coach. Sii conciso e professionale."},
                {"role": "user", "content": f"Dati Squadra: {context_data}\n\nRichiesta Coach: {prompt}"}
            ],
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Errore di connessione AI: {str(e)}"

def make_hashes(password): return hashlib.sha256(str.encode(password)).hexdigest()
def check_hashes(password, hashed_text): return make_hashes(password) == hashed_text

# =================================================================
# 3. MANUALE PDF (LOGO RIDOTTO)
# =================================================================
def generate_branded_manual(team_name, logo_path=None):
    pdf = FPDF()
    pdf.add_page()
    if logo_path and os.path.exists(logo_path):
        try: pdf.image(logo_path, 10, 8, 18); pdf.ln(15)
        except: pdf.ln(10)
    else: pdf.ln(10)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 10, f"{team_name.upper()} | Performance Protocol", ln=True, align='R')
    pdf.line(10, pdf.get_y()+2, 200, pdf.get_y()+2); pdf.ln(15)
    return pdf.output(dest='S').encode('latin-1')

# =================================================================
# 4. GESTIONE LOGIN & SESSION
# =================================================================
if 'logged_in' not in st.session_state: st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🏀 CoachTrack Oracle v19.8")
    t1, t2 = st.tabs(["⊙ Login", "⊕ Registra Team"])
    with t1:
        u, p = st.text_input("Username"), st.text_input("Password", type="password")
        if st.button("Accedi"):
            c = db_conn.cursor(); c.execute("SELECT password, team_name FROM users WHERE username = ?", (u,))
            data = c.fetchone()
            if data and check_hashes(p, data[0]):
                st.session_state.logged_in, st.session_state.username = True, u
                st.session_state.team_name = data[1]; st.rerun()
    with t2:
        nu, nt, np = st.text_input("New User"), st.text_input("Team"), st.text_input("Pw", type="password")
        if st.button("Registra"):
            c = db_conn.cursor(); c.execute("INSERT INTO users VALUES (?,?,?)", (nu, make_hashes(np), nt))
            db_conn.commit(); st.success("Registrato!")
    st.stop()

# =================================================================
# 5. DASHBOARD OPERATIVA
# =================================================================
curr_user, team_name = st.session_state.username, st.session_state.team_name
logo_path = f"logo_{curr_user}.png"

# Sidebar
if os.path.exists(logo_path): st.sidebar.columns([1,2,1])[1].image(logo_path, width=80)
st.sidebar.markdown(f"<h3 style='text-align:center;'>{team_name}</h3>", unsafe_allow_html=True)
if st.sidebar.button("🚪 Logout"): st.session_state.logged_in = False; st.rerun()

df_team = pd.read_sql_query(f"SELECT * FROM player_data WHERE owner = '{curr_user}'", db_conn)

tabs = st.tabs(["⊙ Video", "⊘ Bio-Intelligence", "⧉ Strategy", "🧠 The Oracle", "⚙ Sync & API"])

# --- TAB BIO-INTELLIGENCE ---
with tabs[1]:
    if not df_team.empty:
        sel_p = st.selectbox("Atleta", df_team['player_name'].unique())
        p = df_team[df_team['player_name'] == sel_p].iloc[-1]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Recupero (HRV)", f"{p.get('hrv',0)} ms")
        c2.metric("Massa Muscolare", f"{p.get('muscle_mass',0)} kg")
        c3.metric("Body Fat", f"{p.get('body_fat',0)}%")
        
        # Oracle Quick Insight (IA Integrata nel Widget)
        st.markdown("#### 🧠 Oracle Quick Insight")
        if st.button("Genera Analisi Istantanea"):
            with st.spinner("Llama-3 sta analizzando..."):
                insight = oracle_chat(f"Analizza brevemente lo stato di {sel_p} con HRV {p.get('hrv')} e Massa Muscolare {p.get('muscle_mass')}.")
                st.info(insight)
    else: st.info("Nessun dato presente nel Sync Hub.")

# --- TAB THE ORACLE (CHAT COMPLETA) ---
with tabs[3]:
    st.header("🧠 The Oracle AI Assistant")
    if "messages" not in st.session_state: st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if prompt := st.chat_input("Chiedi strategia, nutrizione o scouting..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            context = df_team.to_string() if not df_team.empty else "Nessun dato atleta caricato."
            response = oracle_chat(prompt, context)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- TAB SYNC HUB ---
with tabs[4]:
    st.header("🔌 Sync Hub & Identity")
    c_logo, c_api = st.columns(2)
    with c_logo:
        up_logo = st.file_uploader("Update Logo", type=['png', 'jpg'])
        if up_logo: Image.open(up_logo).save(logo_path); st.rerun()
    with c_api:
        st.markdown("#### Cloud API Connectors")
        st.text_input("Withings/Whoop Client ID", type="password")
        st.button("Sincronizza Dispositivi Cloud")

    with st.form("manual_entry"):
        st.subheader("Inserimento Manuale Bio-Dati")
        col_m1, col_m2 = st.columns(2)
        p_name = col_m1.text_input("Nome Atleta")
        p_hrv = col_m2.number_input("HRV (ms)", 20, 150, 60)
        p_fat = col_m1.number_input("Body Fat (%)", 3.0, 30.0, 10.0)
        p_musc = col_m2.number_input("Massa Muscolare (kg)", 30, 120, 70)
        if st.form_submit_button("Salva nel Database"):
            c = db_conn.cursor()
            c.execute("INSERT INTO player_data (owner, player_name, timestamp, hrv, body_fat, muscle_mass) VALUES (?,?,?,?,?,?)",
                      (curr_user, p_name, datetime.now().strftime("%Y-%m-%d"), p_hrv, p_fat, p_musc))
            db_conn.commit(); st.success(f"Dati di {p_name} registrati."); st.rerun()
