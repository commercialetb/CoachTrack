import streamlit as st
import pandas as pd
import numpy as np
import cv2
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sqlite3
import hashlib
import time
from fpdf import FPDF
from groq import Groq

# Tentativo di importazione YOLO per evitare crash se non ancora installato
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# =================================================================
# 1. DATABASE E SICUREZZA (MULTI-TENANT)
# =================================================================
def init_db():
    conn = sqlite3.connect('coachtrack_v17.db', check_same_thread=False)
    c = conn.cursor()
    # Tabella Utenti
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, team_name TEXT)''')
    # Tabella Dati Atleti
    c.execute('''CREATE TABLE IF NOT EXISTS player_data 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT, player_name TEXT, 
                  timestamp TEXT, hrv REAL, rpe INTEGER, shot_efficiency REAL, 
                  weight REAL, sleep REAL, video_notes TEXT)''')
    conn.commit()
    return conn

db_conn = init_db()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# =================================================================
# 2. LOGICA "THE ORACLE" AI & PDF
# =================================================================
st.set_page_config(page_title="CoachTrack Oracle v17.1", layout="wide", page_icon="🔮")

# Gestione API Key tramite Secrets o Sidebar
if "GROQ_API_KEY" in st.secrets:
    groq_key = st.secrets["GROQ_API_KEY"]
else:
    groq_key = st.sidebar.text_input("Groq API Key", type="password", help="Inserisci la chiave per attivare THE ORACLE")

client = Groq(api_key=groq_key) if groq_key else None

def oracle_chat(prompt, context=""):
    if not client:
        return "⚠️ THE ORACLE è offline. Inserisci la API Key nella sidebar."
    full_prompt = f"""
    Sei THE ORACLE, l'intelligenza artificiale tattica di una squadra NBA. 
    Dati squadra correnti: {context}.
    Istruzione: Rispondi in modo professionale, tecnico e conciso.
    Domanda del Coach: {prompt}
    """
    try:
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": full_prompt}],
            model="llama3-8b-8192"
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Errore di connessione con THE ORACLE: {str(e)}"

def create_pdf_report(title, content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    clean_text = content.encode('latin-1', 'ignore').decode('latin-1')
    pdf.multi_cell(0, 10, clean_text)
    return pdf.output(dest='S').encode('latin-1')

# =================================================================
# 3. SISTEMA DI LOGIN / REGISTRAZIONE
# =================================================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔮 CoachTrack Oracle v17.1")
    auth_mode = st.sidebar.selectbox("Accesso", ["Login", "Registra Nuova Squadra"])
    
    if auth_mode == "Login":
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        if st.button("Accedi"):
            c = db_conn.cursor()
            c.execute("SELECT password FROM users WHERE username = ?", (user,))
            data = c.fetchone()
            if data and check_hashes(pw, data[0]):
                st.session_state.logged_in = True
                st.session_state.username = user
                st.rerun()
            else:
                st.error("Credenziali errate.")
    else:
        new_user = st.text_input("Scegli Username")
        new_team = st.text_input("Nome della Squadra")
        new_pw = st.text_input("Scegli Password", type="password")
        if st.button("Registra Team"):
            try:
                c = db_conn.cursor()
                c.execute("INSERT INTO users (username, password, team_name) VALUES (?,?,?)", 
                          (new_user, make_hashes(new_pw), new_team))
                db_conn.commit()
                st.success("Squadra registrata con successo! Effettua il login.")
            except:
                st.error("Username già occupato.")
    st.stop()

# =================================================================
# 4. DASHBOARD OPERATIVA
# =================================================================
curr_user = st.session_state.username
st.sidebar.markdown(f"### 🏟️ Coach: {curr_user}")
if st.sidebar.button("Log out"):
    st.session_state.logged_in = False
    st.rerun()

# Recupero dati filtrati per utente
df_team = pd.read_sql_query(f"SELECT * FROM player_data WHERE owner = '{curr_user}'", db_conn)

tabs = st.tabs(["🎥 Video YOLO", "👤 Player 360°", "⚔️ War Room", "📖 Playbook AI", "💬 THE ORACLE", "⌚ Sync Hub"])

# --- TAB 1: VIDEO YOLO ---
with tabs[0]:
    st.header("🎥 Analisi Video & Tracking")
    if not YOLO_AVAILABLE:
        st.error("Libreria 'ultralytics' non installata. Esegui: pip install ultralytics")
    else:
        v_file = st.file_uploader("Carica Match", type=['mp4', 'mov'])
        y_model = st.selectbox("Versione YOLO", ["yolov8n.pt", "yolov11n.pt"])
        if v_file and st.checkbox("Avvia Tracking"):
            with open("temp_match.mp4", "wb") as f:
                f.write(v_file.read())
            model = YOLO(y_model)
            cap = cv2.VideoCapture("temp_match.mp4")
            st_frame = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                results = model.predict(frame, verbose=False, conf=0.3)
                st_frame.image(results[0].plot(), channels="BGR", use_container_width=True)
            cap.release()

# --- TAB 2: PLAYER 360° ---
with tabs[1]:
    st.header("👤 Scheda Atleta Individuale")
    if not df_team.empty:
        sel_p = st.selectbox("Seleziona Giocatore", df_team['player_name'].unique())
        p_data = df_team[df_team['player_name'] == sel_p].iloc[-1]
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Efficienza Tiro", f"{p_data['shot_efficiency']}%")
            st.metric("HRV (Recupero)", f"{p_data['hrv']} ms")
            fig_p = px.line(df_team[df_team['player_name']==sel_p], x='timestamp', y='hrv', title="Trend Recupero")
            st.plotly_chart(fig_p)
        with c2:
            if st.button("Genera Scouting Report AI"):
                report = oracle_chat(f"Crea uno scouting report per {sel_p}. HRV {p_data['hrv']}, Tiro {p_data['shot_efficiency']}%")
                st.markdown(report)
    else:
        st.info("Nessun dato. Usa il Tab Sync per aggiungere giocatori.")

# --- TAB 3: WAR ROOM (SINERGIE) ---
with tabs[2]:
    st.header("⚔️ War Room: Analisi Lineup")
    if len(df_team['player_name'].unique()) >= 2:
        p1 = st.selectbox("Giocatore A", df_team['player_name'].unique(), key="wa")
        p2 = st.selectbox("Giocatore B", df_team['player_name'].unique(), key="wb")
        
        if st.button("Simula Sinergia"):
            ctx = df_team[df_team['player_name'].isin([p1, p2])].to_string()
            st.info(oracle_chat(f"Qual è il fit tattico tra {p1} e {p2}?", ctx))
        
        # Radar di confronto
        d1 = df_team[df_team['player_name'] == p1].iloc[-1]
        d2 = df_team[df_team['player_name'] == p2].iloc[-1]
        
        fig_war = go.Figure()
        fig_war.add_trace(go.Scatterpolar(r=[d1['shot_efficiency'], d1['hrv'], d1['rpe']*10], theta=['Tiro', 'HRV', 'Fatica'], fill='toself', name=p1))
        fig_war.add_trace(go.Scatterpolar(r=[d2['shot_efficiency'], d2['hrv'], d2['rpe']*10], theta=['Tiro', 'HRV', 'Fatica'], fill='toself', name=p2))
        st.plotly_chart(fig_war)
    else:
        st.warning("Servono almeno 2 giocatori nel database.")

# --- TAB 4: PLAYBOOK AI ---
with tabs[3]:
    st.header("📖 Tactical Playbook")
    schema = st.selectbox("Schema", ["Pick & Roll", "Triangolo", "Zona 2-3", "Uscita Blocchi"])
    if st.button("Suggerisci Migliori Esecutori"):
        st.write(oracle_chat(f"Basandoti sui dati squadra, chi deve eseguire {schema}?", df_team.to_string()))
        

# --- TAB 5: THE ORACLE (CHATBOT) ---
with tabs[4]:
    st.header("💬 Talk to THE ORACLE")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            
    if prompt := st.chat_input("Chiedi a THE ORACLE della tua squadra..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        ans = oracle_chat(prompt, df_team.to_string())
        st.session_state.messages.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.markdown(ans)

# --- TAB 6: SYNC HUB ---
with tabs[5]:
    st.header("⌚ Sincronizzazione Roster")
    c_csv, c_man = st.columns(2)
    with c_csv:
        st.subheader("Importazione CSV")
        # Pulsante Download Template
        tmp = pd.DataFrame(columns=["player_name", "hrv", "rpe", "shot_efficiency", "weight", "sleep"])
        st.download_button("📥 Scarica Template CSV", tmp.to_csv(index=False).encode('utf-8'), "template_nba.csv")
        
        up = st.file_uploader("Carica File", type="csv")
        if up and st.button("Conferma Import"):
            df_new = pd.read_csv(up)
            df_new['owner'] = curr_user
            df_new['timestamp'] = datetime.now().strftime("%Y-%m-%d")
            df_new.to_sql('player_data', db_conn, if_exists='append', index=False)
            st.success("Dati caricati!")
            st.rerun()

    with c_man:
        st.subheader("Inserimento Manuale")
        with st.form("man_entry"):
            name_in = st.text_input("Nome")
            hrv_in = st.number_input("HRV", 20, 150, 65)
            eff_in = st.number_input("Shot %", 0, 100, 45)
            fat_in = st.slider("RPE (Fatica)", 1, 10, 5)
            if st.form_submit_button("Salva Giocatore"):
                c = db_conn.cursor()
                c.execute("INSERT INTO player_data (owner, player_name, timestamp, hrv, shot_efficiency, rpe) VALUES (?,?,?,?,?,?)",
                          (curr_user, name_in, datetime.now().strftime("%Y-%m-%d"), hrv_in, eff_in, fat_in))
                db_conn.commit()
                st.rerun()
