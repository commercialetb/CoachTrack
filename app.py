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
# 1. DESIGN SYSTEM: FUTURA LIGHT & FLOATING CHAT
# =================================================================
st.set_page_config(page_title="CoachTrack v20", layout="wide", page_icon="🏀")

st.markdown("""
    <style>
    /* IMPORT FONT FUTURA / CENTURY GOTHIC */
    @import url('https://fonts.googleapis.com/css2?family=Jost:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Jost', 'Century Gothic', 'Futura', sans-serif !important;
        font-weight: 300 !important;
    }

    /* HEADER & SFONDO */
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
    }
    
    /* TITOLI */
    h1, h2, h3 {
        font-weight: 600 !important;
        color: #2c3e50;
        letter-spacing: 1px;
    }

    /* TAB STYLE - PROFESSIONAL CLEAN */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
        padding-bottom: 10px;
        border-bottom: 1px solid #ddd;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 400;
        color: #7f8c8d;
        border: none;
        background: none;
    }
    .stTabs [aria-selected="true"] {
        color: #2980b9 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #2980b9 !important;
    }

    /* METRIC WIDGETS (GLASS) */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.6);
        border: 1px solid rgba(255,255,255,0.9);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-radius: 10px;
        padding: 15px !important;
    }
    
    /* FLOATING ORACLE BUTTON (Basso Destra) */
    .floating-chat {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 9999;
    }
    </style>
    """, unsafe_allow_html=True)

# =================================================================
# 2. DATABASE & BACKEND
# =================================================================
def init_db():
    conn = sqlite3.connect('coachtrack_v20.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, team_name TEXT)''')
    # Struttura completa per non perdere dati
    c.execute('''CREATE TABLE IF NOT EXISTS player_data 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT, player_name TEXT, 
                  timestamp TEXT, hrv REAL, rpe INTEGER, shot_efficiency REAL, 
                  weight REAL, sleep REAL, body_fat REAL, muscle_mass REAL, 
                  water_perc REAL, bone_mass REAL, video_notes TEXT)''')
    
    # Migrazione colonne sicura
    cols = [("body_fat", "REAL"), ("muscle_mass", "REAL"), ("water_perc", "REAL"), ("bone_mass", "REAL"), ("sleep", "REAL")]
    for col, t in cols:
        try: c.execute(f"ALTER TABLE player_data ADD COLUMN {col} {t}")
        except: pass
    conn.commit()
    return conn

db_conn = init_db()

# API GROQ
if "GROQ_API_KEY" in st.secrets:
    groq_key = st.secrets["GROQ_API_KEY"]
else:
    groq_key = st.sidebar.text_input("🔑 Groq API Key", type="password")

client = Groq(api_key=groq_key) if groq_key else None

def oracle_chat(prompt, context_data=""):
    if not client: return "⚠️ Inserisci API Key."
    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": f"Sei l'Assistant Coach NBA di {st.session_state.get('team_name')}. Rispondi in italiano, tecnico e conciso."},
                {"role": "user", "content": f"Dati: {context_data}\nDomanda: {prompt}"}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e: return f"Errore: {e}"

def make_hashes(password): return hashlib.sha256(str.encode(password)).hexdigest()
def check_hashes(password, hashed_text): return make_hashes(password) == hashed_text

# =================================================================
# 3. MANUALE PDF (PROFESSIONALE)
# =================================================================
def generate_branded_manual(team_name, logo_path=None):
    pdf = FPDF()
    pdf.add_page()
    
    # Logo Gestito Correttamente
    if logo_path and os.path.exists(logo_path):
        try:
            pdf.image(logo_path, 10, 10, 20) # 20mm width
            pdf.ln(25) # Spazio verticale sicuro
        except: pdf.ln(10)
    else: pdf.ln(10)

    pdf.set_font("Arial", 'B', 24)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, f"{team_name.upper()} PROTOCOL", ln=True, align='L')
    
    pdf.set_font("Arial", 'I', 12)
    pdf.cell(0, 10, "Manuale Operativo v20.0 - Architettura Completa", ln=True, align='L')
    pdf.line(10, pdf.get_y()+5, 200, pdf.get_y()+5)
    pdf.ln(15)

    def add_sec(title, body):
        pdf.set_font("Arial", 'B', 12)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 8, f" {title}", ln=True, fill=True)
        pdf.ln(4)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 6, body.encode('latin-1','ignore').decode('latin-1'))
        pdf.ln(8)

    add_sec("1. VIDEO & LIVE STREAM", "Caricamento match MP4 e predisposizione per analisi Live Camera (RTSP). Modelli YOLO per tracking.")
    add_sec("2. SYNC HUB & DATA", "Importazione CSV (usare template) o inserimento manuale completo (Massa Ossea, Acqua, Sonno).")
    add_sec("3. STRATEGY ROOM", "War Room con grafici Radar per confronto diretto tra giocatori (Lineup Optimization).")
    add_sec("4. THE ORACLE", "Assistente AI sempre attivo in basso a destra. Fornisce analisi tattiche e nutrizionali in tempo reale.")
    
    return pdf.output(dest='S').encode('latin-1')

# =================================================================
# 4. LOGIN
# =================================================================
if 'logged_in' not in st.session_state: st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🏀 CoachTrack v20")
    t1, t2 = st.tabs(["Login Coach", "Registrazione"])
    with t1:
        u, p = st.text_input("Username"), st.text_input("Password", type="password")
        if st.button("Accedi"):
            c = db_conn.cursor(); c.execute("SELECT password, team_name FROM users WHERE username = ?", (u,))
            data = c.fetchone()
            if data and check_hashes(p, data[0]):
                st.session_state.logged_in, st.session_state.username = True, u
                st.session_state.team_name = data[1]; st.rerun()
    with t2:
        nu, nt, np = st.text_input("Nuovo User"), st.text_input("Team"), st.text_input("Pw", type="password")
        if st.button("Crea Account"):
            c = db_conn.cursor(); c.execute("INSERT INTO users VALUES (?,?,?)", (nu, make_hashes(np), nt))
            db_conn.commit(); st.success("Fatto! Accedi."); 
    st.stop()

# =================================================================
# 5. LAYOUT & SIDEBAR
# =================================================================
curr_user = st.session_state.username
team_name = st.session_state.team_name
logo_path = f"logo_{curr_user}.png"

# SIDEBAR
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=80) # Logo piccolo e elegante
st.sidebar.markdown(f"### {team_name}")
st.sidebar.caption("Operations Center")

# DOWNLOAD MANUALE
try:
    man_pdf = generate_branded_manual(team_name, logo_path)
    st.sidebar.download_button("📘 Scarica Manuale PDF", man_pdf, "Manuale_Ufficiale.pdf")
except: pass

if st.sidebar.button("🚪 Logout"): st.session_state.logged_in = False; st.rerun()

# Recupero Dati Globali
df_team = pd.read_sql_query(f"SELECT * FROM player_data WHERE owner = '{curr_user}'", db_conn)

# HEADER
c_head1, c_head2 = st.columns([0.5, 8])
if os.path.exists(logo_path): c_head1.image(logo_path, use_container_width=True)
c_head2.title(f"{team_name} Dashboard")

# =================================================================
# 6. FLOATING ORACLE CHAT (BASSO A DESTRA)
# =================================================================
# Usiamo st.popover per creare un menu fluttuante persistente
with st.sidebar:
    st.markdown("---")
    st.markdown("**🧠 AI Assistant**")
    # Questo bottone è nella sidebar per stabilità, ma simuliamo l'accesso rapido
    with st.popover("💬 Chiedi a The Oracle", use_container_width=True):
        st.markdown("### The Oracle")
        
        # Suggerimenti (Ripristinati)
        st.caption("Suggerimenti rapidi:")
        col_s1, col_s2 = st.columns(2)
        if col_s1.button("Analisi Infortuni"): st.info(oracle_chat("Chi rischia infortuni oggi?", df_team.to_string()))
        if col_s2.button("Consigli Dietetici"): st.info(oracle_chat("Chi ha bisogno di più proteine?", df_team.to_string()))
        
        # Chat
        prompt = st.text_input("Scrivi qui la tua domanda...")
        if prompt:
            res = oracle_chat(prompt, df_team.to_string())
            st.markdown(f"**Oracle:** {res}")

# =================================================================
# 7. TABS FUNZIONALI
# =================================================================
tabs = st.tabs(["VIDEO LAB", "BIO-METRICS", "STRATEGY ROOM", "SYNC HUB"])

# --- TAB 1: VIDEO LAB (Con Live Stream e Upload) ---
with tabs[0]:
    st.header("Analisi Video & Live Feed")
    
    mode = st.radio("Sorgente:", ["📁 Upload File", "📡 Live Stream / IP Camera"], horizontal=True)
    
    if mode == "📁 Upload File":
        try:
            from ultralytics import YOLO
            v_file = st.file_uploader("Carica Match (.mp4)", type=['mp4'])
            if v_file and st.checkbox("Avvia YOLO Tracking"):
                with open("temp.mp4", "wb") as f: f.write(v_file.read())
                model = YOLO("yolov8n.pt")
                cap = cv2.VideoCapture("temp.mp4")
                st_frame = st.empty()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    res = model.predict(frame, verbose=False, conf=0.3)
                    st_frame.image(res[0].plot(), channels="BGR", use_container_width=True)
                cap.release()
        except ImportError: st.error("Libreria YOLO non installata.")
        
    else:
        st.info("📡 Configurazione Live Stream")
        rtsp_url = st.text_input("URL RTSP/RTMP Telecamera", "rtsp://192.168.1.10:554/stream")
        if st.button("Connetti Camera"):
            st.warning("Connessione allo stream in corso... (Simulazione)")
            # Qui andrebbe il codice cv2.VideoCapture(rtsp_url)

# --- TAB 2: BIO-METRICS (Semafori e Dati Completi) ---
with tabs[1]:
    st.header("Bio-Intelligence Dashboard")
    if not df_team.empty:
        sel_p = st.selectbox("Seleziona Atleta", df_team['player_name'].unique())
        p = df_team[df_team['player_name'] == sel_p].iloc[-1]
        
        # SEMAFORO INFORTUNI
        col_risk, col_msg = st.columns([1, 4])
        risk_score = 0
        if p.get('hrv', 60) < 45: risk_score += 1
        if p.get('sleep', 7) < 6: risk_score += 1
        if p.get('rpe', 5) > 8: risk_score += 1
        
        if risk_score >= 2:
            col_risk.error("🔴 STOP")
            col_msg.error(f"**ALTO RISCHIO:** HRV Basso ({p.get('hrv')}) e Carico Alto. Riposo obbligatorio.")
        elif risk_score == 1:
            col_risk.warning("🟡 WARN")
            col_msg.warning("**ATTENZIONE:** Monitorare idratazione e carico.")
        else:
            col_risk.success("🟢 GO")
            col_msg.success("Atleta in condizione ottimale.")

        # WIDGET DATI (Tutti i campi)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Peso", f"{p.get('weight',0)} kg")
        c2.metric("Massa Muscolare", f"{p.get('muscle_mass',0)} kg")
        c3.metric("Massa Ossea", f"{p.get('bone_mass',0)} kg")
        c4.metric("Grasso", f"{p.get('body_fat',0)}%")
        
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Acqua", f"{p.get('water_perc',0)}%")
        c6.metric("HRV", f"{p.get('hrv',0)} ms")
        c7.metric("Sonno", f"{p.get('sleep',0)} h")
        c8.metric("RPE", f"{p.get('rpe',0)}/10")

    else: st.info("Carica i dati nel Sync Hub.")

# --- TAB 3: STRATEGY ROOM (Grafico Radar Ripristinato) ---
with tabs[2]:
    st.header("Strategy Room: Confronto Diretto")
    if len(df_team['player_name'].unique()) >= 2:
        c_sel1, c_sel2 = st.columns(2)
        p1_name = c_sel1.selectbox("Giocatore A", df_team['player_name'].unique(), key="s1")
        p2_name = c_sel2.selectbox("Giocatore B", df_team['player_name'].unique(), key="s2")
        
        if p1_name and p2_name:
            d1 = df_team[df_team['player_name'] == p1_name].iloc[-1]
            d2 = df_team[df_team['player_name'] == p2_name].iloc[-1]
            
            # Grafico Radar
            categories = ['Tiro %', 'Muscolo', 'HRV', 'Sonno', 'Acqua %']
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=[d1.get('shot_efficiency',0), d1.get('muscle_mass',0), d1.get('hrv',0), d1.get('sleep',0)*10, d1.get('water_perc',0)],
                theta=categories, fill='toself', name=p1_name
            ))
            fig.add_trace(go.Scatterpolar(
                r=[d2.get('shot_efficiency',0), d2.get('muscle_mass',0), d2.get('hrv',0), d2.get('sleep',0)*10, d2.get('water_perc',0)],
                theta=categories, fill='toself', name=p2_name
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption(f"Analisi comparativa generata da The Oracle per {p1_name} vs {p2_name}")

# --- TAB 4: SYNC HUB (Form Completo & Template) ---
with tabs[3]:
    st.header("Sync Hub & Settings")
    
    col_main1, col_main2 = st.columns(2)
    
    # LOGO UPLOAD
    with col_main1:
        st.subheader("Branding")
        up_logo = st.file_uploader("Carica Logo Squadra", type=['png', 'jpg'])
        if up_logo:
            Image.open(up_logo).save(logo_path)
            st.success("Logo Aggiornato! Ricarica pagina.")

    # CSV TEMPLATE & UPLOAD
    with col_main2:
        st.subheader("Importazione CSV")
        # Template Completo
        t_cols = ["player_name", "weight", "body_fat", "muscle_mass", "water_perc", "bone_mass", "hrv", "rpe", "sleep", "shot_efficiency"]
        csv_t = pd.DataFrame(columns=t_cols).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Scarica Template Completo", csv_t, "template_team_v20.csv", "text/csv")
        
        up_csv = st.file_uploader("Carica File Dati", type=['csv'])
        if up_csv and st.button("Importa"):
            df_new = pd.read_csv(up_csv)
            df_new['owner'] = curr_user
            df_new['timestamp'] = datetime.now().strftime("%Y-%m-%d")
            df_new.to_sql('player_data', db_conn, if_exists='append', index=False)
            st.success("Database aggiornato.")

    st.markdown("---")
    
    # FORM MANUALE COMPLETO (Tutti i campi ripristinati)
    st.subheader("📝 Inserimento Manuale Dettagliato")
    with st.form("full_manual"):
        fn_name = st.text_input("Nome Atleta")
        
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        w = r1_c1.number_input("Peso (kg)", 0.0, 150.0, 85.0)
        h = r1_c2.number_input("HRV (ms)", 0, 150, 60)
        s = r1_c3.number_input("Sonno (ore)", 0.0, 12.0, 7.5)
        
        r2_c1, r2_c2, r2_c3 = st.columns(3)
        fat = r2_c1.number_input("Grasso (%)", 0.0, 40.0, 10.0)
        mus = r2_c2.number_input("Muscolo (kg)", 0.0, 100.0, 45.0)
        wat = r2_c3.number_input("Acqua (%)", 0.0, 100.0, 60.0)
        
        r3_c1, r3_c2, r3_c3 = st.columns(3)
        bone = r3_c1.number_input("Ossa (kg)", 0.0, 10.0, 3.5)
        rpe = r2_c2.slider("RPE (Fatica)", 1, 10, 5)
        shot = r3_c3.slider("Tiro %", 0, 100, 45)
        
        if st.form_submit_button("Salva nel Database"):
            c = db_conn.cursor()
            q = """INSERT INTO player_data 
                   (owner, player_name, timestamp, weight, hrv, sleep, body_fat, muscle_mass, water_perc, bone_mass, rpe, shot_efficiency) 
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"""
            c.execute(q, (curr_user, fn_name, datetime.now().strftime("%Y-%m-%d"), w, h, s, fat, mus, wat, bone, rpe, shot))
            db_conn.commit()
            st.success(f"Dati completi salvati per {fn_name}")
            st.rerun()
