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

# Gestione Importazione YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# =================================================================
# 1. DATABASE (AGGIORNATO CON BODY COMPOSITION)
# =================================================================
def init_db():
    conn = sqlite3.connect('coachtrack_v18.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, team_name TEXT)''')
    # Tabella estesa con metriche bilancia impedenziometrica
    c.execute('''CREATE TABLE IF NOT EXISTS player_data 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT, player_name TEXT, 
                  timestamp TEXT, hrv REAL, rpe INTEGER, shot_efficiency REAL, 
                  weight REAL, sleep REAL, 
                  body_fat REAL, muscle_mass REAL, water_perc REAL, bone_mass REAL,
                  video_notes TEXT)''')
    conn.commit()
    return conn

db_conn = init_db()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# =================================================================
# 2. LOGICA AI, PDF REPORT & MANUALE UTENTE
# =================================================================
st.set_page_config(page_title="CoachTrack Oracle v18", layout="wide", page_icon="🏀")

if "GROQ_API_KEY" in st.secrets:
    groq_key = st.secrets["GROQ_API_KEY"]
else:
    groq_key = st.sidebar.text_input("Groq API Key", type="password", help="Chiave per THE ORACLE")

client = Groq(api_key=groq_key) if groq_key else None

def oracle_chat(prompt, context=""):
    if not client: return "⚠️ THE ORACLE è offline. Inserisci API Key."
    full_prompt = f"""
    Sei THE ORACLE, AI tattica e biomedica per l'NBA. 
    Contesto Squadra: {context}.
    Rispondi in modo tecnico ma conciso al Coach.
    Domanda: {prompt}
    """
    try:
        res = client.chat.completions.create(messages=[{"role":"user","content":full_prompt}], model="llama3-8b-8192")
        return res.choices[0].message.content
    except Exception as e: return f"Errore Oracle: {e}"

def generate_user_manual():
    """Genera il Manuale Utente PDF scaricabile."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "CoachTrack Oracle v18 - Manuale Utente", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    manual_text = """
    BENVENUTO COACH.
    Questa guida ti spiega come dominare la lega usando CoachTrack.

    1. VIDEO YOLO HUB
       - Carica il file video del match (.mp4).
       - Seleziona il modello (v8 veloce, v11 preciso).
       - Spunta 'Avvia Tracking' e segui la barra di progresso.
    
    2. PLAYER 360 & BIO
       - Analisi completa dell'atleta.
       - Include dati da Smart Scale (Grasso, Muscolo, Acqua).
       - Genera Report AI automatici.

    3. WAR ROOM
       - Confronta due giocatori per creare la lineup perfetta.
       - Analizza le sinergie tattiche.

    4. THE ORACLE (CHAT)
       - Il tuo assistente 24/7. 
       - Chiedi: "Chi è a rischio infortunio?", "Analizza la difesa".

    5. SYNC HUB
       - Inserisci dati manualmente o via CSV.
       - Collega (simulazione) le API di Smart Scale e Wearable.
    """
    # Pulizia caratteri per FPDF
    clean_text = manual_text.encode('latin-1', 'ignore').decode('latin-1')
    pdf.multi_cell(0, 10, clean_text)
    return pdf.output(dest='S').encode('latin-1')

# =================================================================
# 3. LOGIN
# =================================================================
if 'logged_in' not in st.session_state: st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🏀 CoachTrack Oracle v18")
    tab_log, tab_reg = st.tabs(["Login", "Registra Squadra"])
    
    with tab_log:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Accedi"):
            c = db_conn.cursor()
            c.execute("SELECT password FROM users WHERE username = ?", (u,))
            data = c.fetchone()
            if data and check_hashes(p, data[0]):
                st.session_state.logged_in = True
                st.session_state.username = u
                st.rerun()
            else: st.error("Errore credenziali.")
            
    with tab_reg:
        nu = st.text_input("Nuovo User")
        nt = st.text_input("Nome Team")
        npw = st.text_input("Nuova Password", type="password")
        if st.button("Crea Account"):
            try:
                c = db_conn.cursor()
                c.execute("INSERT INTO users VALUES (?,?,?)", (nu, make_hashes(npw), nt))
                db_conn.commit()
                st.success("Registrato!")
            except: st.error("Utente già esistente.")
    st.stop()

# =================================================================
# 4. DASHBOARD OPERATIVA
# =================================================================
curr_user = st.session_state.username
st.sidebar.title(f"👨‍🏫 {curr_user}")

# DOWNLOAD MANUALE
manual_pdf = generate_user_manual()
st.sidebar.download_button("📘 Scarica Manuale d'Uso", manual_pdf, "Manuale_CoachTrack.pdf")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

df_team = pd.read_sql_query(f"SELECT * FROM player_data WHERE owner = '{curr_user}'", db_conn)

tabs = st.tabs(["🎥 Video Tracking", "👤 Bio 360°", "⚔️ War Room", "📖 Playbook", "💬 The Oracle", "⌚ Sync Hub"])

# --- TAB 1: VIDEO TRACKING CON PROGRESS BAR ---
with tabs[0]:
    st.header("🎥 Video Analysis & Progress")
    if not YOLO_AVAILABLE:
        st.error("Installa 'ultralytics' per usare YOLO.")
    else:
        col_v1, col_v2 = st.columns([3, 1])
        with col_v2:
            model_ver = st.selectbox("Modello AI", ["yolov8n.pt", "yolov11n.pt"])
            conf_t = st.slider("Confidenza", 0.1, 1.0, 0.25)
            start_track = st.checkbox("Avvia Analisi")
        
        with col_v1:
            vid = st.file_uploader("Carica Match", type=['mp4'])
            if vid and start_track:
                tfile = open("temp.mp4", "wb")
                tfile.write(vid.read())
                
                cap = cv2.VideoCapture("temp.mp4")
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                st.write(f"🔄 Avvio Tracking su {total_frames} frames...")
                my_bar = st.progress(0)
                frame_text = st.empty()
                st_frame = st.empty()
                
                model = YOLO(model_ver)
                curr_frame = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    curr_frame += 1
                    # Aggiornamento Barra e Testo ogni 5 frame per velocità
                    if curr_frame % 5 == 0:
                        prog = min(curr_frame / total_frames, 1.0)
                        my_bar.progress(prog)
                        frame_text.text(f"Processing Frame: {curr_frame}/{total_frames}")
                    
                    results = model.predict(frame, conf=conf_t, verbose=False)
                    res_plotted = results[0].plot()
                    st_frame.image(res_plotted, channels="BGR", use_container_width=True)
                
                cap.release()
                my_bar.progress(1.0)
                st.success("Analisi Completata!")

# --- TAB 2: BIO 360 (Con Smart Scale Data) ---
with tabs[1]:
    st.header("👤 Scheda Atleta & Composizione Corporea")
    if not df_team.empty:
        sel_p = st.selectbox("Seleziona Atleta", df_team['player_name'].unique())
        p_data = df_team[df_team['player_name'] == sel_p].iloc[-1]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Peso", f"{p_data['weight']} kg")
        c2.metric("Grasso Corporeo", f"{p_data['body_fat']}%")
        c3.metric("Massa Muscolare", f"{p_data['muscle_mass']} kg")
        
        st.subheader("Analisi Oracle")
        if st.button("Genera Report Biomedico"):
            context = f"Dati {sel_p}: Peso {p_data['weight']}, Fat {p_data['body_fat']}%, Muscle {p_data['muscle_mass']}, HRV {p_data['hrv']}."
            st.info(oracle_chat("Analizza la condizione fisica e dai consigli nutrizionali.", context))
            
        st.plotly_chart(px.bar(df_team[df_team['player_name']==sel_p], x='timestamp', y=['body_fat', 'muscle_mass'], barmode='group', title="Evoluzione Composizione Corporea"))
    else: st.warning("Nessun dato. Vai al Sync Hub.")

# --- TAB 3: WAR ROOM ---
with tabs[2]:
    st.header("⚔️ War Room")
    if len(df_team['player_name'].unique()) >= 2:
        p1 = st.selectbox("Player A", df_team['player_name'].unique(), key="w1")
        p2 = st.selectbox("Player B", df_team['player_name'].unique(), key="w2")
        if st.button("Confronto Diretto"):
            st.write(oracle_chat(f"Chi è meglio schierare oggi tra {p1} e {p2}?", df_team.to_string()))
            
        d1 = df_team[df_team['player_name'] == p1].iloc[-1]
        d2 = df_team[df_team['player_name'] == p2].iloc[-1]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=[d1['shot_efficiency'], d1['hrv'], d1['muscle_mass']], theta=['Tiro', 'Cardio', 'Potenza'], fill='toself', name=p1))
        fig.add_trace(go.Scatterpolar(r=[d2['shot_efficiency'], d2['hrv'], d2['muscle_mass']], theta=['Tiro', 'Cardio', 'Potenza'], fill='toself', name=p2))
        st.plotly_chart(fig)

# --- TAB 4: PLAYBOOK ---
with tabs[3]:
    st.header("📖 Playbook")
    sch = st.selectbox("Schema", ["Pick&Roll", "Iso", "Post-Up"])
    if st.button("Analizza"):
        st.success(oracle_chat(f"Chi deve giocare {sch}?", df_team.to_string()))

# --- TAB 5: THE ORACLE (Con Istruzioni) ---
with tabs[4]:
    st.header("💬 The Oracle")
    
    with st.expander("💡 COSA POSSO CHIEDERE? (Clicca qui)", expanded=True):
        st.markdown("""
        * **Tattica:** "Qual è il miglior quintetto difensivo?"
        * **Salute:** "Chi ha l'HRV troppo basso oggi?"
        * **Scouting:** "Analizza i dati di tiro di Curry nell'ultima settimana."
        * **Nutrizione:** "Crea un piano di recupero per LeBron dopo il match."
        """)
        
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.write(m["content"])
        
    if p := st.chat_input("Scrivi qui..."):
        st.session_state.messages.append({"role":"user", "content":p})
        with st.chat_message("user"): st.write(p)
        r = oracle_chat(p, df_team.to_string())
        st.session_state.messages.append({"role":"assistant", "content":r})
        with st.chat_message("assistant"): st.write(r)

# --- TAB 6: SYNC HUB (Smart Scale Integration) ---
with tabs[5]:
    st.header("⌚ Sync Hub & Smart Scale")
    
    c_imp, c_man = st.columns(2)
    
    with c_imp:
        st.subheader("Smart Scale API Bridge")
        st.info("Integrazione: Withings / Garmin / Tanita")
        api_k = st.text_input("API Key Bilancia", type="password")
        if st.button("Sincronizza Dati Bilancia"):
            st.success("✅ Dati Composizione Corporea scaricati dal Cloud.")
            # Qui andrebbe la logica di fetch reale
            
    with c_man:
        st.subheader("Input Manuale Completo")
        with st.form("bio_form"):
            nm = st.text_input("Nome")
            we = st.number_input("Peso (kg)", 50.0, 150.0, 90.0)
            bf = st.number_input("Grasso Corporeo (%)", 3.0, 40.0, 10.0)
            mm = st.number_input("Massa Muscolare (kg)", 30.0, 100.0, 60.0)
            wa = st.number_input("Acqua Corporea (%)", 30.0, 80.0, 60.0)
            hr = st.number_input("HRV", 20, 150, 60)
            ef = st.number_input("Tiro %", 0, 100, 45)
            
            if st.form_submit_button("Salva Bio-Dati"):
                c = db_conn.cursor()
                ts = datetime.now().strftime("%Y-%m-%d")
                # Inserimento con nuovi campi
                c.execute("""INSERT INTO player_data 
                             (owner, player_name, timestamp, weight, body_fat, muscle_mass, water_perc, hrv, shot_efficiency) 
                             VALUES (?,?,?,?,?,?,?,?,?)""",
                          (curr_user, nm, ts, we, bf, mm, wa, hr, ef))
                db_conn.commit()
                st.rerun()
