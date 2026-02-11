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
# 1. SETUP & STYLE (FUTURA & PROFESSIONAL ICONS)
# =================================================================
st.set_page_config(page_title="CoachTrack Elite v21", layout="wide", page_icon="🏀")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Jost:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Jost', sans-serif !important; }
    .stMetric { background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px; }
    h1, h2, h3 { font-weight: 600 !important; color: #1a1a1a; }
    </style>
    """, unsafe_allow_html=True)

# =================================================================
# 2. DATABASE CORE (PUNTAMENTO v17)
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
# 3. GENERAZIONE MANUALE PDF PROFESSIONALE
# =================================================================
def generate_detailed_manual(team_name, logo_path=None):
    # Definizione interna di una classe per gestire il footer in automatico
    class PDF(FPDF):
        def footer(self):
            # Posizionamento a 1.5 cm dal fondo
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(100)
            # Testo del piè di pagina richiesto
            footer_text = "I dati sono protetti e isolati per ogni singolo Coach tramite Smart Scale API e database criptati."
            self.cell(0, 10, footer_text.encode('latin-1', 'ignore').decode('latin-1'), 0, 0, 'C')

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.add_page()
    
    # --- LOGO PICCOLO E CENTRATO ---
    # Larghezza pagina A4 = 210mm. Logo fissato a 20mm. (210-20)/2 = 95
    if logo_path and os.path.exists(logo_path):
        pdf.image(logo_path, x=95, y=10, w=20)
        pdf.ln(25)
    else:
        pdf.ln(15)
    
    # --- HEADER TITOLO ---
    pdf.set_font("Arial", 'B', 22)
    pdf.set_text_color(0, 51, 102) # Blu Navy Bracciano
    pdf.cell(0, 15, f"Manuale CoachTrack", ln=True, align='C')
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 8, team_name.upper(), ln=True, align='C')
    pdf.ln(10)

    def chapter_title(title):
        pdf.set_font("Arial", 'B', 13)
        pdf.set_fill_color(240, 240, 240)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(0, 10, title, ln=True, fill=True)
        pdf.ln(4)

    def sub_chapter(title, text):
        pdf.set_font("Arial", 'B', 10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 7, title, ln=True)
        pdf.set_font("Arial", '', 10)
        pdf.multi_cell(0, 5, text.encode('latin-1', 'ignore').decode('latin-1'))
        pdf.ln(4)

    # --- 1. ARCHITETTURA E FILOSOFIA ---
    chapter_title("1. ARCHITETTURA E FILOSOFIA DECISIONALE")
    sub_chapter("Ecosistema Integrato", 
        "Il sistema non e una semplice raccolta di dati, ma un'infrastruttura di supporto decisionale. "
        "Si basa sulla triangolazione di: Carico Esterno (Video), Carico Interno (Biometria) e Sintesi "
        "Predittiva (AI). L'obiettivo e trasformare il dato grezzo in vantaggio competitivo immediato.")

    # --- 2. ANALISI VIDEO (YOLO) ---
    chapter_title("2. VIDEO LAB: ANALISI CINEMATICA (YOLO V8/V11)")
    sub_chapter("Tecnologia di Tracciamento", 
        "Utilizziamo Computer Vision basata su YOLO per mappare ogni movimento sul parquet. "
        "Il sistema calcola lo 'Spacing Index' analizzando la dispersione spaziale dei giocatori. "
        "Questo permette di identificare falle strutturali nelle transizioni difensive e "
        "ottimizzare gli angoli di passaggio per i tiratori.")

    # --- 3. BIO-INTELLIGENCE ---
    chapter_title("3. BIO-INTELLIGENCE: SCIENZA DELLA PERFORMANCE")
    sub_chapter("Analisi HRV e Sistema Nervoso", 
        "Il monitoraggio dell'Heart Rate Variability (HRV) e il gold standard per valutare il recupero. "
        "Un trend decrescente indica una saturazione del sistema simpatico, preannunciando cali "
        "di precisione al tiro e tempi di reazione rallentati.")
    
    sub_chapter("Idratazione e Catabolismo", 
        "Il monitoraggio della massa magra e dell'acqua cellulare e vitale. Sotto il 55% di idratazione, "
        "la conducibilita neuromuscolare diminuisce, aumentando il rischio di lesioni da overuse.")

    # --- TABELLA TECNICA ---
    pdf.ln(2)
    pdf.set_fill_color(0, 51, 102); pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(45, 8, "METRICA", 1, 0, 'C', True)
    pdf.cell(45, 8, "OTTIMALE (GO)", 1, 0, 'C', True)
    pdf.cell(45, 8, "MONITORARE", 1, 0, 'C', True)
    pdf.cell(45, 8, "RISCHIO (STOP)", 1, 1, 'C', True)
    
    pdf.set_text_color(0, 0, 0); pdf.set_font("Arial", '', 9)
    metrics = [
        ("HRV Baseline", "> 90%", "75-89%", "< 75%"),
        ("Acqua Corporea", "> 60%", "56-59%", "< 55%"),
        ("RPE (Fatica)", "1 - 5", "6 - 8", "9 - 10")
    ]
    for m, v, g, r in metrics:
        pdf.cell(45, 7, m, 1); pdf.cell(45, 7, v, 1); pdf.cell(45, 7, g, 1); pdf.cell(45, 7, r, 1, 1)
    pdf.ln(6)

    # --- 4. STRATEGY ROOM ---
    chapter_title("4. STRATEGY ROOM: WAR ROOM TATTICA")
    sub_chapter("Lineup Optimization", 
        "La War Room incrocia l'efficienza reale di tiro (Shot %) con i dati di freschezza atletica. "
        "Il sistema suggerisce rotazioni che mantengano l'intensita difensiva costante per tutti "
        "i 40 minuti, identificando i giocatori pronti per il 'clutch time'.")

    # --- 5. THE ORACLE AI ---
    chapter_title("5. INTERAZIONE CON THE ORACLE")
    sub_chapter("Assistenza Strategica", 
        "The Oracle analizza i pattern storici. Puoi richiedere analisi predittive come: "
        "'Analizza il rischio infortuni della lineup titolare' o 'Pianifica un piano di recupero "
        "nutrizionale basato sui cali di massa ossea rilevati'.")

    return pdf.output(dest='S').encode('latin-1')

# =================================================================
# 4. LOGIN SYSTEM
# =================================================================
if 'logged_in' not in st.session_state: st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🏀 CoachTrack Elite v21")
    t_auth1, t_auth2 = st.tabs(["🎯 Login", "📝 Registrazione"])
    with t_auth1:
        u, p = st.text_input("Username"), st.text_input("Password", type="password")
        if st.button("Entra"):
            c = db_conn.cursor()
            c.execute("SELECT password, team_name FROM users WHERE username = ?", (u,))
            data = c.fetchone()
            if data and check_hashes(p, data[0]):
                st.session_state.logged_in, st.session_state.username = True, u
                st.session_state.team_name = data[1]; st.rerun()
            else: st.error("Dati errati.")
    with t_auth2:
        nu, nt, np = st.text_input("Nuovo User"), st.text_input("Nome Team"), st.text_input("Pw", type="password")
        if st.button("Crea Team"):
            c = db_conn.cursor(); c.execute("INSERT INTO users VALUES (?,?,?)", (nu, make_hashes(np), nt))
            db_conn.commit(); st.success("Account creato!")
    st.stop()

# =================================================================
# 5. DASHBOARD OPERATIVA (TUTTE LE FUNZIONI)
# =================================================================
curr_user, team_name = st.session_state.username, st.session_state.team_name
logo_path = f"logo_{curr_user}.png"

# SIDEBAR
if os.path.exists(logo_path): st.sidebar.image(logo_path, width=100)
st.sidebar.title(team_name)

# TASTO DOWNLOAD MANUALE (CORRETTO)
try:
    pdf_file = generate_detailed_manual(team_name, logo_path)
    st.sidebar.download_button(label="📘 Scarica Manuale Tecnico", data=pdf_file, file_name=f"Manuale_{team_name}.pdf", mime="application/pdf")
except Exception as e:
    st.sidebar.error(f"Errore Manuale: {e}")

if st.sidebar.button("Logout"): st.session_state.logged_in = False; st.rerun()

# Recupero Dati
df_team = pd.read_sql_query(f"SELECT * FROM player_data WHERE owner = '{curr_user}'", db_conn)

# TABS PROFESSIONALI
tabs = st.tabs(["🎯 Video Lab", "🧬 Bio-Intelligence", "📊 Strategy Room", "🧠 The Oracle", "🔌 Sync & Settings"])

# --- TAB 1: VIDEO LAB (CARICAMENTO E LIVE) ---
with tabs[0]:
    st.header("Video Analysis Center")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.subheader("Upload Match")
        up_v = st.file_uploader("Carica MP4", type=["mp4", "mov"])
        if up_v: st.video(up_v)
    with col_v2:
        st.subheader("Live Feed")
        url_stream = st.text_input("Indirizzo RTSP/IP Camera", placeholder="rtsp://admin:password@192.168.1.10")
        if st.button("Connetti Camera"): st.info("Connessione in corso allo stream...")

# --- TAB 2: BIO-INTELLIGENCE (WIDGET E SEMAFORO) ---
with tabs[1]:
    st.header("Performance Biometrics")
    if not df_team.empty:
        sel_p = st.selectbox("Seleziona Atleta", df_team['player_name'].unique())
        p = df_team[df_team['player_name'] == sel_p].iloc[-1]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Peso", f"{p.get('weight', 0)} kg")
        c2.metric("HRV", f"{p.get('hrv', 0)} ms")
        c3.metric("Massa Muscolo", f"{p.get('muscle_mass', 0)} kg")
        c4.metric("Body Fat", f"{p.get('body_fat', 0)}%")
        
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Massa Ossea", f"{p.get('bone_mass', 0)} kg")
        c6.metric("Acqua", f"{p.get('water_perc', 0)} %")
        c7.metric("Sonno", f"{p.get('sleep', 0)} h")
        c8.metric("RPE", f"{p.get('rpe', 0)}/10")

        # Logica Semaforo
        st.markdown("---")
        if p.get('hrv', 60) < 45 or p.get('rpe', 5) > 8:
            st.error("🔴 STATUS: RISCHIO INFORTUNIO ELEVATO. Carico da ridurre.")
        elif p.get('hrv', 60) < 55:
            st.warning("🟡 STATUS: AFFATICAMENTO. Monitorare recupero.")
        else:
            st.success("🟢 STATUS: OTTIMALE. Atleta pronto al 100%.")
    else: st.info("Nessun dato atleta trovato. Vai al Sync Hub.")

# --- TAB 3: STRATEGY ROOM (RADAR CHART) ---
with tabs[2]:
    st.header("War Room - Analisi Comparativa")
    if len(df_team['player_name'].unique()) >= 2:
        p1 = st.selectbox("Giocatore 1", df_team['player_name'].unique(), key="p1")
        p2 = st.selectbox("Giocatore 2", df_team['player_name'].unique(), key="p2")
        
        d1 = df_team[df_team['player_name'] == p1].iloc[-1]
        d2 = df_team[df_team['player_name'] == p2].iloc[-1]
        
        fig = go.Figure()
        cat = ['HRV', 'Tiro%', 'Muscolo', 'Sonno', 'RPE (Inv)']
        fig.add_trace(go.Scatterpolar(r=[d1['hrv'], d1['shot_efficiency'], d1['muscle_mass'], d1['sleep']*10, 100-(d1['rpe']*10)], theta=cat, fill='toself', name=p1))
        fig.add_trace(go.Scatterpolar(r=[d2['hrv'], d2['shot_efficiency'], d2['muscle_mass'], d2['sleep']*10, 100-(d2['rpe']*10)], theta=cat, fill='toself', name=p2))
        st.plotly_chart(fig)
    else: st.warning("Necessari almeno 2 atleti per il confronto.")

# --- TAB 4: THE ORACLE (CON FEEDBACK ATTIVAZIONE) ---
with tabs[3]:
    st.header("🧠 The Oracle AI Assistant")
    
    # Inserimento Chiave
    api_k = st.text_input("Configurazione: Inserisci Groq API Key", type="password", help="Inserisci la tua chiave Groq per attivare l'assistente.")
    
    if api_k:
        # FEEDBACK VISIVO RICHIESTO
        st.success(f"✅ Connessione stabilita con successo! Ciao Coach {curr_user.capitalize()}, sono a tua disposizione. Chiedimi qualsiasi analisi tattica o biometrica.")
        
        # Inizializzazione Client
        client = Groq(api_key=api_k)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Chiedi a The Oracle..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    # Passiamo i dati della squadra come contesto
                    context = df_team.to_string() if not df_team.empty else "Nessun dato disponibile."
                    response = client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[
                            {"role": "system", "content": f"Sei l'assistente tecnico del team {team_name}. Analizza i dati e rispondi in modo professionale."},
                            {"role": "user", "content": f"Contesto Squadra:\n{context}\n\nDomanda Coach: {prompt}"}
                        ]
                    )
                    full_res = response.choices[0].message.content
                    st.markdown(full_res)
                    st.session_state.messages.append({"role": "assistant", "content": full_res})
                except Exception as e:
                    st.error(f"Errore durante l'analisi: {e}")
    else:
        st.warning("⚠️ L'intelligenza di The Oracle è attualmente in standby. Inserisci la tua API Key sopra per attivarla.")

# --- TAB 5: SYNC & SETTINGS (TEMPLATE, UPLOAD CSV, FORM MANUALE) ---
with tabs[4]:
    st.header("Sync Hub")
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.subheader("Importazione Dati")
        # Template
        t_cols = ["player_name", "weight", "body_fat", "muscle_mass", "water_perc", "bone_mass", "hrv", "rpe", "sleep", "shot_efficiency"]
        csv_template = pd.DataFrame(columns=t_cols).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Scarica Template CSV", csv_template, "template_coach.csv")
        
        # Upload
        up_csv = st.file_uploader("Carica CSV compilato", type="csv")
        if up_csv:
            new_data = pd.read_csv(up_csv)
            new_data['owner'] = curr_user
            new_data['timestamp'] = datetime.now().strftime("%Y-%m-%d")
            new_data.to_sql('player_data', db_conn, if_exists='append', index=False)
            st.success("Dati importati con successo!")
            
        st.markdown("---")
        st.subheader("Team Branding")
        new_logo = st.file_uploader("Carica Logo (PNG)", type=["png", "jpg"])
        if new_logo:
            Image.open(new_logo).save(logo_path)
            st.success("Logo salvato!"); st.rerun()

    with col_s2:
        st.subheader("Inserimento Manuale Completo")
        with st.form("manual_entry_form"):
            name = st.text_input("Nome Atleta")
            c_m1, c_m2 = st.columns(2)
            w = c_m1.number_input("Peso (kg)", 0.0, 150.0)
            h = c_m2.number_input("HRV (ms)", 0, 150)
            f = c_m1.number_input("Body Fat (%)", 0.0, 30.0)
            m = c_m2.number_input("Massa Muscolo (kg)", 0.0, 100.0)
            o = c_m1.number_input("Massa Ossea (kg)", 0.0, 10.0)
            a = c_m2.number_input("Acqua (%)", 0.0, 100.0)
            s = c_m1.number_input("Sonno (ore)", 0.0, 12.0)
            r = c_m2.slider("RPE Fatica (1-10)", 1, 10)
            sh = st.slider("Efficienza Tiro (%)", 0, 100)
            
            if st.form_submit_button("Sincronizza Dati Atleta"):
                c = db_conn.cursor()
                query = "INSERT INTO player_data (owner, player_name, timestamp, weight, hrv, body_fat, muscle_mass, bone_mass, water_perc, sleep, rpe, shot_efficiency) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
                c.execute(query, (curr_user, name, datetime.now().strftime("%Y-%m-%d"), w, h, f, m, o, a, s, r, sh))
                db_conn.commit(); st.success(f"Dati di {name} salvati."); st.rerun()
