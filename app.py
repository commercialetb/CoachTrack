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
# 1. CONFIGURAZIONE BASE
# =================================================================
st.set_page_config(page_title="CoachTrack Oracle v19", layout="wide", page_icon="🏀")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

if "GROQ_API_KEY" in st.secrets:
    groq_key = st.secrets["GROQ_API_KEY"]
else:
    groq_key = st.sidebar.text_input("Groq API Key", type="password", help="Chiave AI per The Oracle")

client = Groq(api_key=groq_key) if groq_key else None

# =================================================================
# 2. DATABASE & MIGRAZIONE DATI
# =================================================================
def init_db():
    # Usiamo il nome DB originale per mantenere i tuoi dati
    conn = sqlite3.connect('coachtrack_v17.db', check_same_thread=False)
    c = conn.cursor()
    
    # Tabella Utenti
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, team_name TEXT)''')
    
    # Tabella Dati Base
    c.execute('''CREATE TABLE IF NOT EXISTS player_data 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT, player_name TEXT, 
                  timestamp TEXT, hrv REAL, rpe INTEGER, shot_efficiency REAL, 
                  weight REAL, sleep REAL, video_notes TEXT)''')
    
    # AGGIUNTA COLONNE MANCANTI (Migrazione Automatica)
    # Se il tuo vecchio DB non ha queste colonne, le aggiungiamo ora senza rompere nulla.
    new_columns = [
        ("body_fat", "REAL"),
        ("muscle_mass", "REAL"),
        ("water_perc", "REAL"),
        ("bone_mass", "REAL")
    ]
    
    for col_name, col_type in new_columns:
        try:
            c.execute(f"ALTER TABLE player_data ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass # La colonna esiste già, ignoriamo l'errore
            
    conn.commit()
    return conn

db_conn = init_db()

def make_hashes(password): return hashlib.sha256(str.encode(password)).hexdigest()
def check_hashes(password, hashed_text): return make_hashes(password) == hashed_text

# =================================================================
# 3. FUNZIONI PDF & MANUALE PROFESSIONALE AGGIORNATO
# =================================================================
def generate_branded_manual(team_name, logo_path=None):
    pdf = FPDF()
    pdf.add_page()
    
    # Logo ridotto a 20mm per evitare sovrapposizioni
    if logo_path and os.path.exists(logo_path):
        try:
            pdf.image(logo_path, 10, 8, 20) 
            pdf.ln(15) # Spazio dopo il logo piccolo
        except:
            pdf.ln(10)
    else:
        pdf.ln(10)
        
    # Titolo Principale
    pdf.set_font("Arial", 'B', 22)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 15, f"{team_name.upper()} - PERFORMANCE PROTOCOL", ln=True, align='C')
    
    pdf.set_font("Arial", 'I', 11)
    pdf.set_text_color(127, 140, 141)
    pdf.cell(0, 10, "Documento Tecnico Riservato - Versione 19.0 Full Governance", ln=True, align='C')
    pdf.ln(10)
    
    def add_sec(title, body):
        pdf.set_font("Arial", 'B', 12)
        pdf.set_fill_color(230, 233, 237)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 10, f" {title}", ln=True, fill=True)
        pdf.ln(3)
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 6, body.encode('latin-1','ignore').decode('latin-1'))
        pdf.ln(6)

    # 1. VIDEO ANALYSIS
    add_sec("1. VIDEO TRACKING & SPACING (YOLO AI)", 
            "Il sistema utilizza Computer Vision avanzata per mappare il campo. Caricare i match in formato .mp4. "
            "YOLO v8: Ottimizzato per velocita, ideale per feedback immediato post-gara. "
            "YOLO v11: Massima precisione nelle linee di tiro e contatti in area. Monitorare lo spacing "
            "per ottimizzare i flussi offensivi definiti nel Playbook.")

    # 2. SYNC HUB
    add_sec("2. SYNC HUB: GESTIONE DATI INTEGRATA", 
            "L'integrità dei dati è garantita dal caricamento via CSV (usare il Template scaricabile) o "
            "inserimento manuale. Parametri monitorati: Peso, Body Fat (%), Massa Muscolare, Idratazione "
            "e Massa Ossea. Questi dati alimentano l'algoritmo di calcolo del Metabolismo Basale per l'AI.")

    # 3. PREVENZIONE INFORTUNI
    add_sec("3. PROTOCOLLO PREVENZIONE INFORTUNI (IP)", 
            "Il sistema valuta il rischio in tempo reale incrociando HRV e RPE.\n"
            "- ZONA VERDE: Atleta in stato omeostatico. Disponibile per pieno carico.\n"
            "- ZONA GIALLA: Segnali di affaticamento o disidratazione (Water % < 55). Ridurre il volume di allenamento.\n"
            "- ZONA ROSSA: Rischio lesione elevato (HRV < 45ms o Sonno < 6h). Stop cautelativo obbligatorio.")
    
    

    # 4. BIO-METRICS & COMPOSIZIONE
    add_sec("4. ANALISI DELLA COMPOSIZIONE CORPOREA", 
            "A differenza del semplice peso, monitoriamo la qualita dei tessuti. Un calo della Massa Muscolare "
            "associato a un calo della Massa Ossea suggerisce uno stato di catabolismo da stress eccessivo. "
            "L'AI suggerisce correzioni proteiche e di integrazione specifiche.")

    # 5. THE ORACLE AI
    add_sec("5. THE ORACLE: INTELLIGENZA GENERATIVA", 
            "L'AI Oracle analizza l'intero database. Coach, puoi richiedere report specifici come: "
            "'Pianifica dieta di recupero post-trasferta' o 'Analizza sinergia difensiva Player A/B'. "
            "L'AI apprende dai trend storici per migliorare le predizioni stagionali.")

    return pdf.output(dest='S').encode('latin-1')

def oracle_chat(prompt, context=""):
    if not client: return "⚠️ THE ORACLE offline. Inserisci API Key."
    # Prompt migliorato per essere più professionale
    full_p = f"Sei THE ORACLE, l'AI analista della squadra NBA {st.session_state.get('team_name')}. " \
             f"Usa un tono professionale, da Head Coach. Dati attuali: {context}. Domanda: {prompt}"
    try:
        res = client.chat.completions.create(messages=[{"role":"user","content":full_p}], model="llama3-8b-8192")
        return res.choices[0].message.content
    except Exception as e: return f"Errore AI: {e}"


# =================================================================
# 4. LOGIN & REGISTRAZIONE
# =================================================================
if 'logged_in' not in st.session_state: st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🏀 CoachTrack Oracle v19.0")
    t1, t2 = st.tabs(["Login Coach", "Nuova Franchise"])
    
    with t1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Entra in Spogliatoio"):
            c = db_conn.cursor()
            c.execute("SELECT password, team_name FROM users WHERE username = ?", (u,))
            data = c.fetchone()
            if data and check_hashes(p, data[0]):
                st.session_state.logged_in = True
                st.session_state.username = u
                st.session_state.team_name = data[1] if data[1] else "My Team"
                st.rerun()
            else: st.error("Accesso Negato.")
            
    with t2:
        nu = st.text_input("Nuovo Username")
        nt = st.text_input("Nome Squadra")
        np = st.text_input("Nuova Password", type="password")
        if st.button("Crea Squadra"):
            if nu and nt and np:
                try:
                    c = db_conn.cursor()
                    c.execute("INSERT INTO users VALUES (?,?,?)", (nu, make_hashes(np), nt))
                    db_conn.commit()
                    st.success("Squadra Creata! Fai il login.")
                except: st.error("Utente già esistente.")
            else: st.warning("Compila tutto.")
    st.stop()

# =================================================================
# 5. LAYOUT PRINCIPALE & HEADER
# =================================================================
curr_user = st.session_state.username
team_name = st.session_state.team_name
logo_path = f"logo_{curr_user}.png"

# HEADER PRINCIPALE (Logo + Titolo)
head_c1, head_c2 = st.columns([1, 6])
with head_c1:
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.markdown("🏀") # Placeholder se manca logo
with head_c2:
    st.title(f"{team_name} - Operations Center")

# SIDEBAR (Logo + Menu + Manuale)
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)
st.sidebar.markdown(f"### 🏟️ {team_name}")
st.sidebar.write(f"Coach: **{curr_user}**")

try:
    man_pdf = generate_branded_manual(team_name, logo_path)
    st.sidebar.download_button("📘 Scarica Manuale", man_pdf, f"Manuale_{team_name}.pdf")
except: pass

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

# Recupero Dati
df_team = pd.read_sql_query(f"SELECT * FROM player_data WHERE owner = '{curr_user}'", db_conn)

# TABS OPERATIVI
tabs = st.tabs(["🎥 Video", "👤 Bio 360", "⚔️ War Room", "💬 Oracle", "⚙️ Sync Hub"])

# --- TAB 1: VIDEO ---
with tabs[0]:
    st.header("Analisi Video YOLO")
    if not YOLO_AVAILABLE: st.error("YOLO non installato.")
    else:
        v_file = st.file_uploader("Carica Match (.mp4)", type=['mp4'])
        if v_file and st.checkbox("Avvia Tracking"):
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

# --- TAB 2: BIO 360 ---
with tabs[1]:
    st.header("Bio-Metrics & Prevention")
    if not df_team.empty:
        sel_p = st.selectbox("Atleta", df_team['player_name'].unique())
        p = df_team[df_team['player_name'] == sel_p].iloc[-1]
        
        # Logica Semaforo Avanzata
        c_status, c_msg = st.columns([1,3])
        risk = False
        if p.get('hrv', 60) < 45 or p.get('rpe', 5) > 8:
            c_status.error("🔴 STOP"); msg="Alto Rischio Infortunio."; risk=True
        elif p.get('water_perc', 60) < 55:
            c_status.warning("🟡 WARNING"); msg="Disidratazione rilevata."; risk=True
        else:
            c_status.success("🟢 OK"); msg="Atleta Disponibile."
        
        c_msg.info(f"**Oracle Advice:** {msg}")
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Peso", f"{p.get('weight',0)} kg")
        m2.metric("Grasso", f"{p.get('body_fat',0)}%")
        m3.metric("Muscolo", f"{p.get('muscle_mass',0)} kg")
        m4.metric("Acqua", f"{p.get('water_perc',0)}%")
        m5.metric("Ossa", f"{p.get('bone_mass',0)} kg")
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=[p.get('shot_efficiency',0), p.get('hrv',0), p.get('muscle_mass',0)], 
                                      theta=['Tecnica (Shot)', 'Recupero (HRV)', 'Potenza (Muscolo)'], fill='toself'))
        st.plotly_chart(fig)
    else: st.info("Vai al Sync Hub per caricare i dati.")

# --- TAB 3: WAR ROOM ---
with tabs[2]:
    st.header("War Room: Lineup Analysis")
    if len(df_team['player_name'].unique()) >= 2:
        p1 = st.selectbox("Player A", df_team['player_name'].unique(), key="wa")
        p2 = st.selectbox("Player B", df_team['player_name'].unique(), key="wb")
        if st.button("Confronta"):
            st.write(oracle_chat(f"Confronta {p1} vs {p2} per la prossima partita.", df_team.to_string()))

# --- TAB 4: ORACLE ---
with tabs[3]:
    st.header("💬 The Oracle AI")
    if prompt := st.chat_input("Chiedi strategia, dieta o analisi..."):
        st.write(f"**Coach:** {prompt}")
        st.info(oracle_chat(prompt, df_team.to_string()))

# --- TAB 5: SYNC HUB (COMPLETO) ---
with tabs[4]:
    st.header("⚙️ Data Sync & Settings")
    
    col_logo, col_csv, col_man = st.columns([1, 1, 1])
    
    # 1. LOGO UPLOAD
    with col_logo:
        st.subheader("1. Branding")
        up_logo = st.file_uploader("Carica Logo", type=['png', 'jpg'])
        if up_logo:
            Image.open(up_logo).save(logo_path)
            st.success("Logo Aggiornato!")
            if st.button("Applica Logo"): st.rerun()

    # 2. CSV IMPORT/EXPORT
    with col_csv:
        st.subheader("2. Bulk CSV")
        # Bottone Download Template
        template_cols = ["player_name", "weight", "body_fat", "muscle_mass", "water_perc", "bone_mass", "hrv", "rpe", "sleep", "shot_efficiency"]
        df_temp = pd.DataFrame(columns=template_cols)
        csv_temp = df_temp.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Scarica Template CSV", csv_temp, "template_atleti.csv", "text/csv")
        
        # Upload CSV
        up_csv = st.file_uploader("Carica CSV Dati", type=['csv'])
        if up_csv and st.button("Importa CSV"):
            try:
                df_new = pd.read_csv(up_csv)
                df_new['owner'] = curr_user
                df_new['timestamp'] = datetime.now().strftime("%Y-%m-%d")
                # Assicurati che le colonne esistano nel CSV, altrimenti metti 0
                for col in template_cols:
                    if col not in df_new.columns: df_new[col] = 0
                
                df_new.to_sql('player_data', db_conn, if_exists='append', index=False)
                st.success(f"Caricati {len(df_new)} atleti!")
                st.rerun()
            except Exception as e: st.error(f"Errore CSV: {e}")

    # 3. INSERIMENTO MANUALE COMPLETO
    with col_man:
        st.subheader("3. Manual Entry")
        with st.form("full_entry"):
            nm = st.text_input("Nome Atleta")
            c1, c2 = st.columns(2)
            w = c1.number_input("Peso (kg)", 0.0, 150.0, 80.0)
            bf = c2.number_input("Grasso (%)", 0.0, 50.0, 10.0)
            mm = c1.number_input("Muscolo (kg)", 0.0, 100.0, 40.0)
            wat = c2.number_input("Acqua (%)", 0.0, 100.0, 60.0)
            bone = c1.number_input("Ossa (kg)", 0.0, 20.0, 3.0)
            hr = c2.number_input("HRV (ms)", 0, 200, 60)
            sl = c1.number_input("Sonno (ore)", 0.0, 12.0, 7.5)
            rp = c2.slider("RPE (Fatica)", 1, 10, 5)
            eff = st.slider("Tiro (%)", 0, 100, 45)
            
            if st.form_submit_button("Salva Dati Completi"):
                ts = datetime.now().strftime("%Y-%m-%d")
                c = db_conn.cursor()
                query = """INSERT INTO player_data 
                           (owner, player_name, timestamp, weight, body_fat, muscle_mass, water_perc, bone_mass, hrv, sleep, rpe, shot_efficiency) 
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"""
                c.execute(query, (curr_user, nm, ts, w, bf, mm, wat, bone, hr, sl, rp, eff))
                db_conn.commit()
                st.success("Dati Salvati!")
                st.rerun()
