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
# 1. CONFIGURAZIONE E DATABASE
# =================================================================
st.set_page_config(page_title="CoachTrack NBA Ultimate", layout="wide", page_icon="🏀")

def init_db():
    """Inizializza il database con supporto per tutte le metriche v3.2 + NBA"""
    conn = sqlite3.connect('coachtrack_ultimate.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS player_data 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, player_name TEXT, timestamp TEXT,
                  weight REAL, fat REAL, muscle REAL, water REAL, bone REAL, bmr REAL, 
                  hrv REAL, rpe INTEGER, sleep REAL, qsq REAL, 
                  ai_diet TEXT, ai_risk TEXT, scout_report TEXT)''')
    conn.commit()
    return conn

db_conn = init_db()

# =================================================================
# 2. AI & PDF LOGIC
# =================================================================
with st.sidebar:
    st.title("🏀 NBA Front-Office")
    groq_key = st.text_input("Groq API Key", type="password", help="Necessaria per Dieta e Scouting")
    st.markdown("---")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

client = Groq(api_key=groq_key) if groq_key else None

def get_ai_response(prompt_type, data):
    if not client: return "⚠️ API Key mancante."
    
    prompts = {
        "diet": f"Sei un nutrizionista NBA. Crea una dieta per: {data}. Includi macro e timing dei pasti.",
        "risk": f"Sei un trainer NBA. Valuta rischio infortuni: HRV {data['hrv']}, RPE {data['rpe']}, Sonno {data['sleep']}.",
        "scout": f"Sei un GM NBA. Crea scouting report per {data['name']}. Note: {data['notes']}. Includi NBA Comp."
    }
    try:
        res = client.chat.completions.create(messages=[{"role":"user","content":prompts[prompt_type]}], model="llama3-8b-8192")
        return res.choices[0].message.content
    except: return "Errore di connessione AI."

def create_pdf_report(title, content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, content.encode('latin-1', 'ignore').decode('latin-1'))
    return pdf.output(dest='S').encode('latin-1')

# =================================================================
# 3. INTERFACCIA UTENTE
# =================================================================

# Login (come in v3.2)
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if not st.session_state.logged_in:
    st.title("🏀 CoachTrack Elite Login")
    u = st.text_input("Username", value="admin")
    p = st.text_input("Password", type="password", value="admin")
    if st.button("Login", type="primary"):
        if u == "admin" and p == "admin":
            st.session_state.logged_in = True
            st.rerun()
    st.stop()

tabs = st.tabs(["🎥 Video & Spacing", "⚖️ Bio-Intelligence", "📊 NBA Analytics", "🔎 Scouting AI", "💬 Tactical Chat"])

# --- TAB 1: VIDEO (YOLO + SPACING) ---
with tabs[0]:
    st.header("Video Analysis & Spacing Geometry")
    uv = st.file_uploader("Carica Match", type=['mp4','mov'])
    if uv:
        with open("temp.mp4", "wb") as f: f.write(uv.read())
        cap = cv2.VideoCapture("temp.mp4")
        st_frame = st.empty()
        if st.button("Esegui Tracking YOLO"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, (800, 450))
                # Mock Tracking NBA
                cv2.putText(frame, "LIVE SPACING: 64.2m2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                st_frame.image(frame, channels="BGR")
                time.sleep(0.01) # Previene il freeze di Streamlit
            cap.release()

# --- TAB 2: BIOMETRICS (Full v3.2 Integration) ---
with tabs[1]:
    st.header("Monitoraggio Biometrico & Nutrizione")
    with st.form("full_bio_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Nome Giocatore")
            w = st.number_input("Peso (kg)", 50.0, 150.0, 90.0)
            f = st.number_input("Grasso (%)", 5.0, 35.0, 11.0)
        with c2:
            m = st.number_input("Muscolo (kg)", 30.0, 90.0, 45.0)
            h = st.number_input("HRV (ms)", 20.0, 150.0, 65.0)
            s = st.number_input("Sonno (ore)", 4.0, 12.0, 8.0)
        with c3:
            rpe = st.slider("Fatica (RPE)", 1, 10, 5)
            bmr = st.number_input("BMR (kcal)", 1500, 3500, 2100)
            submit = st.form_submit_button("Genera Analisi NBA")
            
        if submit:
            with st.spinner("AI al lavoro..."):
                bio_data = {"weight": w, "fat": f, "hrv": h, "rpe": rpe, "sleep": s, "bmr": bmr}
                diet = get_ai_response("diet", bio_data)
                risk = get_ai_response("risk", bio_data)
                
                cur = db_conn.cursor()
                cur.execute('''INSERT INTO player_data (player_name, timestamp, weight, fat, muscle, hrv, rpe, sleep, bmr, ai_diet, ai_risk) 
                               VALUES (?,?,?,?,?,?,?,?,?,?,?)''', 
                            (name, datetime.now().strftime("%Y-%m-%d"), w, f, m, h, rpe, s, bmr, diet, risk))
                db_conn.commit()
                st.success(f"Analisi per {name} completata!")

# --- TAB 3: NBA ANALYTICS (Radar & Spacing) ---
with tabs[2]:
    st.header("Performance Radar & Load Management")
    df = pd.read_sql_query("SELECT * FROM player_data", db_conn)
    if not df.empty:
        sel = st.selectbox("Seleziona Atleta", df['player_name'].unique())
        latest = df[df['player_name'] == sel].iloc[-1]
        
        col_l, col_r = st.columns(2)
        with col_l:
            # Radar Chart NBA
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[latest['muscle'], latest['hrv'], latest['sleep']*10, (11-latest['rpe'])*10, 70],
                theta=['Muscolo', 'HRV', 'Sonno', 'Recupero', 'Idratazione'],
                fill='toself', line_color='orange'
            ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="NBA Readiness Score")
            st.plotly_chart(fig)
                        
        with col_r:
            st.subheader("⚠️ Injury Risk Advisor")
            st.error(latest['ai_risk'])
            st.subheader("🥗 Piano Nutrizionale")
            st.info(latest['ai_diet'])
            
            # Export
            pdf_data = create_pdf_report(f"Report Atleta: {sel}", latest['ai_diet'])
            st.download_button("📥 Scarica Report PDF", pdf_data, f"Report_{sel}.pdf")

# --- TAB 4: SCOUTING & DRAFT ---
with tabs[3]:
    st.header("🔎 Scouting & Draft Intelligence")
    col_1, col_2 = st.columns(2)
    with col_1:
        s_name = st.text_input("Nome Prospetto")
        s_notes = st.text_area("Note (Tiro, Fisico, Mentalità...)")
        if st.button("Genera Scouting Report"):
            report = get_ai_response("scout", {"name": s_name, "notes": s_notes})
            st.session_state.current_scout = report
            st.markdown(report)
    if "current_scout" in st.session_state:
        with col_2:
            st.subheader("Draft Grade")
            fig_s = px.line_polar(r=[80, 70, 90, 60, 85], theta=['Tiro', 'Difesa', 'Fisico', 'Passaggio', 'IQ'], line_close=True)
            st.plotly_chart(fig_s)
            s_pdf = create_pdf_report(f"Scouting Report: {s_name}", st.session_state.current_scout)
            st.download_button("📥 Scarica Scout PDF", s_pdf, f"Scout_{s_name}.pdf")

# --- TAB 5: TACTICAL CHAT ---
with tabs[4]:
    st.header("💬 AI Tactical Assistant")
    q = st.text_input("Chiedi all'AI basandoti sui dati della squadra")
    if q and client:
        hist = df.tail(10).to_string()
        ans = client.chat.completions.create(
            messages=[{"role":"system","content":f"Sei un assistente NBA. Dati: {hist}"}, {"role":"user","content":q}],
            model="llama3-8b-8192"
        )
        st.chat_message("assistant").write(ans.choices[0].message.content)
