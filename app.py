import streamlit as st
import pandas as pd
import numpy as np
import cv2
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sqlite3
import time
import os
from fpdf import FPDF
from groq import Groq

# =================================================================
# 1. CONFIGURAZIONE E DATABASE
# =================================================================
st.set_page_config(page_title="CoachTrack Elite NBA v6.5", layout="wide", page_icon="🏀")

def init_db():
    conn = sqlite3.connect('coachtrack_nba_v6.db', check_same_thread=False)
    c = conn.cursor()
    # Tabella Unificata: Biometria + NBA Stats + AI
    c.execute('''CREATE TABLE IF NOT EXISTS nba_data 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, player_name TEXT, timestamp TEXT,
                  weight REAL, fat REAL, muscle REAL, water REAL, bone REAL, bmr REAL, 
                  hrv REAL, rpe INTEGER, sleep REAL, qsq REAL, 
                  ai_diet TEXT, ai_risk TEXT, scout_report TEXT)''')
    conn.commit()
    return conn

db_conn = init_db()

# --- PDF GENERATOR CLASS ---
class CoachReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'COACHTRACK ELITE - OFFICIAL NBA REPORT', 0, 1, 'C')
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

# =================================================================
# 2. INTEGRAZIONE AI (GROQ)
# =================================================================
with st.sidebar:
    st.title("🏀 NBA Control Center")
    groq_key = st.text_input("Groq API Key", type="password")
    st.markdown("---")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

client = Groq(api_key=groq_key) if groq_key else None

def get_ai_insight(prompt_type, data):
    if not client: return "⚠️ Inserire API Key per attivare l'AI."
    
    prompts = {
        "diet": f"Sei un nutrizionista NBA. Crea dieta post-gara per: {data}. Specifica macro e integratori.",
        "risk": f"Sei un trainer NBA. Valuta rischio infortuni per: HRV {data['hrv']}, RPE {data['rpe']}, Sonno {data['sleep']}.",
        "scout": f"Sei un GM NBA. Crea report scout per il prospetto {data['name']}. Note: {data['notes']}. Includi NBA Comp e proiezione Draft."
    }
    
    try:
        res = client.chat.completions.create(messages=[{"role":"user","content":prompts[prompt_type]}], model="llama3-8b-8192")
        return res.choices[0].message.content
    except Exception as e: return f"Errore AI: {str(e)}"

# =================================================================
# 3. INTERFACCIA PRINCIPALE
# =================================================================

# --- LOGIN ---
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if not st.session_state.logged_in:
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.title("🏀 CoachTrack Login")
        u = st.text_input("User", value="admin")
        p = st.text_input("Pass", type="password", value="admin")
        if st.button("Entra"):
            if u=="admin" and p=="admin": 
                st.session_state.logged_in = True
                st.rerun()
    st.stop()

# --- TABS ---
tab_vid, tab_bio, tab_nba, tab_scout, tab_chat = st.tabs([
    "🎥 Video Tracking", "⚖️ Bio-Intelligence", "📊 NBA Analytics", "🔎 Scouting & Draft", "💬 Tactical AI"
])

# --- TAB: VIDEO TRACKING (YOLO Integration) ---
with tab_vid:
    st.header("Computer Vision & Play Recognition")
    vid = st.file_uploader("Carica Filmato Match", type=['mp4', 'mov'])
    if vid:
        with open("temp.mp4", "wb") as f: f.write(vid.read())
        cap = cv2.VideoCapture("temp.mp4")
        st_frame = st.empty()
        if st.button("Avvia Analisi Spaziale"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, (800, 450))
                # Mock YOLO Circles
                cv2.circle(frame, (400, 200), 10, (0, 255, 0), -1) # Ball
                cv2.putText(frame, "SPACING: OPTIMAL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                st_frame.image(frame, channels="BGR")
                time.sleep(0.01)
            cap.release()

# --- TAB: BIOMETRICS & DIETA AI ---
with tab_bio:
    st.header("Full Biometrics (v3.2) + AI Nutrition")
    with st.form("bio_nba"):
        c1,c2,c3 = st.columns(3)
        with c1:
            name = st.text_input("Giocatore")
            weight = st.number_input("Peso (kg)", 50.0, 150.0, 90.0)
            fat = st.number_input("Grasso (%)", 5.0, 30.0, 10.0)
        with c2:
            muscle = st.number_input("Muscolo (kg)", 30.0, 100.0, 45.0)
            water = st.number_input("Acqua (%)", 40.0, 80.0, 60.0)
            bmr = st.number_input("BMR (kcal)", 1500, 4000, 2200)
        with c3:
            hrv = st.number_input("HRV (ms)", 20, 150, 60)
            rpe = st.slider("RPE Fatica", 1, 10, 5)
            sleep = st.number_input("Ore Sonno", 4.0, 12.0, 8.0)
        
        if st.form_submit_button("💾 Salva e Genera Dieta AI"):
            data_ai = {"weight":weight, "fat":fat, "bmr":bmr, "hrv":hrv, "rpe":rpe, "sleep":sleep}
            diet = get_ai_insight("diet", data_ai)
            risk = get_ai_insight("risk", data_ai)
            
            cur = db_conn.cursor()
            cur.execute('''INSERT INTO nba_data (player_name, timestamp, weight, fat, muscle, water, bmr, hrv, rpe, sleep, ai_diet, ai_risk) 
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''', 
                        (name, datetime.now().strftime("%Y-%m-%d"), weight, fat, muscle, water, bmr, hrv, rpe, sleep, diet, risk))
            db_conn.commit()
            st.success("Dati Bio-Intelligence sincronizzati!")

# --- TAB: NBA ANALYTICS (Spacing, Gravity, Radar) ---
with tab_nba:
    st.header("Advanced Analytics & Radar Charts")
    df = pd.read_sql_query("SELECT * FROM nba_data", db_conn)
    if not df.empty:
        sel_p = st.selectbox("Seleziona Atleta per Analisi", df['player_name'].unique())
        p_data = df[df['player_name'] == sel_p].iloc[-1]
        
        col_l, col_r = st.columns(2)
        with col_l:
            # RADAR CHART PERFORMANCE
            categories = ['Muscolo', 'HRV', 'Recupero', 'Sonno', 'Idratazione']
            values = [p_data['muscle'], p_data['hrv'], (11-p_data['rpe'])*8, p_data['sleep']*10, p_data['water']]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', line_color='red'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 120])), title="NBA Readiness Radar")
            st.plotly_chart(fig)
            
            
        with col_r:
            st.subheader("🛡️ Load Management (Injury Risk)")
            st.warning(p_data['ai_risk'])
            
            st.subheader("🍎 Dieta Generata dall'AI")
            st.info(p_data['ai_diet'])
            
            # PDF Download
            pdf = CoachReport()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt=f"DIETA E RECUPERO - {sel_p}\n\n{p_data['ai_diet']}")
            pdf_bytes = pdf.output(dest='S').encode('latin-1', 'ignore')
            st.download_button("📥 Scarica Report Dieta PDF", data=pdf_bytes, file_name=f"Dieta_{sel_p}.pdf")

# --- TAB: SCOUTING & DRAFT ---
with tab_scout:
    st.header("🔎 Scouting & Draft Intelligence")
    col_v, col_t = st.columns(2)
    with col_v:
        s_name = st.text_input("Nome Prospetto")
        s_notes = st.text_area("Note Tecniche (Tiro, Fisico, Difesa...)")
        if st.button("Genera Scouting Report AI"):
            with st.spinner("L'AI sta valutando il prospetto..."):
                report = get_ai_insight("scout", {"name":s_name, "notes":s_notes})
                st.session_state.last_scout = report
                st.markdown(report)
                
    if "last_scout" in st.session_state:
        with col_t:
            st.subheader("Visual Draft Grade")
            fig_scout = px.line_polar(r=[80, 90, 70, 85, 60], theta=['Slasher', 'Shooting', 'Defense', 'IQ', 'Strength'], line_close=True)
            st.plotly_chart(fig_scout)
            
            # PDF Scouting
            pdf_s = CoachReport()
            pdf_s.add_page()
            pdf_s.multi_cell(0, 10, txt=f"SCOUTING REPORT: {s_name}\n\n{st.session_state.last_scout}")
            st.download_button("📥 Scarica Scouting Report PDF", data=pdf_s.output(dest='S').encode('latin-1', 'ignore'), file_name="Scout_Report.pdf")

# --- TAB: AI TACTICAL CHAT ---
with tab_chat:
    st.header("💬 AI Assistant Coach")
    user_q = st.text_input("Chiedi consiglio all'AI: (es. Chi è il più stanco della squadra?)")
    if user_q and client:
        context = df.tail(10).to_string()
        res = client.chat.completions.create(
            messages=[{"role":"system","content":f"Sei un assistente NBA. Dati squadra: {context}"}, {"role":"user","content":user_q}],
            model="llama3-8b-8192"
        )
        st.chat_message("assistant").write(res.choices[0].message.content)
