# =================================================================
# COACHTRACK ELITE AI v4.0 - PRO EDITION
# =================================================================

import sys
import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from pathlib import Path
import sqlite3
import json
import os
from collections import deque
import cv2

# Tentativo importazione fpdf per report
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

logging.basicConfig(level=logging.INFO)

# =================================================================
# DATABASE & PERSISTENCE LAYER (Novità v4.0)
# =================================================================
def init_db():
    conn = sqlite3.connect('coachtrack_v4.db')
    c = conn.cursor()
    # Tabella Biometria Estesa
    c.execute('''CREATE TABLE IF NOT EXISTS biometrics
                 (id INTEGER PRIMARY KEY, player_id TEXT, player_name TEXT, 
                  timestamp TEXT, weight_kg REAL, rpe INTEGER, hrv REAL, 
                  fatigue_score REAL, notes TEXT)''')
    # Tabella Eventi Video (Clips)
    c.execute('''CREATE TABLE IF NOT EXISTS video_clips
                 (id INTEGER PRIMARY KEY, filename TEXT, event_type TEXT, 
                  timestamp TEXT, player_id TEXT, duration REAL)''')
    conn.commit()
    conn.close()

def save_bio_db(data):
    conn = sqlite3.connect('coachtrack_v4.db')
    c = conn.cursor()
    c.execute("INSERT INTO biometrics (player_id, player_name, timestamp, weight_kg, rpe, hrv, fatigue_score, notes) VALUES (?,?,?,?,?,?,?,?)",
              (data['player_id'], data['player_name'], str(datetime.now()), 
               data['weight_kg'], data['rpe'], data['hrv'], data['fatigue_score'], data['notes']))
    conn.commit()
    conn.close()

def load_bio_db():
    conn = sqlite3.connect('coachtrack_v4.db')
    df = pd.read_sql_query("SELECT * FROM biometrics", conn)
    conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Inizializza DB all'avvio
init_db()

# =================================================================
# CORE UTILS & PDF REPORTING
# =================================================================
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'CoachTrack Elite - Match Report', 0, 1, 'C')
        self.ln(5)

def generate_report(stats, clips_count):
    if not PDF_AVAILABLE: return None
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Data Report: {datetime.now().strftime('%d/%m/%Y')}", ln=1, align='L')
    pdf.cell(200, 10, txt=f"Azioni Analizzate: {stats.get('total_actions', 0)}", ln=1)
    pdf.cell(200, 10, txt=f"Tiri Rilevati: {stats.get('total_shots', 0)}", ln=1)
    pdf.cell(200, 10, txt=f"Video Clip Generati: {clips_count}", ln=1)
    
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    pdf.output(filename)
    return filename

# =================================================================
# COMPUTER VISION & AUTO-CLIPPING (Novità v4.0)
# =================================================================
def save_video_clip(frames, fps, filename):
    if not frames: return False
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    return True

def add_computer_vision_tab():
    st.header("🎥 CV & Auto-Clipping")
    
    # Mock pipeline se non presente
    try:
        from cv_ai_advanced import CVAIPipeline
    except:
        class CVAIPipeline:
            def initialize(self): return True
            def process_frame(self, f): 
                # Simulazione detection casuale per demo
                import random
                if random.random() < 0.05: return {'action': 'shooting', 'bbox': [100,100,200,200]}
                return None

    cv_tab1, cv_tab2, cv_tab3 = st.tabs(["🔴 Live Analysis", "🎬 Clips Archive", "⚙️ Calibration"])

    with cv_tab1:
        st.subheader("Analisi Video con Auto-Clipping")
        uploaded_video = st.file_uploader("Carica Video Match", type=['mp4', 'mov'])
        
        enable_clipping = st.checkbox("✅ Attiva Auto-Clipping (Salva canestri)", value=True)
        
        if uploaded_video and st.button("🚀 Avvia Analisi"):
            tfile = f"temp_{uploaded_video.name}"
            with open(tfile, 'wb') as f: f.write(uploaded_video.read())
            
            cap = cv2.VideoCapture(tfile)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            
            # BUFFER PER CLIPPING (es. 2 sec prima, 2 sec dopo)
            buffer_seconds = 2
            buffer_size = fps * buffer_seconds
            frame_buffer = deque(maxlen=buffer_size)
            
            pipeline = CVAIPipeline()
            pipeline.initialize()
            
            st_frame = st.empty()
            st_stat = st.empty()
            
            events = []
            clips_generated = 0
            
            # Logica Clipping
            recording_post_event = 0
            current_clip_frames = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # AI Process
                result = pipeline.process_frame(frame)
                
                # Visualizzazione (disegno rettangolo fake)
                if result:
                    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
                    cv2.putText(frame, "ACTION DETECTED", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Gestione Buffer & Clipping
                frame_buffer.append(frame)
                
                if result and result.get('action') == 'shooting' and recording_post_event == 0:
                    # Inizia a registrare un clip
                    recording_post_event = fps * 2 # Registra per altri 2 secondi
                    current_clip_frames = list(frame_buffer) # Prendi i 2 sec precedenti
                    st.toast("🏀 Tiro Rilevato! Generazione Clip...")
                
                if recording_post_event > 0:
                    current_clip_frames.append(frame)
                    recording_post_event -= 1
                    if recording_post_event == 0:
                        # Salva clip
                        clip_name = f"clip_shot_{clips_generated}.mp4"
                        save_video_clip(current_clip_frames, fps, clip_name)
                        clips_generated += 1
                        events.append({'time': cap.get(cv2.CAP_PROP_POS_MSEC)/1000, 'file': clip_name})
                        current_clip_frames = []

                # Display (ridotto per performance)
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 5 == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
                    st_stat.metric("Clip Generati", clips_generated)

            cap.release()
            
            # Generazione Report
            if PDF_AVAILABLE:
                report_file = generate_report({'total_shots': clips_generated}, clips_generated)
                with open(report_file, "rb") as f:
                    st.download_button("📄 Scarica Report PDF", f, "report.pdf")

    with cv_tab2:
        st.subheader("Archivio Highlights")
        # Elenca file mp4 nella directory
        files = [f for f in os.listdir('.') if f.startswith('clip_shot_') and f.endswith('.mp4')]
        if not files:
            st.info("Nessuna clip trovata.")
        else:
            cols = st.columns(3)
            for i, f in enumerate(files):
                with cols[i % 3]:
                    st.video(f)
                    st.caption(f"Action #{i+1}")

# =================================================================
# BIOMETRIC MODULE - ACWR & LOAD MANAGEMENT (Novità v4.0)
# =================================================================
def render_biometric_module():
    st.header("⚖️ Biometria & Carico (ACWR)")
    
    # Carica dati dal DB
    df = load_bio_db()
    
    tab1, tab2 = st.tabs(["➕ Input Giornaliero", "📈 Analisi Carico"])
    
    with tab1:
        with st.form("bio_entry"):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Nome Giocatore *")
                weight = st.number_input("Peso (kg)", 50.0, 130.0, 80.0)
                hrv = st.number_input("HRV (ms) [Opzionale]", 10, 200, 60)
            with c2:
                rpe = st.slider("RPE (Sforzo Percepito 1-10)", 1, 10, 5)
                duration = st.number_input("Durata Allenamento (min)", 0, 180, 90)
                notes = st.text_area("Note Fisiche")
            
            submitted = st.form_submit_button("💾 Salva nel DB")
            if submitted and name:
                # Calcolo Load
                sRPE = rpe * duration # Session RPE
                import hashlib
                pid = hashlib.md5(name.encode()).hexdigest()[:8]
                
                data = {
                    'player_id': pid, 'player_name': name,
                    'weight_kg': weight, 'rpe': sRPE, 'hrv': hrv,
                    'fatigue_score': sRPE / hrv if hrv > 0 else 0,
                    'notes': notes
                }
                save_bio_db(data)
                st.success("✅ Dati salvati con successo nel Database SQLite")
                st.rerun()

    with tab2:
        if df.empty:
            st.warning("Inserisci dati per vedere l'analisi.")
        else:
            player_list = df['player_name'].unique()
            selected_player = st.selectbox("Analizza Giocatore", player_list)
            
            p_data = df[df['player_name'] == selected_player].sort_values('timestamp')
            
            if len(p_data) > 0:
                # Calcolo ACWR (Acute:Chronic Workload Ratio)
                # Acuto = media ultimi 7 gg, Cronico = media ultimi 28 gg
                # Simuliamo il calcolo
                p_data['Load'] = p_data['rpe'] # Qui usiamo RPE come proxy del load
                
                # Grafici
                fig = go.Figure()
                fig.add_trace(go.Bar(x=p_data['timestamp'], y=p_data['Load'], name='Carico Giornaliero'))
                fig.add_trace(go.Scatter(x=p_data['timestamp'], y=p_data['hrv'], name='HRV Trend', yaxis='y2', line=dict(color='red')))
                
                fig.update_layout(
                    title='Monitoraggio Carico vs Recupero (HRV)',
                    yaxis=dict(title='Carico (RPE x Min)'),
                    yaxis2=dict(title='HRV (ms)', overlaying='y', side='right'),
                    legend=dict(x=0, y=1.2, orientation='h')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Logic ACWR
                if len(p_data) >= 7:
                    acute = p_data['Load'].tail(7).mean()
                    chronic = p_data['Load'].tail(28).mean() if len(p_data) >= 28 else p_data['Load'].mean()
                    ratio = acute / chronic if chronic > 0 else 0
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Acute Load (7gg)", f"{acute:.0f}")
                    c2.metric("Chronic Load (28gg)", f"{chronic:.0f}")
                    c3.metric("ACWR Ratio", f"{ratio:.2f}", delta="Ottimale: 0.8-1.3")
                    
                    if ratio > 1.5:
                        st.error("🚨 PERICOLO INFORTUNIO: Carico aumentato troppo velocemente (>1.5)")
                    elif ratio < 0.8:
                        st.warning("⚠️ DETRAINING: Carico troppo basso")
                    else:
                        st.success("🟢 SWEET SPOT: Carico ottimale")

# =================================================================
# ANALYTICS & SHOT CHART (Novità v4.0)
# =================================================================
def draw_court(fig):
    # Funzione helper per disegnare il campo su Plotly
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor='white', width=600, height=500
    )
    # Aggiungi linee campo (semplificato)
    fig.add_shape(type="rect", x0=0, y0=0, x1=15, y1=28, line=dict(color="black")) # Perimetro
    fig.add_shape(type="circle", x0=6, y0=12.25, x1=9, y1=15.75, line=dict(color="black")) # Cerchio centrocampo
    return fig

def add_analytics_tab():
    st.header("📊 Advanced Analytics")
    
    tab1, tab2 = st.tabs(["🔥 Shot Chart & Heatmap", "🧠 Lineup Analysis"])
    
    with tab1:
        st.subheader("Mappa di Tiro")
        # Generazione dati fake per demo se non presenti
        x_shots = np.random.uniform(0, 15, 50) # Larghezza campo 15m
        y_shots = np.random.uniform(0, 14, 50) # Metà campo
        outcomes = np.random.choice(['Made', 'Missed'], 50)
        
        df_shots = pd.DataFrame({'x': x_shots, 'y': y_shots, 'Outcome': outcomes})
        
        fig = px.density_heatmap(df_shots, x='x', y='y', nbinsx=20, nbinsy=20, title="Densità Offensiva", color_continuous_scale="Viridis")
        fig = draw_court(fig)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.write("### Filtri")
            st.multiselect("Giocatori", ["Tutti", "#23", "#0"], default="Tutti")
            made_pct = len(df_shots[df_shots['Outcome']=='Made']) / len(df_shots) * 100
            st.metric("Percentuale dal campo", f"{made_pct:.1f}%")

    with tab2:
        st.info("🧠 Lineup Analysis: Analizza l'efficienza delle combinazioni di 5 giocatori.")
        st.markdown("""
        | Lineup | Minuti | +/- | Off Rtg | Def Rtg | Net Rtg |
        | :--- | :---: | :---: | :---: | :---: | :---: |
        | Smith, Jones, Green, White, Black | 12:30 | +8 | 115.4 | 102.1 | **+13.3** |
        | Smith, Jones, Doe, White, Black | 08:15 | -2 | 98.2 | 105.0 | **-6.8** |
        """)

# =================================================================
# MAIN ENTRY POINT
# =================================================================
st.set_page_config(page_title="CoachTrack Elite v4", page_icon="🏀", layout="wide")

if 'logged_in' not in st.session_state: st.session_state.logged_in = False

# Sidebar Login
with st.sidebar:
    st.title("🏀 CoachTrack PRO")
    if not st.session_state.logged_in:
        u = st.text_input("User")
        p = st.text_input("Pass", type="password")
        if st.button("Login"):
            if u == "admin" and p == "admin":
                st.session_state.logged_in = True
                st.rerun()
    else:
        st.success(f"Loggato come Admin")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

if st.session_state.logged_in:
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "🎥 Video AI & Clips", "⚖️ Biometria & Carico", "📊 Tattica"])
    
    with tab1:
        st.title("Benvenuto Coach.")
        st.metric("Status Sistema", "Online", delta="v4.0 Ready")
        
    with tab2:
        add_computer_vision_tab()
        
    with tab3:
        render_biometric_module()
        
    with tab4:
        add_analytics_tab()
else:
    st.info("Effettua il login dalla sidebar per accedere alla suite PRO.")
