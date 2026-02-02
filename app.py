import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# =========================
# 1. CONFIGURAZIONE & STATO
# =========================
st.set_page_config(page_title="CoachTrack Elite Combine", layout="wide")

# Persistenza dati estesa
if "player_names" not in st.session_state: st.session_state.player_names = {}
if "player_roles" not in st.session_state: st.session_state.player_roles = {}
if "player_bio" not in st.session_state: st.session_state.player_bio = {} # Altezza, Peso, Wingspan, Vertical, Fat%

# =========================
# 2. UI STYLING
# =========================
st.markdown("""
<style>
    header {visibility: hidden;}
    .main { background-color: #0f172a; color: white; }
    .bio-card { background: #1e293b; padding: 15px; border-radius: 12px; border: 1px solid #3b82f6; margin-bottom: 10px; }
    .bio-val { color: #38bdf8; font-weight: 800; font-size: 20px; }
    .ai-box { background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); padding: 25px; border-radius: 20px; border: 1px solid #3b82f6; }
</style>
""", unsafe_allow_html=True)

# =========================
# 3. CARICAMENTO DATI
# =========================
@st.cache_data
def load_data():
    try: return pd.read_csv("data/virtual_uwb_realistic.csv")
    except: return pd.DataFrame({"player_id":["P1"],"x_m":[0],"y_m":[0],"speed_kmh":[0],"timestamp_s":[0]})

uwb = load_data()
all_pids = sorted(uwb["player_id"].unique())

# =========================
# 4. NAVIGAZIONE TAB
# =========================
t_perf, t_bio, t_setup = st.tabs(["📊 Performance AI", "🧬 Bio-Physical Profile", "⚙️ Team Setup"])

# --- TAB SETUP ---
with t_setup:
    st.header("Configurazione Team")
    for pid in all_pids:
        c1, c2, c3 = st.columns([1, 2, 2])
        st.session_state.player_names[str(pid)] = c2.text_input(f"Nome per {pid}", value=st.session_state.player_names.get(str(pid), str(pid)), key=f"n_{pid}")
        st.session_state.player_roles[str(pid)] = c3.selectbox(f"Ruolo per {pid}", ["Guardia", "Ala", "Centro"], index=["Guardia", "Ala", "Centro"].index(st.session_state.player_roles.get(str(pid), "Guardia")), key=f"r_{pid}")

# --- TAB BIO-PHYSICAL ---
with t_bio:
    st.header("🧬 Physical Profile & Combine Data")
    st.info("Inserisci i parametri fisici per calcolare l'efficienza biomeccanica.")
    
    sel_bio = st.selectbox("Seleziona Giocatore:", [st.session_state.player_names.get(str(p), str(p)) for p in all_pids])
    pid_bio = [p for p, name in st.session_state.player_names.items() if name == sel_bio][0]
    
    # Inizializza bio se vuoto
    if pid_bio not in st.session_state.player_bio:
        st.session_state.player_bio[pid_bio] = {"height": 190, "weight": 85, "wingspan": 195, "vertical": 60, "fat": 10}
    
    b = st.session_state.player_bio[pid_bio]
    
    col1, col2, col3 = st.columns(3)
    b["height"] = col1.number_input("Altezza (cm)", value=b["height"], key=f"h_{pid_bio}")
    b["weight"] = col2.number_input("Peso (kg)", value=b["weight"], key=f"w_{pid_bio}")
    b["wingspan"] = col3.number_input("Wingspan (cm)", value=b["wingspan"], key=f"ws_{pid_bio}")
    
    col4, col5 = st.columns(2)
    b["vertical"] = col4.number_input("Vertical Jump (cm)", value=b["vertical"], key=f"vj_{pid_bio}")
    b["fat"] = col5.number_input("Body Fat (%)", value=b["fat"], key=f"bf_{pid_bio}")
    
    st.session_state.player_bio[pid_bio] = b
    
    # Visualizzazione Radar/Card
    st.markdown(f"""
    <div class='bio-card'>
        <span style='color:#94a3b8'>PUNTI DI FORZA BIOMECCANICI:</span><br>
        • Rapporto Altezza/Wingspan: <span class='bio-val'>{b['wingspan']/b['height']:.2f}</span> (NBA Target: >1.03)<br>
        • Power Index: <span class='bio-val'>{ (b['vertical'] * b['weight']) / 100:.1f}</span> (Potenza esplosiva stimata)
    </div>
    """, unsafe_allow_html=True)

# --- TAB PERFORMANCE (AGGIORNATA) ---
uwb["player_label"] = uwb["player_id"].astype(str).map(st.session_state.player_names)
kpi = uwb.groupby('player_label').agg(Distanza=('speed_kmh', 'count'), Vel_Max=('speed_kmh', 'max')).reset_index()

with t_perf:
    sel_p = st.selectbox("Analisi Performance:", kpi['player_label'].unique(), key="perf_sel")
    p_row = kpi[kpi['player_label'] == sel_p].iloc[0]
    curr_pid = [p for p, name in st.session_state.player_names.items() if name == sel_p][0]
    curr_bio = st.session_state.player_bio.get(curr_pid, {"height": 0, "weight": 0, "wingspan": 0, "vertical": 0, "fat": 0})
    
    # Dashboard AI potenziata con dati FISICI
    st.markdown(f"""
    <div class='ai-box'>
        <h2 style='color:#38bdf8'>ELITE ANALYSIS: {sel_p}</h2>
        <div style='display:flex; gap:20px; margin-bottom:15px;'>
            <div class='bio-card'><b>FISICO:</b> {curr_bio['height']}cm | {curr_bio['weight']}kg | Fat {curr_bio['fat']}%</div>
            <div class='bio-card'><b>POTENZA:</b> Vertical {curr_bio['vertical']}cm</div>
        </div>
        
        <p><b>VALUTAZIONE AI:</b></p>
        L'atleta presenta un Body Fat del {curr_bio['fat']}%, ideale per un ruolo di {st.session_state.player_roles.get(curr_pid)}. 
        Con una velocità max di {p_row['Vel_Max']:.1f} km/h e un salto di {curr_bio['vertical']}cm, il profilo esplosivo è 
        {"ECCELLENTE" if p_row['Vel_Max'] > 25 and curr_bio['vertical'] > 70 else "IN SVILUPPO"}.
        
        <p><b>PIANO RECUPERO & NUTRIZIONE:</b></p>
        In base al peso ({curr_bio['weight']}kg) e ai {p_row['Distanza']:.0f}m percorsi, si consiglia un reintegro di 
        <b>{curr_bio['weight'] * 0.04:.1f}g di proteine</b> e <b>{p_row['Distanza']/1000 * 0.6:.1f}L di acqua</b> entro 45 minuti.
    </div>
    """, unsafe_allow_html=True)
    
    st.download_button("⬇️ Scarica Report Completo (Dati Fisici + Tracking)", 
                       data=f"REPORT {sel_p}\nFisico: {curr_bio}\nTracking: {p_row.to_dict()}", 
                       file_name=f"FullReport_{sel_p}.txt")
