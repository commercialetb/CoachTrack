import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# =========================
# 1. SETUP & PERSISTENZA
# =========================
st.set_page_config(page_title="CoachTrack Elite Combine", layout="wide")

# Inizializzazione Session State per dati fisici e nomi
if "player_names" not in st.session_state: st.session_state.player_names = {}
if "player_roles" not in st.session_state: st.session_state.player_roles = {}
if "player_bio" not in st.session_state: st.session_state.player_bio = {}

# =========================
# 2. UI STYLING
# =========================
st.markdown("""
<style>
    header {visibility: hidden;}
    .main { background-color: #0f172a; color: white; }
    .bio-card { background: #1e293b; padding: 15px; border-radius: 12px; border: 1px solid #3b82f6; margin-bottom: 10px; }
    .bio-val { color: #38bdf8; font-weight: 800; font-size: 20px; }
    .ai-box { background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); padding: 25px; border-radius: 20px; border: 1px solid #3b82f6; color: white; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# =========================
# 3. CARICAMENTO & CALCOLO (CORRETTO)
# =========================
@st.cache_data
def load_and_process():
    try:
        df = pd.read_csv("data/virtual_uwb_realistic.csv")
        df = df.sort_values(['player_id', 'timestamp_s'])
        
        # Calcolo distanze (step_m) e velocità (speed_kmh) obbligatori
        df['step_m'] = np.sqrt(df.groupby('player_id')['x_m'].diff()**2 + df.groupby('player_id')['y_m'].diff()**2).fillna(0)
        df['dt'] = df.groupby('player_id')['timestamp_s'].diff().fillna(0.1)
        df['speed_kmh'] = (df['step_m'] / df['dt'] * 3.6).clip(upper=35)
        
        return df
    except Exception as e:
        st.error(f"Errore caricamento dati: {e}")
        return pd.DataFrame()

uwb = load_and_process()
if uwb.empty: st.stop()

all_pids = sorted(uwb["player_id"].unique())

# Inizializzazione valori default se mancano
for pid in all_pids:
    pid_s = str(pid)
    if pid_s not in st.session_state.player_names: st.session_state.player_names[pid_s] = pid_s
    if pid_s not in st.session_state.player_roles: st.session_state.player_roles[pid_s] = "Guardia"
    if pid_s not in st.session_state.player_bio:
        st.session_state.player_bio[pid_s] = {"height": 190, "weight": 85, "wingspan": 195, "vertical": 65, "fat": 10}

# =========================
# 4. NAVIGAZIONE TAB
# =========================
t_perf, t_bio, t_setup = st.tabs(["📊 Performance AI 360", "🧬 Profilo Fisico", "⚙️ Team Setup"])

# --- TAB SETUP ---
with t_setup:
    st.header("Configurazione Squadra")
    for pid in all_pids:
        pid_s = str(pid)
        c1, c2, c3 = st.columns([1, 2, 2])
        c1.text(f"ID: {pid_s}")
        st.session_state.player_names[pid_s] = c2.text_input(f"Nome {pid_s}", value=st.session_state.player_names[pid_s], key=f"n_{pid_s}")
        st.session_state.player_roles[pid_s] = c3.selectbox(f"Ruolo {pid_s}", ["Guardia", "Ala", "Centro"], index=["Guardia", "Ala", "Centro"].index(st.session_state.player_roles[pid_s]), key=f"r_{pid_s}")

# --- TAB BIO-PHYSICAL ---
with t_bio:
    st.header("🧬 Physical Profile & Combine Data")
    sel_bio = st.selectbox("Seleziona Giocatore:", [st.session_state.player_names[str(p)] for p in all_pids], key="bio_sel")
    # Trova l'ID corrispondente al nome
    curr_pid = [p for p, name in st.session_state.player_names.items() if name == sel_bio][0]
    
    b = st.session_state.player_bio[curr_pid]
    
    c1, c2, c3 = st.columns(3)
    b["height"] = c1.number_input("Altezza (cm)", value=int(b["height"]), key=f"h_{curr_pid}")
    b["weight"] = c2.number_input("Peso (kg)", value=int(b["weight"]), key=f"w_{curr_pid}")
    b["wingspan"] = c3.number_input("Wingspan (cm)", value=int(b["wingspan"]), key=f"ws_{curr_pid}")
    
    c4, c5 = st.columns(2)
    b["vertical"] = c4.number_input("Vertical Jump (cm)", value=int(b["vertical"]), key=f"vj_{curr_pid}")
    b["fat"] = c5.number_input("Body Fat (%)", value=int(b["fat"]), key=f"bf_{curr_pid}")
    
    st.session_state.player_bio[curr_pid] = b
    
    st.markdown(f"""
    <div class='bio-card'>
        <b>RIEPILOGO ATLETICO:</b><br>
        • Rapporto Wingspan/Altezza: <span class='bio-val'>{b['wingspan']/b['height']:.2f}</span><br>
        • Power Index (Esplosività): <span class='bio-val'>{(b['vertical'] * b['weight'])/100:.1f}</span>
    </div>
    """, unsafe_allow_html=True)

# --- TAB PERFORMANCE (CORRETTA) ---
# Applichiamo il mapping nomi PRIMA del groupby
uwb["player_label"] = uwb["player_id"].astype(str).map(st.session_state.player_names)
kpi = uwb.groupby('player_label').agg(
    Distanza=('step_m', 'sum'),
    Vel_Max=('speed_kmh', 'max')
).reset_index()

with t_perf:
    sel_p = st.selectbox("Analisi Performance:", kpi['player_label'].unique(), key="perf_sel")
    p_row = kpi[kpi['player_label'] == sel_p].iloc[0]
    
    # Recupero Bio e Ruolo
    p_id = [p for p, name in st.session_state.player_names.items() if name == sel_p][0]
    bio = st.session_state.player_bio[p_id]
    ruolo = st.session_state.player_roles[p_id]

    # Dashboard 360°
    st.markdown(f"""
    <div class='ai-box'>
        <h2 style='color:#38bdf8'>ELITE REPORT 360: {sel_p}</h2>
        <div style='display:flex; gap:15px; margin-bottom:20px;'>
            <div class='bio-card'><b>Profilo:</b> {ruolo} | {bio['height']}cm | {bio['weight']}kg</div>
            <div class='bio-card'><b>Carico:</b> {p_row['Distanza']:.0f}m | Max {p_row['Vel_Max']:.1f}km/h</div>
        </div>

        <h4 style='color:#38bdf8'>🚀 OTTIMIZZAZIONE PRESTAZIONE</h4>
        • <b>Nutrizione:</b> In base al peso ({bio['weight']}kg), assumere {bio['weight']*0.4:.1f}g di proteine post-sforzo.<br>
        • <b>Idratazione:</b> Reintegrare {p_row['Distanza']/1000 * 0.7:.1f} Litri d'acqua.<br>
        • <b>Recupero:</b> {"Crioterapia necessaria" if p_row['Distanza'] > 3500 else "Stretching e Foam Roller"} per 15 min.<br>
        
        <h4 style='color:#38bdf8'>🎯 PIANO D'ALLENAMENTO (60 MIN)</h4>
        - <b>Warm-up (10'):</b> Mobilità specifica per {ruolo}.<br>
        - <b>Speed (15'):</b> 5x20m sprint con 45" recupero.<br>
        - <b>Skill (25'):</b> 100 tiri piazzati + 50 in movimento.<br>
        - <b>Cool-down (10'):</b> Defaticamento e analisi video sessione.
    </div>
    """, unsafe_allow_html=True)
    
    # Download
    full_rep = f"REPORT COMPLETO: {sel_p}\nRuolo: {ruolo}\nDati Bio: {bio}\nTracking: Distanza {p_row['Distanza']:.0f}m, Max Speed {p_row['Vel_Max']:.1f}km/h"
    st.download_button(f"Scarica Report 360° {sel_p}", data=full_rep, file_name=f"CoachTrack_360_{sel_p}.txt", use_container_width=True)
