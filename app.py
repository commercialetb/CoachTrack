import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.spatial import ConvexHull
from datetime import datetime

# =========================
# 1. SETUP & TEMA CHIARO [file:444]
# =========================
st.set_page_config(page_title="CoachTrack Elite Light v3.3", layout="wide")

st.markdown("""
<style>
    header {visibility: hidden;}
    .main { background-color: #f8fafc !important; color: #1e293b !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: #ffffff; border-radius: 12px; padding: 10px; border: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] { height: 60px; color: #64748b !important; font-size: 18px !important; font-weight: 700 !important; }
    .stTabs [aria-selected="true"] { color: #2563eb !important; border-bottom: 4px solid #2563eb !important; }
    .predictive-card { 
        background: #ffffff; padding: 25px; border-radius: 16px; border: 1px solid #e2e8f0; 
        text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin-bottom: 15px;
    }
    .metric-title { color: #64748b !important; font-weight: 800; text-transform: uppercase; font-size: 13px; display: block; margin-bottom: 8px; }
    .metric-value { font-size: 36px !important; font-weight: 900 !important; color: #1e293b !important; }
    .ai-report-light { 
        background: #ffffff; padding: 35px; border-radius: 20px; border: 1px solid #2563eb; 
        color: #1e293b !important; line-height: 1.8; box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.1);
    }
    .highlight-blue { color: #2563eb !important; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# =========================
# 2. CARICAMENTO DATI
# =========================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/virtual_uwb_realistic.csv")
        df['speed_kmh'] = np.random.uniform(10, 28, len(df))
        return df
    except: return pd.DataFrame()

uwb = load_data()
if uwb.empty: st.stop()
all_pids = sorted(uwb["player_id"].unique())

# --- INIZIALIZZAZIONE STATO (RISOLVE KEYERROR) ---
if "p_names" not in st.session_state: st.session_state.p_names = {str(p): str(p) for p in all_pids}
if "p_bio" not in st.session_state: st.session_state.p_bio = {}

for pid in all_pids:
    pid_s = str(pid)
    if pid_s not in st.session_state.p_bio:
        st.session_state.p_bio[pid_s] = {"weight": 85, "height": 190, "vj": 60, "wingspan": 195}

# =========================
# 3. DASHBOARD
# =========================
st.title("🏀 CoachTrack Elite AI v3.3")
t_pred, t_tactic, t_bio, t_setup = st.tabs(["🔮 AI Prediction", "🎯 Tactical", "🧬 Physical", "⚙️ Setup"])

# --- SETUP ---
with t_setup:
    st.header("Configurazione Squadra")
    for pid in all_pids:
        pid_s = str(pid)
        c1, c2 = st.columns([1, 2])
        st.session_state.p_names[pid_s] = c2.text_input(f"Nome ID {pid_s}", value=st.session_state.p_names.get(pid_s, pid_s), key=f"n_{pid_s}")

uwb["player_label"] = uwb["player_id"].astype(str).map(st.session_state.p_names)

# --- TAB PREDICTION ---
with t_pred:
    # Metriche predittive simulate
    real_a = ConvexHull(uwb.sample(5)[['x_m', 'y_m']].values).area if len(uwb) > 5 else 0
    pred_a = real_a * 1.12
    win_p = 0.45 + (real_a / 300)

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"<div class='predictive-card'><span class='metric-title'>Real Spacing Index</span><div class='metric-value'>{real_a:.1f} m²</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='predictive-card'><span class='metric-title'>xSpacing (Next 2s)</span><div class='metric-value'>{pred_a:.1f} m²</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='predictive-card'><span class='metric-title'>Scoring Prob.</span><div class='metric-value'>{(win_p*100):.1f}%</div></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='ai-report-light'>
        <h3 style='color:#2563eb; margin-top:0;'>🤖 AI PREDICTIVE INSIGHTS</h3>
        • <b class='highlight-blue'>Analisi:</b> Spacing previsto in espansione. Probabilità canestro in aumento.<br>
        • <b class='highlight-blue'>Consiglio:</b> Sfruttare la Gravity dei tiratori per isolamenti in post basso.
    </div>
    """, unsafe_allow_html=True)

# --- TAB TATTICA ---
with t_tactic:
    sel_p = st.selectbox("Seleziona Giocatore:", uwb["player_label"].unique())
    fig = px.density_heatmap(uwb[uwb['player_label']==sel_p], x="x_m", y="y_m", nbinsx=40, nbinsy=20, range_x=[0,28], range_y=[0,15], color_continuous_scale="Viridis", title="Heatmap Posizionamento")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB BIO (FISICO) ---
with t_bio:
    st.header("🧬 Physical Profile")
    # Trova ID da nome
    curr_id = [k for k, v in st.session_state.p_names.items() if v == sel_p][0]
    b = st.session_state.p_bio[curr_id]
    
    c1, c2, c3 = st.columns(3)
    b["weight"] = c1.number_input("Peso (kg)", value=int(b.get("weight", 85)), key=f"w_{curr_id}")
    b["height"] = c2.number_input("Altezza (cm)", value=int(b.get("height", 190)), key=f"h_{curr_id}")
    b["vj"] = c3.number_input("Vertical (cm)", value=int(b.get("vj", 60)), key=f"v_{curr_id}")
    
    st.session_state.p_bio[curr_id] = b
    
    st.markdown(f"""
    <div class='ai-report-light'>
        <h4 style='color:#2563eb; margin-top:0;'>RIEPILOGO ATLETICO</h4>
        • Power Index: <b class='highlight-blue'>{(b['vj'] * b['weight'])/100:.1f}</b><br>
        • Target Reintegro: <b class='highlight-blue'>{b['weight']*0.4:.1f}g Proteine</b>
    </div>
    """, unsafe_allow_html=True)
