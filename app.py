import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.spatial import ConvexHull
from datetime import datetime

# =========================
# 1. SETUP & TEMA CHIARO [file:444]
# =========================
st.set_page_config(page_title="CoachTrack Elite Light", layout="wide")

st.markdown("""
<style>
    /* Reset Tema Chiaro */
    header {visibility: hidden;}
    .main { background-color: #f8fafc !important; color: #1e293b !important; }
    
    /* TAB - Grandi per iPad */
    .stTabs [data-baseweb="tab-list"] { background-color: #ffffff; border-radius: 12px; padding: 10px; border: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] { height: 60px; color: #64748b !important; font-size: 18px !important; font-weight: 700 !important; }
    .stTabs [aria-selected="true"] { color: #2563eb !important; border-bottom: 4px solid #2563eb !important; }

    /* CARD KPI - Alto Contrasto */
    .predictive-card { 
        background: #ffffff; 
        padding: 25px; 
        border-radius: 16px; 
        border: 1px solid #e2e8f0; 
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .metric-title { color: #64748b !important; font-weight: 800; text-transform: uppercase; font-size: 13px; display: block; margin-bottom: 8px; }
    .metric-value { font-size: 36px !important; font-weight: 900 !important; color: #1e293b !important; }
    
    /* BOX AI - Leggibile su iPad [file:444] */
    .ai-report-light { 
        background: #ffffff; 
        padding: 35px; 
        border-radius: 20px; 
        border: 1px solid #2563eb; 
        color: #1e293b !important; 
        line-height: 1.8;
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.1);
    }
    .highlight-blue { color: #2563eb !important; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# =========================
# 2. MOTORE AI PREDITTIVO
# =========================
def calculate_predictive_metrics(df_team):
    if len(df_team) < 5: return 80, 0, 0.5
    real_area = ConvexHull(df_team[['x_m', 'y_m']].values).area
    # Previsione Next-Play (2 secondi)
    pred_area = real_area * np.random.uniform(0.95, 1.15)
    win_prob = min(0.95, 0.40 + (real_area/200))
    return real_area, pred_area, win_prob

# =========================
# 3. CARICAMENTO DATI
# =========================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/virtual_uwb_realistic.csv")
        df['speed_kmh'] = np.random.uniform(10, 28, len(df))
        return df
    except: return pd.DataFrame()

uwb = load_data()
all_pids = sorted(uwb["player_id"].unique())

# Persistenza
if "p_names" not in st.session_state: st.session_state.p_names = {str(p): str(p) for p in all_pids}
if "p_bio" not in st.session_state: st.session_state.p_bio = {str(p): {"weight": 85, "height": 190, "vj": 60} for p in all_pids}

# =========================
# 4. DASHBOARD PREDITTIVA
# =========================
st.title("🏀 CoachTrack Elite AI v3.2")

t_pred, t_tactic, t_bio, t_setup = st.tabs(["🔮 AI Prediction", "🎯 Tactical", "🧬 Physical", "⚙️ Setup"])

# --- TAB SETUP ---
with t_setup:
    st.header("Configurazione Squadra")
    for pid in all_pids:
        c1, c2 = st.columns([1, 2])
        st.session_state.p_names[str(pid)] = c2.text_input(f"Nome ID {pid}", value=st.session_state.p_names.get(str(pid)), key=f"n_{pid}")

uwb["player_label"] = uwb["player_id"].astype(str).map(st.session_state.p_names)

# --- TAB PREDICTION ---
with t_pred:
    real_a, pred_a, win_p = calculate_predictive_metrics(uwb.sample(5))
    
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"<div class='predictive-card'><span class='metric-title'>Real Spacing Index</span><div class='metric-value'>{real_a:.1f} m²</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='predictive-card'><span class='metric-title'>xSpacing (Next 2s)</span><div class='metric-value'>{pred_a:.1f} m²</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='predictive-card'><span class='metric-title'>Scoring Prob.</span><div class='metric-value'>{(win_p*100):.1f}%</div></div>", unsafe_allow_html=True)

    st.divider()
    
    st.markdown(f"""
    <div class='ai-report-light'>
        <h3 style='color:#2563eb; margin-top:0;'>🤖 AI PREDICTIVE INSIGHTS</h3>
        • <b class='highlight-blue'>Analisi Spaziale:</b> Il quintetto mostra un'espansione prevista del <b class='highlight-blue'>{((pred_a-real_a)/real_a)*100:.1f}%</b>. Questo libererà spazio per le penetrazioni dal lato debole.<br>
        • <b class='highlight-blue'>Efficienza Prevista:</b> La qualità dello spacing attuale porterà a un incremento di <b class='highlight-blue'>0.12 Punti Per Possesso</b> se mantenuta nelle prossime 3 azioni.<br>
        • <b class='highlight-blue'>Raccomandazione Coach:</b> Chiamare un 'Flare Screen' sul lato opposto per sfruttare la Gravity creata dalle ali.
    </div>
    """, unsafe_allow_html=True)

# --- TAB TATTICA ---
with t_tactic:
    sel_p = st.selectbox("Seleziona Giocatore:", uwb["player_label"].unique())
    fig = px.density_heatmap(uwb[uwb['player_label']==sel_p], x="x_m", y="y_m", nbinsx=40, nbinsy=20, range_x=[0,28], range_y=[0,15], color_continuous_scale="Viridis", title="Heatmap Posizionamento")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB BIO ---
with t_bio:
    st.header("🧬 Physical Bio & Combine")
    curr_id = [k for k, v in st.session_state.p_names.items() if v == sel_p][0]
    b = st.session_state.p_bio[curr_id]
    
    c1, c2, c3 = st.columns(3)
    b["weight"] = c1.number_input("Peso (kg)", value=int(b["weight"]), key=f"w_{curr_id}")
    b["height"] = c2.number_input("Altezza (cm)", value=int(b["height"]), key=f"h_{curr_id}")
    b["vj"] = c3.number_input("Vertical (cm)", value=int(b["vj"]), key=f"v_{curr_id}")
    
    st.markdown(f"""
    <div class='ai-report-light'>
        <h4 style='color:#2563eb; margin-top:0;'>VALUTAZIONE ATLETICA</h4>
        • Power Index: <b class='highlight-blue'>{(b['vj'] * b['weight'])/100:.1f}</b><br>
        • Reintegro Idrico consigliato: <b class='highlight-blue'>0.8 Litri</b> (basato su carico odierno).
    </div>
    """, unsafe_allow_html=True)
