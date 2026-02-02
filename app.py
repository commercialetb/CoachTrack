import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import ConvexHull

# 1. CONFIGURAZIONE PAGINA
st.set_page_config(page_title="CoachTrack Elite AI Complete", layout="wide")

# 2. STILE CSS (Tema Chiaro iPad [file:444])
st.markdown("""
<style>
    header {visibility: hidden;}
    .main { background-color: #f8fafc !important; color: #1e293b !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: #ffffff; border-radius: 12px; padding: 10px; border: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] { height: 60px; color: #64748b !important; font-size: 18px !important; font-weight: 700 !important; }
    .stTabs [aria-selected="true"] { color: #2563eb !important; border-bottom: 4px solid #2563eb !important; }
    .predictive-card { background: #ffffff; padding: 25px; border-radius: 16px; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin-bottom: 15px; }
    .metric-title { color: #64748b !important; font-weight: 800; text-transform: uppercase; font-size: 13px; display: block; margin-bottom: 8px; }
    .metric-value { font-size: 36px !important; font-weight: 900 !important; color: #1e293b !important; }
    .ai-report-light { background: #ffffff; padding: 30px; border-radius: 20px; border: 1px solid #2563eb; color: #1e293b !important; line-height: 1.6; }
    .highlight-blue { color: #2563eb !important; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# 3. FUNZIONI CORE
def draw_court_lines():
    """Linee campo FIBA"""
    return [
        dict(type="rect", x0=0, y0=0, x1=28, y1=15, line=dict(color="#64748b", width=2)),
        dict(type="line", x0=14, y0=0, x1=14, y1=15, line=dict(color="#64748b", width=2)),
        dict(type="circle", x0=12.2, y0=5.7, x1=15.8, y1=9.3, line=dict(color="#64748b", width=2))
    ]

def classify_zone(x, y):
    if x <= 5.8 and 5.05 <= y <= 9.95: return 'Paint'
    if np.sqrt((x-1.575)**2 + (y-7.5)**2) >= 6.75: return '3-Point'
    return 'Mid-Range'

# 4. CARICAMENTO & STATO
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/virtual_uwb_realistic.csv")
        df['speed_kmh'] = np.random.uniform(5, 25, len(df))
        df['zone'] = df.apply(lambda r: classify_zone(r['x_m'], r['y_m']), axis=1)
        return df
    except: return pd.DataFrame()

uwb = load_data()
if uwb.empty: st.error("File CSV non trovato in data/"); st.stop()

all_pids = sorted(uwb["player_id"].unique())
if "p_names" not in st.session_state: st.session_state.p_names = {str(p): f"Player {p}" for p in all_pids}
if "p_bio" not in st.session_state: st.session_state.p_bio = {str(p): {"w": 85, "h": 190, "vj": 60} for p in all_pids}

# 5. UI PRINCIPALE
st.title("🏀 CoachTrack Elite AI")

t_pred, t_tactic, t_phys, t_setup = st.tabs(["🔮 Spacing Index", "🎯 Zone Analysis", "🧬 Physical", "⚙️ Setup"])

# --- TAB SETUP ---
with t_setup:
    st.header("Anagrafica Atleti")
    for pid in all_pids:
        pid_s = str(pid)
        c1, c2 = st.columns([1, 3])
        st.session_state.p_names[pid_s] = c2.text_input(f"Nome ID {pid_s}", value=st.session_state.p_names[pid_s], key=f"n_{pid_s}")

# Aggiorna nomi nel DF
uwb["player_label"] = uwb["player_id"].astype(str).map(st.session_state.p_names)

# --- TAB SPACING (PREDICTIVE) ---
with t_pred:
    # Calcolo Spacing Quintetto (simulazione predittiva) [web:433]
    q_data = uwb.sample(5)[['x_m', 'y_m']].values
    real_area = ConvexHull(q_data).area
    pred_area = real_area * 1.15 # Previsione AI +15%
    win_prob = min(0.95, 0.40 + (real_area/250))

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='predictive-card'><span class='metric-title'>Real Spacing</span><div class='metric-value'>{real_area:.1f} m²</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='predictive-card'><span class='metric-title'>xSpacing (2s)</span><div class='metric-value'>{pred_area:.1f} m²</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='predictive-card'><span class='metric-title'>Prob. Canestro</span><div class='metric-value'>{(win_p*100 if 'win_p' in locals() else win_prob*100):.1f}%</div></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='ai-report-light'>
        <h3 style='color:#2563eb; margin-top:0;'>🤖 AI PREDICTIVE REPORT</h3>
        • <b>Analisi:</b> Spacing previsto in aumento del <span class='highlight-blue'>15%</span> nelle prossime transizioni.<br>
        • <b>Tattica:</b> Ottima occupazione degli angoli. Si prevede un aumento della Gravity per il portatore di palla.
    </div>
    """, unsafe_allow_html=True)

# --- TAB ZONE ANALYSIS ---
with t_tactic:
    sel_p = st.selectbox("Seleziona Giocatore:", uwb["player_label"].unique())
    p_df = uwb[uwb['player_label'] == sel_p]
    
    col_z1, col_z2 = st.columns(2)
    with col_z1:
        z_stats = p_df['zone'].value_counts(normalize=True).mul(100).reset_index()
        fig_pie = px.pie(z_stats, values='proportion', names='zone', title="Shooting/Position Zones", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_z2:
        fig_heat = px.density_heatmap(p_df, x="x_m", y="y_m", range_x=[0,28], range_y=[0,15], color_continuous_scale="Viridis", title="Heatmap Densità")
        fig_heat.update_layout(shapes=draw_court_lines())
        st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB PHYSICAL ---
with t_phys:
    pid_curr = [k for k,v in st.session_state.p_names.items() if v == sel_p][0]
    b = st.session_state.p_bio[pid_curr]
    
    col_b1, col_b2, col_b3 = st.columns(3)
    b["w"] = col_b1.number_input("Peso (kg)", value=int(b["w"]), key=f"w_{pid_curr}")
    b["h"] = col_b2.number_input("Altezza (cm)", value=int(b["h"]), key=f"h_{pid_curr}")
    b["vj"] = col_b3.number_input("Vertical (cm)", value=int(b["vj"]), key=f"v_{pid_curr}")
    
    bmi = b["w"] / ((b["h"]/100)**2)
    p_idx = (b["vj"] * b["w"]) / 100
    
    st.markdown(f"""
    <div class='ai-report-light'>
        <h4 style='color:#2563eb; margin-top:0;'>🧬 PERFORMANCE BIO-PROFILE</h4>
        • BMI: <span class='highlight-blue'>{bmi:.1f}</span> | Power Index: <span class='highlight-blue'>{p_idx:.1f}</span><br>
        • Reintegro Idrico Stimato: <span class='highlight-blue'>{(b['w']*0.04):.1f} L/sessione</span>
    </div>
    """, unsafe_allow_html=True)
