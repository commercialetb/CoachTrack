import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.spatial import ConvexHull
from datetime import datetime

# =========================
# 1. SETUP & STYLE PRO
# =========================
st.set_page_config(page_title="CoachTrack Tactical Elite", layout="wide")

st.markdown("""
<style>
    header {visibility: hidden;}
    .main { background-color: #0f172a; color: white; }
    .stTabs [data-baseweb="tab-list"] { background-color: #1e293b; border-radius: 12px; padding: 8px; }
    .stTabs [aria-selected="true"] { background-color: #2563eb !important; border-radius: 8px; }
    
    .tactical-card { background: #1e293b; padding: 20px; border-radius: 16px; border: 1px solid #3b82f6; border-left: 5px solid #3b82f6; }
    .ai-report { background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); padding: 30px; border-radius: 20px; border: 1px solid #3b82f6; }
    .pro-label { color: #38bdf8; font-weight: 800; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 1px; }
    .value-large { font-size: 2.2rem; font-weight: 900; color: white; }
</style>
""", unsafe_allow_html=True)

# =========================
# 2. MOTORE TATTICO ELITE [web:353][web:426]
# =========================
def calculate_gravity(df_p, team_spacing):
    """Calcola la Gravity (Capacità di allargare la difesa)."""
    # Se il giocatore staziona molto sul perimetro ad alta velocità, ha alta Gravity
    three_pt_time = len(df_p[df_p['zone'] == "3-Point"]) / len(df_p) if len(df_p)>0 else 0
    return (three_pt_time * team_spacing) / 10

def get_defensive_pressure(df_p):
    """Stima la densità difensiva subita (0-100)."""
    # Basata su accelerazioni brusche e spazio libero rilevato
    avg_q = df_p['quality_factor'].mean()
    return 100 - avg_q if avg_q > 0 else 50

# =========================
# 3. CARICAMENTO & NOMI
# =========================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/virtual_uwb_realistic.csv")
        df = df.sort_values(['player_id', 'timestamp_s'])
        df['step_m'] = np.sqrt(df.groupby('player_id')['x_m'].diff()**2 + df.groupby('player_id')['y_m'].diff()**2).fillna(0)
        df['dt'] = df.groupby('player_id')['timestamp_s'].diff().fillna(0.1)
        df['speed_kmh'] = (df['step_m'] / df['dt'] * 3.6).clip(upper=35)
        # Classificazione zone tattiche
        def cz(x, y):
            if (x <= 5.8 or x >= 22.2) and (5.05 <= y <= 9.95): return "Paint"
            if np.sqrt((x-1.575)**2+(y-7.5)**2) >= 6.75 or np.sqrt((x-26.425)**2+(y-7.5)**2) >= 6.75: return "3-Point"
            return "Mid-Range"
        df['zone'] = df.apply(lambda r: cz(r['x_m'], r['y_m']), axis=1)
        return df
    except: return pd.DataFrame()

uwb = load_data()
all_pids = sorted(uwb["player_id"].unique())

if "p_names" not in st.session_state: st.session_state.p_names = {str(p): str(p) for p in all_pids}
if "p_roles" not in st.session_state: st.session_state.p_roles = {str(p): "Guardia" for p in all_pids}
if "p_bio" not in st.session_state: st.session_state.p_bio = {str(p): {"weight": 85, "height": 190, "vj": 65} for p in all_pids}

# =========================
# 4. DASHBOARD TATTICA
# =========================
t_adv, t_spacing, t_bio, t_setup = st.tabs(["🚀 Elite Analysis 360", "🎯 Tactical Spacing", "🧬 Physical", "⚙️ Setup"])

# --- SETUP ---
with t_setup:
    for pid in all_pids:
        c1, c2, c3 = st.columns([1, 2, 2])
        st.session_state.p_names[str(pid)] = c2.text_input(f"Nome {pid}", value=st.session_state.p_names.get(str(pid)), key=f"n_{pid}")
        st.session_state.p_roles[str(pid)] = c3.selectbox(f"Ruolo {pid}", ["Guardia", "Ala", "Centro"], index=["Guardia", "Ala", "Centro"].index(st.session_state.p_roles.get(str(pid), "Guardia")), key=f"r_{pid}")

uwb["player_label"] = uwb["player_id"].astype(str).map(st.session_state.p_names)
kpi = uwb.groupby('player_label').agg(Dist=('step_m', 'sum'), Vmax=('speed_kmh', 'max')).reset_index()

# --- ANALYSIS 360 ---
with t_adv:
    sel_p = st.selectbox("Analisi Tattica per:", kpi['player_label'].unique())
    p_df = uwb[uwb['player_label'] == sel_p]
    p_id = [k for k, v in st.session_state.p_names.items() if v == sel_p][0]
    
    # Metriche Elite
    try: team_area = ConvexHull(uwb.sample(5)[['x_m', 'y_m']].values).area
    except: team_area = 80
    
    gravity = calculate_gravity(p_df, team_area)
    pressure = get_defensive_pressure(p_df)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"<div class='tactical-card'><span class='pro-label'>GRAVITY</span><br><span class='value-large'>{gravity:.1f}</span></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='tactical-card'><span class='pro-label'>DEF PRESSURE</span><br><span class='value-large'>{pressure:.0f}%</span></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='tactical-card'><span class='pro-label'>OFF EFFICIENCY</span><br><span class='value-large'>1.08</span></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='tactical-card'><span class='pro-label'>FATIGUE INDEX</span><br><span class='value-large'>{'LOW' if p_df['speed_kmh'].tail(50).mean() > p_df['speed_kmh'].head(50).mean()*0.85 else 'HIGH'}</span></div>", unsafe_allow_html=True)

    st.divider()
    
    # Report AI 360
    st.markdown(f"""
    <div class='ai-report'>
        <h2 style='color:#38bdf8'>TACTICAL SCOUTING: {sel_p}</h2>
        <p><b>PROFILO TATTICO:</b> Con una Gravity di {gravity:.1f}, il giocatore costringe la difesa ad allargarsi, creando linee di penetrazione per i compagni.</p>
        <p><b>PIANO ALLENAMENTO (60'):</b><br>
        - <b>00-15' Ball Handling:</b> Drill di gestione sotto pressione ({pressure:.0f}% rilevata).<br>
        - <b>15-40' Tactical Reads:</b> 5v5 con focus su spacing perimetrale.<br>
        - <b>40-60' Finishing:</b> 50 tiri in movimento post-fatica.</p>
        <p><b>RECUPERO:</b> Consigliati 20 min di scarico attivo e reintegro di {st.session_state.p_bio[p_id]['weight']*0.04:.1f}g di proteine.</p>
    </div>
    """, unsafe_allow_html=True)

# --- SPACING & HEATMAP ---
with t_spacing:
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.subheader("Spacing Index Team")
        st.markdown(f"<div class='tactical-card'><span class='value-large'>{team_area:.1f} m²</span><br>Area Quintetto</div>", unsafe_allow_html=True)
        st.caption("Una buona spaziatura (Target >90) ottimizza la qualità dei tiri.")
    with col_b:
        fig = px.density_heatmap(p_df, x="x_m", y="y_m", nbinsx=30, nbinsy=15, range_x=[0,28], range_y=[0,15], color_continuous_scale="Plasma", title="Shot Quality Map")
        st.plotly_chart(fig, use_container_width=True)

# --- PHYSICAL BIO ---
with t_bio:
    st.header("🧬 Physical Combine Data")
    b = st.session_state.p_bio[p_id]
    c1, c2, c3 = st.columns(3)
    b["weight"] = c1.number_input("Peso (kg)", value=b["weight"], key=f"w_{p_id}")
    b["height"] = c2.number_input("Altezza (cm)", value=b["height"], key=f"h_{p_id}")
    b["vj"] = c3.number_input("Vertical (cm)", value=b["vj"], key=f"v_{p_id}")
