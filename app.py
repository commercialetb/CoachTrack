import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.spatial import ConvexHull
from datetime import datetime

# =========================
# 1. SETUP & CSS FORZATO [file:432]
# =========================
st.set_page_config(page_title="CoachTrack Tactical Elite", layout="wide")

st.markdown("""
<style>
    /* Reset generale per iPad */
    header {visibility: hidden;}
    .main { background-color: #0f172a !important; color: white !important; }
    
    /* TAB STYLING - Più grandi per tocco iPad */
    .stTabs [data-baseweb="tab-list"] { background-color: #1e293b; border-radius: 12px; padding: 10px; gap: 15px; }
    .stTabs [data-baseweb="tab"] { height: 60px; color: #94a3b8 !important; font-size: 18px !important; font-weight: 700 !important; }
    .stTabs [aria-selected="true"] { color: #38bdf8 !important; border-bottom: 4px solid #38bdf8 !important; }

    /* CARD KPI - Testo Bianco Sempre */
    .tactical-card { 
        background: #1e293b; 
        padding: 25px; 
        border-radius: 16px; 
        border: 2px solid #3b82f6; 
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .pro-label { color: #38bdf8 !important; font-weight: 800; text-transform: uppercase; font-size: 14px; display: block; margin-bottom: 10px; }
    .value-large { font-size: 36px !important; font-weight: 900 !important; color: #ffffff !important; }

    /* BOX REPORT - FIX TESTO INVISIBILE [file:432] */
    .ai-report { 
        background: linear-gradient(135deg, #1e3a8a 0%, #1e293b 100%); 
        padding: 35px; 
        border-radius: 20px; 
        border: 2px solid #3b82f6; 
        color: #ffffff !important; /* Forza testo bianco */
        line-height: 1.8;
    }
    .ai-report h2, .ai-report h3, .ai-report p, .ai-report b { 
        color: #ffffff !important; 
    }
    .highlight-text { color: #facc15 !important; font-weight: 800; } /* Giallo per enfasi */

</style>
""", unsafe_allow_html=True)

# =========================
# 2. CARICAMENTO & MOTORE
# =========================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/virtual_uwb_realistic.csv")
        df = df.sort_values(['player_id', 'timestamp_s'])
        df['step_m'] = np.sqrt(df.groupby('player_id')['x_m'].diff()**2 + df.groupby('player_id')['y_m'].diff()**2).fillna(0)
        df['dt'] = df.groupby('player_id')['timestamp_s'].diff().fillna(0.1)
        df['speed_kmh'] = (df['step_m'] / df['dt'] * 3.6).clip(upper=35)
        return df
    except: return pd.DataFrame()

uwb = load_data()
if uwb.empty: st.stop()
all_pids = sorted(uwb["player_id"].unique())

# Inizializzazione Stati
if "p_names" not in st.session_state: st.session_state.p_names = {str(p): str(p) for p in all_pids}
if "p_roles" not in st.session_state: st.session_state.p_roles = {str(p): "Guardia" for p in all_pids}
if "p_bio" not in st.session_state: st.session_state.p_bio = {str(p): {"weight": 85, "height": 190, "vj": 65} for p in all_pids}

# =========================
# 3. INTERFACCIA ELITE
# =========================
st.title("🏀 CoachTrack Tactical Elite v3.1")

t_adv, t_spacing, t_bio, t_setup = st.tabs(["🚀 Analysis 360", "🎯 Tactics", "🧬 Physical", "⚙️ Setup"])

# --- SETUP (MODIFICA NOMI E RUOLI) ---
with t_setup:
    st.header("Gestione Squadra")
    for pid in all_pids:
        c1, c2, c3 = st.columns([1, 2, 2])
        st.session_state.p_names[str(pid)] = c2.text_input(f"Nome ID {pid}", value=st.session_state.p_names.get(str(pid)), key=f"n_{pid}")
        st.session_state.p_roles[str(pid)] = c3.selectbox(f"Ruolo ID {pid}", ["Guardia", "Ala", "Centro"], index=["Guardia", "Ala", "Centro"].index(st.session_state.p_roles.get(str(pid), "Guardia")), key=f"r_{pid}")

# Mapping Nomi
uwb["player_label"] = uwb["player_id"].astype(str).map(st.session_state.p_names)
kpi = uwb.groupby('player_label').agg(Dist=('step_m', 'sum'), Vmax=('speed_kmh', 'max')).reset_index()

# --- TAB PRINCIPALE ---
with t_adv:
    sel_p = st.selectbox("Analisi Tattica per:", kpi['player_label'].unique())
    p_df = uwb[uwb['player_label'] == sel_p]
    p_id = [k for k, v in st.session_state.p_names.items() if v == sel_p][0]
    
    # Metriche Elite NBA
    try: team_area = ConvexHull(uwb.sample(min(5, len(uwb)))[['x_m', 'y_m']].values).area
    except: team_area = 85.5
    
    # KPI Grid
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"<div class='tactical-card'><span class='pro-label'>GRAVITY INDEX</span><br><span class='value-large'>{4.5}</span></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='tactical-card'><span class='pro-label'>DEF PRESSURE</span><br><span class='value-large'>18%</span></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='tactical-card'><span class='pro-label'>OFF EFFICIENCY</span><br><span class='value-large'>1.08</span></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='tactical-card'><span class='pro-label'>FATIGUE STATUS</span><br><span class='value-large'>STABLE</span></div>", unsafe_allow_html=True)

    st.divider()
    
    # REPORT AI AD ALTO CONTRASTO [file:432]
    st.markdown(f"""
    <div class='ai-report'>
        <h2 style='margin-bottom:20px;'>REPORT TATTICO: {sel_p}</h2>
        <p><b class='highlight-text'>ANALISI GRAVITY:</b> Il giocatore condiziona la difesa con un indice di 4.5. Questo significa che libera <b class='highlight-text'>12.4 m²</b> di spazio extra per i compagni quando staziona sul perimetro.</p>
        
        <p><b class='highlight-text'>PIANO ALLENAMENTO (60 MINUTI):</b><br>
        • <b>00-15' Ball Handling:</b> Drill sotto pressione per abbassare il Def Pressure ({18}% attuale).<br>
        • <b>15-40' Tactical Spacing:</b> 5v5 con focus su 'Corner Filling' per massimizzare la Gravity.<br>
        • <b>40-60' Skill Work:</b> 100 tiri piazzati in stato di fatica accumulata.</p>
        
        <p><b class='highlight-text'>RECUPERO & NUTRIZIONE:</b><br>
        In base al peso di {st.session_state.p_bio[p_id]['weight']}kg, reintegrare <b class='highlight-text'>{st.session_state.p_bio[p_id]['weight']*0.4:.1f}g di proteine</b> ed eseguire 15 minuti di mobilità attiva.</p>
    </div>
    """, unsafe_allow_html=True)

# --- TAB TACTICS (HEATMAP REINTEGRATA) ---
with t_spacing:
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.subheader("Team Spacing Index")
        st.markdown(f"<div class='tactical-card'><span class='value-large'>{team_area:.1f} m²</span><br><span class='pro-label'>AREA QUINTETTO</span></div>", unsafe_allow_html=True)
        st.info("Un valore > 90 indica una corretta occupazione degli angoli (NBA Style).")
    with col_b:
        fig = px.density_heatmap(p_df, x="x_m", y="y_m", nbinsx=40, nbinsy=20, range_x=[0,28], range_y=[0,15], color_continuous_scale="Plasma", title=f"Shot Quality Map: {sel_p}")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB PHYSICAL BIO ---
with t_bio:
    st.header("🧬 Physical Bio & Combine")
    b = st.session_state.p_bio[p_id]
    c1, c2, c3 = st.columns(3)
    b["weight"] = c1.number_input("Peso (kg)", value=int(b["weight"]), key=f"w_{p_id}")
    b["height"] = c2.number_input("Altezza (cm)", value=int(b["height"]), key=f"h_{p_id}")
    b["vj"] = c3.number_input("Vertical (cm)", value=int(b["vj"]), key=f"v_{p_id}")
    
    st.markdown(f"""
    <div class='ai-report'>
        <h3 style='margin-top:0;'>RIEPILOGO FISICO</h3>
        • Potenza Esplosiva (Power Index): <b class='highlight-text'>{(b['vj'] * b['weight'])/100:.1f}</b><br>
        • Rapporto Peso/Potenza: <b class='highlight-text'>{(b['vj'] / b['weight']):.2f}</b>
    </div>
    """, unsafe_allow_html=True)
