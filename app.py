import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.spatial import ConvexHull
from datetime import datetime

# =========================
# 1. SETUP RESPONSIVO
# =========================
st.set_page_config(page_title="CoachTrack Elite v2", layout="wide")

# Persistenza dati
if "player_names" not in st.session_state: st.session_state.player_names = {}
if "history" not in st.session_state: st.session_state.history = []
if "notes" not in st.session_state: st.session_state.notes = {}

# =========================
# 2. UI STYLING (NBA DARK THEME)
# =========================
st.markdown("""
<style>
    header {visibility: hidden;}
    .main { background-color: #0f172a; color: white; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: #1e293b; padding: 10px; border-radius: 12px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #334155; border-radius: 8px; color: white; padding: 0 20px; }
    .stTabs [aria-selected="true"] { background-color: #2563eb !important; border: none; }
    
    .kpi-card { background: #1e293b; padding: 20px; border-radius: 16px; border: 1px solid #334155; text-align: center; }
    .kpi-val { font-size: 32px; font-weight: 800; color: #38bdf8; }
    .kpi-label { font-size: 14px; color: #94a3b8; text-transform: uppercase; }
    
    .ai-box { background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%); padding: 25px; border-radius: 20px; border: 1px solid #3b82f6; }
</style>
""", unsafe_allow_html=True)

# =========================
# 3. MOTORE DI ANALISI ELITE
# =========================
def get_spacing(df):
    if len(df) < 3: return 0
    try: return ConvexHull(df[['x_m', 'y_m']].values).area
    except: return 0

def get_pro_insights(p_label, dist, vmax, role):
    """Genera analisi tattica approfondita."""
    status = "🔴 OVERLOAD" if dist > 3500 else "🟢 OPTIMAL"
    drills = {
        "Guardia": ["Pick & Roll Decision Making", "Transition 3s", "Pressure Dribbling"],
        "Ala": ["Corner Spacing Drills", "Closeout Attack", "Fastbreak Filling"],
        "Centro": ["Rim Protection Timing", "Post-up Footwork", "Screen Accuracy"]
    }
    d_list = drills.get(role, drills["Guardia"])
    
    report = f"""
    ### 🏀 ANALISI TATTICA: {p_label}
    **Status Carico**: {status} | **Ruolo**: {role}
    
    **Valutazione Tecnica**:
    - Il volume di {dist:.0f}m indica un impegno elevato nelle transizioni.
    - Picco di {vmax:.1f} km/h: {"Elite" if vmax > 25 else "Migliorabile"}.
    
    **Drills Consigliati per domani**:
    1. {d_list[0]} (15 min)
    2. {d_list[1]} (High Intensity)
    3. {d_list[2]} (Specifico Ruolo)
    """
    return report

# =========================
# 4. CARICAMENTO DATI
# =========================
@st.cache_data
def load_data():
    try: return pd.read_csv("data/virtual_uwb_realistic.csv")
    except: return pd.DataFrame({"player_id":["P1","P2"], "x_m":[0,5], "y_m":[0,5], "speed_kmh":[10,20], "quality_factor":[90,90]})

uwb = load_data()
all_pids = sorted(uwb["player_id"].unique())

# =========================
# 5. NAVIGAZIONE (NO SIDEBAR)
# =========================
st.title("🏀 CoachTrack Elite AI v2")

t_perf, t_tactic, t_team, t_setup = st.tabs(["📊 Player Performance", "🎯 Tactical Spacing", "📈 Team History", "⚙️ Setup & Nomi"])

# --- TAB SETUP (RIFACIMENTO NOMI) ---
with t_setup:
    st.header("Gestione Squadra")
    st.info("Qui puoi modificare i nomi dei giocatori e i loro ruoli. Le modifiche sono istantanee.")
    
    col_id, col_name, col_role = st.columns(3)
    for pid in all_pids:
        with col_id: st.text(f"ID: {pid}")
        with col_name: 
            st.session_state.player_names[pid] = st.text_input(f"Nome {pid}", value=st.session_state.player_names.get(pid, pid), key=f"n_{pid}")
        with col_role:
            if f"r_{pid}" not in st.session_state: st.session_state[f"r_{pid}"] = "Guardia"
            st.session_state[f"r_{pid}"] = st.selectbox(f"Ruolo {pid}", ["Guardia", "Ala", "Centro"], index=["Guardia", "Ala", "Centro"].index(st.session_state[f"r_{pid}"]), key=f"sel_{pid}")

# Mapping nomi reale
uwb["player_label"] = uwb["player_id"].map(st.session_state.player_names)
kpi = uwb.groupby('player_label').agg(Distanza=('speed_kmh', 'count'), Vel_Max=('speed_kmh', 'max')).reset_index()

# --- TAB PERFORMANCE ---
with t_perf:
    sel_p = st.selectbox("Seleziona Giocatore per il Report:", kpi['player_label'].unique())
    p_row = kpi[kpi['player_label'] == sel_p].iloc[0]
    p_role = st.session_state.get(f"r_{uwb[uwb['player_label']==sel_p]['player_id'].iloc[0]}", "Guardia")

    # KPI CARDS responsive
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Distanza</div><div class="kpi-val">{p_row["Distanza"]:.0f}m</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Top Speed</div><div class="kpi-val">{p_row["Vel_Max"]:.1f}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Stato</div><div class="kpi-val">{"OK" if p_row["Distanza"] < 3500 else "⚠️"}</div></div>', unsafe_allow_html=True)

    st.divider()
    
    # AI & NOTES
    col_ai, col_note = st.columns([1.5, 1])
    with col_ai:
        report_ai = get_pro_insights(sel_p, p_row['Distanza'], p_row['Vel_Max'], p_role)
        st.markdown(f'<div class="ai-box">{report_ai}</div>', unsafe_allow_html=True)
        
        # Download
        full_txt = f"REPORT COACHTRACK - {sel_p}\n{report_ai}\n\nNote Coach: {st.session_state.notes.get(sel_p, '')}"
        st.download_button(f"Scarica Report {sel_p}", data=full_txt, file_name=f"Report_{sel_p}.txt", use_container_width=True)

    with col_note:
        st.subheader("Note Tattiche del Coach")
        st.session_state.notes[sel_p] = st.text_area("Scrivi qui le tue osservazioni...", value=st.session_state.notes.get(sel_p, ""), height=250)
        st.success("Le note vengono salvate automaticamente per il report.")

# --- TAB TACTICAL ---
with t_tactic:
    c_s, c_m = st.columns([1, 2])
    with c_s:
        area = get_spacing(uwb.sample(n=min(5, len(uwb))))
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Spacing Index</div><div class="kpi-val">{area:.1f} m²</div><p>Area occupata dal quintetto</p></div>', unsafe_allow_html=True)
        st.info("Un'area superiore a 90m² indica una corretta occupazione degli angoli (Spacing d'élite).")
    with c_m:
        fig_heat = px.density_heatmap(uwb[uwb['player_label']==sel_p], x="x_m", y="y_m", nbinsx=30, nbinsy=15, range_x=[0,28], range_y=[0,15], color_continuous_scale="Viridis", title="Mappa di Tiro & Posizionamento")
        st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB HISTORY ---
with t_team:
    st.subheader("Andamento Carichi di Lavoro")
    if st.button("💾 Salva Sessione Odierna"):
        st.session_state.history.append({"date": datetime.now().strftime("%H:%M"), "avg_dist": kpi['Distanza'].mean()})
        st.toast("Dati salvati nello storico!")
    
    if st.session_state.history:
        h_df = pd.DataFrame(st.session_state.history)
        fig_h = px.line(h_df, x="date", y="avg_dist", title="Trend Distanza Media Squadra", markers=True)
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info("Nessuna sessione salvata. Clicca il tasto sopra dopo l'allenamento.")
