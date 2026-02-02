import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================
# 1. Configurazione Pagina
# =========================
st.set_page_config(
    page_title="CoachTrack | Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# 2. CSS Personalizzato (Fix Leggibilità & Toolbar)
# =========================
st.markdown("""
<style>
/* Rimuove toolbar Streamlit (Fork/GitHub) */
header[data-testid="stHeader"] { visibility: hidden; }

/* Dashboard Header Bianco */
.ct-header-box {
  background-color: white !important;
  border-radius: 14px;
  padding: 1.2rem;
  margin-bottom: 1.5rem;
  border: 1px solid #e0e0e0;
}
.ct-header-box h2, .ct-header-box div { color: #111827 !important; }

/* KPI Cards: Testo NERO su fondo bianco */
div[data-testid="stMetric"] {
  background-color: white !important;
  border: 1px solid #e0e0e0 !important;
  border-radius: 12px !important;
  padding: 15px !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
}
div[data-testid="stMetricLabel"] > div {
  color: #4B5563 !important;
  font-weight: 600 !important;
}
div[data-testid="stMetricValue"] > div {
  color: #111827 !important;
  font-weight: 800 !important;
}

/* AI Box */
.ai-insight {
  background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
  padding: 1.2rem;
  border-radius: 14px;
  color: white !important;
}

/* iPad Optimization */
.block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# 3. Funzioni Utility
# =========================
def get_label(pid):
    """Converte ID (P1) in Nome (Marco) tramite lo stato della sessione."""
    return st.session_state.player_name_map.get(str(pid), str(pid))

def draw_basketball_court():
    court_length, court_width = 28.0, 15.0
    shapes = []
    shapes.append(dict(type="rect", x0=0, y0=0, x1=court_length, y1=court_width, line=dict(color="white", width=3)))
    shapes.append(dict(type="line", x0=court_length/2, y0=0, x1=court_length/2, y1=court_width, line=dict(color="white", width=2)))
    # Aree e cerchi
    shapes.append(dict(type="circle", x0=14-1.8, y0=7.5-1.8, x1=14+1.8, y1=7.5+1.8, line=dict(color="white", width=2)))
    shapes.append(dict(type="path", path="M 0,0.75 Q 6.75,7.5 0,14.25", line=dict(color="white", width=2)))
    shapes.append(dict(type="path", path="M 28,0.75 Q 21.25,7.5 28,14.25", line=dict(color="white", width=2)))
    return shapes

@st.cache_data
def load_sample():
    uwb = pd.read_csv("data/virtual_uwb_realistic.csv", dtype={"player_id": "category"})
    imu = pd.read_csv("data/virtual_imu_realistic.csv", dtype={"player_id": "category"})
    return uwb, imu

# =========================
# 4. Sidebar & Filtri
# =========================
with st.sidebar:
    st.title("🏀 CoachTrack")
    use_sample = st.toggle("Usa dati demo", value=True)
    quarter = st.selectbox("Periodo", ["Intera Partita", "1° Quarto", "2° Quarto", "3° Quarto", "4° Quarto"])
    min_q = st.slider("Qualità minima (UWB)", 0, 100, 50)
    max_speed_clip = st.slider("Clip velocità (km/h)", 10, 40, 30)
    st.divider()
    enable_ai = st.toggle("Attiva AI Insights", value=True)

# =========================
# 5. Caricamento Dati
# =========================
uwb, imu = load_sample() if use_sample else (None, None)
if uwb is None:
    st.info("Carica i file CSV per iniziare.")
    st.stop()

# Pulizia e calcoli base
uwb = uwb[uwb['quality_factor'] >= min_q].copy()
uwb['step_m'] = np.sqrt(uwb.groupby('player_id')['x_m'].diff()**2 + uwb.groupby('player_id')['y_m'].diff()**2)
uwb['dt'] = uwb.groupby('player_id')['timestamp_s'].diff()
uwb['speed_kmh'] = (uwb['step_m'] / uwb['dt'] * 3.6).clip(upper=max_speed_clip)

# =========================
# 6. Gestione Nomi Giocatori
# =========================
all_player_ids = sorted(uwb["player_id"].unique())

if "player_name_map" not in st.session_state:
    st.session_state.player_name_map = {pid: pid for pid in all_player_ids}

with st.sidebar:
    with st.expander("👥 Modifica Nomi Giocatori"):
        st.caption("Cambia P1, P2... con i nomi reali")
        edited_df = st.data_editor(
            pd.DataFrame({
                "ID": all_player_ids,
                "Nome Visualizzato": [get_label(pid) for pid in all_player_ids]
            }), hide_index=True, use_container_width=True
        )
        if st.button("Salva Nomi"):
            st.session_state.player_name_map = dict(zip(edited_df["ID"], edited_df["Nome Visualizzato"]))
            st.rerun()

# Applicazione nomi reali ai dataset
uwb["player_label"] = uwb["player_id"].map(get_label)

# =========================
# 7. Calcolo KPI
# =========================
kpi = uwb.groupby('player_label').agg(
    Distanza=('step_m', 'sum'),
    Velocità_Media=('speed_kmh', 'mean'),
    Velocità_Max=('speed_kmh', 'max'),
    Qualità=('quality_factor', 'mean')
).reset_index()

# =========================
# 8. Dashboard UI
# =========================
# Header Bianco Pro [file:287]
st.markdown(f"""
<div class="ct-header-box">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h2 style="margin:0;">Dashboard Sessione</h2>
            <div style="font-size:0.9rem; opacity:0.8;">
                Filtri: <b>{quarter}</b> | Qualità min: <b>{min_q}</b> | Clip: <b>{max_speed_clip} km/h</b>
            </div>
        </div>
        <div style="font-weight:bold; color:#2563EB;">LIVE TRACKING</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Metriche Team
m1, m2, m3, m4 = st.columns(4)
m1.metric("Distanza Media", f"{kpi['Distanza'].mean():.0f} m")
m2.metric("Velocità Media", f"{kpi['Velocità_Media'].mean():.1f} km/h")
m3.metric("Picco Sessione", f"{kpi['Velocità_Max'].max():.1f} km/h")
m4.metric("Qualità Segnale", f"{kpi['Qualità'].mean():.0f}/100")

# Tabs principali
t_report, t_heat, t_imu = st.tabs(["📊 Report Squadra", "🔥 Heatmap", "📉 Analisi IMU"])

# --- TAB REPORT ---
with t_report:
    st.subheader("Performance Individuale")
    st.dataframe(kpi, use_container_width=True, hide_index=True)
    
    col_ai, col_plan = st.columns([1.2, 0.8])
    with col_ai:
        st.markdown("### AI Analysis")
        st.markdown("<div class='ai-insight'>Analisi tattica in corso... carica la chiave API Groq per i dettagli.</div>", unsafe_allow_html=True)
    
    with col_plan:
        st.markdown("### Piano Individuale")
        sel_p = st.selectbox("Seleziona Giocatore", kpi['player_label'].unique())
        # Logica download piano [web:272]
        piano_testo = f"CoachTrack Report - {sel_p}\n\nDistanza: {kpi.loc[kpi['player_label']==sel_p, 'Distanza'].values[0]:.0f}m\nVelocità Max: {kpi.loc[kpi['player_label']==sel_p, 'Velocità_Max'].values[0]:.1f} km/h\n\nSuggerimento: Lavoro intermittente ad alta intensità."
        st.download_button(label=f"⬇️ Scarica Piano {sel_p}", data=piano_testo, file_name=f"piano_{sel_p}.txt", mime="text/plain", use_container_width=True)

# --- TAB HEATMAP ---
with t_heat:
    st.subheader("Distribuzione Spaziale")
    sel_h = st.radio("Visualizza:", ["Tutti", "Giocatore Singolo"], horizontal=True)
    
    df_h = uwb
    if sel_h == "Giocatore Singolo":
        sel_p_h = st.selectbox("Scegli Giocatore", kpi['player_label'].unique())
        df_h = uwb[uwb['player_label'] == sel_p_h]
    
    # Heatmap veloce [web:269]
    fig = px.density_heatmap(df_h, x="x_m", y="y_m", nbinsx=60, nbinsy=30, range_x=[0,28], range_y=[0,15],
                             color_continuous_scale="Viridis", title=f"Heatmap: {sel_h}")
    fig.update_layout(shapes=draw_basketball_court(), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# --- TAB IMU ---
with t_imu:
    if imu is not None:
        st.subheader("Carico Meccanico (IMU)")
        imu_p = st.selectbox("Giocatore IMU", imu['player_id'].unique())
        df_imu = imu[imu['player_id'] == imu_p]
        fig_imu = px.line(df_imu, x="timestamp_s", y="accel_z_ms2", title=f"Accelerazione Z: {imu_p}")
        st.plotly_chart(fig_imu, use_container_width=True)
    else:
        st.info("Carica un file IMU per vedere i grafici di accelerazione.")
