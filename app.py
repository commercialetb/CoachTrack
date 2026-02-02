import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

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
# 2. Pulsante "Apri Sidebar" per iPad (JavaScript)
# =========================
# Questo script inietta un pulsante blu flottante che "clicca" la sidebar per te.
components.html(
    """
    <script>
    function openSidebar() {
        // Cerca il pulsante originale di Streamlit (la freccetta >) e lo clicca
        const docs = window.parent.document;
        const buttons = Array.from(docs.querySelectorAll('button'));
        const sidebarButton = buttons.find(el => el.getAttribute('data-testid') === 'stSidebarCollapseButton');
        if (sidebarButton) {
            sidebarButton.click();
        }
    }
    </script>
    <div style="position: fixed; top: 10px; left: 10px; z-index: 99999;">
        <button onclick="openSidebar()" style="
            background-color: #2563EB;
            color: white;
            border: 2px solid white;
            border-radius: 50%;
            width: 45px;
            height: 45px;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            ☰
        </button>
    </div>
    """,
    height=70,
)

# =========================
# 3. CSS Fix (Estetica & Contrast)
# =========================
st.markdown("""
<style>
/* Nasconde toolbar GitHub/Fork */
header[data-testid="stHeader"] { visibility: hidden; }

/* Dashboard Box Bianca */
.ct-header-box {
  background-color: white !important;
  border-radius: 14px;
  padding: 1.2rem;
  margin-bottom: 1rem;
  border: 1px solid #e0e0e0;
}
.ct-header-box h2, .ct-header-box div { color: #111827 !important; }

/* KPI Cards leggibili su iPad */
div[data-testid="stMetric"] {
  background-color: white !important;
  border: 1px solid #e0e0e0 !important;
  border-radius: 12px !important;
  padding: 15px !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
}
div[data-testid="stMetricLabel"] > div { color: #4B5563 !important; font-weight: 600 !important; }
div[data-testid="stMetricValue"] > div { color: #111827 !important; font-weight: 800 !important; }

/* AI Insight Box */
.ai-insight {
  background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
  padding: 1.2rem;
  border-radius: 14px;
  color: white !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 4. Funzioni & Caricamento
# =========================
def get_label(pid):
    return st.session_state.player_name_map.get(str(pid), str(pid))

@st.cache_data
def load_sample():
    # Assicurati che i file siano in questa cartella nel tuo GitHub
    uwb = pd.read_csv("data/virtual_uwb_realistic.csv", dtype={"player_id": "category"})
    imu = pd.read_csv("data/virtual_imu_realistic.csv", dtype={"player_id": "category"})
    return uwb, imu

# --- Sidebar ---
with st.sidebar:
    st.title("🏀 CoachTrack")
    use_sample = st.toggle("Usa dati demo", value=True)
    quarter = st.selectbox("Periodo", ["Intera Partita", "1° Quarto", "2° Quarto", "3° Quarto", "4° Quarto"])
    min_q = st.slider("Qualità min", 0, 100, 50)
    max_speed_clip = st.slider("Clip velocità (km/h)", 10, 40, 30)
    st.divider()
    enable_ai = st.toggle("Attiva AI Insights", value=True)

# --- Caricamento ---
uwb, imu = load_sample() if use_sample else (None, None)
if uwb is None: st.stop()

# Elaborazione UWB
uwb = uwb[uwb['quality_factor'] >= min_q].copy()
uwb['step_m'] = np.sqrt(uwb.groupby('player_id')['x_m'].diff()**2 + uwb.groupby('player_id')['y_m'].diff()**2)
uwb['dt'] = uwb.groupby('player_id')['timestamp_s'].diff()
uwb['speed_kmh'] = (uwb['step_m'] / uwb['dt'] * 3.6).clip(upper=max_speed_clip)

# =========================
# 5. Name Manager (Cambio Nomi)
# =========================
all_pids = sorted(uwb["player_id"].unique())
if "player_name_map" not in st.session_state:
    st.session_state.player_name_map = {str(p): str(p) for p in all_pids}

with st.sidebar:
    st.markdown("### 👥 Gestione Nomi")
    map_df = pd.DataFrame({
        "ID": [str(p) for p in all_pids],
        "Nome": [get_label(p) for p in all_pids]
    })
    edited_df = st.data_editor(map_df, hide_index=True, use_container_width=True)
    if st.button("Salva Nomi"):
        st.session_state.player_name_map = dict(zip(edited_df["ID"], edited_df["Nome"]))
        st.rerun()

# Applicazione Nomi
uwb["player_label"] = uwb["player_id"].map(get_label)
kpi = uwb.groupby('player_label').agg(
    Distanza=('step_m', 'sum'),
    Velocità_Media=('speed_kmh', 'mean'),
    Velocità_Max=('speed_kmh', 'max'),
    Qualità=('quality_factor', 'mean')
).reset_index()

# =========================
# 6. Dashboard Principal
# =========================
st.markdown(f"""
<div class="ct-header-box">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h2 style="margin:0;">Dashboard Sessione</h2>
            <div style="font-size:0.9rem; color:#4B5563;">{quarter} | Qualità: {min_q} | Clip: {max_speed_clip} km/h</div>
        </div>
        <div style="font-weight:bold; color:#2563EB;">LIVE TRACKING</div>
    </div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Dist. Media", f"{kpi['Distanza'].mean():.0f} m")
m2.metric("Vel. Media", f"{kpi['Velocità_Media'].mean():.1f} km/h")
m3.metric("Team Max", f"{kpi['Velocità_Max'].max():.1f} km/h")
m4.metric("Qualità", f"{kpi['Qualità'].mean():.0f}/100")

t_rep, t_heat = st.tabs(["📊 Performance", "🔥 Heatmap"])

with t_rep:
    st.dataframe(kpi, use_container_width=True, hide_index=True)
    c_ai, c_p = st.columns([1, 1])
    with c_ai:
        st.markdown("### AI Analysis")
        st.markdown("<div class='ai-insight'>Analisi tattica in corso... (Sezioni 1-7)</div>", unsafe_allow_html=True)
    with c_p:
        st.markdown("### Piano Individuale")
        sel_p = st.selectbox("Giocatore", kpi['player_label'].unique())
        p_row = kpi[kpi['player_label'] == sel_p].iloc[0]
        testo = f"REPORT: {sel_p}\nDistanza: {p_row['Distanza']:.0f}m\nVel Max: {p_row['Velocità_Max']:.1f} km/h"
        st.download_button("⬇️ Scarica Piano (.txt)", data=testo, file_name=f"piano_{sel_p}.txt")

with t_heat:
    sel_h = st.selectbox("Mostra:", ["Tutti"] + list(kpi['player_label'].unique()))
    df_h = uwb if sel_h == "Tutti" else uwb[uwb['player_label'] == sel_h]
    fig = px.density_heatmap(df_h, x="x_m", y="y_m", nbinsx=60, nbinsy=30, range_x=[0,28], range_y=[0,15], color_continuous_scale="Viridis")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
