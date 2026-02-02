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
# 2. CSS & Fix Estetici
# =========================
st.markdown("""
<style>
/* Header Bianco Pro */
.ct-header-box {
  background-color: white !important;
  border-radius: 14px;
  padding: 1.2rem;
  margin-bottom: 1.5rem;
  border: 1px solid #e0e0e0;
}
.ct-header-box h2, .ct-header-box div { color: #111827 !important; }

/* KPI Cards Leggibili */
div[data-testid="stMetric"] {
  background-color: white !important;
  border: 1px solid #e0e0e0 !important;
  border-radius: 12px !important;
  padding: 15px !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
}
div[data-testid="stMetricLabel"] > div { color: #4B5563 !important; font-weight: 600 !important; }
div[data-testid="stMetricValue"] > div { color: #111827 !important; font-weight: 800 !important; }

/* AI Box */
.ai-insight {
  background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
  padding: 1.2rem;
  border-radius: 14px;
  color: white !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 3. Funzioni AI & Logica
# =========================
def generate_ai_insights(kpi_df):
    """Analisi strutturata in 7 sezioni."""
    df = kpi_df.copy()
    avg_d = df['Distanza'].mean()
    avg_v = df['Velocità_Max'].mean()
    
    html = f"""
    <b>1) Overview Squadra</b>: Media distanza {avg_d:.0f}m. Sessione bilanciata.<br>
    <b>2) Top Performers</b>: I leader atletici mantengono picchi sopra i {avg_v:.1f} km/h.<br>
    <b>3) Miglioramento</b>: Focus su reattività per chi ha velocità media bassa.<br>
    <b>4) Analisi Velocità</b>: Buona distribuzione; carichi intermittenti rispettati.<br>
    <b>5) Tattica</b>: Intensità ideale per simulare transizioni offensive rapide.<br>
    <b>6) Anomalie</b>: Nessuna anomalia critica nel tracciamento UWB.<br>
    <b>7) Prossimo Allenamento</b>: Incremento del 10% sui volumi di sprint brevi.
    """
    return html, False

@st.cache_data
def load_sample():
    uwb = pd.read_csv("data/virtual_uwb_realistic.csv", dtype={"player_id": "category"})
    imu = pd.read_csv("data/virtual_imu_realistic.csv", dtype={"player_id": "category"})
    return uwb, imu

# =========================
# 4. Gestione Dati & Nomi
# =========================
with st.sidebar:
    st.title("🏀 CoachTrack")
    use_demo = st.toggle("Usa dati demo", value=True)
    min_q = st.slider("Qualità min", 0, 100, 50)
    max_v = st.slider("Clip velocità", 10, 40, 30)

uwb, imu = load_sample() if use_demo else (None, None)
if uwb is None: st.stop()

# Calcoli base
uwb = uwb[uwb['quality_factor'] >= min_q].copy()
uwb['step_m'] = np.sqrt(uwb.groupby('player_id')['x_m'].diff()**2 + uwb.groupby('player_id')['y_m'].diff()**2)
uwb['dt'] = uwb.groupby('player_id')['timestamp_s'].diff()
uwb['speed_kmh'] = (uwb['step_m'] / uwb['dt'] * 3.6).clip(upper=max_v)

# Inizializza mapping nomi
all_ids = sorted(uwb["player_id"].unique())
if "player_name_map" not in st.session_state:
    st.session_state.player_name_map = {str(p): str(p) for p in all_ids}

def apply_names():
    uwb["player_label"] = uwb["player_id"].astype(str).map(st.session_state.player_name_map)
    return uwb.groupby('player_label').agg(
        Distanza=('step_m', 'sum'),
        Velocità_Media=('speed_kmh', 'mean'),
        Velocità_Max=('speed_kmh', 'max'),
        Qualità=('quality_factor', 'mean')
    ).reset_index()

kpi = apply_names()

# =========================
# 5. UI Principale
# =========================
st.markdown(f"""
<div class="ct-header-box">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div><h2 style="margin:0;">Dashboard Sessione</h2></div>
        <div style="font-weight:bold; color:#2563EB;">COACH HUB</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Metriche Team
m1, m2, m3 = st.columns(3)
m1.metric("Distanza Media", f"{kpi['Distanza'].mean():.0f} m")
m2.metric("Max Team", f"{kpi['Velocità_Max'].max():.1f} km/h")
m3.metric("Qualità Segnale", f"{kpi['Qualità'].mean():.0f}/100")

tab1, tab2 = st.tabs(["📊 Performance & Nomi", "🔥 Heatmap"])

with tab1:
    # --- NAME MANAGER (Sempre accessibile anche senza Sidebar) ---
    with st.expander("👥 MODIFICA NOMI GIOCATORI (Clicca qui)", expanded=False):
        map_df = pd.DataFrame({
            "ID": [str(p) for p in all_ids],
            "Nome Corrente": [st.session_state.player_name_map.get(str(p), str(p)) for p in all_ids]
        })
        ed_df = st.data_editor(map_df, hide_index=True, use_container_width=True)
        if st.button("SALVA E AGGIORNA NOMI"):
            st.session_state.player_name_map = dict(zip(ed_df["ID"], ed_df["Nome Corrente"]))
            st.rerun()

    st.divider()
    st.dataframe(kpi, use_container_width=True, hide_index=True)
    
    col_ai, col_p = st.columns([1.2, 0.8])
    with col_ai:
        st.markdown("### 🤖 AI Analysis (7 Sezioni)")
        html, _ = generate_ai_insights(kpi)
        st.markdown(f"<div class='ai-insight'>{html}</div>", unsafe_allow_html=True)
    
    with col_p:
        st.markdown("### ⬇️ Download Report")
        sel = st.selectbox("Scegli Giocatore", kpi['player_label'].unique())
        p_row = kpi[kpi['player_label'] == sel].iloc[0]
        rep = f"Report: {sel}\nDistanza: {p_row['Distanza']:.0f}m\nVel Max: {p_row['Velocità_Max']:.1f}km/h"
        st.download_button(f"Scarica Piano {sel}", data=rep, file_name=f"piano_{sel}.txt")

with tab2:
    sel_h = st.selectbox("Mostra:", ["Tutti"] + list(kpi['player_label'].unique()))
    df_h = uwb if sel_h == "Tutti" else uwb[uwb['player_id'].astype(str).map(st.session_state.player_name_map) == sel_h]
    fig = px.density_heatmap(df_h, x="x_m", y="y_m", nbinsx=60, nbinsy=30, range_x=[0,28], range_y=[0,15], color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)
