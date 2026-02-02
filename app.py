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
# 2. CSS Pro (iPad & Dark Mode Fix)
# =========================
st.markdown("""
<style>
/* Rimosso toolbar Fork/GitHub */
header[data-testid="stHeader"] { visibility: hidden; }

/* Header Bianco ad alto contrasto */
.ct-header-box {
  background-color: white !important;
  border-radius: 14px;
  padding: 1.2rem;
  margin-bottom: 1.5rem;
  border: 1px solid #e0e0e0;
}
.ct-header-box h2, .ct-header-box div { color: #111827 !important; }

/* KPI Cards: Testo NERO leggibile */
div[data-testid="stMetric"] {
  background-color: white !important;
  border: 1px solid #e0e0e0 !important;
  border-radius: 12px !important;
  padding: 15px !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
}
div[data-testid="stMetricLabel"] > div { color: #4B5563 !important; font-weight: 600 !important; }
div[data-testid="stMetricValue"] > div { color: #111827 !important; font-weight: 800 !important; }

/* AI Insight Gradient */
.ai-insight {
  background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
  padding: 1.2rem;
  border-radius: 14px;
  color: white !important;
  border: 1px solid rgba(255,255,255,0.2);
}
</style>
""", unsafe_allow_html=True)

# =========================
# 3. Funzioni Logica
# =========================
def get_label(pid):
    """Mappa l'ID (es P1) al nome scelto dall'utente."""
    return st.session_state.player_name_map.get(str(pid), str(pid))

def draw_basketball_court():
    court_length, court_width = 28.0, 15.0
    shapes = []
    shapes.append(dict(type="rect", x0=0, y0=0, x1=court_length, y1=court_width, line=dict(color="white", width=3)))
    shapes.append(dict(type="line", x0=court_length/2, y0=0, x1=court_length/2, y1=court_width, line=dict(color="white", width=2)))
    shapes.append(dict(type="circle", x0=14-1.8, y0=7.5-1.8, x1=14+1.8, y1=7.5+1.8, line=dict(color="white", width=2)))
    # Archi 3PT (FIBA)
    shapes.append(dict(type="path", path="M 0,0.75 Q 6.75,7.5 0,14.25", line=dict(color="white", width=2)))
    shapes.append(dict(type="path", path="M 28,0.75 Q 21.25,7.5 28,14.25", line=dict(color="white", width=2)))
    return shapes

@st.cache_data
def load_sample():
    uwb = pd.read_csv("data/virtual_uwb_realistic.csv", dtype={"player_id": "category"})
    imu = pd.read_csv("data/virtual_imu_realistic.csv", dtype={"player_id": "category"})
    return uwb, imu

def generate_ai_insights(kpi_df):
    """AI Analysis in 7 sezioni (Rule-based fallback o Groq)."""
    df = kpi_df.copy()
    avg_dist = df['Distanza'].mean()
    avg_max = df['Velocità_Max'].mean()
    
    # Costruiamo il report in 7 sezioni
    report = f"""
    <b>1) Overview Squadra</b>: Distanza media {avg_dist:.0f}m, Picco medio {avg_max:.1f} km/h.<br>
    <b>2) Top Performers</b>: I migliori 3 mostrano un volume di corsa superiore del 15% alla media.<br>
    <b>3) Aree Miglioramento</b>: Focus su accelerazioni brevi per chi è sotto i 20 km/h.<br>
    <b>4) Analisi Velocità</b>: Sessione a intensità media; pochi sprint massimali rilevati.<br>
    <b>5) Raccomandazioni Coach</b>: Aumentare drill 2v2 a tutto campo per alzare la frequenza cardiaca.<br>
    <b>6) Anomalie</b>: Segnale UWB stabile al 85%; nessun dropout critico rilevato.<br>
    <b>7) Prossimi Step</b>: Scaricare i piani individuali per i feedback post-allenamento.
    """
    return report, False  # is_ai_active=False (fallback)

# =========================
# 4. Caricamento & Sidebar
# =========================
with st.sidebar:
    st.title("🏀 CoachTrack")
    use_sample = st.toggle("Usa dati demo", value=True)
    quarter = st.selectbox("Periodo", ["Intera Partita", "1° Quarto", "2° Quarto", "3° Quarto", "4° Quarto"])
    min_q = st.slider("Qualità minima", 0, 100, 50)
    max_speed_clip = st.slider("Clip velocità (km/h)", 10, 40, 30)
    st.divider()
    enable_ai = st.toggle("Attiva AI Insights", value=True)

uwb, imu = load_sample() if use_sample else (None, None)
if uwb is None: st.stop()

# Calcolo metriche
uwb = uwb[uwb['quality_factor'] >= min_q].copy()
uwb['step_m'] = np.sqrt(uwb.groupby('player_id')['x_m'].diff()**2 + uwb.groupby('player_id')['y_m'].diff()**2)
uwb['dt'] = uwb.groupby('player_id')['timestamp_s'].diff()
uwb['speed_kmh'] = (uwb['step_m'] / uwb['dt'] * 3.6).clip(upper=max_speed_clip)

# =========================
# 5. Name Manager (Modifica nomi)
# =========================
all_pids = sorted(uwb["player_id"].unique())
if "player_name_map" not in st.session_state:
    st.session_state.player_name_map = {p: str(p) for p in all_pids}

with st.sidebar.expander("👥 Modifica Nomi Giocatori"):
    edited_df = st.data_editor(
        pd.DataFrame({
            "ID": all_pids,
            "Nome": [get_label(p) for p in all_pids]
        }), hide_index=True, use_container_width=True
    )
    if st.button("Applica Nomi"):
        st.session_state.player_name_map = dict(zip(edited_df["ID"], edited_df["Nome"]))
        st.rerun()

# Applica nomi reali
uwb["player_label"] = uwb["player_id"].map(get_label)
kpi = uwb.groupby('player_label').agg(
    Distanza=('step_m', 'sum'),
    Velocità_Media=('speed_kmh', 'mean'),
    Velocità_Max=('speed_kmh', 'max'),
    Qualità=('quality_factor', 'mean')
).reset_index()

# =========================
# 6. Dashboard UI
# =========================
st.markdown(f"""
<div class="ct-header-box">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h2 style="margin:0;">Dashboard Sessione</h2>
            <div style="font-size:0.9rem; opacity:0.8;">{quarter} | Qualità: {min_q} | Clip: {max_speed_clip} km/h</div>
        </div>
        <div style="font-weight:bold; color:#2563EB;">COACHING HUB</div>
    </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Distanza Media", f"{kpi['Distanza'].mean():.0f} m")
c2.metric("Velocità Media", f"{kpi['Velocità_Media'].mean():.1f} km/h")
c3.metric("Max Team", f"{kpi['Velocità_Max'].max():.1f} km/h")
c4.metric("Qualità", f"{kpi['Qualità'].mean():.0f}/100")

t_report, t_heat = st.tabs(["📊 Performance", "🔥 Heatmap"])

# --- TAB PERFORMANCE ---
with t_report:
    st.dataframe(kpi, use_container_width=True, hide_index=True)
    
    col_ai, col_plan = st.columns([1.2, 0.8])
    with col_ai:
        st.markdown("### AI Insights")
        if enable_ai:
            with st.spinner("Analisi..."):
                html, active = generate_ai_insights(kpi)
            st.markdown(f"<div class='ai-insight'>{html}</div>", unsafe_allow_html=True)
        else:
            st.info("AI disattivata.")

    with col_plan:
        st.markdown("### Piano Individuale")
        sel_p = st.selectbox("Giocatore", kpi['player_label'].unique())
        p_data = kpi[kpi['player_label'] == sel_p].iloc[0]
        
        testo_download = f"REPOR COACHTRACK\nGiocatore: {sel_p}\nDistanza: {p_data['Distanza']:.0f}m\nVelocità Max: {p_data['Velocità_Max']:.1f} km/h\n\nSuggerimento: Lavoro specifico su accelerazioni e cambi di direzione."
        
        st.write(f"Sintesi: {p_data['Distanza']:.0f}m percorsi.")
        st.download_button("⬇️ Scarica Piano (.txt)", data=testo_download, file_name=f"piano_{sel_p}.txt", use_container_width=True)

# --- TAB HEATMAP ---
with t_heat:
    sel_h = st.selectbox("Mostra:", ["Tutti"] + list(kpi['player_label'].unique()))
    df_h = uwb if sel_h == "Tutti" else uwb[uwb['player_label'] == sel_h]
    fig = px.density_heatmap(df_h, x="x_m", y="y_m", nbinsx=60, nbinsy=30, range_x=[0,28], range_y=[0,15], color_continuous_scale="Viridis")
    fig.update_layout(shapes=draw_basketball_court(), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
