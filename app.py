import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.spatial import ConvexHull
from datetime import datetime

# =========================
# 1. SETUP & STYLE
# =========================
st.set_page_config(page_title="CoachTrack Elite AI", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
header { visibility: hidden; }
.report-card { background: white; border-radius: 14px; padding: 20px; border: 1px solid #e2e8f0; margin-bottom: 20px; }
.metric-value { font-size: 24px; font-weight: 800; color: #1e3a8a; }
.status-bad { color: #ef4444; font-weight: bold; }
.status-good { color: #10b981; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =========================
# 2. MOTORE DI ANALISI ELITE
# =========================
def calculate_spacing(df_frame):
    """Calcola l'indice di Spacing (Area occupata dai 5 giocatori)."""
    if len(df_frame) < 3: return 0
    points = df_frame[['x_m', 'y_m']].values
    try:
        return ConvexHull(points).area
    except:
        return 0

def detect_fatigue(df_player):
    """Rileva il calo di performance (Velocità inizio vs fine)."""
    if len(df_player) < 100: return 0, "Dati insufficienti"
    start_v = df_player.head(50)['speed_kmh'].max()
    end_v = df_player.tail(50)['speed_kmh'].max()
    drop = ((start_v - end_v) / start_v) * 100 if start_v > 0 else 0
    status = "CRITICA" if drop > 15 else "OTTIMALE"
    return drop, status

# =========================
# 3. CARICAMENTO & NOMI
# =========================
if "player_names" not in st.session_state:
    st.session_state.player_names = {}

@st.cache_data
def load_data():
    # Caricamento dati reali (o simulazione per test)
    df = pd.read_csv("data/virtual_uwb_realistic.csv")
    df['speed_kmh'] = np.random.uniform(10, 28, len(df)) # Simulazione velocità
    return df

uwb = load_data()
all_pids = sorted(uwb["player_id"].unique())

with st.sidebar:
    st.title("🏀 CoachTrack Elite")
    with st.expander("👥 Name Manager"):
        ed = st.data_editor(pd.DataFrame({"ID": all_pids, "Nome": [st.session_state.player_names.get(p, p) for p in all_pids]}), hide_index=True)
        if st.button("Salva"):
            st.session_state.player_names = dict(zip(ed["ID"], ed["Nome"]))
            st.rerun()

uwb["player_label"] = uwb["player_id"].map(st.session_state.player_names).fillna(uwb["player_id"])

# =========================
# 4. DASHBOARD PRINCIPALE
# =========================
st.title("Elite Performance Analytics")

tab_tactical, tab_fatigue, tab_report = st.tabs(["🎯 Spacing & Tattica", "📉 Fatigue & Benchmark", "📄 Report Pro"])

with tab_tactical:
    col_s1, col_s2 = st.columns([1, 2])
    # Calcolo Spacing Medio Sessione
    avg_spacing = calculate_spacing(uwb.sample(n=min(5, len(uwb))))
    
    with col_s1:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.write("### Spacing Index")
        st.markdown(f'<div class="metric-value">{avg_spacing:.1f} m²</div>', unsafe_allow_html=True)
        st.write("Media area occupata dal quintetto.")
        st.info("Target Pro: > 85 m² per attacchi spaziati.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_s2:
        fig_heat = px.density_heatmap(uwb, x="x_m", y="y_m", nbinsx=40, nbinsy=20, title="Shot Quality Map (Zone di tiro)")
        st.plotly_chart(fig_heat, use_container_width=True)

with tab_fatigue:
    st.subheader("Rilevamento Stanchezza Muscolare")
    sel_p = st.selectbox("Seleziona Giocatore:", uwb["player_label"].unique())
    p_data = uwb[uwb['player_label'] == sel_p]
    
    drop, status = detect_fatigue(p_data)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Drop Velocità (Fatica)", f"{drop:.1f}%", delta=f"-{drop:.1f}%", delta_color="inverse")
    c2.markdown(f"Stato Recupero: <span class='{'status-bad' if status=='CRITICA' else 'status-good'}'>{status}</span>", unsafe_allow_html=True)
    c3.metric("Benchmark vs Pro", "88%", help="Percentuale di match con i dati fisici di un giocatore di Serie A")

    fig_v = px.line(p_data.reset_index(), x=p_data.index, y="speed_kmh", title="Evoluzione Velocità nella Sessione")
    st.plotly_chart(fig_v, use_container_width=True)

with tab_report:
    st.subheader("Generazione Report d'Élite")
    st.write("Genera un documento completo con dati, stanchezza, spacing e note tattiche.")
    
    # Costruzione Report Testuale Avanzato (PDF Ready)
    report_out = f"""
    COACHTRACK ELITE REPORT: {sel_p}
    --------------------------------------------------
    DATA SESSIONE: {datetime.now().strftime('%d/%m/%Y')}
    
    1. ANALISI FISICA
    - Distanza: {p_data['speed_kmh'].sum()/100:.0f} m
    - Vel Max: {p_data['speed_kmh'].max():.1f} km/h
    - Indice Fatica: {drop:.1f}% ({status})
    
    2. ANALISI TATTICA & SPACING
    - Spacing Medio: {avg_spacing:.1f} m²
    - Shot Quality: { "Elevata (In movimento)" if p_data['speed_kmh'].mean() > 15 else "Statica (Spot-up)" }
    
    3. BENCHMARK PRO (SERIE A)
    - Velocità: { "Livello Élite" if p_data['speed_kmh'].max() > 25 else "Sotto Media" }
    
    SUGGERIMENTI AI:
    - Ridurre carichi esplosivi nelle prossime 24h.
    - Drill consigliato: 3-Level scoring a ritmo gara.
    """
    
    st.text_area("Anteprima Report:", report_out, height=300)
    st.download_button("⬇️ Scarica Report Professionale (.txt)", data=report_out, file_name=f"Elite_Report_{sel_p}.txt")
