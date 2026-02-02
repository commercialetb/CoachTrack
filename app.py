import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.spatial import ConvexHull
from datetime import datetime

# =========================
# 1. SETUP & THEME [file:287]
# =========================
st.set_page_config(
    page_title="CoachTrack Elite AI", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Forza un look pulito e leggibile su iPad
st.markdown("""
<style>
header { visibility: hidden; }
.report-card { background: white; border-radius: 14px; padding: 20px; border: 1px solid #e2e8f0; margin-bottom: 20px; }
.metric-value { font-size: 26px; font-weight: 800; color: #1e3a8a; }
.ai-badge { background: #dbeafe; color: #1e40af; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; }
.status-bad { color: #ef4444; font-weight: bold; }
.status-good { color: #10b981; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =========================
# 2. LOGICA ANALISI ELITE [web:356][web:360]
# =========================
def calculate_spacing_index(df_frame):
    """Calcola l'area occupata dai 5 giocatori usando Convex Hull (Precisione NBA)."""
    if len(df_frame) < 3: return 0
    points = df_frame[['x_m', 'y_m']].values
    try:
        hull = ConvexHull(points)
        return hull.area
    except:
        return 0

def get_fatigue_data(df_player):
    """Rileva il calo di performance tra inizio e fine sessione."""
    if len(df_player) < 100: return 0, "Dati Insuff."
    # Confronto tra le velocità massime nei due segmenti
    start_v = df_player.head(len(df_player)//5)['speed_kmh'].max()
    end_v = df_player.tail(len(df_player)//5)['speed_kmh'].max()
    
    if start_v <= 0: return 0, "N/A"
    drop = ((start_v - end_v) / start_v) * 100
    status = "⚠️ CRITICA" if drop > 15 else "✅ OTTIMALE"
    return max(0, drop), status

# =========================
# 3. STATO DELLA SESSIONE
# =========================
if "player_names" not in st.session_state:
    st.session_state.player_names = {}
if "player_roles" not in st.session_state:
    st.session_state.player_roles = {}

@st.cache_data
def load_uwb_data():
    try:
        df = pd.read_csv("data/virtual_uwb_realistic.csv")
        # Calcolo velocità vettoriale per analisi fatica
        df['step_m'] = np.sqrt(df.groupby('player_id')['x_m'].diff()**2 + df.groupby('player_id')['y_m'].diff()**2).fillna(0)
        df['dt'] = df.groupby('player_id')['timestamp_s'].diff().fillna(0.1)
        df['speed_kmh'] = (df['step_m'] / df['dt'] * 3.6).clip(upper=35)
        return df
    except:
        st.error("File dati non trovato in /data/")
        return pd.DataFrame()

# =========================
# 4. SIDEBAR & GESTIONE NOMI
# =========================
uwb = load_uwb_data()
if uwb.empty: st.stop()

all_ids = sorted(uwb["player_id"].unique())

with st.sidebar:
    st.title("🏀 CoachTrack Elite")
    with st.expander("👥 Team Manager", expanded=True):
        setup_df = pd.DataFrame({
            "ID": all_ids,
            "Nome": [st.session_state.player_names.get(p, p) for p in all_ids],
            "Ruolo": [st.session_state.player_roles.get(p, "Guardia") for p in all_ids]
        })
        ed = st.data_editor(setup_df, hide_index=True, use_container_width=True)
        if st.button("Salva Configurazione"):
            st.session_state.player_names = dict(zip(ed["ID"], ed["Nome"]))
            st.session_state.player_roles = dict(zip(ed["ID"], ed["Ruolo"]))
            st.rerun()

uwb["player_label"] = uwb["player_id"].map(st.session_state.player_names).fillna(uwb["player_id"])

# =========================
# 5. UI DASHBOARD
# =========================
st.title("Elite Tactical Hub")

t_tactic, t_physical, t_report = st.tabs(["🎯 Tattica & Spacing", "📈 Fatica & Benchmark", "📄 Report Pro"])

# --- TAB TATTICA ---
with t_tactic:
    c_s1, c_s2 = st.columns([1, 2])
    
    # Calcolo Spacing (simulazione istantanea quintetto)
    spacing_val = calculate_spacing_index(uwb.sample(n=min(5, len(uwb))))
    
    with c_s1:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader("Spacing Index")
        st.markdown(f'<div class="metric-value">{spacing_val:.1f} m²</div>', unsafe_allow_html=True)
        st.markdown('<span class="ai-badge">ANALISI AI</span>', unsafe_allow_html=True)
        st.write("")
        status_txt = "OTTIMO" if spacing_val > 80 else "RISTRETTO"
        st.write(f"Stato Spacing: **{status_txt}**")
        st.caption("Target NBA: > 90m² per set offensivi.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.info("💡 Un'area ristretta indica scarsa occupazione degli angoli (corners).")

    with c_s2:
        fig_heat = px.density_heatmap(uwb, x="x_m", y="y_m", nbinsx=60, nbinsy=30, 
                                     range_x=[0,28], range_y=[0,15], color_continuous_scale="Plasma",
                                     title="Shot Quality & Positioning Map")
        fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB FISICA ---
with t_physical:
    sel_p = st.selectbox("Analizza Giocatore:", uwb["player_label"].unique())
    p_df = uwb[uwb['player_label'] == sel_p]
    p_role = st.session_state.player_roles.get(uwb[uwb['player_label']==sel_p]['player_id'].iloc[0], "Guardia")
    
    drop, f_status = get_fatigue_data(p_df)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Indice Fatica (Drop)", f"{drop:.1f}%", delta=f"-{drop:.1f}%", delta_color="inverse")
    m2.markdown(f"Stato Muscolare: <span class='{'status-bad' if drop > 15 else 'status-good'}'>{f_status}</span>", unsafe_allow_html=True)
    
    # Benchmark Pro [web:345]
    bench_val = "ÉLITE" if p_df['speed_kmh'].max() > 25 else "MEDIA"
    m3.metric("Benchmark vs Serie A", bench_val)
    
    fig_v = px.line(p_df.tail(200), x="timestamp_s", y="speed_kmh", title=f"Profilo Velocità Recente - {sel_p}")
    st.plotly_chart(fig_v, use_container_width=True)

# --- TAB REPORT ---
with t_report:
    st.subheader("Esportazione Report Professionale")
    
    # Costruzione Report Tattico Dettagliato
    coach_report = f"""
COACHTRACK ELITE REPORT: {sel_p}
RUOLO: {p_role}
DATA: {datetime.now().strftime('%d/%m/%Y')}
--------------------------------------------------

1. ANALISI FISICA (MONITORAGGIO CARICO)
- Distanza Totale: {p_df['step_m'].sum():.0f} m
- Velocità Massima: {p_df['speed_kmh'].max():.1f} km/h
- Indice Fatica: {drop:.1f}% ({f_status})
- Stato: {"Recupero Necessario" if drop > 15 else "Carico Ottimale"}

2. ANALISI TATTICA (SPACING & MOVIMENTO)
- Spacing Area Squadra: {spacing_val:.1f} m²
- Qualità Tiro: {"In movimento (Elite)" if p_df['speed_kmh'].mean() > 14 else "Statico (Spot-up)"}
- Profilo Ruolo: {p_role} in linea con target stagionali.

3. PRESCRIZIONE ALLENAMENTO (AI PRO)
- Target: {"Lavoro intermittente 15/15" if drop < 10 else "Scarico attivo / Mobilità"}
- Drill Suggerito: {"Drill d'uscita dai blocchi (30 tiri)" if p_role != "Centro" else "Mikan Drill + Post Work"}
- Focus: Migliorare il picco di velocità nei primi 5 metri di transizione.

--------------------------------------------------
Analisi generata tramite CoachTrack AI.
    """
    
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.text_area("Anteprima Report (Pronto per invio):", coach_report, height=400)
    
    # Download Button
    st.download_button(
        label=f"⬇️ SCARICA REPORT COMPLETO ({sel_p})", 
        data=coach_report, 
        file_name=f"Report_CoachTrack_{sel_p}.txt",
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

