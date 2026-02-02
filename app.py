import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.spatial import ConvexHull
from datetime import datetime

# =========================
# 1. SETUP RESPONSIVO
# =========================
st.set_page_config(page_title="CoachTrack Elite v2.1", layout="wide")

# Persistenza dati nello stato della sessione
if "player_names" not in st.session_state: st.session_state.player_names = {}
if "history" not in st.session_state: st.session_state.history = []
if "notes" not in st.session_state: st.session_state.notes = {}
if "player_roles" not in st.session_state: st.session_state.player_roles = {}

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
    
    .ai-box { background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%); padding: 25px; border-radius: 20px; border: 1px solid #3b82f6; color: white; }
</style>
""", unsafe_allow_html=True)

# =========================
# 3. MOTORE DI ANALISI ELITE
# =========================
def get_spacing(df):
    """Calcola l'area del quintetto (Precisione NBA)."""
    if len(df) < 3: return 0
    try: return ConvexHull(df[['x_m', 'y_m']].values).area
    except: return 0

def get_pro_insights(p_label, dist, vmax, role):
    """Genera analisi tattica approfondita basata sui ruoli."""
    status = "🔴 OVERLOAD" if dist > 3500 else "🟢 OTTIMALE"
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
    
    **Piani Allenamento (Drills):**
    1. {d_list[0]} (15 min intensi)
    2. {d_list[1]} (Focus velocità d'esecuzione)
    3. {d_list[2]} (Lavoro specifico post-fatica)
    """
    return report

# =========================
# 4. CARICAMENTO & CALCOLO DATI
# =========================
@st.cache_data
def load_and_process_data():
    try:
        df = pd.read_csv("data/virtual_uwb_realistic.csv")
        # Calcolo velocità e distanze se non presenti
        df = df.sort_values(['player_id', 'timestamp_s'])
        df['step_m'] = np.sqrt(df.groupby('player_id')['x_m'].diff()**2 + df.groupby('player_id')['y_m'].diff()**2).fillna(0)
        df['dt'] = df.groupby('player_id')['timestamp_s'].diff().fillna(0.1)
        # Calcolo speed_kmh obbligatorio
        df['speed_kmh'] = (df['step_m'] / df['dt'] * 3.6).clip(upper=35)
        return df
    except:
        # Fallback se il file manca
        st.error("File 'data/virtual_uwb_realistic.csv' non trovato.")
        return pd.DataFrame(columns=["player_id", "x_m", "y_m", "timestamp_s", "speed_kmh", "quality_factor", "step_m"])

uwb = load_and_process_data()
if uwb.empty: st.stop()

all_pids = sorted(uwb["player_id"].unique())

# =========================
# 5. NAVIGAZIONE TAB
# =========================
st.title("🏀 CoachTrack Elite AI v2.1")

t_perf, t_tactic, t_team, t_setup = st.tabs(["📊 Player Performance", "🎯 Tactical Spacing", "📈 Team History", "⚙️ Setup & Nomi"])

# --- TAB SETUP (MODIFICA NOMI E RUOLI) ---
with t_setup:
    st.header("Gestione Squadra")
    st.info("Modifica qui i nomi e i ruoli. Clicca fuori dal campo dopo aver scritto per salvare.")
    
    for pid in all_pids:
        c1, c2, c3 = st.columns([1, 2, 2])
        with c1: st.text(f"ID: {pid}")
        with c2: 
            st.session_state.player_names[str(pid)] = st.text_input(f"Nome {pid}", value=st.session_state.player_names.get(str(pid), str(pid)), key=f"in_{pid}")
        with c3:
            st.session_state.player_roles[str(pid)] = st.selectbox(f"Ruolo {pid}", ["Guardia", "Ala", "Centro"], index=["Guardia", "Ala", "Centro"].index(st.session_state.player_roles.get(str(pid), "Guardia")), key=f"ro_{pid}")

# Mapping nomi aggiornato
uwb["player_label"] = uwb["player_id"].astype(str).map(st.session_state.player_names).fillna(uwb["player_id"])
kpi = uwb.groupby('player_label').agg(
    Distanza=('step_m', 'sum'), 
    Vel_Max=('speed_kmh', 'max')
).reset_index()

# --- TAB PERFORMANCE ---
with t_perf:
    sel_p = st.selectbox("Seleziona Giocatore:", kpi['player_label'].unique())
    p_row = kpi[kpi['player_label'] == sel_p].iloc[0]
    
    # Recupero ruolo tramite ID originale
    orig_id = uwb[uwb['player_label'] == sel_p]['player_id'].iloc[0]
    p_role = st.session_state.player_roles.get(str(orig_id), "Guardia")

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Distanza</div><div class="kpi-val">{p_row["Distanza"]:.0f} m</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Velocità Max</div><div class="kpi-val">{p_row["Vel_Max"]:.1f}</div></div>', unsafe_allow_html=True)
    with c3: 
        status = "🟢 OK" if p_row["Distanza"] < 3500 else "⚠️ FATICA"
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Stato Carico</div><div class="kpi-val">{status}</div></div>', unsafe_allow_html=True)

    st.divider()
    
    col_ai, col_note = st.columns([1.5, 1])
    with col_ai:
        report_ai = get_pro_insights(sel_p, p_row['Distanza'], p_row['Vel_Max'], p_role)
        st.markdown(f'<div class="ai-box">{report_ai}</div>', unsafe_allow_html=True)
        
        # Download completo
        full_txt = f"REPORT COACHTRACK - {sel_p}\n" + "="*30 + f"\n{report_ai}\n\nNote Coach: {st.session_state.notes.get(sel_p, '')}"
        st.download_button(f"Scarica Report {sel_p}", data=full_txt, file_name=f"Report_{sel_p}.txt", use_container_width=True)

    with col_note:
        st.subheader("Note Coach")
        st.session_state.notes[sel_p] = st.text_area("Osservazioni tecniche...", value=st.session_state.notes.get(sel_p, ""), height=250)

# --- TAB TACTICAL ---
with t_tactic:
    c_s, c_m = st.columns([1, 2])
    with c_s:
        area = get_spacing(uwb.sample(n=min(5, len(uwb))))
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Spacing Area</div><div class="kpi-val">{area:.1f} m²</div></div>', unsafe_allow_html=True)
        st.info("Un'area ampia (Target >90m²) indica che la squadra sta occupando bene gli angoli.")
    with c_m:
        fig_heat = px.density_heatmap(uwb[uwb['player_label']==sel_p], x="x_m", y="y_m", nbinsx=40, nbinsy=20, range_x=[0,28], range_y=[0,15], color_continuous_scale="Plasma", title=f"Posizionamento: {sel_p}")
        st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB HISTORY ---
with t_team:
    if st.button("💾 Salva Dati Sessione"):
        st.session_state.history.append({"date": datetime.now().strftime("%H:%M"), "avg": kpi['Distanza'].mean()})
        st.success("Sessione salvata nello storico!")
    
    if st.session_state.history:
        h_df = pd.DataFrame(st.session_state.history)
        st.plotly_chart(px.line(h_df, x="date", y="avg", title="Trend Carico Medio Squadra", markers=True), use_container_width=True)
