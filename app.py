import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =========================
# 1. Configurazione & Stato
# =========================
st.set_page_config(page_title="CoachTrack AI Ultra", layout="wide")

# Inizializzazione dati persistenti per la sessione
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["player", "dist", "max_v", "date"])
if "coach_notes" not in st.session_state:
    st.session_state.coach_notes = {}
if "player_roles" not in st.session_state:
    st.session_state.player_roles = {}

# =========================
# 2. CSS Avanzato
# =========================
st.markdown("""
<style>
header { visibility: hidden; }
.kpi-card { background: white; border-radius: 12px; padding: 15px; border: 1px solid #eee; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
.alert-red { background-color: #fee2e2; border-left: 5px solid #ef4444; padding: 10px; color: #991b1b; font-weight: bold; border-radius: 4px; }
.ai-box { background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: white; border-radius: 16px; padding: 20px; border: 1px solid rgba(255,255,255,0.2); }
</style>
""", unsafe_allow_html=True)

# =========================
# 3. Funzioni Core
# =========================
@st.cache_data
def load_data():
    # Caricamento dati UWB simulati
    try:
        df = pd.read_csv("data/virtual_uwb_realistic.csv")
        return df
    except:
        # Fallback dati sintetici per test
        return pd.DataFrame({
            "player_id": ["P1"]*100 + ["P2"]*100,
            "x_m": np.random.uniform(0, 28, 200),
            "y_m": np.random.uniform(0, 15, 200),
            "timestamp_s": np.tile(np.arange(100), 2),
            "quality_factor": [80]*200
        })

def get_player_name(pid):
    return st.session_state.get("player_name_map", {}).get(str(pid), str(pid))

# =========================
# 4. Sidebar & Name Manager
# =========================
uwb = load_data()
all_pids = sorted(uwb["player_id"].unique())

with st.sidebar:
    st.title("🏀 CoachTrack Pro")
    with st.expander("👥 Rinomina & Ruoli"):
        role_df = pd.DataFrame({
            "ID": all_pids,
            "Nome": [st.session_state.get("player_name_map", {}).get(str(p), str(p)) for p in all_pids],
            "Ruolo": [st.session_state.player_roles.get(str(p), "Guardia") for p in all_pids]
        })
        ed = st.data_editor(role_df, hide_index=True)
        if st.button("Aggiorna Team"):
            st.session_state.player_name_map = dict(zip(ed["ID"], ed["Nome"]))
            st.session_state.player_roles = dict(zip(ed["ID"], ed["Ruolo"]))
            st.rerun()

# Applicazione nomi
uwb["player_label"] = uwb["player_id"].astype(str).map(st.session_state.get("player_name_map", {p:str(p) for p in all_pids}))

# =========================
# 5. Elaborazione Metriche
# =========================
# Calcolo velocità e distanze
uwb['step_m'] = np.sqrt(uwb.groupby('player_id')['x_m'].diff()**2 + uwb.groupby('player_id')['y_m'].diff()**2)
uwb['dt'] = uwb.groupby('player_id')['timestamp_s'].diff()
uwb['speed_kmh'] = (uwb['step_m'] / uwb['dt'] * 3.6).clip(upper=35)

# 3. Segmenti Tattici [web:320]
uwb['fase'] = np.where(uwb['speed_kmh'] > 12, "Transizione/Contropiede", "Attacco Schierato")

kpi = uwb.groupby('player_label').agg(
    Distanza=('step_m', 'sum'),
    Vel_Max=('speed_kmh', 'max'),
    Qualità=('quality_factor', 'mean')
).reset_index()

# 4. Alert Rischio Carico [web:315]
MEDIA_STORICA_DIST = 3500 # Soglia ipotetica
kpi['Rischio'] = kpi['Distanza'] > (MEDIA_STORICA_DIST * 1.25)

# =========================
# 6. UI Principale
# =========================
st.title("Elite Coach Dashboard")

tab_live, tab_history, tab_tactic = st.tabs(["📊 Sessione Corrente", "📈 Timeline Storica", "🎯 Analisi Tattica"])

# --- TAB SESSIONE ---
with tab_live:
    m1, m2, m3 = st.columns(3)
    m1.metric("Distanza Totale Media", f"{kpi['Distanza'].mean():.0f} m")
    m2.metric("Intensità Sessione", "ALTA" if kpi['Vel_Max'].mean() > 20 else "MEDIA")
    m3.metric("Qualità Segnale", f"{kpi['Qualità'].mean():.0f}%")

    # Alert visivi
    overloaded = kpi[kpi['Rischio'] == True]
    if not overloaded.empty:
        st.markdown(f"<div class='alert-red'>⚠️ ALERT CARICO: {', '.join(overloaded['player_label'].tolist())} hanno superato la soglia di rischio infortuni.</div>", unsafe_allow_html=True)

    st.divider()
    
    col_kpi, col_ai = st.columns([1, 1.2])
    
    with col_kpi:
        st.subheader("Dati Giocatori")
        st.dataframe(kpi, use_container_width=True, hide_index=True)
        
        # 5. Note Coach
        sel_p = st.selectbox("Seleziona Giocatore per Note & Report:", kpi['player_label'].unique())
        st.session_state.coach_notes[sel_p] = st.text_area(f"Note per {sel_p}:", 
                                                         value=st.session_state.coach_notes.get(sel_p, ""), 
                                                         placeholder="Es: Oggi poco reattivo in difesa...")
        
        # Report Pro con Note e Suggerimenti
        report_txt = f"COACHTRACK PRO REPORT - {sel_p}\nData: {datetime.now().strftime('%d/%m/%Y')}\n\n"
        report_txt += f"Metriche: Distanza {kpi[kpi['player_label']==sel_p]['Distanza'].values[0]:.0f}m | Vel Max {kpi[kpi['player_label']==sel_p]['Vel_Max'].values[0]:.1f} km/h\n"
        report_txt += f"\nNote Coach:\n{st.session_state.coach_notes[sel_p]}\n"
        report_txt += f"\nAI Suggerimenti:\n- Intensità target per prossima seduta: +5%.\n- Drill consigliato: Shooting dopo sprint 20m."
        
        st.download_button(f"Scarica Report Completo {sel_p}", data=report_txt, file_name=f"CoachReport_{sel_p}.txt")

    with col_ai:
        st.subheader("AI Tactical Insight")
        st.markdown(f"""
        <div class="ai-box">
            <h4>Analisi individuale per {sel_p}</h4>
            <p>Il giocatore ha operato prevalentemente in fase di <b>Attacco Schierato</b>. 
            Il rischio carico è attualmente <b>{'ALTO' if kpi[kpi['player_label']==sel_p]['Rischio'].values[0] else 'BASSO'}</b>.</p>
            <hr style='opacity:0.2'>
            <p><b>Raccomandazione:</b> Ridurre il volume di salti nella prossima sessione per scaricare i tendini.</p>
        </div>
        """, unsafe_allow_html=True)

# --- TAB TIMELINE ---
with tab_history:
    st.subheader("1. Monitoraggio Trend (Timeline)")
    # Simulazione caricamento dati storici
    if st.button("Salva sessione attuale nello storico"):
        for _, r in kpi.iterrows():
            new_row = pd.DataFrame([{"player": r['player_label'], "dist": r['Distanza'], "date": datetime.now().strftime("%H:%M:%S") }])
            st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
        st.success("Dati aggiunti alla timeline!")

    if not st.session_state.history.empty:
        fig_hist = px.line(st.session_state.history, x="date", y="dist", color="player", title="Carico Distanza nelle ultime sessioni")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Nessuna sessione salvata. Clicca il tasto sopra per iniziare a tracciare la cronologia.")

# --- TAB TATTICA ---
with tab_tactic:
    c_a, c_b = st.columns(2)
    with c_a:
        st.subheader("2. Carico per Ruolo")
        # Aggregazione per Ruoli (dal Name Manager) [web:314]
        kpi['Ruolo'] = kpi['player_label'].map({v: st.session_state.player_roles.get(k, "Guardia") for k, v in st.session_state.get("player_name_map", {}).items()})
        role_agg = kpi.groupby('Ruolo')['Distanza'].mean().reset_index()
        fig_role = px.bar(role_agg, x='Ruolo', y='Distanza', color='Ruolo', title="Distanza Media per Ruolo")
        st.plotly_chart(fig_role, use_container_width=True)

    with c_b:
        st.subheader("3. Segmenti Tattici")
        # Suddivisione fase di gioco [web:320]
        tactic_df = uwb.groupby(['player_label', 'fase']).size().reset_index(name='tempo')
        fig_tactic = px.sunburst(tactic_df, path=['player_label', 'fase'], values='tempo', title="Distribuzione Fasi di Gioco")
        st.plotly_chart(fig_tactic, use_container_width=True)
