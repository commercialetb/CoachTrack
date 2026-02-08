# =================================================================
# COACHTRACK ELITE AI v3.1 - WITH BIOMETRICS MODULE
# =================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from pathlib import Path

# =================================================================
# IMPORTS
# =================================================================

try:
    from ai_functions import calculate_distance, predict_injury_risk
    AI_AVAILABLE = True
except:
    AI_AVAILABLE = False
    def calculate_distance(df):
        if len(df)<2: return 0.0
        dx,dy = np.diff(df['x'].values),np.diff(df['y'].values)
        return float(np.sum(np.sqrt(dx**2+dy**2)))
    def predict_injury_risk(pd,pid):
        d=calculate_distance(pd)
        return {'player_id':pid,'risk_level':'MEDIO','risk_score':40,'acwr':1.2,'fatigue':8,
                'risk_factors':['Distanza elevata'],'recommendations':['Ridurre carico']}

try:
    from ml_models import MLInjuryPredictor, PerformancePredictor
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False
    class MLInjuryPredictor:
        def extract_features(self,pd,phys={}):
            return {'total_distance':calculate_distance(pd) if len(pd)>1 else 0}
        def predict(self,f):
            rp=min(35+f.get('total_distance',0)/100,85)
            return {'risk_level':'MEDIO','risk_probability':rp,'confidence':'Media',
                    'top_risk_factors':[('Distanza',0.35)],'recommendations':['Monitorare']}
    class PerformancePredictor:
        def __init__(self): self.is_trained=False
        def extract_features(self,s,o,i=None): return {'avg_points_last5':15}
        def predict_next_game(self,f): return {'points':17.5,'assists':5,'rebounds':6,'efficiency':48,'confidence':'MEDIA'}

try:
    from physical_nutrition import generate_enhanced_nutrition, create_body_composition_viz
    PHYSICAL_AVAILABLE = True
except:
    PHYSICAL_AVAILABLE = False
    def generate_enhanced_nutrition(pid,ph,act,goal):
        w,bmr=ph.get('weight_kg',80),ph.get('bmr',2000)
        cal=int(bmr*1.55)
        return {'player_id':pid,'target_calories':cal,'protein_g':int(w*2.2),'carbs_g':int(cal*0.5/4),
                'fats_g':int(cal*0.25/9),'recommendations':['Carbs pre-workout','Proteine post'],'supplements':['Whey','Creatina'],
                'meals':[{'name':'Colazione','timing':'7:00','calories':int(cal*0.25),'protein':int(w*0.4),
                          'carbs':int(cal*0.15/4),'fats':int(cal*0.06/9),'examples':'Avena, uova'}]}
    def create_body_composition_viz(ph):
        fig=go.Figure()
        fig.add_trace(go.Pie(labels=['Muscoli','Grasso','Acqua','Altro'],
                              values=[ph.get('muscle_pct',45),ph.get('body_fat_pct',12),15,28],hole=0.4))
        fig.update_layout(title="Body Composition",height=400)
        return fig

CV_AVAILABLE=False
try:
    from cv_processor import CoachTrackVisionProcessor
    CV_AVAILABLE=True
except: pass

def add_computer_vision_tab():
    st.header("🎥 Computer Vision")
    if not CV_AVAILABLE:
        st.error("❌ CV non disponibile")
        missing_pkgs=[]
        try: import cv2
        except: missing_pkgs.append('opencv-python')
        try: from ultralytics import YOLO
        except: missing_pkgs.append('ultralytics')
        if missing_pkgs: st.error(f"Mancanti: {','.join(missing_pkgs)}")
        st.info("Aggiungi a requirements.txt per Streamlit Cloud")

# =================================================================
# BIOMETRIC MODULE (NUOVO)
# =================================================================

def render_biometric_module():
    '''Modulo biometrico completo con input manuale'''

    st.header("⚖️ Monitoraggio Biometrico")

    # Initialize biometric data in session state
    if 'biometric_data' not in st.session_state:
        st.session_state.biometric_data = pd.DataFrame(columns=[
            'player_id', 'player_name', 'timestamp', 
            'weight_kg', 'body_fat_pct', 'muscle_mass_kg',
            'water_pct', 'bone_mass_kg', 'bmr_kcal',
            'measurement_type', 'source', 'notes'
        ])

    # Sub-tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Dashboard",
        "➕ Inserimento Dati",
        "📈 Analisi Trend"
    ])

    # TAB 1: DASHBOARD
    with tab1:
        st.subheader("📊 Overview Squadra")

        if st.session_state.biometric_data.empty:
            st.info("👋 Nessun dato disponibile. Inizia inserendo misurazioni nella tab 'Inserimento Dati'.")
        else:
            # Latest measurements per player
            latest = st.session_state.biometric_data.sort_values('timestamp').groupby('player_id').last()

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Giocatori Monitorati", len(latest))

            with col2:
                avg_weight = latest['weight_kg'].mean()
                st.metric("Peso Medio", f"{avg_weight:.1f} kg" if pd.notna(avg_weight) else "N/A")

            with col3:
                avg_bf = latest['body_fat_pct'].mean()
                st.metric("Body Fat Medio", f"{avg_bf:.1f}%" if pd.notna(avg_bf) else "N/A")

            with col4:
                recent = st.session_state.biometric_data[
                    st.session_state.biometric_data['timestamp'] >= 
                    datetime.now() - timedelta(days=7)
                ]
                st.metric("Misurazioni (7gg)", len(recent))

            st.divider()

            # Alerts
            st.subheader("🚨 Alert Attivi")
            alerts = []

            for player_id in latest.index:
                player_data = st.session_state.biometric_data[
                    st.session_state.biometric_data['player_id'] == player_id
                ]

                if len(player_data) >= 2:
                    latest_weight = latest.loc[player_id, 'weight_kg']
                    avg_7d = player_data.tail(7)['weight_kg'].mean()
                    weight_change = latest_weight - avg_7d

                    if abs(weight_change) > 2.0:
                        player_name = latest.loc[player_id, 'player_name']
                        alerts.append({
                            'player': player_name,
                            'message': f"Peso {weight_change:+.1f}kg vs media 7gg",
                            'severity': 'high' if abs(weight_change) > 3 else 'medium'
                        })

                    # Hydration check
                    water = latest.loc[player_id, 'water_pct']
                    if pd.notna(water) and water < 55:
                        player_name = latest.loc[player_id, 'player_name']
                        alerts.append({
                            'player': player_name,
                            'message': f"Possibile disidratazione: {water:.1f}% acqua",
                            'severity': 'high'
                        })

            if alerts:
                for alert in alerts:
                    if alert['severity'] == 'high':
                        st.error(f"**{alert['player']}**: ⚠️ {alert['message']}")
                    else:
                        st.warning(f"**{alert['player']}**: {alert['message']}")
            else:
                st.success("✅ Nessun alert attivo - Tutti i parametri nella norma")

            st.divider()

            # Table
            st.subheader("📋 Ultime Misurazioni")

            display_df = latest[[
                'player_name', 'timestamp', 'weight_kg', 'body_fat_pct', 
                'muscle_mass_kg', 'water_pct', 'source'
            ]].copy()

            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%d/%m/%Y %H:%M')
            display_df.columns = ['Giocatore', 'Data', 'Peso (kg)', 'Grasso (%)', 
                                  'Muscolo (kg)', 'Acqua (%)', 'Fonte']

            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # TAB 2: INPUT MANUALE
    with tab2:
        st.subheader("➕ Inserimento Misurazione Manuale")

        st.info("💡 Inserisci i dati manualmente. Compila almeno il peso, gli altri campi sono opzionali.")

        with st.form("manual_measurement_form"):
            col1, col2 = st.columns(2)

            with col1:
                player_name = st.text_input(
                    "Nome Giocatore *",
                    placeholder="es. Mario Rossi"
                )

                weight = st.number_input(
                    "Peso (kg) *",
                    min_value=40.0,
                    max_value=150.0,
                    value=75.0,
                    step=0.1,
                    format="%.1f"
                )

                body_fat = st.number_input(
                    "Grasso Corporeo (%)",
                    min_value=3.0,
                    max_value=50.0,
                    value=None,
                    step=0.1,
                    format="%.1f",
                    help="Opzionale - Misurato con plicometro o BIA"
                )

                muscle_mass = st.number_input(
                    "Massa Muscolare (kg)",
                    min_value=20.0,
                    max_value=80.0,
                    value=None,
                    step=0.1,
                    format="%.1f",
                    help="Opzionale"
                )

            with col2:
                water = st.number_input(
                    "Acqua Corporea (%)",
                    min_value=40.0,
                    max_value=75.0,
                    value=None,
                    step=0.1,
                    format="%.1f",
                    help="Opzionale - Normale: 55-65%"
                )

                bone_mass = st.number_input(
                    "Massa Ossea (kg)",
                    min_value=2.0,
                    max_value=5.0,
                    value=None,
                    step=0.1,
                    format="%.1f",
                    help="Opzionale"
                )

                measurement_type = st.selectbox(
                    "Momento Misurazione",
                    ["Pre-allenamento", "Post-allenamento", "Mattina", "Altro"]
                )

                notes = st.text_area(
                    "Note",
                    placeholder="Eventuali annotazioni...",
                    height=100
                )

            submitted = st.form_submit_button("💾 Salva Misurazione", type="primary", use_container_width=True)

            if submitted:
                if not player_name:
                    st.error("❌ Il nome del giocatore è obbligatorio!")
                else:
                    # Generate player ID
                    import hashlib
                    player_id = hashlib.md5(player_name.encode()).hexdigest()[:8]

                    # Calculate BMR if enough data
                    bmr = None
                    if muscle_mass and body_fat:
                        bmr = int(370 + (21.6 * muscle_mass))

                    # Create new measurement
                    new_row = pd.DataFrame([{
                        'player_id': player_id,
                        'player_name': player_name,
                        'timestamp': datetime.now(),
                        'weight_kg': weight,
                        'body_fat_pct': body_fat,
                        'muscle_mass_kg': muscle_mass,
                        'water_pct': water,
                        'bone_mass_kg': bone_mass,
                        'bmr_kcal': bmr,
                        'measurement_type': measurement_type.lower().replace('-', '_'),
                        'source': 'manual',
                        'notes': notes
                    }])

                    st.session_state.biometric_data = pd.concat([
                        st.session_state.biometric_data, 
                        new_row
                    ], ignore_index=True)

                    st.success(f"✅ Misurazione salvata per {player_name}!")
                    st.balloons()

                    # Check for alerts
                    player_data = st.session_state.biometric_data[
                        st.session_state.biometric_data['player_id'] == player_id
                    ]

                    if len(player_data) >= 2:
                        avg_7d = player_data.tail(7)['weight_kg'].mean()
                        weight_change = weight - avg_7d

                        if abs(weight_change) > 2.0:
                            st.warning(f"⚠️ Alert: Peso {weight_change:+.1f}kg vs media 7 giorni")

                        if water and water < 55:
                            st.error(f"🚨 Alert: Possibile disidratazione ({water:.1f}% acqua)")

    # TAB 3: ANALISI TREND
    with tab3:
        st.subheader("📈 Analisi Trend Biometrici")

        if st.session_state.biometric_data.empty:
            st.info("Nessun dato disponibile per l'analisi")
        else:
            players = st.session_state.biometric_data['player_name'].unique()
            selected_player = st.selectbox("Seleziona Giocatore", players)

            if selected_player:
                player_id = st.session_state.biometric_data[
                    st.session_state.biometric_data['player_name'] == selected_player
                ]['player_id'].iloc[0]

                days = st.slider("Periodo analisi (giorni)", 7, 180, 30)

                cutoff_date = datetime.now() - timedelta(days=days)
                player_df = st.session_state.biometric_data[
                    (st.session_state.biometric_data['player_id'] == player_id) &
                    (st.session_state.biometric_data['timestamp'] >= cutoff_date)
                ].sort_values('timestamp')

                if len(player_df) < 2:
                    st.warning("Dati insufficienti per analisi trend (minimo 2 misurazioni)")
                else:
                    # Trend summary
                    weight_change = player_df['weight_kg'].iloc[-1] - player_df['weight_kg'].iloc[0]
                    weight_change_pct = (weight_change / player_df['weight_kg'].iloc[0]) * 100

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Variazione Peso",
                            f"{weight_change:+.1f} kg",
                            delta=f"{weight_change_pct:+.1f}%"
                        )

                    with col2:
                        if player_df['body_fat_pct'].notna().sum() >= 2:
                            bf_change = player_df['body_fat_pct'].iloc[-1] - player_df['body_fat_pct'].iloc[0]
                            st.metric(
                                "Variazione Grasso",
                                f"{bf_change:+.1f} %",
                                delta="↓ Bene" if bf_change < 0 else "↑ Attenzione" if bf_change > 0 else "→ Stabile"
                            )
                        else:
                            st.metric("Variazione Grasso", "N/A")

                    with col3:
                        if player_df['muscle_mass_kg'].notna().sum() >= 2:
                            muscle_change = player_df['muscle_mass_kg'].iloc[-1] - player_df['muscle_mass_kg'].iloc[0]
                            st.metric(
                                "Variazione Muscolo",
                                f"{muscle_change:+.1f} kg",
                                delta="↑ Bene" if muscle_change > 0 else "↓ Attenzione" if muscle_change < 0 else "→ Stabile"
                            )
                        else:
                            st.metric("Variazione Muscolo", "N/A")

                    st.divider()

                    # Weight trend chart
                    fig_weight = go.Figure()
                    fig_weight.add_trace(go.Scatter(
                        x=player_df['timestamp'],
                        y=player_df['weight_kg'],
                        mode='lines+markers',
                        name='Peso',
                        line=dict(color='#3498DB', width=3),
                        marker=dict(size=8)
                    ))

                    fig_weight.update_layout(
                        title="📊 Trend Peso",
                        xaxis_title="Data",
                        yaxis_title="Peso (kg)",
                        hovermode='x unified',
                        height=400
                    )

                    st.plotly_chart(fig_weight, use_container_width=True)

                    # Body composition chart (if data available)
                    if player_df['body_fat_pct'].notna().any():
                        fig_comp = go.Figure()

                        fig_comp.add_trace(go.Scatter(
                            x=player_df['timestamp'],
                            y=player_df['body_fat_pct'],
                            mode='lines+markers',
                            name='Grasso %',
                            line=dict(color='#E74C3C', width=2)
                        ))

                        if player_df['muscle_mass_kg'].notna().any():
                            fig_comp.add_trace(go.Scatter(
                                x=player_df['timestamp'],
                                y=player_df['muscle_mass_kg'],
                                mode='lines+markers',
                                name='Muscolo kg',
                                yaxis='y2',
                                line=dict(color='#27AE60', width=2)
                            ))

                        fig_comp.update_layout(
                            title="📊 Body Composition",
                            xaxis_title="Data",
                            yaxis_title="Grasso (%)",
                            yaxis2=dict(
                                title="Muscolo (kg)",
                                overlaying='y',
                                side='right'
                            ),
                            hovermode='x unified',
                            height=400
                        )

                        st.plotly_chart(fig_comp, use_container_width=True)

# =================================================================
# MAIN
# =================================================================

st.set_page_config(page_title="CoachTrack Elite",page_icon="🏀",layout="wide")

if 'logged_in' not in st.session_state: st.session_state.logged_in=False

if not st.session_state.logged_in:
    st.title("🏀 CoachTrack Elite AI")
    col1,col2,col3=st.columns([1,2,1])
    with col2:
        u=st.text_input("Username",value="admin")
        p=st.text_input("Password",type="password",value="admin")
        if st.button("Login",type="primary",use_container_width=True):
            if u=="admin" and p=="admin":
                st.session_state.logged_in=True
                st.rerun()
        st.info("admin / admin")
    st.stop()

if 'tracking_data' not in st.session_state: st.session_state.tracking_data={}
if 'physical_profiles' not in st.session_state: st.session_state.physical_profiles={}
if 'ml_injury_model' not in st.session_state: st.session_state.ml_injury_model=MLInjuryPredictor()
if 'performance_model' not in st.session_state: st.session_state.performance_model=PerformancePredictor()

with st.sidebar:
    st.title("🏀 CoachTrack")
    st.markdown("---")
    col1,col2=st.columns(2)
    with col1:
        st.success("✅" if AI_AVAILABLE else "❌"); st.caption("AI")
        st.success("✅" if ML_AVAILABLE else "❌"); st.caption("ML")
    with col2:
        st.success("✅" if CV_AVAILABLE else "❌"); st.caption("CV")
        st.success("✅" if PHYSICAL_AVAILABLE else "❌"); st.caption("PH")
    st.markdown("---")
    st.metric("Players",len(st.session_state.tracking_data))
    st.metric("Physical",len(st.session_state.physical_profiles))
    # NEW: Biometric count
    bio_count = 0
    if 'biometric_data' in st.session_state and not st.session_state.biometric_data.empty:
        bio_count = len(st.session_state.biometric_data['player_id'].unique())
    st.metric("Biometric", bio_count)
    st.markdown("---")
    if st.button("Logout",use_container_width=True):
        st.session_state.logged_in=False
        st.rerun()

st.title("🏀 CoachTrack Elite AI v3.1")
st.markdown("**Complete:** AI + ML + CV + Physical + Nutrition + **Biometrics** + Analytics")

# MODIFIED: Added Biometrics tab
tab1,tab2,tab3,tab4,tab5,tab6,tab7=st.tabs([
    "⚙️ Config",
    "🤖 AI Features",
    "🎥 CV",
    "🧠 ML",
    "💪 Physical",
    "⚖️ Biometrics",  # NEW TAB
    "📊 Analytics"
])

# (Il resto del codice delle tabs rimane identico...)
