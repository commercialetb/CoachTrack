# =================================================================
# COACHTRACK ELITE AI v3.0 - MAIN APPLICATION + CV INTEGRATION
# =================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO
import json
import time
import os
from pathlib import Path

# =================================================================
# COMPUTER VISION IMPORTS (con fallback)
# =================================================================

try:
    from cv_processor import CoachTrackVisionProcessor
    from cv_camera import CameraManager, CourtCalibrator
    from cv_tracking import PlayerDetector, SimpleTracker, BallDetector
    CV_AVAILABLE = True
    CV_STATUS = "✅ Computer Vision disponibile"
except ImportError as e:
    CV_AVAILABLE = False
    CV_STATUS = f"⚠️ CV non disponibile: {e}"
    print(CV_STATUS)

# =================================================================
# GROQ CLIENT INITIALIZATION
# =================================================================

try:
    from groq import Groq
    GROQ_INSTALLED = True
except ImportError:
    GROQ_INSTALLED = False
    Groq = None

def initialize_groq_client():
    if not GROQ_INSTALLED:
        return None, False, "Groq library non installata"

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except:
            pass

    if api_key:
        try:
            client = Groq(api_key=api_key)
            return client, True, "✅ Groq connesso"
        except Exception as e:
            return None, False, f"Errore Groq: {str(e)}"
    return None, False, "API Key non configurata"

GROQ_CLIENT, GROQ_AVAILABLE, GROQ_STATUS = initialize_groq_client()

def call_groq_llm(prompt, system_message="Sei un esperto di sport science.", temperature=0.7, max_tokens=2000):
    if not GROQ_AVAILABLE or GROQ_CLIENT is None:
        return f"⚠️ Groq non disponibile"
    try:
        response = GROQ_CLIENT.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Errore: {str(e)}"

# =================================================================
# UTILITY FUNCTIONS
# =================================================================

def calculate_distance(df):
    if len(df) < 2:
        return 0.0
    dx = np.diff(df['x'].values)
    dy = np.diff(df['y'].values)
    return float(np.sum(np.sqrt(dx**2 + dy**2)))

def calculate_speed(df):
    if len(df) < 2:
        return df
    if 'speed_kmh_calc' in df.columns:
        return df
    df = df.copy()
    dx = np.diff(df['x'].values)
    dy = np.diff(df['y'].values)
    dt = np.diff(df['timestamp'].values).astype(float) / 1000.0
    dt[dt == 0] = 0.001
    speeds = (np.sqrt(dx**2 + dy**2) / dt) * 3.6
    df.loc[df.index[1:], 'speed_kmh_calc'] = speeds
    df.loc[df.index[0], 'speed_kmh_calc'] = 0.0
    return df

def detect_jumps_imu(df, threshold_g=1.5):
    if 'az' not in df.columns:
        return []
    jumps = []
    for i, az in enumerate(df['az'].values):
        if az > threshold_g:
            jumps.append({
                'timestamp': int(df['timestamp'].iloc[i]),
                'peak_g': round(float(az), 2),
                'duration_ms': 200,
                'estimated_height_cm': round((az-1)*20, 1)
            })
    return jumps[:10]

# =================================================================
# AI FUNCTIONS
# =================================================================

def predict_injury_risk(player_data, player_id):
    if len(player_data) < 10:
        return {
            'player_id': player_id,
            'risk_level': 'BASSO',
            'risk_score': 10,
            'acwr': 1.0,
            'asymmetry': 5.0,
            'fatigue': 5.0,
            'risk_factors': ['Dati insufficienti'],
            'recommendations': ['Raccogliere più dati']
        }

    distance = calculate_distance(player_data)
    player_data_with_speed = calculate_speed(player_data)
    avg_speed = player_data_with_speed['speed_kmh_calc'].mean()

    risk_score = 25 if distance < 200 else 40 if distance < 500 else 60
    risk_level = 'ALTO' if risk_score > 60 else 'MEDIO' if risk_score > 30 else 'BASSO'

    risk_factors = [f'Distanza: {distance:.1f}m', f'Velocità: {avg_speed:.1f} km/h']
    recommendations = ['Monitorare carico', 'Bilanciare recupero']

    return {
        'player_id': player_id,
        'risk_level': risk_level,
        'risk_score': risk_score,
        'acwr': 1.2,
        'asymmetry': 10.0,
        'fatigue': 8.0,
        'risk_factors': risk_factors,
        'recommendations': recommendations
    }

def recommend_offensive_plays(player_data):
    if len(player_data) < 5:
        return {'recommended_plays': ['Dati insufficienti'], 'reasoning': ['Caricare dati']}

    distance = calculate_distance(player_data)
    return {
        'recommended_plays': ['Pick and Roll', 'Motion Offense', 'Fast Break'],
        'reasoning': [f'Movimento: {distance:.0f}m']
    }

def analyze_movement_patterns(player_data, player_id):
    if len(player_data) < 10:
        return {'player_id': player_id, 'pattern_type': 'UNKNOWN', 'insights': ['Dati insufficienti'], 'anomalies': []}

    distance = calculate_distance(player_data)
    pattern = 'DYNAMIC' if distance > 100 else 'BALANCED'
    return {'player_id': player_id, 'pattern_type': pattern, 'insights': [f'Distanza: {distance:.1f}m'], 'anomalies': []}

def test_groq_connection():
    return GROQ_AVAILABLE, GROQ_STATUS

# =================================================================
# ML MODELS MOCK
# =================================================================

class MLInjuryPredictor:
    def extract_features(self, player_data, physical_data=None, player_age=25, previous_injuries=0, training_history=None):
        return {'acwr': 1.2, 'asymmetry': 10.0, 'fatigue': 8.0, 'workload': 100.0, 'restdays': 2, 'age': player_age}

    def predict(self, features):
        risk_prob = 35
        risk_level = 'MEDIO'
        return {
            'risk_level': risk_level,
            'risk_probability': risk_prob,
            'confidence': 'Media',
            'top_risk_factors': [('ACWR', 0.25), ('Fatigue', 0.20)],
            'recommendations': ['Monitorare carico', 'Aumentare recupero']
        }

# =================================================================
# CV TAB FUNCTION
# =================================================================

def add_computer_vision_tab():
    """Tab Computer Vision completo"""
    st.header("🎥 Computer Vision System")

    if not CV_AVAILABLE:
        st.error("❌ Moduli CV non disponibili")
        st.info("📦 Installa dipendenze:")
        st.code("pip install -r requirements_cv.txt", language="bash")
        st.markdown("### 📋 File necessari:")
        st.markdown("- `cv_camera.py`\n- `cv_processor.py`\n- `cv_tracking.py`")
        return

    st.success(CV_STATUS)

    # Tabs CV
    cv_tab1, cv_tab2, cv_tab3, cv_tab4 = st.tabs([
        "📹 Live Tracking",
        "📁 Process Video",
        "🎯 Calibrazione",
        "📊 Analysis"
    ])

    # ============ TAB 1: LIVE TRACKING ============
    with cv_tab1:
        st.subheader("📹 Live Camera Tracking")

        col1, col2 = st.columns([2, 1])
        with col1:
            camera_source = st.text_input(
                "Camera Source",
                value="0",
                help="0 = USB webcam, rtsp://ip/stream = WiFi camera"
            )
        with col2:
            duration = st.number_input(
                "Durata (sec)",
                min_value=0,
                max_value=3600,
                value=60,
                help="0 = infinito"
            )

        col3, col4 = st.columns(2)
        with col3:
            output_file = st.text_input("Output JSON", value="live_tracking.json")
        with col4:
            visualize = st.checkbox("Mostra Video", value=False, help="Disabilita per migliori performance")

        if st.button("🎬 Start Live Tracking", type="primary"):
            with st.spinner("Inizializzazione camera..."):
                processor = CoachTrackVisionProcessor(camera_source)
                if processor.initialize():
                    st.success("✅ Camera connessa")

                    if not processor.is_calibrated:
                        st.warning("⚠️ Sistema non calibrato. Vai alla tab Calibrazione.")

                    st.info("🎬 Processing avviato... Premi 'q' nella finestra video per fermare")

                    try:
                        processor.run_realtime(
                            output_file=output_file,
                            visualize=visualize,
                            duration=duration
                        )
                        st.success(f"✅ Tracking completato! Dati: {output_file}")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Errore: {e}")
                else:
                    st.error("❌ Impossibile connettersi alla camera")

        # Show recent files
        st.markdown("---")
        st.subheader("📂 Recent Tracking Files")
        tracking_files = list(Path(".").glob("*tracking*.json"))
        if tracking_files:
            selected_file = st.selectbox("Seleziona file", tracking_files)
            if st.button("📊 Carica Dati"):
                with open(selected_file, 'r') as f:
                    data = json.load(f)

                st.json(data['metadata'])

                # Convert to dataframe
                frames_data = []
                for frame in data['frames']:
                    for player in frame.get('players', []):
                        frames_data.append({
                            'frame': frame['frame_number'],
                            'timestamp': frame['timestamp'],
                            'player_id': player['player_id'],
                            'x': player['x'],
                            'y': player['y'],
                            'conf': player['conf']
                        })

                if frames_data:
                    df = pd.DataFrame(frames_data)
                    st.dataframe(df.head(50))
                    st.session_state.cv_tracking_data = df
                    st.session_state.tracking_data = {pid: df[df['player_id']==pid] for pid in df['player_id'].unique()}
                    st.success(f"✅ Caricati {len(df)} punti tracking per {df['player_id'].nunique()} giocatori")
        else:
            st.info("ℹ️ Nessun file tracking trovato")

    # ============ TAB 2: PROCESS VIDEO ============
    with cv_tab2:
        st.subheader("📁 Process Video File")

        uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

        if uploaded_video:
            video_path = f"uploaded_{uploaded_video.name}"
            with open(video_path, 'wb') as f:
                f.write(uploaded_video.read())
            st.success(f"✅ Video caricato: {video_path}")

            col1, col2 = st.columns(2)
            with col1:
                output_json = st.text_input("Output JSON", value="video_tracking.json")
            with col2:
                create_annotated = st.checkbox("Crea Video Annotato", value=False)

            if st.button("🎬 Process Video", type="primary"):
                with st.spinner("Processing video..."):
                    processor = CoachTrackVisionProcessor(video_path)
                    output_video = f"annotated_{video_path}" if create_annotated else None

                    success = processor.process_video_file(
                        video_path,
                        output_json,
                        output_video
                    )

                    if success:
                        st.success("✅ Video processato!")

                        with open(output_json, 'r') as f:
                            json_data = f.read()
                        st.download_button(
                            "📥 Download JSON",
                            data=json_data,
                            file_name=output_json,
                            mime="application/json"
                        )
                    else:
                        st.error("❌ Errore processing")

    # ============ TAB 3: CALIBRAZIONE ============
    with cv_tab3:
        st.subheader("🎯 Court Calibration")

        st.markdown("""
        ### 📐 Procedura Calibrazione:
        1. Connetti la camera
        2. Assicurati che il campo sia **completamente visibile**
        3. Clicca sui 4 angoli del campo nell'ordine:
           - **Basso-Sinistra**
           - **Basso-Destra**
           - **Alto-Destra**
           - **Alto-Sinistra**
        4. Premi 'q' per salvare
        """)

        camera_source_calib = st.text_input(
            "Camera Source",
            value="0",
            key="calib_source"
        )

        if st.button("🎯 Start Calibration", type="primary"):
            with st.spinner("Inizializzazione..."):
                processor = CoachTrackVisionProcessor(camera_source_calib)
                if processor.initialize(calibration_file=None):
                    st.info("📍 Clicca sui 4 angoli nella finestra che si apre...")
                    success = processor.calibrate_court()
                    if success:
                        st.success("✅ Calibrazione completata e salvata!")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Calibrazione fallita")
                else:
                    st.error("❌ Impossibile connettersi alla camera")

        # Show current calibration
        st.markdown("---")
        st.subheader("Current Calibration")
        calib_file = Path("camera_calibration.json")
        if calib_file.exists():
            with open(calib_file, 'r') as f:
                calib = json.load(f)
            st.json(calib)
            if st.button("🗑️ Delete Calibration"):
                calib_file.unlink()
                st.success("✅ Calibrazione eliminata")
                st.rerun()
        else:
            st.info("ℹ️ Nessuna calibrazione presente")

    # ============ TAB 4: ANALYSIS ============
    with cv_tab4:
        st.subheader("📊 Tracking Data Analysis")

        if 'cv_tracking_data' not in st.session_state:
            st.info("ℹ️ Carica tracking data dalla tab 'Live Tracking'")
            return

        df = st.session_state.cv_tracking_data

        # Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Frames", df['frame'].nunique())
        with col2:
            st.metric("Players", df['player_id'].nunique())
        with col3:
            st.metric("Avg Confidence", f"{df['conf'].mean():.2%}")
        with col4:
            duration = (df['timestamp'].max() - df['timestamp'].min())
            st.metric("Duration", f"{duration:.1f}s")

        # Player selection
        selected_player = st.selectbox(
            "Select Player",
            options=sorted(df['player_id'].unique())
        )

        player_df = df[df['player_id'] == selected_player]

        # Trajectory plot
        st.subheader(f"Player {selected_player} Trajectory")
        fig = go.Figure()

        # Court background
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=28, y1=15,
            line=dict(color="white", width=2),
            fillcolor="rgba(0,100,0,0.1)"
        )

        # Trajectory
        fig.add_trace(go.Scatter(
            x=player_df['x'],
            y=player_df['y'],
            mode='lines+markers',
            name=f'Player {selected_player}',
            line=dict(color='red', width=2),
            marker=dict(size=4)
        ))

        fig.update_layout(
            title=f"Court Position - Player {selected_player}",
            xaxis_title="X (meters)",
            yaxis_title="Y (meters)",
            width=800,
            height=500,
            plot_bgcolor='darkgreen'
        )

        st.plotly_chart(fig, use_container_width=True)

# =================================================================
# STREAMLIT APP
# =================================================================

st.set_page_config(
    page_title="CoachTrack Elite AI",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Login
def check_login(username, password):
    return username == "admin" and password == "admin"

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🏀 CoachTrack Elite AI")
    st.markdown("### 🔐 Login")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password", value="admin")
        if st.button("Login", use_container_width=True, type="primary"):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Credenziali errate")
        st.info("💡 Default: admin / admin")
    st.stop()

# Session state
if 'tracking_data' not in st.session_state:
    st.session_state.tracking_data = {}
if 'imu_data' not in st.session_state:
    st.session_state.imu_data = {}
if 'physical_profiles' not in st.session_state:
    st.session_state.physical_profiles = {}
if 'ml_injury_model' not in st.session_state:
    st.session_state.ml_injury_model = MLInjuryPredictor()

# Sidebar
with st.sidebar:
    st.title("🏀 CoachTrack Elite")
    st.markdown("---")

    st.markdown("### 🔌 System Status")
    col1, col2 = st.columns(2)
    with col1:
        if GROQ_AVAILABLE:
            st.success("Groq ✅")
        else:
            st.error("Groq ❌")
    with col2:
        if CV_AVAILABLE:
            st.success("CV ✅")
        else:
            st.warning("CV ⚠️")

    st.markdown("---")
    st.markdown("### 📈 Data Summary")
    uwb_count = len(st.session_state.tracking_data)
    phys_count = len(st.session_state.physical_profiles)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Players", uwb_count)
    with col2:
        st.metric("Physical", phys_count)

    if uwb_count == 0 and not CV_AVAILABLE:
        st.caption("⚠️ Carica dati CSV o usa CV")

    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

# Main
st.title("🏀 CoachTrack Elite AI v3.0")
st.markdown("Sistema completo: ML, Groq AI, Computer Vision e Tactical Analytics")

# TABS PRINCIPALI
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚙️ Configurazione",
    "🤖 AI Features",
    "🎥 Computer Vision",
    "🧠 ML Advanced",
    "📊 Analytics"
])

# TAB 1: CONFIGURAZIONE
with tab1:
    st.header("⚙️ Configurazione Dati")
    st.markdown("### 📡 Carica Dati Tracking UWB")

    uploaded_uwb = st.file_uploader("Carica CSV Tracking", type=['csv'], key='uwb_upload')

    if uploaded_uwb:
        try:
            df = pd.read_csv(uploaded_uwb, sep=';')
            st.success(f"✅ File caricato: {len(df)} righe")

            required_cols = ['player_id', 'timestamp', 'x', 'y']
            if all(col in df.columns for col in required_cols):
                for player_id in df['player_id'].unique():
                    player_df = df[df['player_id'] == player_id].copy()
                    st.session_state.tracking_data[player_id] = player_df
                st.success(f"✅ Dati importati per {len(df['player_id'].unique())} giocatori")
                with st.expander("📋 Anteprima"):
                    st.dataframe(df.head(20))
            else:
                st.error(f"❌ CSV deve contenere: {', '.join(required_cols)}")
        except Exception as e:
            st.error(f"❌ Errore: {e}")

# TAB 2: AI FEATURES
with tab2:
    st.header("🤖 AI Elite Features")

    if not st.session_state.tracking_data:
        st.warning("⚠️ Carica dati in Tab 1 o usa Computer Vision in Tab 3")
    else:
        st.success(f"✅ {len(st.session_state.tracking_data)} giocatori disponibili")

        selected_ai = st.selectbox("Giocatore", list(st.session_state.tracking_data.keys()), key='ai_player')
        ai_feature = st.selectbox("Funzionalità", [
            "🩺 Injury Risk",
            "🏀 Offensive Plays",
            "🏃 Movement Patterns"
        ])

        if st.button("🚀 Esegui Analisi", type="primary"):
            player_data = st.session_state.tracking_data[selected_ai]

            if "Injury" in ai_feature:
                result = predict_injury_risk(player_data, selected_ai)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rischio", result['risk_level'])
                with col2:
                    st.metric("Score", result['risk_score'])
                with col3:
                    st.metric("ACWR", result['acwr'])

                st.markdown("#### Fattori di Rischio")
                for f in result['risk_factors']:
                    st.write(f"- {f}")

                st.markdown("#### Raccomandazioni")
                for r in result['recommendations']:
                    st.info(f"✓ {r}")

            elif "Offensive" in ai_feature:
                result = recommend_offensive_plays(player_data)
                st.markdown("#### Giocate Consigliate")
                for play in result['recommended_plays']:
                    st.success(f"🏀 {play}")

            elif "Movement" in ai_feature:
                result = analyze_movement_patterns(player_data, selected_ai)
                st.metric("Pattern Type", result['pattern_type'])
                for insight in result['insights']:
                    st.info(insight)

# TAB 3: COMPUTER VISION
with tab3:
    add_computer_vision_tab()

# TAB 4: ML ADVANCED
with tab4:
    st.header("🧠 ML Advanced")
    st.info("Funzionalità ML avanzate disponibili")

# TAB 5: ANALYTICS
with tab5:
    st.header("📊 Analytics")
    st.info("Dashboard analytics in sviluppo")

st.caption("CoachTrack Elite AI v3.0 © 2026")
