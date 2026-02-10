# =================================================================
# COACHTRACK ELITE AI v3.2 - FINAL WORKING VERSION
# =================================================================

import sys
import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

print("="*70)
print("🚀 COACHTRACK APP STARTING")
print("="*70)

# =================================================================
# CHECK OPENCV
# =================================================================
CV_AVAILABLE = False
try:
    import cv2
    CV_AVAILABLE = True
    print(f"✅ OpenCV {cv2.__version__}")
except ImportError:
    print("⚠️ OpenCV non disponibile")

# =================================================================
# AI ADVANCED MODULE (YOLOv8)
# =================================================================
try:
    from cv_ai_advanced import CVAIPipeline
    AI_ADVANCED_AVAILABLE = True
    YOLO_AVAILABLE = True
    print("✅ CV AI Pipeline v5.0 (YOLOv8)")
except ImportError:
    AI_ADVANCED_AVAILABLE = False
    YOLO_AVAILABLE = False
    print("⚠️ CV AI not available")

# =================================================================
# HELPER FUNCTIONS
# =================================================================
def calculate_distance(df):
    if len(df) < 2:
        return 0.0
    dx, dy = np.diff(df['x'].values), np.diff(df['y'].values)
    return float(np.sum(np.sqrt(dx**2 + dy**2)))

# =================================================================
# COMPUTER VISION TAB
# =================================================================
def add_computer_vision_tab():
    """Computer Vision with YOLOv8 AI Analysis"""

    # IMPORT GLOBALI
    import pandas as pd
    import plotly.express as px
    from pathlib import Path
    import json
    import cv2
    import os
    import time
    import numpy as np

    st.header("🎥 Computer Vision")

    if not CV_AVAILABLE:
        st.error("❌ OpenCV non disponibile")
        return

    st.success("✅ Computer Vision Online")

    cv_tab1, cv_tab2, cv_tab3, cv_tab4 = st.tabs([
        "🎬 Video Info",
        "🎯 Calibration", 
        "📊 Dashboard",
        "🧠 AI Analysis"
    ])

    # TAB 1: VIDEO INFO
    with cv_tab1:
        st.subheader("🎬 Video Info")
        st.info("📹 Upload video - Usa 'AI Analysis' per processing")
        uv = st.file_uploader("Carica Video", type=['mp4','avi','mov','mkv'], key="vid_info")
        if uv:
            vp = f"temp_{uv.name}"
            with open(vp,'wb') as f:
                f.write(uv.read())
            st.success(f"✅ {uv.name}")
            try:
                cap = cv2.VideoCapture(vp)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                dur = fc/fps if fps>0 else 0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("⏱️ Durata", f"{dur:.1f}s")
                c2.metric("🎞️ FPS", fps)
                c3.metric("📸 Frame", f"{fc:,}")
                c4.metric("📐 Risoluzione", f"{w}x{h}")
                st.success("✅ Vai tab 'AI Analysis' per processing!")
            except Exception as e:
                st.error(f"❌ {e}")
            finally:
                if os.path.exists(vp):
                    try:
                        os.remove(vp)
                    except:
                        pass

    # TAB 2: CALIBRATION
    with cv_tab2:
        st.subheader("🎯 Court Calibration")
        st.info("📐 Feature in sviluppo")

    # TAB 3: DASHBOARD
    with cv_tab3:
        st.subheader("📊 Analysis Dashboard")
        st.info("📥 Upload JSON da AI Analysis")

        uj = st.file_uploader("📥 Carica JSON", type=['json'], key="json_up")

        if uj:
            try:
                data = json.load(uj)
                st.success(f"✅ {uj.name}")

                if 'statistics' in data:
                    s = data['statistics']
                    c1,c2,c3 = st.columns(3)
                    c1.metric("📸 Pose", s.get('total_poses_detected',0))
                    c2.metric("🎯 Actions", s.get('total_actions',0))
                    c3.metric("🏀 Shots", s.get('total_shots',0))

                st.markdown("---")

                if 'actions' in data and len(data['actions'])>0:
                    st.markdown("### 🎯 Actions")
                    adf = pd.DataFrame(data['actions'])
                    st.dataframe(adf, use_container_width=True)

                    if 'action' in adf.columns:
                        ac = adf['action'].value_counts()
                        fig = px.bar(x=ac.index, y=ac.values, 
                                    labels={'x':'Azione','y':'Conteggio'})
                        st.plotly_chart(fig, use_container_width=True)

                if 'shots' in data and len(data['shots'])>0:
                    st.markdown("### 🏀 Shots")
                    sdf = pd.DataFrame(data['shots'])
                    st.dataframe(sdf, use_container_width=True)
                    if 'form_score' in sdf.columns:
                        avg = sdf['form_score'].mean()
                        st.metric("📊 Form Score", f"{avg:.1f}/100")

                with st.expander("📄 Raw JSON"):
                    st.json(data)

            except Exception as e:
                st.error(f"❌ {e}")
        else:
            jf = list(Path('.').glob('*.json'))
            if jf:
                st.info(f"📁 {len(jf)} JSON sul server")
                sel = st.selectbox("Seleziona", [f.name for f in jf])
                if st.button("📊 Carica"):
                    with open(sel,'r') as f:
                        st.json(json.load(f))
            else:
                st.warning("⚠️ Usa AI Analysis per generare JSON")

    # TAB 4: AI ANALYSIS
    with cv_tab4:
        st.subheader("🧠 AI Advanced Analysis")
        st.markdown("---")

        if not AI_ADVANCED_AVAILABLE:
            st.error("❌ AI module non disponibile")
            st.code("Verifica cv_ai_advanced.py")
            return

        st.success("✅ YOLOv8 Pose Analysis attiva!")
        st.info("🤖 Features: Pose Detection + Action Recognition + Shot Analysis")

        st.markdown("### 📹 Upload Video")
        uploaded_video_ai = st.file_uploader(
            "Carica video", 
            type=['mp4','avi','mov','mkv'],
            key="ai_video",
            help="Video di partita/allenamento"
        )

        if uploaded_video_ai:
            import os
            video_path = f"temp_ai_{uploaded_video_ai.name}"
            with st.spinner("📤 Caricamento..."):
                with open(video_path,'wb') as f:
                    f.write(uploaded_video_ai.read())

            st.success(f"✅ {uploaded_video_ai.name}")

            st.markdown("### ⚙️ Opzioni")
            col1, col2 = st.columns(2)
            with col1:
                analyze_actions = st.checkbox("🎯 Action Recognition", value=True)
                analyze_shots = st.checkbox("🏀 Shot Tracking", value=True)
            with col2:
                analyze_pose = st.checkbox("🤸 Pose Analysis", value=True)
                output_json = st.text_input("📄 Output", "ai_analysis.json")

            st.markdown("---")

            if st.button("🚀 Avvia AI Analysis", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    status_text.text("🤖 Inizializzazione...")
                    progress_bar.progress(0.1)

                    status_text.text("🎬 Processing video...")
                    progress_bar.progress(0.3)

                    pipeline = CVAIPipeline()
                    if not pipeline.initialize():
                        raise Exception("YOLOv8 init failed")

                    cap = cv2.VideoCapture(video_path)
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    results = {
                        'video_info': {'fps':fps, 'frame_count':fc, 'duration':fc/fps if fps>0 else 0},
                        'actions': [],
                        'shots': [],
                        'pose_data': [],
                        'statistics': {'total_poses_detected':0, 'total_actions':0, 'total_shots':0}
                    }

                    fi = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if fi % 5 == 0:
                            fr = pipeline.process_frame(frame)
                            if fr:
                                results['statistics']['total_poses_detected'] += 1
                                act = fr.get('action', 'unknown')
                                if act != 'unknown':
                                    results['actions'].append({
                                        'frame': int(fi),
                                        'action': act,
                                        'timestamp': float(fi/fps if fps>0 else 0)
                                    })
                                    results['statistics']['total_actions'] += 1

                                if act == 'shooting' and 'shooting_form' in fr:
                                    form = fr['shooting_form']
                                    results['shots'].append({
                                        'frame': int(fi),
                                        'elbow_angle': float(form['elbow_angle']),
                                        'knee_angle': float(form['knee_angle']),
                                        'form_score': float(form['form_score']),
                                        'timestamp': float(fi/fps if fps>0 else 0)
                                    })
                                    results['statistics']['total_shots'] += 1

                        fi += 1
                        if fi % 100 == 0:
                            progress_bar.progress(min(0.3 + (fi/fc)*0.7, 1.0))

                    cap.release()

                    with open(output_json, 'w') as f:
                        json.dump(results, f, indent=2)

                    result = results

                    progress_bar.progress(1.0)
                    status_text.text("✅ Completato!")
                    st.balloons()

                    # VISUALIZZA RISULTATI
                    st.markdown("### 📊 Risultati")

                    stats = result.get('statistics', {})
                    c1,c2,c3 = st.columns(3)
                    c1.metric("📸 Pose", stats.get('total_poses_detected',0))
                    c2.metric("🎯 Actions", stats.get('total_actions',0))
                    c3.metric("🏀 Shots", stats.get('total_shots',0))

                    st.markdown("---")

                    if analyze_actions and result.get('actions'):
                        st.markdown("#### 🎯 Actions")
                        actions = result['actions']
                        if len(actions) > 0:
                            adf = pd.DataFrame(actions)
                            st.dataframe(adf, use_container_width=True)

                            if 'action' in adf.columns:
                                ac = adf['action'].value_counts()
                                fig = px.bar(x=ac.index, y=ac.values)
                                st.plotly_chart(fig, use_container_width=True)

                    if analyze_shots and result.get('shots'):
                        st.markdown("#### 🏀 Shots")
                        shots = result['shots']
                        if len(shots) > 0:
                            sdf = pd.DataFrame(shots)
                            st.dataframe(sdf, use_container_width=True)

                    st.markdown("---")

                    with open(output_json, 'r') as f:
                        json_data = f.read()

                    st.download_button(
                        "⬇️ Download JSON",
                        json_data,
                        output_json,
                        "application/json",
                        use_container_width=True
                    )

                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Errore: {str(e)}")

                    with st.expander("🔍 Dettagli"):
                        import traceback
                        st.code(traceback.format_exc())

                finally:
                    if os.path.exists(video_path):
                        try:
                            time.sleep(0.5)
                            os.remove(video_path)
                        except:
                            pass

# =================================================================
# BIOMETRIC MODULE
# =================================================================
def render_biometric_module():
    st.header("⚖️ Biometrics")

    if 'biometric_data' not in st.session_state:
        st.session_state.biometric_data = pd.DataFrame(columns=[
            'player_id','player_name','timestamp','weight_kg','body_fat_pct',
            'muscle_mass_kg','water_pct','bone_mass_kg','bmr_kcal',
            'measurement_type','source','notes'
        ])

    tab1, tab2 = st.tabs(["📊 Dashboard", "➕ Input"])

    with tab1:
        st.subheader("📊 Dashboard")
        if st.session_state.biometric_data.empty:
            st.info("Nessun dato - Vai tab Input")
        else:
            latest = st.session_state.biometric_data.sort_values('timestamp').groupby('player_id').last()
            st.metric("Giocatori", len(latest))
            st.dataframe(latest[['player_name','weight_kg','body_fat_pct']], use_container_width=True)

    with tab2:
        st.subheader("➕ Input Manuale")
        with st.form("bio_form"):
            name = st.text_input("Nome")
            weight = st.number_input("Peso (kg)", 40.0, 150.0, 75.0)
            submitted = st.form_submit_button("💾 Salva")

            if submitted and name:
                import hashlib
                pid = hashlib.md5(name.encode()).hexdigest()[:8]
                new_row = pd.DataFrame([{
                    'player_id': pid,
                    'player_name': name,
                    'timestamp': datetime.now(),
                    'weight_kg': weight,
                    'body_fat_pct': None,
                    'muscle_mass_kg': None,
                    'water_pct': None,
                    'bone_mass_kg': None,
                    'bmr_kcal': None,
                    'measurement_type': 'manual',
                    'source': 'manual',
                    'notes': ''
                }])
                st.session_state.biometric_data = pd.concat([
                    st.session_state.biometric_data, new_row
                ], ignore_index=True)
                st.success(f"✅ Salvato: {name}")
                st.rerun()

# =================================================================
# MAIN APP
# =================================================================
st.set_page_config(page_title="CoachTrack Elite", page_icon="🏀", layout="wide")

# Session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'tracking_data' not in st.session_state:
    st.session_state.tracking_data = {}

# Login
if not st.session_state.logged_in:
    st.title("🏀 CoachTrack Elite")
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        u = st.text_input("Username", value="admin")
        p = st.text_input("Password", type="password", value="admin")
        if st.button("Login", type="primary", use_container_width=True):
            if u == "admin" and p == "admin":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Credenziali errate")
    st.stop()

# Sidebar
with st.sidebar:
    st.title("🏀 CoachTrack")
    st.markdown("---")
    st.caption("v3.2 - YOLOv8")
    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

# Main
st.title("🏀 CoachTrack Elite AI")
st.markdown("Complete AI + ML + CV + Biometrics")

tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "🎥 CV", "⚖️ Biometrics"])

with tab1:
    st.header("📊 Dashboard")
    st.info("Welcome to CoachTrack Elite v3.2")
    col1,col2,col3 = st.columns(3)
    col1.metric("Players", len(st.session_state.tracking_data))
    col2.metric("CV Status", "✅" if CV_AVAILABLE else "❌")
    col3.metric("AI Status", "✅" if AI_ADVANCED_AVAILABLE else "❌")

with tab2:
    add_computer_vision_tab()

with tab3:
    render_biometric_module()
