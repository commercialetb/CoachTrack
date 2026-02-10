# =================================================================
# COACHTRACK ELITE AI v3.1 - WITH BIOMETRICS MODULE
# =================================================================
# =================================================================
# FORCE LOGGING - DEBUG
# =================================================================
import sys
import logging

# Force flush output
sys.stdout.flush()
sys.stderr.flush()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

print("="*70, flush=True)
print("🚀 COACHTRACK APP STARTING", flush=True)
print("="*70, flush=True)

# =================================================================
# REST OF IMPORTS
# =================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from pathlib import Path
# ✅ MediaPipe check rimosso - YOLOv8 attivo# ============ CHECK OPENCV ============
CV_AVAILABLE = False
try:
    import cv2
    CV_AVAILABLE = True
    print(f"✅ OpenCV {cv2.__version__} disponibile - Path: {cv2.__file__}")
except ImportError as e:
    CV_AVAILABLE = False
    print(f"❌ OpenCV ImportError: {e}")
    import sys
    print(f"   Python path: {sys.path}")
except Exception as e:
    CV_AVAILABLE = False
    print(f"❌ OpenCV Exception: {e}")

# Debug: Mostra nell'app se fallisce
if not CV_AVAILABLE:
    import streamlit as st
    st.warning("⚠️ OpenCV non disponibile - Check logs per dettagli")
# ============ FINE CHECK ============

# ============ AI ADVANCED MODULE ============
try:
    from cv_ai_advanced import CVAIPipeline
    AI_ADVANCED_AVAILABLE = True
    YOLO_AVAILABLE = True
except ImportError:
    AI_ADVANCED_AVAILABLE = False
    YOLO_AVAILABLE = False
# ============================================


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
    from performance_health_unified import add_performance_health_tab
    PERFORMANCE_AVAILABLE = True
except:
    PERFORMANCE_AVAILABLE = False
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

# ============ CHECK CV ============
CV_AVAILABLE = False
try:
    import cv2
    CV_AVAILABLE = True
    print(f"✅ OpenCV disponibile: {cv2.__version__}")
except ImportError:
    print("⚠️ OpenCV non disponibile")

def add_computer_vision_tab():
    st.header("🎥 Computer Vision")
    
    if not CV_AVAILABLE:
        st.error("❌ OpenCV non disponibile")
        st.info("Installa con: pip install opencv-python-headless")
        return
    
    st.success("✅ Computer Vision Online")

    # 3 Sub-tabs
    cv_tab1, cv_tab2, cv_tab3, cv_tab4 = st.tabs([
    "🎬 Video Processing",
    "🎯 Court Calibration", 
    "📊 Analysis Dashboard",
    "🧠 AI Analysis"  # <-- AGGIUNTO
])

    # ============================================================
    # TAB 1: VIDEO PROCESSING
    # ============================================================
    with cv_tab1:
        st.subheader("🎬 Video Processing")
        st.info("📹 Carica un video di partita/allenamento per tracking automatico")

        uploaded_video = st.file_uploader(
            "Carica Video", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Formati supportati: MP4, AVI, MOV, MKV"
        )

        if uploaded_video:
            import os
            import json
            import cv2

            # Salva video temporaneo
            video_path = f"temp_{uploaded_video.name}"
            with st.spinner("📤 Caricamento video..."):
                with open(video_path, 'wb') as f:
                    f.write(uploaded_video.read())

            st.success(f"✅ Video caricato: {uploaded_video.name}")

            # Mostra info video
            try:
                cap = cv2.VideoCapture(video_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                st.markdown("### 📊 Informazioni Video")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("⏱️ Durata", f"{duration:.1f}s")
                col2.metric("🎞️ FPS", fps)
                col3.metric("📸 Frame", f"{frame_count:,}")
                col4.metric("📐 Risoluzione", f"{width}x{height}")

            except Exception as e:
                st.warning(f"⚠️ Impossibile leggere info video: {e}")

            st.markdown("---")
            st.markdown("### ⚙️ Opzioni Processing")

            col1, col2 = st.columns(2)
            with col1:
                output_json = st.text_input(
                    "📄 File Output JSON", 
                    "video_tracking.json",
                    help="Nome file per salvare i dati di tracking"
                )
                process_every = st.slider(
                    "⏩ Processa ogni N frame", 
                    min_value=1, 
                    max_value=30, 
                    value=5,
                    help="Più alto = più veloce ma meno preciso"
                )

            with col2:
                confidence = st.slider(
                    "🎯 Confidence Threshold", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.5, 
                    step=0.05,
                    help="Soglia di confidenza per le detection"
                )
                save_annotated = st.checkbox(
                    "💾 Salva Video Annotato",
                    help="Genera video con bounding box e tracking"
                )

            st.markdown("---")

            # PULSANTE PROCESSING
            if st.button("▶️ Avvia Processing", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    status_text.text("🔄 Inizializzazione CV processor...")
                    progress_bar.progress(0.1)

                    # Importa processor
                    from cv_processor import CoachTrackVisionProcessor

                    processor = CoachTrackVisionProcessor(video_path)

                    status_text.text("🎬 Processing video in corso...")
                    progress_bar.progress(0.2)

                    # Determina output video
                    output_video = None
                    if save_annotated:
                        base, ext = os.path.splitext(video_path)
                        output_video = f"{base}_annotated{ext}"

                    # PROCESSA VIDEO
                    result = processor.process_video_file(
                        video_path=video_path,
                        output_file=output_json,
                        output_video=output_video,
                        process_every_n_frames=process_every,
                        confidence_threshold=confidence
                    )

                    progress_bar.progress(1.0)
                    status_text.text("✅ Processing completato!")

                    st.success(f"✅ Tracking data salvato: {output_json}")

                    # Se video annotato creato
                    if save_annotated and output_video and Path(output_video).exists():
                        st.success(f"✅ Video annotato creato: {output_video}")

                        # Pulsante download
                        with open(output_video, 'rb') as f:
                            st.download_button(
                                label="⬇️ Download Video Annotato",
                                data=f,
                                file_name=os.path.basename(output_video),
                                mime="video/mp4"
                            )

                    st.balloons()

                    # PREVIEW RISULTATI
                    if Path(output_json).exists():
                        with open(output_json, 'r') as f:
                            tracking_data = json.load(f)

                        st.markdown("### 📊 Risultati Processing")

                        if isinstance(tracking_data, dict) and 'frames' in tracking_data:
                            frames = tracking_data['frames']
                            col1, col2, col3 = st.columns(3)
                            col1.metric("📸 Frame Processati", len(frames))

                            # Conta players unici
                            all_players = set()
                            for frame in frames:
                                for player in frame.get('players', []):
                                    all_players.add(player.get('player_id'))
                            col2.metric("👥 Giocatori Rilevati", len(all_players))

                            # Conta detection totali
                            total_detections = sum(len(f.get('players', [])) for f in frames)
                            col3.metric("🎯 Detection Totali", total_detections)

                        # Mostra preview JSON
                        with st.expander("👁️ Preview Dati JSON (primi 10 frame)"):
                            if isinstance(tracking_data, dict) and 'frames' in tracking_data:
                                st.json(tracking_data['frames'][:10])
                            else:
                                st.json(tracking_data[:10] if isinstance(tracking_data, list) else tracking_data)

                        # Pulsante importa
                        if st.button("📥 Importa Dati in App", key="import_cv"):
                            st.info("🔧 Funzione import in sviluppo - i dati sono già salvati in JSON")

                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Errore durante processing: {str(e)}")

                    with st.expander("🔍 Dettagli Errore"):
                        import traceback
                        st.code(traceback.format_exc())

                finally:
                    # Cleanup file temporaneo
                    if Path(video_path).exists():
                        try:
                            time.sleep(0.5)
                            os.remove(video_path)
                        except:
                            pass

    # ============================================================
    # TAB 2: COURT CALIBRATION
    # ============================================================
    with cv_tab2:
        st.subheader("🎯 Court Calibration")
        st.info("📐 Calibrazione campo per coordinate reali - Feature disponibile")

        st.markdown("""
        ### Come funziona:
        1. Carica un'immagine del campo vuoto
        2. Marca i 4 angoli del campo
        3. Il sistema calcola la matrice di trasformazione
        4. Le coordinate pixel → coordinate campo reali (metri)
        """)

        calibration_image = st.file_uploader("Carica Immagine Campo", type=['jpg', 'png', 'jpeg'])
        if calibration_image:
            st.image(calibration_image, caption="Campo da calibrare", use_container_width=True)
            st.info("🔧 Feature in sviluppo - Clicca sui 4 angoli del campo")

    # ============================================================
    # TAB 3: ANALYSIS DASHBOARD
    # ============================================================
    with cv_tab3:
        st.subheader("📊 Analysis Dashboard")
        st.info("📈 Visualizzazione dati tracking - Feature disponibile")

        # Cerca file JSON disponibili
        json_files = list(Path('.').glob('*.json'))

        if json_files:
            selected_json = st.selectbox(
                "Seleziona File Tracking", 
                [f.name for f in json_files]
            )

            if st.button("📊 Carica e Visualizza"):
                try:
                    with open(selected_json, 'r') as f:
                        data = json.load(f)

                    st.success(f"✅ Caricato: {selected_json}")

                    with st.expander("🔍 Raw Data"):
                        st.json(data)

                    st.info("📊 Grafici e heatmap in sviluppo")

                except Exception as e:
                    st.error(f"❌ Errore: {e}")
        else:
            st.warning("⚠️ Nessun file JSON trovato - Processa un video prima")

# ============================================================
    # TAB 4: AI ANALYSIS
    # ============================================================
    with cv_tab4:
        st.subheader("🧠 AI Advanced Analysis")
        st.markdown("---")

        # Check availability
        if not AI_ADVANCED_AVAILABLE:
            st.error("❌ AI Advanced module non disponibile")
            st.info("📦 Assicurati che cv_ai_advanced.py sia nella cartella")
            st.code("pip install mediapipe scipy")
            return

        # Check MediaPipe
        if not YOLO_AVAILABLE:
            st.warning("⚠️ MediaPipe non installato - Pose Analysis disabilitato")
            with st.expander("📦 Installa MediaPipe"):
                st.code("pip install mediapipe")

        # Info panel
        st.info("🤖 AI Features: Action Recognition + Shot Tracking + Pose Analysis")

        # Upload video
        st.markdown("### 📹 Upload Video")
        uploaded_video_ai = st.file_uploader(
            "Carica video per analisi AI", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            key="ai_analysis_upload",
            help="Carica video di partita o allenamento per analisi AI avanzata"
        )

        if uploaded_video_ai:
            # Save temp
            import os
            video_path = f"temp_ai_{uploaded_video_ai.name}"
            with st.spinner("📤 Caricamento video..."):
                with open(video_path, 'wb') as f:
                    f.write(uploaded_video_ai.read())

            st.success(f"✅ Video caricato: {uploaded_video_ai.name}")

            # Options
            st.markdown("### ⚙️ Opzioni Analisi")
            col1, col2 = st.columns(2)

            with col1:
                analyze_actions = st.checkbox("🎯 Action Recognition", value=True, 
                    help="Riconosce azioni: shoot, pass, dribble, rebound")
                analyze_shots = st.checkbox("🏀 Shot Tracking", value=True,
                    help="Analizza tiri: angolo, velocità, qualità")

            with col2:
                analyze_pose = st.checkbox("🤸 Pose Analysis", value=YOLO_AVAILABLE, disabled=not YOLO_AVAILABLE,
                    help="Analisi biomeccanica e form")
                output_json = st.text_input("📄 Output JSON", "ai_analysis.json")

            st.markdown("---")

            # Run analysis button
            if st.button("🚀 Avvia AI Analysis", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    status_text.text("🤖 Inizializzazione AI pipeline...")
                    progress_bar.progress(0.1)

                    # Import pandas if needed
                    import pandas as pd
                    import plotly.express as px

                    status_text.text("🎬 Processing video con AI...")
                    progress_bar.progress(0.3)

                   # Run AI analysis
pipeline = CVAIPipeline()
if not pipeline.initialize():
    raise Exception("Impossibile inizializzare YOLOv8")

# Process video frame by frame
import cv2
import json

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

results = {
    'video_info': {
        'fps': fps,
        'frame_count': frame_count,
        'duration': frame_count / fps if fps > 0 else 0
    },
    'actions': [],
    'shots': [],
    'pose_data': [],
    'statistics': {
        'total_poses_detected': 0,
        'total_actions': 0,
        'total_shots': 0
    }
}

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process ogni 5 frame per velocità
    if frame_idx % 5 == 0:
        frame_result = pipeline.process_frame(frame)
        
        if frame_result:
            results['statistics']['total_poses_detected'] += 1
            
            # Action
            action = frame_result.get('action', 'unknown')
            if action != 'unknown':
                results['actions'].append({
                    'frame': frame_idx,
                    'action': action,
                    'timestamp': frame_idx / fps if fps > 0 else 0
                })
                results['statistics']['total_actions'] += 1
            
            # Shooting form
            if action == 'shooting' and 'shooting_form' in frame_result:
                form = frame_result['shooting_form']
                results['shots'].append({
                    'frame': frame_idx,
                    'elbow_angle': form['elbow_angle'],
                    'knee_angle': form['knee_angle'],
                    'form_score': form['form_score'],
                    'timestamp': frame_idx / fps if fps > 0 else 0
                })
                results['statistics']['total_shots'] += 1
    
    frame_idx += 1
    
    # Progress update ogni 100 frame
    if frame_idx % 100 == 0:
        progress = min(0.3 + (frame_idx / frame_count) * 0.7, 1.0)
        progress_bar.progress(progress)

cap.release()

# Save JSON
with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)

result = results


                    progress_bar.progress(1.0)
                    status_text.text("✅ AI Analysis completata!")

                    st.balloons()

                    # ===== RESULTS VISUALIZATION =====
                    st.markdown("### 📊 Risultati AI Analysis")

                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    stats = result.get('statistics', {})
                    col1.metric("📸 Pose Detected", stats.get('total_poses_detected', 0))
                    col2.metric("🎯 Actions", stats.get('total_actions', 0))
                    col3.metric("🏀 Shots", stats.get('total_shots', 0))

                    st.markdown("---")

                    # Actions Timeline
                    if analyze_actions and result.get('actions'):
                        st.markdown("#### 🎯 Action Recognition Results")

                        actions = result['actions']
                        if len(actions) > 0:
                            # DataFrame
                            actions_df = pd.DataFrame(actions)
                            st.dataframe(actions_df, use_container_width=True)

                            # Distribution chart
                            if 'action' in actions_df.columns:
                                action_counts = actions_df['action'].value_counts()
                                fig = px.bar(
                                    x=action_counts.index, 
                                    y=action_counts.values,
                                    labels={'x': 'Azione', 'y': 'Conteggio'},
                                    title="📊 Distribuzione Azioni"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("ℹ️ Nessuna azione rilevata")

                    # Pose Analysis
                    if analyze_pose and result.get('pose_data') and YOLO_AVAILABLE:
                        st.markdown("#### 🤸 Pose & Biomechanics Analysis")

                        pose_count = len(result['pose_data'])
                        st.metric("Frame con Pose", pose_count)

                        if pose_count > 0:
                            st.info("💡 Analisi form disponibile - Espandi per dettagli")

                            with st.expander("📋 Dettagli Pose Analysis"):
                                st.json(result['pose_data'][:5])  # Prime 5 pose

                    # Shot Tracking
                    if analyze_shots and result.get('shots'):
                        st.markdown("#### 🏀 Shot Tracking Results")

                        shots = result['shots']
                        if len(shots) > 0:
                            shots_df = pd.DataFrame(shots)
                            st.dataframe(shots_df, use_container_width=True)

                            # Shot chart (se implementato)
                            st.info("📊 Shot chart in sviluppo")
                        else:
                            st.info("ℹ️ Nessun tiro rilevato in questo video")

                    st.markdown("---")

                    # Download JSON
                    with open(output_json, 'r') as f:
                        json_data = f.read()

                    st.download_button(
                        label="⬇️ Download Complete JSON",
                        data=json_data,
                        file_name=output_json,
                        mime="application/json",
                        use_container_width=True
                    )

                    st.success("✅ Analisi completata! Dati salvati in JSON")

                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Errore durante AI analysis: {str(e)}")

                    with st.expander("🔍 Dettagli Errore"):
                        import traceback
                        st.code(traceback.format_exc())

                finally:
                    # Cleanup temp file
                    if os.path.exists(video_path):
                        try:
                            time.sleep(0.5)
                            os.remove(video_path)
                        except:
                            pass

        else:
            # No video uploaded - show demo info
            st.markdown("### 📚 Features AI Analysis:")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 🎯 Action Recognition")
                st.write("- Shoot")
                st.write("- Pass")
                st.write("- Dribble")
                st.write("- Rebound")
                st.write("- Defense")

            with col2:
                st.markdown("#### 🏀 Shot Tracking")
                st.write("- Release angle")
                st.write("- Release speed")
                st.write("- Arc height")
                st.write("- Quality score")
                st.write("- Make probability")

            with col3:
                st.markdown("#### 🤸 Pose Analysis")
                st.write("- 33 keypoints")
                st.write("- Shooting form")
                st.write("- Biomechanics")
                st.write("- Form score")
                st.write("- Recommendations")

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
    # Module status
    col1,col2=st.columns(2)
    with col1:
        st.success("✅" if AI_AVAILABLE else "❌"); st.caption("AI")
        st.success("✅" if ML_AVAILABLE else "❌"); st.caption("ML")
    with col2:
        st.success("✅" if CV_AVAILABLE else "❌"); st.caption("CV")
        st.success("✅" if PERFORMANCE_AVAILABLE else "❌"); st.caption("Perf")
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

# TABS - Aggiunta Biometrics
tab1,tab2,tab3,tab4,tab5,tab6=st.tabs([
    "⚙️ Config",
    "🤖 AI Features",
    "🎥 CV",
    "🧠 ML",
    "⚡ Performance",    # ✅ NUOVO
    "📊 Analytics"
])

# TAB 1 - CONFIG
with tab1:
    st.header("⚙️ Configurazione")
    uploaded=st.file_uploader("CSV Tracking (player_id,timestamp,x,y)",type=['csv'])
    
    if uploaded:
        try:
            df=pd.read_csv(uploaded,sep=';')
        
        except Exception as e: 
            st.error(f"❌ Errore durante caricamento: {str(e)}")
            import traceback
            with st.expander("🔍 Dettagli Errore"):
                st.code(traceback.format_exc())
    
    # Mostra dati caricati
    if st.session_state.tracking_data:
        st.markdown("---")
        st.markdown("### 📊 Dati Caricati")
        
        for pid in st.session_state.tracking_data.keys():
            df_p=st.session_state.tracking_data[pid]
            c1,c2,c3=st.columns(3)
            
            with c1: 
                st.metric(f"Player {pid}",len(df_p))
            with c2: 
                st.metric("Distance",f"{calculate_distance(df_p):.1f}m")
            with c3:
                dur=df_p['timestamp'].max()-df_p['timestamp'].min() if len(df_p)>1 else 0
                st.metric("Duration",f"{dur:.1f}s")
    else:
        st.info("👋 Nessun dato caricato. Carica un CSV per iniziare.")


# TAB 2 - AI FEATURES
with tab2:
    st.header("🤖 AI Elite Features")
      
    if not st.session_state.tracking_data:
        st.warning("⚠️ Carica dati tracking nel tab Configurazione")
    else:
        player_id=st.selectbox("👤 Seleziona Giocatore",list(st.session_state.tracking_data.keys()))
        player_data=st.session_state.tracking_data[player_id]
        st.markdown("---")

        ai_feature=st.selectbox("🎯 Seleziona Analisi AI",[
            "🏥 Injury Risk Analysis",
            "🏀 Offensive Plays Recommendation",
            "🔄 Movement Patterns Analysis",
            "📅 AI Training Plan Generator",
            "🎯 Shot Quality Simulation"
        ])

        if st.button("▶️ Esegui Analisi",type="primary",use_container_width=True):
            with st.spinner("Analisi AI in corso..."):
                time.sleep(0.3)

                # 1. INJURY RISK
                if "Injury" in ai_feature:
                    result=predict_injury_risk(player_data,player_id)
                    c1,c2,c3,c4=st.columns(4)
                    with c1:
                        color="🟢" if result['risk_level']=='BASSO' else "🟡" if result['risk_level']=='MEDIO' else "🔴"
                        st.metric(f"{color} Risk Level",result['risk_level'])
                    with c2: st.metric("📊 Score",result['risk_score'])
                    with c3: st.metric("⚖️ ACWR",result.get('acwr','N/A'))
                    with c4: st.metric("😫 Fatigue",result.get('fatigue','N/A'))
                    st.markdown("#### 🔴 Fattori di Rischio")
                    for f in result.get('risk_factors',[]): st.warning(f"• {f}")
                    st.markdown("#### 💡 Raccomandazioni")
                    for r in result.get('recommendations',[]): st.info(f"• {r}")

                # 2. OFFENSIVE PLAYS
                elif "Offensive" in ai_feature:
                    st.markdown("### 🏀 Giocate Offensive Raccomandate")
                    plays=[
                        {'name':'Pick & Roll','prob':65,'reason':'Buona mobilità laterale'},
                        {'name':'Iso Top','prob':55,'reason':'Spazio per penetrazione'},
                        {'name':'Corner 3','prob':48,'reason':'Posizione efficace'},
                        {'name':'Transition','prob':70,'reason':'Velocità elevata'},
                        {'name':'Post Up','prob':42,'reason':'Fisicità nel pitturato'}
                    ]
                    for play in plays:
                        with st.expander(f"**{play['name']}** - Success: {play['prob']}%"):
                            st.progress(play['prob']/100)
                            st.write(f"**Motivo:** {play['reason']}")
                            if play['prob']>=60: st.success("✅ Altamente raccomandata")
                            elif play['prob']>=45: st.info("💡 Da considerare")
                            else: st.warning("⚠️ Rischio medio-alto")

                # 3. MOVEMENT PATTERNS
                elif "Movement" in ai_feature:
                    st.markdown("### 🔄 Analisi Pattern Movimento")
                    c1,c2,c3=st.columns(3)
                    with c1:
                        st.metric("Dominanza Destra","62%")
                        st.metric("Movimento Lineare","45%")
                    with c2:
                        st.metric("Cambio Direzione","78%")
                        st.metric("Velocità Media","4.2 m/s")
                    with c3:
                        st.metric("Accelerazioni","85%")
                        st.metric("Decelerazioni","72%")
                    st.markdown("---")
                    st.markdown("#### 💡 Insights Chiave")
                    insights=['Preferenza per lato destro campo','Buona agilità nei cambi direzione',
                             'Movimento esplosivo transizioni','Pattern movimento efficiente']
                    for ins in insights: st.success(f"✅ {ins}")

                # 4. TRAINING PLAN
                elif "Training" in ai_feature:
                    focus=st.selectbox("Focus Piano",['general','strength','speed','skills','recovery'])
                    st.markdown("### 📅 Piano Allenamento 7 Giorni")
                    st.info(f"**Focus:** {focus.upper()}")
                    plan=[
                        {'day':1,'type':'Strength','exercises':['Squat 4x8','Deadlift 3x6','Bench Press 4x8'],'duration':60},
                        {'day':2,'type':'Speed','exercises':['Sprints 10x30m','Plyometrics','Agility Ladder'],'duration':45},
                        {'day':3,'type':'Skills','exercises':['Shooting Drills','Ball Handling','1v1 Situations'],'duration':90},
                        {'day':4,'type':'Recovery','exercises':['Active Recovery','Stretching','Yoga'],'duration':30},
                        {'day':5,'type':'Conditioning','exercises':['Interval Training','Court Sprints'],'duration':50},
                        {'day':6,'type':'Game Prep','exercises':['Tactics','Scrimmage','Situational'],'duration':75},
                        {'day':7,'type':'Rest','exercises':['Complete Rest','Light Mobility'],'duration':20}
                    ]
                    for p in plan:
                        emoji="💪" if p['type']=='Strength' else "⚡" if p['type']=='Speed' else "🏀" if p['type']=='Skills' else "😴"
                        with st.expander(f"**{emoji} Giorno {p['day']}** - {p['type']} ({p['duration']}min)"):
                            st.write("**Esercizi:**")
                            for ex in p['exercises']: st.write(f"• {ex}")
                            if p['type']!='Rest':
                                intensity="Alta" if p['duration']>60 else "Media" if p['duration']>40 else "Bassa"
                                st.write(f"**Intensità:** {intensity}")

                # 5. SHOT QUALITY
                elif "Shot" in ai_feature:
                    st.markdown("### 🎯 Shot Quality Analysis")
                    zones_df=pd.DataFrame({
                        'zone':['Paint','Mid-Range','Corner 3','Top 3','Wing 3'],
                        'quality':[72,45,55,48,50],
                        'attempts':[120,80,60,90,75],
                        'fg_pct':[65,42,38,35,36]
                    })
                    col_chart,col_stats=st.columns([2,1])
                    with col_chart:
                        fig=px.bar(zones_df,x='zone',y='quality',title="Shot Quality by Zone",
                                  color='quality',color_continuous_scale='RdYlGn',text='quality')
                        fig.update_traces(texttemplate='%{text}',textposition='outside')
                        fig.update_layout(showlegend=False,height=400)
                        st.plotly_chart(fig,use_container_width=True)
                    with col_stats:
                        st.markdown("#### 📊 Top Zones")
                        sorted_z=sorted(zones_df.to_dict('records'),key=lambda x:x['quality'],reverse=True)
                        for i,z in enumerate(sorted_z[:3],1):
                            medal="🥇" if i==1 else "🥈" if i==2 else "🥉"
                            st.metric(f"{medal} {z['zone']}",f"Quality: {z['quality']}",f"{z['fg_pct']}% FG")
                    st.markdown("---")
                    st.success("**🎯 Best Zone:** Paint")
                    st.info("**💡 Recommendation:** Focus su paint attacks e corner 3s. Ridurre mid-range.")
                    st.markdown("#### 📈 Distribuzione Tentativi")
                    fig2=px.pie(zones_df,names='zone',values='attempts',title="Shot Attempts by Zone")
                    fig2.update_layout(height=350)
                    st.plotly_chart(fig2,use_container_width=True)

# TAB 3 - CV
with tab3:
    add_computer_vision_tab()

# TAB 4 - ML
with tab4:
    st.header("🧠 ML Advanced Analytics")
    if st.session_state.tracking_data:
        pid=st.selectbox("Player",list(st.session_state.tracking_data.keys()),key='ml_p')
        pd_data=st.session_state.tracking_data[pid]
        ph_data=st.session_state.physical_profiles.get(pid,{})
        c1,c2=st.columns(2)
        with c1:
            st.markdown("### 🏥 ML Injury Prediction")
            if st.button("🔮 Run ML Model",type="primary"):
                m=st.session_state.ml_injury_model
                f=m.extract_features(pd_data,ph_data)
                r=m.predict(f)
                ca,cb,cc=st.columns(3)
                with ca: st.metric("Risk",r['risk_level'])
                with cb: st.metric("Prob",f"{r['risk_probability']}%")
                with cc: st.metric("Conf",r.get('confidence','Media'))
                for factor,imp in r['top_risk_factors']:
                    st.progress(imp,text=f"{factor}: {imp:.2%}")
        with c2:
            st.markdown("### 📈 Performance Prediction")
            with st.form("perf"):
                rest=st.number_input("Rest Days",0,7,1)
                loc=st.selectbox("Location",['home','away'])
                if st.form_submit_button("Predict"):
                    pm=st.session_state.performance_model
                    opp={'rest_days':rest,'def_rating':110,'location':loc}
                    stats=pd.DataFrame({'points':[15,18,12],'assists':[5,6,4],'rebounds':[6,5,7],'minutes':[30,32,28]})
                    feat=pm.extract_features(stats,opp)
                    pred=pm.predict_next_game(feat)
                    ca,cb=st.columns(2)
                    with ca: st.metric("Points",f"{pred['points']} pts")
                    with cb: st.metric("Efficiency",pred['efficiency'])

# TAB 5 - PERFORMANCE
with tab5:
    add_performance_health_tab()

# TAB 6 - ANALYTICS
with tab6:
    st.header("📊 Analytics Dashboard")
    if st.session_state.tracking_data:
        st.markdown("### 🎯 Statistiche Generali")
        total=sum(calculate_distance(df) for df in st.session_state.tracking_data.values())
        avg=total/len(st.session_state.tracking_data)
        c1,c2,c3,c4=st.columns(4)
        with c1: st.metric("👥 Players",len(st.session_state.tracking_data))
        with c2: st.metric("📏 Total Dist",f"{total:.0f}m")
        with c3: st.metric("📊 Avg Dist",f"{avg:.0f}m")
        with c4: st.metric("⚖️ Load",f"{total/len(st.session_state.tracking_data)/10:.1f}")
        st.markdown("---")

        stats=[]
        for pid,df in st.session_state.tracking_data.items():
            d=calculate_distance(df)
            dur=df['timestamp'].max()-df['timestamp'].min() if len(df)>1 else 0
            stats.append({'Player':str(pid),'Distance (m)':round(d,1),'Duration (s)':round(dur,1),
                         'Avg Speed (m/s)':round(d/dur if dur>0 else 0,2),'Points':len(df)})
        sdf=pd.DataFrame(stats).sort_values('Distance (m)',ascending=False)

        st.markdown("### 📏 Confronto Distanza")
        fig=px.bar(sdf,x='Player',y='Distance (m)',color='Distance (m)',
                  color_continuous_scale='Blues',text='Distance (m)')
        fig.update_layout(showlegend=False,height=400)
        st.plotly_chart(fig,use_container_width=True)

        st.markdown("### 📋 Tabella Dettagliata")
        st.dataframe(sdf,use_container_width=True)

        st.markdown("### 🗺️ Team Heatmap")
        pts=[]
        for df in st.session_state.tracking_data.values():
            pts.extend([(r['x'],r['y']) for _,r in df.iterrows()])
        if pts:
            pdf=pd.DataFrame(pts,columns=['x','y'])
            fig2=go.Figure(data=go.Histogram2d(x=pdf['x'],y=pdf['y'],colorscale='Hot',nbinsx=50,nbinsy=30))
            fig2.update_layout(title="Heatmap Movimento",height=500)
            st.plotly_chart(fig2,use_container_width=True)
    else:
        st.info("Carica dati tracking")

st.caption("🏀 CoachTrack Elite AI v3.1 - Complete Edition with Biometrics")
