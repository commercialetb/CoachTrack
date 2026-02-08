# =================================================================
# COACHTRACK ELITE AI v3.1 - WITH CV VIDEO PROCESSING
# =================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from pathlib import Path
import json
import os

# CV imports
CV_AVAILABLE = False
CV_ERROR = None
try:
    import cv2
    from cv_processor import CoachTrackVisionProcessor
    CV_AVAILABLE = True
except Exception as e:
    CV_ERROR = str(e)

# =================================================================
# CV TAB COMPLETO
# =================================================================

def add_computer_vision_tab():
    '''TAB CV con Video Processing completo'''
    st.header("🎥 Computer Vision System")

    if not CV_AVAILABLE:
        st.error("❌ CV non disponibile")
        if CV_ERROR:
            with st.expander("Errore"):
                st.code(CV_ERROR)

        # Check packages
        missing = []
        try: import cv2
        except: missing.append("opencv-python-headless")
        try: from ultralytics import YOLO
        except: missing.append("ultralytics")

        if missing:
            st.code(f"pip install {' '.join(missing)}")
        return

    # 4 Sub-tabs
    cv_tab1, cv_tab2, cv_tab3, cv_tab4 = st.tabs([
        "📹 Live Camera",
        "🎬 Video Processing", 
        "🎯 Calibration",
        "📊 Analysis"
    ])

    # VIDEO PROCESSING TAB
    with cv_tab2:
        st.subheader("🎬 Video Processing")
        st.info("Carica un video per estrarre tracking automatico")

        uploaded_video = st.file_uploader(
            "Carica Video",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="MP4, AVI, MOV, MKV"
        )

        if uploaded_video:
            video_path = f"temp_{uploaded_video.name}"
            with st.spinner("Caricamento..."):
                with open(video_path, 'wb') as f:
                    f.write(uploaded_video.read())

            st.success(f"✅ {uploaded_video.name}")

            # Video info
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frames / fps if fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Durata", f"{duration:.1f}s")
                c2.metric("FPS", fps)
                c3.metric("Frame", f"{frames:,}")
                c4.metric("Ris", f"{width}x{height}")
            except: pass

            st.markdown("---")
            st.markdown("### Opzioni")

            col1, col2 = st.columns(2)
            with col1:
                output_json = st.text_input("Output JSON", "video_tracking.json")
                process_n = st.slider("Processa ogni N frame", 1, 30, 5)
            with col2:
                confidence = st.slider("Confidence", 0.0, 1.0, 0.5, 0.05)
                save_video = st.checkbox("Salva Video Annotato")

            st.markdown("---")

            if st.button("🎬 Avvia Processing", type="primary", use_container_width=True):
                progress = st.progress(0)
                status = st.empty()

                try:
                    status.text("Inizializzazione...")
                    progress.progress(0.1)

                    processor = CoachTrackVisionProcessor(video_path)

                    status.text("Processing video...")
                    progress.progress(0.2)

                    output_video = video_path.replace('.', '_ann.') if save_video else None

                    result = processor.process_video_file(
                        video_path=video_path,
                        output_file=output_json,
                        output_video=output_video,
                        process_every_n_frames=process_n,
                        confidence_threshold=confidence
                    )

                    progress.progress(1.0)
                    status.text("✅ Completato!")

                    st.success(f"✅ Tracking: {output_json}")

                    if save_video and output_video and Path(output_video).exists():
                        st.success(f"✅ Video: {output_video}")
                        with open(output_video, 'rb') as f:
                            st.download_button("Download Video", f, os.path.basename(output_video), "video/mp4")

                    st.balloons()

                    # Preview
                    if Path(output_json).exists():
                        with open(output_json) as f:
                            data = json.load(f)

                        st.markdown("### Risultati")

                        if isinstance(data, list):
                            c1,c2,c3 = st.columns(3)
                            c1.metric("Frame", len(data))
                            c2.metric("Players", len(set(d.get('player_id','') for d in data)))
                            c3.metric("Detections", sum(1 for d in data if d.get('detections')))

                        with st.expander("Preview Dati"):
                            st.json(data[:10] if isinstance(data, list) else data)

                        if st.button("📥 Importa in App"):
                            st.info("Funzione import in sviluppo...")

                except Exception as e:
                    progress.empty()
                    status.empty()
                    st.error(f"❌ Errore: {str(e)}")
                    with st.expander("Dettagli"):
                        import traceback
                        st.code(traceback.format_exc())

                finally:
                    if Path(video_path).exists():
                        try: os.remove(video_path)
                        except: pass

    # Altri tabs semplificati
    with cv_tab1:
        st.info("Live Camera Tracking - Feature disponibile con camera connessa")

    with cv_tab3:
        st.info("Court Calibration - Feature per calibrazione campo")

    with cv_tab4:
        st.info("Analysis - Visualizzazione dati CV")

# Resto codice app.py (mantengo versione semplificata)
print("✅ File creato!")
