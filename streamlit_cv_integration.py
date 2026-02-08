# =================================================================
# COACHTRACK ELITE AI v3.0 - STREAMLIT CV INTEGRATION
# Integrazione modulo Computer Vision nell'app principale
# =================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
from pathlib import Path

# Import moduli CV
try:
    from cv_processor import CoachTrackVisionProcessor
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    st.warning("⚠️ Moduli Computer Vision non trovati. Installa dipendenze.")


def add_computer_vision_tab():
    """Aggiunge tab Computer Vision a Streamlit app"""

    st.header("🎥 Computer Vision System")

    if not CV_AVAILABLE:
        st.error("❌ Moduli CV non disponibili. Verifica installazione.")
        st.code("pip install opencv-python ultralytics numpy", language="bash")
        return

    # Tabs CV
    cv_tab1, cv_tab2, cv_tab3, cv_tab4 = st.tabs([
        "📹 Live Tracking",
        "📁 Process Video", 
        "🎯 Calibrazione",
        "📊 Analysis"
    ])

    # ============ TAB 1: LIVE TRACKING ============
    with cv_tab1:
        st.subheader("Live Camera Tracking")

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
            visualize = st.checkbox("Mostra Video", value=True)

        if st.button("🎬 Start Live Tracking", type="primary"):
            with st.spinner("Inizializzazione..."):
                processor = CoachTrackVisionProcessor(camera_source)

                if processor.initialize():
                    st.success("✅ Camera connessa")

                    # Check calibration
                    if not processor.is_calibrated:
                        st.warning("⚠️ Sistema non calibrato. I dati potrebbero essere imprecisi.")

                    # Start processing (questo blocca, quindi meglio in thread separato)
                    st.info("🎬 Processing avviato... Premi 'q' nella finestra video per fermare")

                    try:
                        processor.run_realtime(
                            output_file=output_file,
                            visualize=visualize,
                            duration=duration
                        )
                        st.success(f"✅ Tracking completato! Dati salvati in {output_file}")
                    except Exception as e:
                        st.error(f"❌ Errore: {e}")
                else:
                    st.error("❌ Impossibile connettersi alla camera")

        # Show recent tracking data
        st.markdown("---")
        st.subheader("Recent Tracking Data")

        tracking_files = list(Path(".").glob("*tracking*.json"))
        if tracking_files:
            selected_file = st.selectbox("Select file", tracking_files)

            if st.button("📊 Load Data"):
                with open(selected_file, 'r') as f:
                    data = json.load(f)

                st.json(data['metadata'])

                # Convert to dataframe
                frames_data = []
                for frame in data['frames']:
                    for player in frame['players']:
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

                    # Store in session state for analysis
                    st.session_state.cv_tracking_data = df
                    st.success(f"✅ Loaded {len(df)} tracking points")

    # ============ TAB 2: PROCESS VIDEO ============
    with cv_tab2:
        st.subheader("Process Video File")

        uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

        if uploaded_video:
            # Save uploaded file
            video_path = f"uploaded_{uploaded_video.name}"
            with open(video_path, 'wb') as f:
                f.write(uploaded_video.read())

            st.success(f"✅ Video uploaded: {video_path}")

            col1, col2 = st.columns(2)
            with col1:
                output_json = st.text_input("Output JSON", value="video_tracking.json")
            with col2:
                create_annotated = st.checkbox("Crea Video Annotato", value=False)

            output_video = "annotated_" + video_path if create_annotated else None

            if st.button("🎬 Process Video", type="primary"):
                with st.spinner("Processing video..."):
                    processor = CoachTrackVisionProcessor(video_path)

                    success = processor.process_video_file(
                        video_path,
                        output_json,
                        output_video
                    )

                    if success:
                        st.success("✅ Video processato!")

                        # Download buttons
                        with open(output_json, 'r') as f:
                            json_data = f.read()

                        st.download_button(
                            "📥 Download JSON",
                            data=json_data,
                            file_name=output_json,
                            mime="application/json"
                        )

                        if output_video and Path(output_video).exists():
                            with open(output_video, 'rb') as f:
                                video_data = f.read()

                            st.download_button(
                                "📥 Download Video Annotato",
                                data=video_data,
                                file_name=output_video,
                                mime="video/mp4"
                            )
                    else:
                        st.error("❌ Errore processing video")

    # ============ TAB 3: CALIBRAZIONE ============
    with cv_tab3:
        st.subheader("🎯 Court Calibration")

        st.markdown("""
        La calibrazione è necessaria per convertire coordinate immagine in coordinate campo reali.

        **Procedura**:
        1. Connetti la camera
        2. Assicurati che il campo sia completamente visibile
        3. Clicca sui 4 angoli del campo nell'ordine: Basso-SX, Basso-DX, Alto-DX, Alto-SX
        4. Salva la calibrazione
        """)

        camera_source_calib = st.text_input(
            "Camera Source (Calibration)",
            value="0",
            key="calib_source"
        )

        if st.button("🎯 Start Calibration", type="primary"):
            with st.spinner("Inizializzazione camera..."):
                processor = CoachTrackVisionProcessor(camera_source_calib)

                if processor.initialize(calibration_file=None):
                    st.info("📍 Clicca sui 4 angoli del campo nella finestra che si apre...")

                    success = processor.calibrate_court()

                    if success:
                        st.success("✅ Calibrazione completata e salvata!")
                        st.balloons()
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
            st.info("ℹ️ Carica tracking data dalla tab 'Live Tracking' o 'Process Video'")
            return

        df = st.session_state.cv_tracking_data

        # Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Frames", df['frame'].nunique())
        with col2:
            st.metric("Players Tracked", df['player_id'].nunique())
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

        # Draw court
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=28, y1=15,
            line=dict(color="white", width=2),
            fillcolor="rgba(0,100,0,0.1)"
        )

        # Player trajectory
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

        # Heatmap
        st.subheader("Position Heatmap")

        fig_heat = go.Figure(data=go.Histogram2d(
            x=player_df['x'],
            y=player_df['y'],
            colorscale='Hot',
            nbinsx=28,
            nbinsy=15
        ))

        fig_heat.update_layout(
            title="Position Density",
            xaxis_title="X (meters)",
            yaxis_title="Y (meters)",
            width=800,
            height=500
        )

        st.plotly_chart(fig_heat, use_container_width=True)

        # Speed analysis
        st.subheader("Speed Analysis")

        player_df = player_df.sort_values('frame')
        player_df['dx'] = player_df['x'].diff()
        player_df['dy'] = player_df['y'].diff()
        player_df['dt'] = player_df['timestamp'].diff()
        player_df['speed'] = np.sqrt(player_df['dx']**2 + player_df['dy']**2) / player_df['dt']
        player_df['speed'] = player_df['speed'].fillna(0).clip(0, 15)  # Max 15 m/s

        fig_speed = px.line(
            player_df,
            x='frame',
            y='speed',
            title=f"Speed Over Time - Player {selected_player}",
            labels={'speed': 'Speed (m/s)', 'frame': 'Frame'}
        )

        st.plotly_chart(fig_speed, use_container_width=True)

        # Export data
        st.markdown("---")
        if st.button("📥 Export Player Data to CSV"):
            csv = player_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                data=csv,
                file_name=f"player_{selected_player}_tracking.csv",
                mime="text/csv"
            )


# =================================================================
# INTEGRARE NELL'APP PRINCIPALE
# =================================================================

def integrate_cv_in_main_app():
    """
    Funzione da chiamare nell'app.py principale per aggiungere CV tab

    Usage nell'app.py:

    from streamlit_cv_integration import add_computer_vision_tab

    # Dopo le altre tabs
    with st.tabs(...):
        ...
        with tab_cv:
            add_computer_vision_tab()
    """
    pass
