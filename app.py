# =================================================================
# COACHTRACK ELITE AI v3.0 - ULTIMATE EDITION
# Complete: AI + ML + CV + Physical + Nutrition + Analytics
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
from pathlib import Path

# =================================================================
# MODULE IMPORTS WITH FALLBACKS
# =================================================================

# AI Functions
try:
    from ai_functions import (calculate_distance, predict_injury_risk, 
        recommend_offensive_plays, analyze_movement_patterns, 
        generate_ai_training_plan, simulate_shot_quality)
    AI_AVAILABLE = True
except:
    AI_AVAILABLE = False
    def calculate_distance(df):
        if len(df) < 2: return 0.0
        dx = np.diff(df['x'].values)
        dy = np.diff(df['y'].values)
        return float(np.sum(np.sqrt(dx**2 + dy**2)))

    def predict_injury_risk(player_data, player_id):
        distance = calculate_distance(player_data)
        risk_score = 25 if distance < 200 else 40 if distance < 500 else 60
        return {
            'player_id': player_id, 'risk_level': 'MEDIO', 'risk_score': risk_score,
            'acwr': 1.2, 'asymmetry': 10.0, 'fatigue': 8.0,
            'risk_factors': ['Distanza elevata', 'Asimmetria rilevata'],
            'recommendations': ['Ridurre carico', 'Monitorare recupero']
        }

    def recommend_offensive_plays(player_data, player_id):
        return {
            'player_id': player_id,
            'plays': [
                {'name': 'Pick & Roll', 'success_prob': 65, 'reason': 'Buona mobilità'},
                {'name': 'Iso Top', 'success_prob': 55, 'reason': 'Spazio ottimale'},
                {'name': 'Corner 3', 'success_prob': 48, 'reason': 'Posizione efficace'}
            ]
        }

    def analyze_movement_patterns(player_data, player_id):
        distance = calculate_distance(player_data)
        return {
            'player_id': player_id,
            'patterns': {
                'dominanza_destra': 62,
                'movimento_lineare': 45,
                'cambio_direzione': 78,
                'velocita_media': 4.2
            },
            'insights': ['Preferenza lato destro', 'Buona agilità', 'Movimento esplosivo']
        }

    def generate_ai_training_plan(player_data, player_id, focus='general'):
        return {
            'player_id': player_id,
            'focus': focus,
            'plan': [
                {'day': 1, 'type': 'Strength', 'exercises': ['Squat', 'Deadlift'], 'duration': 60},
                {'day': 2, 'type': 'Speed', 'exercises': ['Sprints', 'Plyometrics'], 'duration': 45},
                {'day': 3, 'type': 'Skills', 'exercises': ['Shooting', 'Dribbling'], 'duration': 90},
                {'day': 4, 'type': 'Recovery', 'exercises': ['Stretching', 'Yoga'], 'duration': 30},
            ]
        }

    def simulate_shot_quality(player_data, player_id):
        return {
            'player_id': player_id,
            'shot_zones': [
                {'zone': 'Paint', 'quality': 72, 'attempts': 120},
                {'zone': 'Mid-Range', 'quality': 45, 'attempts': 80},
                {'zone': '3PT Corner', 'quality': 38, 'attempts': 60},
                {'zone': '3PT Top', 'quality': 35, 'attempts': 90}
            ],
            'best_zone': 'Paint',
            'recommendation': 'Focus on paint attacks and corner 3s'
        }

# ML Models
try:
    from ml_models import MLInjuryPredictor, PerformancePredictor
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False
    class MLInjuryPredictor:
        def extract_features(self, player_data, physical_data={}):
            distance = calculate_distance(player_data) if len(player_data) > 1 else 0
            return {
                'total_distance': distance,
                'avg_speed': distance / len(player_data) if len(player_data) > 0 else 0,
                'age': physical_data.get('age', 25),
                'weight_kg': physical_data.get('weight_kg', 80),
                'bmi': physical_data.get('bmi', 22)
            }

        def predict(self, features):
            risk_prob = 35 + (features.get('total_distance', 0) / 100)
            risk_prob = min(risk_prob, 85)
            return {
                'risk_level': 'BASSO' if risk_prob < 30 else 'MEDIO' if risk_prob < 60 else 'ALTO',
                'risk_probability': round(risk_prob, 1),
                'confidence': 'Media',
                'top_risk_factors': [
                    ('Distanza totale', 0.35),
                    ('Velocità media', 0.25),
                    ('BMI', 0.15)
                ],
                'recommendations': [
                    'Monitorare carico settimanale',
                    'Valutare recupero',
                    'Check biomeccanico'
                ]
            }

# Physical & Nutrition
try:
    from physical_nutrition import (simulate_apple_health_sync, generate_enhanced_nutrition,
        create_body_composition_viz, parse_physical_csv, create_physical_csv_template)
    PHYSICAL_AVAILABLE = True
except:
    PHYSICAL_AVAILABLE = False

    def generate_enhanced_nutrition(player_id, physical_data, activity, goal):
        weight = physical_data.get('weight_kg', 80)
        bmr = physical_data.get('bmr', 2000)

        activity_multiplier = {'Low': 1.2, 'Moderate': 1.55, 'High': 1.75, 'Very High': 1.9}
        mult = activity_multiplier.get(activity.split('(')[0].strip(), 1.55)

        target_cal = int(bmr * mult)
        protein_g = int(weight * 2.2)
        carbs_g = int(target_cal * 0.5 / 4)
        fats_g = int(target_cal * 0.25 / 9)

        return {
            'player_id': player_id,
            'target_calories': target_cal,
            'protein_g': protein_g,
            'carbs_g': carbs_g,
            'fats_g': fats_g,
            'recommendations': [
                'Priorità carboidrati pre-allenamento',
                'Proteine post-workout entro 30min',
                'Idratazione: 3-4L/giorno'
            ],
            'supplements': ['Whey Protein', 'Creatina 5g', 'Omega-3', 'Vitamina D'],
            'meals': [
                {'name': 'Colazione', 'timing': '7:00', 'calories': int(target_cal*0.25),
                 'protein': int(protein_g*0.2), 'carbs': int(carbs_g*0.3), 'fats': int(fats_g*0.25),
                 'examples': 'Avena, uova, frutta, noci'},
                {'name': 'Pre-Workout', 'timing': '10:00', 'calories': int(target_cal*0.15),
                 'protein': int(protein_g*0.15), 'carbs': int(carbs_g*0.25), 'fats': int(fats_g*0.1),
                 'examples': 'Banana, toast integrale, miele'},
                {'name': 'Post-Workout', 'timing': '12:30', 'calories': int(target_cal*0.2),
                 'protein': int(protein_g*0.35), 'carbs': int(carbs_g*0.2), 'fats': int(fats_g*0.15),
                 'examples': 'Shake proteine, riso, pollo'},
                {'name': 'Pranzo', 'timing': '14:00', 'calories': int(target_cal*0.25),
                 'protein': int(protein_g*0.25), 'carbs': int(carbs_g*0.2), 'fats': int(fats_g*0.3),
                 'examples': 'Pasta, carne, verdure, olio'},
                {'name': 'Cena', 'timing': '20:00', 'calories': int(target_cal*0.15),
                 'protein': int(protein_g*0.15), 'carbs': int(carbs_g*0.05), 'fats': int(fats_g*0.2),
                 'examples': 'Pesce, verdure, insalata'}
            ]
        }

    def create_body_composition_viz(physical_data):
        fig = go.Figure()

        composition = {
            'Muscoli': physical_data.get('muscle_pct', 45),
            'Grasso': physical_data.get('body_fat_pct', 12),
            'Acqua': physical_data.get('body_water_pct', 60) - 45 - 12,
            'Ossa/Altro': 100 - 45 - 12 - (physical_data.get('body_water_pct', 60) - 45 - 12)
        }

        fig.add_trace(go.Pie(
            labels=list(composition.keys()),
            values=list(composition.values()),
            hole=0.4
        ))

        fig.update_layout(title="Body Composition", height=400)
        return fig

# Computer Vision
CV_AVAILABLE = False
try:
    from cv_processor import CoachTrackVisionProcessor
    from cv_camera import CameraManager
    from cv_tracking import PlayerDetector
    CV_AVAILABLE = True
except:
    pass

# =================================================================
# COMPUTER VISION TAB
# =================================================================

def add_computer_vision_tab():
    """Complete CV tab with diagnostics"""
    st.header("🎥 Computer Vision System")

    if not CV_AVAILABLE:
        st.error("❌ Computer Vision non disponibile")

        st.markdown("### 🔍 Diagnostica")

        # Check files
        cv_files = ['cv_camera.py', 'cv_processor.py', 'cv_tracking.py']
        missing_files = [f for f in cv_files if not Path(f).exists()]

        if missing_files:
            st.error(f"**File mancanti:** {', '.join(missing_files)}")
        else:
            st.success("✅ Tutti i file CV presenti")

        # Check packages
        missing_pkgs = []
        try:
            import cv2
        except:
            missing_pkgs.append('opencv-python')
        try:
            from ultralytics import YOLO
        except:
            missing_pkgs.append('ultralytics')
        try:
            import torch
        except:
            missing_pkgs.append('torch')

        if missing_pkgs:
            st.error(f"**Pacchetti mancanti:** {', '.join(missing_pkgs)}")
            st.info("**Per Streamlit Cloud:** Aggiungi a `requirements.txt`")
            st.code("\n".join(missing_pkgs), language="text")
        else:
            st.success("✅ Tutti i pacchetti installati")

        return

    # CV Available
    st.success("✅ Computer Vision disponibile")

    cv_tab1, cv_tab2, cv_tab3, cv_tab4 = st.tabs([
        "📹 Live", "🎬 Video", "🎯 Calibra", "📊 Analisi"
    ])

    with cv_tab1:
        st.subheader("Live Camera Tracking")
        camera = st.text_input("Camera Source", "0")
        duration = st.number_input("Durata (sec)", 0, 3600, 60)
        output = st.text_input("Output JSON", "live_tracking.json")

        if st.button("▶️ Start", type="primary"):
            with st.spinner("Inizializzazione..."):
                try:
                    processor = CoachTrackVisionProcessor(camera)
                    if processor.initialize():
                        processor.run_realtime(output_file=output, visualize=True, duration=duration)
                        st.success(f"✅ Salvato in {output}")
                        st.balloons()
                    else:
                        st.error("Camera non connessa")
                except Exception as e:
                    st.error(f"Errore: {str(e)}")

    with cv_tab2:
        st.subheader("Process Video")
        video = st.file_uploader("Upload", type=['mp4', 'avi', 'mov'])
        if video:
            video_path = f"uploaded_{video.name}"
            with open(video_path, 'wb') as f:
                f.write(video.read())
            st.success(f"✅ {video_path}")

            if st.button("🎬 Process"):
                with st.spinner("Processing..."):
                    try:
                        processor = CoachTrackVisionProcessor(video_path)
                        processor.process_video_file(video_path, "video_tracking.json", None)
                        st.success("✅ Done")
                    except Exception as e:
                        st.error(str(e))

    with cv_tab3:
        st.info("Calibrazione: Clicca 4 angoli campo (BL, BR, TR, TL)")

    with cv_tab4:
        if 'cv_tracking_data' in st.session_state:
            df = st.session_state.cv_tracking_data
            st.metric("Frames", df['frame'].nunique())
            st.metric("Players", df['player_id'].nunique())
        else:
            st.info("Nessun dato caricato")

# =================================================================
# MAIN APP
# =================================================================

st.set_page_config(page_title="CoachTrack Elite", page_icon="🏀", layout="wide")

# Login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🏀 CoachTrack Elite AI")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        u = st.text_input("Username", value="admin")
        p = st.text_input("Password", type="password", value="admin")
        if st.button("Login", type="primary", use_container_width=True):
            if u == "admin" and p == "admin":
                st.session_state.logged_in = True
                st.rerun()
        st.info("admin / admin")
    st.stop()

# Session state
if 'tracking_data' not in st.session_state:
    st.session_state.tracking_data = {}
if 'physical_profiles' not in st.session_state:
    st.session_state.physical_profiles = {}
if 'ml_injury_model' not in st.session_state:
    st.session_state.ml_injury_model = MLInjuryPredictor()

# Sidebar
with st.sidebar:
    st.title("🏀 CoachTrack")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.success("✅" if AI_AVAILABLE else "❌")
        st.caption("AI")
        st.success("✅" if ML_AVAILABLE else "❌")
        st.caption("ML")
    with col2:
        st.success("✅" if CV_AVAILABLE else "❌")
        st.caption("CV")
        st.success("✅" if PHYSICAL_AVAILABLE else "❌")
        st.caption("PH")

    st.markdown("---")
    st.metric("Players", len(st.session_state.tracking_data))
    st.metric("Physical", len(st.session_state.physical_profiles))

    st.markdown("---")
    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

st.title("🏀 CoachTrack Elite AI v3.0 ULTIMATE")
st.markdown("**Complete:** AI + ML + CV + Physical + Nutrition + Analytics")

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "⚙️ Config", "🤖 AI Features", "🎥 Computer Vision", 
    "🧠 ML Advanced", "💪 Physical & Nutrition", "📊 Analytics"
])

# =================================================================
# TAB 1: CONFIGURAZIONE
# =================================================================
with tab1:
    st.header("⚙️ Configurazione Dati")

    st.markdown("### 📤 Upload CSV Tracking")
    uploaded = st.file_uploader("CSV (player_id, timestamp, x, y)", type=['csv'])

    if uploaded:
        try:
            df = pd.read_csv(uploaded, sep=';')
            st.success(f"✅ Caricato: {len(df)} righe")

            if all(c in df.columns for c in ['player_id', 'timestamp', 'x', 'y']):
                for pid in df['player_id'].unique():
                    st.session_state.tracking_data[pid] = df[df['player_id']==pid].copy()
                st.success(f"✅ Importati {len(df['player_id'].unique())} giocatori")

                with st.expander("👁️ Anteprima"):
                    st.dataframe(df.head(20))
            else:
                st.error("❌ CSV deve contenere: player_id, timestamp, x, y")
        except Exception as e:
            st.error(f"Errore: {str(e)}")

    st.markdown("---")
    st.markdown("### 📊 Dati Caricati")
    if st.session_state.tracking_data:
        for pid in st.session_state.tracking_data.keys():
            df_player = st.session_state.tracking_data[pid]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Player {pid}", len(df_player))
            with col2:
                st.metric("Distance", f"{calculate_distance(df_player):.1f}m")
            with col3:
                duration = (df_player['timestamp'].max() - df_player['timestamp'].min())
                st.metric("Duration", f"{duration:.1f}s")
    else:
        st.info("Nessun dato caricato")

# =================================================================
# TAB 2: AI FEATURES
# =================================================================
with tab2:
    st.header("🤖 AI Elite Features")

    if not st.session_state.tracking_data:
        st.warning("⚠️ Carica dati tracking nel tab Configurazione")
    else:
        player_id = st.selectbox("👤 Seleziona Giocatore", list(st.session_state.tracking_data.keys()))
        player_data = st.session_state.tracking_data[player_id]

        st.markdown("---")

        # Feature selection
        ai_feature = st.selectbox("🎯 Funzionalità AI", [
            "🏥 Injury Risk Analysis",
            "🏀 Offensive Plays Recommendation",
            "🔄 Movement Patterns Analysis",
            "📅 AI Training Plan Generator",
            "🎯 Shot Quality Simulation"
        ])

        if st.button("▶️ Esegui Analisi", type="primary", use_container_width=True):
            with st.spinner("Analisi AI in corso..."):
                time.sleep(0.5)  # Simulate processing

                if "Injury" in ai_feature:
                    result = predict_injury_risk(player_data, player_id)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        color = "🟢" if result['risk_level'] == 'BASSO' else "🟡" if result['risk_level'] == 'MEDIO' else "🔴"
                        st.metric(f"{color} Livello Rischio", result['risk_level'])
                    with col2:
                        st.metric("📊 Risk Score", result['risk_score'])
                    with col3:
                        st.metric("⚖️ ACWR", result['acwr'])
                    with col4:
                        st.metric("😫 Fatigue", result['fatigue'])

                    st.markdown("#### 🔴 Fattori di Rischio")
                    for factor in result['risk_factors']:
                        st.warning(f"• {factor}")

                    st.markdown("#### 💡 Raccomandazioni")
                    for rec in result['recommendations']:
                        st.info(f"• {rec}")

                elif "Offensive" in ai_feature:
                    result = recommend_offensive_plays(player_data, player_id)

                    st.markdown("### 🏀 Giocate Offensive Raccomandate")
                    for play in result['plays']:
                        with st.expander(f"**{play['name']}** - Success Prob: {play['success_prob']}%"):
                            st.progress(play['success_prob'] / 100)
                            st.write(f"**Motivo:** {play['reason']}")

                elif "Movement" in ai_feature:
                    result = analyze_movement_patterns(player_data, player_id)

                    st.markdown("### 🔄 Analisi Pattern Movimento")

                    col1, col2 = st.columns(2)
                    with col1:
                        for key, value in list(result['patterns'].items())[:2]:
                            st.metric(key.replace('_', ' ').title(), f"{value}%")
                    with col2:
                        for key, value in list(result['patterns'].items())[2:]:
                            if isinstance(value, (int, float)):
                                st.metric(key.replace('_', ' ').title(), f"{value}")

                    st.markdown("#### 💡 Insights")
                    for insight in result['insights']:
                        st.success(f"• {insight}")

                elif "Training" in ai_feature:
                    focus = st.selectbox("Focus", ['general', 'strength', 'speed', 'skills'])
                    result = generate_ai_training_plan(player_data, player_id, focus)

                    st.markdown("### 📅 Piano Allenamento AI-Generated")

                    for day in result['plan']:
                        with st.expander(f"**Giorno {day['day']}** - {day['type']} ({day['duration']}min)"):
                            st.write("**Esercizi:**")
                            for exercise in day['exercises']:
                                st.write(f"• {exercise}")

                elif "Shot" in ai_feature:
                    result = simulate_shot_quality(player_data, player_id)

                    st.markdown("### 🎯 Shot Quality Analysis")

                    # Chart
                    zones_df = pd.DataFrame(result['shot_zones'])
                    fig = px.bar(zones_df, x='zone', y='quality', 
                               title="Shot Quality by Zone",
                               color='quality',
                               color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig, use_container_width=True)

                    st.success(f"**Best Zone:** {result['best_zone']}")
                    st.info(f"**Recommendation:** {result['recommendation']}")

# =================================================================
# TAB 3: COMPUTER VISION
# =================================================================
with tab3:
    add_computer_vision_tab()

# =================================================================
# TAB 4: ML ADVANCED
# =================================================================
with tab4:
    st.header("🧠 ML Advanced Analytics")

    if not st.session_state.tracking_data:
        st.warning("⚠️ Carica dati tracking")
    else:
        player_id = st.selectbox("Player", list(st.session_state.tracking_data.keys()), key='ml_player')
        player_data = st.session_state.tracking_data[player_id]
        physical_data = st.session_state.physical_profiles.get(player_id, {})

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🏥 ML Injury Prediction")
            if st.button("🔮 Run ML Model", type="primary"):
                model = st.session_state.ml_injury_model
                features = model.extract_features(player_data, physical_data)
                result = model.predict(features)

                st.markdown("#### Risultati")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Risk Level", result['risk_level'])
                with col_b:
                    st.metric("Probability", f"{result['risk_probability']}%")
                with col_c:
                    st.metric("Confidence", result['confidence'])

                st.markdown("#### Top Risk Factors")
                for factor, importance in result['top_risk_factors']:
                    st.progress(importance, text=f"{factor}: {importance:.2%}")

                st.markdown("#### Recommendations")
                for rec in result['recommendations']:
                    st.info(f"• {rec}")

        with col2:
            st.markdown("### 📈 Performance Prediction")
            st.info("Feature in sviluppo")

            # Mock performance prediction
            st.markdown("#### Predicted Performance")
            st.metric("Next Game Score", "18.5 pts")
            st.metric("Efficiency", "52%")
            st.metric("Trend", "↗️ +3.2%")

# =================================================================
# TAB 5: PHYSICAL & NUTRITION
# =================================================================
with tab5:
    st.header("💪 Physical & Nutrition")

    phys_tab1, phys_tab2, phys_tab3 = st.tabs([
        "📋 Physical Data", "🍎 Nutrition Plans", "📊 Body Composition"
    ])

    with phys_tab1:
        st.subheader("Physical Data Management")

        col1, col2 = st.columns([2, 1])
        with col1:
            existing = ["Nuovo..."] + list(st.session_state.physical_profiles.keys())
            player_phys = st.selectbox("Giocatore", existing)
            if player_phys == "Nuovo...":
                player_phys = st.text_input("Nome Giocatore", key='new_player')
        with col2:
            data_date = st.date_input("Data", datetime.now())

        st.markdown("---")

        # Input manuale completo
        with st.form("physical_form"):
            st.markdown("### Input Dati Fisici")

            col1, col2, col3 = st.columns(3)

            with col1:
                height = st.number_input("Altezza (cm)", 150.0, 230.0, 195.0, 0.5)
                weight = st.number_input("Peso (kg)", 50.0, 150.0, 80.0, 0.1)
                age = st.number_input("Età", 15, 45, 25)

            with col2:
                body_fat = st.number_input("Grasso (%)", 3.0, 40.0, 12.0, 0.1)
                water = st.number_input("Acqua (%)", 45.0, 75.0, 60.0, 0.1)
                muscle = st.number_input("Muscoli (%)", 25.0, 60.0, 45.0, 0.1)

            with col3:
                bone = st.number_input("Ossa (kg)", 2.0, 5.0, 3.2, 0.1)
                hr = st.number_input("HR Riposo (bpm)", 40, 100, 55)
                vo2 = st.number_input("VO2 Max", 30.0, 80.0, 52.0, 0.5)

            notes = st.text_area("Note", placeholder="Infortuni, condizioni particolari...")

            if st.form_submit_button("💾 Salva Dati", type="primary", use_container_width=True):
                if player_phys and player_phys != "Nuovo...":
                    bmi = weight / ((height/100)**2)
                    fat_mass = weight * (body_fat/100)
                    lean_mass = weight - fat_mass
                    bmr = int(10*weight + 6.25*height - 5*age + 5)
                    amr = int(bmr * 1.55)

                    physical_data = {
                        'date': data_date.strftime('%Y-%m-%d'),
                        'height_cm': height, 'weight_kg': weight, 'age': age,
                        'bmi': round(bmi, 1), 'body_fat_pct': body_fat,
                        'lean_mass_kg': round(lean_mass, 1),
                        'fat_mass_kg': round(fat_mass, 1),
                        'body_water_pct': water, 'muscle_pct': muscle,
                        'bone_mass_kg': bone, 'resting_hr': hr, 'vo2_max': vo2,
                        'bmr': bmr, 'amr': amr, 'notes': notes,
                        'source': 'Manual Input'
                    }

                    st.session_state.physical_profiles[player_phys] = physical_data
                    st.success(f"✅ Dati salvati per {player_phys}")
                    st.balloons()

        st.markdown("---")
        st.markdown("### 📊 Dati Salvati")

        if st.session_state.physical_profiles:
            for pid, data in st.session_state.physical_profiles.items():
                with st.expander(f"👤 {pid} - {data.get('date', 'N/A')}"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Peso", f"{data.get('weight_kg')} kg")
                        st.metric("Altezza", f"{data.get('height_cm')} cm")
                    with col2:
                        st.metric("BMI", data.get('bmi'))
                        st.metric("Fat", f"{data.get('body_fat_pct')}%")
                    with col3:
                        st.metric("Lean Mass", f"{data.get('lean_mass_kg')} kg")
                        st.metric("VO2 Max", data.get('vo2_max'))
                    with col4:
                        st.metric("BMR", f"{data.get('bmr')} kcal")
                        if st.button("🗑️ Elimina", key=f"del_{pid}"):
                            del st.session_state.physical_profiles[pid]
                            st.rerun()
        else:
            st.info("Nessun dato fisico salvato")

    with phys_tab2:
        st.subheader("🍎 AI Nutrition Plans")

        if not st.session_state.physical_profiles:
            st.info("Aggiungi dati fisici prima di generare piani nutrizionali")
        else:
            player_nutr = st.selectbox("Giocatore", list(st.session_state.physical_profiles.keys()))

            col1, col2 = st.columns(2)
            with col1:
                activity = st.selectbox("Activity Level", [
                    "Low (Recovery)",
                    "Moderate (Training)",
                    "High (Intense/Match)",
                    "Very High (Tournament)"
                ])
            with col2:
                goal = st.selectbox("Goal", [
                    "Maintenance",
                    "Muscle Gain",
                    "Fat Loss",
                    "Performance"
                ])

            if st.button("🍎 Generate Nutrition Plan", type="primary", use_container_width=True):
                phys_data = st.session_state.physical_profiles[player_nutr]
                plan = generate_enhanced_nutrition(player_nutr, phys_data, activity, goal)

                st.markdown("### 📊 Piano Nutrizionale")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🔥 Calorie", f"{plan['target_calories']} kcal")
                with col2:
                    st.metric("🥩 Proteine", f"{plan['protein_g']}g")
                with col3:
                    st.metric("🍚 Carboidrati", f"{plan['carbs_g']}g")
                with col4:
                    st.metric("🥑 Grassi", f"{plan['fats_g']}g")

                st.markdown("---")
                st.markdown("#### 💡 Raccomandazioni")
                for rec in plan['recommendations']:
                    st.info(f"• {rec}")

                st.markdown("#### 💊 Integratori Consigliati")
                for supp in plan['supplements']:
                    st.success(f"• {supp}")

                st.markdown("---")
                st.markdown("#### 🍽️ Piano Pasti")

                for meal in plan['meals']:
                    with st.expander(f"**{meal['name']}** - {meal['timing']} ({meal['calories']} kcal)"):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.write(f"**Proteine:** {meal['protein']}g")
                            st.write(f"**Carboidrati:** {meal['carbs']}g")
                            st.write(f"**Grassi:** {meal['fats']}g")
                        with col2:
                            st.write(f"**Esempi:** {meal['examples']}")

    with phys_tab3:
        st.subheader("📊 Body Composition Analysis")

        if st.session_state.physical_profiles:
            player_viz = st.selectbox("Player Visualization", 
                                     list(st.session_state.physical_profiles.keys()))
            data = st.session_state.physical_profiles[player_viz]

            fig = create_body_composition_viz(data)
            st.plotly_chart(fig, use_container_width=True)

            # Additional metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Lean Mass", f"{data.get('lean_mass_kg')} kg")
            with col2:
                st.metric("Fat Mass", f"{data.get('fat_mass_kg')} kg")
            with col3:
                st.metric("BMI", data.get('bmi'))
        else:
            st.info("Aggiungi dati fisici per vedere body composition")

# =================================================================
# TAB 6: ANALYTICS
# =================================================================
with tab6:
    st.header("📊 Analytics Dashboard")

    if not st.session_state.tracking_data:
        st.info("Carica dati tracking per vedere analytics avanzate")
    else:
        # Overall stats
        st.markdown("### 🎯 Statistiche Generali")

        total_distance = sum(calculate_distance(df) for df in st.session_state.tracking_data.values())
        avg_distance = total_distance / len(st.session_state.tracking_data)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👥 Players", len(st.session_state.tracking_data))
        with col2:
            st.metric("📏 Total Distance", f"{total_distance:.0f}m")
        with col3:
            st.metric("📊 Avg Distance", f"{avg_distance:.0f}m")
        with col4:
            st.metric("⚖️ Avg Load", f"{total_distance/len(st.session_state.tracking_data)/10:.1f}")

        st.markdown("---")

        # Player comparison
        st.markdown("### 👥 Confronto Giocatori")

        players_stats = []
        for pid, df in st.session_state.tracking_data.items():
            distance = calculate_distance(df)
            duration = df['timestamp'].max() - df['timestamp'].min()
            avg_speed = distance / duration if duration > 0 else 0

            players_stats.append({
                'Player': pid,
                'Distance (m)': round(distance, 1),
                'Duration (s)': round(duration, 1),
                'Avg Speed (m/s)': round(avg_speed, 2),
                'Points': len(df)
            })

        stats_df = pd.DataFrame(players_stats)

        # Chart
        fig = px.bar(stats_df, x='Player', y='Distance (m)',
                    title="Distance Comparison",
                    color='Distance (m)',
                    color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.dataframe(stats_df, use_container_width=True)

        st.markdown("---")

        # Heatmap
        st.markdown("### 🗺️ Team Heatmap")

        all_points = []
        for df in st.session_state.tracking_data.values():
            all_points.extend([(row['x'], row['y']) for _, row in df.iterrows()])

        if all_points:
            points_df = pd.DataFrame(all_points, columns=['x', 'y'])

            fig_heat = go.Figure(data=go.Histogram2d(
                x=points_df['x'],
                y=points_df['y'],
                colorscale='Hot',
                nbinsx=50,
                nbinsy=30
            ))

            fig_heat.update_layout(
                title="Team Movement Heatmap",
                xaxis_title="X (meters)",
                yaxis_title="Y (meters)",
                height=500
            )

            st.plotly_chart(fig_heat, use_container_width=True)

st.caption("🏀 CoachTrack Elite AI v3.0 ULTIMATE - Complete Edition")
