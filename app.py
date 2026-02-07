# =================================================================
# COACHTRACK ELITE AI v3.0 - MAIN APPLICATION
# Complete with Physical/Nutrition AI + ML Models + Tactical AI + Groq
# =================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO
import json

# Import all modules
from ai_functions import (
    calculate_distance, calculate_speed, detect_jumps_imu,
    predict_injury_risk, recommend_offensive_plays,
    optimize_defensive_matchups, analyze_movement_patterns,
    simulate_shot_quality, generate_ai_training_plan
)

from physical_nutrition import (
    PHYSICAL_METRICS, parse_physical_csv, validate_physical_data,
    simulate_apple_health_sync, calculate_advanced_bmr, calculate_amr,
    estimate_body_composition, generate_enhanced_nutrition,
    create_body_composition_viz, export_physical_data_excel,
    create_physical_csv_template
)

from groq_integration import (
    generate_nutrition_report_nlg, generate_training_plan_nlg,
    generate_scout_report_nlg, game_assistant_chat,
    generate_performance_summary, test_groq_connection
)

from ml_models import (
    MLInjuryPredictor, PerformancePredictor, ShotFormAnalyzer
)

from tactical_ai import (
    TacticalPatternRecognizer, ScoutReportGenerator,
    LineupOptimizer, GameAssistant, simulate_opponent_stats
)

# PDF support (optional)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="CoachTrack Elite AI v3.0",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================================================================
# AUTHENTICATION
# =================================================================

def check_login(username, password):
    """Simple authentication"""
    return username == "admin" and password == "admin"

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🏀 CoachTrack Elite AI v3.0")
    st.markdown("### 🔐 Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password", value="admin")
        
        if st.button("🚀 Login", use_container_width=True, type="primary"):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Credenziali errate")
    
    st.info("💡 Default: admin / admin")
    st.stop()

# =================================================================
# SESSION STATE INITIALIZATION
# =================================================================

if 'language' not in st.session_state:
    st.session_state.language = 'it'

if 'tracking_data' not in st.session_state:
    st.session_state.tracking_data = {}

if 'imu_data' not in st.session_state:
    st.session_state.imu_data = {}

if 'physical_profiles' not in st.session_state:
    st.session_state.physical_profiles = {}

if 'physical_history' not in st.session_state:
    st.session_state.physical_history = {}

if 'current_nutrition_plan' not in st.session_state:
    st.session_state.current_nutrition_plan = None

if 'ml_injury_model' not in st.session_state:
    st.session_state.ml_injury_model = MLInjuryPredictor()

if 'performance_model' not in st.session_state:
    st.session_state.performance_model = PerformancePredictor()

if 'lineup_optimizer' not in st.session_state:
    st.session_state.lineup_optimizer = LineupOptimizer()

if 'game_assistant' not in st.session_state:
    st.session_state.game_assistant = GameAssistant()

# =================================================================
# TRANSLATIONS
# =================================================================

TRANSLATIONS = {
    'it': {
        'title': '🏀 CoachTrack Elite AI v3.0 - Sistema Completo',
        'welcome': 'Sistema completo con ML Models, Groq AI e Tactical Analytics',
        'config': 'Configurazione & Dati',
        'physical': 'Profilo Fisico & AI',
        'ai_features': 'AI Elite Features',
        'ml_advanced': 'ML Advanced',
        'tactical': 'Tactical AI & Scout',
        'analytics': 'Analytics & Reports',
        'logout': 'Logout'
    },
    'en': {
        'title': '🏀 CoachTrack Elite AI v3.0 - Complete System',
        'welcome': 'Complete system with ML Models, Groq AI and Tactical Analytics',
        'config': 'Configuration & Data',
        'physical': 'Physical Profile & AI',
        'ai_features': 'Elite AI Features',
        'ml_advanced': 'ML Advanced',
        'tactical': 'Tactical AI & Scout',
        'analytics': 'Analytics & Reports',
        'logout': 'Logout'
    }
}

t = TRANSLATIONS[st.session_state.language]

# =================================================================
# SIDEBAR
# =================================================================

with st.sidebar:
    st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=CoachTrack+v3.0", use_container_width=True)
    st.title("CoachTrack Elite")
    st.caption("v3.0 - Complete Edition")
    st.markdown("---")
    
    # Language selector
    lang = st.selectbox("🌐 Language", ["IT", "EN"], index=0 if st.session_state.language == 'it' else 1)
    if lang == "IT" and st.session_state.language != 'it':
        st.session_state.language = 'it'
        st.rerun()
    elif lang == "EN" and st.session_state.language != 'en':
        st.session_state.language = 'en'
        st.rerun()
    
    st.markdown("---")
    
    # Test Groq connection
    st.markdown("### 🤖 Groq Status")
    if st.button("Test Groq Connection", use_container_width=True):
        with st.spinner("Testing..."):
            success, message = test_groq_connection()
            if success:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
    
    st.markdown("---")
    
# Stats summary
st.markdown("### 📊 Data Summary")

# Corretto: aggiunto il nome completo della variabile e i due punti
if st.session_state.tracking_data:
    st.metric("Players UWB", len(st.session_state.tracking_data))

if st.session_state.physical_profiles:
    st.metric("Players Physical", len(st.session_state.physical_profiles))

st.markdown("---")

if st.button("🚪 " + t['logout'], use_container_width=True):
    st.session_state.logged_in = False
    st.rerun()



# =================================================================
# MAIN UI - TABS
# =================================================================

st.title(t['title'])
st.markdown(t['welcome'])

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 " + t['config'],
    "🏋️ " + t['physical'],
    "🤖 " + t['ai_features'],
    "🧠 " + t['ml_advanced'],
    "⚔️ " + t['tactical'],
    "📈 " + t['analytics']
])

# =================================================================
# TAB 1: CONFIGURATION & DATA UPLOAD
# =================================================================

with tab1:
    st.header("⚙️ Configurazione e Caricamento Dati")
    
    st.markdown("### 📡 Carica Dati Tracking UWB")
    
    uploaded_uwb = st.file_uploader("Carica CSV Tracking UWB", type=['csv'], key='uwb_upload')
    
    if uploaded_uwb:
        try:
            df = pd.read_csv(uploaded_uwb)
            
            st.success(f"✅ File caricato: {len(df)} righe")
            
            required_cols = ['player_id', 'timestamp', 'x', 'y']
            if all(col in df.columns for col in required_cols):
                # Group by player
                for player_id in df['player_id'].unique():
                    player_df = df[df['player_id'] == player_id].copy()
                    st.session_state.tracking_data[player_id] = player_df
                
                st.success(f"✅ Dati importati per {len(df['player_id'].unique())} giocatori")
                
                # Preview
                with st.expander("👁️ Anteprima Dati"):
                    st.dataframe(df.head(20))
            else:
                st.error(f"❌ CSV deve contenere: {', '.join(required_cols)}")
        
        except Exception as e:
            st.error(f"❌ Errore lettura file: {e}")
    
    st.markdown("---")
    
    # IMU Data Upload
    st.markdown("### 📱 Carica Dati IMU (Salti)")
    
    uploaded_imu = st.file_uploader("Carica CSV IMU", type=['csv'], key='imu_upload')
    
    if uploaded_imu:
        try:
            imu_df = pd.read_csv(uploaded_imu)
            
            required_imu = ['player_id', 'timestamp', 'ax', 'ay', 'az']
            if all(col in imu_df.columns for col in required_imu):
                for player_id in imu_df['player_id'].unique():
                    player_imu = imu_df[imu_df['player_id'] == player_id].copy()
                    st.session_state.imu_data[player_id] = player_imu
                
                st.success(f"✅ Dati IMU importati per {len(imu_df['player_id'].unique())} giocatori")
            else:
                st.error(f"❌ CSV IMU deve contenere: {', '.join(required_imu)}")
        except Exception as e:
            st.error(f"❌ Errore: {e}")

# =================================================================
# TAB 2: PHYSICAL PROFILE & AI NUTRITION (ENHANCED)
# =================================================================

with tab2:
    st.header("🏋️ Profilo Fisico & AI Nutrition Enhanced")
    
    # Section 1: Upload Physical Data
    st.markdown("### 📊 Gestione Dati Fisici Completi")
    
    with st.expander("📥 Carica Dati Fisici", expanded=False):
        upload_tab, apple_tab, manual_tab = st.tabs(["📄 CSV Upload", "🍎 Apple Health", "✏️ Manuale"])
        
        with upload_tab:
            st.markdown("#### Upload CSV con Dati Fisici")
            
            # Template download
            template_csv = create_physical_csv_template()
            st.download_button(
                "📥 Scarica Template CSV",
                template_csv,
                "physical_data_template.csv",
                "text/csv",
                help="Scarica template con formato corretto"
            )
            
            uploaded_physical = st.file_uploader("Carica CSV Dati Fisici", type=['csv'], key='physical_csv')
            
            if uploaded_physical:
                df_physical, error = parse_physical_csv(uploaded_physical)
                
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.success(f"✅ File caricato: {len(df_physical)} righe")
                    
                    # Preview
                    st.dataframe(df_physical.head())
                    
                    if st.button("💾 Salva Dati Fisici", type="primary"):
                        saved_count = 0
                        for _, row in df_physical.iterrows():
                            player_id = row['player_id']
                            
                            # Create physical profile
                            profile = {
                                'date': row.get('date', datetime.now().strftime('%Y-%m-%d')),
                                'source': 'CSV Upload'
                            }
                            
                            # Add all available metrics
                            for metric in PHYSICAL_METRICS.keys():
                                if metric in row and pd.notna(row[metric]):
                                    profile[metric] = float(row[metric])
                            
                            # Validate
                            warnings = validate_physical_data(profile)
                            if warnings:
                                st.warning(f"⚠️ {player_id}: " + ", ".join(warnings))
                            
                            # Save
                            st.session_state.physical_profiles[player_id] = profile
                            
                            # Save to history
                            if player_id not in st.session_state.physical_history:
                                st.session_state.physical_history[player_id] = []
                            st.session_state.physical_history[player_id].append(profile)
                            
                            saved_count += 1
                        
                        st.success(f"✅ Salvati profili per {saved_count} giocatori")
                        st.rerun()
        
        with apple_tab:
            st.markdown("#### 🍎 Sincronizza Apple Health (Demo)")
            st.info("ℹ️ Questa è una simulazione. In produzione userà Apple HealthKit API.")
            
            if st.session_state.tracking_data:
                player_for_sync = st.selectbox("Seleziona Giocatore", list(st.session_state.tracking_data.keys()))
                
                if st.button("🔄 Sincronizza Apple Health (Demo)", type="primary"):
                    with st.spinner("Sincronizzazione in corso..."):
                        import time
                        time.sleep(1)
                        
                        health_data = simulate_apple_health_sync(player_for_sync)
                        st.session_state.physical_profiles[player_for_sync] = health_data
                        
                        # Add to history
                        if player_for_sync not in st.session_state.physical_history:
                            st.session_state.physical_history[player_for_sync] = []
                        st.session_state.physical_history[player_for_sync].append(health_data)
                    
                    st.success(f"✅ Dati sincronizzati per {player_for_sync}")
                    st.json(health_data)
            else:
                st.warning("⚠️ Carica prima dati UWB in Tab 1")
        
        with manual_tab:
            st.markdown("#### ✏️ Inserimento Manuale")
            
            if st.session_state.tracking_data:
                player_manual = st.selectbox("Giocatore", list(st.session_state.tracking_data.keys()), key='manual_player')
                
                col1, col2, col3 = st.columns(3)
                
                manual_data = {}
                
                with col1:
                    manual_data['weight_kg'] = st.number_input("Peso (kg)", 50.0, 150.0, 80.0, 0.1)
                    manual_data['bmi'] = st.number_input("BMI", 15.0, 35.0, 22.5, 0.1)
                    manual_data['lean_mass_kg'] = st.number_input("Massa Magra (kg)", 40.0, 120.0, 68.0, 0.1)
                
                with col2:
                    manual_data['body_fat_pct'] = st.number_input("Grasso (%)", 3.0, 40.0, 12.0, 0.1)
                    manual_data['body_water_pct'] = st.number_input("Acqua (%)", 45.0, 75.0, 60.0, 0.1)
                    manual_data['muscle_pct'] = st.number_input("Muscoli (%)", 25.0, 60.0, 45.0, 0.1)
                
                with col3:
                    manual_data['bone_mass_kg'] = st.number_input("Ossa (kg)", 2.0, 5.0, 3.2, 0.1)
                    manual_data['bmr'] = st.number_input("BMR (kcal)", 1200, 3000, 1800, 10)
                    manual_data['amr'] = st.number_input("AMR (kcal)", 1800, 5000, 2700, 10)
                
                if st.button("💾 Salva Dati Manuali", type="primary"):
                    manual_data['date'] = datetime.now().strftime('%Y-%m-%d')
                    manual_data['source'] = 'Manual Entry'
                    
                    warnings = validate_physical_data(manual_data)
                    if warnings:
                        st.warning("⚠️ " + ", ".join(warnings))
                    
                    st.session_state.physical_profiles[player_manual] = manual_data
                    
                    if player_manual not in st.session_state.physical_history:
                        st.session_state.physical_history[player_manual] = []
                    st.session_state.physical_history[player_manual].append(manual_data)
                    
                    st.success(f"✅ Dati salvati per {player_manual}")
            else:
                st.warning("⚠️ Carica prima dati UWB")
    
    st.divider()
    
    # Section 2: Body Composition Dashboard
    if st.session_state.physical_profiles:
        st.markdown("### 📈 Dashboard Composizione Corporea")
        
        selected_viz = st.selectbox("Seleziona Giocatore", 
                                    list(st.session_state.physical_profiles.keys()), key='viz_player')
        
        if selected_viz in st.session_state.physical_profiles:
            viz_data = st.session_state.physical_profiles[selected_viz]
            
            # Metrics overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Peso", f"{viz_data.get('weight_kg', 'N/A')} kg")
                st.metric("BMI", f"{viz_data.get('bmi', 'N/A')}")
            
            with col2:
                st.metric("Massa Magra", f"{viz_data.get('lean_mass_kg', 'N/A')} kg")
                st.metric("Grasso", f"{viz_data.get('body_fat_pct', 'N/A')}%")
            
            with col3:
                st.metric("Muscoli", f"{viz_data.get('muscle_pct', 'N/A')}%")
                st.metric("Acqua", f"{viz_data.get('body_water_pct', 'N/A')}%")
            
            with col4:
                st.metric("BMR", f"{viz_data.get('bmr', 'N/A')} kcal")
                st.metric("AMR", f"{viz_data.get('amr', 'N/A')} kcal")
            
            # Visualizations
            fig = create_body_composition_viz(viz_data)
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Section 3: Enhanced AI Nutrition WITH GROQ
    st.markdown("### 🥗 Piano Nutrizionale AI con Groq (Natural Language)")
    
    if st.session_state.physical_profiles:
        selected_nutrition = st.selectbox("Giocatore Nutrition", 
                                         list(st.session_state.physical_profiles.keys()),
                                         key='nutrition_player')
        
        col1, col2 = st.columns(2)
        
        with col1:
            activity_level = st.selectbox(
                "Livello Attività",
                ["Low (Recovery)", "Moderate (Training)", "High (Intense/Match)", "Very High (Tournament)"],
                index=1
            )
        
        with col2:
            goal = st.selectbox(
                "Obiettivo",
                ["Maintenance", "Muscle Gain", "Fat Loss", "Performance"],
                index=3
            )
        
        use_groq = st.checkbox("🤖 Usa Groq per report dettagliato (Natural Language)", value=True)
        
        if st.button("🚀 Genera Piano Nutrizionale AI", type="primary"):
            with st.spinner("Generazione piano in corso..."):
                nutrition_plan = generate_enhanced_nutrition(
                    selected_nutrition,
                    st.session_state.physical_profiles[selected_nutrition],
                    activity_level,
                    goal
                )
                
                st.session_state.current_nutrition_plan = nutrition_plan
                
                st.success(f"✅ Piano generato per {selected_nutrition}")
                
                # Targets
                st.markdown("#### 🎯 Target Giornalieri")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Calorie", f"{nutrition_plan['target_calories']} kcal")
                with col2:
                    st.metric("Proteine", f"{nutrition_plan['protein_g']}g")
                with col3:
                    st.metric("Carboidrati", f"{nutrition_plan['carbs_g']}g")
                with col4:
                    st.metric("Grassi", f"{nutrition_plan['fats_g']}g")
                
                # GROQ NATURAL LANGUAGE REPORT
                if use_groq:
                    st.markdown("#### 📝 Report Nutrizionale Dettagliato (Groq AI)")
                    with st.spinner("Groq sta generando report..."):
                        groq_report = generate_nutrition_report_nlg(
                            selected_nutrition,
                            nutrition_plan,
                            st.session_state.physical_profiles[selected_nutrition],
                            'it'
                        )
                        st.markdown(groq_report)
                
                # Meal Plan
                st.markdown("#### 🍽️ Piano Pasti")
                for meal in nutrition_plan['meals']:
                    with st.expander(f"{meal['name']} - {meal['calories']} kcal ({meal.get('timing', '')})"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Proteine:** {meal['protein']}g")
                        with col2:
                            st.write(f"**Carboidrati:** {meal['carbs']}g")
                        with col3:
                            st.write(f"**Grassi:** {meal['fats']}g")
                        
                        if 'examples' in meal:
                            st.info(f"💡 {meal['examples']}")
    else:
        st.warning("⚠️ Carica prima dati fisici")
# =================================================================
# TAB 3: AI ELITE FEATURES (Base AI)
# =================================================================

with tab3:
    st.header("🤖 Funzionalità AI Elite (Base)")
    
    if not st.session_state.tracking_data:
        st.warning("⚠️ Carica prima dati tracking in Tab 1")
        st.stop()
    
    ai_feature = st.selectbox(
        "Seleziona Funzionalità AI",
        [
            "🩺 Injury Risk Predictor (Base)",
            "🏀 Offensive Play Recommender",
            "🛡️ Defensive Matchup Optimizer",
            "🏃 Movement Pattern Analyzer",
            "🎯 Shot Quality Simulator"
        ]
    )
    
    selected_player_ai = st.selectbox("Seleziona Giocatore", list(st.session_state.tracking_data.keys()))
    
    if st.button("🚀 Esegui Analisi AI", type="primary"):
        player_data = st.session_state.tracking_data[selected_player_ai]
        
        with st.spinner("Analisi AI in corso..."):
            
            if "Injury Risk" in ai_feature:
                result = predict_injury_risk(player_data, selected_player_ai)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Livello Rischio", result['risk_level'], 
                             delta=f"{result['risk_score']}/100",
                             delta_color="inverse")
                with col2:
                    st.metric("ACWR", result['acwr'])
                with col3:
                    st.metric("Asimmetria", f"{result['asymmetry']}%")
                
                st.markdown("#### 📊 Fattori di Rischio")
                for factor in result['risk_factors']:
                    st.write(f"- {factor}")
                
                st.markdown("#### 💡 Raccomandazioni")
                for rec in result['recommendations']:
                    st.write(f"- {rec}")
            
            elif "Offensive Play" in ai_feature:
                result = recommend_offensive_plays(player_data)
                
                st.markdown("#### 🏀 Giocate Offensive Raccomandate")
                for play in result['recommended_plays']:
                    st.write(f"- {play}")
                
                st.markdown("#### 📊 Analisi")
                for reason in result['reasoning']:
                    st.info(reason)
            
            elif "Defensive Matchup" in ai_feature:
                result = optimize_defensive_matchups({selected_player_ai: player_data})
                
                st.markdown("#### 🛡️ Matchup Difensivi Ottimali")
                for match in result:
                    st.write(f"**{match['opponent']}** → {match['defender']} (Score: {match['match_score']})")
            
            elif "Movement Pattern" in ai_feature:
                result = analyze_movement_patterns(player_data, selected_player_ai)
                
                st.metric("Pattern Type", result['pattern_type'])
                
                st.markdown("#### 📊 Insights")
                for insight in result['insights']:
                    st.write(f"- {insight}")
                
                if result['anomalies']:
                    st.markdown("#### ⚠️ Anomalie Rilevate")
                    for anomaly in result['anomalies']:
                        st.warning(anomaly)
            
            elif "Shot Quality" in ai_feature:
                result = simulate_shot_quality(player_data, selected_player_ai)
                
                st.metric("Qualità Media Tiri", f"{result['avg_quality']}/100")
                
                if result['shots']:
                    st.markdown("#### 🎯 Analisi Tiri")
                    shots_df = pd.DataFrame(result['shots'])
                    st.dataframe(shots_df)
                
                st.markdown("#### 💡 Raccomandazioni")
                for rec in result['recommendations']:
                    st.write(f"- {rec}")

# =================================================================
# TAB 4: ML ADVANCED (Random Forest + Gradient Boosting)
# =================================================================

with tab4:
    st.header("🧠 ML Advanced - Machine Learning Models")
    
    if not st.session_state.tracking_data:
        st.warning("⚠️ Carica prima dati tracking in Tab 1")
        st.stop()
    
    ml_feature = st.selectbox(
        "Seleziona ML Feature",
        [
            "🤖 ML Injury Risk Predictor (Random Forest)",
            "📊 Performance Predictor Next Game (Gradient Boosting)",
            "📹 Shot Form Analyzer (Computer Vision - Placeholder)"
        ]
    )
    
    selected_ml_player = st.selectbox("Seleziona Giocatore", list(st.session_state.tracking_data.keys()), key='ml_player')
    
    if "ML Injury" in ml_feature:
        st.markdown("### 🤖 ML Injury Risk Predictor")
        st.info("Usa Random Forest con 12 features: ACWR, asymmetry, fatigue, workload, rest, physical data, etc.")
        
        if st.button("🚀 Predici Rischio Infortuni (ML)", type="primary"):
            with st.spinner("Training Random Forest model..."):
                player_data = st.session_state.tracking_data[selected_ml_player]
                physical_data = st.session_state.physical_profiles.get(selected_ml_player)
                
                # Extract features
                features = st.session_state.ml_injury_model.extract_features(
                    player_data, 
                    physical_data, 
                    player_age=25
                )
                
                # Predict
                prediction = st.session_state.ml_injury_model.predict(features)
                
                st.success("✅ Predizione completata!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    risk_color = "🔴" if prediction['risk_level'] == "ALTO" else "🟡" if prediction['risk_level'] == "MEDIO" else "🟢"
                    st.metric("Rischio ML", f"{risk_color} {prediction['risk_level']}", 
                             f"{prediction['risk_probability']}%")
                with col2:
                    st.metric("Model Confidence", "Alta" if prediction['risk_probability'] > 70 or prediction['risk_probability'] < 30 else "Media")
                with col3:
                    st.metric("Features Used", len(features))
                
                # Top risk factors
                st.markdown("#### 🔍 Top 5 Fattori di Rischio (Feature Importance)")
                for i, (feature, importance) in enumerate(prediction['top_risk_factors'], 1):
                    st.write(f"{i}. **{feature}**: {importance:.3f} (importance score)")
                
                # Recommendations
                st.markdown("#### 💡 Raccomandazioni ML-Based")
                for rec in prediction['recommendations']:
                    st.write(f"- {rec}")
                
                # Feature values used
                with st.expander("📊 Feature Values utilizzati"):
                    features_df = pd.DataFrame([features]).T
                    features_df.columns = ['Value']
                    st.dataframe(features_df)
    
    elif "Performance Predictor" in ml_feature:
        st.markdown("### 📊 Performance Predictor Next Game")
        st.info("Usa Gradient Boosting per predire points, assists, rebounds, efficiency della prossima partita")
        
        # Mock player stats history
        if st.button("🚀 Predici Performance Prossima Partita", type="primary"):
            with st.spinner("Training Gradient Boosting models..."):
                
                # Generate synthetic stats history
                stats_history = pd.DataFrame({
                    'points': np.random.randint(10, 25, 10),
                    'assists': np.random.randint(3, 8, 10),
                    'rebounds': np.random.randint(4, 10, 10),
                    'minutes': np.random.randint(25, 38, 10)
                })
                
                opponent_info = {
                    'rest_days': st.slider("Rest Days", 0, 4, 1),
                    'def_rating': st.slider("Opponent Def Rating", 100, 120, 110),
                    'location': st.radio("Location", ['home', 'away']),
                    'usage_rate': 25,
                    'fatigue': 0.1
                }
                
                # Get injury risk for context
                injury_risk = None
                if selected_ml_player in st.session_state.physical_profiles:
                    player_data = st.session_state.tracking_data[selected_ml_player]
                    physical_data = st.session_state.physical_profiles[selected_ml_player]
                    features = st.session_state.ml_injury_model.extract_features(player_data, physical_data)
                    injury_risk = st.session_state.ml_injury_model.predict(features)
                
                # Extract features for performance
                perf_features = st.session_state.performance_model.extract_features(
                    stats_history,
                    opponent_info,
                    injury_risk
                )
                
                # Predict
                predictions = st.session_state.performance_model.predict_next_game(perf_features)
                
                st.success("✅ Predizioni completate!")
                
                # Show predictions
                st.markdown("#### 🎯 Predizioni Prossima Partita")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Punti", f"{predictions['points']:.1f}")
                with col2:
                    st.metric("Assist", f"{predictions['assists']:.1f}")
                with col3:
                    st.metric("Rimbalzi", f"{predictions['rebounds']:.1f}")
                with col4:
                    st.metric("Efficiency", f"{predictions['efficiency']:.1f}")
                
                st.metric("Confidence Level", predictions['confidence'])
                
                # Stats history comparison
                st.markdown("#### 📈 Confronto con Storico (Ultimi 10 match)")
                comparison_df = pd.DataFrame({
                    'Metric': ['Points', 'Assists', 'Rebounds'],
                    'Avg Last 10': [stats_history['points'].mean(), 
                                   stats_history['assists'].mean(), 
                                   stats_history['rebounds'].mean()],
                    'Predicted Next': [predictions['points'], 
                                      predictions['assists'], 
                                      predictions['rebounds']]
                })
                st.dataframe(comparison_df)
                
                # GROQ SUMMARY
                if st.checkbox("🤖 Genera Analisi Dettagliata con Groq"):
                    with st.spinner("Groq sta generando analisi..."):
                        stats_summary = f"""
Statistiche ultimi 10 match:
- Punti: media {stats_history['points'].mean():.1f}, max {stats_history['points'].max()}, min {stats_history['points'].min()}
- Assist: media {stats_history['assists'].mean():.1f}
- Rimbalzi: media {stats_history['rebounds'].mean():.1f}

Predizioni prossima partita:
- Punti: {predictions['points']:.1f}
- Assist: {predictions['assists']:.1f}
- Rimbalzi: {predictions['rebounds']:.1f}
- Confidence: {predictions['confidence']}
"""
                        groq_analysis = generate_performance_summary(
                            selected_ml_player,
                            stats_summary,
                            predictions,
                            'it'
                        )
                        st.markdown("#### 📝 Analisi Groq")
                        st.markdown(groq_analysis)
    
    elif "Shot Form" in ml_feature:
        st.markdown("### 📹 Shot Form Analyzer")
        st.warning("⚠️ Feature in sviluppo - Richiede MediaPipe/OpenCV integration")
        
        shot_analyzer = ShotFormAnalyzer()
        result = shot_analyzer.analyze_shot_video("placeholder.mp4")
        
        st.info(result['message'])
        
        st.markdown("#### 🔜 Next Steps:")
        for step in result['next_steps']:
            st.write(f"- {step}")

# =================================================================
# TAB 5: TACTICAL AI & SCOUT
# =================================================================

with tab5:
    st.header("⚔️ Tactical AI & Auto-Scout Report")
    
    tactical_feature = st.selectbox(
        "Seleziona Feature Tattica",
        [
            "🔍 Auto-Scout Report Generator (con Groq)",
            "⚔️ Lineup Optimizer (ML-Based)",
            "📊 Tactical Pattern Recognition",
            "💬 Real-Time Game Assistant (Chat)"
        ]
    )
    
    if "Auto-Scout" in tactical_feature:
        st.markdown("### 🔍 Auto-Scout Report Generator")
        st.info("⭐ KILLER FEATURE: Genera report scouting completi in stile NBA usando Groq AI")
        
        opponent_team_name = st.text_input("Nome Squadra Avversaria", "Lakers")
        
        # Generate mock opponent stats
        if st.button("📊 Genera Scout Report Completo", type="primary"):
            with st.spinner("Analisi tattica in corso + Generazione report Groq..."):
                
                # Generate opponent stats
                opponent_stats = simulate_opponent_stats()
                
                # Use team tracking data as opponent data (for demo)
                opponent_tracking = st.session_state.tracking_data if st.session_state.tracking_data else {}
                
                # Generate full report
                scout_gen = ScoutReportGenerator()
                report = scout_gen.generate_full_report(
                    opponent_team_name,
                    opponent_tracking,
                    opponent_stats,
                    language='it'
                )
                
                st.success("✅ Scout Report generato!")
                
                # Display stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Punti/Game", opponent_stats['points_per_game'])
                    st.metric("3PT%", f"{opponent_stats['three_pt_pct']}%")
                with col2:
                    st.metric("Assist/Game", opponent_stats['assists_per_game'])
                    st.metric("Pace", opponent_stats['pace'])
                with col3:
                    st.metric("Def Rating", opponent_stats['def_rating'])
                    st.metric("Perse/Game", opponent_stats['turnovers_per_game'])
                
                # Pattern summary
                if report['patterns']:
                    st.markdown("#### 📊 Pattern Tattici Identificati")
                    patterns_df = pd.DataFrame(report['patterns'])
                    st.dataframe(patterns_df)
                
                # Strengths & Weaknesses
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### ✅ Punti di Forza")
                    for strength in report['strengths']:
                        st.success(strength)
                
                with col2:
                    st.markdown("#### ⚠️ Punti Deboli")
                    for weakness in report['weaknesses']:
                        st.warning(weakness)
                
                # GROQ NATURAL LANGUAGE REPORT (NBA-STYLE)
                st.markdown("---")
                st.markdown("### 📝 Scout Report Completo (Groq AI - Stile NBA)")
                st.markdown(report['report_text'])
                
                # Game plan recommendations
                st.markdown("---")
                st.markdown("#### 🎯 Piano di Gioco Raccomandato")
                for rec in report['recommendations']:
                    st.write(rec)
    
    elif "Lineup Optimizer" in tactical_feature:
        st.markdown("### ⚔️ Lineup Optimizer")
        
        if not st.session_state.tracking_data or len(st.session_state.tracking_data) < 5:
            st.warning("⚠️ Servono almeno 5 giocatori con dati tracking")
        else:
            # Calculate player ratings
            st.markdown("#### 1️⃣ Calcola Rating Giocatori")
            
            if st.button("📊 Calcola Ratings", type="primary"):
                # Mock stats for demo
                player_stats = {}
                for player_id in st.session_state.tracking_data.keys():
                    player_stats[player_id] = {
                        'points': np.random.randint(8, 22),
                        'assists': np.random.randint(2, 8),
                        'rebounds': np.random.randint(3, 10),
                        'steals': np.random.randint(0, 3),
                        'blocks': np.random.randint(0, 2),
                        'turnovers': np.random.randint(1, 4),
                        'minutes': np.random.randint(25, 35)
                    }
                
                ratings = st.session_state.lineup_optimizer.calculate_player_ratings(player_stats)
                
                st.success("✅ Ratings calcolati!")
                
                ratings_df = pd.DataFrame([
                    {'Player': pid, 'Rating': rating} 
                    for pid, rating in sorted(ratings.items(), key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(ratings_df)
            
            # Calculate chemistry
            st.markdown("#### 2️⃣ Calcola Chemistry Matrix")
            
            if st.button("🧪 Calcola Chemistry"):
                chemistry_matrix = st.session_state.lineup_optimizer.calculate_chemistry(
                    st.session_state.tracking_data
                )
                
                st.success("✅ Chemistry matrix calcolata!")
                st.dataframe(chemistry_matrix.style.background_gradient(cmap='RdYlGn'))
            
            # Optimize lineup
            st.markdown("#### 3️⃣ Ottimizza Lineup")
            
            lineup_size = st.slider("Dimensione Lineup", 3, 8, 5)
            
            if st.button("⚔️ Trova Lineup Ottimale", type="primary"):
                available_players = list(st.session_state.tracking_data.keys())
                
                result = st.session_state.lineup_optimizer.optimize_lineup(
                    available_players,
                    lineup_size=lineup_size
                )
                
                if 'error' in result:
                    st.error(result['error'])
                else:
                    st.success("✅ Lineup ottimale trovato!")
                    
                    st.metric("Overall Score", result['score'])
                    st.metric("Chemistry Score", result['chemistry_score'])
                    
                    st.markdown("#### 🏀 Lineup Raccomandato")
                    for i, detail in enumerate(result['details'], 1):
                        st.write(f"{i}. **{detail['player_id']}** - Rating: {detail['rating']:.1f}")
                    
                    st.info(result['recommendation'])
    
    elif "Pattern Recognition" in tactical_feature:
        st.markdown("### 📊 Tactical Pattern Recognition")
        
        if st.session_state.tracking_data:
            recognizer = TacticalPatternRecognizer()
            
            if st.button("🔍 Analizza Pattern Tattici", type="primary"):
                with st.spinner("Analisi pattern in corso..."):
                    # Simulate possession outcomes
                    n_possessions = 50
                    outcomes = np.random.choice(['score', 'miss', 'turnover'], n_possessions, p=[0.45, 0.45, 0.10])
                    
                    patterns = recognizer.analyze_team_patterns(st.session_state.tracking_data, outcomes)
                    pattern_summary = recognizer.get_pattern_summary()
                    
                    st.success("✅ Analisi completata!")
                    
                    # Display patterns
                    if pattern_summary:
                        patterns_df = pd.DataFrame(pattern_summary)
                        st.dataframe(patterns_df)
                        
                        # Chart
                        fig = px.bar(patterns_df, x='pattern', y='frequency', 
                                    color='success_rate',
                                    title='Pattern Frequency & Success Rate')
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Carica dati tracking prima")
    
    elif "Game Assistant" in tactical_feature:
        st.markdown("### 💬 Real-Time Game Assistant")
        st.info("🎮 GAME CHANGER: Chat con AI durante la partita per suggerimenti tattici real-time")
        
        # Update game state
        st.markdown("#### ⚙️ Game State Attuale")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            score_us = st.number_input("Score Nostro", 0, 150, 75)
            score_opp = st.number_input("Score Avversario", 0, 150, 78)
        
        with col2:
            quarter = st.selectbox("Periodo", [1, 2, 3, 4], index=3)
            time_remaining = st.text_input("Tempo Rimanente", "2:30")
        
        with col3:
            timeouts = st.number_input("Timeout Rimasti", 0, 7, 3)
        
        st.session_state.game_assistant.update_game_state(
            score_us=score_us,
            score_opponent=score_opp,
            quarter=quarter,
            time_remaining=time_remaining,
            timeouts_remaining=timeouts
        )
        
        # Chat interface
        st.markdown("#### 💬 Chiedi all'Assistente AI")
        
        question = st.text_input(
            "La tua domanda:",
            placeholder="Es: Siamo sotto 3 punti, ultimi 2 minuti, cosa faccio?"
        )
        
        if st.button("🤖 Chiedi a Groq AI", type="primary") and question:
            with st.spinner("Groq sta pensando..."):
                response = st.session_state.game_assistant.ask_assistant(question, 'it')
                
                st.markdown("#### 🎯 Risposta AI:")
                st.success(response['answer'])
        
        # Timeout recommendation
        st.markdown("---")
        if st.button("⏱️ Valuta se chiamare Timeout"):
            recommendations = st.session_state.game_assistant.get_timeout_recommendation()
            
            for rec in recommendations:
                if "TIMEOUT CONSIGLIATO" in rec:
                    st.error(rec)
                elif "TIMEOUT STRATEGICO" in rec:
                    st.warning(rec)
                else:
                    st.success(rec)

# =================================================================
# TAB 6: ANALYTICS & REPORTS
# =================================================================

with tab6:
    st.header("📈 Analytics & Report Generation")
    
    if not st.session_state.tracking_data:
        st.warning("⚠️ Carica prima dati in Tab 1")
        st.stop()
    
    st.markdown("### 📊 Statistiche Team")
    
    # Team overview
    total_distance = 0
    total_players = len(st.session_state.tracking_data)
    
    for player_data in st.session_state.tracking_data.values():
        total_distance += calculate_distance(player_data)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Giocatori Totali", total_players)
    col2.metric("Distanza Totale Team", f"{total_distance:.1f} m")
    col3.metric("Media per Giocatore", f"{total_distance/total_players:.1f} m")
    col4.metric("Players con Physical Data", len(st.session_state.physical_profiles))
    
    st.markdown("---")
    
    # Export options
    st.markdown("### 📥 Export Dati & Report")
    
    export_type = st.selectbox(
        "Tipo Export",
        [
            "📊 Excel Report Completo",
            "📄 PDF Summary Report",
            "📁 CSV Physical Data",
            "🤖 Groq AI Summary Report"
        ]
    )
    
    if "Excel" in export_type:
        if st.button("📊 Genera Excel Report", type="primary"):
            if st.session_state.physical_profiles:
                excel_data = export_physical_data_excel(st.session_state.physical_profiles)
                st.download_button(
                    "💾 Download Excel",
                    excel_data,
                    "coachtrack_complete_report.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("⚠️ Nessun dato fisico da esportare")
    
    elif "PDF" in export_type:
        st.info("📄 PDF generation in sviluppo - Richiede ReportLab configuration")
    
    elif "CSV" in export_type:
        if st.button("📁 Esporta CSV", type="primary"):
            if st.session_state.physical_profiles:
                df = pd.DataFrame([
                    {'Player': pid, **data} 
                    for pid, data in st.session_state.physical_profiles.items()
                ])
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 Download CSV",
                    csv,
                    "physical_data.csv",
                    "text/csv"
                )
    
    elif "Groq AI" in export_type:
        st.markdown("### 🤖 Groq AI Summary Report Generator")
        
        if st.button("🚀 Genera Report AI Completo", type="primary"):
            with st.spinner("Groq sta generando report completo..."):
                
                # Collect all data for summary
                summary_data = {
                    'total_players': len(st.session_state.tracking_data),
                    'players_with_physical': len(st.session_state.physical_profiles),
                    'total_distance': total_distance,
                    'avg_distance': total_distance / total_players if total_players > 0 else 0
                }
                
                # Create comprehensive summary prompt
                prompt = f"""Genera un report di sintesi completo per il team.

DATI DISPONIBILI:
- Giocatori con tracking UWB: {summary_data['total_players']}
- Giocatori con dati fisici: {summary_data['players_with_physical']}
- Distanza totale percorsa: {summary_data['total_distance']:.1f} m
- Media per giocatore: {summary_data['avg_distance']:.1f} m

GENERA UN REPORT CHE INCLUDA:
1. Overview situazione team
2. Analisi performance fisica
3. Raccomandazioni per miglioramento
4. Next steps consigliati

Lunghezza: 400-500 parole, italiano professionale."""

                # Use Groq (simplified call)
                from groq_integration import client, DEFAULT_MODEL
                
                if client:
                    response = client.chat.completions.create(
                        model=DEFAULT_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=800
                    )
                    
                    report = response.choices[0].message.content
                    
                    st.markdown("---")
                    st.markdown("### 📝 Report Completo Generato da Groq AI")
                    st.markdown(report)
                    
                    # Download button
                    st.download_button(
                        "💾 Download Report (TXT)",
                        report,
                        "team_report_ai.txt",
                        "text/plain"
                    )
                else:
                    st.error("❌ Groq non configurato. Aggiungi GROQ_API_KEY nel .env")

st.markdown("---")
st.markdown("**CoachTrack Elite AI v3.0** © 2026 - Complete Edition with ML, Groq AI & Tactical Analytics")
st.caption("🚀 Powered by: Random Forest, Gradient Boosting, Groq Llama 3.1, Natural Language Generation")
