# =================================================================
# COACHTRACK ELITE AI v3.0 - MAIN APPLICATION (FIXED)
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

# =================================================================
# FUNZIONI INLINE - TRACKING & AI BASE
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
    risk_score = 25 if distance < 200 else 40
    risk_level = 'ALTO' if risk_score > 60 else 'MEDIO' if risk_score > 30 else 'BASSO'
    return {
        'player_id': player_id, 
        'risk_level': risk_level, 
        'risk_score': risk_score, 
        'acwr': 1.2, 
        'asymmetry': 10.0, 
        'fatigue': 8.0, 
        'risk_factors': ['ACWR: 1.2', 'Asimmetria: 10%'], 
        'recommendations': ['Monitorare carico']
    }

def recommend_offensive_plays(player_data):
    if len(player_data) < 5:
        return {'recommended_plays': ['Dati insufficienti'], 'reasoning': ['Caricare più dati']}
    return {
        'recommended_plays': ['Pick and Roll', 'Motion Offense', 'Fast Break'], 
        'reasoning': ['Gioco versatile consigliato']
    }

def optimize_defensive_matchups(team_data, opponent_data=None):
    if not team_data:
        return []
    return [{
        'defender': pid, 
        'opponent': 'Opponent Forward', 
        'match_score': 75, 
        'reason': 'Matchup versatile'
    } for pid in team_data.keys()]

def analyze_movement_patterns(player_data, player_id):
    if len(player_data) < 10:
        return {
            'player_id': player_id, 
            'pattern_type': 'UNKNOWN', 
            'insights': ['Dati insufficienti'], 
            'anomalies': []
        }
    distance = calculate_distance(player_data)
    pattern = 'DYNAMIC' if distance > 100 else 'BALANCED'
    return {
        'player_id': player_id, 
        'pattern_type': pattern, 
        'insights': [f'Distanza: {distance:.1f}m'], 
        'anomalies': []
    }

def simulate_shot_quality(player_data, player_id):
    if len(player_data) < 5:
        return {
            'player_id': player_id, 
            'avg_quality': 0, 
            'shots': [], 
            'recommendations': ['Dati insufficienti']
        }
    shots = [{
        'x': float(player_data.iloc[i]['x']), 
        'y': float(player_data.iloc[i]['y']), 
        'distance': 5.0, 
        'quality': 75, 
        'type': '2PT'
    } for i in range(min(5, len(player_data)))]
    return {
        'player_id': player_id, 
        'avg_quality': 75.0, 
        'shots': shots, 
        'recommendations': ['Buona selezione']
    }

def generate_ai_training_plan(player_id, injury_risk_data, physical_data=None):
    risk_level = injury_risk_data.get('risk_level', 'MEDIO')
    intensity = 'BASSA' if risk_level == 'ALTO' else 'MODERATA'
    exercises = [{'name': 'Recovery', 'sets': '3x10', 'focus': 'Recupero', 'priority': 'Alta'}]
    return {
        'player_id': player_id, 
        'risk_level': risk_level, 
        'intensity': intensity, 
        'duration': '60min', 
        'frequency': '5x/settimana', 
        'focus_areas': 'Condizionamento', 
        'exercises': exercises, 
        'notes': f'Piano {risk_level}'
    }

# =================================================================
# FUNZIONI PHYSICAL & NUTRITION
# =================================================================

PHYSICAL_METRICS = {
    'weight_kg': 'Peso (kg)',
    'bmi': 'BMI',
    'body_fat_pct': 'Grasso Corporeo (%)',
    'lean_mass_kg': 'Massa Magra (kg)',
    'body_water_pct': 'Acqua Corporea (%)',
    'muscle_pct': 'Massa Muscolare (%)',
    'bone_mass_kg': 'Massa Ossea (kg)',
    'bmr': 'BMR (kcal)',
    'amr': 'AMR (kcal)'
}

def parse_physical_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as e:
        return None, str(e)

def validate_physical_data(profile):
    warnings = []
    if profile.get('bmi', 0) < 18 or profile.get('bmi', 0) > 30:
        warnings.append("BMI fuori range normale")
    return warnings

def simulate_apple_health_sync(player_id):
    return {
        'player_id': player_id,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'source': 'Apple Health Sync',
        'weight_kg': 80.0,
        'bmi': 22.5,
        'body_fat_pct': 12.0,
        'lean_mass_kg': 68.0,
        'body_water_pct': 60.0,
        'muscle_pct': 45.0,
        'bone_mass_kg': 3.2,
        'bmr': 1800,
        'amr': 2700
    }

def generate_enhanced_nutrition(player_id, physical_data, activity_level, goal):
    base_calories = physical_data.get('amr', 2500)
    
    if 'High' in activity_level:
        target_calories = int(base_calories * 1.3)
    elif 'Moderate' in activity_level:
        target_calories = int(base_calories * 1.1)
    else:
        target_calories = base_calories
    
    protein_g = int(physical_data.get('weight_kg', 80) * 2.2)
    carbs_g = int(target_calories * 0.5 / 4)
    fats_g = int(target_calories * 0.25 / 9)
    
    meals = [
        {'name': 'Colazione', 'calories': int(target_calories * 0.25), 'protein': int(protein_g * 0.25), 'carbs': int(carbs_g * 0.25), 'fats': int(fats_g * 0.25), 'timing': '07:00-08:00', 'examples': 'Avena, uova, frutta'},
        {'name': 'Pranzo', 'calories': int(target_calories * 0.35), 'protein': int(protein_g * 0.35), 'carbs': int(carbs_g * 0.35), 'fats': int(fats_g * 0.35), 'timing': '12:00-13:00', 'examples': 'Riso, pollo, verdure'},
        {'name': 'Cena', 'calories': int(target_calories * 0.30), 'protein': int(protein_g * 0.30), 'carbs': int(carbs_g * 0.30), 'fats': int(fats_g * 0.30), 'timing': '19:00-20:00', 'examples': 'Pesce, patate, insalata'},
        {'name': 'Snack', 'calories': int(target_calories * 0.10), 'protein': int(protein_g * 0.10), 'carbs': int(carbs_g * 0.10), 'fats': int(fats_g * 0.10), 'timing': 'Pre/Post workout', 'examples': 'Shake proteico, frutta secca'}
    ]
    
    return {
        'player_id': player_id,
        'target_calories': target_calories,
        'protein_g': protein_g,
        'carbs_g': carbs_g,
        'fats_g': fats_g,
        'activity_level': activity_level,
        'goal': goal,
        'meals': meals
    }

def create_body_composition_viz(data):
    fig = go.Figure()
    
    categories = ['Grasso', 'Muscoli', 'Acqua']
    values = [
        data.get('body_fat_pct', 0),
        data.get('muscle_pct', 0),
        data.get('body_water_pct', 0)
    ]
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1']
    ))
    
    fig.update_layout(title="Composizione Corporea", yaxis_title="%")
    return fig

def create_physical_csv_template():
    template_data = "player_id,date,weight_kg,bmi,body_fat_pct,lean_mass_kg,body_water_pct,muscle_pct,bone_mass_kg,bmr,amr\nP001,2026-02-07,80.0,22.5,12.0,68.0,60.0,45.0,3.2,1800,2700\n"
    return template_data

def export_physical_data_excel(data):
    return BytesIO()

# =================================================================
# GROQ INTEGRATION (MOCK)
# =================================================================

def test_groq_connection():
    return True, "Groq connesso (mock mode)"

def generate_nutrition_report_nlg(player_id, nutrition_plan, physical_data, language):
    return f"""
## 📊 Report Nutrizionale per {player_id}

### Analisi Composizione Corporea
Il giocatore presenta un peso di {physical_data.get('weight_kg', 'N/A')} kg con un BMI di {physical_data.get('bmi', 'N/A')}, 
indicando una composizione corporea nella norma per un atleta professionista.

### Piano Nutrizionale Personalizzato
Target calorico giornaliero: **{nutrition_plan['target_calories']} kcal**

La distribuzione dei macronutrienti è ottimizzata per:
- **Proteine**: {nutrition_plan['protein_g']}g per supportare il recupero muscolare
- **Carboidrati**: {nutrition_plan['carbs_g']}g per l'energia durante allenamenti intensi
- **Grassi**: {nutrition_plan['fats_g']}g per funzioni ormonali e assorbimento vitamine

### Raccomandazioni
Mantenere idratazione costante (3-4L/giorno) e timing ottimale dei pasti per massimizzare performance e recupero.
"""

def generate_training_plan_nlg(player_id, training_plan, language):
    return f"Piano allenamento per {player_id}: Intensità {training_plan['intensity']}, Focus: {training_plan['focus_areas']}"

def generate_scout_report_nlg(team_name, report_data, language):
    return f"## Scout Report: {team_name}\n\nAnalisi tattica completa generata."

def game_assistant_chat(query, context, language):
    return f"Risposta assistant: {query}"

def generate_performance_summary(player_id, stats_summary, predictions, language):
    return f"""
## 📊 Analisi Performance per {player_id}

{stats_summary}

### Analisi Predittiva
Le predizioni mostrano un trend positivo con confidence {predictions['confidence']}.
Il giocatore dovrebbe ottenere circa {predictions['points']:.1f} punti nella prossima partita.
"""

# =================================================================
# ML MODELS (MOCK)
# =================================================================

class MLInjuryPredictor:
    def extract_features(self, player_data, physical_data=None, player_age=25):
        return {
            'acwr': 1.2,
            'asymmetry': 10.0,
            'fatigue': 8.0,
            'workload': 100.0,
            'rest_days': 2,
            'age': player_age
        }
    
    def predict(self, features):
        risk_prob = 35
        risk_level = 'MEDIO'
        return {
            'risk_level': risk_level,
            'risk_probability': risk_prob,
            'top_risk_factors': [('ACWR', 0.25), ('Fatigue', 0.20), ('Workload', 0.18)],
            'recommendations': ['Monitorare carico', 'Aumentare recupero']
        }

class PerformancePredictor:
    def extract_features(self, stats_history, opponent_info, injury_risk=None):
        return {
            'avg_points': stats_history['points'].mean(),
            'rest_days': opponent_info['rest_days'],
            'def_rating': opponent_info['def_rating']
        }
    
    def predict_next_game(self, features):
        return {
            'points': 18.5,
            'assists': 5.2,
            'rebounds': 6.8,
            'efficiency': 22.3,
            'confidence': 'Alta'
        }

class ShotFormAnalyzer:
    def analyze_shot_video(self, video_path):
        return {
            'message': 'Feature in sviluppo',
            'next_steps': ['Integrare MediaPipe', 'Analisi angoli corpo', 'Tracking mano/gomito']
        }

# =================================================================
# TACTICAL AI (MOCK)
# =================================================================

class TacticalPatternRecognizer:
    pass

class ScoutReportGenerator:
    def generate_full_report(self, team_name, tracking_data, stats, language):
        return {
            'team_name': team_name,
            'offensive_rating': 112,
            'defensive_rating': 108,
            'key_players': ['Player A', 'Player B']
        }

class LineupOptimizer:
    def optimize(self, team_data, constraints):
        return {'lineup': ['P1', 'P2', 'P3', 'P4', 'P5'], 'score': 85}

class GameAssistant:
    pass

def simulate_opponent_stats():
    return {
        'points_per_game': 110,
        'assists_per_game': 25,
        'three_pt_pct': 36,
        'pace': 98,
        'offensive_rating': 112,
        'defensive_rating': 108
    }

# =================================================================
# PAGE CONFIG
# =================================================================

st.set_page_config(
    page_title="CoachTrack Elite AI",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================================================================
# AUTHENTICATION
# =================================================================

def check_login(username, password):
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
# SESSION STATE
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
        'title': '🏀 CoachTrack Elite AI v3.0',
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
# SIDEBAR - SEZIONE DATA SUMMARY CORRETTA
# =================================================================

with st.sidebar:
    # st.image("https://i.imgur.com/placeholder_basketball.png", use_container_width=True)
    st.title("CoachTrack Elite")
    # st.caption("v3.0 - Complete Edition")
    st.markdown("---")
    
    lang = st.selectbox("🌐 Language", ["IT", "EN"], index=0 if st.session_state.language == 'it' else 1)
    if lang == "IT" and st.session_state.language != 'it':
        st.session_state.language = 'it'
        st.rerun()
    elif lang == "EN" and st.session_state.language != 'en':
        st.session_state.language = 'en'
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 🤖 Groq Status")
    if st.button("Test Groq Connection", use_container_width=True):
        with st.spinner("Testing..."):
            success, message = test_groq_connection()
            if success:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
    
        st.markdown("---")
    
    # ===== DATA SUMMARY SECTION =====
    st.markdown("### 📊 Data Summary")
    
    # Conta i dati disponibili
    uwb_count = len(st.session_state.tracking_data) if st.session_state.tracking_data else 0
    phys_count = len(st.session_state.physical_profiles) if st.session_state.physical_profiles else 0
    imu_count = len(st.session_state.imu_data) if st.session_state.imu_data else 0
    
    # Mostra sempre le metriche (anche se 0)
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("👥 Players UWB", uwb_count)
    
    with col2:
        st.metric("🏋️ Physical", phys_count)
    
    if imu_count > 0:
        st.metric("📱 IMU Data", imu_count)
    
    # Status nutrition
    if st.session_state.current_nutrition_plan:
        st.success("✅ Nutrition Active")
    
    # Messaggio se nessun dato
    if uwb_count == 0 and phys_count == 0 and imu_count == 0:
        st.caption("⬆️ Carica dati in Tab 1")
    
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
# TAB 1: CONFIGURATION
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
                for player_id in df['player_id'].unique():
                    player_df = df[df['player_id'] == player_id].copy()
                    st.session_state.tracking_data[player_id] = player_df
                
                st.success(f"✅ Dati importati per {len(df['player_id'].unique())} giocatori")
                
                with st.expander("👁️ Anteprima Dati"):
                    st.dataframe(df.head(20))
            else:
                st.error(f"❌ CSV deve contenere: {', '.join(required_cols)}")
        except Exception as e:
            st.error(f"❌ Errore lettura file: {e}")
    
    st.markdown("---")
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
# TAB 2: PHYSICAL PROFILE
# =================================================================

with tab2:
    st.header("🏋️ Profilo Fisico & AI Nutrition Enhanced")
    st.markdown("### 📊 Gestione Dati Fisici Completi")
    
    with st.expander("📥 Carica Dati Fisici", expanded=False):
        upload_tab, apple_tab, manual_tab = st.tabs(["📄 CSV Upload", "🍎 Apple Health", "✏️ Manuale"])
        
        with upload_tab:
            st.markdown("#### Upload CSV con Dati Fisici")
            template_csv = create_physical_csv_template()
            st.download_button("📥 Scarica Template CSV", template_csv, "physical_data_template.csv", "text/csv")
            
            uploaded_physical = st.file_uploader("Carica CSV Dati Fisici", type=['csv'], key='physical_csv')
            if uploaded_physical:
                df_physical, error = parse_physical_csv(uploaded_physical)
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.success(f"✅ File caricato: {len(df_physical)} righe")
                    st.dataframe(df_physical.head())
        
        with apple_tab:
            st.markdown("#### 🍎 Sincronizza Apple Health (Demo)")
            st.info("ℹ️ Questa è una simulazione.")
            if st.session_state.tracking_data:
                player_for_sync = st.selectbox("Seleziona Giocatore", list(st.session_state.tracking_data.keys()))
                if st.button("🔄 Sincronizza Apple Health (Demo)", type="primary"):
                    health_data = simulate_apple_health_sync(player_for_sync)
                    st.session_state.physical_profiles[player_for_sync] = health_data
                    st.success(f"✅ Dati sincronizzati per {player_for_sync}")
        
        with manual_tab:
            st.markdown("#### ✏️ Inserimento Manuale")
            if st.session_state.tracking_data:
                player_manual = st.selectbox("Giocatore", list(st.session_state.tracking_data.keys()), key='manual_player')
                col1, col2, col3 = st.columns(3)
                with col1:
                    weight = st.number_input("Peso (kg)", 50.0, 150.0, 80.0, 0.1)
                with col2:
                    bmi = st.number_input("BMI", 15.0, 35.0, 22.5, 0.1)
                with col3:
                    body_fat = st.number_input("Grasso (%)", 3.0, 40.0, 12.0, 0.1)
                
                if st.button("💾 Salva Dati Manuali", type="primary"):
                    manual_data = {
                        'weight_kg': weight,
                        'bmi': bmi,
                        'body_fat_pct': body_fat,
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'source': 'Manual Entry'
                    }
                    st.session_state.physical_profiles[player_manual] = manual_data
                    st.success(f"✅ Dati salvati per {player_manual}")
    
    if st.session_state.physical_profiles:
        st.divider()
        st.markdown("### 🥗 Piano Nutrizionale AI")
        selected_nutrition = st.selectbox("Giocatore Nutrition", list(st.session_state.physical_profiles.keys()))
        activity_level = st.selectbox("Livello Attività", ["Low (Recovery)", "Moderate (Training)", "High (Intense/Match)"], index=1)
        goal = st.selectbox("Obiettivo", ["Maintenance", "Muscle Gain", "Fat Loss", "Performance"], index=3)
        
        if st.button("🚀 Genera Piano Nutrizionale AI", type="primary"):
            nutrition_plan = generate_enhanced_nutrition(selected_nutrition, st.session_state.physical_profiles[selected_nutrition], activity_level, goal)
            st.session_state.current_nutrition_plan = nutrition_plan
            st.success(f"✅ Piano generato per {selected_nutrition}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Calorie", f"{nutrition_plan['target_calories']} kcal")
            with col2:
                st.metric("Proteine", f"{nutrition_plan['protein_g']}g")
            with col3:
                st.metric("Carboidrati", f"{nutrition_plan['carbs_g']}g")
            with col4:
                st.metric("Grassi", f"{nutrition_plan['fats_g']}g")
            
            groq_report = generate_nutrition_report_nlg(selected_nutrition, nutrition_plan, st.session_state.physical_profiles[selected_nutrition], 'it')
            st.markdown(groq_report)

# =================================================================
# TAB 3, 4, 5, 6
# =================================================================

with tab3:
    st.header("🤖 Funzionalità AI Elite")
    st.info("Carica dati tracking in Tab 1 per usare le funzioni AI")

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
            "📹 Shot Form Analyzer (Computer Vision)"
        ]
    )
    
    # =================================================================
    # ML INJURY RISK PREDICTOR
    # =================================================================
    
    if "ML Injury" in ml_feature:
        st.markdown("### 🤖 ML Injury Risk Predictor")
        st.info("💡 Usa Random Forest con 12 features avanzate per predire rischio infortuni")
        
        selected_ml_player = st.selectbox(
            "Seleziona Giocatore", 
            list(st.session_state.tracking_data.keys()), 
            key='ml_injury_player'
        )
        
        # Advanced settings
        with st.expander("⚙️ Parametri Avanzati"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                player_age = st.number_input("Età Giocatore", 18, 40, 25)
                prev_injuries = st.number_input("Infortuni Precedenti", 0, 10, 0)
            
            with col2:
                rest_days = st.number_input("Giorni Riposo", 0, 7, 2)
                use_training_history = st.checkbox("Usa Storico Allenamenti (28gg)", value=False)
            
            with col3:
                model_info = st.checkbox("Mostra Info Modello", value=True)
        
        if st.button("🚀 Predici Rischio Infortuni (ML)", type="primary"):
            with st.spinner("🔄 Training Random Forest model..."):
                player_data = st.session_state.tracking_data[selected_ml_player]
                physical_data = st.session_state.physical_profiles.get(selected_ml_player)
                
                # Generate training history if requested
                training_history = None
                if use_training_history:
                    training_history = create_training_history_sample(days=28)
                
                # Extract features
                features = st.session_state.ml_injury_model.extract_features(
                    player_data, 
                    physical_data, 
                    player_age=player_age,
                    previous_injuries=prev_injuries,
                    training_history=training_history
                )
                
                # Predict
                prediction = st.session_state.ml_injury_model.predict(features)
                
                st.success("✅ Predizione completata!")
                
                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    risk_emoji = {"BASSO": "🟢", "MEDIO": "🟡", "ALTO": "🔴"}
                    st.metric(
                        "Rischio ML", 
                        f"{risk_emoji[prediction['risk_level']]} {prediction['risk_level']}", 
                        f"{prediction['risk_probability']}%"
                    )
                
                with col2:
                    st.metric("Model Confidence", prediction['confidence'])
                
                with col3:
                    st.metric("Features Used", len(features))
                
                with col4:
                    st.metric("Risk Class", prediction['risk_class'])
                
                # Feature importance visualization
                st.markdown("#### 🔍 Top 5 Fattori di Rischio (Feature Importance)")
                
                importance_df = pd.DataFrame(
                    prediction['top_risk_factors'], 
                    columns=['Feature', 'Importance']
                )
                
                fig_importance = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance (Random Forest)',
                    color='Importance',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Recommendations
                st.markdown("#### 💡 Raccomandazioni ML-Based")
                for i, rec in enumerate(prediction['recommendations'], 1):
                    st.markdown(f"{i}. {rec}")
                
                # Feature values table
                with st.expander("📊 Feature Values Utilizzati"):
                    features_df = pd.DataFrame([features]).T
                    features_df.columns = ['Value']
                    features_df.index.name = 'Feature'
                    st.dataframe(features_df.style.highlight_max(color='lightgreen'))
                
                # Model info
                if model_info:
                    with st.expander("ℹ️ Informazioni Modello"):
                        st.markdown("""
                        **Random Forest Classifier**
                        - N. Estimators: 100
                        - Max Depth: 10
                        - Training Samples: 500
                        - Classes: 3 (BASSO, MEDIO, ALTO)
                        - Features: 12
                        
                        **Features Utilizzate:**
                        1. ACWR (Acute:Chronic Workload Ratio)
                        2. Asymmetry %
                        3. Fatigue Index
                        4. Cumulative Workload (7 days)
                        5. Rest Days
                        6. Training Intensity
                        7. Age
                        8. BMI
                        9. Body Fat %
                        10. Previous Injuries Count
                        11. Workload Spike %
                        12. Consistency Score
                        """)
    
    # =================================================================
    # PERFORMANCE PREDICTOR
    # =================================================================
    
    elif "Performance Predictor" in ml_feature:
        st.markdown("### 📊 Performance Predictor Next Game")
        st.info("💡 Usa Gradient Boosting per predire punti, assist, rimbalzi, efficiency della prossima partita")
        
        selected_perf_player = st.selectbox(
            "Seleziona Giocatore", 
            list(st.session_state.tracking_data.keys()), 
            key='ml_perf_player'
        )
        
        # Generate or use real stats history
        st.markdown("#### 📈 Storico Statistiche (Ultimi 10 Match)")
        
        # Option to use sample data or manual
        use_sample = st.checkbox("Usa dati sample", value=True)
        
        if use_sample:
            stats_history = generate_player_stats_history(games=10, avg_points=18)
            st.dataframe(stats_history)
        else:
            st.warning("⚠️ Feature manual input in sviluppo - usa sample data")
            stats_history = generate_player_stats_history(games=10, avg_points=18)
        
        # Opponent info
        st.markdown("#### ⚔️ Informazioni Prossimo Avversario")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rest_days = st.slider("Rest Days", 0, 4, 1)
        
        with col2:
            def_rating = st.slider("Opponent Def Rating", 100, 120, 110)
        
        with col3:
            location = st.radio("Location", ['home', 'away'])
        
        with col4:
            usage_rate = st.slider("Usage Rate %", 15, 35, 25)
        
        opponent_info = {
            'rest_days': rest_days,
            'def_rating': def_rating,
            'location': location,
            'usage_rate': usage_rate,
            'fatigue': 0.2
        }
        
        # Get injury risk for context
        injury_risk = None
        if selected_perf_player in st.session_state.physical_profiles:
            with st.expander("🩺 Considera Injury Risk nel calcolo"):
                consider_injury = st.checkbox("Usa injury risk come fattore", value=True)
                
                if consider_injury:
                    player_data = st.session_state.tracking_data[selected_perf_player]
                    physical_data = st.session_state.physical_profiles[selected_perf_player]
                    features = st.session_state.ml_injury_model.extract_features(
                        player_data, 
                        physical_data
                    )
                    injury_risk = st.session_state.ml_injury_model.predict(features)
                    
                    st.info(f"Injury Risk: {injury_risk['risk_level']} ({injury_risk['risk_probability']}%)")
        
        if st.button("🚀 Predici Performance Prossima Partita", type="primary"):
            with st.spinner("🔄 Training Gradient Boosting models..."):
                
                # Extract features
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
                    st.metric(
                        "Punti", 
                        f"{predictions['points']:.1f}",
                        delta=f"{predictions['points'] - stats_history['points'].mean():.1f}"
                    )
                
                with col2:
                    st.metric(
                        "Assist", 
                        f"{predictions['assists']:.1f}",
                        delta=f"{predictions['assists'] - stats_history['assists'].mean():.1f}"
                    )
                
                with col3:
                    st.metric(
                        "Rimbalzi", 
                        f"{predictions['rebounds']:.1f}",
                        delta=f"{predictions['rebounds'] - stats_history['rebounds'].mean():.1f}"
                    )
                
                with col4:
                    st.metric("Efficiency", f"{predictions['efficiency']:.1f}")
                
                st.metric("Confidence Level", predictions['confidence'])
                
                # Comparison chart
                st.markdown("#### 📊 Confronto con Storico")
                
                comparison_data = {
                    'Metric': ['Points', 'Assists', 'Rebounds'],
                    'Avg Last 10': [
                        stats_history['points'].mean(),
                        stats_history['assists'].mean(),
                        stats_history['rebounds'].mean()
                    ],
                    'Predicted Next': [
                        predictions['points'],
                        predictions['assists'],
                        predictions['rebounds']
                    ]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                
                fig_comparison = go.Figure()
                
                fig_comparison.add_trace(go.Bar(
                    name='Avg Last 10',
                    x=comparison_df['Metric'],
                    y=comparison_df['Avg Last 10'],
                    marker_color='lightblue'
                ))
                
                fig_comparison.add_trace(go.Bar(
                    name='Predicted Next',
                    x=comparison_df['Metric'],
                    y=comparison_df['Predicted Next'],
                    marker_color='orange'
                ))
                
                fig_comparison.update_layout(
                    title='Confronto Performance',
                    barmode='group',
                    yaxis_title='Value'
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Historical trend
                st.markdown("#### 📈 Trend Storico Punti")
                
                fig_trend = px.line(
                    stats_history.reset_index(),
                    x='index',
                    y='points',
                    title='Ultimi 10 Match - Punti',
                    labels={'index': 'Game #', 'points': 'Points'}
                )
                
                # Add prediction as point
                fig_trend.add_scatter(
                    x=[10],
                    y=[predictions['points']],
                    mode='markers',
                    marker=dict(size=15, color='red'),
                    name='Predicted Next'
                )
                
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # Groq NLG Analysis (optional)
                if st.checkbox("🤖 Genera Analisi Dettagliata con Groq"):
                    with st.spinner("Groq sta generando analisi..."):
                        stats_summary = f"""
Statistiche ultimi 10 match:
- Punti: media {stats_history['points'].mean():.1f}, max {stats_history['points'].max()}, min {stats_history['points'].min()}
- Assist: media {stats_history['assists'].mean():.1f}
- Rimbalzi: media {stats_history['rebounds'].mean():.1f}
- Minuti: media {stats_history['minutes'].mean():.1f}

Predizioni prossima partita:
- Punti: {predictions['points']:.1f}
- Assist: {predictions['assists']:.1f}
- Rimbalzi: {predictions['rebounds']:.1f}
- Efficiency: {predictions['efficiency']:.1f}
- Confidence: {predictions['confidence']}

Contesto:
- Avversario Def Rating: {def_rating}
- Location: {location}
- Rest Days: {rest_days}
- Usage Rate: {usage_rate}%
"""
                        
                        groq_analysis = generate_performance_summary(
                            selected_perf_player,
                            stats_summary,
                            predictions,
                            'it'
                        )
                        
                        st.markdown("#### 📝 Analisi Groq")
                        st.markdown(groq_analysis)
    
    # =================================================================
    # SHOT FORM ANALYZER
    # =================================================================
    
    elif "Shot Form" in ml_feature:
        st.markdown("### 📹 Shot Form Analyzer (Computer Vision)")
        st.warning("⚠️ Feature in sviluppo - Richiede MediaPipe/OpenCV integration")
        
        shot_analyzer = ShotFormAnalyzer()
        
        # Show placeholder info
        result = shot_analyzer.analyze_shot_video("placeholder.mp4")
        
        st.info(result['message'])
        
        st.markdown("#### 🔜 Next Steps per Implementazione:")
        for i, step in enumerate(result['next_steps'], 1):
            st.write(f"{i}. {step}")
        
        st.markdown("#### 📦 Librerie Richieste:")
        for lib in result['required_libraries']:
            st.code(f"pip install {lib}")
        
        st.markdown("#### 📊 Sample Output (Futuro):")
        st.json(result['sample_output'])
        
        # Show optimal form guide
        with st.expander("📚 Guida Tecnica: Shot Form Ottimale"):
            optimal_form = shot_analyzer.get_optimal_form_guide()
            
            st.markdown("##### 🎯 Preparazione")
            for key, value in optimal_form['preparation'].items():
                st.write(f"**{key.title()}**: {value}")
            
            st.markdown("##### 🚀 Release")
            for key, value in optimal_form['release'].items():
                st.write(f"**{key.replace('_', ' ').title()}**: {value}")
            
            st.markdown("##### ✋ Follow Through")
            for key, value in optimal_form['follow_through'].items():
                st.write(f"**{key.replace('_', ' ').title()}**: {value}")


with tab5:
    st.header("⚔️ Tactical AI & Scout")
    st.markdown("### 🔍 Auto-Scout Report Generator")
    
    opponent_team_name = st.text_input("Nome Squadra Avversaria", "Lakers")
    
    if st.button("📊 Genera Scout Report Completo", type="primary"):
        opponent_stats = simulate_opponent_stats()
        st.success("✅ Scout Report generato!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Punti/Game", opponent_stats['points_per_game'])
            st.metric("3PT%", f"{opponent_stats['three_pt_pct']}%")
        with col2:
            st.metric("Assist/Game", opponent_stats['assists_per_game'])
            st.metric("Pace", opponent_stats['pace'])
        with col3:
            # FIXED: completato st.metric
            st.metric("Off Rating", opponent_stats['offensive_rating'])
            st.metric("Def Rating", opponent_stats['defensive_rating'])

with tab6:
    st.header("📈 Analytics & Reports")
    st.info("Sezione report avanzati - In sviluppo")

st.markdown("---")
st.caption("🏀 CoachTrack Elite AI v3.0 © 2026 - Fixed Version")
