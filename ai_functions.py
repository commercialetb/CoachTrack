"""
AI Functions Module - CoachTrack Elite AI v3.0
Base AI-powered analysis functions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =================================================================
# CORE CALCULATION FUNCTIONS
# =================================================================

def calculate_distance(df):
    """Calculate total distance traveled from tracking data"""
    if len(df) < 2:
        return 0.0
    
    try:
        dx = np.diff(df['x'].values)
        dy = np.diff(df['y'].values)
        distances = np.sqrt(dx**2 + dy**2)
        return float(np.sum(distances))
    except Exception as e:
        print(f"Error calculating distance: {e}")
        return 0.0

def calculate_speed(df):
    """Calculate speed in km/h and add to dataframe"""
    if len(df) < 2:
        return df
    
    try:
        if 'speed_kmh_calc' not in df.columns:
            dx = np.diff(df['x'].values)
            dy = np.diff(df['y'].values)
            dt = np.diff(df['timestamp'].values).astype(float) / 1000.0
            dt[dt == 0] = 0.001
            
            distances = np.sqrt(dx**2 + dy**2)
            speeds = (distances / dt) * 3.6
            
            df = df.copy()
            df.loc[df.index[1:], 'speed_kmh_calc'] = speeds
            df.loc[df.index[0], 'speed_kmh_calc'] = 0.0
        
        return df
    except Exception as e:
        print(f"Error calculating speed: {e}")
        return df

def detect_jumps_imu(df, threshold_g=1.5):
    """Detect jumps from IMU acceleration data"""
    if 'az' not in df.columns or len(df) == 0:
        return []
    
    try:
        jumps = []
        az = df['az'].values
        timestamps = df['timestamp'].values
        
        in_jump = False
        jump_start = None
        jump_peak_g = 0
        
        for i in range(len(az)):
            if az[i] > threshold_g and not in_jump:
                in_jump = True
                jump_start = i
                jump_peak_g = az[i]
            elif in_jump:
                if az[i] > jump_peak_g:
                    jump_peak_g = az[i]
                if az[i] < 0.5 or i == len(az) - 1:
                    jump_duration_ms = (timestamps[i] - timestamps[jump_start]) if i > jump_start else 0
                    estimated_height_cm = ((jump_peak_g - 1.0) * 9.81 * (jump_duration_ms/1000)**2 / 2) * 100
                    estimated_height_cm = max(0, min(estimated_height_cm, 120))
                    
                    jumps.append({
                        'timestamp': int(timestamps[jump_start]),
                        'peak_g': round(float(jump_peak_g), 2),
                        'duration_ms': int(jump_duration_ms),
                        'estimated_height_cm': round(float(estimated_height_cm), 1)
                    })
                    in_jump = False
        
        return jumps
    except Exception as e:
        print(f"Error detecting jumps: {e}")
        return []

# =================================================================
# INJURY RISK PREDICTION (BASE VERSION)
# =================================================================

def predict_injury_risk(player_data, player_id):
    """
    Predict injury risk based on workload metrics
    Returns: dict with risk_level, risk_score, and recommendations
    """
    try:
        if len(player_data) < 10:
            return {
                'player_id': player_id,
                'risk_level': 'UNKNOWN',
                'risk_score': 0,
                'acwr': 1.0,
                'asymmetry': 0,
                'fatigue': 0,
                'risk_factors': ['Dati insufficienti per analisi (min 10 righe)'],
                'recommendations': ['Raccogliere più dati tracking']
            }
        
        # Calculate ACWR (Acute:Chronic Workload Ratio)
        total_rows = len(player_data)
        recent_window = max(1, int(total_rows * 0.15))
        chronic_window = total_rows
        
        recent_distance = calculate_distance(player_data.tail(recent_window))
        chronic_distance = calculate_distance(player_data) / chronic_window * recent_window
        
        acwr = recent_distance / chronic_distance if chronic_distance > 0 else 1.0
        
        # Calculate asymmetry
        if 'dx' in player_data.columns:
            left_moves = len(player_data[player_data['dx'] < -0.5])
            right_moves = len(player_data[player_data['dx'] > 0.5])
            total_lateral = left_moves + right_moves
            asymmetry = abs(left_moves - right_moves) / total_lateral * 100 if total_lateral > 0 else 0
        else:
            player_data_copy = player_data.copy()
            player_data_copy['dx'] = player_data_copy['x'].diff()
            left_moves = len(player_data_copy[player_data_copy['dx'] < -0.5])
            right_moves = len(player_data_copy[player_data_copy['dx'] > 0.5])
            total_lateral = left_moves + right_moves
            asymmetry = abs(left_moves - right_moves) / total_lateral * 100 if total_lateral > 0 else 0
        
        # Calculate fatigue index
        player_data_with_speed = calculate_speed(player_data.copy())
        if 'speed_kmh_calc' in player_data_with_speed.columns:
            first_quarter = player_data_with_speed.head(len(player_data_with_speed)//4)
            last_quarter = player_data_with_speed.tail(len(player_data_with_speed)//4)
            first_q_speed = first_quarter['speed_kmh_calc'].mean()
            last_q_speed = last_quarter['speed_kmh_calc'].mean()
            fatigue = (first_q_speed - last_q_speed) / first_q_speed if first_q_speed > 0 else 0
            fatigue = max(0, fatigue) * 100
        else:
            fatigue = 0
        
        # Calculate risk score
        risk_score = 0
        risk_factors = []
        
        if acwr > 1.5:
            risk_score += 30
            risk_factors.append(f"ACWR elevato ({acwr:.2f}) - carico acuto troppo alto")
        elif acwr < 0.8:
            risk_score += 15
            risk_factors.append(f"ACWR basso ({acwr:.2f}) - deconditioning risk")
        
        if asymmetry > 15:
            risk_score += 25
            risk_factors.append(f"Asimmetria elevata ({asymmetry:.1f}%) - rischio sovraccarico unilaterale")
        
        if fatigue > 15:
            risk_score += 20
            risk_factors.append(f"Indice fatica alto ({fatigue:.1f}%) - recupero insufficiente")
        
        # Determine risk level
        if risk_score >= 60:
            risk_level = "ALTO"
        elif risk_score >= 30:
            risk_level = "MEDIO"
        else:
            risk_level = "BASSO"
        
        # Generate recommendations
        recommendations = []
        if risk_level == "ALTO":
            recommendations.append("🚨 RIPOSO IMMEDIATO: 48-72h senza attività intensa")
            recommendations.append("👨‍⚕️ Valutazione medica/fisioterapista consigliata")
        elif risk_level == "MEDIO":
            recommendations.append("⚠️ Ridurre intensità allenamento 20-30%")
            recommendations.append("🔄 Aumentare recupero attivo")
        else:
            recommendations.append("✅ Continuare monitoraggio regolare")
        
        if asymmetry > 15:
            recommendations.append("⚖️ Esercizi di equilibrio e rinforzo unilaterale")
        
        if fatigue > 15:
            recommendations.append("😴 Ottimizzare riposo e nutrizione post-allenamento")
        
        if not risk_factors:
            risk_factors.append("Nessun fattore di rischio significativo identificato")
        
        return {
            'player_id': player_id,
            'risk_level': risk_level,
            'risk_score': int(risk_score),
            'acwr': round(float(acwr), 2),
            'asymmetry': round(float(asymmetry), 1),
            'fatigue': round(float(fatigue), 1),
            'risk_factors': risk_factors,
            'recommendations': recommendations
        }
    
    except Exception as e:
        print(f"Error in predict_injury_risk: {e}")
        return {
            'player_id': player_id,
            'risk_level': 'ERROR',
            'risk_score': 0,
            'acwr': 1.0,
            'asymmetry': 0,
            'fatigue': 0,
            'risk_factors': [f'Errore analisi: {str(e)}'],
            'recommendations': ['Verificare dati input']
        }

# =================================================================
# OFFENSIVE PLAY RECOMMENDER
# =================================================================

def recommend_offensive_plays(player_data):
    """
    Recommend offensive plays based on player movement patterns
    """
    try:
        if len(player_data) < 5:
            return {
                'recommended_plays': ['Dati insufficienti per raccomandazioni'],
                'reasoning': ['Caricare più dati tracking']
            }
        
        total_distance = calculate_distance(player_data)
        avg_x = player_data['x'].mean()
        avg_y = player_data['y'].mean()
        
        player_data_with_speed = calculate_speed(player_data.copy())
        avg_speed = player_data_with_speed['speed_kmh_calc'].mean() if 'speed_kmh_calc' in player_data_with_speed.columns else 10
        
        plays = []
        reasoning = []
        
        # High movement - transition plays
        if total_distance > 100:
            plays.append("🏃 Fast Break / Transizione")
            reasoning.append(f"Alta mobilità rilevata (distanza: {total_distance:.1f}m)")
        
        # Positioning analysis
        if avg_x < 15:
            plays.append("🎯 Post-Up / Inside Game")
            reasoning.append(f"Posizionamento vicino canestro (x: {avg_x:.1f})")
        elif avg_x > 20:
            plays.append("🏀 Pick & Roll esterno")
            plays.append("🎯 Spot-Up 3PT")
            reasoning.append(f"Posizionamento perimetrale (x: {avg_x:.1f})")
        
        # Speed analysis
        if avg_speed > 12:
            plays.append("⚡ Motion Offense")
            reasoning.append(f"Velocità media alta ({avg_speed:.1f} km/h)")
        else:
            plays.append("🎯 Half-Court Set Play")
            reasoning.append(f"Gioco posizionale ({avg_speed:.1f} km/h)")
        
        if not plays:
            plays = ["🏀 Offense standard", "⚔️ Iso 1vs1"]
            reasoning = ["Pattern neutrale - flessibilità tattica"]
        
        return {
            'recommended_plays': plays,
            'reasoning': reasoning
        }
    
    except Exception as e:
        print(f"Error in recommend_offensive_plays: {e}")
        return {
            'recommended_plays': ['Errore analisi'],
            'reasoning': [str(e)]
        }

# =================================================================
# DEFENSIVE MATCHUP OPTIMIZER
# =================================================================

def optimize_defensive_matchups(team_data, opponent_data=None):
    """
    Optimize defensive matchups based on player characteristics
    """
    try:
        if not team_
            return []
        
        matchups = []
        
        # Analyze each player
        for player_id, player_df in team_data.items():
            if len(player_df) < 5:
                continue
            
            player_df_with_speed = calculate_speed(player_df.copy())
            avg_speed = player_df_with_speed['speed_kmh_calc'].mean() if 'speed_kmh_calc' in player_df_with_speed.columns else 10
            total_distance = calculate_distance(player_df)
            
            # Determine defensive role
            if avg_speed > 13 and total_distance > 80:
                opponent_type = "Guardia veloce / Wing scorer"
                match_score = 85
            elif avg_speed < 10:
                opponent_type = "Centro / Post player"
                match_score = 80
            else:
                opponent_type = "Forward versatile"
                match_score = 75
            
            matchups.append({
                'defender': player_id,
                'opponent': opponent_type,
                'match_score': match_score,
                'reason': f"Speed: {avg_speed:.1f} km/h, Mobility: {total_distance:.1f}m"
            })
        
        return sorted(matchups, key=lambda x: x['match_score'], reverse=True)
    
    except Exception as e:
        print(f"Error in optimize_defensive_matchups: {e}")
        return []

# =================================================================
# MOVEMENT PATTERN ANALYZER
# =================================================================

def analyze_movement_patterns(player_data, player_id):
    """
    Analyze and classify player movement patterns
    """
    try:
        if len(player_data) < 10:
            return {
                'player_id': player_id,
                'pattern_type': 'UNKNOWN',
                'insights': ['Dati insufficienti'],
                'anomalies': []
            }
        
        total_distance = calculate_distance(player_data)
        player_data_with_speed = calculate_speed(player_data.copy())
        avg_speed = player_data_with_speed['speed_kmh_calc'].mean() if 'speed_kmh_calc' in player_data_with_speed.columns else 10
        max_speed = player_data_with_speed['speed_kmh_calc'].max() if 'speed_kmh_calc' in player_data_with_speed.columns else 15
        
        # Classify pattern
        if avg_speed > 15 and total_distance > 150:
            pattern_type = "HIGH MOBILITY - Transition Player"
        elif avg_speed > 12:
            pattern_type = "DYNAMIC - Motion Offense"
        elif avg_speed < 8:
            pattern_type = "STATIC - Post/Spot-Up Player"
        else:
            pattern_type = "BALANCED - Versatile Movement"
        
        # Generate insights
        insights = [
            f"Distanza totale: {total_distance:.1f}m",
            f"Velocità media: {avg_speed:.1f} km/h",
            f"Velocità massima: {max_speed:.1f} km/h"
        ]
        
        # Detect anomalies
        anomalies = []
        if max_speed > 25:
            anomalies.append(f"Picco velocità insolito: {max_speed:.1f} km/h (verificare dato)")
        
        if avg_speed < 5 and len(player_data) > 50:
            anomalies.append("Mobilità estremamente bassa - possibile inattività")
        
        return {
            'player_id': player_id,
            'pattern_type': pattern_type,
            'insights': insights,
            'anomalies': anomalies
        }
    
    except Exception as e:
        print(f"Error in analyze_movement_patterns: {e}")
        return {
            'player_id': player_id,
            'pattern_type': 'ERROR',
            'insights': [str(e)],
            'anomalies': []
        }

# =================================================================
# SHOT QUALITY SIMULATOR
# =================================================================

def simulate_shot_quality(player_data, player_id):
    """
    Simulate and analyze shot quality based on positioning
    """
    try:
        if len(player_data) < 5:
            return {
                'player_id': player_id,
                'avg_quality': 0,
                'shots': [],
                'recommendations': ['Dati insufficienti']
            }
        
        shots = []
        
        # Sample potential shot locations
        sample_indices = np.linspace(0, len(player_data)-1, min(10, len(player_data)), dtype=int)
        
        for idx in sample_indices:
            x = player_data.iloc[idx]['x']
            y = player_data.iloc[idx]['y']
            
            # Calculate distance from basket (assume basket at 0,0)
            distance_from_basket = np.sqrt(x**2 + y**2)
            
            # Quality based on distance
            if distance_from_basket < 3:
                quality = 95
                shot_type = "Layup"
            elif distance_from_basket < 6:
                quality = 75
                shot_type = "Close 2PT"
            elif distance_from_basket < 7:
                quality = 55
                shot_type = "Mid-range"
            else:
                quality = 40
                shot_type = "3PT"
            
            shots.append({
                'x': round(float(x), 2),
                'y': round(float(y), 2),
                'distance': round(float(distance_from_basket), 2),
                'quality': quality,
                'type': shot_type
            })
        
        avg_quality = np.mean([s['quality'] for s in shots]) if shots else 0
        
        # Recommendations
        recommendations = []
        if avg_quality > 70:
            recommendations.append("✅ Ottima selezione tiro - continuare posizionamento")
        elif avg_quality < 50:
            recommendations.append("⚠️ Migliorare selezione tiro - cercare posizioni migliori")
            recommendations.append("🎯 Focus su tiri vicino canestro")
        else:
            recommendations.append("👍 Selezione tiro accettabile")
        
        return {
            'player_id': player_id,
            'avg_quality': round(float(avg_quality), 1),
            'shots': shots,
            'recommendations': recommendations
        }
    
    except Exception as e:
        print(f"Error in simulate_shot_quality: {e}")
        return {
            'player_id': player_id,
            'avg_quality': 0,
            'shots': [],
            'recommendations': [f'Errore: {str(e)}']
        }

# =================================================================
# AI TRAINING PLAN GENERATOR
# =================================================================

def generate_ai_training_plan(player_id, injury_risk_data, physical_data=None):
    """
    Generate personalized AI training plan based on injury risk
    """
    try:
        risk_level = injury_risk_data.get('risk_level', 'MEDIO')
        risk_score = injury_risk_data.get('risk_score', 50)
        
        # Determine intensity
        if risk_level == "ALTO" or risk_score > 60:
            intensity = "BASSA (Recovery)"
            duration = "30-40 min"
            frequency = "3-4 sessioni/settimana"
            focus = "Recovery, mobilità, prevenzione"
        elif risk_level == "MEDIO" or risk_score > 30:
            intensity = "MODERATA"
            duration = "45-60 min"
            frequency = "4-5 sessioni/settimana"
            focus = "Condizionamento, forza, tecnica"
        else:
            intensity = "ALTA (Performance)"
            duration = "60-90 min"
            frequency = "5-6 sessioni/settimana"
            focus = "Performance, esplosività, endurance"
        
        # Generate exercises
        exercises = []
        
        if risk_level == "ALTO":
            exercises = [
                {'name': 'Mobility Drills', 'sets': '3x10', 'focus': 'Mobilità articolare', 'priority': 'Alta'},
                {'name': 'Light Cardio', 'sets': '20min', 'focus': 'Recupero attivo', 'priority': 'Alta'},
                {'name': 'Core Stability', 'sets': '3x15', 'focus': 'Stabilizzazione', 'priority': 'Media'},
                {'name': 'Stretching', 'sets': '15min', 'focus': 'Flessibilità', 'priority': 'Alta'}
            ]
        elif risk_level == "MEDIO":
            exercises = [
                {'name': 'Dynamic Warm-up', 'sets': '10min', 'focus': 'Attivazione', 'priority': 'Alta'},
                {'name': 'Strength Training', 'sets': '4x8-10', 'focus': 'Forza generale', 'priority': 'Alta'},
                {'name': 'Basketball Drills', 'sets': '30min', 'focus': 'Tecnica', 'priority': 'Media'},
                {'name': 'Conditioning', 'sets': '20min', 'focus': 'Aerobico', 'priority': 'Media'},
                {'name': 'Cool-down', 'sets': '10min', 'focus': 'Recupero', 'priority': 'Alta'}
            ]
        else:
            exercises = [
                {'name': 'Sport-Specific Warm-up', 'sets': '15min', 'focus': 'Preparazione', 'priority': 'Alta'},
                {'name': 'Plyometrics', 'sets': '4x10', 'focus': 'Esplosività', 'priority': 'Alta'},
                {'name': 'On-Court Training', 'sets': '45min', 'focus': 'Game situations', 'priority': 'Alta'},
                {'name': 'Speed/Agility', 'sets': '20min', 'focus': 'Velocità', 'priority': 'Media'},
                {'name': 'Scrimmage', 'sets': '5x5 15min', 'focus': 'Tattica', 'priority': 'Media'},
                {'name': 'Recovery Protocol', 'sets': '15min', 'focus': 'Defaticamento', 'priority': 'Alta'}
            ]
        
        return {
            'player_id': player_id,
            'risk_level': risk_level,
            'intensity': intensity,
            'duration': duration,
            'frequency': frequency,
            'focus_areas': focus,
            'exercises': exercises,
            'notes': f"Piano generato in base a risk level {risk_level} (score: {risk_score})"
        }
    
    except Exception as e:
        print(f"Error in generate_ai_training_plan: {e}")
        return {
            'player_id': player_id,
            'risk_level': 'ERROR',
            'intensity': 'N/A',
            'duration': 'N/A',
            'frequency': 'N/A',
            'focus_areas': 'N/A',
            'exercises': [],
            'notes': f'Errore: {str(e)}'
        }
