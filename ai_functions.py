"""
AI Functions Module - CoachTrack Elite AI (Minimal Version)
"""

import pandas as pd
import numpy as np

def calculate_distance(df):
    """Calculate total distance"""
    if len(df) < 2:
        return 0.0
    dx = np.diff(df['x'].values)
    dy = np.diff(df['y'].values)
    return float(np.sum(np.sqrt(dx**2 + dy**2)))

def calculate_speed(df):
    """Calculate speed and add to dataframe"""
    if len(df) < 2:
        return df
    if 'speed_kmh_calc' in df.columns:
        return df
    df = df.copy()
    dx = np.diff(df['x'].values)
    dy = np.diff(df['y'].values)
    dt = np.diff(df['timestamp'].values).astype(float) / 1000.0
    dt[dt == 0] = 0.001
    distances = np.sqrt(dx**2 + dy**2)
    speeds = (distances / dt) * 3.6
    df.loc[df.index[1:], 'speed_kmh_calc'] = speeds
    df.loc[df.index[0], 'speed_kmh_calc'] = 0.0
    return df

def detect_jumps_imu(df, threshold_g=1.5):
    """Detect jumps from IMU data"""
    if 'az' not in df.columns:
        return []
    jumps = []
    az = df['az'].values
    timestamps = df['timestamp'].values
    for i in range(len(az)):
        if az[i] > threshold_g:
            jumps.append({
                'timestamp': int(timestamps[i]),
                'peak_g': round(float(az[i]), 2),
                'duration_ms': 200,
                'estimated_height_cm': round(float((az[i] - 1.0) * 20), 1)
            })
    return jumps[:10]

def predict_injury_risk(player_data, player_id):
    """Predict injury risk"""
    if len(player_data) < 10:
        return {
            'player_id': player_id,
            'risk_level': 'BASSO',
            'risk_score': 10,
            'acwr': 1.0,
            'asymmetry': 5.0,
            'fatigue': 5.0,
            'risk_factors': ['Dati insufficienti per analisi completa'],
            'recommendations': ['Raccogliere piu dati tracking']
        }
    
    total_distance = calculate_distance(player_data)
    acwr = 1.2
    asymmetry = 10.0
    fatigue = 8.0
    risk_score = 25
    
    if total_distance > 200:
        risk_score += 15
        acwr = 1.5
    
    risk_level = 'ALTO' if risk_score > 60 else 'MEDIO' if risk_score > 30 else 'BASSO'
    
    return {
        'player_id': player_id,
        'risk_level': risk_level,
        'risk_score': risk_score,
        'acwr': acwr,
        'asymmetry': asymmetry,
        'fatigue': fatigue,
        'risk_factors': [f'ACWR: {acwr}', f'Asimmetria: {asymmetry}%'],
        'recommendations': ['Monitorare carico allenamento', 'Recupero adeguato']
    }

def recommend_offensive_plays(player_data):
    """Recommend offensive plays"""
    if len(player_data) < 5:
        return {
            'recommended_plays': ['Dati insufficienti'],
            'reasoning': ['Caricare piu dati']
        }
    
    distance = calculate_distance(player_data)
    plays = []
    reasons = []
    
    if distance > 100:
        plays.append('Fast Break')
        reasons.append('Alta mobilita rilevata')
    
    plays.extend(['Pick and Roll', 'Motion Offense', 'Iso 1vs1'])
    reasons.append('Gioco versatile consigliato')
    
    return {
        'recommended_plays': plays,
        'reasoning': reasons
    }

def optimize_defensive_matchups(team_data, opponent_data=None):
    """Optimize defensive matchups"""
    if not team_
        return []
    
    matchups = []
    for player_id in team_data.keys():
        matchups.append({
            'defender': player_id,
            'opponent': 'Opponent Forward',
            'match_score': 75,
            'reason': 'Matchup versatile consigliato'
        })
    
    return matchups

def analyze_movement_patterns(player_data, player_id):
    """Analyze movement patterns"""
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
        'insights': [f'Distanza totale: {distance:.1f}m', 'Pattern di movimento analizzato'],
        'anomalies': []
    }

def simulate_shot_quality(player_data, player_id):
    """Simulate shot quality"""
    if len(player_data) < 5:
        return {
            'player_id': player_id,
            'avg_quality': 0,
            'shots': [],
            'recommendations': ['Dati insufficienti']
        }
    
    shots = []
    for i in range(min(5, len(player_data))):
        x = player_data.iloc[i]['x']
        y = player_data.iloc[i]['y']
        distance = np.sqrt(x**2 + y**2)
        quality = 90 if distance < 5 else 60
        shots.append({
            'x': float(x),
            'y': float(y),
            'distance': float(distance),
            'quality': quality,
            'type': 'Close 2PT' if distance < 5 else 'Mid-range'
        })
    
    avg_quality = np.mean([s['quality'] for s in shots])
    
    return {
        'player_id': player_id,
        'avg_quality': round(avg_quality, 1),
        'shots': shots,
        'recommendations': ['Buona selezione tiro']
    }

def generate_ai_training_plan(player_id, injury_risk_data, physical_data=None):
    """Generate AI training plan"""
    risk_level = injury_risk_data.get('risk_level', 'MEDIO')
    
    if risk_level == 'ALTO':
        intensity = 'BASSA'
        duration = '30-40 min'
        frequency = '3-4 sessioni/settimana'
        exercises = [
            {'name': 'Recovery', 'sets': '3x10', 'focus': 'Recupero', 'priority': 'Alta'},
            {'name': 'Mobility', 'sets': '15min', 'focus': 'Mobilita', 'priority': 'Alta'}
        ]
    else:
        intensity = 'MODERATA'
        duration = '60 min'
        frequency = '5 sessioni/settimana'
        exercises = [
            {'name': 'Strength', 'sets': '4x8', 'focus': 'Forza', 'priority': 'Alta'},
            {'name': 'Basketball Drills', 'sets': '30min', 'focus': 'Tecnica', 'priority': 'Media'},
            {'name': 'Conditioning', 'sets': '20min', 'focus': 'Aerobico', 'priority': 'Media'}
        ]
    
    return {
        'player_id': player_id,
        'risk_level': risk_level,
        'intensity': intensity,
        'duration': duration,
        'frequency': frequency,
        'focus_areas': 'Condizionamento generale',
        'exercises': exercises,
        'notes': f'Piano basato su risk level {risk_level}'
    }
