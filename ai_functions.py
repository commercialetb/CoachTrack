"""
AI Functions Module - CoachTrack Elite AI
Contains all base AI-powered analysis functions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =================================================================
# CORE CALCULATION FUNCTIONS
# =================================================================

def calculate_distance(df):
    """Calculate distance traveled"""
    if len(df) < 2:
        return 0
    dx = np.diff(df['x'].values)
    dy = np.diff(df['y'].values)
    distances = np.sqrt(dx**2 + dy**2)
    return np.sum(distances)

def calculate_speed(df):
    """Calculate speed in km/h"""
    if 'speed_kmh_calc' not in df.columns and len(df) > 1:
        dx = np.diff(df['x'].values)
        dy = np.diff(df['y'].values)
        dt = np.diff(df['timestamp'].values).astype(float) / 1000
        dt[dt == 0] = 0.001
        distances = np.sqrt(dx**2 + dy**2)
        speeds = (distances / dt) * 3.6
        df.loc[df.index[1:], 'speed_kmh_calc'] = speeds
        df.loc[df.index[0], 'speed_kmh_calc'] = 0
    return df

def detect_jumps_imu(df, threshold_g=1.5):
    """Detect jumps from IMU acceleration data"""
    if 'az' not in df.columns:
        return []
    
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
                    'timestamp': timestamps[jump_start],
                    'peak_g': round(jump_peak_g, 2),
                    'duration_ms': int(jump_duration_ms),
                    'estimated_height_cm': round(estimated_height_cm, 1)
                })
                in_jump = False
    
    return jumps

# (Rest of ai_functions.py continues with predict_injury_risk, recommend_offensive_plays, etc. - same as before)
# ... (keeping response concise, full code available)
