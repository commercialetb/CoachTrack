"""
Physical Data Management & Enhanced AI Nutrition Module
CoachTrack Elite AI v3.0
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from io import BytesIO

# =================================================================
# PHYSICAL METRICS DEFINITIONS
# =================================================================

PHYSICAL_METRICS = {
    "bmi": {"label": "BMI", "label_en": "BMI", "unit": "", "default": 22.5, "min": 15.0, "max": 35.0},
    "weight_kg": {"label": "Peso", "label_en": "Weight", "unit": "kg", "default": 80.0, "min": 50.0, "max": 150.0},
    "lean_mass_kg": {"label": "Massa Magra", "label_en": "Lean Mass", "unit": "kg", "default": 68.0, "min": 40.0, "max": 120.0},
    "body_fat_pct": {"label": "Grasso", "label_en": "Body Fat", "unit": "%", "default": 12.0, "min": 3.0, "max": 40.0},
    "body_water_pct": {"label": "Acqua", "label_en": "Body Water", "unit": "%", "default": 60.0, "min": 45.0, "max": 75.0},
    "muscle_pct": {"label": "Muscoli", "label_en": "Muscle", "unit": "%", "default": 45.0, "min": 25.0, "max": 60.0},
    "bone_mass_kg": {"label": "Ossa", "label_en": "Bone Mass", "unit": "kg", "default": 3.2, "min": 2.0, "max": 5.0},
    "bmr": {"label": "BMR", "label_en": "BMR", "unit": "kcal", "default": 1800, "min": 1200, "max": 3000},
    "amr": {"label": "AMR", "label_en": "AMR", "unit": "kcal", "default": 2700, "min": 1800, "max": 5000}
}

# =================================================================
# DATA PARSING & VALIDATION
# =================================================================

def parse_physical_csv(uploaded_file):
    """Parse CSV with physical data"""
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = ['player_id', 'date']
        
        if not all(col in df.columns for col in required_cols):
            return None, "CSV must contain 'player_id' and 'date' columns"
        
        return df, None
    except Exception as e:
        return None, f"Error parsing CSV: {str(e)}"

def validate_physical_data(data_dict):
    """Validate physical data ranges"""
    warnings = []
    
    for metric, value in data_dict.items():
        if metric in PHYSICAL_METRICS:
            metric_info = PHYSICAL_METRICS[metric]
            if value < metric_info['min'] or value > metric_info['max']:
                warnings.append(f"{metric_info['label']}: {value} fuori range ({metric_info['min']}-{metric_info['max']})")
    
    return warnings

# =================================================================
# APPLE HEALTH SIMULATION
# =================================================================

def simulate_apple_health_sync(player_id):
    """Simulate Apple Health data sync (demo)"""
    base_weight = 75 + np.random.uniform(-10, 15)
    body_fat = 8 + np.random.uniform(0, 8)
    
    data = {
        'weight_kg': round(base_weight, 1),
        'body_fat_pct': round(body_fat, 1),
        'lean_mass_kg': round(base_weight * (1 - body_fat/100), 1),
        'body_water_pct': round(55 + np.random.uniform(0, 10), 1),
        'muscle_pct': round(40 + np.random.uniform(0, 10), 1),
        'bone_mass_kg': round(2.8 + np.random.uniform(0, 0.8), 1),
        'bmi': round(base_weight / (1.95**2), 1),
        'bmr': int(1600 + np.random.uniform(0, 400)),
        'amr': int(2400 + np.random.uniform(0, 800)),
        'date': datetime.now().strftime('%Y-%m-%d'),
        'source': 'Apple Health (Demo)'
    }
    
    return data

# =================================================================
# ADVANCED CALCULATIONS
# =================================================================

def calculate_advanced_bmr(lean_mass_kg, weight_kg, age=25, height_cm=195):
    """Calculate BMR using Katch-McArdle (uses lean mass)"""
    if lean_mass_kg and lean_mass_kg > 0:
        bmr = 370 + (21.6 * lean_mass_kg)
    else:
        bmr = 88.362 + (13.397 * weight_kg) + (4.799 * height_cm) - (5.677 * age)
    
    return int(bmr)

def calculate_amr(bmr, activity_level):
    """Calculate AMR based on activity level"""
    multipliers = {
        "Low (Recovery)": 1.3,
        "Moderate (Training)": 1.55,
        "High (Intense/Match)": 1.75,
        "Very High (Tournament)": 1.95
    }
    
    multiplier = multipliers.get(activity_level, 1.55)
    return int(bmr * multiplier)

def estimate_body_composition(weight_kg, body_fat_pct):
    """Estimate other body composition metrics if missing"""
    fat_mass = weight_kg * (body_fat_pct / 100)
    lean_mass = weight_kg - fat_mass
    
    estimates = {
        'lean_mass_kg': round(lean_mass, 1),
        'body_water_pct': round(55 + (20 - body_fat_pct) * 0.5, 1),
        'muscle_pct': round(lean_mass / weight_kg * 100 * 0.65, 1),
        'bone_mass_kg': round(weight_kg * 0.04, 1)
    }
    
    return estimates

# =================================================================
# ENHANCED AI NUTRITION
# =================================================================

def generate_enhanced_nutrition(player_id, physical_data, activity_level, goal):
    """Generate personalized nutrition plan with AI"""
    
    weight = physical_data.get('weight_kg', 80)
    lean_mass = physical_data.get('lean_mass_kg', 68)
    body_fat = physical_data.get('body_fat_pct', 12)
    bmr = physical_data.get('bmr')
    
    if not bmr:
        bmr = calculate_advanced_bmr(lean_mass, weight)
    
    amr = calculate_amr(bmr, activity_level)
    
    goal_adjustments = {
        "Maintenance": 1.0,
        "Muscle Gain": 1.15,
        "Fat Loss": 0.85,
        "Performance": 1.1
    }
    
    target_calories = int(amr * goal_adjustments.get(goal, 1.0))
    
    protein_g = int(lean_mass * 2.2) if goal == "Muscle Gain" else int(lean_mass * 1.8)
    
    carb_pct = 0.50 if activity_level == "High (Intense/Match)" else 0.45
    fat_pct = 0.25 if goal == "Fat Loss" else 0.30
    
    remaining_cals = target_calories - (protein_g * 4)
    carbs_g = int((remaining_cals * carb_pct) / 4)
    fats_g = int((remaining_cals * fat_pct) / 9)
    
    recommendations = []
    if body_fat < 8:
        recommendations.append("⚠️ Grasso corporeo molto basso - aumenta apporto calorico")
    if body_fat > 18:
        recommendations.append("📉 Considera deficit calorico moderato (-300 kcal)")
    if protein_g / weight < 1.6:
        recommendations.append("🥩 Aumenta proteine per supporto muscolare")
    
    if activity_level == "High (Intense/Match)":
        recommendations.append("⚡ Pre-gara: 50-70g carboidrati 2-3h prima")
        recommendations.append("💧 Idratazione: 3-4L acqua durante giorno gara")
    
    supplements = []
    if goal == "Muscle Gain":
        supplements.extend(["Creatina 5g/giorno", "Whey Protein post-workout", "BCAA durante allenamento"])
    if activity_level in ["High (Intense/Match)", "Very High (Tournament)"]:
        supplements.extend(["Magnesio 400mg", "Omega-3 2g", "Vitamina D 4000 IU"])
    if goal == "Performance":
        supplements.extend(["Beta-alanina 3-6g", "Caffeina 200mg pre-gara", "Elettroliti durante match"])
    
    meals = [
        {
            "name": "Colazione",
            "timing": "7:00-8:00 (Entro 1h da sveglia)",
            "calories": int(target_calories * 0.25),
            "protein": int(protein_g * 0.25),
            "carbs": int(carbs_g * 0.30),
            "fats": int(fats_g * 0.25),
            "examples": "Avena 80g + Whey 30g + Banana + Burro arachidi 15g"
        },
        {
            "name": "Spuntino Pre-Training",
            "timing": "10:30-11:00 (1-2h prima allenamento)",
            "calories": int(target_calories * 0.15),
            "protein": int(protein_g * 0.15),
            "carbs": int(carbs_g * 0.20),
            "fats": int(fats_g * 0.10),
            "examples": "Yogurt greco 200g + Miele 20g + Mandorle 20g"
        },
        {
            "name": "Pranzo Post-Training",
            "timing": "13:00-14:00 (Entro 2h da allenamento)",
            "calories": int(target_calories * 0.30),
            "protein": int(protein_g * 0.35),
            "carbs": int(carbs_g * 0.30),
            "fats": int(fats_g * 0.25),
            "examples": "Pollo 200g + Riso 120g + Verdure + Olio oliva 15ml"
        },
        {
            "name": "Spuntino Pomeriggio",
            "timing": "17:00-17:30",
            "calories": int(target_calories * 0.10),
            "protein": int(protein_g * 0.10),
            "carbs": int(carbs_g * 0.10),
            "fats": int(fats_g * 0.15),
            "examples": "Frutta + Noci 30g"
        },
        {
            "name": "Cena",
            "timing": "20:00-21:00 (3-4h prima sonno)",
            "calories": int(target_calories * 0.20),
            "protein": int(protein_g * 0.15),
            "carbs": int(carbs_g * 0.10),
            "fats": int(fats_g * 0.25),
            "examples": "Salmone 150g + Patate dolci 150g + Insalata + Avocado 50g"
        }
    ]
    
    return {
        "player_id": player_id,
        "target_calories": target_calories,
        "protein_g": protein_g,
        "carbs_g": carbs_g,
        "fats_g": fats_g,
        "bmr": bmr,
        "amr": amr,
        "recommendations": recommendations,
        "supplements": supplements,
        "meals": meals,
        "activity_level": activity_level,
        "goal": goal
    }

# =================================================================
# VISUALIZATIONS
# =================================================================

def create_body_composition_viz(physical_data):
    """Create comprehensive body composition dashboard"""
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('BMI', 'Body Fat %', 'Muscle %', 'Lean Mass', 'Water %', 'BMR/AMR'),
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'bar'}]]
    )
    
    bmi = physical_data.get('bmi', 22)
    bmi_color = "green" if 18.5 <= bmi <= 24.9 else "orange" if bmi < 18.5 else "red"
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=bmi,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [15, 35]},
               'bar': {'color': bmi_color},
               'steps': [
                   {'range': [15, 18.5], 'color': "lightgray"},
                   {'range': [18.5, 24.9], 'color': "lightgreen"},
                   {'range': [24.9, 35], 'color': "lightyellow"}]}
    ), row=1, col=1)
    
    bf = physical_data.get('body_fat_pct', 12)
    bf_color = "green" if 8 <= bf <= 15 else "orange"
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=bf,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [3, 30]},
               'bar': {'color': bf_color}}
    ), row=1, col=2)
    
    muscle = physical_data.get('muscle_pct', 45)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=muscle,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [25, 60]},
               'bar': {'color': "darkblue"}}
    ), row=1, col=3)
    
    lean = physical_data.get('lean_mass_kg', 68)
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=lean,
        delta={'reference': 65, 'valueformat': ".1f"},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=2, col=1)
    
    water = physical_data.get('body_water_pct', 60)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=water,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [45, 75]},
               'bar': {'color': "cyan"}}
    ), row=2, col=2)
    
    bmr = physical_data.get('bmr', 1800)
    amr = physical_data.get('amr', 2700)
    fig.add_trace(go.Bar(
        x=['BMR', 'AMR'],
        y=[bmr, amr],
        marker_color=['lightblue', 'darkblue'],
        text=[f'{bmr} kcal', f'{amr} kcal'],
        textposition='auto'
    ), row=2, col=3)
    
    fig.update_layout(height=600, showlegend=False, title_text="Body Composition Dashboard")
    
    return fig

# =================================================================
# EXPORT FUNCTIONS
# =================================================================

def export_physical_data_excel(physical_profiles):
    """Export physical data to formatted Excel"""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
        
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            data_rows = []
            for player_id, data in physical_profiles.items():
                row = {'Player': player_id}
                row.update(data)
                data_rows.append(row)
            
            df = pd.DataFrame(data_rows)
            df.to_excel(writer, sheet_name='Physical Data', index=False)
            
            workbook = writer.book
            worksheet = writer.sheets['Physical Data']
            
            for cell in worksheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
                cell.alignment = Alignment(horizontal='center')
        
        output.seek(0)
        return output
        
    except ImportError:
        df = pd.DataFrame([{'Player': pid, **data} for pid, data in physical_profiles.items()])
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return output

def create_physical_csv_template():
    """Create CSV template for upload"""
    template_data = {
        'player_id': ['Player1', 'Player2'],
        'date': ['2026-02-07', '2026-02-07'],
        'weight_kg': [82.5, 78.0],
        'body_fat_pct': [11.5, 9.5],
        'lean_mass_kg': [73.0, 70.5],
        'body_water_pct': [61.5, 63.0],
        'muscle_pct': [46.0, 47.5],
        'bone_mass_kg': [3.2, 3.0],
        'bmi': [22.8, 21.5],
        'bmr': [1850, 1950],
        'amr': [2775, 2925]
    }
    
    df = pd.DataFrame(template_data)
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return output
