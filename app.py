import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import ConvexHull
from io import BytesIO
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# PDF Generation
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# =================================================================
# CONFIG
# =================================================================
st.set_page_config(page_title='CoachTrack Elite AI', layout='wide', initial_sidebar_state='collapsed')

st.markdown("""
<style>
    header {visibility: hidden;}
    .main { background-color: #f8fafc !important; color: #1e293b !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: #ffffff; border-radius: 12px; padding: 10px; border: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] { height: 60px; color: #64748b !important; font-size: 16px !important; font-weight: 700 !important; }
    .stTabs [aria-selected="true"] { color: #2563eb !important; border-bottom: 4px solid #2563eb !important; }
    .predictive-card { background: #ffffff; padding: 20px; border-radius: 15px; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .ai-report-light { background: #ffffff; padding: 30px; border-radius: 15px; border-left: 5px solid #2563eb; line-height: 1.6; margin: 15px 0; }
    .physical-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 15px; margin: 10px 0; }
    .contact-card { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 20px; border-radius: 12px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# =================================================================
# CORE FUNCTIONS
# =================================================================
def draw_basketball_court():
    court_length, court_width = 28.0, 15.0
    shapes = []
    shapes.append(dict(type="rect", x0=0, y0=0, x1=court_length, y1=court_width,
                      line=dict(color="white", width=3), fillcolor="rgba(0,0,0,0)"))
    shapes.append(dict(type="line", x0=court_length/2, y0=0, x1=court_length/2, y1=court_width,
                      line=dict(color="white", width=2)))
    shapes.append(dict(type="circle", x0=court_length/2-1.8, y0=court_width/2-1.8,
                      x1=court_length/2+1.8, y1=court_width/2+1.8,
                      line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)"))
    shapes.append(dict(type="path", path=f"M 0,{court_width/2-6.75} Q 6.75,{court_width/2} 0,{court_width/2+6.75}",
                      line=dict(color="white", width=2)))
    shapes.append(dict(type="path", path=f"M {court_length},{court_width/2-6.75} Q {court_length-6.75},{court_width/2} {court_length},{court_width/2+6.75}",
                      line=dict(color="white", width=2)))
    for x_pos in [5.8, court_length-5.8]:
        shapes.append(dict(type="circle", x0=x_pos-1.8, y0=court_width/2-1.8,
                          x1=x_pos+1.8, y1=court_width/2+1.8,
                          line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)"))
    return shapes

def classify_zone(x, y):
    court_length, court_width = 28.0, 15.0
    if (x <= 5.8 and abs(y - court_width/2) <= 2.45) or (x >= court_length - 5.8 and abs(y - court_width/2) <= 2.45):
        return 'Paint'
    dist_left = np.sqrt((x - 1.575)**2 + (y - court_width/2)**2)
    dist_right = np.sqrt((x - (court_length - 1.575))**2 + (y - court_width/2)**2)
    if dist_left >= 6.75 or dist_right >= 6.75:
        return '3-Point'
    return 'Mid-Range'

# AI FUNCTIONS
def calculate_injury_risk(player_data, player_id):
    if len(player_data) < 100: return 0, 1.0, 0.1, 5, "🟢 LOW"
    recent = player_data.tail(min(100, len(player_data)))['speed_kmh_calc'].sum()
    chronic = player_data['speed_kmh_calc'].mean() * 100
    acwr = recent / chronic if chronic > 0 else 1.0
    left_moves = (player_data['dx'] < -0.5).sum()
    right_moves = (player_data['dx'] > 0.5).sum()
    asymmetry = abs(left_moves - right_moves) / max(left_moves + right_moves, 1)
    q1_speed = player_data.head(len(player_data)//4)['speed_kmh_calc'].mean()
    q4_speed = player_data.tail(len(player_data)//4)['speed_kmh_calc'].mean()
    fatigue = abs((q1_speed - q4_speed) / q1_speed * 100) if q1_speed > 0 else 5
    risk = 0
    if acwr > 1.5: risk += 40
    if acwr < 0.8: risk += 20
    if asymmetry > 0.25: risk += 30
    if fatigue > 15: risk += 30
    risk = min(risk, 100)
    level = "🔴 HIGH" if risk > 60 else "🟡 MEDIUM" if risk > 30 else "🟢 LOW"
    return risk, acwr, asymmetry, fatigue, level

def recommend_offensive_play(spacing, quarter, score_diff):
    plays = {
        "Pick & Roll Top": {"ppp": 1.15, "rate": 0.82, "when": "Switch-heavy defense"},
        "Motion Offense": {"ppp": 1.05, "rate": 0.75, "when": "Against zone"},
        "Flare Screen": {"ppp": 1.08, "rate": 0.79, "when": "High spacing"},
        "Transition": {"ppp": 1.28, "rate": 0.89, "when": "After steal"}
    }
    if spacing > 90 and quarter >= 3: best = "Pick & Roll Top"
    elif score_diff < -5 and quarter == 4: best = "Transition"
    elif spacing > 85: best = "Flare Screen"
    else: best = "Motion Offense"
    return best, plays[best]["ppp"], plays[best]["rate"], plays[best]["when"]

def optimize_defensive_matchups(defenders):
    matchups = []
    threats = ["⭐ Star", "🎯 Shooter", "💪 Physical", "⚡ Fast"]
    for i, defender in enumerate(defenders):
        threat = threats[i % len(threats)]
        stop_rate = f"{np.random.randint(55, 88)}%"
        rec = "✅ Keep" if np.random.random() > 0.3 else "🔄 Switch"
        matchups.append({"your_player": defender, "opponent_threat": threat, "stop_rate": stop_rate, "recommendation": rec})
    return matchups

def analyze_movement_patterns(player_data):
    if len(player_data) < 50: return {"dominant_direction": "Right (N/A)", "avg_speed": 0, "preferred_zone": "N/A", "confidence": 0}
    right_moves = (player_data['dx'] > 0.5).sum()
    left_moves = (player_data['dx'] < -0.5).sum()
    direction = f"Right ({right_moves})" if right_moves > left_moves else f"Left ({left_moves})"
    zone_counts = player_data['zone'].value_counts()
    preferred = zone_counts.idxmax() if len(zone_counts) > 0 else "Mid-Range"
    avg_speed = player_data['speed_kmh_calc'].mean()
    confidence = min(len(player_data) / 500, 1.0)
    return {"dominant_direction": direction, "avg_speed": avg_speed, "preferred_zone": preferred, "confidence": confidence}

def calculate_shot_quality(x, y, spacing):
    dist_left = np.sqrt((x - 1.575)**2 + (y - 7.5)**2)
    dist_right = np.sqrt((x - 26.425)**2 + (y - 7.5)**2)
    dist_basket = min(dist_left, dist_right)
    if dist_basket < 2: base_prob = 0.65
    elif dist_basket < 4: base_prob = 0.55
    elif 6 < dist_basket < 7.5: base_prob = 0.38
    else: base_prob = 0.42
    spacing_factor = spacing / 85.0
    base_prob *= (0.9 + spacing_factor * 0.2)
    return min(base_prob, 0.95)

# =================================================================
# NEW: AI TRAINING & NUTRITION FUNCTIONS
# =================================================================

def calculate_bmr(weight_kg, height_cm, age, gender):
    """Calculate Basal Metabolic Rate using Mifflin-St Jeor"""
    if gender == "Male":
        return (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        return (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161

def calculate_tdee(bmr, activity_level):
    """Calculate Total Daily Energy Expenditure"""
    activity_multipliers = {
        "Low (Recovery)": 1.3,
        "Moderate (Training)": 1.55,
        "High (Intense/Match)": 1.85,
        "Very High (Tournament)": 2.1
    }
    return bmr * activity_multipliers.get(activity_level, 1.55)

def generate_personalized_training(player_id, player_data, physical_profile, injury_risk_data):
    """AI-Generated Personalized Training Program"""
    
    risk, acwr, asymmetry, fatigue, level = injury_risk_data
    patterns = analyze_movement_patterns(player_data)
    
    # Analyze player metrics
    avg_speed = patterns['avg_speed']
    max_speed = player_data['speed_kmh_calc'].max() if len(player_data) > 0 else 0
    distance = player_data['step_m'].sum() if len(player_data) > 0 else 0
    preferred_zone = patterns['preferred_zone']
    
    # AI Decision Logic
    training_plan = {
        "volume": "Moderate",
        "intensity": "Moderate", 
        "focus_areas": [],
        "exercises": [],
        "recovery": "Standard",
        "warnings": []
    }
    
    # Volume adjustment based on ACWR
    if acwr > 1.5:
        training_plan["volume"] = "Reduced (High ACWR)"
        training_plan["warnings"].append("⚠️ ACWR elevato - ridurre carico di lavoro")
    elif acwr < 0.8:
        training_plan["volume"] = "Increased (Low ACWR)"
        training_plan["warnings"].append("✅ ACWR basso - può aumentare carico")
    
    # Intensity based on fatigue
    if fatigue > 15:
        training_plan["intensity"] = "Low (High Fatigue)"
        training_plan["recovery"] = "Extended - 48-72h"
        training_plan["warnings"].append("🛑 Fatica elevata - priorità recupero")
    elif avg_speed > 18:
        training_plan["intensity"] = "High"
        training_plan["warnings"].append("💪 Ottima velocità media - mantieni intensità")
    
    # Asymmetry correction
    if asymmetry > 0.25:
        training_plan["focus_areas"].append("Correzione asimmetria laterale")
        training_plan["exercises"].extend([
            "🔄 Single-leg drills (lato debole)",
            "🔄 Lateral bounds con focus su equilibrio",
            "🔄 Defensive slides con enfasi lato debole"
        ])
    
    # Zone-specific training
    if preferred_zone == "Paint":
        training_plan["focus_areas"].append("Potenza esplosiva sotto canestro")
        training_plan["exercises"].extend([
            "🏀 Mikan drill (50 rep)",
            "💥 Box jumps (4x8)",
            "💪 Post moves con contatto"
        ])
    elif preferred_zone == "3-Point":
        training_plan["focus_areas"].append("Meccanica di tiro e condizionamento")
        training_plan["exercises"].extend([
            "🎯 Spot shooting da 7 zone (10 tiri/zona)",
            "🏃 Transition 3-point drills",
            "⚡ Quick release drills"
        ])
    else:
        training_plan["focus_areas"].append("Versatilità mid-range")
        training_plan["exercises"].extend([
            "🎯 Pull-up jumpers (5 spots x 8 rep)",
            "🔄 Pick & Roll finishing",
            "⚡ Catch-and-shoot drills"
        ])
    
    # Speed development
    if avg_speed < 12:
        training_plan["focus_areas"].append("Sviluppo velocità")
        training_plan["exercises"].extend([
            "⚡ Sprint drills 10-20m (6x3)",
            "🏃 Acceleration ladder drills",
            "💨 Resistance band sprints"
        ])
    
    # Conditioning based on distance
    if distance < 2000:
        training_plan["focus_areas"].append("Condizionamento aerobico")
        training_plan["exercises"].extend([
            "🔄 Continuous movement drills (12 min)",
            "🏃 Transition runs full court (8x)"
        ])
    
    # Recovery protocols
    if risk > 60:
        training_plan["recovery"] = "Priority - 72h minimum"
        training_plan["exercises"].insert(0, "🧘 Active recovery: stretching dinamico 20 min")
        training_plan["exercises"].insert(1, "❄️ Ice bath 10-12 min")
    
    return training_plan

def generate_personalized_nutrition(player_id, physical_profile, activity_level, goal):
    """AI-Generated Personalized Nutrition Plan"""
    
    weight = physical_profile.get('weight_kg', 80)
    height = physical_profile.get('height_cm', 190)
    age = physical_profile.get('age', 25)
    gender = physical_profile.get('gender', 'Male')
    body_fat = physical_profile.get('body_fat_pct', 12)
    
    # Calculate energy needs
    bmr = calculate_bmr(weight, height, age, gender)
    tdee = calculate_tdee(bmr, activity_level)
    
    # Adjust calories based on goal
    if goal == "Muscle Gain":
        target_calories = tdee + 300
        protein_ratio = 0.30
        carb_ratio = 0.45
        fat_ratio = 0.25
    elif goal == "Fat Loss":
        target_calories = tdee - 400
        protein_ratio = 0.35
        carb_ratio = 0.35
        fat_ratio = 0.30
    elif goal == "Performance":
        target_calories = tdee + 100
        protein_ratio = 0.25
        carb_ratio = 0.50
        fat_ratio = 0.25
    else:  # Maintenance
        target_calories = tdee
        protein_ratio = 0.25
        carb_ratio = 0.45
        fat_ratio = 0.30
    
    # Calculate macros
    protein_cal = target_calories * protein_ratio
    carb_cal = target_calories * carb_ratio
    fat_cal = target_calories * fat_ratio
    
    protein_g = protein_cal / 4
    carb_g = carb_cal / 4
    fat_g = fat_cal / 9
    
    # Generate meal plan
    nutrition_plan = {
        "target_calories": int(target_calories),
        "bmr": int(bmr),
        "tdee": int(tdee),
        "protein_g": int(protein_g),
        "carbs_g": int(carb_g),
        "fats_g": int(fat_g),
        "water_liters": round(weight * 0.035, 1),
        "meals": []
    }
    
    # Distribute macros across meals
    meals_structure = [
        {"name": "Colazione Pre-Allenamento", "cal_pct": 0.25},
        {"name": "Snack Post-Allenamento", "cal_pct": 0.15},
        {"name": "Pranzo", "cal_pct": 0.30},
        {"name": "Snack Pomeridiano", "cal_pct": 0.10},
        {"name": "Cena", "cal_pct": 0.20}
    ]
    
    for meal in meals_structure:
        meal_calories = target_calories * meal['cal_pct']
        meal_protein = (meal_calories * protein_ratio) / 4
        meal_carbs = (meal_calories * carb_ratio) / 4
        meal_fats = (meal_calories * fat_ratio) / 9
        
        nutrition_plan["meals"].append({
            "name": meal["name"],
            "calories": int(meal_calories),
            "protein": int(meal_protein),
            "carbs": int(meal_carbs),
            "fats": int(meal_fats)
        })
    
    # Add recommendations
    nutrition_plan["recommendations"] = []
    
    if body_fat > 15 and goal != "Fat Loss":
        nutrition_plan["recommendations"].append("⚠️ Body fat elevato - considera ridurre carboidrati del 10%")
    
    if activity_level == "High (Intense/Match)":
        nutrition_plan["recommendations"].append("💧 Aumenta idratazione: " + str(nutrition_plan["water_liters"] + 0.5) + "L")
        nutrition_plan["recommendations"].append("⚡ Aggiungi carboidrati veloci pre-gara (banana, gel)")
    
    if protein_g / weight < 1.6:
        nutrition_plan["recommendations"].append("💪 Proteine ottimali: 1.6-2.2g/kg per atleti")
    
    nutrition_plan["recommendations"].append(f"🥗 Verdure ad ogni pasto principale (minimo 200g)")
    nutrition_plan["recommendations"].append(f"🐟 Omega-3: 2-3 porzioni pesce/settimana")
    
    return nutrition_plan

# =================================================================
# EMAIL SENDING FUNCTION
# =================================================================

def send_email_with_pdf(recipient_email, recipient_name, subject, body, pdf_data, pdf_filename, smtp_config):
    """
    Send email with PDF attachment
    
    Args:
        recipient_email: Email address of recipient
        recipient_name: Name of recipient
        subject: Email subject
        body: Email body (HTML)
        pdf_data: PDF file bytes
        pdf_filename: Name for PDF attachment
        smtp_config: Dict with smtp_server, smtp_port, smtp_user, smtp_password
    
    Returns:
        success: Boolean
        message: Success or error message
    """
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = smtp_config['smtp_user']
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # Add HTML body
        html_part = MIMEText(body, 'html')
        msg.attach(html_part)
        
        # Attach PDF
        if pdf_data:
            pdf_attachment = MIMEApplication(pdf_data, _subtype='pdf')
            pdf_attachment.add_header('Content-Disposition', 'attachment', filename=pdf_filename)
            msg.attach(pdf_attachment)
        
        # Send email
        server = smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port'])
        server.starttls()
        server.login(smtp_config['smtp_user'], smtp_config['smtp_password'])
        server.send_message(msg)
        server.quit()
        
        return True, f"✅ Email inviata con successo a {recipient_name} ({recipient_email})"
    
    except Exception as e:
        return False, f"❌ Errore invio email: {str(e)}"

# =================================================================
# PDF GENERATION FUNCTIONS (EXTENDED)
# =================================================================

def generate_team_pdf(team_name, kpi_df, brand_color, session_type):
    if not PDF_AVAILABLE: return None
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm)
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=22, textColor=colors.HexColor(brand_color), alignment=TA_CENTER)
    story.append(Paragraph(f"📊 {team_name} - {session_type} REPORT", title_style))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    
    table_data = [['Player', 'Distance (m)', 'Max Speed', 'Avg Speed', 'Quality']]
    for _, row in kpi_df.iterrows():
        table_data.append([str(row['player_id']), f"{row['distance_m']:.0f}", f"{row['max_speed_kmh']:.1f}", f"{row['avg_speed_kmh']:.1f}", f"{row['avg_quality']:.0f}"])
    
    t = Table(table_data, colWidths=[3*cm, 3*cm, 3*cm, 3*cm, 2.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(brand_color)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    story.append(t)
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(f"<b>SUMMARY</b>", styles['Heading2']))
    story.append(Paragraph(f"• Total Distance: {kpi_df['distance_m'].sum():.0f} m", styles['Normal']))
    story.append(Paragraph(f"• Max Speed: {kpi_df['max_speed_kmh'].max():.1f} km/h", styles['Normal']))
    doc.build(story)
    return buffer.getvalue()

def generate_player_pdf(player_id, player_data, team_name, brand_color, session_type):
    if not PDF_AVAILABLE: return None
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm)
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor(brand_color), alignment=TA_CENTER)
    story.append(Paragraph(f"👤 {player_id} - {session_type} REPORT", title_style))
    story.append(Paragraph(f"{team_name} | {datetime.now().strftime('%d/%m/%Y')}", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    
    risk, acwr, asym, fat, level = calculate_injury_risk(player_data, player_id)
    story.append(Paragraph(f"<b>🏥 INJURY RISK: {level}</b> ({risk}/100)", styles['Heading2']))
    story.append(Paragraph(f"• ACWR: {acwr:.2f} | Asymmetry: {asym*100:.1f}% | Fatigue: {fat:.1f}%", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    
    patterns = analyze_movement_patterns(player_data)
    story.append(Paragraph(f"<b>👣 MOVEMENT PROFILE</b>", styles['Heading2']))
    story.append(Paragraph(f"• Dominant: {patterns['dominant_direction']}", styles['Normal']))
    story.append(Paragraph(f"• Avg Speed: {patterns['avg_speed']:.1f} km/h", styles['Normal']))
    story.append(Paragraph(f"• Preferred Zone: {patterns['preferred_zone']}", styles['Normal']))
    
    doc.build(story)
    return buffer.getvalue()

def generate_training_pdf(player_id, player_name, training_plan, physical_profile, brand_color):
    """Generate PDF for personalized training plan"""
    if not PDF_AVAILABLE: return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=24, 
                                 textColor=colors.HexColor(brand_color), alignment=TA_CENTER)
    story.append(Paragraph(f"🏋️ PERSONALIZED TRAINING PLAN", title_style))
    story.append(Paragraph(f"{player_name} ({player_id})", styles['Heading2']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.8*cm))
    
    # Physical Profile
    story.append(Paragraph("<b>📊 PHYSICAL PROFILE</b>", styles['Heading2']))
    profile_data = [
        ['Metric', 'Value'],
        ['Height', f"{physical_profile.get('height_cm', 'N/A')} cm"],
        ['Weight', f"{physical_profile.get('weight_kg', 'N/A')} kg"],
        ['Age', f"{physical_profile.get('age', 'N/A')} years"],
        ['Body Fat', f"{physical_profile.get('body_fat_pct', 'N/A')}%"],
        ['Vertical Jump', f"{physical_profile.get('vertical_jump_cm', 'N/A')} cm"]
    ]
    
    t = Table(profile_data, colWidths=[6*cm, 8*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(brand_color)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige)
    ]))
    story.append(t)
    story.append(Spacer(1, 0.8*cm))
    
    # Training Parameters
    story.append(Paragraph("<b>⚙️ TRAINING PARAMETERS</b>", styles['Heading2']))
    story.append(Paragraph(f"<b>Volume:</b> {training_plan['volume']}", styles['Normal']))
    story.append(Paragraph(f"<b>Intensity:</b> {training_plan['intensity']}", styles['Normal']))
    story.append(Paragraph(f"<b>Recovery:</b> {training_plan['recovery']}", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    
    # Warnings
    if training_plan['warnings']:
        story.append(Paragraph("<b>⚠️ IMPORTANT ALERTS</b>", styles['Heading2']))
        for warning in training_plan['warnings']:
            story.append(Paragraph(f"• {warning}", styles['Normal']))
        story.append(Spacer(1, 0.5*cm))
    
    # Focus Areas
    story.append(Paragraph("<b>🎯 FOCUS AREAS</b>", styles['Heading2']))
    for area in training_plan['focus_areas']:
        story.append(Paragraph(f"• {area}", styles['Normal']))
    story.append(Spacer(1, 0.8*cm))
    
    # Exercises
    story.append(Paragraph("<b>💪 RECOMMENDED EXERCISES</b>", styles['Heading2']))
    for idx, exercise in enumerate(training_plan['exercises'], 1):
        story.append(Paragraph(f"{idx}. {exercise}", styles['Normal']))
    
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("<i>This plan is generated by AI based on performance data and should be reviewed by qualified coaching staff.</i>", 
                          styles['Italic']))
    
    doc.build(story)
    return buffer.getvalue()

def generate_nutrition_pdf(player_id, player_name, nutrition_plan, physical_profile, activity_level, goal, brand_color):
    """Generate PDF for personalized nutrition plan"""
    if not PDF_AVAILABLE: return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=24,
                                 textColor=colors.HexColor(brand_color), alignment=TA_CENTER)
    story.append(Paragraph(f"🍽️ PERSONALIZED NUTRITION PLAN", title_style))
    story.append(Paragraph(f"{player_name} ({player_id})", styles['Heading2']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.8*cm))
    
    # Physical Profile
    story.append(Paragraph("<b>📊 ATHLETE PROFILE</b>", styles['Heading2']))
    profile_data = [
        ['Metric', 'Value'],
        ['Weight', f"{physical_profile.get('weight_kg', 'N/A')} kg"],
        ['Height', f"{physical_profile.get('height_cm', 'N/A')} cm"],
        ['Age', f"{physical_profile.get('age', 'N/A')} years"],
        ['Body Fat', f"{physical_profile.get('body_fat_pct', 'N/A')}%"],
        ['Activity Level', activity_level],
        ['Goal', goal]
    ]
    
    t = Table(profile_data, colWidths=[6*cm, 8*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(brand_color)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige)
    ]))
    story.append(t)
    story.append(Spacer(1, 0.8*cm))
    
    # Caloric Needs
    story.append(Paragraph("<b>⚡ CALORIC REQUIREMENTS</b>", styles['Heading2']))
    cal_data = [
        ['Metric', 'Value'],
        ['BMR (Basal Metabolic Rate)', f"{nutrition_plan['bmr']} kcal"],
        ['TDEE (Total Daily Energy)', f"{nutrition_plan['tdee']} kcal"],
        ['Target Calories', f"{nutrition_plan['target_calories']} kcal"],
        ['Daily Water', f"{nutrition_plan['water_liters']} liters"]
    ]
    
    t = Table(cal_data, colWidths=[8*cm, 6*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(brand_color)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue)
    ]))
    story.append(t)
    story.append(Spacer(1, 0.8*cm))
    
    # Macronutrients
    story.append(Paragraph("<b>🥗 DAILY MACRONUTRIENTS</b>", styles['Heading2']))
    macro_data = [
        ['Macronutrient', 'Grams', 'Calories', '% of Total'],
        ['Protein', f"{nutrition_plan['protein_g']}g", 
         f"{nutrition_plan['protein_g']*4:.0f} kcal",
         f"{(nutrition_plan['protein_g']*4/nutrition_plan['target_calories']*100):.0f}%"],
        ['Carbohydrates', f"{nutrition_plan['carbs_g']}g",
         f"{nutrition_plan['carbs_g']*4:.0f} kcal",
         f"{(nutrition_plan['carbs_g']*4/nutrition_plan['target_calories']*100):.0f}%"],
        ['Fats', f"{nutrition_plan['fats_g']}g",
         f"{nutrition_plan['fats_g']*9:.0f} kcal",
         f"{(nutrition_plan['fats_g']*9/nutrition_plan['target_calories']*100):.0f}%"]
    ]
    
    t = Table(macro_data, colWidths=[4*cm, 3*cm, 3.5*cm, 3*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(brand_color)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen)
    ]))
    story.append(t)
    story.append(Spacer(1, 0.8*cm))
    
    # Meal Distribution
    story.append(Paragraph("<b>🍴 MEAL DISTRIBUTION</b>", styles['Heading2']))
    meal_data = [['Meal', 'Calories', 'Protein', 'Carbs', 'Fats']]
    for meal in nutrition_plan['meals']:
        meal_data.append([
            meal['name'],
            f"{meal['calories']} kcal",
            f"{meal['protein']}g",
            f"{meal['carbs']}g",
            f"{meal['fats']}g"
        ])
    
    t = Table(meal_data, colWidths=[5*cm, 3*cm, 2*cm, 2*cm, 2*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(brand_color)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightyellow)
    ]))
    story.append(t)
    story.append(Spacer(1, 0.8*cm))
    
    # Recommendations
    if nutrition_plan['recommendations']:
        story.append(Paragraph("<b>💡 PERSONALIZED RECOMMENDATIONS</b>", styles['Heading2']))
        for rec in nutrition_plan['recommendations']:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
    
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("<i>This nutrition plan is generated by AI and should be reviewed by a qualified sports nutritionist or dietitian.</i>",
                          styles['Italic']))
    
    doc.build(story)
    return buffer.getvalue()

# DATA LOADING
@st.cache_data
def load_sample():
    uwb = pd.read_csv('data/virtual_uwb_realistic.csv', dtype={'player_id': 'category', 'quality_factor': 'int16'})
    imu = pd.read_csv('data/virtual_imu_realistic.csv', dtype={'player_id': 'category', 'jump_detected': 'int8'})
    return uwb, imu

@st.cache_data
def load_uploaded(uwb_bytes, imu_bytes):
    uwb = pd.read_csv(uwb_bytes)
    imu = pd.read_csv(imu_bytes) if imu_bytes else None
    return uwb, imu

# =================================================================
# MAIN APP
# =================================================================

# HOMEPAGE
st.title("🏀 CoachTrack Elite AI - Professional Analytics")

# TABS (ADDED NEW TAB)
tab_analytics, tab_physical, tab_ai, tab_config = st.tabs([
    "⚙️ Configuration",
    "🏃 Physical Profile & AI", 
    "🧠 AI Elite Features", 
    "📊 Analytics & Reports"
])

# =================================================================
# TAB 4: CONFIGURATION
# =================================================================
with tab_config:
    st.header("⚙️ System Configuration")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        team_name = st.text_input("🏆 Team Name", "Elite Basketball Academy", key='team_name')
    with col2:
        session_type = st.selectbox("📋 Session Type", ["Match", "Training"], key='session_type')
    with col3:
        brand_color = st.color_picker("🎨 Brand Color", "#2563eb", key='brand_color')
    
    st.subheader("📁 Data Source")
    use_sample = st.toggle("Use sample data (recommended)", value=True)
    uwb_file, imu_file = None, None
    if not use_sample:
        col_f1, col_f2 = st.columns(2)
        with col_f1: uwb_file = st.file_uploader("UWB CSV", type=['csv'])
        with col_f2: imu_file = st.file_uploader("IMU CSV (optional)", type=['csv'])
    
    st.subheader("⏱️ Period Filter")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1: quarter = st.selectbox("Period", ['Full Session', 'Q1 (0-10min)', 'Q2 (10-20min)', 'Q3 (20-30min)', 'Q4 (30-40min)'])
    with col_p2: min_q = st.slider("Min Quality", 0, 100, 50)
    with col_p3: max_speed_clip = st.slider("Max Speed Clip", 10, 40, 30)
    
    st.divider()
    
    
    # Player name mapping
    if 'player_names' not in st.session_state:
        st.session_state.player_names = {p: p for p in all_players_temp}
    
    col_map1, col_map2, col_map3 = st.columns(3)
    for idx, pid in enumerate(all_players_temp):
        col = [col_map1, col_map2, col_map3][idx % 3]
        with col:
            st.session_state.player_names[pid] = st.text_input(f"Player {pid}", value=st.session_state.player_names.get(pid, pid), key=f'name_{pid}')
    
    
    
    # EMAIL CONFIGURATION
    st.subheader("📧 Email Configuration (Optional)")
    st.info("💡 Configure SMTP settings to enable email sending of reports to athletes")
    
    col_smtp1, col_smtp2 = st.columns(2)
    with col_smtp1:
        smtp_server = st.text_input("SMTP Server", "smtp.gmail.com", key='smtp_server',
                                     help="For Gmail: smtp.gmail.com")
        smtp_user = st.text_input("SMTP User (Email)", "", key='smtp_user',
                                  help="Your email address")
    with col_smtp2:
        smtp_port = st.number_input("SMTP Port", 587, 587, 587, key='smtp_port',
                                    help="Standard: 587 (TLS)")
        smtp_password = st.text_input("SMTP Password", "", type="password", key='smtp_password',
                                      help="For Gmail: use App Password")
    
    if smtp_user and smtp_password:
        st.success("✅ Email configuration saved")
        smtp_config = {
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'smtp_user': smtp_user,
            'smtp_password': smtp_password
        }
    else:
        smtp_config = None
        st.warning("⚠️ Email sending disabled - configure SMTP settings to enable")
    
    st.divider()
    
    st.subheader("👥 Player Name Mapping")
    st.info("💡 **Change Player Names:** Edit the mappings below to use custom names (e.g., Player_1 → 'LeBron James')")
    
    # Load data first to get player IDs
    if use_sample:
        uwb_temp, _ = load_sample()
    else:
        if not uwb_file:
            st.warning("Upload UWB file to configure player names")
            st.stop()
        uwb_temp, _ = load_uploaded(uwb_file, None)
    
    all_players_temp = sorted(uwb_temp['player_id'].unique())
    
   

# LOAD DATA
if use_sample:
    uwb, imu = load_sample()
else:
    if not uwb_file:
        st.info("Upload UWB file in Configuration tab")
        st.stop()
    uwb, imu = load_uploaded(uwb_file, imu_file)

# Validate
required = ['timestamp_s', 'player_id', 'x_m', 'y_m', 'quality_factor']
missing = [c for c in required if c not in uwb.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

uwb = uwb.sort_values(['player_id', 'timestamp_s']).copy()

# Apply filters
if quarter != 'Full Session':
    quarter_map = {'Q1 (0-10min)': (0, 600), 'Q2 (10-20min)': (600, 1200), 'Q3 (20-30min)': (1200, 1800), 'Q4 (30-40min)': (1800, 2400)}
    t_min, t_max = quarter_map[quarter]
    uwb = uwb[(uwb['timestamp_s'] >= t_min) & (uwb['timestamp_s'] < t_max)].copy()

uwb = uwb[uwb['quality_factor'] >= min_q].copy()

# Calculate metrics
uwb['dx'] = uwb.groupby('player_id')['x_m'].diff()
uwb['dy'] = uwb.groupby('player_id')['y_m'].diff()
uwb['dt'] = uwb.groupby('player_id')['timestamp_s'].diff()
uwb['step_m'] = np.sqrt(uwb['dx']**2 + uwb['dy']**2)
uwb['speed_ms_calc'] = uwb['step_m'] / uwb['dt']
uwb['speed_kmh_calc'] = (uwb['speed_ms_calc'] * 3.6).clip(upper=max_speed_clip)
uwb['accel_calc'] = uwb.groupby('player_id')['speed_kmh_calc'].diff() / uwb['dt']
uwb['zone'] = uwb.apply(lambda row: classify_zone(row['x_m'], row['y_m']), axis=1)

# Apply name mapping
uwb['player_name'] = uwb['player_id'].map(st.session_state.player_names)

# KPI
@st.cache_data
def calculate_kpi(df):
    return df.groupby(['player_id', 'player_name']).agg(
        points=('timestamp_s', 'count'),
        distance_m=('step_m', 'sum'),
        avg_speed_kmh=('speed_kmh_calc', 'mean'),
        max_speed_kmh=('speed_kmh_calc', 'max'),
        avg_quality=('quality_factor', 'mean')
    ).reset_index()

kpi = calculate_kpi(uwb)
all_players = sorted(uwb['player_id'].unique())

# =================================================================
# TAB 2: PHYSICAL PROFILE & AI
# =================================================================
with tab_physical:
    st.header("🏃 Physical Profile & AI Personalization")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 15px; margin-bottom: 20px;'>
        <h3>👤 Athlete Physical Data Management</h3>
        <p>Insert physical and anthropometric data for each athlete to enable AI-powered personalized training and nutrition plans.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize physical profiles in session state
    if 'physical_profiles' not in st.session_state:
        st.session_state.physical_profiles = {}
        # Initialize with default values for all players
        for pid in all_players:
            st.session_state.physical_profiles[pid] = {
                'height_cm': 190,
                'weight_kg': 85,
                'age': 25,
                'gender': 'Male',
                'body_fat_pct': 12,
                'vertical_jump_cm': 65,
                'wingspan_cm': 200,
                'position': 'Guard',
                'email': '',
                'phone': '',
                'birthdate': '2001-01-01',
                'nationality': ''
            }
    
    # Player selection
    physical_player = st.selectbox("👤 Select Player to Edit Physical Profile", all_players, key='physical_player_select')
    pname = st.session_state.player_names.get(physical_player, physical_player)
    
    st.subheader(f"📋 Physical Profile: {pname}")
    
    # PERSONAL DATA SECTION
    st.markdown("#### 👤 Personal Information")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    
    with col_p1:
        email = st.text_input("📧 Email", 
                             st.session_state.physical_profiles[physical_player].get('email', ''),
                             key=f'email_{physical_player}',
                             help="Email address for sending reports")
    with col_p2:
        phone = st.text_input("📱 Phone", 
                             st.session_state.physical_profiles[physical_player].get('phone', ''),
                             key=f'phone_{physical_player}',
                             help="Contact phone number")
    with col_p3:
        birthdate = st.date_input("🎂 Birthdate", 
                                  value=datetime.strptime(st.session_state.physical_profiles[physical_player].get('birthdate', '2001-01-01'), '%Y-%m-%d'),
                                  key=f'birthdate_{physical_player}',
                                  min_value=datetime(1980, 1, 1),
                                  max_value=datetime.now())
    with col_p4:
        nationality = st.text_input("🌍 Nationality", 
                                   st.session_state.physical_profiles[physical_player].get('nationality', ''),
                                   key=f'nationality_{physical_player}')
    
    st.divider()
    
    # Physical data input form
    st.markdown("#### 📏 Physical Measurements")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        height = st.number_input("📏 Height (cm)", 150, 230, 
                                st.session_state.physical_profiles[physical_player]['height_cm'],
                                key=f'height_{physical_player}')
    with col2:
        weight = st.number_input("⚖️ Weight (kg)", 50, 150,
                                st.session_state.physical_profiles[physical_player]['weight_kg'],
                                key=f'weight_{physical_player}')
    with col3:
        age = st.number_input("🎂 Age", 16, 45,
                             st.session_state.physical_profiles[physical_player]['age'],
                             key=f'age_{physical_player}')
    with col4:
        gender = st.selectbox("⚧ Gender", ['Male', 'Female'],
                             index=0 if st.session_state.physical_profiles[physical_player]['gender'] == 'Male' else 1,
                             key=f'gender_{physical_player}')
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        body_fat = st.number_input("📊 Body Fat (%)", 5, 30,
                                  st.session_state.physical_profiles[physical_player]['body_fat_pct'],
                                  key=f'bodyfat_{physical_player}')
    with col6:
        vertical = st.number_input("🦘 Vertical Jump (cm)", 30, 100,
                                  st.session_state.physical_profiles[physical_player]['vertical_jump_cm'],
                                  key=f'vertical_{physical_player}')
    with col7:
        wingspan = st.number_input("🦅 Wingspan (cm)", 150, 250,
                                  st.session_state.physical_profiles[physical_player]['wingspan_cm'],
                                  key=f'wingspan_{physical_player}')
    with col8:
        position = st.selectbox("🏀 Position", ['Point Guard', 'Shooting Guard', 'Small Forward', 'Power Forward', 'Center'],
                               index=['Point Guard', 'Shooting Guard', 'Small Forward', 'Power Forward', 'Center'].index(
                                   st.session_state.physical_profiles[physical_player].get('position', 'Guard') 
                                   if st.session_state.physical_profiles[physical_player].get('position', 'Guard') in ['Point Guard', 'Shooting Guard', 'Small Forward', 'Power Forward', 'Center']
                                   else 'Shooting Guard'
                               ),
                               key=f'position_{physical_player}')
    
    # Update profile button
    if st.button("💾 Save Physical Profile", type="primary", key=f'save_profile_{physical_player}'):
        st.session_state.physical_profiles[physical_player] = {
            'height_cm': height,
            'weight_kg': weight,
            'age': age,
            'gender': gender,
            'body_fat_pct': body_fat,
            'vertical_jump_cm': vertical,
            'wingspan_cm': wingspan,
            'position': position,
            'email': email,
            'phone': phone,
            'birthdate': birthdate.strftime('%Y-%m-%d'),
            'nationality': nationality
        }
        st.success(f"✅ Profile saved for {pname}!")
        
        # Show contact card if email/phone provided
        if email or phone:
            st.markdown(f"""
            <div class='contact-card'>
                <h4>📞 Contact Information Updated</h4>
                <p><b>Email:</b> {email if email else 'Not provided'}</p>
                <p><b>Phone:</b> {phone if phone else 'Not provided'}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # =================================================================
    # AI TRAINING GENERATION
    # =================================================================
    
    st.subheader("🤖 AI-Powered Personalized Training Plan")
    
    st.info("🧠 The AI analyzes performance data, injury risk, movement patterns, and physical profile to generate a customized training program.")
    
    training_player = st.selectbox("Select Player for Training Plan", all_players, key='training_player_select')
    training_pname = st.session_state.player_names.get(training_player, training_player)
    
    # Get player data
    player_data_training = uwb[uwb['player_id'] == training_player]
    injury_data = calculate_injury_risk(player_data_training, training_player)
    physical_profile = st.session_state.physical_profiles.get(training_player, {})
    
    if st.button("🎯 Generate AI Training Plan", type="primary", key='generate_training'):
        with st.spinner("🤖 AI analyzing performance data and generating plan..."):
            training_plan = generate_personalized_training(
                training_player,
                player_data_training,
                physical_profile,
                injury_data
            )
            
            st.session_state['current_training_plan'] = training_plan
            st.session_state['current_training_player'] = training_player
            st.session_state['current_training_pname'] = training_pname
            
            st.markdown(f"""
            <div class='ai-report-light'>
                <h3 style='color:#2563eb;'>🏋️ AI TRAINING PLAN: {training_pname}</h3>
                <p><b>Volume:</b> {training_plan['volume']}</p>
                <p><b>Intensity:</b> {training_plan['intensity']}</p>
                <p><b>Recovery:</b> {training_plan['recovery']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Warnings
            if training_plan['warnings']:
                st.warning("⚠️ **Important Alerts:**")
                for warning in training_plan['warnings']:
                    st.markdown(f"- {warning}")
            
            # Focus Areas
            st.markdown("#### 🎯 Focus Areas")
            for area in training_plan['focus_areas']:
                st.markdown(f"- {area}")
            
            # Exercises
            st.markdown("#### 💪 Recommended Exercises")
            for idx, exercise in enumerate(training_plan['exercises'], 1):
                st.markdown(f"{idx}. {exercise}")
            
            st.divider()
            
            # DOWNLOAD & EMAIL SECTION
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                # Generate PDF
                if PDF_AVAILABLE:
                    training_pdf = generate_training_pdf(
                        training_player,
                        training_pname,
                        training_plan,
                        physical_profile,
                        brand_color
                    )
                    
                    if training_pdf:
                        st.download_button(
                            "📥 Download Training Plan PDF",
                            data=training_pdf,
                            file_name=f"{training_pname}_Training_Plan.pdf",
                            mime="application/pdf",
                            key='download_training_pdf'
                        )
            
            with col_dl2:
                # Email sending
                player_email = physical_profile.get('email', '')
                if smtp_config and player_email:
                    if st.button("📧 Send via Email", key='send_training_email'):
                        training_pdf = generate_training_pdf(
                            training_player,
                            training_pname,
                            training_plan,
                            physical_profile,
                            brand_color
                        )
                        
                        email_body = f"""
                        <html>
                        <body style='font-family: Arial, sans-serif;'>
                            <h2 style='color: {brand_color};'>🏋️ Your Personalized Training Plan</h2>
                            <p>Ciao <b>{training_pname}</b>,</p>
                            <p>Il tuo piano di allenamento personalizzato AI è allegato a questa email.</p>
                            <p><b>Highlights:</b></p>
                            <ul>
                                <li>Volume: {training_plan['volume']}</li>
                                <li>Intensity: {training_plan['intensity']}</li>
                                <li>Recovery: {training_plan['recovery']}</li>
                            </ul>
                            <p>Segui attentamente le indicazioni del piano e consulta il tuo coach per qualsiasi domanda.</p>
                            <hr>
                            <p style='color: #666; font-size: 12px;'>Generated by CoachTrack Elite AI - {team_name}</p>
                        </body>
                        </html>
                        """
                        
                        success, message = send_email_with_pdf(
                            player_email,
                            training_pname,
                            f"🏋️ Your Training Plan - {team_name}",
                            email_body,
                            training_pdf,
                            f"{training_pname}_Training_Plan.pdf",
                            smtp_config
                        )
                        
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                else:
                    if not smtp_config:
                        st.warning("⚠️ Configure SMTP in Configuration tab")
                    elif not player_email:
                        st.warning("⚠️ Add player email to enable sending")
    
    st.divider()
    
    # =================================================================
    # AI NUTRITION GENERATION
    # =================================================================
    
    st.subheader("🍽️ AI-Powered Personalized Nutrition Plan")
    
    st.info("🥗 The AI calculates caloric needs (BMR, TDEE) and macronutrient distribution based on physical profile, activity level, and goals.")
    
    nutrition_player = st.selectbox("Select Player for Nutrition Plan", all_players, key='nutrition_player_select')
    nutrition_pname = st.session_state.player_names.get(nutrition_player, nutrition_player)
    
    col_n1, col_n2 = st.columns(2)
    
    with col_n1:
        activity_level = st.selectbox(
            "🏃 Activity Level",
            ["Low (Recovery)", "Moderate (Training)", "High (Intense/Match)", "Very High (Tournament)"],
            index=1,
            key='activity_level'
        )
    
    with col_n2:
        nutrition_goal = st.selectbox(
            "🎯 Goal",
            ["Maintenance", "Muscle Gain", "Fat Loss", "Performance"],
            index=3,
            key='nutrition_goal'
        )
    
    physical_profile_nutrition = st.session_state.physical_profiles.get(nutrition_player, {})
    
    if st.button("🥗 Generate AI Nutrition Plan", type="primary", key='generate_nutrition'):
        with st.spinner("🤖 AI calculating caloric needs and macros..."):
            nutrition_plan = generate_personalized_nutrition(
                nutrition_player,
                physical_profile_nutrition,
                activity_level,
                nutrition_goal
            )
            
            st.markdown(f"""
            <div class='ai-report-light'>
                <h3 style='color:#10b981;'>🍽️ AI NUTRITION PLAN: {nutrition_pname}</h3>
                <p><b>Goal:</b> {nutrition_goal} | <b>Activity:</b> {activity_level}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Caloric Overview
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🔥 BMR", f"{nutrition_plan['bmr']} kcal")
            col2.metric("⚡ TDEE", f"{nutrition_plan['tdee']} kcal")
            col3.metric("🎯 Target", f"{nutrition_plan['target_calories']} kcal")
            col4.metric("💧 Water", f"{nutrition_plan['water_liters']} L")
            
            st.divider()
            
            # Macronutrients
            st.markdown("#### 🥗 Daily Macronutrients")
            col1, col2, col3 = st.columns(3)
            col1.metric("💪 Protein", f"{nutrition_plan['protein_g']}g", 
                       f"{nutrition_plan['protein_g']/physical_profile_nutrition.get('weight_kg', 80):.1f}g/kg")
            col2.metric("🍚 Carbs", f"{nutrition_plan['carbs_g']}g")
            col3.metric("🥑 Fats", f"{nutrition_plan['fats_g']}g")
            
            # Macros visualization
            fig_macros = go.Figure(data=[go.Pie(
                labels=['Protein', 'Carbohydrates', 'Fats'],
                values=[nutrition_plan['protein_g']*4, nutrition_plan['carbs_g']*4, nutrition_plan['fats_g']*9],
                marker=dict(colors=['#ef4444', '#f59e0b', '#10b981']),
                hole=0.4
            )])
            fig_macros.update_layout(title="Macronutrient Distribution (Calories)", height=350)
            st.plotly_chart(fig_macros, use_container_width=True)
            
            st.divider()
            
            # Meal Distribution
            st.markdown("#### 🍴 Meal Distribution")
            meal_df = pd.DataFrame(nutrition_plan['meals'])
            st.dataframe(meal_df, use_container_width=True, hide_index=True)
            
            # Recommendations
            if nutrition_plan['recommendations']:
                st.markdown("#### 💡 Personalized Recommendations")
                for rec in nutrition_plan['recommendations']:
                    st.markdown(f"- {rec}")
            
            st.divider()
            
            # DOWNLOAD & EMAIL SECTION
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                # Generate PDF
                if PDF_AVAILABLE:
                    nutrition_pdf = generate_nutrition_pdf(
                        nutrition_player,
                        nutrition_pname,
                        nutrition_plan,
                        physical_profile_nutrition,
                        activity_level,
                        nutrition_goal,
                        brand_color
                    )
                    
                    if nutrition_pdf:
                        st.download_button(
                            "📥 Download Nutrition Plan PDF",
                            data=nutrition_pdf,
                            file_name=f"{nutrition_pname}_Nutrition_Plan.pdf",
                            mime="application/pdf",
                            key='download_nutrition_pdf'
                        )
            
            with col_dl2:
                # Email sending
                player_email = physical_profile_nutrition.get('email', '')
                if smtp_config and player_email:
                    if st.button("📧 Send via Email", key='send_nutrition_email'):
                        nutrition_pdf = generate_nutrition_pdf(
                            nutrition_player,
                            nutrition_pname,
                            nutrition_plan,
                            physical_profile_nutrition,
                            activity_level,
                            nutrition_goal,
                            brand_color
                        )
                        
                        email_body = f"""
                        <html>
                        <body style='font-family: Arial, sans-serif;'>
                            <h2 style='color: {brand_color};'>🍽️ Your Personalized Nutrition Plan</h2>
                            <p>Ciao <b>{nutrition_pname}</b>,</p>
                            <p>Il tuo piano nutrizionale personalizzato AI è allegato a questa email.</p>
                            <p><b>Summary:</b></p>
                            <ul>
                                <li>Target Calories: {nutrition_plan['target_calories']} kcal/day</li>
                                <li>Protein: {nutrition_plan['protein_g']}g</li>
                                <li>Carbs: {nutrition_plan['carbs_g']}g</li>
                                <li>Fats: {nutrition_plan['fats_g']}g</li>
                                <li>Water: {nutrition_plan['water_liters']}L</li>
                            </ul>
                            <p>Segui attentamente il piano e consulta un nutrizionista sportivo per personalizzazioni ulteriori.</p>
                            <hr>
                            <p style='color: #666; font-size: 12px;'>Generated by CoachTrack Elite AI - {team_name}</p>
                        </body>
                        </html>
                        """
                        
                        success, message = send_email_with_pdf(
                            player_email,
                            nutrition_pname,
                            f"🍽️ Your Nutrition Plan - {team_name}",
                            email_body,
                            nutrition_pdf,
                            f"{nutrition_pname}_Nutrition_Plan.pdf",
                            smtp_config
                        )
                        
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                else:
                    if not smtp_config:
                        st.warning("⚠️ Configure SMTP in Configuration tab")
                    elif not player_email:
                        st.warning("⚠️ Add player email to enable sending")
    
    st.divider()
    
    # =================================================================
    # TEAM OVERVIEW
    # =================================================================
    
    st.subheader("👥 Team Physical Overview")
    
    # Create overview dataframe
    overview_data = []
    for pid in all_players:
        profile = st.session_state.physical_profiles.get(pid, {})
        pname = st.session_state.player_names.get(pid, pid)
        
        # Calculate BMI
        height_m = profile.get('height_cm', 190) / 100
        weight = profile.get('weight_kg', 85)
        bmi = weight / (height_m ** 2)
        
        overview_data.append({
            'Player': pname,
            'Email': profile.get('email', 'N/A'),
            'Phone': profile.get('phone', 'N/A'),
            'Position': profile.get('position', 'N/A'),
            'Height (cm)': profile.get('height_cm', 'N/A'),
            'Weight (kg)': profile.get('weight_kg', 'N/A'),
            'Age': profile.get('age', 'N/A'),
            'BMI': f"{bmi:.1f}",
            'Body Fat (%)': profile.get('body_fat_pct', 'N/A'),
            'Vertical (cm)': profile.get('vertical_jump_cm', 'N/A')
        })
    
    overview_df = pd.DataFrame(overview_data)
    st.dataframe(overview_df, use_container_width=True, hide_index=True)
    
    # Export physical profiles
    csv_physical = overview_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Export Physical Profiles CSV",
        data=csv_physical,
        file_name=f"{team_name}_Physical_Profiles.csv",
        mime="text/csv"
    )

# =================================================================
# TAB 3: AI ELITE FEATURES (UNCHANGED)
# =================================================================
with tab_ai:
    st.header("🧠 AI Elite Features")
    
    ai_tab1, ai_tab2, ai_tab3, ai_tab4, ai_tab5, ai_tab6 = st.tabs([
        "🏥 Injury Risk", "🎯 Offensive AI", "🛡️ Defense AI", 
        "👣 Movement AI", "🏀 Shot Quality", "📉 IMU Jumps"
    ])
    
    # INJURY TAB
    with ai_tab1:
        st.subheader("🏥 Injury Risk Predictor")
        cols = st.columns(min(3, len(all_players)))
        for idx, pid in enumerate(all_players):
            pdata = uwb[uwb['player_id'] == pid]
            risk, acwr, asym, fat, level = calculate_injury_risk(pdata, pid)
            with cols[idx % 3]:
                pname = st.session_state.player_names.get(pid, pid)
                st.markdown(f"""
                <div class='predictive-card'>
                    <b>{pname}</b><br>
                    <span style='font-size:24px;'>{level}</span><br>
                    <span style='font-size:16px;'>{risk}/100</span>
                </div>
                """, unsafe_allow_html=True)
                st.metric("ACWR", f"{acwr:.2f}", "⚠️" if acwr > 1.5 else "✅")
    
    # TACTICS TAB
    with ai_tab2:
        st.subheader("🎯 Offensive Play Recommender")
        col1, col2, col3 = st.columns(3)
        with col1: curr_q = st.selectbox("Quarter", [1,2,3,4], index=3)
        with col2: score_diff = st.number_input("Score Diff", -20, 20, -3)
        with col3: curr_spacing = st.number_input("Spacing (m²)", 50, 120, 78)
        
        play, ppp, rate, when = recommend_offensive_play(curr_spacing, curr_q, score_diff)
        st.markdown(f"""
        <div class='ai-report-light'>
            <h3 style='color:#10b981;'>✅ RECOMMENDED: {play}</h3>
            <p><b>PPP:</b> {ppp:.2f} | <b>Success:</b> {rate*100:.0f}%</p>
            <p>💡 {when}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # DEFENSE TAB
    with ai_tab3:
        st.subheader("🛡️ Defensive Matchup Optimizer")
        matchups = optimize_defensive_matchups(all_players[:5])
        for m in matchups:
            pname = st.session_state.player_names.get(m['your_player'], m['your_player'])
            st.markdown(f"""
            <div style='padding:15px; background:#fff; border-left:4px solid #2563eb; margin:10px 0;'>
                <b>{pname}</b> vs {m['opponent_threat']} | Stop: {m['stop_rate']} | {m['recommendation']}
            </div>
            """, unsafe_allow_html=True)
    
    # MOVEMENT TAB
    with ai_tab4:
        st.subheader("👣 Movement Pattern Analyzer")
        pattern_player = st.selectbox("Select Player", all_players)
        pdata = uwb[uwb['player_id'] == pattern_player]
        patterns = analyze_movement_patterns(pdata)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Direction", patterns['dominant_direction'])
        col2.metric("Avg Speed", f"{patterns['avg_speed']:.1f} km/h")
        col3.metric("Zone", patterns['preferred_zone'])
        
        fig = go.Figure()
        fig.add_trace(go.Histogram2d(x=pdata['x_m'], y=pdata['y_m'], colorscale='Reds', nbinsx=40, nbinsy=20))
        fig.update_layout(shapes=draw_basketball_court(), xaxis=dict(range=[0,28], showgrid=False), 
                         yaxis=dict(range=[0,15], scaleanchor='x', scaleratio=1, showgrid=False),
                         plot_bgcolor='rgba(34,139,34,0.2)', height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # SHOT QUALITY TAB
    with ai_tab5:
        st.subheader("🏀 Shot Quality Predictor (qSQ)")
        shot_player = st.selectbox("Select Player for Shot Quality", all_players, key='shot_player')
        
        # Generate map for selected player's positions
        pdata_shot = uwb[uwb['player_id'] == shot_player]
        
        x_grid = np.linspace(0, 28, 40)
        y_grid = np.linspace(0, 15, 20)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i,j] = calculate_shot_quality(X[i,j], Y[i,j], 75) * 100
        
        fig_qsq = go.Figure(data=go.Heatmap(x=x_grid, y=y_grid, z=Z, colorscale='RdYlGn', zmin=0, zmax=100))
        fig_qsq.update_layout(shapes=draw_basketball_court(), xaxis=dict(range=[0,28], showgrid=False),
                             yaxis=dict(range=[0,15], scaleanchor='x', scaleratio=1, showgrid=False),
                             plot_bgcolor='rgba(34,139,34,0.2)', 
                             title=f'Shot Quality Map - {st.session_state.player_names.get(shot_player, shot_player)}',
                             height=500)
        st.plotly_chart(fig_qsq, use_container_width=True)
        
        # Simulator
        st.markdown("### 🎯 Shot Simulator")
        col1, col2, col3 = st.columns(3)
        with col1: sim_x = st.slider("X Position", 0.0, 28.0, 6.75, 0.5)
        with col2: sim_y = st.slider("Y Position", 0.0, 15.0, 7.5, 0.5)
        with col3: sim_spacing = st.slider("Spacing", 50, 120, 75)
        
        qsq = calculate_shot_quality(sim_x, sim_y, sim_spacing)
        color = "#10b981" if qsq > 0.50 else "#f59e0b" if qsq > 0.40 else "#ef4444"
        rec = "🟢 GREAT SHOT" if qsq > 0.50 else "🟡 ACCEPTABLE" if qsq > 0.40 else "🔴 FORCED"
        
        st.markdown(f"""
        <div class='ai-report-light' style='border-color:{color};'>
            <h3 style='color:{color};'>{rec}</h3>
            <h1 style='color:{color};'>{qsq*100:.1f}%</h1>
            <p>qSQ Score vs League Avg (46%): {(qsq-0.46)*100:+.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # IMU JUMPS TAB
    with ai_tab6:
        st.subheader("📉 IMU Jump Detection")
        if imu is None:
            st.info("No IMU data loaded")
        else:
            if quarter != 'Full Session':
                imu = imu[(imu['timestamp_s'] >= t_min) & (imu['timestamp_s'] < t_max)]
            
            jumps = (imu.get('jump_detected', 0) == 1).sum() if 'jump_detected' in imu.columns else 0
            st.metric("Total Jumps", jumps)
            
            imu_player = st.selectbox("Select Player IMU", sorted(imu['player_id'].unique()))
            imu_p = imu[imu['player_id'] == imu_player]
            
            fig_imu = px.line(imu_p, x='timestamp_s', y='accel_z_ms2', 
                             title=f'Vertical Accel - {st.session_state.player_names.get(imu_player, imu_player)}')
            
            if 'jump_detected' in imu_p.columns:
                jumps_p = imu_p[imu_p['jump_detected'] == 1]
                if not jumps_p.empty:
                    fig_imu.add_scatter(x=jumps_p['timestamp_s'], y=jumps_p['accel_z_ms2'],
                                       mode='markers', marker=dict(color='red', size=10, symbol='star'), name='Jumps')
            
            st.plotly_chart(fig_imu, use_container_width=True)

# =================================================================
# TAB 1: ANALYTICS & REPORTS
# =================================================================
with tab_analytics:
    st.header("📊 Analytics & Reports")
    
    # PDF DOWNLOADS
    if PDF_AVAILABLE:
        st.subheader("📄 PDF Reports")
        col1, col2 = st.columns(2)
        with col1:
            team_pdf = generate_team_pdf(team_name, kpi, brand_color, session_type)
            if team_pdf:
                st.download_button("📥 Team Report PDF", data=team_pdf, 
                                  file_name=f"{team_name}_Report.pdf", mime="application/pdf")
        with col2:
            player_pdf_sel = st.selectbox("Player for PDF", all_players)
            player_pdf = generate_player_pdf(player_pdf_sel, uwb[uwb['player_id']==player_pdf_sel], 
                                            team_name, brand_color, session_type)
            if player_pdf:
                pname = st.session_state.player_names.get(player_pdf_sel, player_pdf_sel)
                st.download_button(f"📥 {pname} PDF", data=player_pdf, 
                                  file_name=f"{pname}_Report.pdf", mime="application/pdf")
    
    st.divider()
    
    # KPI TABLE
    st.subheader(f"📊 Team KPI - {quarter} - {session_type}")
    st.dataframe(kpi, use_container_width=True)
    
    # CSV EXPORT
    csv = kpi.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export KPI as CSV", data=csv, file_name=f"{team_name}_KPI.csv", mime="text/csv")
    
    st.divider()
    
    # VISUALIZATIONS
    st.subheader("📍 Court Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🗺️ Player Trajectories**")
        fig_traj = go.Figure()
        sample_data = uwb.sample(min(2000, len(uwb)))
        for pid in sample_data['player_id'].unique():
            pdata = sample_data[sample_data['player_id'] == pid]
            pname = st.session_state.player_names.get(pid, pid)
            fig_traj.add_trace(go.Scatter(x=pdata['x_m'], y=pdata['y_m'], mode='markers',
                                         name=pname, marker=dict(size=4, opacity=0.6)))
        fig_traj.update_layout(shapes=draw_basketball_court(), xaxis=dict(range=[0,28], showgrid=False),
                              yaxis=dict(range=[0,15], scaleanchor='x', scaleratio=1, showgrid=False),
                              plot_bgcolor='rgba(34,139,34,0.2)', height=500, showlegend=True)
        st.plotly_chart(fig_traj, use_container_width=True)
    
    with col2:
        st.markdown("**🔥 Density Heatmap**")
        heat_player = st.selectbox("Select Player", all_players, key='heat_viz')
        heat_data = uwb[uwb['player_id'] == heat_player]
        
        fig_heat = go.Figure()
        fig_heat.add_trace(go.Histogram2d(x=heat_data['x_m'], y=heat_data['y_m'],
                                         colorscale='Viridis', nbinsx=40, nbinsy=20))
        fig_heat.update_layout(shapes=draw_basketball_court(), xaxis=dict(range=[0,28], showgrid=False),
                              yaxis=dict(range=[0,15], scaleanchor='x', scaleratio=1, showgrid=False),
                              plot_bgcolor='rgba(34,139,34,0.2)', height=500)
        st.plotly_chart(fig_heat, use_container_width=True)
    
    st.divider()
    
    # ACCELERATIONS
    st.subheader("⚡ Acceleration Analysis")
    accel_player = st.selectbox("Select Player for Accelerations", all_players, key='accel_player')
    accel_data = uwb[uwb['player_id'] == accel_player].dropna(subset=['accel_calc'])
    accel_data = accel_data[np.abs(accel_data['accel_calc']) < 50]
    
    fig_accel = go.Figure()
    fig_accel.add_trace(go.Scatter(x=accel_data['x_m'], y=accel_data['y_m'], mode='markers',
                                   marker=dict(size=8, color=accel_data['accel_calc'], 
                                             colorscale='RdYlGn', showscale=True, 
                                             colorbar=dict(title="Accel"))))
    fig_accel.update_layout(shapes=draw_basketball_court(), xaxis=dict(range=[0,28], showgrid=False),
                           yaxis=dict(range=[0,15], scaleanchor='x', scaleratio=1, showgrid=False),
                           plot_bgcolor='rgba(34,139,34,0.2)', 
                           title=f'Acceleration Map - {st.session_state.player_names.get(accel_player, accel_player)}',
                           height=500)
    st.plotly_chart(fig_accel, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Max Acceleration", f"{accel_data['accel_calc'].max():.1f} km/h/s")
    col2.metric("Max Deceleration", f"{accel_data['accel_calc'].min():.1f} km/h/s")
    col3.metric("Avg Abs Accel", f"{accel_data['accel_calc'].abs().mean():.1f} km/h/s")
    
    st.divider()
    
    # PLAYER COMPARISON
    st.subheader("🔄 Player Comparison Tool")
    col_cmp1, col_cmp2 = st.columns(2)
    with col_cmp1: player_cmp_a = st.selectbox("Player A", all_players, key='cmp_a')
    with col_cmp2: player_cmp_b = st.selectbox("Player B", all_players, index=min(1, len(all_players)-1), key='cmp_b')
    
    kpi_a = kpi[kpi['player_id'] == player_cmp_a].iloc[0]
    kpi_b = kpi[kpi['player_id'] == player_cmp_b].iloc[0]
    
    fig_radar = go.Figure()
    categories = ['Distance', 'Avg Speed', 'Max Speed', 'Quality']
    
    # Normalize values
    max_dist = kpi['distance_m'].max()
    max_avg_speed = kpi['avg_speed_kmh'].max()
    max_max_speed = kpi['max_speed_kmh'].max()
    
    values_a = [kpi_a['distance_m']/max_dist*100, kpi_a['avg_speed_kmh']/max_avg_speed*100, 
                kpi_a['max_speed_kmh']/max_max_speed*100, kpi_a['avg_quality']]
    values_b = [kpi_b['distance_m']/max_dist*100, kpi_b['avg_speed_kmh']/max_avg_speed*100,
                kpi_b['max_speed_kmh']/max_max_speed*100, kpi_b['avg_quality']]
    
    fig_radar.add_trace(go.Scatterpolar(r=values_a, theta=categories, fill='toself', 
                                       name=st.session_state.player_names.get(player_cmp_a, player_cmp_a)))
    fig_radar.add_trace(go.Scatterpolar(r=values_b, theta=categories, fill='toself',
                                       name=st.session_state.player_names.get(player_cmp_b, player_cmp_b)))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=400)
    st.plotly_chart(fig_radar, use_container_width=True)

st.divider()
st.caption(f"© 2026 {team_name} | CoachTrack Elite AI v8.0 - Email Integration | Powered by Perplexity AI")
