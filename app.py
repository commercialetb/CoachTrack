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
# LANGUAGE + LOGIN
# =================================================================
if "language" not in st.session_state:
    st.session_state.language = None
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

USERNAME = "coach"
PASSWORD = "basket2026"

# TRADUZIONI COMPLETE
translations = {
    "it": {
        "logout": "Esci",
        "welcome": "Benvenuto Coach!",
        "tab_config": "Configurazione",
        "tab_physical": "Profilo Fisico & AI",
        "tab_ai": "Funzioni AI Elite",
        "tab_analytics": "Analisi & Report",
        "system_config": "Configurazione Sistema",
        "team_name": "Nome Squadra",
        "session_type": "Tipo Sessione",
        "brand_color": "Colore Brand",
        "data_source": "Sorgente Dati",
        "use_sample": "Usa dati di esempio (raccomandato)",
        "period_filter": "Filtro Periodo",
        "min_quality": "Qualita Minima",
        "max_speed": "Velocita Massima",
        "email_config": "Configurazione Email (Opzionale)",
        "player_mapping": "Mappatura Nomi Giocatori",
        "match": "Partita",
        "training": "Allenamento",
        "period": "Periodo",
        "full_session": "Sessione Completa"
    },
    "en": {
        "logout": "Logout",
        "welcome": "Welcome Coach!",
        "tab_config": "Configuration",
        "tab_physical": "Physical Profile & AI",
        "tab_ai": "AI Elite Features",
        "tab_analytics": "Analytics & Reports",
        "system_config": "System Configuration",
        "team_name": "Team Name",
        "session_type": "Session Type",
        "brand_color": "Brand Color",
        "data_source": "Data Source",
        "use_sample": "Use sample data (recommended)",
        "period_filter": "Period Filter",
        "min_quality": "Min Quality",
        "max_speed": "Max Speed Clip",
        "email_config": "Email Configuration (Optional)",
        "player_mapping": "Player Name Mapping",
        "match": "Match",
        "training": "Training",
        "period": "Period",
        "full_session": "Full Session"
    }
}

def t(key):
    """Translation function"""
    return translations[st.session_state.language].get(key, key)

# LANGUAGE SELECTOR
if st.session_state.language is None:
    st.markdown("<div style='text-align:center; padding:50px 0 30px 0;'><h1 style='color:#2563eb;'>CoachTrack Elite AI</h1><p style='color:#64748b; font-size:18px;'>Professional Basketball Analytics Platform</p></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("### Select Language / Seleziona Lingua")
        if st.button("English", use_container_width=True, type="primary", key="btn_lang_en"):
            st.session_state.language = "en"
            st.rerun()
        if st.button("Italiano", use_container_width=True, type="primary", key="btn_lang_it"):
            st.session_state.language = "it"
            st.rerun()
    st.stop()

# LOGIN
if not st.session_state.authenticated:
    texts = {"it": {"title": "CoachTrack Elite AI", "subtitle": "Piattaforma Professionale di Analisi Basketball", "login_title": "Accedi", "username": "Username", "password": "Password", "login_btn": "Accedi", "error": "Username o password errati", "change_lang": "Cambia Lingua"}, "en": {"title": "CoachTrack Elite AI", "subtitle": "Professional Basketball Analytics Platform", "login_title": "Login", "username": "Username", "password": "Password", "login_btn": "Login", "error": "Invalid credentials", "change_lang": "Change Language"}}
    t_login = texts[st.session_state.language]
    st.markdown(f"<div style='text-align:center; padding:50px 0 30px 0;'><h1 style='color:#2563eb;'>{t_login['title']}</h1><p style='color:#64748b; font-size:18px;'>{t_login['subtitle']}</p></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.markdown(f"### {t_login['login_title']}")
            username = st.text_input(t_login['username'], placeholder="coach", key="login_username")
            password = st.text_input(t_login['password'], type="password", key="login_password")
            submit = st.form_submit_button(t_login['login_btn'], use_container_width=True)
            if submit:
                if username == USERNAME and password == PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error(t_login['error'])
        if st.button(t_login['change_lang'], use_container_width=True, key="btn_change_lang"):
            st.session_state.language = None
            st.rerun()
    st.stop()

def logout():
    """Logout function"""
    if st.sidebar.button(t("logout"), use_container_width=True, key="btn_logout"):
        st.session_state.authenticated = False
        st.session_state.language = None
        st.rerun()

# =================================================================
# CONFIG
# =================================================================
st.set_page_config(page_title='CoachTrack Elite AI', layout='wide', initial_sidebar_state='collapsed')
st.markdown("<style>header {visibility: hidden;}.main { background-color: #f8fafc !important; color: #1e293b !important; }.stTabs [data-baseweb='tab-list'] { background-color: #ffffff; border-radius: 12px; padding: 10px; border: 1px solid #e2e8f0; }.stTabs [data-baseweb='tab'] { height: 60px; color: #64748b !important; font-size: 16px !important; font-weight: 700 !important; }.stTabs [aria-selected='true'] { color: #2563eb !important; border-bottom: 4px solid #2563eb !important; }</style>", unsafe_allow_html=True)

st.sidebar.title(t("welcome"))
logout()
st.sidebar.divider()


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
    if len(player_data) < 100: return 0, 1.0, 0.1, 5, " LOW"
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
    level = " HIGH" if risk > 60 else " MEDIUM" if risk > 30 else " LOW"
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
    threats = [" Star", " Shooter", " Physical", " Fast"]
    for i, defender in enumerate(defenders):
        threat = threats[i % len(threats)]
        stop_rate = f"{np.random.randint(55, 88)}%"
        rec = " Keep" if np.random.random() > 0.3 else " Switch"
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
        training_plan["warnings"].append("  ACWR elevato - ridurre carico di lavoro")
    elif acwr < 0.8:
        training_plan["volume"] = "Increased (Low ACWR)"
        training_plan["warnings"].append(" ACWR basso - pu aumentare carico")
    
    # Intensity based on fatigue
    if fatigue > 15:
        training_plan["intensity"] = "Low (High Fatigue)"
        training_plan["recovery"] = "Extended - 48-72h"
        training_plan["warnings"].append(" Fatica elevata - priorit  recupero")
    elif avg_speed > 18:
        training_plan["intensity"] = "High"
        training_plan["warnings"].append(" Ottima velocit  media - mantieni intensit ")
    
    # Asymmetry correction
    if asymmetry > 0.25:
        training_plan["focus_areas"].append("Correzione asimmetria laterale")
        training_plan["exercises"].extend([
            " Single-leg drills (lato debole)",
            " Lateral bounds con focus su equilibrio",
            " Defensive slides con enfasi lato debole"
        ])
    
    # Zone-specific training
    if preferred_zone == "Paint":
        training_plan["focus_areas"].append("Potenza esplosiva sotto canestro")
        training_plan["exercises"].extend([
            " Mikan drill (50 rep)",
            " Box jumps (4x8)",
            " Post moves con contatto"
        ])
    elif preferred_zone == "3-Point":
        training_plan["focus_areas"].append("Meccanica di tiro e condizionamento")
        training_plan["exercises"].extend([
            " Spot shooting da 7 zone (10 tiri/zona)",
            " Transition 3-point drills",
            " Quick release drills"
        ])
    else:
        training_plan["focus_areas"].append("Versatilit  mid-range")
        training_plan["exercises"].extend([
            " Pull-up jumpers (5 spots x 8 rep)",
            " Pick & Roll finishing",
            " Catch-and-shoot drills"
        ])
    
    # Speed development
    if avg_speed < 12:
        training_plan["focus_areas"].append("Sviluppo velocit ")
        training_plan["exercises"].extend([
            " Sprint drills 10-20m (6x3)",
            " Acceleration ladder drills",
            " Resistance band sprints"
        ])
    
    # Conditioning based on distance
    if distance < 2000:
        training_plan["focus_areas"].append("Condizionamento aerobico")
        training_plan["exercises"].extend([
            " Continuous movement drills (12 min)",
            " Transition runs full court (8x)"
        ])
    
    # Recovery protocols
    if risk > 60:
        training_plan["recovery"] = "Priority - 72h minimum"
        training_plan["exercises"].insert(0, " Active recovery: stretching dinamico 20 min")
        training_plan["exercises"].insert(1, " Ice bath 10-12 min")
    
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
        nutrition_plan["recommendations"].append("  Body fat elevato - considera ridurre carboidrati del 10%")
    
    if activity_level == "High (Intense/Match)":
        nutrition_plan["recommendations"].append(" Aumenta idratazione: " + str(nutrition_plan["water_liters"] + 0.5) + "L")
        nutrition_plan["recommendations"].append(" Aggiungi carboidrati veloci pre-gara (banana, gel)")
    
    if protein_g / weight < 1.6:
        nutrition_plan["recommendations"].append(" Proteine ottimali: 1.6-2.2g/kg per atleti")
    
    nutrition_plan["recommendations"].append(f" Verdure ad ogni pasto principale (minimo 200g)")
    nutrition_plan["recommendations"].append(f" Omega-3: 2-3 porzioni pesce/settimana")
    
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
        
        return True, f" Email inviata con successo a {recipient_name} ({recipient_email})"
    
    except Exception as e:
        return False, f" Errore invio email: {str(e)}"

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
    story.append(Paragraph(f" {team_name} - {session_type} REPORT", title_style))
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
    story.append(Paragraph(f" Total Distance: {kpi_df['distance_m'].sum():.0f} m", styles['Normal']))
    story.append(Paragraph(f" Max Speed: {kpi_df['max_speed_kmh'].max():.1f} km/h", styles['Normal']))
    doc.build(story)
    return buffer.getvalue()

def generate_player_pdf(player_id, player_data, team_name, brand_color, session_type):
    if not PDF_AVAILABLE: return None
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm)
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor(brand_color), alignment=TA_CENTER)
    story.append(Paragraph(f" {player_id} - {session_type} REPORT", title_style))
    story.append(Paragraph(f"{team_name} | {datetime.now().strftime('%d/%m/%Y')}", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    
    risk, acwr, asym, fat, level = calculate_injury_risk(player_data, player_id)
    story.append(Paragraph(f"<b> INJURY RISK: {level}</b> ({risk}/100)", styles['Heading2']))
    story.append(Paragraph(f" ACWR: {acwr:.2f} | Asymmetry: {asym*100:.1f}% | Fatigue: {fat:.1f}%", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    
    patterns = analyze_movement_patterns(player_data)
    story.append(Paragraph(f"<b> MOVEMENT PROFILE</b>", styles['Heading2']))
    story.append(Paragraph(f" Dominant: {patterns['dominant_direction']}", styles['Normal']))
    story.append(Paragraph(f" Avg Speed: {patterns['avg_speed']:.1f} km/h", styles['Normal']))
    story.append(Paragraph(f" Preferred Zone: {patterns['preferred_zone']}", styles['Normal']))
    
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
    story.append(Paragraph(f" PERSONALIZED TRAINING PLAN", title_style))
    story.append(Paragraph(f"{player_name} ({player_id})", styles['Heading2']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.8*cm))
    
    # Physical Profile
    story.append(Paragraph("<b> PHYSICAL PROFILE</b>", styles['Heading2']))
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
    story.append(Paragraph("<b> TRAINING PARAMETERS</b>", styles['Heading2']))
    story.append(Paragraph(f"<b>Volume:</b> {training_plan['volume']}", styles['Normal']))
    story.append(Paragraph(f"<b>Intensity:</b> {training_plan['intensity']}", styles['Normal']))
    story.append(Paragraph(f"<b>Recovery:</b> {training_plan['recovery']}", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    
    # Warnings
    if training_plan['warnings']:
        story.append(Paragraph("<b>  IMPORTANT ALERTS</b>", styles['Heading2']))
        for warning in training_plan['warnings']:
            story.append(Paragraph(f" {warning}", styles['Normal']))
        story.append(Spacer(1, 0.5*cm))
    
    # Focus Areas
    story.append(Paragraph("<b> FOCUS AREAS</b>", styles['Heading2']))
    for area in training_plan['focus_areas']:
        story.append(Paragraph(f" {area}", styles['Normal']))
    story.append(Spacer(1, 0.8*cm))
    
    # Exercises
    story.append(Paragraph("<b> RECOMMENDED EXERCISES</b>", styles['Heading2']))
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
    story.append(Paragraph(f" PERSONALIZED NUTRITION PLAN", title_style))
    story.append(Paragraph(f"{player_name} ({player_id})", styles['Heading2']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.8*cm))
    
    # Physical Profile
    story.append(Paragraph("<b> ATHLETE PROFILE</b>", styles['Heading2']))
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
    story.append(Paragraph("<b> CALORIC REQUIREMENTS</b>", styles['Heading2']))
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
    story.append(Paragraph("<b> DAILY MACRONUTRIENTS</b>", styles['Heading2']))
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
    story.append(Paragraph("<b> MEAL DISTRIBUTION</b>", styles['Heading2']))
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
        story.append(Paragraph("<b> PERSONALIZED RECOMMENDATIONS</b>", styles['Heading2']))
        for rec in nutrition_plan['recommendations']:
            story.append(Paragraph(f" {rec}", styles['Normal']))
    
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

# =================================================================
# MAIN APP
# =================================================================
st.title("CoachTrack Elite AI")

tab_config, tab_physical, tab_ai, tab_analytics = st.tabs([
    t("tab_config"),
    t("tab_physical"),
    t("tab_ai"),
    t("tab_analytics")
])

with tab_config:
    st.header(t("system_config"))

    col1, col2, col3 = st.columns(3)
    with col1:
        team_name = st.text_input(t("team_name"), "Elite Basketball Academy", key='config_team_name')
    with col2:
        session_options = [t("match"), t("training")]
        session_type = st.selectbox(t("session_type"), session_options, key='config_session_type')
    with col3:
        brand_color = st.color_picker(t("brand_color"), "#2563eb", key='config_brand_color')

    st.subheader(t("data_source"))
    use_sample = st.toggle(t("use_sample"), value=True, key='config_use_sample')
    uwb_file, imu_file = None, None
    if not use_sample:
        col_f1, col_f2 = st.columns(2)
        with col_f1: uwb_file = st.file_uploader("UWB CSV", type=['csv'], key='config_uwb_file')
        with col_f2: imu_file = st.file_uploader("IMU CSV (optional)", type=['csv'], key='config_imu_file')

    st.subheader(t("period_filter"))
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1: 
        period_options = [t("full_session"), 'Q1 (0-10min)', 'Q2 (10-20min)', 'Q3 (20-30min)', 'Q4 (30-40min)']
        quarter = st.selectbox(t("period"), period_options, key='config_quarter')
    with col_p2: min_q = st.slider(t("min_quality"), 0, 100, 50, key='config_min_q')
    with col_p3: max_speed_clip = st.slider(t("max_speed"), 10, 40, 30, key='config_max_speed')

    st.divider()
    st.subheader(t("email_config"))

    col_smtp1, col_smtp2 = st.columns(2)
    with col_smtp1:
        smtp_server = st.text_input("SMTP Server", "smtp.gmail.com", key='config_smtp_server')
        smtp_user = st.text_input("SMTP User (Email)", "", key='config_smtp_user')
    with col_smtp2:
        smtp_port = st.number_input("SMTP Port", 587, 587, 587, key='config_smtp_port')
        smtp_password = st.text_input("SMTP Password", "", type="password", key='config_smtp_password')

    if smtp_user and smtp_password:
        st.success("Email configuration saved")
        smtp_config = {'smtp_server': smtp_server, 'smtp_port': smtp_port, 'smtp_user': smtp_user, 'smtp_password': smtp_password}
    else:
        smtp_config = None
        st.warning("Email sending disabled - configure SMTP settings")

    st.divider()
    st.subheader(t("player_mapping"))
    st.info("Edit the mappings below to use custom names")

    if use_sample:
        uwb_temp, _ = load_sample()
    else:
        if not uwb_file:
            st.warning("Upload UWB file to configure player names")
            st.stop()
        uwb_temp, _ = load_uploaded(uwb_file, None)

    all_players_temp = sorted(uwb_temp['player_id'].unique())

    if 'player_names' not in st.session_state:
        st.session_state.player_names = {p: p for p in all_players_temp}

    col_map1, col_map2, col_map3 = st.columns(3)
    for idx, pid in enumerate(all_players_temp):
        col = [col_map1, col_map2, col_map3][idx % 3]
        with col:
            st.session_state.player_names[pid] = st.text_input(f"Player {pid}", value=st.session_state.player_names.get(pid, pid), key=f'config_name_{pid}')

# LOAD DATA
if use_sample:
    uwb, imu = load_sample()
else:
    if not uwb_file:
        st.info("Upload UWB file in Configuration tab")
        st.stop()
    uwb, imu = load_uploaded(uwb_file, imu_file)

required = ['timestamp_s', 'player_id', 'x_m', 'y_m', 'quality_factor']
missing = [c for c in required if c not in uwb.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

uwb = uwb.sort_values(['player_id', 'timestamp_s']).copy()

if quarter != period_options[0]:
    quarter_map = {'Q1 (0-10min)': (0, 600), 'Q2 (10-20min)': (600, 1200), 'Q3 (20-30min)': (1200, 1800), 'Q4 (30-40min)': (1800, 2400)}
    if quarter in quarter_map:
        t_min, t_max = quarter_map[quarter]
        uwb = uwb[(uwb['timestamp_s'] >= t_min) & (uwb['timestamp_s'] < t_max)].copy()

uwb = uwb[uwb['quality_factor'] >= min_q].copy()
uwb['dx'] = uwb.groupby('player_id')['x_m'].diff()
uwb['dy'] = uwb.groupby('player_id')['y_m'].diff()
uwb['dt'] = uwb.groupby('player_id')['timestamp_s'].diff()
uwb['step_m'] = np.sqrt(uwb['dx']**2 + uwb['dy']**2)
uwb['speed_ms_calc'] = uwb['step_m'] / uwb['dt']
uwb['speed_kmh_calc'] = (uwb['speed_ms_calc'] * 3.6).clip(upper=max_speed_clip)
uwb['accel_calc'] = uwb.groupby('player_id')['speed_kmh_calc'].diff() / uwb['dt']
uwb['zone'] = uwb.apply(lambda row: classify_zone(row['x_m'], row['y_m']), axis=1)
uwb['player_name'] = uwb['player_id'].map(st.session_state.player_names)

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

with tab_physical:
    st.header(t("tab_physical"))
    st.info("Physical Profile & AI features - Coming soon")

with tab_ai:
    st.header(t("tab_ai"))
    st.info("AI Elite Features - Coming soon")

with tab_analytics:
    st.header(t("tab_analytics"))
    st.subheader("Team Performance Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Distance", f"{kpi['distance_m'].sum():.0f} m")
    col2.metric("Max Speed", f"{kpi['max_speed_kmh'].max():.1f} km/h")
    col3.metric("Avg Speed", f"{kpi['avg_speed_kmh'].mean():.1f} km/h")
    col4.metric("Avg Quality", f"{kpi['avg_quality'].mean():.0f}%")

    st.dataframe(kpi, use_container_width=True)
