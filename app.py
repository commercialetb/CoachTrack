import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    from reportlab.platypus import (
        SimpleDocTemplate,
        Table,
        TableStyle,
        Paragraph,
        Spacer,
        PageBreak,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# =================================================================
# CONFIGURAZIONE BASE STREAMLIT
# =================================================================
st.set_page_config(
    page_title="CoachTrack Elite AI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    header {visibility: hidden;}
    .main { background-color: #f8fafc !important; color: #1e293b !important; }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 10px;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        color: #64748b !important;
        font-size: 16px !important;
        font-weight: 700 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        border-bottom: 4px solid #2563eb !important;
    }
    .predictive-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .ai-report-light {
        background: #ffffff;
        padding: 30px;
        border-radius: 15px;
        border-left: 5px solid #2563eb;
        line-height: 1.6;
        margin: 15px 0;
    }
    .physical-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .contact-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =================================================================
# FUNZIONI CORE
# =================================================================
def draw_basketball_court():
    court_length, court_width = 28.0, 15.0
    shapes = []

    shapes.append(
        dict(
            type="rect",
            x0=0,
            y0=0,
            x1=court_length,
            y1=court_width,
            line=dict(color="white", width=3),
            fillcolor="rgba(0,0,0,0)",
        )
    )

    shapes.append(
        dict(
            type="line",
            x0=court_length / 2,
            y0=0,
            x1=court_length / 2,
            y1=court_width,
            line=dict(color="white", width=2),
        )
    )

    shapes.append(
        dict(
            type="circle",
            x0=court_length / 2 - 1.8,
            y0=court_width / 2 - 1.8,
            x1=court_length / 2 + 1.8,
            y1=court_width / 2 + 1.8,
            line=dict(color="white", width=2),
            fillcolor="rgba(0,0,0,0)",
        )
    )

    shapes.append(
        dict(
            type="path",
            path=f"M 0,{court_width/2-6.75} Q 6.75,{court_width/2} 0,{court_width/2+6.75}",
            line=dict(color="white", width=2),
        )
    )
    shapes.append(
        dict(
            type="path",
            path=f"M {court_length},{court_width/2-6.75} Q {court_length-6.75},{court_width/2} {court_length},{court_width/2+6.75}",
            line=dict(color="white", width=2),
        )
    )

    for x_pos in [5.8, court_length - 5.8]:
        shapes.append(
            dict(
                type="circle",
                x0=x_pos - 1.8,
                y0=court_width / 2 - 1.8,
                x1=x_pos + 1.8,
                y1=court_width / 2 + 1.8,
                line=dict(color="white", width=2),
                fillcolor="rgba(0,0,0,0)",
            )
        )
    return shapes


def classify_zone(x, y):
    court_length, court_width = 28.0, 15.0
    if (x <= 5.8 and abs(y - court_width / 2) <= 2.45) or (
        x >= court_length - 5.8 and abs(y - court_width / 2) <= 2.45
    ):
        return "Paint"
    dist_left = np.sqrt((x - 1.575) ** 2 + (y - court_width / 2) ** 2)
    dist_right = np.sqrt((x - (court_length - 1.575)) ** 2 + (y - court_width / 2) ** 2)
    if dist_left >= 6.75 or dist_right >= 6.75:
        return "3-Point"
    return "Mid-Range"


# =================================================================
# AI FUNCTIONS
# =================================================================
def calculate_injury_risk(player_ pd.DataFrame, player_id):
    if len(player_data) < 100:
        return 0, 1.0, 0.1, 5, "🟢 LOW"

    recent = player_data.tail(min(100, len(player_data)))["speed_kmh_calc"].sum()
    chronic = player_data["speed_kmh_calc"].mean() * 100
    acwr = recent / chronic if chronic > 0 else 1.0

    left_moves = (player_data["dx"] < -0.5).sum()
    right_moves = (player_data["dx"] > 0.5).sum()
    asymmetry = abs(left_moves - right_moves) / max(left_moves + right_moves, 1)

    q1_speed = player_data.head(len(player_data) // 4)["speed_kmh_calc"].mean()
    q4_speed = player_data.tail(len(player_data) // 4)["speed_kmh_calc"].mean()
    fatigue = abs((q1_speed - q4_speed) / q1_speed * 100) if q1_speed > 0 else 5

    risk = 0
    if acwr > 1.5:
        risk += 40
    if acwr < 0.8:
        risk += 20
    if asymmetry > 0.25:
        risk += 30
    if fatigue > 15:
        risk += 30

    risk = min(risk, 100)
    if risk > 60:
        level = "🔴 HIGH"
    elif risk > 30:
        level = "🟡 MEDIUM"
    else:
        level = "🟢 LOW"

    return risk, acwr, asymmetry, fatigue, level


def recommend_offensive_play(spacing: float, quarter: int, score_diff: int):
    plays = {
        "Pick & Roll Top": {
            "ppp": 1.15,
            "rate": 0.82,
            "when": "Switch-heavy defense",
        },
        "Motion Offense": {
            "ppp": 1.05,
            "rate": 0.75,
            "when": "Against zone",
        },
        "Flare Screen": {
            "ppp": 1.08,
            "rate": 0.79,
            "when": "High spacing",
        },
        "Transition": {
            "ppp": 1.28,
            "rate": 0.89,
            "when": "After steal",
        },
    }

    if spacing > 90 and quarter >= 3:
        best = "Pick & Roll Top"
    elif score_diff < -5 and quarter == 4:
        best = "Transition"
    elif spacing > 85:
        best = "Flare Screen"
    else:
        best = "Motion Offense"

    return best, plays[best]["ppp"], plays[best]["rate"], plays[best]["when"]


def optimize_defensive_matchups(defenders):
    matchups = []
    threats = ["⭐ Star", "🎯 Shooter", "💪 Physical", "⚡ Fast"]
    for i, defender in enumerate(defenders):
        threat = threats[i % len(threats)]
        stop_rate = f"{np.random.randint(55, 88)}%"
        rec = "✅ Keep" if np.random.random() > 0.3 else "🔄 Switch"
        matchups.append(
            {
                "your_player": defender,
                "opponent_threat": threat,
                "stop_rate": stop_rate,
                "recommendation": rec,
            }
        )
    return matchups


def analyze_movement_patterns(player_ pd.DataFrame):
    if len(player_data) < 50:
        return {
            "dominant_direction": "Right (N/A)",
            "avg_speed": 0,
            "preferred_zone": "N/A",
            "confidence": 0,
        }

    right_moves = (player_data["dx"] > 0.5).sum()
    left_moves = (player_data["dx"] < -0.5).sum()
    direction = (
        f"Right ({right_moves})" if right_moves > left_moves else f"Left ({left_moves})"
    )

    zone_counts = player_data["zone"].value_counts()
    preferred = zone_counts.idxmax() if len(zone_counts) > 0 else "Mid-Range"
    avg_speed = player_data["speed_kmh_calc"].mean()
    confidence = min(len(player_data) / 500, 1.0)

    return {
        "dominant_direction": direction,
        "avg_speed": avg_speed,
        "preferred_zone": preferred,
        "confidence": confidence,
    }


def calculate_shot_quality(x: float, y: float, spacing: float):
    dist_left = np.sqrt((x - 1.575) ** 2 + (y - 7.5) ** 2)
    dist_right = np.sqrt((x - 26.425) ** 2 + (y - 7.5) ** 2)
    dist_basket = min(dist_left, dist_right)

    if dist_basket < 2:
        base_prob = 0.65
    elif dist_basket < 4:
        base_prob = 0.55
    elif 6 < dist_basket < 7.5:
        base_prob = 0.38
    else:
        base_prob = 0.42

    spacing_factor = spacing / 85.0
    base_prob *= 0.9 + spacing_factor * 0.2
    return min(base_prob, 0.95)


# =================================================================
# AI TRAINING & NUTRITION FUNCTIONS
# =================================================================
def calculate_bmr(weight_kg, height_cm, age, gender):
    if gender == "Male":
        return (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        return (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161


def calculate_tdee(bmr, activity_level):
    activity_multipliers = {
        "Low (Recovery)": 1.3,
        "Moderate (Training)": 1.55,
        "High (Intense/Match)": 1.85,
        "Very High (Tournament)": 2.1,
    }
    return bmr * activity_multipliers.get(activity_level, 1.55)


def generate_personalized_training(player_id, player_data, physical_profile, injury_data):
    risk, acwr, asymmetry, fatigue, level = injury_data
    patterns = analyze_movement_patterns(player_data)

    avg_speed = patterns["avg_speed"]
    max_speed = player_data["speed_kmh_calc"].max() if len(player_data) > 0 else 0
    distance = player_data["step_m"].sum() if len(player_data) > 0 else 0
    preferred_zone = patterns["preferred_zone"]

    training_plan = {
        "volume": "Moderate",
        "intensity": "Moderate",
        "focus_areas": [],
        "exercises": [],
        "recovery": "Standard",
        "warnings": [],
    }

    # Volume (ACWR)
    if acwr > 1.5:
        training_plan["volume"] = "Reduced (High ACWR)"
        training_plan["warnings"].append(
            "⚠️ ACWR elevato - ridurre carico di lavoro"
        )
    elif acwr < 0.8:
        training_plan["volume"] = "Increased (Low ACWR)"
        training_plan["warnings"].append(
            "✅ ACWR basso - può aumentare carico"
        )

    # Intensità (fatigue / speed)
    if fatigue > 15:
        training_plan["intensity"] = "Low (High Fatigue)"
        training_plan["recovery"] = "Extended - 48-72h"
        training_plan["warnings"].append(
            "🛑 Fatica elevata - priorità recupero"
        )
    elif avg_speed > 18:
        training_plan["intensity"] = "High"
        training_plan["warnings"].append(
            "💪 Ottima velocità media - mantieni intensità"
        )

    # Asimmetria
    if asymmetry > 0.25:
        training_plan["focus_areas"].append("Correzione asimmetria laterale")
        training_plan["exercises"].extend(
            [
                "🔄 Single-leg drills (lato debole)",
                "🔄 Lateral bounds con focus su equilibrio",
                "🔄 Defensive slides con enfasi lato debole",
            ]
        )

    # Zona preferita
    if preferred_zone == "Paint":
        training_plan["focus_areas"].append("Potenza esplosiva sotto canestro")
        training_plan["exercises"].extend(
            [
                "🏀 Mikan drill (50 rep)",
                "💥 Box jumps (4x8)",
                "💪 Post moves con contatto",
            ]
        )
    elif preferred_zone == "3-Point":
        training_plan["focus_areas"].append("Meccanica di tiro e condizionamento")
        training_plan["exercises"].extend(
            [
                "🎯 Spot shooting da 7 zone (10 tiri/zona)",
                "🏃 Transition 3-point drills",
                "⚡ Quick release drills",
            ]
        )
    else:
        training_plan["focus_areas"].append("Versatilità mid-range")
        training_plan["exercises"].extend(
            [
                "🎯 Pull-up jumpers (5 spots x 8 rep)",
                "🔄 Pick & Roll finishing",
                "⚡ Catch-and-shoot drills",
            ]
        )

    # Velocità
    if avg_speed < 12:
        training_plan["focus_areas"].append("Sviluppo velocità")
        training_plan["exercises"].extend(
            [
                "⚡ Sprint drills 10-20m (6x3)",
                "🏃 Acceleration ladder drills",
                "💨 Resistance band sprints",
            ]
        )

    # Distanza
    if distance < 2000:
        training_plan["focus_areas"].append("Condizionamento aerobico")
        training_plan["exercises"].extend(
            [
                "🔄 Continuous movement drills (12 min)",
                "🏃 Transition runs full court (8x)",
            ]
        )

    # Recovery extra
    if risk > 60:
        training_plan["recovery"] = "Priority - 72h minimum"
        training_plan["exercises"].insert(
            0, "🧘 Active recovery: stretching dinamico 20 min"
        )
        training_plan["exercises"].insert(1, "❄️ Ice bath 10-12 min")

    return training_plan


def generate_personalized_nutrition(player_id, physical_profile, activity_level, goal):
    weight = physical_profile.get("weight_kg", 80)
    height = physical_profile.get("height_cm", 190)
    age = physical_profile.get("age", 25)
    gender = physical_profile.get("gender", "Male")
    bodyfat = physical_profile.get("body_fat_pct", 12)

    bmr = calculate_bmr(weight, height, age, gender)
    tdee = calculate_tdee(bmr, activity_level)

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
    else:
        target_calories = tdee
        protein_ratio = 0.25
        carb_ratio = 0.45
        fat_ratio = 0.30

    protein_cal = target_calories * protein_ratio
    carb_cal = target_calories * carb_ratio
    fat_cal = target_calories * fat_ratio

    protein_g = protein_cal / 4
    carbs_g = carb_cal / 4
    fats_g = fat_cal / 9

    nutrition_plan = {
        "target_calories": int(target_calories),
        "bmr": int(bmr),
        "tdee": int(tdee),
        "protein_g": int(protein_g),
        "carbs_g": int(carbs_g),
        "fats_g": int(fats_g),
        "water_liters": round(weight * 0.035, 1),
        "meals": [],
        "recommendations": [],
    }

    # Distribuzione pasti
    meals_structure = [
        {"name": "Colazione / Pre-Allenamento", "cal_pct": 0.25},
        {"name": "Snack Post-Allenamento", "cal_pct": 0.15},
        {"name": "Pranzo", "cal_pct": 0.30},
        {"name": "Snack Pomeridiano", "cal_pct": 0.10},
        {"name": "Cena", "cal_pct": 0.20},
    ]

    for meal in meals_structure:
        meal_calories = target_calories * meal["cal_pct"]
        meal_protein = meal_calories * protein_ratio / 4
        meal_carbs = meal_calories * carb_ratio / 4
        meal_fats = meal_calories * fat_ratio / 9
        nutrition_plan["meals"].append(
            {
                "name": meal["name"],
                "calories": int(meal_calories),
                "protein": int(meal_protein),
                "carbs": int(meal_carbs),
                "fats": int(meal_fats),
            }
        )

    rec = nutrition_plan["recommendations"]
    if bodyfat > 15 and goal != "Fat Loss":
        rec.append("Body fat elevato - considera ridurre carboidrati del 10%.")
    if activity_level == "High (Intense/Match)":
        rec.append(
            f"Aumenta idratazione a {nutrition_plan['water_liters'] + 0.5} L."
        )
        rec.append("Aggiungi carboidrati veloci pre-gara (banana, gel, bevanda isotonica).")
    if protein_g / weight >= 1.6:
        rec.append("Proteine ottimali 1.6-2.2 g/kg per atleti.")
    rec.append("Verdure ad ogni pasto principale (minimo 200 g).")
    rec.append("Omega-3: 2-3 porzioni di pesce a settimana.")

    return nutrition_plan


# =================================================================
# PDF GENERATION HELPERS
# =================================================================
def generate_team_pdf(team_name, kpi_df, brand_color, session_type):
    if not PDF_AVAILABLE:
        return None

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=22,
        textColor=colors.HexColor(brand_color),
        alignment=TA_CENTER,
    )

    story = []

    story.append(Paragraph(f"{team_name} - {session_type} REPORT", title_style))
    story.append(
        Paragraph(datetime.now().strftime("Date: %d/%m/%Y %H:%M"), styles["Normal"])
    )
    story.append(Spacer(1, 0.5 * cm))

    table_data = [["Player", "Distance (m)", "Max Speed", "Avg Speed", "Quality"]]
    for _, row in kpi_df.iterrows():
        table_data.append(
            [
                str(row["player_id"]),
                f"{row['distance_m']:.0f}",
                f"{row['max_speed_kmh']:.1f}",
                f"{row['avg_speed_kmh']:.1f}",
                f"{row['avg_quality']:.0f}",
            ]
        )

    t = Table(table_data, colWidths=[3 * cm, 3 * cm, 3 * cm, 3 * cm, 2.5 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(brand_color)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 1, colors.grey),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 1 * cm))

    story.append(Paragraph("<b>SUMMARY</b>", styles["Heading2"]))
    story.append(
        Paragraph(
            f"Total Distance: {kpi_df['distance_m'].sum():.0f} m", styles["Normal"]
        )
    )
    story.append(
        Paragraph(
            f"Max Speed: {kpi_df['max_speed_kmh'].max():.1f} km/h", styles["Normal"]
        )
    )

    doc.build(story)
    return buffer.getvalue()


def generate_player_pdf(player_id, player_data, team_name, brand_color, session_type):
    if not PDF_AVAILABLE:
        return None

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.HexColor(brand_color),
        alignment=TA_CENTER,
    )

    story = []

    story.append(Paragraph(f"{player_id} - {session_type} REPORT", title_style))
    story.append(
        Paragraph(
            f"{team_name} - {datetime.now().strftime('%d/%m/%Y')}", styles["Normal"]
        )
    )
    story.append(Spacer(1, 0.5 * cm))

    risk, acwr, asym, fat, level = calculate_injury_risk(player_data, player_id)
    story.append(
        Paragraph(f"<b>INJURY RISK {level}</b> ({risk}%)", styles["Heading2"])
    )
    story.append(
        Paragraph(
            f"ACWR: {acwr:.2f}  |  Asymmetry: {asym*100:.1f}%  |  Fatigue: {fat:.1f}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.5 * cm))

    patterns = analyze_movement_patterns(player_data)
    story.append(Paragraph("<b>MOVEMENT PROFILE</b>", styles["Heading2"]))
    story.append(
        Paragraph(
            f"Dominant: {patterns['dominant_direction']}", styles["Normal"]
        )
    )
    story.append(
        Paragraph(
            f"Avg Speed: {patterns['avg_speed']:.1f} km/h", styles["Normal"]
        )
    )
    story.append(
        Paragraph(f"Preferred Zone: {patterns['preferred_zone']}", styles["Normal"])
    )

    doc.build(story)
    return buffer.getvalue()


def generate_training_pdf(player_id, player_name, training_plan, physical_profile, brand_color):
    if not PDF_AVAILABLE:
        return None

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor(brand_color),
        alignment=TA_CENTER,
    )

    story = []

    story.append(Paragraph("PERSONALIZED TRAINING PLAN", title_style))
    story.append(
        Paragraph(f"{player_name} ({player_id})", styles["Heading2"])
    )
    story.append(
        Paragraph(
            datetime.now().strftime("Generated: %d/%m/%Y %H:%M"), styles["Normal"]
        )
    )
    story.append(Spacer(1, 0.8 * cm))

    story.append(Paragraph("<b>PHYSICAL PROFILE</b>", styles["Heading2"]))
    profile_data = [
        ["Metric", "Value"],
        ["Height", f"{physical_profile.get('height_cm', 'NA')} cm"],
        ["Weight", f"{physical_profile.get('weight_kg', 'NA')} kg"],
        ["Age", f"{physical_profile.get('age', 'NA')} years"],
        ["Body Fat", f"{physical_profile.get('body_fat_pct', 'NA')} %"],
        ["Vertical Jump", f"{physical_profile.get('vertical_jump_cm', 'NA')} cm"],
    ]
    t = Table(profile_data, colWidths=[6 * cm, 8 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(brand_color)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 0.8 * cm))

    story.append(Paragraph("<b>TRAINING PARAMETERS</b>", styles["Heading2"]))
    story.append(Paragraph(f"<b>Volume:</b> {training_plan['volume']}", styles["Normal"]))
    story.append(Paragraph(f"<b>Intensity:</b> {training_plan['intensity']}", styles["Normal"]))
    story.append(Paragraph(f"<b>Recovery:</b> {training_plan['recovery']}", styles["Normal"]))
    story.append(Spacer(1, 0.5 * cm))

    if training_plan["warnings"]:
        story.append(Paragraph("<b>IMPORTANT ALERTS</b>", styles["Heading2"]))
        for w in training_plan["warnings"]:
            story.append(Paragraph(f"- {w}", styles["Normal"]))
        story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("<b>FOCUS AREAS</b>", styles["Heading2"]))
    for area in training_plan["focus_areas"]:
        story.append(Paragraph(f"- {area}", styles["Normal"]))
    story.append(Spacer(1, 0.8 * cm))

    story.append(Paragraph("<b>RECOMMENDED EXERCISES</b>", styles["Heading2"]))
    for idx, ex in enumerate(training_plan["exercises"], 1):
        story.append(Paragraph(f"{idx}. {ex}", styles["Normal"]))
    story.append(Spacer(1, 1 * cm))

    story.append(
        Paragraph(
            "This plan is generated by AI based on performance data and should be reviewed by qualified coaching staff.",
            styles["Italic"],
        )
    )

    doc.build(story)
    return buffer.getvalue()


def generate_nutrition_pdf(
    player_id,
    player_name,
    nutrition_plan,
    physical_profile,
    activity_level,
    goal,
    brand_color,
):
    if not PDF_AVAILABLE:
        return None

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor(brand_color),
        alignment=TA_CENTER,
    )

    story = []

    story.append(Paragraph("PERSONALIZED NUTRITION PLAN", title_style))
    story.append(
        Paragraph(f"{player_name} ({player_id})", styles["Heading2"])
    )
    story.append(
        Paragraph(
            datetime.now().strftime("Generated: %d/%m/%Y %H:%M"), styles["Normal"]
        )
    )
    story.append(Spacer(1, 0.8 * cm))

    story.append(Paragraph("<b>ATHLETE PROFILE</b>", styles["Heading2"]))
    profile_data = [
        ["Metric", "Value"],
        ["Weight", f"{physical_profile.get('weight_kg', 'NA')} kg"],
        ["Height", f"{physical_profile.get('height_cm', 'NA')} cm"],
        ["Age", f"{physical_profile.get('age', 'NA')} years"],
        ["Body Fat", f"{physical_profile.get('body_fat_pct', 'NA')} %"],
        ["Activity Level", activity_level],
        ["Goal", goal],
    ]
    t = Table(profile_data, colWidths=[6 * cm, 8 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(brand_color)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 0.8 * cm))

    story.append(Paragraph("<b>CALORIC REQUIREMENTS</b>", styles["Heading2"]))
    cal_data = [
        ["Metric", "Value"],
        ["BMR (Basal Metabolic Rate)", f"{nutrition_plan['bmr']} kcal"],
        ["TDEE (Total Daily Energy)", f"{nutrition_plan['tdee']} kcal"],
        ["Target Calories", f"{nutrition_plan['target_calories']} kcal"],
        ["Daily Water", f"{nutrition_plan['water_liters']} liters"],
    ]
    t2 = Table(cal_data, colWidths=[8 * cm, 6 * cm])
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(brand_color)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.lightblue),
            ]
        )
    )
    story.append(t2)
    story.append(Spacer(1, 0.8 * cm))

    story.append(Paragraph("<b>DAILY MACRONUTRIENTS</b>", styles["Heading2"]))
    macro_data = [
        ["Macronutrient", "Grams", "Calories", "% of Total"],
        [
            "Protein",
            f"{nutrition_plan['protein_g']} g",
            f"{nutrition_plan['protein_g'] * 4:.0f} kcal",
            f"{nutrition_plan['protein_g']*4/nutrition_plan['target_calories']*100:.0f}%",
        ],
        [
            "Carbohydrates",
            f"{nutrition_plan['carbs_g']} g",
            f"{nutrition_plan['carbs_g'] * 4:.0f} kcal",
            f"{nutrition_plan['carbs_g']*4/nutrition_plan['target_calories']*100:.0f}%",
        ],
        [
            "Fats",
            f"{nutrition_plan['fats_g']} g",
            f"{nutrition_plan['fats_g'] * 9:.0f} kcal",
            f"{nutrition_plan['fats_g']*9/nutrition_plan['target_calories']*100:.0f}%",
        ],
    ]
    t3 = Table(macro_data, colWidths=[4 * cm, 3 * cm, 3.5 * cm, 3 * cm])
    t3.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(brand_color)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.lightgreen),
            ]
        )
    )
    story.append(t3)
    story.append(Spacer(1, 0.8 * cm))

    story.append(Paragraph("<b>MEAL DISTRIBUTION</b>", styles["Heading2"]))
    meal_data = [["Meal", "Calories", "Protein", "Carbs", "Fats"]]
    for meal in nutrition_plan["meals"]:
        meal_data.append(
            [
                meal["name"],
                f"{meal['calories']} kcal",
                f"{meal['protein']} g",
                f"{meal['carbs']} g",
                f"{meal['fats']} g",
            ]
        )
    t4 = Table(meal_data, colWidths=[5 * cm, 3 * cm, 2 * cm, 2 * cm, 2 * cm])
    t4.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(brand_color)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.lightyellow),
            ]
        )
    )
    story.append(t4)
    story.append(Spacer(1, 0.8 * cm))

    if nutrition_plan["recommendations"]:
        story.append(Paragraph("<b>PERSONALIZED RECOMMENDATIONS</b>", styles["Heading2"]))
        for r in nutrition_plan["recommendations"]:
            story.append(Paragraph(f"- {r}", styles["Normal"]))
        story.append(Spacer(1, 1 * cm))

    story.append(
        Paragraph(
            "This nutrition plan is generated by AI and should be reviewed by a qualified sports nutritionist or dietitian.",
            styles["Italic"],
        )
    )

    doc.build(story)
    return buffer.getvalue()


# =================================================================
# EMAIL
# =================================================================
def send_email_with_pdf(
    recipient_email, recipient_name, subject, body_html, pdf_data, pdf_filename, smtp_config
):
    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = smtp_config["smtp_user"]
        msg["To"] = recipient_email
        msg["Subject"] = subject

        html_part = MIMEText(body_html, "html")
        msg.attach(html_part)

        if pdf_
            pdf_attachment = MIMEApplication(pdf_data, _subtype="pdf")
            pdf_attachment.add_header(
                "Content-Disposition", "attachment", filename=pdf_filename
            )
            msg.attach(pdf_attachment)

        server = smtplib.SMTP(smtp_config["smtp_server"], smtp_config["smtp_port"])
        server.starttls()
        server.login(smtp_config["smtp_user"], smtp_config["smtp_password"])
        server.send_message(msg)
        server.quit()
        return True, f"Email inviata con successo a {recipient_name} ({recipient_email})"
    except Exception as e:
        return False, f"Errore invio email: {str(e)}"


# =================================================================
# DATA LOADING
# =================================================================
@st.cache_data
def load_sample():
    uwb = pd.read_csv(
        "data_virtual_uwb_realistic.csv",
        dtype={"player_id": "category", "quality_factor": "int16"},
    )
    imu = pd.read_csv(
        "data_virtual_imu_realistic.csv",
        dtype={"player_id": "category", "jump_detected": "int8"},
    )
    return uwb, imu


@st.cache_data
def load_uploaded_uwb(uwb_bytes, imu_bytes):
    uwb = pd.read_csv(uwb_bytes)
    imu = pd.read_csv(imu_bytes) if imu_bytes else None
    return uwb, imu


@st.cache_data
def calculate_kpi(df: pd.DataFrame):
    return (
        df.groupby(["player_id", "player_name"])
        .agg(
            points=("timestamp_s", "count"),
            distance_m=("step_m", "sum"),
            avg_speed_kmh=("speed_kmh_calc", "mean"),
            max_speed_kmh=("speed_kmh_calc", "max"),
            avg_quality=("quality_factor", "mean"),
        )
        .reset_index()
    )


# =================================================================
# TITOLO E TABS
# =================================================================
st.title("CoachTrack Elite AI - Professional Analytics")

tab_analytics, tab_physical, tab_ai, tab_config = st.tabs(
    ["📊 Analytics & Reports", "🏃 Physical Profile & AI", "🧠 AI Elite Features", "⚙️ Configuration"]
)


# =================================================================
# CONFIG TAB (⚙️) - messo in fondo ma usato come base per gli altri
# =================================================================
with tab_config:
    st.header("System Configuration")

    col1, col2, col3 = st.columns(3)
    with col1:
        team_name = st.text_input(
            "Team Name", "Elite Basketball Academy", key="team_name"
        )
    with col2:
        session_type = st.selectbox(
            "Session Type", ["Match", "Training"], key="session_type"
        )
    with col3:
        brand_color = st.color_picker("Brand Color", "#2563eb", key="brand_color")

    st.subheader("Data Source")
    use_sample = st.toggle("Use sample data (recommended)", value=True, key="use_sample")
    uwb_file, imu_file = None, None

    if not use_sample:
        colf1, colf2 = st.columns(2)
        with colf1:
            uwb_file = st.file_uploader("UWB CSV", type="csv")
        with colf2:
            imu_file = st.file_uploader("IMU CSV (optional)", type="csv")

    st.subheader("Period Filter")
    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        quarter = st.selectbox(
            "Period",
            ["Full Session", "Q1 (0-10min)", "Q2 (10-20min)", "Q3 (20-30min)", "Q4 (30-40min)"],
            key="quarter",
        )
    with colp2:
        min_q = st.slider("Min Quality", 0, 100, 50, key="min_q")
    with colp3:
        max_speed_clip = st.slider("Max Speed Clip", 10, 40, 30, key="max_speed_clip")

    st.divider()

    # Carico dati per mapping nomi
    if use_sample:
        uwb_temp, imu_temp = load_sample()
    else:
        if not uwb_file:
            st.warning("Upload UWB file to configure player names")
            st.stop()
        uwb_temp, imu_temp = load_uploaded_uwb(uwb_file, imu_file)

    all_players_temp = sorted(uwb_temp["player_id"].unique())

    st.subheader("Player Name Mapping")
    st.info(
        "Change Player Names. Edit the mappings below to use custom names "
        "(es. Player1 → LeBron James)."
    )

    if "player_names" not in st.session_state:
        st.session_state.player_names = {p: str(p) for p in all_players_temp}

    col_map1, col_map2, col_map3 = st.columns(3)
    cols_map = [col_map1, col_map2, col_map3]

    for idx, pid in enumerate(all_players_temp):
        col = cols_map[idx % 3]
        with col:
            st.session_state.player_names[pid] = st.text_input(
                f"Player {pid}",
                value=st.session_state.player_names.get(pid, str(pid)),
                key=f"name_{pid}",
            )

    st.divider()

    st.subheader("Email Configuration (Optional)")
    st.info("Configure SMTP settings to enable email sending of reports to athletes.")

    colsmtp1, colsmtp2 = st.columns(2)
    with colsmtp1:
        smtp_server = st.text_input(
            "SMTP Server",
            "smtp.gmail.com",
            key="smtp_server",
            help="For Gmail: smtp.gmail.com",
        )
        smtp_user = st.text_input(
            "SMTP User Email", "", key="smtp_user", help="Your email address"
        )
    with colsmtp2:
        smtp_port = st.number_input(
            "SMTP Port", 1, 65535, 587, key="smtp_port", help="Standard 587 (TLS)"
        )
        smtp_password = st.text_input(
            "SMTP Password",
            "",
            type="password",
            key="smtp_password",
            help="For Gmail use an App Password",
        )

    if smtp_user and smtp_password:
        st.success("Email configuration saved.")
        smtp_config = {
            "smtp_server": smtp_server,
            "smtp_port": smtp_port,
            "smtp_user": smtp_user,
            "smtp_password": smtp_password,
        }
    else:
        smtp_config = None
        st.warning("Email sending disabled - configure SMTP settings to enable.")

    st.caption("Configuration complete. Use other tabs to analyze data and generate AI content.")


# =================================================================
# CARICAMENTO DATI PRINCIPALE (usato da tutti gli altri tab)
# =================================================================
if st.session_state.get("use_sample", True):
    uwb, imu = load_sample()
else:
    # Se non in sample mode, ci affidiamo ai widget di tab_config
    if "uploaded_uwb" in st.session_state:
        uwb, imu = load_uploaded_uwb(
            st.session_state["uploaded_uwb"], st.session_state.get("uploaded_imu")
        )
    else:
        uwb, imu = load_sample()  # fallback


# Validazione base
required_cols = ["timestamp_s", "player_id", "x_m", "y_m", "quality_factor"]
missing = [c for c in required_cols if c not in uwb.columns]
if missing:
    st.error(f"Missing columns in UWB  {missing}")
    st.stop()

uwb = uwb.sort_values(["player_id", "timestamp_s"]).copy()

# Quarter filter
q = st.session_state.get("quarter", "Full Session")
if q != "Full Session":
    quarter_map = {
        "Q1 (0-10min)": (0, 600),
        "Q2 (10-20min)": (600, 1200),
        "Q3 (20-30min)": (1200, 1800),
        "Q4 (30-40min)": (1800, 2400),
    }
    t_min, t_max = quarter_map[q]
    uwb = uwb[(uwb["timestamp_s"] >= t_min) & (uwb["timestamp_s"] <= t_max)].copy()

uwb = uwb[uwb["quality_factor"] >= st.session_state.get("min_q", 50)].copy()

# Metrics
uwb["dx"] = uwb.groupby("player_id")["x_m"].diff()
uwb["dy"] = uwb.groupby("player_id")["y_m"].diff()
uwb["dt"] = uwb.groupby("player_id")["timestamp_s"].diff().replace(0, np.nan)

uwb["step_m"] = np.sqrt(uwb["dx"] ** 2 + uwb["dy"] ** 2).fillna(0)
uwb["speed_ms_calc"] = (uwb["step_m"] / uwb["dt"]).replace([np.inf, -np.inf], np.nan)
uwb["speed_kmh_calc"] = (
    uwb["speed_ms_calc"].fillna(0) * 3.6
).clip(upper=st.session_state.get("max_speed_clip", 30))
uwb["accel_calc"] = uwb.groupby("player_id")["speed_kmh_calc"].diff() / uwb["dt"]
uwb["zone"] = uwb.apply(lambda r: classify_zone(r["x_m"], r["y_m"]), axis=1)

uwb["player_name"] = uwb["player_id"].map(st.session_state.get("player_names", {}))

all_players = sorted(uwb["player_id"].unique())
kpi = calculate_kpi(uwb)


# =================================================================
# TAB 🏃 Physical Profile & AI
# =================================================================
with tab_physical:
    st.header("Physical Profile & AI")

    st.markdown(
        """
<div class='physical-card'>
  <h3>Athlete Physical Data Management</h3>
  <p>Insert physical and anthropometric data for each athlete to enable AI-powered personalized training and nutrition plans.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    if "physical_profiles" not in st.session_state:
        st.session_state.physical_profiles = {}
        for pid in all_players:
            st.session_state.physical_profiles[pid] = {
                "height_cm": 190,
                "weight_kg": 85,
                "age": 25,
                "gender": "Male",
                "body_fat_pct": 12,
                "vertical_jump_cm": 65,
                "wingspan_cm": 200,
                "position": "Guard",
                "email": "",
                "phone": "",
                "birthdate": "2001-01-01",
                "nationality": "",
            }

    physical_player = st.selectbox(
        "Select Player to Edit Physical Profile", all_players, key="physical_player"
    )
    pname = st.session_state.player_names.get(physical_player, physical_player)

    st.subheader(f"Physical Profile - {pname}")

    profile = st.session_state.physical_profiles.get(physical_player, {})

    st.markdown("**Personal Information**")
    colp1, colp2, colp3, colp4 = st.columns(4)
    with colp1:
        email = st.text_input(
            "Email",
            profile.get("email", ""),
            key=f"email_{physical_player}",
            help="Email address for sending reports",
        )
    with colp2:
        phone = st.text_input(
            "Phone",
            profile.get("phone", ""),
            key=f"phone_{physical_player}",
            help="Contact phone number",
        )
    with colp3:
        birthdate = st.date_input(
            "Birthdate",
            value=datetime.strptime(profile.get("birthdate", "2001-01-01"), "%Y-%m-%d"),
            key=f"birthdate_{physical_player}",
            min_value=datetime(1980, 1, 1),
            max_value=datetime.now(),
        )
    with colp4:
        nationality = st.text_input(
            "Nationality",
            profile.get("nationality", ""),
            key=f"nationality_{physical_player}",
        )

    st.divider()

    st.markdown("**Physical Measurements**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        height = st.number_input(
            "Height (cm)",
            150,
            230,
            int(profile.get("height_cm", 190)),
            key=f"height_{physical_player}",
        )
    with col2:
        weight = st.number_input(
            "Weight (kg)",
            50,
            150,
            int(profile.get("weight_kg", 85)),
            key=f"weight_{physical_player}",
        )
    with col3:
        age = st.number_input(
            "Age",
            16,
            45,
            int(profile.get("age", 25)),
            key=f"age_{physical_player}",
        )
    with col4:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            index=0 if profile.get("gender", "Male") == "Male" else 1,
            key=f"gender_{physical_player}",
        )

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        bodyfat = st.number_input(
            "Body Fat (%)",
            5,
            30,
            int(profile.get("body_fat_pct", 12)),
            key=f"bodyfat_{physical_player}",
        )
    with col6:
        vertical = st.number_input(
            "Vertical Jump (cm)",
            30,
            100,
            int(profile.get("vertical_jump_cm", 65)),
            key=f"vertical_{physical_player}",
        )
    with col7:
        wingspan = st.number_input(
            "Wingspan (cm)",
            150,
            250,
            int(profile.get("wingspan_cm", 200)),
            key=f"wingspan_{physical_player}",
        )
    with col8:
        pos_list = [
            "Point Guard",
            "Shooting Guard",
            "Small Forward",
            "Power Forward",
            "Center",
        ]
        current_pos = profile.get("position", "Guard")
        if current_pos not in pos_list:
            current_pos = "Shooting Guard"
        position = st.selectbox(
            "Position",
            pos_list,
            index=pos_list.index(current_pos),
            key=f"position_{physical_player}",
        )

    if st.button("💾 Save Physical Profile", type="primary", key=f"save_profile_{physical_player}"):
        st.session_state.physical_profiles[physical_player] = {
            "height_cm": height,
            "weight_kg": weight,
            "age": age,
            "gender": gender,
            "body_fat_pct": bodyfat,
            "vertical_jump_cm": vertical,
            "wingspan_cm": wingspan,
            "position": position,
            "email": email,
            "phone": phone,
            "birthdate": birthdate.strftime("%Y-%m-%d"),
            "nationality": nationality,
        }
        st.success(f"✅ Profile saved for {pname}!")

        if email or phone:
            st.markdown(
                f"""
            <div class='contact-card'>
                <h4>📞 Contact Information Updated</h4>
                <p><b>Email:</b> {email if email else 'Not provided'}</p>
                <p><b>Phone:</b> {phone if phone else 'Not provided'}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.divider()

    # AI Training
    st.subheader("🤖 AI-Powered Personalized Training Plan")
    st.info(
        "🧠 The AI analyzes performance data, injury risk, movement patterns, and physical profile to generate a customized training program."
    )

    training_player = st.selectbox(
        "Select Player for Training Plan", all_players, key="training_player"
    )
    training_pname = st.session_state.player_names.get(training_player, training_player)

    player_data_training = uwb[uwb["player_id"] == training_player]
    injury_data = calculate_injury_risk(player_data_training, training_player)
    physical_profile_training = st.session_state.physical_profiles.get(
        training_player, {}
    )

    if st.button("🎯 Generate AI Training Plan", type="primary", key="generate_training"):
        with st.spinner("🤖 AI analyzing performance data and generating plan..."):
            training_plan = generate_personalized_training(
                training_player,
                player_data_training,
                physical_profile_training,
                injury_data,
            )
            st.session_state["current_training_plan"] = training_plan
            st.session_state["current_training_player"] = training_player
            st.session_state["current_training_pname"] = training_pname

            st.markdown(
                f"""
            <div class='ai-report-light'>
                <h3 style='color:#2563eb;'>🏋️ AI TRAINING PLAN: {training_pname}</h3>
                <p><b>Volume:</b> {training_plan['volume']}</p>
                <p><b>Intensity:</b> {training_plan['intensity']}</p>
                <p><b>Recovery:</b> {training_plan['recovery']}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            if training_plan["warnings"]:
                st.warning("⚠️ Important Alerts:")
                for w in training_plan["warnings"]:
                    st.markdown(f"- {w}")

            st.markdown("#### 🎯 Focus Areas")
            for area in training_plan["focus_areas"]:
                st.markdown(f"- {area}")

            st.markdown("#### 💪 Recommended Exercises")
            for idx, ex in enumerate(training_plan["exercises"], 1):
                st.markdown(f"{idx}. {ex}")

            st.divider()

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                if PDF_AVAILABLE:
                    training_pdf = generate_training_pdf(
                        training_player,
                        training_pname,
                        training_plan,
                        physical_profile_training,
                        st.session_state.get("brand_color", "#2563eb"),
                    )
                    if training_pdf:
                        st.download_button(
                            "Download Training Plan PDF",
                            data=training_pdf,
                            file_name=f"{training_pname}_TrainingPlan.pdf",
                            mime="application/pdf",
                            key="download_training_pdf",
                        )

            with col_dl2:
                smtp_conf = smtp_config if "smtp_config" in locals() else None
                player_email = physical_profile_training.get("email", "")
                if smtp_conf and player_email:
                    if st.button("Send via Email", key="send_training_email"):
                        email_body = f"""
                        <html><body style="font-family: Arial, sans-serif;">
                        <h2 style="color:{st.session_state.get('brand_color', '#2563eb')};">Your Personalized Training Plan</h2>
                        <p>Ciao <b>{training_pname}</b>,</p>
                        <p>Il tuo piano di allenamento personalizzato AI è allegato a questa email.</p>
                        <p><b>Highlights</b></p>
                        <ul>
                            <li>Volume: {training_plan['volume']}</li>
                            <li>Intensity: {training_plan['intensity']}</li>
                            <li>Recovery: {training_plan['recovery']}</li>
                        </ul>
                        <p>Segui attentamente le indicazioni del piano e consulta il tuo coach per qualsiasi domanda.</p>
                        <hr>
                        <p style="color:#666;font-size:12px;">Generated by CoachTrack Elite AI - {team_name}</p>
                        </body></html>
                        """
                        success, message = send_email_with_pdf(
                            player_email,
                            training_pname,
                            f"Your Training Plan - {team_name}",
                            email_body,
                            training_pdf,
                            f"{training_pname}_TrainingPlan.pdf",
                            smtp_conf,
                        )
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                elif not smtp_conf:
                    st.warning("Configure SMTP in Configuration tab.")
                elif not player_email:
                    st.warning("Add player email in physical profile to enable sending.")

    st.divider()

    # AI Nutrition
    st.subheader("🥗 AI-Powered Personalized Nutrition Plan")
    st.info(
        "The AI calculates caloric needs (BMR, TDEE) and macronutrient distribution based on physical profile, activity level, and goals."
    )

    nutrition_player = st.selectbox(
        "Select Player for Nutrition Plan", all_players, key="nutrition_player"
    )
    nutrition_pname = st.session_state.player_names.get(
        nutrition_player, nutrition_player
    )

    coln1, coln2 = st.columns(2)
    with coln1:
        activity_level = st.selectbox(
            "Activity Level",
            ["Low (Recovery)", "Moderate (Training)", "High (Intense/Match)", "Very High (Tournament)"],
            index=1,
            key="activity_level",
        )
    with coln2:
        nutrition_goal = st.selectbox(
            "Goal",
            ["Maintenance", "Muscle Gain", "Fat Loss", "Performance"],
            index=3,
            key="nutrition_goal",
        )

    physical_profile_nutrition = st.session_state.physical_profiles.get(
        nutrition_player, {}
    )

    if st.button("🍽 Generate AI Nutrition Plan", type="primary", key="generate_nutrition"):
        with st.spinner("🤖 AI calculating caloric needs and macros..."):
            nutrition_plan = generate_personalized_nutrition(
                nutrition_player,
                physical_profile_nutrition,
                activity_level,
                nutrition_goal,
            )

            st.markdown(
                f"""
            <div class='ai-report-light'>
                <h3 style='color:#10b981;'>AI NUTRITION PLAN: {nutrition_pname}</h3>
                <p><b>Goal:</b> {nutrition_goal} | <b>Activity:</b> {activity_level}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("BMR", f"{nutrition_plan['bmr']} kcal")
            with col2:
                st.metric("TDEE", f"{nutrition_plan['tdee']} kcal")
            with col3:
                st.metric("Target", f"{nutrition_plan['target_calories']} kcal")
            with col4:
                st.metric("Water", f"{nutrition_plan['water_liters']} L")

            st.markdown("**Daily Macronutrients**")
            colm1, colm2, colm3 = st.columns(3)
            with colm1:
                st.metric(
                    "Protein",
                    f"{nutrition_plan['protein_g']} g",
                    f"{nutrition_plan['protein_g']/max(physical_profile_nutrition.get('weight_kg',80),1):.1f} g/kg",
                )
            with colm2:
                st.metric("Carbs", f"{nutrition_plan['carbs_g']} g")
            with colm3:
                st.metric("Fats", f"{nutrition_plan['fats_g']} g")

            st.markdown
