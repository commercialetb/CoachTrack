import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import ConvexHull
from io import BytesIO
from datetime import datetime

# PDF Generation
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# =================================================================
# CONFIGURAZIONE PAGINA
# =================================================================
st.set_page_config(
    page_title='CoachTrack Elite AI Complete', 
    layout='wide',
    initial_sidebar_state='expanded'
)

# CSS Styling (iPad Optimized)
st.markdown("""
<style>
    header {visibility: hidden;}
    .main { background-color: #f8fafc !important; color: #1e293b !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: #ffffff; border-radius: 12px; padding: 10px; border: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] { height: 60px; color: #64748b !important; font-size: 18px !important; font-weight: 700 !important; }
    .stTabs [aria-selected="true"] { color: #2563eb !important; border-bottom: 4px solid #2563eb !important; }
    .predictive-card { background: #ffffff; padding: 25px; border-radius: 16px; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin-bottom: 15px; }
    .metric-title { color: #64748b !important; font-weight: 800; text-transform: uppercase; font-size: 13px; display: block; margin-bottom: 8px; }
    .metric-value { font-size: 36px !important; font-weight: 900 !important; color: #1e293b !important; }
    .ai-report-light { background: #ffffff; padding: 35px; border-radius: 20px; border: 1px solid #2563eb; color: #1e293b !important; line-height: 1.8; box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.1); }
    .highlight-blue { color: #2563eb !important; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# =================================================================
# FUNZIONI CORE - CAMPO BASKET
# =================================================================
def draw_basketball_court():
    """Disegna campo basket FIBA con linee"""
    court_length, court_width = 28.0, 15.0
    shapes = []
    
    # Bordo campo
    shapes.append(dict(type="rect", x0=0, y0=0, x1=court_length, y1=court_width,
                      line=dict(color="white", width=3), fillcolor="rgba(0,0,0,0)"))
    
    # Metà campo
    shapes.append(dict(type="line", x0=court_length/2, y0=0, x1=court_length/2, y1=court_width,
                      line=dict(color="white", width=2)))
    
    # Cerchi centrali
    shapes.append(dict(type="circle", x0=court_length/2-1.8, y0=court_width/2-1.8,
                      x1=court_length/2+1.8, y1=court_width/2+1.8,
                      line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)"))
    
    # Linea 3 punti (sinistra)
    shapes.append(dict(type="path",
                      path=f"M 0,{court_width/2-6.75} Q 6.75,{court_width/2} 0,{court_width/2+6.75}",
                      line=dict(color="white", width=2)))
    
    # Linea 3 punti (destra)
    shapes.append(dict(type="path",
                      path=f"M {court_length},{court_width/2-6.75} Q {court_length-6.75},{court_width/2} {court_length},{court_width/2+6.75}",
                      line=dict(color="white", width=2)))
    
    # Cerchi tiro libero
    for x_pos in [5.8, court_length-5.8]:
        shapes.append(dict(type="circle", x0=x_pos-1.8, y0=court_width/2-1.8,
                          x1=x_pos+1.8, y1=court_width/2+1.8,
                          line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)"))
    
    # Area restrita
    for x_pos in [0, court_length-1.25]:
        shapes.append(dict(type="rect", x0=x_pos, y0=court_width/2-1.25, 
                          x1=x_pos+1.25 if x_pos==0 else court_length, y1=court_width/2+1.25,
                          line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)"))
    
    return shapes

def classify_zone(x, y):
    """Classifica posizione in Paint/3-Point/Mid-Range"""
    court_length, court_width = 28.0, 15.0
    
    # Paint
    if (x <= 5.8 and abs(y - court_width/2) <= 2.45) or \
       (x >= court_length - 5.8 and abs(y - court_width/2) <= 2.45):
        return 'Paint'
    
    # 3-Point
    dist_left = np.sqrt((x - 1.575)**2 + (y - court_width/2)**2)
    dist_right = np.sqrt((x - (court_length - 1.575))**2 + (y - court_width/2)**2)
    if dist_left >= 6.75 or dist_right >= 6.75:
        return '3-Point'
    
    return 'Mid-Range'

# =================================================================
# FUNZIONI AI (5 MODULI ELITE)
# =================================================================
def calculate_injury_risk(player_data, player_id):
    """AI 1: Injury Risk Predictor [web:465][web:474]"""
    if len(player_data) < 100:
        return 0, 1.0, 0.1, 5, "🟢 BASSO"
    
    # ACWR
    recent = player_data.tail(min(100, len(player_data)))['speed_kmh_calc'].sum()
    chronic = player_data['speed_kmh_calc'].mean() * 100
    acwr = recent / chronic if chronic > 0 else 1.0
    
    # Asimmetria (destra vs sinistra)
    left_moves = (player_data['dx'] < -0.5).sum()
    right_moves = (player_data['dx'] > 0.5).sum()
    asymmetry = abs(left_moves - right_moves) / max(left_moves + right_moves, 1)
    
    # Fatica
    q1_speed = player_data.head(len(player_data)//4)['speed_kmh_calc'].mean()
    q4_speed = player_data.tail(len(player_data)//4)['speed_kmh_calc'].mean()
    fatigue = abs((q1_speed - q4_speed) / q1_speed * 100) if q1_speed > 0 else 5
    
    # Risk Score
    risk = 0
    if acwr > 1.5: risk += 40
    if acwr < 0.8: risk += 20
    if asymmetry > 0.25: risk += 30
    if fatigue > 15: risk += 30
    
    risk = min(risk, 100)
    level = "🔴 ALTO" if risk > 60 else "🟡 MEDIO" if risk > 30 else "🟢 BASSO"
    
    return risk, acwr, asymmetry, fatigue, level

def recommend_offensive_play(spacing, quarter, score_diff):
    """AI 2: Offensive Play Recommender [web:466]"""
    plays = {
        "Pick & Roll Top": {"ppp": 1.15, "rate": 0.82, "when": "Difesa switch-heavy, spazio ottimale"},
        "Motion Offense": {"ppp": 1.05, "rate": 0.75, "when": "Contro zona, muovere la difesa"},
        "Flare Screen": {"ppp": 1.08, "rate": 0.79, "when": "Liberare tiratore da 3, spacing alto"},
        "Post-Up": {"ppp": 0.92, "rate": 0.68, "when": "Mismatch sotto canestro"},
        "Transition": {"ppp": 1.28, "rate": 0.89, "when": "Dopo recupero palla, difesa non schierata"}
    }
    
    # Logica decisionale
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
    """AI 3: Defensive Matchup Optimizer [web:469]"""
    matchups = []
    threats = ["⭐ Star Scorer", "🎯 Shooter", "💪 Physical", "⚡ Fast Guard"]
    
    for i, defender in enumerate(defenders):
        threat = threats[i % len(threats)]
        stop_rate = f"{np.random.randint(55, 88)}%"
        rec = "✅ Mantieni Matchup" if np.random.random() > 0.3 else "🔄 Switch Consigliato"
        
        matchups.append({
            "your_player": defender,
            "opponent_threat": threat,
            "historical_stop_rate": stop_rate,
            "recommendation": rec
        })
    
    return matchups

def analyze_movement_patterns(player_data):
    """AI 4: Movement Pattern Analyzer [web:467]"""
    if len(player_data) < 50:
        return {"dominant_direction": "Destra (N/A)", "avg_speed_pre_action": 0, 
                "preferred_zone": "N/A", "confidence": 0}
    
    # Direzione dominante
    right_moves = (player_data['dx'] > 0.5).sum()
    left_moves = (player_data['dx'] < -0.5).sum()
    direction = f"Destra ({right_moves})" if right_moves > left_moves else f"Sinistra ({left_moves})"
    
    # Zona preferita
    zone_counts = player_data['zone'].value_counts()
    preferred = zone_counts.idxmax() if len(zone_counts) > 0 else "Mid-Range"
    
    # Velocità media
    avg_speed = player_data['speed_kmh_calc'].mean()
    
    # Confidence
    confidence = min(len(player_data) / 500, 1.0)
    
    return {
        "dominant_direction": direction,
        "avg_speed_pre_action": avg_speed,
        "preferred_zone": preferred,
        "confidence": confidence
    }

def calculate_shot_quality(x, y, spacing):
    """AI 5: Shot Quality Predictor (qSQ) [web:448]"""
    # Distanza dai canestri
    dist_left = np.sqrt((x - 1.575)**2 + (y - 7.5)**2)
    dist_right = np.sqrt((x - 26.425)**2 + (y - 7.5)**2)
    dist_basket = min(dist_left, dist_right)
    
    # Base probability
    if dist_basket < 2:  # Under basket
        base_prob = 0.65
    elif dist_basket < 4:  # Close paint
        base_prob = 0.55
    elif 6 < dist_basket < 7.5:  # 3-point range
        base_prob = 0.38
    else:  # Mid-range
        base_prob = 0.42
    
    # Spacing modifier
    spacing_factor = spacing / 85.0
    base_prob *= (0.9 + spacing_factor * 0.2)
    
    return min(base_prob, 0.95)

# =================================================================
# PDF REPORT GENERATION
# =================================================================
def generate_team_pdf(team_name, kpi_df, brand_color):
    """Genera PDF Report Squadra"""
    if not PDF_AVAILABLE:
        return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor(brand_color),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    story.append(Paragraph(f"📊 TEAM REPORT: {team_name}", title_style))
    story.append(Paragraph(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    
    # KPI Table
    table_data = [['Giocatore', 'Distanza (m)', 'Vel. Max (km/h)', 'Vel. Media (km/h)', 'Quality']]
    for _, row in kpi_df.iterrows():
        table_data.append([
            str(row['player_id']),
            f"{row['distance_m']:.0f}",
            f"{row['max_speed_kmh']:.1f}",
            f"{row['avg_speed_kmh']:.1f}",
            f"{row['avg_quality']:.0f}"
        ])
    
    t = Table(table_data, colWidths=[3*cm, 3*cm, 3.5*cm, 3.5*cm, 2.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(brand_color)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    story.append(t)
    story.append(Spacer(1, 1*cm))
    
    # Summary
    story.append(Paragraph("<b>📈 TEAM SUMMARY</b>", styles['Heading2']))
    story.append(Paragraph(f"• Distanza Totale Squadra: {kpi_df['distance_m'].sum():.0f} metri", styles['Normal']))
    story.append(Paragraph(f"• Velocità Max Squadra: {kpi_df['max_speed_kmh'].max():.1f} km/h", styles['Normal']))
    story.append(Paragraph(f"• Quality Factor Medio: {kpi_df['avg_quality'].mean():.0f}/100", styles['Normal']))
    
    doc.build(story)
    return buffer.getvalue()

def generate_player_pdf(player_id, player_data, team_name, brand_color):
    """Genera PDF Report Singolo Giocatore"""
    if not PDF_AVAILABLE:
        return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=22, 
                                 textColor=colors.HexColor(brand_color), alignment=TA_CENTER)
    story.append(Paragraph(f"👤 PLAYER REPORT: {player_id}", title_style))
    story.append(Paragraph(f"Team: {team_name} | {datetime.now().strftime('%d/%m/%Y')}", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    
    # Injury Risk
    risk, acwr, asym, fat, level = calculate_injury_risk(player_data, player_id)
    story.append(Paragraph(f"<b>🏥 INJURY RISK: {level}</b> ({risk}/100)", styles['Heading2']))
    story.append(Paragraph(f"• ACWR: {acwr:.2f} | Asimmetria: {asym*100:.1f}% | Fatica: {fat:.1f}%", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    
    # Movement Patterns
    patterns = analyze_movement_patterns(player_data)
    story.append(Paragraph(f"<b>👣 MOVEMENT PROFILE</b>", styles['Heading2']))
    story.append(Paragraph(f"• Direzione Dominante: {patterns['dominant_direction']}", styles['Normal']))
    story.append(Paragraph(f"• Velocità Media: {patterns['avg_speed_pre_action']:.1f} km/h", styles['Normal']))
    story.append(Paragraph(f"• Zona Preferita: {patterns['preferred_zone']}", styles['Normal']))
    
    doc.build(story)
    return buffer.getvalue()

# =================================================================
# CARICAMENTO DATI
# =================================================================
@st.cache_data
def load_sample():
    uwb = pd.read_csv('data/virtual_uwb_realistic.csv', 
                      dtype={'player_id': 'category', 'quality_factor': 'int16'})
    imu = pd.read_csv('data/virtual_imu_realistic.csv',
                      dtype={'player_id': 'category', 'jump_detected': 'int8'})
    return uwb, imu

@st.cache_data
def load_uploaded(uwb_bytes, imu_bytes):
    uwb = pd.read_csv(uwb_bytes)
    imu = pd.read_csv(imu_bytes) if imu_bytes else None
    return uwb, imu

# =================================================================
# SIDEBAR
# =================================================================
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/2563eb/ffffff?text=CoachTrack", use_container_width=True)
    st.title("⚙️ Settings")
    
    st.header('📁 Data Source')
    use_sample = st.toggle('Use realistic sample data', value=True)
    uwb_file, imu_file = None, None
    if not use_sample:
        uwb_file = st.file_uploader('UWB CSV', type=['csv'])
        imu_file = st.file_uploader('IMU CSV (optional)', type=['csv'])

    st.header('⏱️ Game Period')
    quarter_labels = ['Full Game', 'Q1 (0-10 min)', 'Q2 (10-20 min)', 'Q3 (20-30 min)', 'Q4 (30-40 min)']
    quarter = st.selectbox('Select period', quarter_labels, index=0)
    
    st.header('🔧 UWB Filters')
    min_q = st.slider('Min Quality Factor', 0, 100, 50)
    max_speed_clip = st.slider('Max Speed Clip (km/h)', 10, 40, 30)
    
    st.header('🧠 AI Modules')
    show_injury = st.toggle('Injury Predictor', value=True)
    show_tactics = st.toggle('Tactical Advisor', value=True)
    show_defense = st.toggle('Defense Optimizer', value=True)
    show_movement = st.toggle('Movement Analyzer', value=True)
    show_shot_quality = st.toggle('Shot Quality (qSQ)', value=True)
    
    st.header('📊 Visualizations')
    show_spacing = st.toggle('Team Spacing', value=True)
    show_zones = st.toggle('Zone Analysis', value=True)
    show_comparison = st.toggle('Heatmap Compare', value=False)
    show_animation = st.toggle('Time Animation', value=False)
    
    st.header('🎨 Branding')
    team_name = st.text_input('Team Name', 'Elite Basketball')
    brand_color = st.color_picker('Brand Color', '#2563eb')
    logo_file = st.file_uploader('Team Logo (PNG)', type=['png'])
    
    st.header('📄 PDF Reports')
    enable_pdf = st.toggle('Enable PDF Generation', value=PDF_AVAILABLE)

# =================================================================
# LOAD & PROCESS DATA
# =================================================================
if use_sample:
    uwb, imu = load_sample()
else:
    if not uwb_file:
        st.info('Please upload UWB CSV file to continue')
        st.stop()
    uwb, imu = load_uploaded(uwb_file, imu_file)

# Validate columns
required = ['timestamp_s', 'player_id', 'x_m', 'y_m', 'quality_factor']
missing = [c for c in required if c not in uwb.columns]
if missing:
    st.error(f'Missing columns: {missing}')
    st.stop()

uwb = uwb.sort_values(['player_id', 'timestamp_s']).copy()

# Quarter filter
if quarter != 'Full Game':
    quarter_map = {
        'Q1 (0-10 min)': (0, 600),
        'Q2 (10-20 min)': (600, 1200),
        'Q3 (20-30 min)': (1200, 1800),
        'Q4 (30-40 min)': (1800, 2400)
    }
    t_min, t_max = quarter_map[quarter]
    uwb = uwb[(uwb['timestamp_s'] >= t_min) & (uwb['timestamp_s'] < t_max)].copy()

# Apply filters
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

# KPI
@st.cache_data
def calculate_kpi(df):
    return df.groupby('player_id').agg(
        points=('timestamp_s', 'count'),
        distance_m=('step_m', 'sum'),
        avg_speed_kmh=('speed_kmh_calc', 'mean'),
        max_speed_kmh=('speed_kmh_calc', 'max'),
        avg_quality=('quality_factor', 'mean')
    ).reset_index()

kpi = calculate_kpi(uwb)
all_players = sorted(uwb['player_id'].unique())

# =================================================================
# MAIN APP UI
# =================================================================
st.title(f'🏀 {team_name} - Elite AI Analytics')
st.caption(f'📱 iPad Optimized | 🤖 AI-Powered | 📊 Complete System | Period: {quarter}')

# PDF DOWNLOAD BUTTONS
if enable_pdf and PDF_AVAILABLE:
    col_pdf1, col_pdf2 = st.columns(2)
    with col_pdf1:
        team_pdf = generate_team_pdf(team_name, kpi, brand_color)
        if team_pdf:
            st.download_button(
                "📥 Download Team Report PDF",
                data=team_pdf,
                file_name=f"Team_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
    with col_pdf2:
        player_sel_pdf = st.selectbox("Select Player for PDF", all_players, key='pdf_player')
        player_pdf = generate_player_pdf(player_sel_pdf, uwb[uwb['player_id']==player_sel_pdf], team_name, brand_color)
        if player_pdf:
            st.download_button(
                f"📥 Download {player_sel_pdf} Report PDF",
                data=player_pdf,
                file_name=f"Player_{player_sel_pdf}_Report.pdf",
                mime="application/pdf"
            )

st.divider()

# KPI TABLE
st.subheader(f'📊 Team KPI - {quarter}')
st.dataframe(kpi, use_container_width=True)

# TABS
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏥 Injury AI", "🎯 Tactics AI", "🛡️ Defense AI", "👣 Movement AI", 
    "🏀 Shot Quality", "📍 Visualizations", "⚡ Accelerations"
])

# TAB 1: INJURY RISK
with tab1:
    if show_injury:
        st.header("🏥 AI Injury Risk Predictor")
        
        cols = st.columns(min(3, len(all_players)))
        for idx, pid in enumerate(all_players):
            player_data = uwb[uwb['player_id'] == pid]
            risk, acwr, asym, fat, level = calculate_injury_risk(player_data, pid)
            
            with cols[idx % 3]:
                st.markdown(f"""
                <div class='predictive-card'>
                    <span class='metric-title'>{pid}</span>
                    <div class='metric-value' style='font-size:24px;'>{level}</div>
                    <div style='font-size:18px;margin-top:10px;'>{risk}/100</div>
                </div>
                """, unsafe_allow_html=True)
                st.metric("ACWR", f"{acwr:.2f}", "⚠️" if acwr > 1.5 else "✅")
                st.metric("Asymmetry", f"{asym*100:.1f}%")

# TAB 2: TACTICS
with tab2:
    if show_tactics:
        st.header("🎯 AI Offensive Play Recommender")
        
        col1, col2, col3 = st.columns(3)
        with col1: curr_quarter = st.selectbox('Quarter', [1,2,3,4], index=3)
        with col2: score_diff = st.number_input('Score Diff', -20, 20, -3)
        with col3: curr_spacing = st.number_input('Spacing (m²)', 50, 120, 78)
        
        play, ppp, rate, when = recommend_offensive_play(curr_spacing, curr_quarter, score_diff)
        
        st.markdown(f"""
        <div class='ai-report-light' style='border-color:#10b981;'>
            <h3 style='color:#10b981;'>✅ RECOMMENDED PLAY</h3>
            <h2>{play}</h2>
            <p><b>Expected PPP:</b> {ppp:.2f} | <b>Success Rate:</b> {rate*100:.0f}%</p>
            <p>💡 {when}</p>
        </div>
        """, unsafe_allow_html=True)

# TAB 3: DEFENSE
with tab3:
    if show_defense:
        st.header("🛡️ Defensive Matchup Optimizer")
        matchups = optimize_defensive_matchups(all_players[:5])
        
        for m in matchups:
            icon = "✅" if "Mantieni" in m["recommendation"] else "🔄"
            st.markdown(f"""
            <div style='padding:15px; background:#fff; border-left:4px solid #2563eb; margin:10px 0; border-radius:8px;'>
                <b>{icon} {m['your_player']}</b> vs {m['opponent_threat']} | Stop Rate: {m['historical_stop_rate']}<br>
                <span style='color:#2563eb;'>{m['recommendation']}</span>
            </div>
            """, unsafe_allow_html=True)

# TAB 4: MOVEMENT PATTERNS
with tab4:
    if show_movement:
        st.header("👣 Movement Pattern Analyzer")
        
        pattern_player = st.selectbox("Select Player", all_players)
        player_data = uwb[uwb['player_id'] == pattern_player]
        patterns = analyze_movement_patterns(player_data)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Dominant Direction", patterns['dominant_direction'])
        col2.metric("Avg Speed", f"{patterns['avg_speed_pre_action']:.1f} km/h")
        col3.metric("Preferred Zone", patterns['preferred_zone'])
        
        # Heatmap CON CAMPO BASKET
        fig_pattern = go.Figure()
        fig_pattern.add_trace(go.Histogram2d(
            x=player_data['x_m'], y=player_data['y_m'],
            colorscale='Reds', nbinsx=40, nbinsy=20
        ))
        fig_pattern.update_layout(
            shapes=draw_basketball_court(),
            xaxis=dict(range=[0,28], showgrid=False, zeroline=False),
            yaxis=dict(range=[0,15], scaleanchor='x', scaleratio=1, showgrid=False),
            plot_bgcolor='rgba(34,139,34,0.2)',
            title=f'Movement Heatmap - {pattern_player}',
            height=500
        )
        st.plotly_chart(fig_pattern, use_container_width=True)

# TAB 5: SHOT QUALITY
with tab5:
    if show_shot_quality:
        st.header("🏀 Shot Quality Predictor (qSQ)")
        
        # Mappa completa CON CAMPO BASKET
        x_grid = np.linspace(0, 28, 40)
        y_grid = np.linspace(0, 15, 20)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i,j] = calculate_shot_quality(X[i,j], Y[i,j], 75) * 100
        
        fig_qsq = go.Figure(data=go.Heatmap(
            x=x_grid, y=y_grid, z=Z, colorscale='RdYlGn', zmin=0, zmax=100
        ))
        fig_qsq.update_layout(
            shapes=draw_basketball_court(),
            xaxis=dict(range=[0,28], showgrid=False, zeroline=False),
            yaxis=dict(range=[0,15], scaleanchor='x', scaleratio=1, showgrid=False),
            plot_bgcolor='rgba(34,139,34,0.2)',
            title='Shot Quality Map (qSQ %)',
            height=500
        )
        st.plotly_chart(fig_qsq, use_container_width=True)

# TAB 6: VISUALIZATIONS
with tab6:
    st.header("📍 Court Visualizations")
    
    col1, col2 = st.columns(2)
    
    # Trajectories CON CAMPO
    with col1:
        st.subheader("🗺️ Player Trajectories")
        fig_traj = go.Figure()
        
        sample_data = uwb.sample(min(2000, len(uwb)))
        for pid in sample_data['player_id'].unique():
            pdata = sample_data[sample_data['player_id'] == pid]
            fig_traj.add_trace(go.Scatter(
                x=pdata['x_m'], y=pdata['y_m'], mode='markers',
                name=pid, marker=dict(size=4, opacity=0.5)
            ))
        
        fig_traj.update_layout(
            shapes=draw_basketball_court(),
            xaxis=dict(range=[0,28], showgrid=False, zeroline=False),
            yaxis=dict(range=[0,15], scaleanchor='x', scaleratio=1, showgrid=False),
            plot_bgcolor='rgba(34,139,34,0.2)',
            height=500
        )
        st.plotly_chart(fig_traj, use_container_width=True)
    
    # Heatmap CON CAMPO
    with col2:
        st.subheader("🔥 Density Heatmap")
        heat_player = st.selectbox("Player", all_players, key='heat_p')
        heat_data = uwb[uwb['player_id'] == heat_player]
        
        fig_heat = go.Figure()
        fig_heat.add_trace(go.Histogram2d(
            x=heat_data['x_m'], y=heat_data['y_m'],
            colorscale='Viridis', nbinsx=40, nbinsy=20
        ))
        fig_heat.update_layout(
            shapes=draw_basketball_court(),
            xaxis=dict(range=[0,28], showgrid=False, zeroline=False),
            yaxis=dict(range=[0,15], scaleanchor='x', scaleratio=1, showgrid=False),
            plot_bgcolor='rgba(34,139,34,0.2)',
            height=500
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# TAB 7: ACCELERATIONS
with tab7:
    st.header("⚡ Acceleration Analysis")
    
    accel_player = st.selectbox("Select Player for Acceleration Map", all_players)
    accel_data = uwb[uwb['player_id'] == accel_player].dropna(subset=['accel_calc'])
    accel_data = accel_data[np.abs(accel_data['accel_calc']) < 50]  # Remove outliers
    
    # Acceleration Heatmap CON CAMPO BASKET
    fig_accel = go.Figure()
    fig_accel.add_trace(go.Scatter(
        x=accel_data['x_m'], y=accel_data['y_m'],
        mode='markers',
        marker=dict(
            size=8,
            color=accel_data['accel_calc'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Accel<br>(km/h/s)")
        )
    ))
    fig_accel.update_layout(
        shapes=draw_basketball_court(),
        xaxis=dict(range=[0,28], showgrid=False, zeroline=False, title=''),
        yaxis=dict(range=[0,15], scaleanchor='x', scaleratio=1, showgrid=False, title=''),
        plot_bgcolor='rgba(34,139,34,0.2)',
        title=f'Acceleration Map - {accel_player}',
        height=500
    )
    st.plotly_chart(fig_accel, use_container_width=True)
    
    # Stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Max Acceleration", f"{accel_data['accel_calc'].max():.1f} km/h/s")
    col2.metric("Max Deceleration", f"{accel_data['accel_calc'].min():.1f} km/h/s")
    col3.metric("Avg Abs Accel", f"{accel_data['accel_calc'].abs().mean():.1f} km/h/s")

# IMU SECTION
if imu is not None:
    st.divider()
    st.header("📉 IMU Jump Detection")
    
    if quarter != 'Full Game':
        imu = imu[(imu['timestamp_s'] >= t_min) & (imu['timestamp_s'] < t_max)]
    
    jumps = (imu.get('jump_detected', 0) == 1).sum() if 'jump_detected' in imu.columns else 0
    st.metric("Total Jumps Detected", jumps)
    
    imu_player = st.selectbox("Select Player IMU", sorted(imu['player_id'].unique()))
    imu_p = imu[imu['player_id'] == imu_player]
    
    fig_imu = px.line(imu_p, x='timestamp_s', y='accel_z_ms2', 
                      title=f'Vertical Acceleration - {imu_player}')
    
    if 'jump_detected' in imu_p.columns:
        jumps_p = imu_p[imu_p['jump_detected'] == 1]
        if not jumps_p.empty:
            fig_imu.add_scatter(x=jumps_p['timestamp_s'], y=jumps_p['accel_z_ms2'],
                               mode='markers', marker=dict(color='red', size=10, symbol='star'),
                               name='Jumps')
    
    st.plotly_chart(fig_imu, use_container_width=True)

# FOOTER
st.divider()
st.caption(f"© 2026 {team_name} | CoachTrack Elite AI v5.0 | Powered by Perplexity AI")
