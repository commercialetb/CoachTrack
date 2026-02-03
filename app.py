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

# PDF GENERATION
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

# CONFIGURATION TAB
tab_config, tab_ai, tab_analytics = st.tabs(["⚙️ Configuration", "🧠 AI Elite Features", "📊 Analytics & Reports"])

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
    
    # Player name mapping
    if 'player_names' not in st.session_state:
        st.session_state.player_names = {p: p for p in all_players_temp}
    
    col_map1, col_map2, col_map3 = st.columns(3)
    for idx, pid in enumerate(all_players_temp):
        col = [col_map1, col_map2, col_map3][idx % 3]
        with col:
            st.session_state.player_names[pid] = st.text_input(f"Player {pid}", value=st.session_state.player_names.get(pid, pid), key=f'name_{pid}')

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

# AI TAB
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

# ANALYTICS TAB
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
    
    # PLAYER COMPARISON (MY EXTRA)
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
st.caption(f"© 2026 {team_name} | CoachTrack Elite AI v6.0 | Powered by Perplexity AI")
