import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='Basketball Tracking MVP (Realistico)', layout='wide')
st.title('🏀 Basketball Tracking MVP - Test Realistico')
st.caption('Dataset include dropout e outlier NLOS per simulare condizioni reali indoor.')

# Basketball court drawing function
def draw_basketball_court():
    """Draw basketball court lines (half court, FIBA dimensions)"""
    court_length = 28.0  # meters
    court_width = 15.0   # meters
    
    shapes = []
    
    # Court outline
    shapes.append(dict(type="rect", x0=0, y0=0, x1=court_length, y1=court_width,
                      line=dict(color="white", width=3), fillcolor="rgba(0,0,0,0)"))
    
    # Center line
    shapes.append(dict(type="line", x0=court_length/2, y0=0, x1=court_length/2, y1=court_width,
                      line=dict(color="white", width=2)))
    
    # Center circle
    shapes.append(dict(type="circle", xref="x", yref="y",
                      x0=court_length/2-1.8, y0=court_width/2-1.8,
                      x1=court_length/2+1.8, y1=court_width/2+1.8,
                      line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)"))
    
    # 3-point line (arc) - left side
    shapes.append(dict(type="path",
                      path=f"M 0,{court_width/2-6.75} Q 6.75,{court_width/2} 0,{court_width/2+6.75}",
                      line=dict(color="white", width=2)))
    
    # 3-point line (arc) - right side
    shapes.append(dict(type="path",
                      path=f"M {court_length},{court_width/2-6.75} Q {court_length-6.75},{court_width/2} {court_length},{court_width/2+6.75}",
                      line=dict(color="white", width=2)))
    
    # Free throw circles - left
    shapes.append(dict(type="circle", x0=5.8-1.8, y0=court_width/2-1.8,
                      x1=5.8+1.8, y1=court_width/2+1.8,
                      line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)"))
    
    # Free throw circles - right
    shapes.append(dict(type="circle", x0=court_length-5.8-1.8, y0=court_width/2-1.8,
                      x1=court_length-5.8+1.8, y1=court_width/2+1.8,
                      line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)"))
    
    # Restricted area - left
    shapes.append(dict(type="rect", x0=0, y0=court_width/2-1.25, 
                      x1=1.25, y1=court_width/2+1.25,
                      line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)"))
    
    # Restricted area - right
    shapes.append(dict(type="rect", x0=court_length-1.25, y0=court_width/2-1.25,
                      x1=court_length, y1=court_width/2+1.25,
                      line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)"))
    
    return shapes

with st.sidebar:
    st.header('📁 Dati')
    use_sample = st.toggle('Usa sample realistici inclusi (consigliato)', value=True)
    uwb_file = None
    imu_file = None
    if not use_sample:
        uwb_file = st.file_uploader('UWB CSV', type=['csv'])
        imu_file = st.file_uploader('IMU CSV', type=['csv'])

    st.header('⏱️ Periodo di Gioco')
    quarter_labels = ['Intera Partita', '1° Quarto (0-10 min)', '2° Quarto (10-20 min)', 
                     '3° Quarto (20-30 min)', '4° Quarto (30-40 min)']
    quarter = st.selectbox('Seleziona periodo', quarter_labels, index=0)
    
    st.header('🔧 Filtri UWB')
    min_q = st.slider('Quality factor minima (0-100)', 0, 100, 50, 1)
    max_speed_clip = st.slider('Clip velocità (km/h) per togliere outlier', 10, 40, 30, 1)

@st.cache_data
def load_sample():
    uwb = pd.read_csv('data/virtual_uwb_realistic.csv')
    imu = pd.read_csv('data/virtual_imu_realistic.csv')
    return uwb, imu

@st.cache_data
def load_uploaded(uwb_bytes, imu_bytes):
    uwb = pd.read_csv(uwb_bytes)
    imu = pd.read_csv(imu_bytes) if imu_bytes is not None else None
    return uwb, imu

if use_sample:
    uwb, imu = load_sample()
else:
    if uwb_file is None:
        st.info('Carica almeno un file UWB per continuare.')
        st.stop()
    uwb, imu = load_uploaded(uwb_file, imu_file)

required = ['timestamp_s','player_id','x_m','y_m','quality_factor']
missing = [c for c in required if c not in uwb.columns]
if missing:
    st.error(f'UWB CSV: colonne mancanti: {missing}. Colonne disponibili: {list(uwb.columns)}')
    st.stop()

uwb = uwb.sort_values(['player_id','timestamp_s']).copy()

# Apply quarter filter
if quarter != 'Intera Partita':
    quarter_map = {
        '1° Quarto (0-10 min)': (0, 600),
        '2° Quarto (10-20 min)': (600, 1200),
        '3° Quarto (20-30 min)': (1200, 1800),
        '4° Quarto (30-40 min)': (1800, 2400)
    }
    t_min, t_max = quarter_map[quarter]
    uwb = uwb[(uwb['timestamp_s'] >= t_min) & (uwb['timestamp_s'] < t_max)].copy()

uwb = uwb[uwb['quality_factor'] >= min_q].copy()

st.subheader(f'📊 KPI per giocatore - {quarter}')

uwb['dx'] = uwb.groupby('player_id')['x_m'].diff()
uwb['dy'] = uwb.groupby('player_id')['y_m'].diff()
uwb['dt'] = uwb.groupby('player_id')['timestamp_s'].diff()
uwb['step_m'] = np.sqrt(uwb['dx']**2 + uwb['dy']**2)
uwb['speed_ms_calc'] = uwb['step_m'] / uwb['dt']
uwb['speed_kmh_calc'] = (uwb['speed_ms_calc'] * 3.6).clip(upper=max_speed_clip)

kpi = (uwb.groupby('player_id')
       .agg(points=('timestamp_s','count'),
            distance_m=('step_m','sum'),
            avg_speed_kmh=('speed_kmh_calc','mean'),
            max_speed_kmh=('speed_kmh_calc','max'),
            avg_quality=('quality_factor','mean'))
       .reset_index())

kpi['distance_m'] = kpi['distance_m'].fillna(0)
st.dataframe(kpi, use_container_width=True)

c1, c2 = st.columns([1,1])
with c1:
    st.subheader('🗺️ Traiettorie su Campo')
    fig = go.Figure()
    
    # Add player trajectories
    for player in uwb['player_id'].unique():
        player_data = uwb[uwb['player_id'] == player]
        fig.add_trace(go.Scatter(
            x=player_data['x_m'],
            y=player_data['y_m'],
            mode='markers',
            name=player,
            opacity=0.5,
            marker=dict(size=4)
        ))
    
    # Add court lines
    fig.update_layout(
        shapes=draw_basketball_court(),
        xaxis=dict(range=[0, 28], constrain='domain', showgrid=False, zeroline=False),
        yaxis=dict(range=[0, 15], scaleanchor='x', scaleratio=1, showgrid=False, zeroline=False),
        plot_bgcolor='rgba(139,69,19,0.3)',  # Wood floor color
        title='Posizioni UWB su Campo Basket',
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader('🔥 Heatmap Densità su Campo')
    
    fig2 = go.Figure()
    
    # Create 2D histogram for heatmap
    fig2.add_trace(go.Histogram2d(
        x=uwb['x_m'],
        y=uwb['y_m'],
        colorscale='Hot',
        nbinsx=60,
        nbinsy=32,
        colorbar=dict(title="Densità")
    ))
    
    # Add court lines on top
    fig2.update_layout(
        shapes=draw_basketball_court(),
        xaxis=dict(range=[0, 28], constrain='domain', showgrid=False, zeroline=False),
        yaxis=dict(range=[0, 15], scaleanchor='x', scaleratio=1, showgrid=False, zeroline=False),
        plot_bgcolor='rgba(139,69,19,0.3)',
        title='Heatmap Densità Posizioni',
        height=500
    )
    
    st.plotly_chart(fig2, use_container_width=True)

st.subheader('📈 Velocità nel tempo')
players = sorted(uwb['player_id'].unique())
sel = st.multiselect('Seleziona giocatori', players, default=players[:2] if len(players) >= 2 else players)
plot_df = uwb[uwb['player_id'].isin(sel)].copy()
fig3 = px.line(plot_df, x='timestamp_s', y='speed_kmh_calc', color='player_id', 
               title=f'Speed (km/h) - {quarter}')
st.plotly_chart(fig3, use_container_width=True)

st.subheader('📉 IMU (con rumore/bias + dropout)')
if imu is None:
    st.info('Nessun file IMU caricato (ok per test UWB-only).')
else:
    if 'timestamp_s' not in imu.columns or 'accel_z_ms2' not in imu.columns:
        st.warning(f'IMU CSV: colonne richieste mancanti. Colonne disponibili: {list(imu.columns)}')
    else:
        # Apply quarter filter to IMU
        if quarter != 'Intera Partita':
            imu = imu[(imu['timestamp_s'] >= t_min) & (imu['timestamp_s'] < t_max)].copy()
        
        jumps = int((imu.get('jump_detected', pd.Series([0]*len(imu)))==1).sum()) if 'jump_detected' in imu.columns else 0
        st.write(f'🏀 Salti rilevati in {quarter}:', jumps)
        psel = st.selectbox('Giocatore IMU', sorted(imu['player_id'].unique()))
        imu_p = imu[imu['player_id']==psel].sort_values('timestamp_s')
        fig4 = px.line(imu_p, x='timestamp_s', y='accel_z_ms2', title=f'Accel Z (m/s²) - {psel} - {quarter}')
        st.plotly_chart(fig4, use_container_width=True)