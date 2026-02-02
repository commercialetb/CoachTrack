import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title='Basketball Tracking MVP (Realistico)', 
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .stMultiSelect, .stSelectbox, .stSlider {
            font-size: 0.9rem;
        }
        .stDataFrame {
            font-size: 0.85rem;
        }
        /* Make columns stack on mobile */
        .element-container {
            width: 100% !important;
        }
    }
    
    /* Improve readability */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        margin-top: 1rem;
    }
    
    /* Better spacing for buttons */
    .stButton button {
        width: 100%;
    }
    
    /* Zone analysis cards */
    .zone-card {
        padding: 1rem;
        border-radius: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 0.5rem 0;
    }
    
    /* AI insights box */
    .ai-insight {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title('🏀 Basketball Tracking MVP - Enhanced Edition')
st.caption('Dataset include dropout e outlier NLOS per simulare condizioni reali indoor.')
st.caption('📱 **Responsive Design** + 🤖 **AI Analysis** + 🎯 **Zone Tracking**')

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

# Zone classification function
def classify_zone(x, y):
    """Classify court position into zones (FIBA dimensions)"""
    court_length = 28.0
    court_width = 15.0
    
    # Paint area: within 5.8m from basket (free throw line)
    # Left basket
    if x <= 5.8 and (court_width/2 - 2.45) <= y <= (court_width/2 + 2.45):
        return 'Paint'
    # Right basket
    if x >= (court_length - 5.8) and (court_width/2 - 2.45) <= y <= (court_width/2 + 2.45):
        return 'Paint'
    
    # 3-point zone: beyond 6.75m from basket
    # Left side
    left_basket_x, left_basket_y = 1.575, court_width/2
    dist_left = np.sqrt((x - left_basket_x)**2 + (y - left_basket_y)**2)
    if dist_left >= 6.75:
        return '3-Point'
    
    # Right side
    right_basket_x, right_basket_y = court_length - 1.575, court_width/2
    dist_right = np.sqrt((x - right_basket_x)**2 + (y - right_basket_y)**2)
    if dist_right >= 6.75:
        return '3-Point'
    
    # Everything else is mid-range
    return 'Mid-Range'

# AI Analysis function (uses Groq API if available)
def generate_ai_insights(kpi_df, zone_df=None):
    """Generate AI insights using Groq API or fallback to rule-based"""
    try:
        # Try to import groq
        import groq
        
        # Check if API key exists in secrets
        if 'GROQ_API_KEY' in st.secrets:
            client = groq.Groq(api_key=st.secrets["GROQ_API_KEY"])
            
            # Prepare data summary
            data_summary = f"""
            Analizza questi dati di performance basket:
            
            KPI Giocatori:
            {kpi_df.to_string()}
            
            Fornisci insights brevi e actionable in italiano (max 3 punti).
            """
            
            response = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{
                    "role": "user",
                    "content": data_summary
                }],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        else:
            raise Exception("API key not found")
    
    except Exception as e:
        # Fallback to rule-based insights
        insights = []
        
        # Top performer
        top_player = kpi_df.loc[kpi_df['distance_m'].idxmax()]
        insights.append(f"🏆 **Top Performer**: {top_player['player_id']} ha percorso {top_player['distance_m']:.0f}m - il più attivo!")
        
        # Speed analysis
        fastest = kpi_df.loc[kpi_df['max_speed_kmh'].idxmax()]
        insights.append(f"⚡ **Velocità Massima**: {fastest['player_id']} ha raggiunto {fastest['max_speed_kmh']:.1f} km/h!")
        
        # Quality check
        avg_quality = kpi_df['avg_quality'].mean()
        if avg_quality < 60:
            insights.append(f"⚠️ **Attenzione**: Quality factor medio basso ({avg_quality:.0f}) - possibili interferenze NLOS")
        else:
            insights.append(f"✅ **Tracking Quality**: Ottima ({avg_quality:.0f}/100) - dati affidabili!")
        
        return "\n\n".join(insights)

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
    
    st.header('🤖 AI Analysis')
    enable_ai = st.toggle('Abilita AI Insights', value=True)
    
    st.header('🎯 Advanced Features')
    show_zones = st.toggle('Mostra Zone Analysis', value=True)
    show_comparison = st.toggle('Confronto Heatmap', value=False)
    show_animation = st.toggle('Animazione Temporale', value=False)

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

# Calculate derived metrics
uwb['dx'] = uwb.groupby('player_id')['x_m'].diff()
uwb['dy'] = uwb.groupby('player_id')['y_m'].diff()
uwb['dt'] = uwb.groupby('player_id')['timestamp_s'].diff()
uwb['step_m'] = np.sqrt(uwb['dx']**2 + uwb['dy']**2)
uwb['speed_ms_calc'] = uwb['step_m'] / uwb['dt']
uwb['speed_kmh_calc'] = (uwb['speed_ms_calc'] * 3.6).clip(upper=max_speed_clip)

# Add zone classification
uwb['zone'] = uwb.apply(lambda row: classify_zone(row['x_m'], row['y_m']), axis=1)

st.subheader(f'📊 KPI per giocatore - {quarter}')

@st.cache_data
def calculate_kpi(uwb_data):
    kpi = (uwb_data.groupby('player_id')
           .agg(points=('timestamp_s','count'),
                distance_m=('step_m','sum'),
                avg_speed_kmh=('speed_kmh_calc','mean'),
                max_speed_kmh=('speed_kmh_calc','max'),
                avg_quality=('quality_factor','mean'))
           .reset_index())
    kpi['distance_m'] = kpi['distance_m'].fillna(0)
    return kpi

cache_key = f"{len(uwb)}_{uwb['timestamp_s'].min()}_{uwb['timestamp_s'].max()}"
kpi = calculate_kpi(uwb.copy())

st.dataframe(kpi, use_container_width=True)

# AI Insights Section
if enable_ai:
    with st.expander('🤖 AI Insights & Recommendations', expanded=True):
        with st.spinner('Analyzing performance data...'):
            insights = generate_ai_insights(kpi)
            st.markdown(f'<div class="ai-insight">{insights}</div>', unsafe_allow_html=True)
            st.caption('💡 Powered by Groq AI (Llama 3.1 70B) - Configure API key in Streamlit secrets for enhanced insights')

# Zone Analysis Section
if show_zones:
    st.subheader('🎯 Zone Analysis - Distribuzione sul Campo')
    
    @st.cache_data
    def calculate_zone_stats(uwb_data):
        zone_stats = (uwb_data.groupby(['player_id', 'zone'])
                      .size()
                      .reset_index(name='count'))
        zone_totals = zone_stats.groupby('player_id')['count'].transform('sum')
        zone_stats['percentage'] = (zone_stats['count'] / zone_totals * 100).round(1)
        return zone_stats
    
    zone_stats = calculate_zone_stats(uwb[['player_id', 'zone']].copy())
    
    # Select player for zone analysis
    zone_player = st.selectbox('Seleziona giocatore per zone analysis', 
                               sorted(uwb['player_id'].unique()), 
                               key='zone_player_select')
    
    player_zones = zone_stats[zone_stats['player_id'] == zone_player]
    
    col_z1, col_z2 = st.columns([1, 1])
    
    with col_z1:
        st.write('**Tabella Zone**')
        st.dataframe(player_zones[['zone', 'count', 'percentage']], use_container_width=True)
    
    with col_z2:
        st.write('**Distribuzione Percentuale**')
        fig_pie = px.pie(player_zones, values='percentage', names='zone', 
                         title=f'Zone Distribution - {zone_player}',
                         color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_pie, use_container_width=True)

# Player filter section
st.subheader('👤 Filtro Giocatore')
all_players = sorted(uwb['player_id'].unique())
player_filter = st.multiselect(
    'Seleziona giocatori da visualizzare (lascia vuoto per tutti)',
    options=all_players,
    default=all_players,
    help='Seleziona uno o più giocatori per filtrare le visualizzazioni'
)

# Apply player filter
if player_filter:
    uwb_filtered = uwb[uwb['player_id'].isin(player_filter)].copy()
else:
    uwb_filtered = uwb.copy()

# Heatmap Comparison (Side-by-Side)
if show_comparison and len(all_players) >= 2:
    st.subheader('🔥 Confronto Heatmap - Side by Side')
    
    col_cmp1, col_cmp2 = st.columns(2)
    
    with col_cmp1:
        player_a = st.selectbox('Giocatore A', all_players, index=0, key='cmp_a')
    
    with col_cmp2:
        player_b = st.selectbox('Giocatore B', all_players, index=min(1, len(all_players)-1), key='cmp_b')
    
    # Create side-by-side subplots
    fig_comparison = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'{player_a}', f'{player_b}'),
        specs=[[{'type': 'histogram2d'}, {'type': 'histogram2d'}]]
    )
    
    data_a = uwb_filtered[uwb_filtered['player_id'] == player_a]
    data_b = uwb_filtered[uwb_filtered['player_id'] == player_b]
    
    fig_comparison.add_trace(
        go.Histogram2d(x=data_a['x_m'], y=data_a['y_m'], 
                      colorscale='Hot', showscale=False,
                      nbinsx=40, nbinsy=20),
        row=1, col=1
    )
    
    fig_comparison.add_trace(
        go.Histogram2d(x=data_b['x_m'], y=data_b['y_m'], 
                      colorscale='Viridis', showscale=True,
                      nbinsx=40, nbinsy=20),
        row=1, col=2
    )
    
    fig_comparison.update_xaxes(range=[0, 28], row=1, col=1)
    fig_comparison.update_xaxes(range=[0, 28], row=1, col=2)
    fig_comparison.update_yaxes(range=[0, 15], row=1, col=1)
    fig_comparison.update_yaxes(range=[0, 15], row=1, col=2)
    
    fig_comparison.update_layout(height=400, showlegend=False)
    
    st.plotly_chart(fig_comparison, use_container_width=True)

# Temporal Animation
if show_animation:
    st.subheader('⏯️ Animazione Temporale - Evoluzione Heatmap')
    
    anim_player = st.selectbox('Giocatore per animazione', all_players, key='anim_player')
    time_window = st.slider('Finestra temporale (secondi)', 30, 300, 60, 30)
    
    anim_data = uwb_filtered[uwb_filtered['player_id'] == anim_player].copy()
    
    if not anim_data.empty:
        # Create time bins
        min_time = anim_data['timestamp_s'].min()
        max_time = anim_data['timestamp_s'].max()
        time_bins = np.arange(min_time, max_time, time_window)
        
        anim_data['time_bin'] = pd.cut(anim_data['timestamp_s'], bins=time_bins, 
                                        labels=[f'{int(t)}-{int(t+time_window)}s' for t in time_bins[:-1]])
        
        fig_anim = px.density_heatmap(
            anim_data, x='x_m', y='y_m',
            animation_frame='time_bin',
            range_x=[0, 28], range_y=[0, 15],
            nbinsx=40, nbinsy=20,
            color_continuous_scale='Plasma',
            title=f'Evoluzione Posizionale - {anim_player}'
        )
        
        fig_anim.update_layout(height=500)
        st.plotly_chart(fig_anim, use_container_width=True)
    else:
        st.info('Nessun dato disponibile per questo giocatore')

# Original visualizations (kept from previous version)
c1, c2 = st.columns([1,1])

with c1:
    st.subheader('🗺️ Traiettorie su Campo')
    
    with st.expander('⚙️ Opzioni Traiettorie'):
        show_all = st.checkbox('Mostra tutti i giocatori', value=True, key='traj_all')
        if not show_all:
            traj_player = st.selectbox('Giocatore singolo', all_players, key='traj_player')
        marker_size = st.slider('Dimensione marker', 2, 10, 4, key='traj_size')
        marker_opacity = st.slider('Opacità marker', 0.1, 1.0, 0.5, 0.1, key='traj_opacity')
    
    fig = go.Figure()
    
    plot_data = uwb_filtered if show_all else uwb_filtered[uwb_filtered['player_id'] == traj_player]
    if len(plot_data) > 5000:
        plot_data = plot_data.iloc[::max(1, len(plot_data)//5000)]
    
    for player in plot_data['player_id'].unique():
        player_data = plot_data[plot_data['player_id'] == player]
        fig.add_trace(go.Scatter(
            x=player_data['x_m'],
            y=player_data['y_m'],
            mode='markers',
            name=player,
            opacity=marker_opacity,
            marker=dict(size=marker_size)
        ))
    
    fig.update_layout(
        shapes=draw_basketball_court(),
        xaxis=dict(range=[0, 28], constrain='domain', showgrid=False, zeroline=False),
        yaxis=dict(range=[0, 15], scaleanchor='x', scaleratio=1, showgrid=False, zeroline=False),
        plot_bgcolor='rgba(34,139,34,0.2)',
        title='Posizioni UWB su Campo Basket',
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader('🔥 Heatmap Densità su Campo')
    
    with st.expander('⚙️ Opzioni Heatmap'):
        heatmap_player_all = st.checkbox('Tutti i giocatori', value=True, key='heat_all')
        if not heatmap_player_all:
            heatmap_player = st.selectbox('Giocatore singolo', all_players, key='heat_player')
        
        colorscale_options = {
            'Hot (Rosso-Giallo)': 'Hot',
            'Viridis (Blu-Verde-Giallo)': 'Viridis',
            'Plasma (Viola-Rosa-Giallo)': 'Plasma',
            'Inferno (Nero-Rosso-Giallo)': 'Inferno',
            'Jet (Blu-Verde-Rosso)': 'Jet',
            'Portland (Blu-Bianco-Rosso)': 'Portland',
            'Blues (Bianco-Blu)': 'Blues',
            'Reds (Bianco-Rosso)': 'Reds',
            'YlOrRd (Giallo-Arancio-Rosso)': 'YlOrRd',
            'RdYlGn (Rosso-Giallo-Verde)': 'RdYlGn'
        }
        
        colorscale_choice = st.selectbox(
            'Schema colori heatmap',
            options=list(colorscale_options.keys()),
            index=2,
            key='heat_color'
        )
        
        nbins_x = st.slider('Risoluzione orizzontale', 20, 100, 60, 5, key='heat_binsx')
        nbins_y = st.slider('Risoluzione verticale', 10, 60, 32, 2, key='heat_binsy')
        
        reverse_color = st.checkbox('Inverti colori', value=False, key='heat_reverse')
    
    fig2 = go.Figure()
    
    heatmap_data = uwb_filtered if heatmap_player_all else uwb_filtered[uwb_filtered['player_id'] == heatmap_player]
    
    colorscale = colorscale_options[colorscale_choice]
    if reverse_color:
        colorscale = colorscale + '_r'
    
    fig2.add_trace(go.Histogram2d(
        x=heatmap_data['x_m'],
        y=heatmap_data['y_m'],
        colorscale=colorscale,
        nbinsx=nbins_x,
        nbinsy=nbins_y,
        colorbar=dict(title="Densità")
    ))
    
    fig2.update_layout(
        shapes=draw_basketball_court(),
        xaxis=dict(range=[0, 28], constrain='domain', showgrid=False, zeroline=False, title=''),
        yaxis=dict(range=[0, 15], scaleanchor='x', scaleratio=1, showgrid=False, zeroline=False, title=''),
        plot_bgcolor='rgba(34,139,34,0.2)',
        title=f"Heatmap Densità - {'Tutti' if heatmap_player_all else heatmap_player}",
        height=500
    )
    
    st.plotly_chart(fig2, use_container_width=True)

st.subheader('📈 Velocità nel tempo')

with st.expander('⚙️ Opzioni Grafico Velocità'):
    speed_players = st.multiselect(
        'Giocatori da mostrare',
        options=all_players,
        default=all_players[:2] if len(all_players) >= 2 else all_players,
        key='speed_players'
    )
    show_avg = st.checkbox('Mostra media velocità', value=False, key='speed_avg')
    show_max_line = st.checkbox('Mostra linea velocità massima', value=False, key='speed_max')

plot_df = uwb_filtered[uwb_filtered['player_id'].isin(speed_players)].copy() if speed_players else uwb_filtered.copy()

fig3 = px.line(plot_df, x='timestamp_s', y='speed_kmh_calc', color='player_id', 
               title=f'Speed (km/h) - {quarter}',
               labels={'timestamp_s': 'Tempo (secondi)', 'speed_kmh_calc': 'Velocità (km/h)'})

if show_avg and not plot_df.empty:
    avg_speed = plot_df['speed_kmh_calc'].mean()
    fig3.add_hline(y=avg_speed, line_dash="dash", line_color="gray", 
                   annotation_text=f"Media: {avg_speed:.1f} km/h")

if show_max_line and not plot_df.empty:
    max_speed = plot_df['speed_kmh_calc'].max()
    fig3.add_hline(y=max_speed, line_dash="dot", line_color="red",
                   annotation_text=f"Max: {max_speed:.1f} km/h")

st.plotly_chart(fig3, use_container_width=True)

st.subheader('💾 Export Grafici come PNG')

col_exp1, col_exp2, col_exp3 = st.columns(3)

with col_exp1:
    if st.button('📥 Download Traiettorie PNG', key='export_traj'):
        st.info('Usa il pulsante 📷 nella toolbar del grafico sopra per salvare come PNG')

with col_exp2:
    if st.button('📥 Download Heatmap PNG', key='export_heat'):
        st.info('Usa il pulsante 📷 nella toolbar del grafico sopra per salvare come PNG')

with col_exp3:
    if st.button('📥 Download Velocità PNG', key='export_speed'):
        st.info('Usa il pulsante 📷 nella toolbar del grafico sopra per salvare come PNG')

st.caption('💡 Tip: Ogni grafico Plotly ha una toolbar interattiva (visibile al passaggio del mouse) con opzioni di export, zoom, pan e reset.')

st.subheader('📉 IMU (con rumore/bias + dropout)')
if imu is None:
    st.info('Nessun file IMU caricato (ok per test UWB-only).')
else:
    if 'timestamp_s' not in imu.columns or 'accel_z_ms2' not in imu.columns:
        st.warning(f'IMU CSV: colonne richieste mancanti. Colonne disponibili: {list(imu.columns)}')
    else:
        if quarter != 'Intera Partita':
            imu = imu[(imu['timestamp_s'] >= t_min) & (imu['timestamp_s'] < t_max)].copy()
        
        jumps = int((imu.get('jump_detected', pd.Series([0]*len(imu)))==1).sum()) if 'jump_detected' in imu.columns else 0
        st.write(f'🏀 Salti rilevati in {quarter}:', jumps)
        
        imu_players = sorted(imu['player_id'].unique())
        psel = st.selectbox('Giocatore IMU', imu_players, key='imu_player')
        
        imu_p = imu[imu['player_id']==psel].sort_values('timestamp_s')
        fig4 = px.line(imu_p, x='timestamp_s', y='accel_z_ms2', 
                      title=f'Accel Z (m/s²) - {psel} - {quarter}',
                      labels={'timestamp_s': 'Tempo (secondi)', 'accel_z_ms2': 'Accelerazione Z (m/s²)'})
        
        if 'jump_detected' in imu_p.columns:
            jump_points = imu_p[imu_p['jump_detected'] == 1]
            if not jump_points.empty:
                fig4.add_scatter(x=jump_points['timestamp_s'], y=jump_points['accel_z_ms2'],
                               mode='markers', marker=dict(color='red', size=10, symbol='star'),
                               name='Salti rilevati')
        
        st.plotly_chart(fig4, use_container_width=True)

# Footer with setup instructions
with st.expander('⚙️ Setup Instructions - AI Features'):
    st.markdown("""
    ### 🤖 Enable AI Insights with Groq API (FREE)
    
    1. Get FREE API key at: [console.groq.com](https://console.groq.com)
    2. Create `.streamlit/secrets.toml` file:
       GROQ_API_KEY = "your_api_key_here"
    3. Restart Streamlit app
    4. AI insights will automatically activate!
    
    **Benefits:**
    - ✅ 14,400 free requests/day
    - ✅ 10x faster than GPT-4
    - ✅ Llama 3.1 70B model
    - ✅ Natural language insights
    
    Without API key, app uses rule-based insights (still useful!).
    """)