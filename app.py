import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title='Basketball Tracking MVP (Realistico)', layout='wide')
st.title('🏀 Basketball Tracking MVP - Test Realistico')
st.caption('Dataset include dropout e outlier NLOS per simulare condizioni reali indoor.')

with st.sidebar:
    st.header('📁 Dati')
    use_sample = st.toggle('Usa sample realistici inclusi (consigliato)', value=True)
    uwb_file = None
    imu_file = None
    if not use_sample:
        uwb_file = st.file_uploader('UWB CSV', type=['csv'])
        imu_file = st.file_uploader('IMU CSV', type=['csv'])

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

# UWB checks
required = ['timestamp_s','player_id','x_m','y_m','quality_factor']
missing = [c for c in required if c not in uwb.columns]
if missing:
    st.error(f'UWB CSV: colonne mancanti: {missing}. Colonne disponibili: {list(uwb.columns)}')
    st.stop()

uwb = uwb.sort_values(['player_id','timestamp_s']).copy()
uwb = uwb[uwb['quality_factor'] >= min_q].copy()

# Clip speed to mitigate NLOS spikes
if 'speed_kmh' in uwb.columns:
    uwb['speed_kmh_clip'] = uwb['speed_kmh'].clip(upper=max_speed_clip)
else:
    uwb['speed_kmh_clip'] = np.nan

st.subheader('📊 KPI per giocatore (dopo filtri)')

# Recompute distance and speed from positions (robust to missing samples)
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
    st.subheader('🗺️ Traiettorie (con dropout/outlier)')
    fig = px.scatter(uwb, x='x_m', y='y_m', color='player_id', opacity=0.45,
                     title='Posizioni UWB (realistiche)')
    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader('🔥 Heatmap densità')
    fig2 = px.density_heatmap(uwb, x='x_m', y='y_m', nbinsx=60, nbinsy=32,
                              title='Heatmap densità posizioni (dopo filtri)')
    fig2.update_yaxes(scaleanchor='x', scaleratio=1)
    st.plotly_chart(fig2, use_container_width=True)

st.subheader('📈 Velocità nel tempo (robusta)')
players = sorted(uwb['player_id'].unique())
sel = st.multiselect('Seleziona giocatori', players, default=players[:2])
plot_df = uwb[uwb['player_id'].isin(sel)].copy()
fig3 = px.line(plot_df, x='timestamp_s', y='speed_kmh_calc', color='player_id', title='Speed (km/h) da UWB (clippata)')
st.plotly_chart(fig3, use_container_width=True)

st.subheader('📉 IMU (con rumore/bias + dropout)')
if imu is None:
    st.info('Nessun file IMU caricato (ok per test UWB-only).')
else:
    if 'timestamp_s' not in imu.columns or 'accel_z_ms2' not in imu.columns:
        st.warning(f'IMU CSV: colonne richieste mancanti. Colonne disponibili: {list(imu.columns)}')
    else:
        jumps = int((imu.get('jump_detected', pd.Series([0]*len(imu)))==1).sum()) if 'jump_detected' in imu.columns else 0
        st.write('🏀 Salti rilevati (virtuali):', jumps)
        psel = st.selectbox('Giocatore IMU', sorted(imu['player_id'].unique()))
        imu_p = imu[imu['player_id']==psel].sort_values('timestamp_s')
        fig4 = px.line(imu_p, x='timestamp_s', y='accel_z_ms2', title=f'Accel Z (m/s²) - {psel}')
        st.plotly_chart(fig4, use_container_width=True)