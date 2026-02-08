# =================================================================
# COACHTRACK ELITE AI v3.0 - FINAL COMPLETE EDITION
# All Features: AI + ML + CV + Physical + Nutrition + Analytics
# =================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
from pathlib import Path

# =================================================================
# IMPORT MODULES
# =================================================================

# AI Functions
try:
    from ai_functions import (calculate_distance, predict_injury_risk, 
        recommend_offensive_plays, analyze_movement_patterns, 
        generate_ai_training_plan, simulate_shot_quality)
    AI_AVAILABLE = True
except:
    AI_AVAILABLE = False
    def calculate_distance(df):
        if len(df) < 2: return 0.0
        dx, dy = np.diff(df['x'].values), np.diff(df['y'].values)
        return float(np.sum(np.sqrt(dx**2 + dy**2)))

    def predict_injury_risk(player_data, player_id):
        distance = calculate_distance(player_data)
        risk_score = 25 if distance < 200 else 40 if distance < 500 else 60
        return {
            'player_id': player_id, 'risk_level': 'MEDIO', 'risk_score': risk_score,
            'acwr': 1.2, 'asymmetry': 10.0, 'fatigue': 8.0,
            'risk_factors': ['Distanza elevata', 'Asimmetria rilevata'],
            'recommendations': ['Ridurre carico', 'Monitorare recupero']
        }

# ML Models
try:
    from ml_models import MLInjuryPredictor, PerformancePredictor
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False
    class MLInjuryPredictor:
        def extract_features(self, player_data, physical_data={}):
            distance = calculate_distance(player_data) if len(player_data) > 1 else 0
            return {'total_distance': distance, 'avg_speed': distance/len(player_data) if len(player_data)>0 else 0}
        def predict(self, features):
            risk_prob = min(35 + features.get('total_distance',0)/100, 85)
            return {
                'risk_level': 'BASSO' if risk_prob<30 else 'MEDIO' if risk_prob<60 else 'ALTO',
                'risk_probability': round(risk_prob,1), 'confidence': 'Media',
                'top_risk_factors': [('Distanza',0.35)], 'recommendations': ['Monitorare carico']
            }

    class PerformancePredictor:
        def __init__(self): self.is_trained = False
        def extract_features(self, stats, opponent, injury=None):
            return {'avg_points_last5':15.0,'rest_days':1,'opponent_def_rating':110,'home_away':1}
        def predict_next_game(self, features):
            return {'points':17.5,'assists':5.0,'rebounds':6.0,'efficiency':48.0,'confidence':'MEDIA'}

# Physical
try:
    from physical_nutrition import generate_enhanced_nutrition, create_body_composition_viz
    PHYSICAL_AVAILABLE = True
except:
    PHYSICAL_AVAILABLE = False
    def generate_enhanced_nutrition(player_id, phys, activity, goal):
        w, bmr = phys.get('weight_kg',80), phys.get('bmr',2000)
        mult = {'Low':1.2,'Moderate':1.55,'High':1.75,'Very High':1.9}.get(activity.split('(')[0].strip(),1.55)
        cal = int(bmr*mult)
        return {
            'player_id':player_id,'target_calories':cal,'protein_g':int(w*2.2),
            'carbs_g':int(cal*0.5/4),'fats_g':int(cal*0.25/9),
            'recommendations':['Carbs pre-workout','Proteine post-workout','3-4L acqua'],
            'supplements':['Whey','Creatina','Omega-3','Vit D'],
            'meals':[
                {'name':'Colazione','timing':'7:00','calories':int(cal*0.25),'protein':int(w*0.44),
                 'carbs':int(cal*0.15/4),'fats':int(cal*0.06/9),'examples':'Avena, uova, frutta'},
                {'name':'Pranzo','timing':'14:00','calories':int(cal*0.35),'protein':int(w*0.55),
                 'carbs':int(cal*0.175/4),'fats':int(cal*0.09/9),'examples':'Pasta, carne, verdure'}
            ]
        }
    def create_body_composition_viz(phys):
        fig = go.Figure()
        fig.add_trace(go.Pie(labels=['Muscoli','Grasso','Acqua','Altro'],
                            values=[phys.get('muscle_pct',45),phys.get('body_fat_pct',12),15,28],hole=0.4))
        fig.update_layout(title="Body Composition",height=400)
        return fig

# CV
CV_AVAILABLE = False
try:
    from cv_processor import CoachTrackVisionProcessor
    CV_AVAILABLE = True
except: pass

# =================================================================
# CV TAB
# =================================================================
def add_computer_vision_tab():
    st.header("🎥 Computer Vision System")
    if not CV_AVAILABLE:
        st.error("❌ Computer Vision non disponibile")
        cv_files = ['cv_camera.py','cv_processor.py','cv_tracking.py']
        missing_files = [f for f in cv_files if not Path(f).exists()]
        if missing_files: st.error(f"File mancanti: {','.join(missing_files)}")
        else: st.success("✅ File CV presenti")
        missing_pkgs = []
        try: import cv2
        except: missing_pkgs.append('opencv-python')
        try: from ultralytics import YOLO
        except: missing_pkgs.append('ultralytics')
        if missing_pkgs:
            st.error(f"Pacchetti mancanti: {','.join(missing_pkgs)}")
            st.info("Per Streamlit Cloud: Aggiungi a requirements.txt")
        return
    st.success("✅ CV disponibile")

# =================================================================
# MAIN APP
# =================================================================
st.set_page_config(page_title="CoachTrack Elite",page_icon="🏀",layout="wide")

if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if not st.session_state.logged_in:
    st.title("🏀 CoachTrack Elite AI")
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        u = st.text_input("Username",value="admin")
        p = st.text_input("Password",type="password",value="admin")
        if st.button("Login",type="primary",use_container_width=True):
            if u=="admin" and p=="admin":
                st.session_state.logged_in=True
                st.rerun()
        st.info("admin / admin")
    st.stop()

if 'tracking_data' not in st.session_state: st.session_state.tracking_data = {}
if 'physical_profiles' not in st.session_state: st.session_state.physical_profiles = {}
if 'ml_injury_model' not in st.session_state: st.session_state.ml_injury_model = MLInjuryPredictor()
if 'performance_model' not in st.session_state: st.session_state.performance_model = PerformancePredictor()

with st.sidebar:
    st.title("🏀 CoachTrack")
    st.markdown("---")
    col1,col2 = st.columns(2)
    with col1:
        st.success("✅" if AI_AVAILABLE else "❌"); st.caption("AI")
        st.success("✅" if ML_AVAILABLE else "❌"); st.caption("ML")
    with col2:
        st.success("✅" if CV_AVAILABLE else "❌"); st.caption("CV")
        st.success("✅" if PHYSICAL_AVAILABLE else "❌"); st.caption("PH")
    st.markdown("---")
    st.metric("Players",len(st.session_state.tracking_data))
    st.metric("Physical",len(st.session_state.physical_profiles))
    st.markdown("---")
    if st.button("Logout",use_container_width=True):
        st.session_state.logged_in=False
        st.rerun()

st.title("🏀 CoachTrack Elite AI v3.0 FINAL")
st.markdown("**Complete:** AI + ML + CV + Physical + Nutrition + Analytics")

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["⚙️ Config","🤖 AI","🎥 CV","🧠 ML","💪 Physical","📊 Analytics"])

# TAB 1
with tab1:
    st.header("⚙️ Configurazione")
    uploaded = st.file_uploader("CSV Tracking",type=['csv'])
    if uploaded:
        try:
            df = pd.read_csv(uploaded,sep=';')
            if all(c in df.columns for c in ['player_id','timestamp','x','y']):
                for pid in df['player_id'].unique():
                    st.session_state.tracking_data[pid] = df[df['player_id']==pid].copy()
                st.success(f"✅ {len(df['player_id'].unique())} giocatori")
        except Exception as e: st.error(str(e))

# TAB 2  
with tab2:
    st.header("🤖 AI Features")
    if st.session_state.tracking_data:
        pid = st.selectbox("Player",list(st.session_state.tracking_data.keys()))
        if st.button("🏥 Injury Risk",type="primary"):
            r = predict_injury_risk(st.session_state.tracking_data[pid],pid)
            col1,col2 = st.columns(2)
            with col1: st.metric("Risk",r['risk_level'])
            with col2: st.metric("Score",r['risk_score'])

# TAB 3
with tab3:
    add_computer_vision_tab()

# TAB 4
with tab4:
    st.header("🧠 ML Advanced")
    if st.session_state.tracking_data:
        pid = st.selectbox("Player ML",list(st.session_state.tracking_data.keys()),key='ml')
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("### 🏥 ML Injury")
            if st.button("Run ML"):
                m = st.session_state.ml_injury_model
                f = m.extract_features(st.session_state.tracking_data[pid],
                                      st.session_state.physical_profiles.get(pid,{}))
                r = m.predict(f)
                st.metric("Risk",r['risk_level'])
                st.metric("Prob",f"{r['risk_probability']}%")
        with col2:
            st.markdown("### 📈 Performance")
            with st.form("perf"):
                rest = st.number_input("Rest Days",0,7,1)
                loc = st.selectbox("Location",['home','away'])
                if st.form_submit_button("Predict"):
                    pm = st.session_state.performance_model
                    opp = {'rest_days':rest,'def_rating':110,'location':loc}
                    stats = pd.DataFrame({'points':[15,18,12],'assists':[5,6,4],'rebounds':[6,5,7],'minutes':[30,32,28]})
                    feat = pm.extract_features(stats,opp)
                    pred = pm.predict_next_game(feat)
                    st.metric("Points",f"{pred['points']} pts")
                    st.metric("Efficiency",pred['efficiency'])

# TAB 5 - PHYSICAL & NUTRITION COMPLETO
with tab5:
    st.header("💪 Physical & Nutrition")

    t1,t2,t3 = st.tabs(["📋 Physical","🍎 Nutrition","📊 Body Comp"])

    with t1:
        st.subheader("Physical Data")
        col1,col2 = st.columns([2,1])
        with col1:
            existing = ["Nuovo..."] + list(st.session_state.physical_profiles.keys())
            pname = st.selectbox("Giocatore",existing)
            if pname=="Nuovo...": pname = st.text_input("Nome",key='new')
        with col2:
            ddate = st.date_input("Data",datetime.now())

        with st.form("phys"):
            st.markdown("### Input Dati Fisici")
            c1,c2,c3 = st.columns(3)
            with c1:
                h = st.number_input("Altezza (cm)",150.0,230.0,195.0,0.5)
                w = st.number_input("Peso (kg)",50.0,150.0,80.0,0.1)
                age = st.number_input("Età",15,45,25)
            with c2:
                bf = st.number_input("Grasso (%)",3.0,40.0,12.0,0.1)
                water = st.number_input("Acqua (%)",45.0,75.0,60.0,0.1)
                muscle = st.number_input("Muscoli (%)",25.0,60.0,45.0,0.1)
            with c3:
                bone = st.number_input("Ossa (kg)",2.0,5.0,3.2,0.1)
                hr = st.number_input("HR Riposo",40,100,55)
                vo2 = st.number_input("VO2 Max",30.0,80.0,52.0,0.5)

            if st.form_submit_button("💾 Salva",type="primary",use_container_width=True):
                if pname and pname!="Nuovo...":
                    bmi = w/((h/100)**2)
                    fm = w*(bf/100)
                    lm = w-fm
                    bmr = int(10*w+6.25*h-5*age+5)
                    st.session_state.physical_profiles[pname] = {
                        'date':ddate.strftime('%Y-%m-%d'),'height_cm':h,'weight_kg':w,'age':age,
                        'bmi':round(bmi,1),'body_fat_pct':bf,'lean_mass_kg':round(lm,1),
                        'fat_mass_kg':round(fm,1),'body_water_pct':water,'muscle_pct':muscle,
                        'bone_mass_kg':bone,'resting_hr':hr,'vo2_max':vo2,
                        'bmr':bmr,'amr':int(bmr*1.55),'source':'Manual'
                    }
                    st.success(f"✅ Salvato {pname}")
                    st.balloons()

        st.markdown("### 📊 Dati Salvati")
        if st.session_state.physical_profiles:
            for pid,data in st.session_state.physical_profiles.items():
                with st.expander(f"👤 {pid}"):
                    c1,c2,c3 = st.columns(3)
                    with c1:
                        st.metric("Peso",f"{data.get('weight_kg')}kg")
                        st.metric("BMI",data.get('bmi'))
                    with c2:
                        st.metric("Fat",f"{data.get('body_fat_pct')}%")
                        st.metric("Lean",f"{data.get('lean_mass_kg')}kg")
                    with c3:
                        st.metric("BMR",f"{data.get('bmr')}kcal")
                        if st.button("🗑️",key=f"del_{pid}"):
                            del st.session_state.physical_profiles[pid]
                            st.rerun()

    with t2:
        st.subheader("🍎 Nutrition Plans")
        if st.session_state.physical_profiles:
            pn = st.selectbox("Player",list(st.session_state.physical_profiles.keys()))
            c1,c2 = st.columns(2)
            with c1:
                act = st.selectbox("Activity",["Low (Recovery)","Moderate (Training)","High (Intense)","Very High (Tournament)"])
            with c2:
                goal = st.selectbox("Goal",["Maintenance","Muscle Gain","Fat Loss","Performance"])

            if st.button("🍎 Generate",type="primary",use_container_width=True):
                plan = generate_enhanced_nutrition(pn,st.session_state.physical_profiles[pn],act,goal)
                st.markdown("### 📊 Piano Nutrizionale")
                c1,c2,c3,c4 = st.columns(4)
                with c1: st.metric("🔥 Cal",f"{plan['target_calories']}")
                with c2: st.metric("🥩 Prot",f"{plan['protein_g']}g")
                with c3: st.metric("🍚 Carbs",f"{plan['carbs_g']}g")
                with c4: st.metric("🥑 Fats",f"{plan['fats_g']}g")

                st.markdown("#### 💡 Raccomandazioni")
                for rec in plan['recommendations']: st.info(f"• {rec}")
                st.markdown("#### 💊 Integratori")
                for supp in plan['supplements']: st.success(f"✅ {supp}")
                st.markdown("#### 🍽️ Pasti")
                for meal in plan['meals']:
                    with st.expander(f"{meal['name']} - {meal['timing']}"):
                        st.write(f"**Cal:** {meal['calories']} | P{meal['protein']}g C{meal['carbs']}g F{meal['fats']}g")
                        st.write(f"**Esempi:** {meal['examples']}")

    with t3:
        st.subheader("📊 Body Composition")
        if st.session_state.physical_profiles:
            pv = st.selectbox("Player Viz",list(st.session_state.physical_profiles.keys()),key='viz')
            c1,c2 = st.columns([2,1])
            with c1:
                fig = create_body_composition_viz(st.session_state.physical_profiles[pv])
                st.plotly_chart(fig,use_container_width=True)
            with c2:
                d = st.session_state.physical_profiles[pv]
                st.metric("Lean",f"{d.get('lean_mass_kg')}kg")
                st.metric("Fat",f"{d.get('fat_mass_kg')}kg")
                st.metric("VO2",d.get('vo2_max'))

# TAB 6 - ANALYTICS COMPLETO  
with tab6:
    st.header("📊 Analytics Dashboard")

    if not st.session_state.tracking_data:
        st.info("📥 Carica dati tracking")
    else:
        st.markdown("### 🎯 Statistiche Generali")
        total = sum(calculate_distance(df) for df in st.session_state.tracking_data.values())
        avg = total/len(st.session_state.tracking_data)

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("👥 Players",len(st.session_state.tracking_data))
        with c2: st.metric("📏 Total Dist",f"{total:.0f}m")
        with c3: st.metric("📊 Avg Dist",f"{avg:.0f}m")
        with c4: st.metric("⚖️ Avg Load",f"{total/len(st.session_state.tracking_data)/10:.1f}")

        st.markdown("---")
        st.markdown("### 👥 Confronto Giocatori")

        stats = []
        for pid,df in st.session_state.tracking_data.items():
            dist = calculate_distance(df)
            dur = df['timestamp'].max()-df['timestamp'].min() if len(df)>1 else 0
            stats.append({
                'Player':str(pid),'Distance (m)':round(dist,1),'Duration (s)':round(dur,1),
                'Avg Speed (m/s)':round(dist/dur if dur>0 else 0,2),'Points':len(df)
            })

        sdf = pd.DataFrame(stats).sort_values('Distance (m)',ascending=False)

        # Chart distanza
        fig = px.bar(sdf,x='Player',y='Distance (m)',title="📏 Confronto Distanza",
                    color='Distance (m)',color_continuous_scale='Blues',text='Distance (m)')
        fig.update_traces(texttemplate='%{text:.0f}m',textposition='outside')
        fig.update_layout(showlegend=False,height=400)
        st.plotly_chart(fig,use_container_width=True)

        st.markdown("---")

        # Chart velocità
        st.markdown("### ⚡ Velocità Media")
        fig2 = px.bar(sdf,x='Player',y='Avg Speed (m/s)',title="⚡ Velocità",
                     color='Avg Speed (m/s)',color_continuous_scale='Reds',text='Avg Speed (m/s)')
        fig2.update_traces(texttemplate='%{text:.2f}m/s',textposition='outside')
        fig2.update_layout(showlegend=False,height=400)
        st.plotly_chart(fig2,use_container_width=True)

        st.markdown("---")
        st.markdown("### 📋 Tabella Dettagliata")
        st.dataframe(sdf.style.background_gradient(subset=['Distance (m)'],cmap='Blues')
                    .background_gradient(subset=['Avg Speed (m/s)'],cmap='Reds'),
                    use_container_width=True,height=300)

        st.markdown("---")
        st.markdown("### 🗺️ Team Heatmap")

        pts = []
        for df in st.session_state.tracking_data.values():
            pts.extend([(r['x'],r['y']) for _,r in df.iterrows()])

        if pts:
            pdf = pd.DataFrame(pts,columns=['x','y'])
            fig3 = go.Figure(data=go.Histogram2d(x=pdf['x'],y=pdf['y'],
                            colorscale='Hot',nbinsx=50,nbinsy=30))
            fig3.update_layout(title="🗺️ Heatmap Squadra",
                             xaxis_title="X (m)",yaxis_title="Y (m)",height=500)
            st.plotly_chart(fig3,use_container_width=True)
            st.info(f"📊 Punti tracciati: {len(pdf):,}")

        st.markdown("---")
        st.markdown("### 💾 Export")
        if st.button("📥 Download CSV"):
            csv = sdf.to_csv(index=False)
            st.download_button("⬇️ Download",csv,f"stats_{datetime.now().strftime('%Y%m%d')}.csv","text/csv")

st.caption("🏀 CoachTrack Elite AI v3.0 FINAL - Complete Edition")
