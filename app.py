# =================================================================
# COACHTRACK ELITE AI v3.2 - COMPLETE EDITION
# Analytics + ML + CV + Biometrics
# =================================================================

import sys
import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

print("="*70)
print("🚀 COACHTRACK ELITE v3.2 STARTING")
print("="*70)

# =================================================================
# CHECK MODULES
# =================================================================
CV_AVAILABLE = False
try:
    import cv2
    CV_AVAILABLE = True
    print(f"✅ OpenCV {cv2.__version__}")
except ImportError:
    print("⚠️ OpenCV not available")

try:
    from cv_ai_advanced import CVAIPipeline
    AI_ADVANCED_AVAILABLE = True
    YOLO_AVAILABLE = True
    print("✅ CV AI Pipeline v5.0 (YOLOv8)")
except ImportError:
    AI_ADVANCED_AVAILABLE = False
    YOLO_AVAILABLE = False
    print("⚠️ AI module not available")

# =================================================================
# HELPER FUNCTIONS
# =================================================================
def calculate_distance(df):
    """Calcola distanza percorsa da dataframe tracking"""
    if len(df) < 2:
        return 0.0
    if 'x' not in df.columns or 'y' not in df.columns:
        return 0.0
    dx, dy = np.diff(df['x'].values), np.diff(df['y'].values)
    return float(np.sum(np.sqrt(dx**2 + dy**2)))

# =================================================================
# COMPUTER VISION MODULE
# =================================================================
def add_computer_vision_tab():
    """Computer Vision with YOLOv8 AI Analysis"""

    import pandas as pd
    import plotly.express as px
    from pathlib import Path
    import json
    import cv2
    import os
    import time
    import numpy as np

    st.header("🎥 Computer Vision")

    if not CV_AVAILABLE:
        st.error("❌ OpenCV non disponibile")
        return

    st.success("✅ Computer Vision Online")

    cv_tab1, cv_tab2, cv_tab3, cv_tab4 = st.tabs([
        "🎬 Video Info", "🎯 Calibration", "📊 Dashboard", "🧠 AI Analysis"
    ])

    with cv_tab1:
        st.subheader("🎬 Video Info")
        st.info("📹 Upload per info - Usa 'AI Analysis' per processing")
        uv = st.file_uploader("Carica Video", type=['mp4','avi','mov','mkv'], key="vid_info")
        if uv:
            vp = f"temp_{uv.name}"
            with open(vp,'wb') as f:
                f.write(uv.read())
            st.success(f"✅ {uv.name}")
            try:
                cap = cv2.VideoCapture(vp)
                fps,fc = int(cap.get(cv2.CAP_PROP_FPS)),int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                dur,w,h = fc/fps if fps>0 else 0,int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                c1,c2,c3,c4=st.columns(4)
                c1.metric("⏱️",f"{dur:.1f}s")
                c2.metric("🎞️",fps)
                c3.metric("📸",f"{fc:,}")
                c4.metric("📐",f"{w}x{h}")
                st.success("✅ Vai 'AI Analysis'!")
            except Exception as e:
                st.error(f"❌ {e}")
            finally:
                if os.path.exists(vp):
                    try: os.remove(vp)
                    except: pass

    with cv_tab2:
        st.subheader("🎯 Court Calibration")
        st.info("📐 Feature in sviluppo")

    with cv_tab3:
        st.subheader("📊 Analysis Dashboard")
        st.info("📥 Upload JSON da AI Analysis")
        uj = st.file_uploader("📥 Carica JSON", type=['json'], key="json_up")
        if uj:
            try:
                data = json.load(uj)
                st.success(f"✅ {uj.name}")
                if 'statistics' in data:
                    s=data['statistics']
                    c1,c2,c3=st.columns(3)
                    c1.metric("📸",s.get('total_poses_detected',0))
                    c2.metric("🎯",s.get('total_actions',0))
                    c3.metric("🏀",s.get('total_shots',0))
                st.markdown("---")
                if 'actions' in data and len(data['actions'])>0:
                    st.markdown("### 🎯 Actions")
                    adf=pd.DataFrame(data['actions'])
                    st.dataframe(adf,use_container_width=True)
                    if 'action' in adf.columns:
                        ac=adf['action'].value_counts()
                        fig=px.bar(x=ac.index,y=ac.values,labels={'x':'Azione','y':'Conteggio'})
                        st.plotly_chart(fig,use_container_width=True)
                if 'shots' in data and len(data['shots'])>0:
                    st.markdown("### 🏀 Shots")
                    sdf=pd.DataFrame(data['shots'])
                    st.dataframe(sdf,use_container_width=True)
                    if 'form_score' in sdf.columns:
                        st.metric("Form",f"{sdf['form_score'].mean():.1f}/100")
                with st.expander("📄 JSON"):
                    st.json(data)
            except Exception as e:
                st.error(f"❌ {e}")
        else:
            jf=list(Path('.').glob('*.json'))
            if jf:
                st.info(f"📁 {len(jf)} JSON sul server")
                sel=st.selectbox("Seleziona",[f.name for f in jf])
                if st.button("📊 Carica"):
                    with open(sel,'r') as f:
                        st.json(json.load(f))
            else:
                st.warning("⚠️ Usa AI Analysis")

    with cv_tab4:
        st.subheader("🧠 AI Advanced Analysis")
        st.markdown("---")
        if not AI_ADVANCED_AVAILABLE:
            st.error("❌ AI module non disponibile")
            return
        st.success("✅ YOLOv8 Pose Analysis")
        st.info("🤖 Pose + Actions + Shot Analysis")
        st.markdown("### 📹 Upload")
        uva = st.file_uploader("Video",type=['mp4','avi','mov','mkv'],key="ai_video")
        if uva:
            vp=f"temp_ai_{uva.name}"
            with st.spinner("📤..."):
                with open(vp,'wb') as f: f.write(uva.read())
            st.success(f"✅ {uva.name}")
            st.markdown("### ⚙️ Opzioni")
            c1,c2=st.columns(2)
            with c1:
                aa=st.checkbox("🎯 Actions",value=True)
                ash=st.checkbox("🏀 Shots",value=True)
            with c2:
                ap=st.checkbox("🤸 Pose",value=True)
                oj=st.text_input("📄 Output","ai_analysis.json")
            st.markdown("---")
            if st.button("🚀 Avvia",type="primary",use_container_width=True):
                pb,st_=st.progress(0),st.empty()
                try:
                    st_.text("🤖 Init...")
                    pb.progress(0.1)
                    st_.text("🎬 Processing...")
                    pb.progress(0.3)
                    pip=CVAIPipeline()
                    if not pip.initialize(): raise Exception("YOLOv8 fail")
                    cap=cv2.VideoCapture(vp)
                    fps,fc=int(cap.get(cv2.CAP_PROP_FPS)),int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    res={'video_info':{'fps':fps,'frame_count':fc,'duration':fc/fps if fps>0 else 0},'actions':[],'shots':[],'pose_data':[],'statistics':{'total_poses_detected':0,'total_actions':0,'total_shots':0}}
                    fi=0
                    while cap.isOpened():
                        ret,frame=cap.read()
                        if not ret: break
                        if fi%5==0:
                            fr=pip.process_frame(frame)
                            if fr:
                                res['statistics']['total_poses_detected']+=1
                                act=fr.get('action','unknown')
                                if act!='unknown':
                                    res['actions'].append({'frame':int(fi),'action':act,'timestamp':float(fi/fps if fps>0 else 0)})
                                    res['statistics']['total_actions']+=1
                                if act=='shooting' and 'shooting_form' in fr:
                                    form=fr['shooting_form']
                                    res['shots'].append({'frame':int(fi),'elbow_angle':float(form['elbow_angle']),'knee_angle':float(form['knee_angle']),'form_score':float(form['form_score']),'timestamp':float(fi/fps if fps>0 else 0)})
                                    res['statistics']['total_shots']+=1
                        fi+=1
                        if fi%100==0: pb.progress(min(0.3+(fi/fc)*0.7,1.0))
                    cap.release()
                    with open(oj,'w') as f: json.dump(res,f,indent=2)
                    pb.progress(1.0)
                    st_.text("✅ Done!")
                    st.balloons()
                    st.markdown("### 📊 Risultati")
                    s=res.get('statistics',{})
                    c1,c2,c3=st.columns(3)
                    c1.metric("📸",s.get('total_poses_detected',0))
                    c2.metric("🎯",s.get('total_actions',0))
                    c3.metric("🏀",s.get('total_shots',0))
                    st.markdown("---")
                    if aa and res.get('actions'):
                        st.markdown("#### 🎯 Actions")
                        if len(res['actions'])>0:
                            adf=pd.DataFrame(res['actions'])
                            st.dataframe(adf,use_container_width=True)
                            if 'action' in adf.columns:
                                ac=adf['action'].value_counts()
                                fig=px.bar(x=ac.index,y=ac.values)
                                st.plotly_chart(fig,use_container_width=True)
                    if ash and res.get('shots'):
                        st.markdown("#### 🏀 Shots")
                        if len(res['shots'])>0:
                            sdf=pd.DataFrame(res['shots'])
                            st.dataframe(sdf,use_container_width=True)
                    st.markdown("---")
                    with open(oj,'r') as f: jd=f.read()
                    st.download_button("⬇️ Download JSON",jd,oj,"application/json",use_container_width=True)
                except Exception as e:
                    pb.empty()
                    st_.empty()
                    st.error(f"❌ {str(e)}")
                    with st.expander("🔍"):
                        import traceback
                        st.code(traceback.format_exc())
                finally:
                    if os.path.exists(vp):
                        try: time.sleep(0.5);os.remove(vp)
                        except: pass

# =================================================================
# BIOMETRIC MODULE
# =================================================================
def render_biometric_module():
    st.header("⚖️ Biometrics")
    if 'biometric_data' not in st.session_state:
        st.session_state.biometric_data=pd.DataFrame(columns=['player_id','player_name','timestamp','weight_kg','body_fat_pct','muscle_mass_kg','water_pct','bone_mass_kg','bmr_kcal','measurement_type','source','notes'])
    tab1,tab2=st.tabs(["📊 Dashboard","➕ Input"])
    with tab1:
        st.subheader("📊 Dashboard")
        if st.session_state.biometric_data.empty:
            st.info("Nessun dato")
        else:
            latest=st.session_state.biometric_data.sort_values('timestamp').groupby('player_id').last()
            st.metric("Giocatori",len(latest))
            st.dataframe(latest[['player_name','weight_kg','body_fat_pct']],use_container_width=True)
    with tab2:
        st.subheader("➕ Input")
        with st.form("bio_form"):
            name=st.text_input("Nome")
            weight=st.number_input("Peso (kg)",40.0,150.0,75.0)
            submitted=st.form_submit_button("💾 Salva")
            if submitted and name:
                import hashlib
                pid=hashlib.md5(name.encode()).hexdigest()[:8]
                new_row=pd.DataFrame([{'player_id':pid,'player_name':name,'timestamp':datetime.now(),'weight_kg':weight,'body_fat_pct':None,'muscle_mass_kg':None,'water_pct':None,'bone_mass_kg':None,'bmr_kcal':None,'measurement_type':'manual','source':'manual','notes':''}])
                st.session_state.biometric_data=pd.concat([st.session_state.biometric_data,new_row],ignore_index=True)
                st.success(f"✅ {name}")
                st.rerun()

# =================================================================
# ANALYTICS MODULE
# =================================================================
def add_analytics_tab():
    """Analytics Dashboard"""
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    st.header("📊 Analytics Dashboard")
    if not st.session_state.tracking_data:
        st.info("📥 Carica CSV tracking")
        st.markdown("### 📁 Upload CSV")
        up=st.file_uploader("CSV (player_id,timestamp,x,y)",type=['csv'])
        if up:
            try:
                df=pd.read_csv(up)
                if all(c in df.columns for c in ['player_id','x','y']):
                    for pid in df['player_id'].unique():
                        st.session_state.tracking_data[str(pid)]=df[df['player_id']==pid].reset_index(drop=True)
                    st.success(f"✅ {len(df['player_id'].unique())} players")
                    st.rerun()
                else:
                    st.error("❌ Serve: player_id, x, y")
            except Exception as e:
                st.error(f"❌ {e}")
        return
    st.markdown("### 📈 Statistiche")
    total=sum(calculate_distance(df) for df in st.session_state.tracking_data.values())
    avg=total/len(st.session_state.tracking_data) if st.session_state.tracking_data else 0
    c1,c2,c3,c4=st.columns(4)
    c1.metric("👥",len(st.session_state.tracking_data))
    c2.metric("📏",f"{total:.0f}m")
    c3.metric("📊",f"{avg:.0f}m")
    c4.metric("⚖️",f"{total/len(st.session_state.tracking_data)/10:.1f}" if st.session_state.tracking_data else "0")
    st.markdown("---")
    st.markdown("### 📊 Confronto")
    stats=[]
    for pid,df in st.session_state.tracking_data.items():
        d=calculate_distance(df)
        stats.append({'Player':str(pid),'Distance (m)':round(d,1),'Points':len(df)})
    if stats:
        sdf=pd.DataFrame(stats).sort_values('Distance (m)',ascending=False)
        fig=px.bar(sdf,x='Player',y='Distance (m)',color='Distance (m)',color_continuous_scale='Blues',text='Distance (m)')
        fig.update_layout(showlegend=False,height=400)
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(sdf,use_container_width=True)
    st.markdown("---")
    st.markdown("### 🔥 Heatmap")
    pts=[]
    for df in st.session_state.tracking_data.values():
        if 'x' in df.columns and 'y' in df.columns:
            pts.extend([(r['x'],r['y']) for _,r in df.iterrows()])
    if pts:
        pdf=pd.DataFrame(pts,columns=['x','y'])
        fig2=go.Figure(data=go.Histogram2d(x=pdf['x'],y=pdf['y'],colorscale='Hot',nbinsx=50,nbinsy=30))
        fig2.update_layout(title='Team Movement',height=500)
        st.plotly_chart(fig2,use_container_width=True)
    else:
        st.info("No position data")

# =================================================================
# ML MODULE
# =================================================================
def add_ml_tab():
    """ML Analytics"""
    import pandas as pd
    st.header("🤖 ML Advanced Analytics")
    if not st.session_state.tracking_data:
        st.warning("⚠️ Carica tracking prima")
        return
    tab1,tab2=st.tabs(["🚑 Injury","📈 Performance"])
    with tab1:
        st.subheader("🚑 Injury Risk")
        st.info("ML basato su distanza/carico")
        pid=st.selectbox("Player",list(st.session_state.tracking_data.keys()),key="ml_inj")
        if st.button("🔍 Run",type="primary"):
            pd_data=st.session_state.tracking_data[pid]
            dist=calculate_distance(pd_data)
            risk=min(35+(dist/100)*85,100)
            if risk<40: lvl,col="BASSO","🟢"
            elif risk<70: lvl,col="MEDIO","🟡"
            else: lvl,col="ALTO","🔴"
            c1,c2,c3=st.columns(3)
            c1.metric(f"{col} Risk",lvl)
            c2.metric("Score",f"{risk:.0f}/100")
            c3.metric("Dist",f"{dist:.0f}m")
            st.markdown("---")
            st.markdown("#### 📋 Fattori")
            if dist>5000: st.warning("⚠️ Dist elevata")
            if len(pd_data)>1000: st.warning("⚠️ Alto carico")
            st.markdown("#### 💡 Raccomandazioni")
            if lvl=="ALTO": st.error("🔴 Ridurre 20-30%")
            elif lvl=="MEDIO": st.warning("🟡 Monitorare")
            else: st.success("🟢 OK")
    with tab2:
        st.subheader("📈 Performance")
        st.info("Predizione prossima partita")
        with st.form("perf"):
            rest=st.number_input("Riposo (gg)",0,7,1)
            loc=st.selectbox("Location",["home","away"])
            opp=st.slider("Rating avversario",80,120,100)
            sub=st.form_submit_button("🔮 Predict",type="primary")
            if sub:
                pts=15+rest*1.5+(3 if loc=="home" else 0)-(opp-100)*0.15
                eff=45+rest*2-(opp-100)*0.2
                st.markdown("### 🎯 Predizioni")
                c1,c2=st.columns(2)
                c1.metric("📊 Punti",f"{pts:.1f}")
                c2.metric("⚡ Efficiency",f"{eff:.1f}%")
                st.markdown("---")
                if rest<2: st.warning("⚠️ Poco riposo")
                if loc=="away": st.info("🏟️ Trasferta")
                if opp>110: st.warning("💪 Avversario forte")

# =================================================================
# MAIN APP
# =================================================================
st.set_page_config(page_title="CoachTrack Elite",page_icon="🏀",layout="wide")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in=False
if 'tracking_data' not in st.session_state:
    st.session_state.tracking_data={}

if not st.session_state.logged_in:
    st.title("🏀 CoachTrack Elite")
    col1,col2,col3=st.columns([1,2,1])
    with col2:
        u=st.text_input("Username",value="admin")
        p=st.text_input("Password",type="password",value="admin")
        if st.button("Login",type="primary",use_container_width=True):
            if u=="admin" and p=="admin":
                st.session_state.logged_in=True
                st.rerun()
            else:
                st.error("❌ Wrong")
    st.stop()

with st.sidebar:
    st.title("🏀 CoachTrack")
    st.markdown("---")
    st.caption("v3.2 Complete")
    if st.button("Logout",use_container_width=True):
        st.session_state.logged_in=False
        st.rerun()

st.title("🏀 CoachTrack Elite AI v3.2")
st.markdown("Complete: AI + ML + CV + Biometrics + Analytics")

tab1,tab2,tab3,tab4,tab5=st.tabs(["🏠 Dashboard","🎥 CV","⚖️ Bio","📊 Analytics","🤖 ML"])

with tab1:
    st.header("📊 Dashboard")
    st.info("Welcome to CoachTrack Elite v3.2 Complete Edition")
    col1,col2,col3,col4=st.columns(4)
    col1.metric("Players",len(st.session_state.tracking_data))
    col2.metric("CV","✅" if CV_AVAILABLE else "❌")
    col3.metric("AI","✅" if AI_ADVANCED_AVAILABLE else "❌")
    col4.metric("Status","🟢 Online")

with tab2:
    add_computer_vision_tab()

with tab3:
    render_biometric_module()

with tab4:
    add_analytics_tab()

with tab5:
    add_ml_tab()
