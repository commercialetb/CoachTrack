# =================================================================
# COACHTRACK ELITE AI v3.2 - COMPLETE WITH FULL BIOMETRICS
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

logging.basicConfig(level=logging.INFO)
print("="*70)
print("🚀 COACHTRACK v3.2 - COMPLETE BIOMETRIC")
print("="*70)

# Check modules
CV_AVAILABLE = False
try:
    import cv2
    CV_AVAILABLE = True
    print(f"✅ OpenCV {cv2.__version__}")
except: print("⚠️ OpenCV not available")

try:
    from cv_ai_advanced import CVAIPipeline
    AI_ADVANCED_AVAILABLE = True
    YOLO_AVAILABLE = True
    print("✅ AI Pipeline (YOLOv8)")
except:
    AI_ADVANCED_AVAILABLE = False
    YOLO_AVAILABLE = False
    print("⚠️ AI not available")

def calculate_distance(df):
    if len(df)<2 or 'x' not in df.columns or 'y' not in df.columns:
        return 0.0
    dx,dy=np.diff(df['x'].values),np.diff(df['y'].values)
    return float(np.sum(np.sqrt(dx**2+dy**2)))

# =================================================================
# COMPUTER VISION
# =================================================================
def add_computer_vision_tab():
    import pandas as pd
    import plotly.express as px
    from pathlib import Path
    import json,cv2,os,time,numpy as np

    st.header("🎥 Computer Vision")
    if not CV_AVAILABLE:
        st.error("❌ OpenCV not available")
        return
    st.success("✅ CV Online")

    cv_tab1,cv_tab2,cv_tab3,cv_tab4=st.tabs(["🎬 Video","🎯 Calib","📊 Dashboard","🧠 AI"])

    with cv_tab1:
        st.subheader("🎬 Video Info")
        uv=st.file_uploader("Video",type=['mp4','avi','mov','mkv'],key="vid")
        if uv:
            vp=f"temp_{uv.name}"
            with open(vp,'wb') as f: f.write(uv.read())
            st.success(f"✅ {uv.name}")
            try:
                cap=cv2.VideoCapture(vp)
                fps,fc=int(cap.get(cv2.CAP_PROP_FPS)),int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                dur,w,h=fc/fps if fps>0 else 0,int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                c1,c2,c3,c4=st.columns(4)
                c1.metric("⏱️",f"{dur:.1f}s")
                c2.metric("🎞️",fps)
                c3.metric("📸",f"{fc:,}")
                c4.metric("📐",f"{w}x{h}")
            except Exception as e: st.error(f"❌ {e}")
            finally:
                if os.path.exists(vp):
                    try: os.remove(vp)
                    except: pass

    with cv_tab2:
        st.subheader("🎯 Calibration")
        st.info("In sviluppo")

    with cv_tab3:
        st.subheader("📊 Dashboard")
        uj=st.file_uploader("📥 JSON",type=['json'],key="ju")
        if uj:
            try:
                data=json.load(uj)
                st.success(f"✅ {uj.name}")
                if 'statistics' in data:
                    s=data['statistics']
                    c1,c2,c3=st.columns(3)
                    c1.metric("📸",s.get('total_poses_detected',0))
                    c2.metric("🎯",s.get('total_actions',0))
                    c3.metric("🏀",s.get('total_shots',0))
                st.markdown("---")
                if 'actions' in data and len(data['actions'])>0:
                    adf=pd.DataFrame(data['actions'])
                    st.dataframe(adf,use_container_width=True)
                    if 'action' in adf.columns:
                        ac=adf['action'].value_counts()
                        fig=px.bar(x=ac.index,y=ac.values)
                        st.plotly_chart(fig,use_container_width=True)
                if 'shots' in data and len(data['shots'])>0:
                    sdf=pd.DataFrame(data['shots'])
                    st.dataframe(sdf,use_container_width=True)
                    if 'form_score' in sdf.columns:
                        st.metric("Form",f"{sdf['form_score'].mean():.1f}/100")
                with st.expander("JSON"):
                    st.json(data)
            except Exception as e: st.error(f"❌ {e}")

    with cv_tab4:
        st.subheader("🧠 AI Analysis")
        if not AI_ADVANCED_AVAILABLE:
            st.error("❌ AI not available")
            return
        st.success("✅ YOLOv8")
        uva=st.file_uploader("Video AI",type=['mp4','avi','mov','mkv'],key="ai")
        if uva:
            vp=f"temp_ai_{uva.name}"
            with open(vp,'wb') as f: f.write(uva.read())
            st.success(f"✅ {uva.name}")
            oj=st.text_input("Output","ai_analysis.json")
            if st.button("🚀 Run",type="primary",use_container_width=True):
                pb,st_=st.progress(0),st.empty()
                try:
                    st_.text("🤖 Init...")
                    pb.progress(0.1)
                    pip=CVAIPipeline()
                    if not pip.initialize(): raise Exception("Fail")
                    cap=cv2.VideoCapture(vp)
                    fps,fc=int(cap.get(cv2.CAP_PROP_FPS)),int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    res={'video_info':{'fps':fps,'frame_count':fc,'duration':fc/fps if fps>0 else 0},'actions':[],'shots':[],'statistics':{'total_poses_detected':0,'total_actions':0,'total_shots':0}}
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
                    s=res.get('statistics',{})
                    c1,c2,c3=st.columns(3)
                    c1.metric("📸",s.get('total_poses_detected',0))
                    c2.metric("🎯",s.get('total_actions',0))
                    c3.metric("🏀",s.get('total_shots',0))
                    with open(oj,'r') as f: jd=f.read()
                    st.download_button("⬇️ Download",jd,oj,"application/json",use_container_width=True)
                except Exception as e:
                    pb.empty()
                    st_.empty()
                    st.error(f"❌ {str(e)}")
                finally:
                    if os.path.exists(vp):
                        try: time.sleep(0.5);os.remove(vp)
                        except: pass

# =================================================================
# BIOMETRIC MODULE - COMPLETO
# =================================================================
def render_biometric_module():
    import pandas as pd
    import plotly.graph_objects as go
    from datetime import datetime,timedelta

    st.header("⚖️ Monitoraggio Biometrico")

    if 'biometric_data' not in st.session_state:
        st.session_state.biometric_data=pd.DataFrame(columns=[
            'player_id','player_name','timestamp','weight_kg','body_fat_pct',
            'muscle_mass_kg','water_pct','bone_mass_kg','bmr_kcal',
            'measurement_type','source','notes'
        ])

    tab1,tab2,tab3=st.tabs(["📊 Dashboard","➕ Input","📈 Trend"])

    # TAB 1: DASHBOARD
    with tab1:
        st.subheader("📊 Overview")
        if st.session_state.biometric_data.empty:
            st.info("Nessun dato")
        else:
            latest=st.session_state.biometric_data.sort_values('timestamp').groupby('player_id').last()
            c1,c2,c3,c4=st.columns(4)
            c1.metric("👥",len(latest))
            avg_w=latest['weight_kg'].mean()
            c2.metric("⚖️",f"{avg_w:.1f} kg" if pd.notna(avg_w) else "N/A")
            avg_bf=latest['body_fat_pct'].mean()
            c3.metric("📊",f"{avg_bf:.1f}%" if pd.notna(avg_bf) else "N/A")
            recent=st.session_state.biometric_data[st.session_state.biometric_data['timestamp']>=(datetime.now()-timedelta(days=7))]
            c4.metric("📅 7gg",len(recent))
            st.divider()
            st.subheader("⚠️ Alert")
            alerts=[]
            for pid in latest.index:
                pd_=st.session_state.biometric_data[st.session_state.biometric_data['player_id']==pid]
                if len(pd_)>=2:
                    lw=latest.loc[pid,'weight_kg']
                    avg7=pd_.tail(7)['weight_kg'].mean()
                    wc=lw-avg7
                    if abs(wc)>=2.0:
                        pn=latest.loc[pid,'player_name']
                        alerts.append({'p':pn,'m':f"Peso {wc:+.1f}kg vs 7gg",'s':'high' if abs(wc)>=3 else 'med'})
            if alerts:
                for a in alerts:
                    if a['s']=='high': st.error(f"🔴 {a['p']}: {a['m']}")
                    else: st.warning(f"🟡 {a['p']}: {a['m']}")
            else: st.success("✅ Tutto OK")
            st.divider()
            st.subheader("📋 Ultime")
            disp=latest[['player_name','timestamp','weight_kg','body_fat_pct','muscle_mass_kg','water_pct','source']].copy()
            disp['timestamp']=pd.to_datetime(disp['timestamp']).dt.strftime('%d/%m/%Y %H:%M')
            disp.columns=['Giocatore','Data','Peso(kg)','Grasso(%)','Muscolo(kg)','Acqua(%)','Fonte']
            st.dataframe(disp,use_container_width=True,hide_index=True)

    # TAB 2: INPUT
    with tab2:
        st.subheader("➕ Input Manuale")
        st.info("Compila almeno nome e peso")
        with st.form("bio"):
            c1,c2=st.columns(2)
            with c1:
                name=st.text_input("Nome *")
                weight=st.number_input("Peso(kg) *",40.0,150.0,75.0,0.1)
                bf=st.number_input("Grasso(%)",3.0,50.0,value=None,step=0.1)
                mm=st.number_input("Muscolo(kg)",20.0,80.0,value=None,step=0.1)
            with c2:
                water=st.number_input("Acqua(%)",40.0,75.0,value=None,step=0.1)
                bone=st.number_input("Ossa(kg)",2.0,5.0,value=None,step=0.1)
                bmr=st.number_input("BMR(kcal)",1200,3000,value=None,step=10)
                mtype=st.selectbox("Momento",["Pre-allenamento","Post-allenamento","Mattina","Altro"])
            notes=st.text_area("Note")
            sub=st.form_submit_button("💾 Salva",type="primary",use_container_width=True)
            if sub:
                if not name:
                    st.error("❌ Nome obbligatorio")
                else:
                    import hashlib
                    pid=hashlib.md5(name.encode()).hexdigest()[:8]
                    pd_=st.session_state.biometric_data[st.session_state.biometric_data['player_id']==pid]
                    if len(pd_)>=2:
                        avg7=pd_.tail(7)['weight_kg'].mean()
                        wc=weight-avg7
                        if abs(wc)>=2.0: st.warning(f"⚠️ Peso {wc:+.1f}kg vs 7gg")
                    if water and water<55: st.error(f"🚨 Disidratazione {water:.1f}%")
                    new=pd.DataFrame([{'player_id':pid,'player_name':name,'timestamp':datetime.now(),'weight_kg':weight,'body_fat_pct':bf,'muscle_mass_kg':mm,'water_pct':water,'bone_mass_kg':bone,'bmr_kcal':bmr,'measurement_type':mtype.lower().replace('-','_'),'source':'manual','notes':notes}])
                    st.session_state.biometric_data=pd.concat([st.session_state.biometric_data,new],ignore_index=True)
                    st.success(f"✅ {name}")
                    st.balloons()

    # TAB 3: TREND
    with tab3:
        st.subheader("📈 Trend")
        if st.session_state.biometric_data.empty:
            st.info("Nessun dato")
        else:
            players=st.session_state.biometric_data['player_name'].unique()
            sel=st.selectbox("Giocatore",players)
            if sel:
                pid=st.session_state.biometric_data[st.session_state.biometric_data['player_name']==sel]['player_id'].iloc[0]
                days=st.slider("Giorni",7,180,30)
                cutoff=datetime.now()-timedelta(days=days)
                pdf=st.session_state.biometric_data[(st.session_state.biometric_data['player_id']==pid)&(st.session_state.biometric_data['timestamp']>=cutoff)].sort_values('timestamp')
                if len(pdf)<2:
                    st.warning("Minimo 2 misurazioni")
                else:
                    st.markdown("#### ⚖️ Peso")
                    fig=go.Figure()
                    fig.add_trace(go.Scatter(x=pdf['timestamp'],y=pdf['weight_kg'],mode='lines+markers',name='Peso',line=dict(color='#3498DB',width=2)))
                    fig.update_layout(xaxis_title="Data",yaxis_title="kg",height=300)
                    st.plotly_chart(fig,use_container_width=True)
                    if pdf['body_fat_pct'].notna().any():
                        st.markdown("#### 📊 Composition")
                        fig2=go.Figure()
                        fig2.add_trace(go.Scatter(x=pdf['timestamp'],y=pdf['body_fat_pct'],mode='lines+markers',name='Grasso%',line=dict(color='#E74C3C',width=2)))
                        if pdf['muscle_mass_kg'].notna().any():
                            fig2.add_trace(go.Scatter(x=pdf['timestamp'],y=pdf['muscle_mass_kg'],mode='lines+markers',name='Muscolo kg',yaxis='y2',line=dict(color='#27AE60',width=2)))
                        fig2.update_layout(xaxis_title="Data",yaxis_title="Grasso%",yaxis2=dict(title="Muscolo kg",overlaying='y',side='right'),height=400)
                        st.plotly_chart(fig2,use_container_width=True)
                    st.markdown("#### 📈 Stats")
                    c1,c2,c3=st.columns(3)
                    wc=pdf['weight_kg'].iloc[-1]-pdf['weight_kg'].iloc[0]
                    c1.metric("Variazione",f"{wc:+.1f} kg",delta=f"{wc:+.1f}")
                    c2.metric("Medio",f"{pdf['weight_kg'].mean():.1f} kg")
                    c3.metric("Misurazioni",len(pdf))

# =================================================================
# ANALYTICS
# =================================================================
def add_analytics_tab():
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    st.header("📊 Analytics")
    if not st.session_state.tracking_data:
        st.info("Upload CSV")
        up=st.file_uploader("CSV (player_id,x,y)",type=['csv'])
        if up:
            try:
                df=pd.read_csv(up)
                if all(c in df.columns for c in ['player_id','x','y']):
                    for pid in df['player_id'].unique():
                        st.session_state.tracking_data[str(pid)]=df[df['player_id']==pid].reset_index(drop=True)
                    st.success(f"✅ {len(df['player_id'].unique())} players")
                    st.rerun()
            except Exception as e: st.error(f"❌ {e}")
        return
    total=sum(calculate_distance(df) for df in st.session_state.tracking_data.values())
    avg=total/len(st.session_state.tracking_data) if st.session_state.tracking_data else 0
    c1,c2,c3,c4=st.columns(4)
    c1.metric("👥",len(st.session_state.tracking_data))
    c2.metric("📏",f"{total:.0f}m")
    c3.metric("📊",f"{avg:.0f}m")
    c4.metric("⚖️",f"{total/len(st.session_state.tracking_data)/10:.1f}" if st.session_state.tracking_data else "0")
    st.markdown("---")
    stats=[]
    for pid,df in st.session_state.tracking_data.items():
        d=calculate_distance(df)
        stats.append({'Player':str(pid),'Distance(m)':round(d,1),'Points':len(df)})
    if stats:
        sdf=pd.DataFrame(stats).sort_values('Distance(m)',ascending=False)
        fig=px.bar(sdf,x='Player',y='Distance(m)',color='Distance(m)',color_continuous_scale='Blues',text='Distance(m)')
        fig.update_layout(showlegend=False,height=400)
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(sdf,use_container_width=True)
    st.markdown("---")
    pts=[]
    for df in st.session_state.tracking_data.values():
        if 'x' in df.columns and 'y' in df.columns:
            pts.extend([(r['x'],r['y']) for _,r in df.iterrows()])
    if pts:
        pdf=pd.DataFrame(pts,columns=['x','y'])
        fig2=go.Figure(data=go.Histogram2d(x=pdf['x'],y=pdf['y'],colorscale='Hot',nbinsx=50,nbinsy=30))
        fig2.update_layout(title='Heatmap',height=500)
        st.plotly_chart(fig2,use_container_width=True)

# =================================================================
# ML
# =================================================================
def add_ml_tab():
    st.header("🤖 ML")
    if not st.session_state.tracking_data:
        st.warning("Carica tracking")
        return
    tab1,tab2=st.tabs(["🚑 Injury","📈 Perf"])
    with tab1:
        st.subheader("🚑 Injury")
        pid=st.selectbox("Player",list(st.session_state.tracking_data.keys()),key="ml")
        if st.button("🔍 Run",type="primary"):
            pd_=st.session_state.tracking_data[pid]
            dist=calculate_distance(pd_)
            risk=min(35+(dist/100)*85,100)
            if risk<40: lvl,col="BASSO","🟢"
            elif risk<70: lvl,col="MEDIO","🟡"
            else: lvl,col="ALTO","🔴"
            c1,c2,c3=st.columns(3)
            c1.metric(f"{col}",lvl)
            c2.metric("Score",f"{risk:.0f}/100")
            c3.metric("Dist",f"{dist:.0f}m")
            if dist>5000: st.warning("⚠️ Dist alta")
            if lvl=="ALTO": st.error("🔴 Ridurre 20-30%")
            elif lvl=="MEDIO": st.warning("🟡 Monitorare")
            else: st.success("🟢 OK")
    with tab2:
        st.subheader("📈 Performance")
        with st.form("perf"):
            rest=st.number_input("Riposo(gg)",0,7,1)
            loc=st.selectbox("Loc",["home","away"])
            opp=st.slider("Avversario",80,120,100)
            sub=st.form_submit_button("🔮",type="primary")
            if sub:
                pts=15+rest*1.5+(3 if loc=="home" else 0)-(opp-100)*0.15
                eff=45+rest*2-(opp-100)*0.2
                c1,c2=st.columns(2)
                c1.metric("Punti",f"{pts:.1f}")
                c2.metric("Eff",f"{eff:.1f}%")

# =================================================================
# MAIN
# =================================================================
st.set_page_config(page_title="CoachTrack",page_icon="🏀",layout="wide")

if 'logged_in' not in st.session_state: st.session_state.logged_in=False
if 'tracking_data' not in st.session_state: st.session_state.tracking_data={}

if not st.session_state.logged_in:
    st.title("🏀 CoachTrack")
    c1,c2,c3=st.columns([1,2,1])
    with c2:
        u=st.text_input("User",value="admin")
        p=st.text_input("Pass",type="password",value="admin")
        if st.button("Login",type="primary",use_container_width=True):
            if u=="admin" and p=="admin":
                st.session_state.logged_in=True
                st.rerun()
    st.stop()

with st.sidebar:
    st.title("🏀 CoachTrack")
    st.markdown("---")
    st.caption("v3.2 Complete Bio")
    if st.button("Logout",use_container_width=True):
        st.session_state.logged_in=False
        st.rerun()

st.title("🏀 CoachTrack Elite v3.2")
st.markdown("Complete: AI + ML + CV + **Full Biometrics** + Analytics")

tab1,tab2,tab3,tab4,tab5=st.tabs(["🏠 Home","🎥 CV","⚖️ Biometrics","📊 Analytics","🤖 ML"])

with tab1:
    st.header("📊 Dashboard")
    st.info("CoachTrack Elite v3.2 - Complete Biometric Edition")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Players",len(st.session_state.tracking_data))
    c2.metric("CV","✅" if CV_AVAILABLE else "❌")
    c3.metric("AI","✅" if AI_ADVANCED_AVAILABLE else "❌")
    if 'biometric_data' in st.session_state:
        bio_count=len(st.session_state.biometric_data['player_id'].unique()) if not st.session_state.biometric_data.empty else 0
    else: bio_count=0
    c4.metric("Bio Players",bio_count)

with tab2:
    add_computer_vision_tab()

with tab3:
    render_biometric_module()

with tab4:
    add_analytics_tab()

with tab5:
    add_ml_tab()
