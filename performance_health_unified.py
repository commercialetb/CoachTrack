# =================================================================
# PERFORMANCE & HEALTH UNIFIED MODULE - OPZIONE B
# Unifica Physical + Nutrition + Biometrics + Load
# =================================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import hashlib

# =================================================================
# HELPER FUNCTIONS (da physical_nutrition.py)
# =================================================================

def generate_enhanced_nutrition(player_id, physical_data, activity_level, goal):
    """Generate personalized nutrition plan"""
    weight = physical_data.get('weight_kg', 80)
    bmr = physical_data.get('bmr', 1800)

    # Activity multipliers
    multipliers = {
        'Low (Recovery)': 1.3,
        'Moderate (Training)': 1.55,
        'High (Intense)': 1.75
    }

    amr = int(bmr * multipliers.get(activity_level, 1.55))

    # Goal adjustments
    goal_adj = {'Maintenance': 1.0, 'Muscle Gain': 1.15, 'Fat Loss': 0.85}
    target_cal = int(amr * goal_adj.get(goal, 1.0))

    protein_g = int(weight * 2.2)
    carbs_g = int(target_cal * 0.5 / 4)
    fats_g = int(target_cal * 0.25 / 9)

    return {
        'player_id': player_id,
        'target_calories': target_cal,
        'protein_g': protein_g,
        'carbs_g': carbs_g,
        'fats_g': fats_g,
        'recommendations': ['Carbs pre-workout', 'Proteine post-workout'],
        'supplements': ['Whey', 'Creatina'],
        'meals': [
            {'name': 'Colazione', 'calories': int(target_cal * 0.25), 'protein': int(protein_g * 0.25)},
            {'name': 'Pranzo', 'calories': int(target_cal * 0.35), 'protein': int(protein_g * 0.35)},
            {'name': 'Cena', 'calories': int(target_cal * 0.30), 'protein': int(protein_g * 0.30)},
            {'name': 'Snack', 'calories': int(target_cal * 0.10), 'protein': int(protein_g * 0.10)}
        ]
    }


def create_body_composition_viz(physical_data):
    """Create body composition pie chart"""
    fig = go.Figure()

    labels = ['Muscoli', 'Grasso', 'Acqua', 'Altro']
    values = [
        physical_data.get('muscle_pct', 45),
        physical_data.get('body_fat_pct', 12),
        15,
        28
    ]

    fig.add_trace(go.Pie(labels=labels, values=values, hole=0.4))
    fig.update_layout(title="Body Composition", height=400)

    return fig


# =================================================================
# MAIN TAB FUNCTION
# =================================================================

def add_performance_health_tab():
    """TAB UNIFICATO: Physical + Nutrition + Biometrics + Load"""

    st.header("⚡ Performance & Health")
    st.markdown("**Dashboard unificato:** Body Composition + Nutrition + Biometrics + Load Tracking")

    # Initialize session state
    if 'physical_profiles' not in st.session_state:
        st.session_state.physical_profiles = {}

    if 'biometric_data' not in st.session_state:
        st.session_state.biometric_data = pd.DataFrame(columns=[
            'player_id', 'player_name', 'timestamp', 'weight_kg', 'body_fat_pct',
            'muscle_mass_kg', 'water_pct', 'bone_mass_kg', 'bmr_kcal',
            'measurement_type', 'source', 'notes'
        ])

    # =================================================================
    # 4 SUB-TABS
    # =================================================================

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Body Composition",
        "🍽️ Nutrition & Diet",
        "❤️ Biometrics & Recovery",
        "🏋️ Load & Performance"
    ])

    # =================================================================
    # TAB 1: BODY COMPOSITION (Physical Data)
    # =================================================================

    with tab1:
        st.subheader("📊 Body Composition Management")

        existing = ["Nuovo..."] + list(st.session_state.physical_profiles.keys())
        pname = st.selectbox("Seleziona Giocatore", existing, key='perf_player')

        if pname == "Nuovo...":
            pname = st.text_input("Nome Nuovo Giocatore", key='perf_new_name')

        with st.form("body_composition_form"):
            st.markdown("### 📋 Input Dati Fisici")

            col1, col2, col3 = st.columns(3)

            with col1:
                h = st.number_input("Altezza (cm)", 150.0, 230.0, 195.0, 0.5)
                w = st.number_input("Peso (kg)", 50.0, 150.0, 80.0, 0.1)
                age = st.number_input("Età", 15, 45, 25)

            with col2:
                bf = st.number_input("Grasso (%)", 3.0, 40.0, 12.0, 0.1)
                water = st.number_input("Acqua (%)", 45.0, 75.0, 60.0, 0.1)
                muscle = st.number_input("Muscoli (%)", 25.0, 60.0, 45.0, 0.1)

            with col3:
                bone = st.number_input("Ossa (kg)", 2.0, 5.0, 3.2, 0.1)
                hr = st.number_input("HR Riposo", 40, 100, 55)
                vo2 = st.number_input("VO2 Max", 30.0, 80.0, 52.0, 0.5)

            submitted = st.form_submit_button("💾 Salva Profilo", type="primary", use_container_width=True)

            if submitted:
                if pname and pname != "Nuovo...":
                    # Calculate metrics
                    bmi = w / ((h/100)**2)
                    fm = w * (bf/100)
                    lm = w - fm
                    bmr = int(10*w + 6.25*h - 5*age + 5)

                    st.session_state.physical_profiles[pname] = {
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'height_cm': h, 'weight_kg': w, 'age': age,
                        'bmi': round(bmi, 1), 'body_fat_pct': bf,
                        'lean_mass_kg': round(lm, 1), 'fat_mass_kg': round(fm, 1),
                        'body_water_pct': water, 'muscle_pct': muscle,
                        'bone_mass_kg': bone, 'resting_hr': hr,
                        'vo2_max': vo2, 'bmr': bmr, 'amr': int(bmr * 1.55)
                    }

                    st.success(f"✅ Profilo salvato per {pname}!")
                    st.balloons()
                    st.rerun()

        # Show saved profiles
        if st.session_state.physical_profiles:
            st.markdown("---")
            st.markdown("### 📊 Profili Salvati")

            for pid, data in st.session_state.physical_profiles.items():
                with st.expander(f"👤 {pid}"):
                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.metric("Peso", f"{data.get('weight_kg')}kg")
                        st.metric("BMI", data.get('bmi'))

                    with c2:
                        st.metric("Grasso", f"{data.get('body_fat_pct')}%")
                        st.metric("Massa Magra", f"{data.get('lean_mass_kg')}kg")

                    with c3:
                        st.metric("BMR", f"{data.get('bmr')} kcal")
                        st.metric("VO2 Max", data.get('vo2_max'))

                    # Visualization
                    fig = create_body_composition_viz(data)
                    st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # TAB 2: NUTRITION & DIET
    # =================================================================

    with tab2:
        st.subheader("🍽️ Nutrition & Diet Planning")

        if not st.session_state.physical_profiles:
            st.warning("⚠️ Crea prima un profilo fisico nel tab Body Composition")
        else:
            pn = st.selectbox("Seleziona Giocatore", list(st.session_state.physical_profiles.keys()), key='nutr_player')

            col1, col2 = st.columns(2)

            with col1:
                act = st.selectbox("Livello Attività", ["Low (Recovery)", "Moderate (Training)", "High (Intense)"])

            with col2:
                goal = st.selectbox("Obiettivo", ["Maintenance", "Muscle Gain", "Fat Loss"])

            if st.button("🍎 Genera Piano Nutrizionale", type="primary", use_container_width=True):
                plan = generate_enhanced_nutrition(pn, st.session_state.physical_profiles[pn], act, goal)

                st.success("✅ Piano nutrizionale generato!")

                # Macros
                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.metric("🔥 Calorie", plan['target_calories'])

                with c2:
                    st.metric("🥩 Proteine", f"{plan['protein_g']}g")

                with c3:
                    st.metric("🍚 Carboidrati", f"{plan['carbs_g']}g")

                with c4:
                    st.metric("🥑 Grassi", f"{plan['fats_g']}g")

                st.markdown("---")

                # Meals
                st.markdown("### 🍽️ Piano Pasti")

                for meal in plan['meals']:
                    with st.expander(f"**{meal['name']}** - {meal['calories']} kcal"):
                        st.write(f"**Proteine:** {meal['protein']}g")

                # Recommendations
                st.markdown("### 💡 Raccomandazioni")
                for rec in plan['recommendations']:
                    st.info(f"• {rec}")

                # Supplements
                st.markdown("### 💊 Integratori Consigliati")
                for supp in plan['supplements']:
                    st.success(f"✅ {supp}")

    # =================================================================
    # TAB 3: BIOMETRICS & RECOVERY
    # =================================================================

    with tab3:
        st.subheader("❤️ Biometrics & Recovery Monitoring")

        subtab1, subtab2, subtab3 = st.tabs(["📊 Dashboard", "➕ Nuovo Dato", "📈 Trend"])

        # SUBTAB 1: Dashboard
        with subtab1:
            if st.session_state.biometric_data.empty:
                st.info("👋 Nessun dato biometrico. Inizia inserendo una misurazione nel tab 'Nuovo Dato'")
            else:
                latest = st.session_state.biometric_data.sort_values('timestamp').groupby('player_id').last()

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Giocatori Monitorati", len(latest))

                with col2:
                    avg_weight = latest['weight_kg'].mean()
                    st.metric("Peso Medio", f"{avg_weight:.1f} kg" if pd.notna(avg_weight) else "N/A")

                with col3:
                    avg_bf = latest['body_fat_pct'].mean()
                    st.metric("Body Fat Medio", f"{avg_bf:.1f}%" if pd.notna(avg_bf) else "N/A")

                with col4:
                    recent = st.session_state.biometric_data[
                        st.session_state.biometric_data['timestamp'] >= datetime.now() - timedelta(days=7)
                    ]
                    st.metric("Misurazioni (7gg)", len(recent))

        # SUBTAB 2: Insert Data
        with subtab2:
            st.markdown("➕ **Inserisci Nuova Misurazione**")

            with st.form("biometric_form"):
                col1, col2 = st.columns(2)

                with col1:
                    player_name = st.text_input("Nome Giocatore *", placeholder="es. Mario Rossi")
                    weight = st.number_input("Peso (kg) *", 40.0, 150.0, 75.0, 0.1)
                    body_fat = st.number_input("Grasso Corporeo (%)", 3.0, 50.0, value=None, step=0.1)
                    muscle_mass = st.number_input("Massa Muscolare (kg)", 20.0, 80.0, value=None, step=0.1)

                with col2:
                    water = st.number_input("Acqua Corporea (%)", 40.0, 75.0, value=None, step=0.1)
                    bone_mass = st.number_input("Massa Ossea (kg)", 2.0, 5.0, value=None, step=0.1)
                    measurement_type = st.selectbox("Momento", ["Pre-allenamento", "Post-allenamento", "Mattina", "Altro"])
                    notes = st.text_area("Note", height=100)

                submitted = st.form_submit_button("💾 Salva Misurazione", type="primary", use_container_width=True)

                if submitted:
                    if not player_name:
                        st.error("❌ Nome giocatore obbligatorio!")
                    else:
                        player_id = hashlib.md5(player_name.encode()).hexdigest()[:8]

                        bmr = None
                        if muscle_mass and body_fat:
                            bmr = int(370 + (21.6 * muscle_mass))

                        new_row = pd.DataFrame([{
                            'player_id': player_id,
                            'player_name': player_name,
                            'timestamp': datetime.now(),
                            'weight_kg': weight,
                            'body_fat_pct': body_fat,
                            'muscle_mass_kg': muscle_mass,
                            'water_pct': water,
                            'bone_mass_kg': bone_mass,
                            'bmr_kcal': bmr,
                            'measurement_type': measurement_type.lower().replace('-', '_'),
                            'source': 'manual',
                            'notes': notes
                        }])

                        st.session_state.biometric_data = pd.concat([
                            st.session_state.biometric_data,
                            new_row
                        ], ignore_index=True)

                        st.success(f"✅ Misurazione salvata per {player_name}!")
                        st.balloons()
                        st.rerun()

        # SUBTAB 3: Trends
        with subtab3:
            if st.session_state.biometric_data.empty:
                st.info("Nessun dato per analisi trend")
            else:
                players = st.session_state.biometric_data['player_name'].unique()
                selected_player = st.selectbox("Seleziona Giocatore", players, key='trend_player')

                if selected_player:
                    player_id = st.session_state.biometric_data[
                        st.session_state.biometric_data['player_name'] == selected_player
                    ]['player_id'].iloc[0]

                    days = st.slider("Periodo (giorni)", 7, 180, 30)
                    cutoff = datetime.now() - timedelta(days=days)

                    player_df = st.session_state.biometric_data[
                        (st.session_state.biometric_data['player_id'] == player_id) &
                        (st.session_state.biometric_data['timestamp'] >= cutoff)
                    ].sort_values('timestamp')

                    if len(player_df) >= 2:
                        # Weight trend
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=player_df['timestamp'],
                            y=player_df['weight_kg'],
                            mode='lines+markers',
                            name='Peso',
                            line=dict(color='#3498DB', width=3)
                        ))
                        fig.update_layout(title="📊 Trend Peso", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Almeno 2 misurazioni necessarie per trend")

    # =================================================================
    # TAB 4: LOAD & PERFORMANCE
    # =================================================================

    with tab4:
        st.subheader("🏋️ Training Load & Performance Tracking")

        st.info("💡 Monitora carico di allenamento, ACWR e injury risk")

        # Demo content
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("ACWR (7:28)", "1.2", delta="⚠️ Medio")

        with col2:
            st.metric("Training Load (7d)", "2,500 AU", delta="+15%")

        with col3:
            st.metric("Recovery Score", "78/100", delta="👍 Buono")

        st.markdown("---")

        # Load chart
        dates = pd.date_range(end=datetime.now(), periods=14, freq='D')
        load = [1800, 2100, 1950, 2300, 2200, 1900, 1500, 2400, 2500, 2300, 2100, 2200, 2400, 2300]

        fig_load = px.line(x=dates, y=load, labels={'x': 'Data', 'y': 'Load (AU)'}, title="📈 Training Load (14 giorni)")
        st.plotly_chart(fig_load, use_container_width=True)

        st.markdown("### 💡 Raccomandazioni")
        st.success("✅ Load controllato - continua monitoraggio")
        st.info("📊 ACWR leggermente elevato - considera 1 giorno recovery")
