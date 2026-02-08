"""
CoachTrack Elite - Biometric Monitoring Module
Gestione dati biometrici con input automatico (bilancia) o manuale
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path

# Import integrations (opzionali)
try:
    from integrations.xiaomi_scale import XiaomiScaleIntegration
    XIAOMI_AVAILABLE = True
except ImportError:
    XIAOMI_AVAILABLE = False

try:
    from integrations.withings_scale import WithingsScaleIntegration
    WITHINGS_AVAILABLE = True
except ImportError:
    WITHINGS_AVAILABLE = False


class BiometricModule:
    """Modulo completo gestione biometria atleti"""
    
    def __init__(self, data_dir="data/biometrics"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.measurements_file = self.data_dir / "measurements.csv"
        self.config_file = self.data_dir / "config.json"
        
        # Load existing data
        self.load_data()
        
        # Initialize integrations if configured
        self.scale_integration = None
        self.init_scale_integration()
    
    def load_data(self):
        """Carica dati esistenti"""
        if self.measurements_file.exists():
            self.df = pd.read_csv(self.measurements_file, parse_dates=['timestamp'])
        else:
            # Create empty DataFrame
            self.df = pd.DataFrame(columns=[
                'player_id', 'player_name', 'timestamp', 
                'weight_kg', 'body_fat_pct', 'muscle_mass_kg',
                'water_pct', 'bone_mass_kg', 'bmr_kcal',
                'measurement_type', 'source', 'notes'
            ])
    
    def save_data(self):
        """Salva dati su CSV"""
        self.df.to_csv(self.measurements_file, index=False)
    
    def init_scale_integration(self):
        """Inizializza integrazione bilancia se configurata"""
        if not self.config_file.exists():
            return
        
        with open(self.config_file) as f:
            config = json.load(f)
        
        scale_type = config.get('scale_type')
        
        if scale_type == 'xiaomi' and XIAOMI_AVAILABLE:
            self.scale_integration = XiaomiScaleIntegration(config['xiaomi'])
        elif scale_type == 'withings' and WITHINGS_AVAILABLE:
            self.scale_integration = WithingsScaleIntegration(config['withings'])
    
    def add_measurement_manual(self, player_id, player_name, measurements, notes=""):
        """Aggiungi misurazione manuale"""
        new_row = {
            'player_id': player_id,
            'player_name': player_name,
            'timestamp': datetime.now(),
            'weight_kg': measurements.get('weight'),
            'body_fat_pct': measurements.get('body_fat'),
            'muscle_mass_kg': measurements.get('muscle_mass'),
            'water_pct': measurements.get('water'),
            'bone_mass_kg': measurements.get('bone_mass'),
            'bmr_kcal': measurements.get('bmr'),
            'measurement_type': measurements.get('type', 'manual'),
            'source': 'manual',
            'notes': notes
        }
        
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self.save_data()
        
        return True
    
    def add_measurement_auto(self, measurement_data):
        """Aggiungi misurazione automatica da bilancia"""
        new_row = {
            'player_id': measurement_data['player_id'],
            'player_name': measurement_data['player_name'],
            'timestamp': measurement_data['timestamp'],
            'weight_kg': measurement_data['weight'],
            'body_fat_pct': measurement_data.get('body_fat'),
            'muscle_mass_kg': measurement_data.get('muscle_mass'),
            'water_pct': measurement_data.get('water'),
            'bone_mass_kg': measurement_data.get('bone_mass'),
            'bmr_kcal': measurement_data.get('bmr'),
            'measurement_type': measurement_data.get('type', 'pre_training'),
            'source': measurement_data.get('source', 'scale'),
            'notes': ''
        }
        
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self.save_data()
        
        return True
    
    def get_player_data(self, player_id, days=30):
        """Ottieni dati giocatore ultimi N giorni"""
        cutoff_date = datetime.now() - timedelta(days=days)
        player_df = self.df[
            (self.df['player_id'] == player_id) & 
            (self.df['timestamp'] >= cutoff_date)
        ].sort_values('timestamp')
        
        return player_df
    
    def get_latest_measurement(self, player_id):
        """Ottieni ultima misurazione giocatore"""
        player_df = self.df[self.df['player_id'] == player_id]
        if player_df.empty:
            return None
        return player_df.sort_values('timestamp', ascending=False).iloc[0]
    
    def calculate_trends(self, player_id, days=30):
        """Calcola trend biometrici"""
        player_df = self.get_player_data(player_id, days)
        
        if len(player_df) < 2:
            return None
        
        trends = {}
        
        # Weight trend
        if player_df['weight_kg'].notna().sum() >= 2:
            weight_change = player_df['weight_kg'].iloc[-1] - player_df['weight_kg'].iloc[0]
            trends['weight_change_kg'] = weight_change
            trends['weight_change_pct'] = (weight_change / player_df['weight_kg'].iloc[0]) * 100
        
        # Body fat trend
        if player_df['body_fat_pct'].notna().sum() >= 2:
            bf_change = player_df['body_fat_pct'].iloc[-1] - player_df['body_fat_pct'].iloc[0]
            trends['bodyfat_change_pct'] = bf_change
        
        # Muscle mass trend
        if player_df['muscle_mass_kg'].notna().sum() >= 2:
            muscle_change = player_df['muscle_mass_kg'].iloc[-1] - player_df['muscle_mass_kg'].iloc[0]
            trends['muscle_change_kg'] = muscle_change
        
        return trends
    
    def check_alerts(self, player_id):
        """Verifica anomalie e genera alert"""
        alerts = []
        
        latest = self.get_latest_measurement(player_id)
        if latest is None:
            return alerts
        
        # Get 7-day average
        df_7d = self.get_player_data(player_id, days=7)
        if len(df_7d) < 2:
            return alerts
        
        avg_7d = df_7d['weight_kg'].mean()
        
        # Alert 1: Weight anomaly
        weight_change = latest['weight_kg'] - avg_7d
        if abs(weight_change) > 2.0:
            alerts.append({
                'type': 'weight_anomaly',
                'severity': 'medium' if abs(weight_change) < 3 else 'high',
                'message': f"⚠️ Peso {weight_change:+.1f}kg rispetto media 7 giorni",
                'value': weight_change
            })
        
        # Alert 2: Dehydration
        if pd.notna(latest['water_pct']) and latest['water_pct'] < 55:
            alerts.append({
                'type': 'dehydration',
                'severity': 'high',
                'message': f"🚨 Possibile disidratazione: {latest['water_pct']:.1f}% acqua corporea",
                'value': latest['water_pct']
            })
        
        # Alert 3: Low body fat (underweight risk)
        if pd.notna(latest['body_fat_pct']) and latest['body_fat_pct'] < 6:
            alerts.append({
                'type': 'low_bodyfat',
                'severity': 'medium',
                'message': f"⚠️ Grasso corporeo molto basso: {latest['body_fat_pct']:.1f}%",
                'value': latest['body_fat_pct']
            })
        
        return alerts


def render_biometric_module():
    """Render Streamlit UI per modulo biometrico"""
    
    st.header("⚖️ Monitoraggio Biometrico")
    
    # Initialize module
    if 'biometric_module' not in st.session_state:
        st.session_state.biometric_module = BiometricModule()
    
    bio_module = st.session_state.biometric_module
    
    # Tabs interni
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "➕ Inserimento Dati",
        "📈 Analisi Trend",
        "⚙️ Configurazione"
    ])
    
    # ========== TAB 1: DASHBOARD ==========
    with tab1:
        render_dashboard(bio_module)
    
    # ========== TAB 2: INPUT DATI ==========
    with tab2:
        render_input_form(bio_module)
    
    # ========== TAB 3: ANALISI ==========
    with tab3:
        render_analysis(bio_module)
    
    # ========== TAB 4: CONFIG ==========
    with tab4:
        render_configuration(bio_module)


def render_dashboard(bio_module):
    """Dashboard overview tutti i giocatori"""
    
    st.subheader("📊 Overview Squadra")
    
    if bio_module.df.empty:
        st.info("👋 Nessun dato disponibile. Inizia inserendo misurazioni nella tab 'Inserimento Dati'.")
        return
    
    # Get latest measurements per player
    latest_per_player = bio_module.df.sort_values('timestamp').groupby('player_id').last()
    
    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Giocatori Monitorati",
            len(latest_per_player),
            delta=None
        )
    
    with col2:
        avg_weight = latest_per_player['weight_kg'].mean()
        st.metric(
            "Peso Medio",
            f"{avg_weight:.1f} kg" if pd.notna(avg_weight) else "N/A"
        )
    
    with col3:
        avg_bf = latest_per_player['body_fat_pct'].mean()
        st.metric(
            "Body Fat Medio",
            f"{avg_bf:.1f}%" if pd.notna(avg_bf) else "N/A"
        )
    
    with col4:
        # Count recent measurements (last 7 days)
        recent = bio_module.df[
            bio_module.df['timestamp'] >= datetime.now() - timedelta(days=7)
        ]
        st.metric(
            "Misurazioni (7gg)",
            len(recent)
        )
    
    st.divider()
    
    # Alert section
    st.subheader("🚨 Alert Attivi")
    
    all_alerts = []
    for player_id in latest_per_player.index:
        player_alerts = bio_module.check_alerts(player_id)
        if player_alerts:
            player_name = latest_per_player.loc[player_id, 'player_name']
            for alert in player_alerts:
                alert['player'] = player_name
                all_alerts.append(alert)
    
    if all_alerts:
        for alert in all_alerts:
            if alert['severity'] == 'high':
                st.error(f"**{alert['player']}**: {alert['message']}")
            else:
                st.warning(f"**{alert['player']}**: {alert['message']}")
    else:
        st.success("✅ Nessun alert attivo - Tutti i parametri nella norma")
    
    st.divider()
    
    # Table ultima misurazione per giocatore
    st.subheader("📋 Ultime Misurazioni")
    
    display_df = latest_per_player[[
        'player_name', 'timestamp', 'weight_kg', 'body_fat_pct', 
        'muscle_mass_kg', 'water_pct', 'source'
    ]].copy()
    
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%d/%m/%Y %H:%M')
    display_df.columns = ['Giocatore', 'Data', 'Peso (kg)', 'Grasso (%)', 
                          'Muscolo (kg)', 'Acqua (%)', 'Fonte']
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


def render_input_form(bio_module):
    """Form inserimento dati (manuale o automatico)"""
    
    st.subheader("➕ Inserimento Misurazione")
    
    # Mode selection
    input_mode = st.radio(
        "Modalità inserimento:",
        ["📝 Manuale", "📡 Automatico (Bilancia)"],
        horizontal=True
    )
    
    if input_mode == "📝 Manuale":
        render_manual_input(bio_module)
    else:
        render_auto_input(bio_module)


def render_manual_input(bio_module):
    """Form input manuale"""
    
    st.info("💡 Inserisci i dati manualmente. Compila almeno il peso, gli altri campi sono opzionali.")
    
    with st.form("manual_measurement_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Player selection
            player_name = st.text_input(
                "Nome Giocatore *",
                placeholder="es. Mario Rossi"
            )
            
            # Weight (required)
            weight = st.number_input(
                "Peso (kg) *",
                min_value=40.0,
                max_value=150.0,
                value=75.0,
                step=0.1,
                format="%.1f"
            )
            
            # Body fat (optional)
            body_fat = st.number_input(
                "Grasso Corporeo (%)",
                min_value=3.0,
                max_value=50.0,
                value=None,
                step=0.1,
                format="%.1f",
                help="Opzionale - Misurato con plicometro o BIA"
            )
            
            # Muscle mass (optional)
            muscle_mass = st.number_input(
                "Massa Muscolare (kg)",
                min_value=20.0,
                max_value=80.0,
                value=None,
                step=0.1,
                format="%.1f",
                help="Opzionale"
            )
        
        with col2:
            # Water percentage (optional)
            water = st.number_input(
                "Acqua Corporea (%)",
                min_value=40.0,
                max_value=75.0,
                value=None,
                step=0.1,
                format="%.1f",
                help="Opzionale - Normale: 55-65%"
            )
            
            # Measurement type
            measurement_type = st.selectbox(
                "Momento Misurazione",
                ["Pre-allenamento", "Post-allenamento", "Mattina", "Altro"]
            )
            
            # Notes
            notes = st.text_area(
                "Note",
                placeholder="Eventuali annotazioni...",
                height=100
            )
        
        # Submit button
        submitted = st.form_submit_button("💾 Salva Misurazione", type="primary", use_container_width=True)
        
        if submitted:
            if not player_name:
                st.error("❌ Il nome del giocatore è obbligatorio!")
            else:
                # Generate player ID (simple hash)
                import hashlib
                player_id = hashlib.md5(player_name.encode()).hexdigest()[:8]
                
                measurements = {
                    'weight': weight,
                    'body_fat': body_fat if body_fat else None,
                    'muscle_mass': muscle_mass if muscle_mass else None,
                    'water': water if water else None,
                    'type': measurement_type.lower().replace('-', '_')
                }
                
                success = bio_module.add_measurement_manual(
                    player_id=player_id,
                    player_name=player_name,
                    measurements=measurements,
                    notes=notes
                )
                
                if success:
                    st.success(f"✅ Misurazione salvata per {player_name}!")
                    st.balloons()
                    
                    # Check alerts
                    alerts = bio_module.check_alerts(player_id)
                    if alerts:
                        st.warning("⚠️ Nuovi alert generati:")
                        for alert in alerts:
                            st.warning(alert['message'])
                else:
                    st.error("❌ Errore nel salvataggio")


def render_auto_input(bio_module):
    """Input automatico da bilancia"""
    
    if bio_module.scale_integration is None:
        st.warning("⚙️ Bilancia non configurata. Vai nella tab 'Configurazione' per impostare l'integrazione.")
        
        st.info("""
        **Bilance supportate:**
        - 🇨🇳 Xiaomi Mi Body Composition 2 (€35)
        - 🇫🇷 Withings Body Comp (€180)
        
        Dopo la configurazione, le misurazioni saranno sincronizzate automaticamente!
        """)
        return
    
    st.info("📡 Modalità automatica attiva. La bilancia sincronizzerà i dati in tempo reale.")
    
    # Status
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Stato Connessione", "🟢 Connessa" if bio_module.scale_integration.is_connected() else "🔴 Disconnessa")
    
    with col2:
        last_sync = bio_module.scale_integration.get_last_sync()
        st.metric("Ultima Sincronizzazione", last_sync.strftime("%H:%M:%S") if last_sync else "Mai")
    
    # Manual sync button
    if st.button("🔄 Sincronizza Ora", use_container_width=True):
        with st.spinner("Sincronizzazione in corso..."):
            new_measurements = bio_module.scale_integration.sync()
            
            if new_measurements:
                for measurement in new_measurements:
                    bio_module.add_measurement_auto(measurement)
                
                st.success(f"✅ {len(new_measurements)} nuove misurazioni sincronizzate!")
            else:
                st.info("ℹ️ Nessuna nuova misurazione disponibile")


def render_analysis(bio_module):
    """Analisi trend dettagliata"""
    
    st.subheader("📈 Analisi Trend Biometrici")
    
    if bio_module.df.empty:
        st.info("Nessun dato disponibile per l'analisi")
        return
    
    # Player selection
    players = bio_module.df['player_name'].unique()
    selected_player = st.selectbox("Seleziona Giocatore", players)
    
    if not selected_player:
        return
    
    player_id = bio_module.df[bio_module.df['player_name'] == selected_player]['player_id'].iloc[0]
    
    # Time range
    days = st.slider("Periodo analisi (giorni)", 7, 180, 30)
    
    player_df = bio_module.get_player_data(player_id, days)
    
    if len(player_df) < 2:
        st.warning("Dati insufficienti per analisi trend (minimo 2 misurazioni)")
        return
    
    # Trend summary
    trends = bio_module.calculate_trends(player_id, days)
    
    if trends:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            weight_change = trends.get('weight_change_kg', 0)
            st.metric(
                "Variazione Peso",
                f"{weight_change:+.1f} kg",
                delta=f"{trends.get('weight_change_pct', 0):.1f}%"
            )
        
        with col2:
            bf_change = trends.get('bodyfat_change_pct', 0)
            st.metric(
                "Variazione Grasso",
                f"{bf_change:+.1f} %",
                delta="↓ Bene" if bf_change < 0 else "↑ Attenzione" if bf_change > 0 else "→ Stabile"
            )
        
        with col3:
            muscle_change = trends.get('muscle_change_kg', 0)
            st.metric(
                "Variazione Muscolo",
                f"{muscle_change:+.1f} kg",
                delta="↑ Bene" if muscle_change > 0 else "↓ Attenzione" if muscle_change < 0 else "→ Stabile"
            )
    
    # Charts
    st.divider()
    
    # Weight trend
    fig_weight = go.Figure()
    fig_weight.add_trace(go.Scatter(
        x=player_df['timestamp'],
        y=player_df['weight_kg'],
        mode='lines+markers',
        name='Peso',
        line=dict(color='#3498DB', width=3),
        marker=dict(size=8)
    ))
    
    fig_weight.update_layout(
        title="📊 Trend Peso",
        xaxis_title="Data",
        yaxis_title="Peso (kg)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_weight, use_container_width=True)
    
    # Body composition (if available)
    if player_df['body_fat_pct'].notna().any():
        fig_composition = go.Figure()
        
        fig_composition.add_trace(go.Scatter(
            x=player_df['timestamp'],
            y=player_df['body_fat_pct'],
            mode='lines+markers',
            name='Grasso %',
            line=dict(color='#E74C3C', width=2)
        ))
        
        if player_df['muscle_mass_kg'].notna().any():
            fig_composition.add_trace(go.Scatter(
                x=player_df['timestamp'],
                y=player_df['muscle_mass_kg'],
                mode='lines+markers',
                name='Muscolo kg',
                yaxis='y2',
                line=dict(color='#27AE60', width=2)
            ))
        
        fig_composition.update_layout(
            title="📊 Body Composition",
            xaxis_title="Data",
            yaxis_title="Grasso (%)",
            yaxis2=dict(
                title="Muscolo (kg)",
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_composition, use_container_width=True)


def render_configuration(bio_module):
    """Configurazione integrazione bilancia"""
    
    st.subheader("⚙️ Configurazione Bilancia")
    
    scale_type = st.selectbox(
        "Tipo di Bilancia",
        ["Nessuna (Solo manuale)", "Xiaomi Mi Body", "Withings Body Comp"]
    )
    
    if scale_type == "Nessuna (Solo manuale)":
        st.info("✏️ Modalità manuale: inserisci i dati tramite form nella tab 'Inserimento Dati'")
        return
    
    elif scale_type == "Xiaomi Mi Body":
        st.markdown("""
        ### 🇨🇳 Xiaomi Mi Body Composition 2
        
        **Setup:**
        1. Scarica app **Mi Fit** su smartphone
        2. Crea account Xiaomi e associa bilancia
        3. Inserisci credenziali qui sotto per sync automatico
        
        **Costo:** ~€35  
        **Accuracy:** ±0.2kg peso, ±2% body fat
        """)
        
        with st.form("xiaomi_config"):
            xiaomi_email = st.text_input("Email Account Xiaomi")
            xiaomi_password = st.text_input("Password", type="password")
            
            submitted = st.form_submit_button("💾 Salva Configurazione")
            
            if submitted:
                config = {
                    'scale_type': 'xiaomi',
                    'xiaomi': {
                        'email': xiaomi_email,
                        'password': xiaomi_password
                    }
                }
                
                with open(bio_module.config_file, 'w') as f:
                    json.dump(config, f)
                
                st.success("✅ Configurazione salvata! Riavvia l'app per attivare l'integrazione.")
    
    elif scale_type == "Withings Body Comp":
        st.markdown("""
        ### 🇫🇷 Withings Body Comp
        
        **Setup:**
        1. Crea account su [Withings Developer](https://developer.withings.com)
        2. Crea applicazione e ottieni API keys
        3. Inserisci credenziali qui sotto
        
        **Costo:** ~€180  
        **Accuracy:** Medical-grade, ±0.1kg peso
        """)
        
        with st.form("withings_config"):
            client_id = st.text_input("Client ID")
            client_secret = st.text_input("Client Secret", type="password")
            
            submitted = st.form_submit_button("💾 Salva Configurazione")
            
            if submitted:
                config = {
                    'scale_type': 'withings',
                    'withings': {
                        'client_id': client_id,
                        'client_secret': client_secret
                    }
                }
                
                with open(bio_module.config_file, 'w') as f:
                    json.dump(config, f)
                
                st.success("✅ Configurazione salvata! Riavvia l'app per attivare l'integrazione.")
    
    st.divider()
    
    # Export/Import data
    st.subheader("📁 Gestione Dati")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Esporta Dati (CSV)", use_container_width=True):
            csv = bio_module.df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=f"biometric_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        uploaded_file = st.file_uploader("📤 Importa Dati (CSV)", type=['csv'])
        if uploaded_file:
            import_df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
            bio_module.df = pd.concat([bio_module.df, import_df], ignore_index=True)
            bio_module.save_data()
            st.success(f"✅ Importati {len(import_df)} record!")


# Export function per app.py
__all__ = ['render_biometric_module']
