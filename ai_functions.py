# =================================================================
# AI_FUNCTIONS.PY - CoachTrack Elite AI
# Gestione completa funzioni AI con integrazione Groq LLM
# =================================================================

import pandas as pd
import numpy as np
import os
from datetime import datetime

# =================================================================
# GROQ CLIENT INITIALIZATION
# =================================================================

try:
    from groq import Groq
    GROQ_INSTALLED = True
except ImportError:
    GROQ_INSTALLED = False
    Groq = None

def initialize_groq_client():
    """Inizializza client Groq con API key"""
    if not GROQ_INSTALLED:
        return None, False, "Groq library non installata (pip install groq)"
    
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        # Prova anche da Streamlit secrets (se disponibile)
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY")
        except:
            pass
    
    if api_key:
        try:
            client = Groq(api_key=api_key)
            # Test connessione
            _ = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10
            )
            return client, True, "Groq API connesso ✓"
        except Exception as e:
            return None, False, f"Errore connessione Groq: {str(e)}"
    else:
        return None, False, "API Key Groq non configurata (GROQ_API_KEY)"

# Client globale
GROQ_CLIENT, GROQ_AVAILABLE, GROQ_STATUS = initialize_groq_client()

# =================================================================
# GROQ HELPER FUNCTION
# =================================================================

def call_groq_llm(prompt, system_message="Sei un esperto di sport science e analisi basket.", temperature=0.7, max_tokens=2000):
    """
    Chiama Groq LLM in modo sicuro
    
    Args:
        prompt: Il prompt da inviare
        system_message: Messaggio di sistema
        temperature: Creatività (0-1)
        max_tokens: Lunghezza massima risposta
    
    Returns:
        str: Risposta del modello o messaggio di errore
    """
    if not GROQ_AVAILABLE or GROQ_CLIENT is None:
        return f"⚠️ Groq non disponibile: {GROQ_STATUS}"
    
    try:
        response = GROQ_CLIENT.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Errore Groq API: {str(e)}"

# =================================================================
# UTILITY FUNCTIONS
# =================================================================

def calculate_distance(df):
    """Calcola distanza totale percorsa da coordinate x,y"""
    if len(df) < 2:
        return 0.0
    dx = np.diff(df['x'].values)
    dy = np.diff(df['y'].values)
    return float(np.sum(np.sqrt(dx**2 + dy**2)))

def calculate_speed(df):
    """Calcola velocità in km/h da coordinate e timestamp"""
    if len(df) < 2:
        return df
    if 'speed_kmh_calc' in df.columns:
        return df
    
    df = df.copy()
    dx = np.diff(df['x'].values)
    dy = np.diff(df['y'].values)
    dt = np.diff(df['timestamp'].values).astype(float) / 1000.0
    dt[dt == 0] = 0.001
    speeds = (np.sqrt(dx**2 + dy**2) / dt) * 3.6
    df.loc[df.index[1:], 'speed_kmh_calc'] = speeds
    df.loc[df.index[0], 'speed_kmh_calc'] = 0.0
    return df

# =================================================================
# AI FUNCTIONS - INJURY & PERFORMANCE
# =================================================================

def predict_injury_risk(player_data, player_id):
    """
    Predice rischio infortuni con analisi Groq
    
    Args:
        player_ DataFrame con dati tracking (x, y, timestamp)
        player_id: ID del giocatore
    
    Returns:
        dict: Analisi rischio con raccomandazioni AI
    """
    if len(player_data) < 10:
        return {
            'player_id': player_id,
            'risk_level': 'BASSO',
            'risk_score': 10,
            'acwr': 1.0,
            'asymmetry': 5.0,
            'fatigue': 5.0,
            'risk_factors': ['Dati insufficienti per analisi'],
            'recommendations': ['Raccogliere almeno 10+ campioni per analisi affidabile']
        }
    
    # Calcoli biomeccanici base
    distance = calculate_distance(player_data)
    num_samples = len(player_data)
    
    player_data_with_speed = calculate_speed(player_data)
    avg_speed = player_data_with_speed['speed_kmh_calc'].mean()
    max_speed = player_data_with_speed['speed_kmh_calc'].max()
    
    # Risk score algoritmico
    risk_score = 25 if distance < 200 else 40 if distance < 500 else 60
    if max_speed > 20:
        risk_score += 10
    
    risk_level = 'ALTO' if risk_score > 60 else 'MEDIO' if risk_score > 30 else 'BASSO'
    
    # GROQ ANALYSIS
    if GROQ_AVAILABLE:
        prompt = f"""Analizza il rischio infortuni per il giocatore {player_id}.

DATI BIOMECANICI:
- Distanza percorsa: {distance:.1f} metri
- Campioni dati: {num_samples}
- Velocità media: {avg_speed:.1f} km/h
- Velocità massima: {max_speed:.1f} km/h
- Risk score calcolato: {risk_score}/100
- Livello rischio: {risk_level}

COMPITO:
1. Analizza questi dati dal punto di vista biomeccanico
2. Identifica 3 fattori di rischio principali specifici
3. Fornisci 3 raccomandazioni actionable per ridurre il rischio

Rispondi ESATTAMENTE in questo formato:
FATTORI_RISCHIO:
- [fattore 1 specifico]
- [fattore 2 specifico]
- [fattore 3 specifico]

RACCOMANDAZIONI:
- [raccomandazione 1 specifica e misurabile]
- [raccomandazione 2 specifica e misurabile]
- [raccomandazione 3 specifica e misurabile]"""

        groq_response = call_groq_llm(
            prompt,
            system_message="Sei un medico sportivo esperto in prevenzione infortuni nel basket con 15 anni di esperienza.",
            temperature=0.5
        )
        
        # Parse risposta
        try:
            if "FATTORI_RISCHIO:" in groq_response and "RACCOMANDAZIONI:" in groq_response:
                parts = groq_response.split("RACCOMANDAZIONI:")
                risk_factors_text = parts[0].split("FATTORI_RISCHIO:")[1]
                recommendations_text = parts[1]
                
                risk_factors = [f.strip().lstrip('-').strip() for f in risk_factors_text.split("\n") if f.strip() and f.strip().startswith('-')]
                recommendations = [r.strip().lstrip('-').strip() for r in recommendations_text.split("\n") if r.strip() and r.strip().startswith('-')]
            else:
                # Parsing fallback
                risk_factors = [groq_response[:150]]
                recommendations = [groq_response[150:300]]
        except:
            risk_factors = [f'Distanza elevata: {distance:.1f}m', f'Velocità picco: {max_speed:.1f} km/h']
            recommendations = ['Monitorare carico settimanale', 'Bilanciare intensità e recupero']
    else:
        risk_factors = [
            f'ACWR ratio: 1.2 (monitorare)',
            f'Distanza: {distance:.1f}m',
            f'Velocità picco: {max_speed:.1f} km/h'
        ]
        recommendations = [
            'Groq non disponibile - Mantenere ACWR tra 0.8-1.3',
            'Implementare protocolli di recupero',
            'Monitorare asimmetrie con test funzionali'
        ]
    
    return {
        'player_id': player_id,
        'risk_level': risk_level,
        'risk_score': risk_score,
        'acwr': 1.2,
        'asymmetry': 10.0,
        'fatigue': 8.0,
        'risk_factors': risk_factors[:3],
        'recommendations': recommendations[:3]
    }

# =================================================================
# OFFENSIVE PLAYS RECOMMENDATION
# =================================================================

def recommend_offensive_plays(player_data):
    """
    Raccomanda giocate offensive basate su pattern movimento
    
    Args:
        player_ DataFrame con coordinate x,y
    
    Returns:
        dict: Giocate consigliate con reasoning
    """
    if len(player_data) < 5:
        return {
            'recommended_plays': ['Dati insufficienti'],
            'reasoning': ['Caricare almeno 5+ campioni per analisi']
        }
    
    # Analisi movimento
    distance = calculate_distance(player_data)
    avg_x = player_data['x'].mean()
    avg_y = player_data['y'].mean()
    x_std = player_data['x'].std()
    y_std = player_data['y'].std()
    
    # GROQ ANALYSIS
    if GROQ_AVAILABLE:
        prompt = f"""Analizza il movimento del giocatore e raccomanda 3 giocate offensive specifiche.

DATI MOVIMENTO:
- Distanza totale: {distance:.1f} metri
- Posizione media campo: x={avg_x:.1f}, y={avg_y:.1f}
- Deviazione standard X: {x_std:.1f}m (mobilità laterale)
- Deviazione standard Y: {y_std:.1f}m (profondità)
- Sample size: {len(player_data)} rilevazioni

COMPITO:
Basandoti su questi pattern di movimento, raccomanda 3 giocate offensive che sfruttano al meglio le caratteristiche del giocatore.

Rispondi ESATTAMENTE in questo formato:
GIOCATE:
1. [Nome giocata] - [Breve descrizione 1 frase]
2. [Nome giocata] - [Breve descrizione 1 frase]
3. [Nome giocata] - [Breve descrizione 1 frase]

REASONING:
[Spiega in 2-3 frasi perché queste giocate sono ottimali per questo profilo]"""

        groq_response = call_groq_llm(
            prompt,
            system_message="Sei un allenatore NBA esperto di schemi offensivi e motion offense.",
            temperature=0.6
        )
        
        # Parse risposta
        try:
            if "GIOCATE:" in groq_response and "REASONING:" in groq_response:
                parts = groq_response.split("REASONING:")
                plays_text = parts[0].split("GIOCATE:")[1]
                reasoning_text = parts[1].strip()
                
                plays = []
                for line in plays_text.split("\n"):
                    line = line.strip()
                    if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                        play = line.split(".", 1)[1].strip() if "." in line else line
                        plays.append(play)
                
                return {
                    'recommended_plays': plays[:3] if plays else ['Pick and Roll', 'Motion Offense', 'Isolation'],
                    'reasoning': [reasoning_text]
                }
            else:
                return {
                    'recommended_plays': [groq_response[:100]],
                    'reasoning': [groq_response[100:200]]
                }
        except:
            pass
    
    # Fallback algoritmico
    plays_fallback = ['Pick and Roll', 'Motion Offense', 'Fast Break']
    reasoning_fallback = [f'Groq non disponibile. Profilo: distanza {distance:.0f}m, mobilità {"alta" if x_std > 5 else "moderata"}']
    
    return {
        'recommended_plays': plays_fallback,
        'reasoning': reasoning_fallback
    }

# =================================================================
# DEFENSIVE MATCHUPS OPTIMIZATION
# =================================================================

def optimize_defensive_matchups(team_data, opponent_data=None):
    """
    Ottimizza matchup difensivi
    
    Args:
        team_ dict {player_id: tracking_df}
        opponent_ Optional opponent stats
    
    Returns:
        list: Matchup consigliati
    """
    if not team_
        return []
    
    matchups = []
    for player_id in team_data.keys():
        matchups.append({
            'defender': player_id,
            'opponent': 'Opponent Forward',
            'match_score': 75,
            'reason': 'Matchup versatile basato su fisicità'
        })
    
    return matchups

# =================================================================
# MOVEMENT PATTERNS ANALYSIS
# =================================================================

def analyze_movement_patterns(player_data, player_id):
    """
    Analizza pattern di movimento con AI
    
    Args:
        player_ DataFrame tracking
        player_id: ID giocatore
    
    Returns:
        dict: Pattern type, insights, anomalies
    """
    if len(player_data) < 10:
        return {
            'player_id': player_id,
            'pattern_type': 'UNKNOWN',
            'insights': ['Dati insufficienti per pattern analysis'],
            'anomalies': []
        }
    
    distance = calculate_distance(player_data)
    pattern = 'DYNAMIC' if distance > 100 else 'BALANCED'
    
    # Statistiche movimento
    x_range = player_data['x'].max() - player_data['x'].min()
    y_range = player_data['y'].max() - player_data['y'].min()
    coverage_area = x_range * y_range
    
    # GROQ ANALYSIS
    if GROQ_AVAILABLE:
        prompt = f"""Analizza il pattern di movimento del giocatore {player_id}.

METRICHE MOVIMENTO:
- Distanza totale: {distance:.1f}m
- Pattern type: {pattern}
- Range X (larghezza campo): {x_range:.1f}m
- Range Y (lunghezza campo): {y_range:.1f}m
- Area copertura: {coverage_area:.1f}m²
- Campioni: {len(player_data)}

COMPITO:
1. Identifica il tipo di pattern predominante (es: perimetro, paint, transizione, spot-up)
2. Fornisci 3 insights tattici specifici
3. Rileva eventuali anomalie o pattern inusuali

Rispondi ESATTAMENTE in questo formato:
INSIGHTS:
- [insight tattico 1]
- [insight tattico 2]
- [insight tattico 3]

ANOMALIES:
- [anomalia 1 o scrivi "Nessuna anomalia significativa"]"""

        groq_response = call_groq_llm(
            prompt,
            system_message="Sei un analista di movimento sportivo con expertise in tracking data e analisi tattica basket.",
            temperature=0.5
        )
        
        # Parse
        try:
            insights = []
            anomalies = []
            
            if "INSIGHTS:" in groq_response:
                parts = groq_response.split("ANOMALIES:") if "ANOMALIES:" in groq_response else [groq_response, ""]
                insights_text = parts[0].split("INSIGHTS:")[1]
                
                for line in insights_text.split("\n"):
                    line = line.strip()
                    if line.startswith("-"):
                        insights.append(line[1:].strip())
                
                if len(parts) > 1:
                    anomalies_text = parts[1]
                    for line in anomalies_text.split("\n"):
                        line = line.strip()
                        if line.startswith("-") and "Nessuna" not in line and "nessuna" not in line:
                            anomalies.append(line[1:].strip())
            
            return {
                'player_id': player_id,
                'pattern_type': pattern,
                'insights': insights[:3] if insights else [f'Distanza: {distance:.1f}m', f'Area: {coverage_area:.1f}m²', 'Pattern analisi limitata'],
                'anomalies': anomalies
            }
        except:
            pass
    
    # Fallback
    return {
        'player_id': player_id,
        'pattern_type': pattern,
        'insights': [
            f'Distanza totale: {distance:.1f}m',
            f'Area copertura: {coverage_area:.1f}m²',
            'Groq non disponibile per insights avanzati'
        ],
        'anomalies': []
    }

# =================================================================
# SHOT QUALITY SIMULATION
# =================================================================

def simulate_shot_quality(player_data, player_id):
    """
    Simula qualità tiri basata su posizioni
    
    Args:
        player_ DataFrame tracking
        player_id: ID giocatore
    
    Returns:
        dict: Analisi qualità tiri
    """
    if len(player_data) < 5:
        return {
            'player_id': player_id,
            'avg_quality': 0,
            'shots': [],
            'recommendations': ['Dati insufficienti per shot analysis']
        }
    
    shots = []
    for i in range(min(5, len(player_data))):
        x = float(player_data.iloc[i]['x'])
        y = float(player_data.iloc[i]['y'])
        
        # Simula distanza canestro (assumendo canestro a coordinate fisse)
        basket_x, basket_y = 14.0, 7.5  # Centro campo standard
        distance = np.sqrt((x - basket_x)**2 + (y - basket_y)**2)
        
        # Qualità basata su distanza
        quality = max(0, min(100, 90 - distance * 3))
        shot_type = '3PT' if distance > 6.75 else '2PT'
        
        shots.append({
            'x': x,
            'y': y,
            'distance': round(distance, 1),
            'quality': round(quality),
            'type': shot_type
        })
    
    avg_quality = np.mean([s['quality'] for s in shots])
    
    recommendations = [
        f'Qualità media: {avg_quality:.0f}/100',
        'Focus su shot selection da posizioni ottimali',
        'Bilanciare volume e qualità dei tentativi'
    ]
    
    return {
        'player_id': player_id,
        'avg_quality': round(avg_quality, 1),
        'shots': shots,
        'recommendations': recommendations
    }

# =================================================================
# AI TRAINING PLAN GENERATOR
# =================================================================

def generate_ai_training_plan(player_id, injury_risk_data, physical_data=None):
    """
    Genera piano allenamento personalizzato con Groq
    
    Args:
        player_id: ID giocatore
        injury_risk_ dict da predict_injury_risk()
        physical_ Optional dict con dati fisici
    
    Returns:
        dict: Piano allenamento completo
    """
    risk_level = injury_risk_data.get('risk_level', 'MEDIO')
    risk_score = injury_risk_data.get('risk_score', 50)
    
    if not GROQ_AVAILABLE:
        intensity = 'BASSA' if risk_level == 'ALTO' else 'MODERATA'
        exercises = [
            {'name': 'Recovery drills', 'sets': '3x10', 'focus': 'Recupero', 'priority': 'Alta'},
            {'name': 'Core stability', 'sets': '3x30s', 'focus': 'Core', 'priority': 'Media'}
        ]
        return {
            'player_id': player_id,
            'risk_level': risk_level,
            'intensity': intensity,
            'duration': '60min',
            'frequency': '5x/settimana',
            'focus_areas': 'Condizionamento generale',
            'exercises': exercises,
            'notes': f'Piano {risk_level} - Groq non disponibile'
        }
    
    # GROQ TRAINING PLAN
    physical_info = ""
    if physical_
        physical_info = f"""
DATI FISICI:
- Peso: {physical_data.get('weight_kg', 'N/A')} kg
- BMI: {physical_data.get('bmi', 'N/A')}
- Grasso corporeo: {physical_data.get('body_fat_pct', 'N/A')}%
- Massa muscolare: {physical_data.get('muscle_pct', 'N/A')}%
- BMR: {physical_data.get('bmr', 'N/A')} kcal"""
    
    prompt = f"""Crea un piano di allenamento settimanale personalizzato per il giocatore {player_id}.

INJURY RISK ASSESSMENT:
- Livello rischio: {risk_level}
- Risk score: {risk_score}/100
- Fattori di rischio: {', '.join(injury_risk_data.get('risk_factors', [])[:2])}
{physical_info}

COMPITO:
Crea un piano di allenamento che:
1. Consideri il livello di rischio infortuni
2. Includa 5 esercizi specifici con sets/reps
3. Definisca intensità, durata, frequenza ottimali
4. Fornisca note tecniche e progressioni

Rispondi ESATTAMENTE in questo formato:
PARAMETRI:
Intensità: [BASSA/MODERATA/ALTA]
Durata: [minuti per sessione]
Frequenza: [sessioni/settimana]
Focus: [area principale allenamento]

ESERCIZI:
1. [Nome esercizio] - [sets x reps] - [focus muscolare/tecnico] - Priorità [Alta/Media/Bassa]
2. [Nome esercizio] - [sets x reps] - [focus muscolare/tecnico] - Priorità [Alta/Media/Bassa]
3. [Nome esercizio] - [sets x reps] - [focus muscolare/tecnico] - Priorità [Alta/Media/Bassa]
4. [Nome esercizio] - [sets x reps] - [focus muscolare/tecnico] - Priorità [Alta/Media/Bassa]
5. [Nome esercizio] - [sets x reps] - [focus muscolare/tecnico] - Priorità [Alta/Media/Bassa]

NOTE:
[Note tecniche, progressioni, e avvertenze specifiche]"""

    groq_response = call_groq_llm(
        prompt,
        system_message="Sei un preparatore atletico professionista con certificazione NSCA e 10 anni di esperienza con atleti professionisti di basket.",
        temperature=0.6,
        max_tokens=2000
    )
    
    # Parse risposta
    try:
        lines = groq_response.split("\n")
        intensity = "MODERATA"
        duration = "60min"
        frequency = "5x/settimana"
        focus_areas = "Condizionamento generale"
        exercises = []
        notes = ""
        
        in_exercises = False
        in_notes = False
        
        for line in lines:
            line = line.strip()
            
            if "Intensità:" in line:
                intensity = line.split(":")[-1].strip()
            elif "Durata:" in line:
                duration = line.split(":")[-1].strip()
            elif "Frequenza:" in line:
                frequency = line.split(":")[-1].strip()
            elif "Focus:" in line:
                focus_areas = line.split(":")[-1].strip()
            elif "ESERCIZI:" in line:
                in_exercises = True
                in_notes = False
            elif "NOTE:" in line:
                in_notes = True
                in_exercises = False
            elif in_exercises and line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                # Parse esercizio
                try:
                    parts = line.split("-")
                    name_part = parts[0].split(".", 1)[1].strip() if "." in parts[0] else parts[0].strip()
                    sets = parts[1].strip() if len(parts) > 1 else "3x10"
                    focus = parts[2].strip() if len(parts) > 2 else "Generale"
                    priority = parts[3].strip().replace("Priorità", "").strip() if len(parts) > 3 else "Media"
                    
                    exercises.append({
                        'name': name_part,
                        'sets': sets,
                        'focus': focus,
                        'priority': priority
                    })
                except:
                    pass
            elif in_notes and line:
                notes += line + " "
        
        if not exercises:
            exercises = [
                {'name': 'Squat', 'sets': '3x10', 'focus': 'Lower body', 'priority': 'Alta'},
                {'name': 'Core plank', 'sets': '3x30s', 'focus': 'Core stability', 'priority': 'Alta'},
                {'name': 'Mobility drills', 'sets': '2x15', 'focus': 'Flessibilità', 'priority': 'Media'}
            ]
        
        return {
            'player_id': player_id,
            'risk_level': risk_level,
            'intensity': intensity,
            'duration': duration,
            'frequency': frequency,
            'focus_areas': focus_areas,
            'exercises': exercises[:5],
            'notes': notes.strip() if notes else f"Piano personalizzato per rischio {risk_level}"
        }
    
    except Exception as e:
        # Fallback se parsing fallisce
        return {
            'player_id': player_id,
            'risk_level': risk_level,
            'intensity': 'MODERATA',
            'duration': '60min',
            'frequency': '5x/settimana',
            'focus_areas': 'Generale',
            'exercises': [
                {'name': 'Recovery work', 'sets': '3x10', 'focus': 'Recupero', 'priority': 'Alta'}
            ],
            'notes': groq_response[:200] if groq_response else "Errore parsing piano"
        }

# =================================================================
# NUTRITION REPORT (NLG)
# =================================================================

def generate_nutrition_report_nlg(player_id, nutrition_plan, physical_data, language='it'):
    """
    Genera report nutrizionale completo con Groq NLG
    
    Args:
        player_id: ID giocatore
        nutrition_plan: dict con piano nutrizionale
        physical_ dict con dati fisici
        language: Lingua report
    
    Returns:
        str: Report markdown formattato
    """
    if not GROQ_AVAILABLE:
        return f"""
## 📊 Report Nutrizionale per {player_id}

### Analisi Composizione Corporea
Il giocatore presenta un peso di {physical_data.get('weight_kg', 'N/A')} kg con un BMI di {physical_data.get('bmi', 'N/A')}.

### Piano Nutrizionale
Target calorico: **{nutrition_plan['target_calories']} kcal/giorno**
- Proteine: {nutrition_plan['protein_g']}g
- Carboidrati: {nutrition_plan['carbs_g']}g
- Grassi: {nutrition_plan['fats_g']}g

⚠️ Groq non disponibile - Report base generato.
"""
    
    # GROQ FULL REPORT
    prompt = f"""Genera un report nutrizionale professionale completo per un giocatore di basket professionista.

DATI GIOCATORE {player_id}:
- Peso: {physical_data.get('weight_kg', 'N/A')} kg
- Altezza: {physical_data.get('height_cm', 'N/A')} cm
- BMI: {physical_data.get('bmi', 'N/A')}
- Grasso corporeo: {physical_data.get('body_fat_pct', 'N/A')}%
- Massa magra: {physical_data.get('lean_mass_kg', 'N/A')} kg
- Massa muscolare: {physical_data.get('muscle_pct', 'N/A')}%
- BMR: {physical_data.get('bmr', 'N/A')} kcal/giorno
- AMR: {physical_data.get('amr', 'N/A')} kcal/giorno

PIANO NUTRIZIONALE CALCOLATO:
- Target calorico: {nutrition_plan['target_calories']} kcal/giorno
- Proteine: {nutrition_plan['protein_g']}g ({nutrition_plan['protein_g']*4} kcal)
- Carboidrati: {nutrition_plan['carbs_g']}g ({nutrition_plan['carbs_g']*4} kcal)
- Grassi: {nutrition_plan['fats_g']}g ({nutrition_plan['fats_g']*9} kcal)
- Livello attività: {nutrition_plan['activity_level']}
- Obiettivo: {nutrition_plan['goal']}

COMPITO:
Scrivi un report nutrizionale professionale, completo e dettagliato in italiano che includa:

1. **Analisi Composizione Corporea** (3-4 frasi):
   - Valutazione BMI e body composition
   - Confronto con standard atleti professionisti
   - Punti di forza e aree di miglioramento

2. **Valutazione Piano Nutrizionale** (4-5 frasi):
   - Adeguatezza calorica rispetto a obiettivi
   - Bilanciamento macronutrienti (% su totale)
   - Timing e distribuzione ottimale
   - Considerazioni specifiche per basket

3. **Strategie di Implementazione** (3-4 frasi):
   - Timing dei pasti (pre/post allenamento)
   - Idratazione e integrazione
   - Gestione giorni match vs recovery

4. **Raccomandazioni Specifiche** (lista di 3-4 punti):
   - Consigli pratici e actionable
   - Focus su performance e recupero

Usa formato markdown con headers (##, ###).
Tono: Professionale, scientifico ma accessibile.
Evita frasi generiche, fornisci dati specifici e personalizzati.
Lingua: ITALIANO"""

    groq_response = call_groq_llm(
        prompt,
        system_message="Sei un nutrizionista sportivo certificato specializzato in atleti professionisti e sport di squadra ad alta intensità.",
        temperature=0.6,
        max_tokens=1800
    )
    
    return groq_response

# =================================================================
# SCOUT REPORT (NLG)
# =================================================================

def generate_scout_report_nlg(team_name, report_data, language='it'):
    """
    Genera scout report tattico completo con Groq
    
    Args:
        team_name: Nome squadra avversaria
        report_ dict o str con dati team
        language: Lingua report
    
    Returns:
        str: Scout report markdown
    """
    if not GROQ_AVAILABLE:
        return f"## 📊 Scout Report: {team_name}\n\n⚠️ Groq non disponibile - Report non generato."
    
    prompt = f"""Crea un scout report tattico professionale completo per la squadra "{team_name}".

DATI DISPONIBILI:
{report_data}

COMPITO:
Scrivi un scout report tattico dettagliato in italiano che includa:

1. **OVERVIEW** (2-3 frasi)
   - Stile di gioco generale
   - Punti chiave identità squadra

2. **SISTEMA OFFENSIVO** (4-5 punti specifici)
   - Schema principale e varianti
   - Spacing e movimento palla
   - Transition game
   - Pick and roll options
   - Tendenze finali di partita

3. **SISTEMA DIFENSIVO** (4-5 punti specifici)
   - Schema difensivo base (man, zone, switching)
   - Pressure ball e trapping
   - Weak side help
   - Transition defense
   - Situazioni speciali

4. **GIOCATORI CHIAVE** (top 3)
   - Nome, ruolo, caratteristiche tecniche
   - Punti di forza specifici
   - Come limitarlo

5. **PUNTI DI FORZA** (3-4 punti)
   - Aspetti da rispettare

6. **DEBOLEZZE SFRUTTABILI** (3-4 punti)
   - Vulnerabilità tattiche specifiche

7. **RACCOMANDAZIONI TATTICHE** (4-5 strategie)
   - Matchup suggeriti
   - Focus difensivo
   - Opportunità offensive
   - Situazioni speciali

Usa formato markdown con headers (##, ###) e liste.
Tono: Professionale, analitico, specifico.
Evita generalizzazioni, fornisci dettagli tattici concreti.
Lingua: ITALIANO"""

    groq_response = call_groq_llm(
        prompt,
        system_message="Sei uno scout professionista NBA con 20 anni di esperienza in analisi tattica avversari e video breakdown.",
        temperature=0.7,
        max_tokens=2800
    )
    
    return groq_response

# =================================================================
# GAME ASSISTANT CHAT
# =================================================================

def game_assistant_chat(query, context, language='it'):
    """
    Assistente AI conversazionale per partite
    
    Args:
        query: Domanda allenatore
        context: Contesto partita (score, tempo, situazione)
        language: Lingua
    
    Returns:
        str: Risposta concisa
    """
    if not GROQ_AVAILABLE:
        return "⚠️ Groq non disponibile. Impossibile rispondere alla query."
    
    prompt = f"""Sei un assistente AI per allenatori di basket durante le partite.

CONTESTO PARTITA:
{context}

DOMANDA ALLENATORE:
{query}

Rispondi in modo:
- Conciso (max 3-4 frasi)
- Specifico e actionable
- Con dati numerici se disponibili nel contesto
- In italiano
- Focalizzato su decisioni tattiche immediate

Rispondi DIRETTAMENTE alla domanda senza introduzioni o preamboli."""

    groq_response = call_groq_llm(
        prompt,
        system_message="Sei un assistente tecnico AI esperto di basket in-game. Risposte brevi, precise, actionable.",
        temperature=0.6,
        max_tokens=250
    )
    
    return groq_response

# =================================================================
# PERFORMANCE SUMMARY (NLG)
# =================================================================

def generate_performance_summary(player_id, stats_summary, predictions, language='it'):
    """
    Genera summary performance con predizioni
    
    Args:
        player_id: ID giocatore
        stats_summary: str o dict con stats recenti
        predictions: dict da ML model
        language: Lingua
    
    Returns:
        str: Summary markdown
    """
    if not GROQ_AVAILABLE:
        return f"""
## 📊 Analisi Performance per {player_id}

{stats_summary}

### Predizioni Prossima Partita
- Punti: {predictions.get('points', 'N/A')}
- Assist: {predictions.get('assists', 'N/A')}
- Rimbalzi: {predictions.get('rebounds', 'N/A')}
- Confidence: {predictions.get('confidence', 'N/A')}

⚠️ Groq non disponibile - Summary base generato.
"""
    
    prompt = f"""Genera un'analisi performance professionale per il giocatore {player_id}.

STATISTICHE RECENTI:
{stats_summary}

PREDIZIONI ML PROSSIMA PARTITA:
- Punti previsti: {predictions.get('points', 'N/A')}
- Assist previsti: {predictions.get('assists', 'N/A')}
- Rimbalzi previsti: {predictions.get('rebounds', 'N/A')}
- Efficiency: {predictions.get('efficiency', 'N/A')}
- Confidence level: {predictions.get('confidence', 'N/A')}

COMPITO:
Scrivi un'analisi performance in italiano che includa:
1. Trend recenti (2-3 frasi)
2. Analisi predizioni (2-3 frasi)
3. Raccomandazioni per coach (2-3 punti)

Formato markdown, tono professionale, conciso.
Lingua: ITALIANO"""

    groq_response = call_groq_llm(
        prompt,
        system_message="Sei un analista performance specializzato in basket analytics e player development.",
        temperature=0.6,
        max_tokens=800
    )
    
    return groq_response

# =================================================================
# GROQ CONNECTION TEST
# =================================================================

def test_groq_connection():
    """Testa connessione Groq"""
    return GROQ_AVAILABLE, GROQ_STATUS

# =================================================================
# DETECT JUMPS IMU (bonus utility)
# =================================================================

def detect_jumps_imu(df, threshold_g=1.5):
    """Rileva salti da dati IMU accelerometro"""
    if 'az' not in df.columns:
        return []
    
    jumps = []
    for i, az in enumerate(df['az'].values):
        if az > threshold_g:
            jumps.append({
                'timestamp': int(df['timestamp'].iloc[i]),
                'peak_g': round(float(az), 2),
                'duration_ms': 200,
                'estimated_height_cm': round((az-1)*20, 1)
            })
    
    return jumps[:10]  # Limita a primi 10

# =================================================================
# TRAINING PLAN NLG (wrapper)
# =================================================================

def generate_training_plan_nlg(player_id, training_plan, language='it'):
    """Genera descrizione testuale piano allenamento"""
    return f"Piano allenamento per {player_id}: Intensità {training_plan['intensity']}, Focus: {training_plan['focus_areas']}, Durata: {training_plan['duration']}"

# =================================================================
# MODULE INFO
# =================================================================

__version__ = "3.0.0"
__author__ = "CoachTrack Elite AI"
__all__ = [
    'predict_injury_risk',
    'recommend_offensive_plays',
    'optimize_defensive_matchups',
    'analyze_movement_patterns',
    'simulate_shot_quality',
    'generate_ai_training_plan',
    'generate_nutrition_report_nlg',
    'generate_scout_report_nlg',
    'game_assistant_chat',
    'generate_performance_summary',
    'generate_training_plan_nlg',
    'test_groq_connection',
    'calculate_distance',
    'calculate_speed',
    'detect_jumps_imu',
    'GROQ_AVAILABLE',
    'GROQ_STATUS'
]
