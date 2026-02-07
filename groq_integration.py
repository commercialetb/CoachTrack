"""
Groq LLM Integration Module
Natural Language Generation for Reports, Analysis, and Game Assistant
"""

import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    print("⚠️ WARNING: GROQ_API_KEY not found in environment variables")
    print("   Create a .env file with: GROQ_API_KEY=your_key_here")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# =================================================================
# GROQ CONFIGURATION
# =================================================================

GROQ_MODELS = {
    'fast': 'llama-3.1-8b-instant',      # Fastest, good for simple tasks
    'balanced': 'llama-3.1-70b-versatile', # Best balance speed/quality
    'quality': 'llama-3.2-90b-text-preview' # Highest quality
}

DEFAULT_MODEL = GROQ_MODELS['balanced']

# =================================================================
# NUTRITION REPORT GENERATOR
# =================================================================

def generate_nutrition_report_nlg(player_id, nutrition_plan, physical_data, language='it'):
    """Generate natural language nutrition report using Groq"""
    
    if not client:
        return "⚠️ Groq non configurato. Aggiungi GROQ_API_KEY nel file .env"
    
    lang_prompts = {
        'it': f"""Sei un nutrizionista sportivo esperto specializzato in basket professionistico.

Genera un report nutrizionale dettagliato e personalizzato per il giocatore {player_id}.

DATI FISICI ATTUALI:
- Peso: {physical_data.get('weight_kg', 'N/A')} kg
- Grasso corporeo: {physical_data.get('body_fat_pct', 'N/A')}%
- Massa magra: {physical_data.get('lean_mass_kg', 'N/A')} kg
- BMI: {physical_data.get('bmi', 'N/A')}
- Acqua corporea: {physical_data.get('body_water_pct', 'N/A')}%
- Massa muscolare: {physical_data.get('muscle_pct', 'N/A')}%

PIANO NUTRIZIONALE CALCOLATO:
- Obiettivo: {nutrition_plan['goal']}
- Livello attività: {nutrition_plan['activity_level']}
- BMR (metabolismo basale): {nutrition_plan['bmr']} kcal
- AMR (fabbisogno totale): {nutrition_plan['amr']} kcal
- Target calorico giornaliero: {nutrition_plan['target_calories']} kcal
- Proteine: {nutrition_plan['protein_g']}g ({nutrition_plan['protein_g']*4} kcal)
- Carboidrati: {nutrition_plan['carbs_g']}g ({nutrition_plan['carbs_g']*4} kcal)
- Grassi: {nutrition_plan['fats_g']}g ({nutrition_plan['fats_g']*9} kcal)

GENERA UN REPORT COMPLETO CHE INCLUDA:

1. **Analisi Composizione Corporea**
   - Valutazione stato fisico attuale
   - Punti di forza e aree di miglioramento
   - Comparazione con standard atleti basket professionisti

2. **Strategia Nutrizionale Personalizzata**
   - Razionale dietro target calorici e macro
   - Perché questi valori sono ottimali per l'obiettivo
   - Adattamenti per giorni di gara vs allenamento

3. **Raccomandazioni Pratiche**
   - Timing ottimale nutrizione (pre/durante/post allenamento)
   - Idratazione specifica (quantità e timing)
   - Alimenti consigliati e da evitare

4. **Piano Integrazione**
   - Integratori specifici consigliati e dosaggi
   - Timing assunzione (quando e perché)
   - Sinergie tra integratori

5. **Strategie di Implementazione**
   - Come strutturare i pasti nella giornata
   - Esempi di pasti concreti
   - Trucchi per aderenza al piano

Scrivi in italiano professionale ma accessibile, con tono esperto ma amichevole.
Lunghezza: 600-800 parole.""",
        
        'en': f"""You are an expert sports nutritionist specialized in professional basketball.

Generate a detailed and personalized nutrition report for player {player_id}.

CURRENT PHYSICAL DATA:
- Weight: {physical_data.get('weight_kg', 'N/A')} kg
- Body Fat: {physical_data.get('body_fat_pct', 'N/A')}%
- Lean Mass: {physical_data.get('lean_mass_kg', 'N/A')} kg
- BMI: {physical_data.get('bmi', 'N/A')}
- Body Water: {physical_data.get('body_water_pct', 'N/A')}%
- Muscle Mass: {physical_data.get('muscle_pct', 'N/A')}%

CALCULATED NUTRITION PLAN:
- Goal: {nutrition_plan['goal']}
- Activity Level: {nutrition_plan['activity_level']}
- BMR (basal metabolic rate): {nutrition_plan['bmr']} kcal
- AMR (total daily energy): {nutrition_plan['amr']} kcal
- Daily calorie target: {nutrition_plan['target_calories']} kcal
- Protein: {nutrition_plan['protein_g']}g ({nutrition_plan['protein_g']*4} kcal)
- Carbohydrates: {nutrition_plan['carbs_g']}g ({nutrition_plan['carbs_g']*4} kcal)
- Fats: {nutrition_plan['fats_g']}g ({nutrition_plan['fats_g']*9} kcal)

GENERATE A COMPLETE REPORT INCLUDING:

1. Body Composition Analysis
2. Personalized Nutrition Strategy
3. Practical Recommendations
4. Supplement Plan
5. Implementation Strategies

Write in professional but accessible English, expert but friendly tone.
Length: 600-800 words."""
    }
    
    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{
                "role": "user",
                "content": lang_prompts[language]
            }],
            temperature=0.7,
            max_tokens=1200
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"⚠️ Errore generazione report: {str(e)}"

# =================================================================
# TRAINING PLAN REPORT GENERATOR
# =================================================================

def generate_training_plan_nlg(player_id, training_plan, injury_risk, physical_data=None, language='it'):
    """Generate natural language training plan using Groq"""
    
    if not client:
        return "⚠️ Groq non configurato"
    
    lang_prompts = {
        'it': f"""Sei un preparatore atletico esperto specializzato in basket professionistico.

Genera un report dettagliato del piano di allenamento per {player_id}.

ANALISI RISCHIO INFORTUNI:
- Livello rischio: {injury_risk['risk_level']}
- Score: {injury_risk['risk_score']}/100
- ACWR (carico acuto/cronico): {injury_risk['acwr']}
- Asimmetria laterale: {injury_risk['asymmetry']}%
- Indice fatica: {injury_risk['fatigue']}%

PIANO ALLENAMENTO PROPOSTO:
- Intensità: {training_plan['intensity']}
- Durata sessioni: {training_plan['duration']}
- Frequenza settimanale: {training_plan['frequency']}
- Focus principale: {training_plan['focus_areas']}

ESERCIZI SPECIFICI:
{chr(10).join([f"- {ex['name']}: {ex['sets']} ({ex['focus']}) - Priorità: {ex['priority']}" for ex in training_plan['exercises']])}

GENERA UN REPORT CHE INCLUDA:

1. **Analisi Situazione Attuale**
   - Valutazione rischio infortuni e sue cause
   - Impatto su programmazione allenamento
   - Timeline recupero/progressione

2. **Razionale Scientifico**
   - Perché questa intensità e frequenza
   - Come il piano riduce rischio infortuni
   - Adattamenti fisiologici attesi

3. **Dettagli Implementazione**
   - Struttura settimanale esatta
   - Progressione carichi nelle settimane
   - Warm-up e cool-down specifici

4. **Monitoraggio e Adattamenti**
   - KPI da tracciare
   - Segnali di warning da osservare
   - Quando e come modificare il piano

5. **Raccomandazioni Integrative**
   - Recovery (sonno, riposo attivo)
   - Nutrizione per supportare il piano
   - Strategie mentali

Scrivi in italiano professionale, tono da coach esperto.
Lunghezza: 500-700 parole.""",
        
        'en': f"""You are an expert athletic trainer specialized in professional basketball.

Generate a detailed training plan report for {player_id}.

[Similar structure in English...]"""
    }
    
    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": lang_prompts[language]}],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"⚠️ Errore: {str(e)}"

# =================================================================
# AUTO-SCOUT REPORT GENERATOR (KILLER FEATURE)
# =================================================================

def generate_scout_report_nlg(opponent_team, opponent_stats, patterns, strengths, weaknesses, language='it'):
    """Generate NBA-style scouting report using Groq"""
    
    if not client:
        return "⚠️ Groq non configurato"
    
    prompt_it = f"""Sei uno scout professionista NBA con 20 anni di esperienza.

Genera un report di scouting dettagliato in stile NBA per l'avversario: {opponent_team}

STATISTICHE CHIAVE:
{opponent_stats}

PATTERN TATTICI IDENTIFICATI:
{patterns}

PUNTI DI FORZA:
{strengths}

PUNTI DEBOLI:
{weaknesses}

GENERA UN REPORT PROFESSIONALE CHE INCLUDA:

1. **Executive Summary** (2-3 frasi)
   - Valutazione complessiva squadra
   - Livello minaccia (scala 1-10)

2. **Analisi Offensiva**
   - Sistemi di gioco principali
   - Giocatori chiave e loro ruoli
   - Tendenze in situazioni cruciali (ultimi 5 min, comeback)
   - Efficienza (eFG%, TS%, pace)

3. **Analisi Difensiva**
   - Schema difensivo prevalente (zone, man, switch)
   - Vulnerabilità principali
   - Come attaccano pick&roll, spot-up, transition

4. **Matchup Individuali Critici**
   - Chi dobbiamo fermare assolutamente
   - Chi possiamo attaccare in difesa
   - Suggerimenti difensive specifici

5. **Strategia Consigliata**
   - Piano di gioco offensivo (3-4 punti chiave)
   - Piano di gioco difensivo (3-4 punti chiave)
   - Situazioni speciali (ATO, last shot, etc.)

6. **Keys to Victory** (3 punti essenziali)

Scrivi in italiano professionale stile report NBA.
Lunghezza: 700-900 parole.
Usa dati e numeri quando possibile."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODELS['quality'],  # Use highest quality for scouting
            messages=[{"role": "user", "content": prompt_it}],
            temperature=0.6,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"⚠️ Errore: {str(e)}"

# =================================================================
# REAL-TIME GAME ASSISTANT (GAME CHANGER)
# =================================================================

def game_assistant_chat(question, game_context, language='it'):
    """Real-time game assistant - answers coach questions during game"""
    
    if not client:
        return "⚠️ Groq non configurato"
    
    system_prompt_it = """Sei un assistente AI esperto di basket professionistico che supporta l'allenatore durante la partita.

RUOLO:
- Analizza situazione di gioco in tempo reale
- Suggerisci strategie tattiche immediate
- Proponi cambi lineup e timeout
- Rispondi velocemente (max 150 parole)

STILE:
- Diretto e conciso
- Usa dati quando disponibili
- Priorità a suggerimenti actionable
- Tono professionale da assistant coach"""

    context_str = f"""CONTESTO PARTITA:
Score: {game_context.get('score', 'N/A')}
Tempo rimanente: {game_context.get('time_remaining', 'N/A')}
Periodo: {game_context.get('quarter', 'N/A')}
Lineup attuale: {game_context.get('current_lineup', 'N/A')}
Statistiche chiave: {game_context.get('stats', 'N/A')}

DOMANDA COACH: {question}"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODELS['fast'],  # Fast response for real-time
            messages=[
                {"role": "system", "content": system_prompt_it},
                {"role": "user", "content": context_str}
            ],
            temperature=0.3,  # Lower temp for more factual responses
            max_tokens=300
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"⚠️ Errore: {str(e)}"

# =================================================================
# PERFORMANCE ANALYSIS SUMMARY
# =================================================================

def generate_performance_summary(player_id, stats_history, ml_predictions=None, language='it'):
    """Generate performance analysis summary with trends"""
    
    if not client:
        return "⚠️ Groq non configurato"
    
    prompt = f"""Analizza le performance recenti di {player_id} e genera un report conciso.

STATISTICHE ULTIME 10 PARTITE:
{stats_history}

{"PREDIZIONI ML PROSSIMA PARTITA:\n" + str(ml_predictions) if ml_predictions else ""}

GENERA:
1. Trend performance (miglioramento/calo/stabile)
2. Statistiche chiave in evidenza
3. Pattern positivi identificati
4. Aree di miglioramento
5. Raccomandazioni specifiche (2-3 punti)

Lunghezza: 300-400 parole, italiano professionale."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=600
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"⚠️ Errore: {str(e)}"

# =================================================================
# UTILITY FUNCTIONS
# =================================================================

def test_groq_connection():
    """Test Groq API connection"""
    if not client:
        return False, "API key non configurata"
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODELS['fast'],
            messages=[{"role": "user", "content": "Test connection. Reply with 'OK' only."}],
            max_tokens=10
        )
        return True, "Connessione OK"
    except Exception as e:
        return False, f"Errore: {str(e)}"

def get_available_models():
    """Get list of available Groq models"""
    return GROQ_MODELS
