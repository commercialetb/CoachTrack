import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================
# Page
# =========================
st.set_page_config(
    page_title="CoachTrack | Basketball Tracking",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Global styles (minimal + pro)
# =========================
st.markdown(
    """
<style>
/* tighter top padding */
div.block-container { padding-top: 1.2rem; padding-bottom: 2.0rem; }

/* hide Streamlit footer/menu (optional; keep if you want super clean) */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* KPI cards */
div[data-testid="stMetric"]{
  background: white;
  border: 1px solid rgba(17,24,39,0.10);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 10px 24px rgba(17,24,39,0.06);
}
div[data-testid="stMetricLabel"]{ font-weight: 650; letter-spacing: 0.2px; }
div[data-testid="stMetricValue"]{ font-weight: 750; }

/* boxed sections */
.ct-box{
  background: white;
  border: 1px solid rgba(17,24,39,0.10);
  border-radius: 16px;
  padding: 16px 16px;
  box-shadow: 0 10px 24px rgba(17,24,39,0.06);
}

/* AI box */
.ai-insight {
  background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
  padding: 1.1rem 1.2rem;
  border-radius: 14px;
  color: white;
  border: 1px solid rgba(255,255,255,0.25);
}

/* mobile */
@media (max-width: 768px) {
  div.block-container { padding-left: 0.9rem; padding-right: 0.9rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Helpers
# =========================
def section_title(title: str, subtitle: str | None = None):
    if subtitle:
        st.markdown(f"### {title}\n<span style='color: rgba(17,24,39,0.70);'>{subtitle}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"### {title}")

def draw_basketball_court():
    court_length = 28.0
    court_width = 15.0
    shapes = []

    shapes.append(dict(type="rect", x0=0, y0=0, x1=court_length, y1=court_width,
                       line=dict(color="white", width=3), fillcolor="rgba(0,0,0,0)"))

    shapes.append(dict(type="line", x0=court_length/2, y0=0, x1=court_length/2, y1=court_width,
                       line=dict(color="white", width=2)))

    shapes.append(dict(type="circle", xref="x", yref="y",
                       x0=court_length/2-1.8, y0=court_width/2-1.8,
                       x1=court_length/2+1.8, y1=court_width/2+1.8,
                       line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)"))

    shapes.append(dict(type="path",
                       path=f"M 0,{court_width/2-6.75} Q 6.75,{court_width/2} 0,{court_width/2+6.75}",
                       line=dict(color="white", width=2)))

    shapes.append(dict(type="path",
                       path=f"M {court_length},{court_width/2-6.75} Q {court_length-6.75},{court_width/2} {court_length},{court_width/2+6.75}",
                       line=dict(color="white", width=2)))

    shapes.append(dict(type="circle", x0=5.8-1.8, y0=court_width/2-1.8,
                       x1=5.8+1.8, y1=court_width/2+1.8,
                       line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)"))

    shapes.append(dict(type="circle", x0=court_length-5.8-1.8, y0=court_width/2-1.8,
                       x1=court_length-5.8+1.8, y1=court_width/2+1.8,
                       line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)"))

    shapes.append(dict(type="rect", x0=0, y0=court_width/2-1.25,
                       x1=1.25, y1=court_width/2+1.25,
                       line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)"))

    shapes.append(dict(type="rect", x0=court_length-1.25, y0=court_width/2-1.25,
                       x1=court_length, y1=court_width/2+1.25,
                       line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)"))

    return shapes

def classify_zone(x, y):
    court_length = 28.0
    court_width = 15.0

    # Paint
    if x <= 5.8 and (court_width/2 - 2.45) <= y <= (court_width/2 + 2.45):
        return "Paint"
    if x >= (court_length - 5.8) and (court_width/2 - 2.45) <= y <= (court_width/2 + 2.45):
        return "Paint"

    # 3PT
    left_basket_x, left_basket_y = 1.575, court_width/2
    if np.sqrt((x - left_basket_x)**2 + (y - left_basket_y)**2) >= 6.75:
        return "3-Point"

    right_basket_x, right_basket_y = court_length - 1.575, court_width/2
    if np.sqrt((x - right_basket_x)**2 + (y - right_basket_y)**2) >= 6.75:
        return "3-Point"

    return "Mid-Range"


@st.cache_data
def load_sample():
    uwb = pd.read_csv(
        "data/virtual_uwb_realistic.csv",
        dtype={"player_id": "category", "quality_factor": "int16"},
    )
    imu = pd.read_csv(
        "data/virtual_imu_realistic.csv",
        dtype={"player_id": "category", "jump_detected": "int8"},
    )
    return uwb, imu


@st.cache_data
def load_uploaded(uwb_bytes, imu_bytes):
    uwb = pd.read_csv(uwb_bytes)
    imu = pd.read_csv(imu_bytes) if imu_bytes is not None else None
    return uwb, imu


@st.cache_data
def calculate_kpi(uwb_data):
    kpi = (
        uwb_data.groupby("player_id")
        .agg(
            points=("timestamp_s", "count"),
            distance_m=("step_m", "sum"),
            avg_speed_kmh=("speed_kmh_calc", "mean"),
            max_speed_kmh=("speed_kmh_calc", "max"),
            avg_quality=("quality_factor", "mean"),
        )
        .reset_index()
    )
    kpi["distance_m"] = kpi["distance_m"].fillna(0)
    return kpi


@st.cache_data
def calculate_zone_stats(uwb_zone_df):
    zone_stats = (
        uwb_zone_df.groupby(["player_id", "zone"])
        .size()
        .reset_index(name="count")
    )
    totals = zone_stats.groupby("player_id")["count"].transform("sum")
    zone_stats["percentage"] = (zone_stats["count"] / totals * 100).round(1)
    return zone_stats


@st.cache_data
def calculate_zone_stats_for_ai(uwb_zone_df):
    z = calculate_zone_stats(uwb_zone_df)
    return z[["player_id", "zone", "percentage"]].copy()


def build_next_training_plan(kpi_df, zone_df=None):
    df = kpi_df.copy()
    bullets = []
    if df.empty:
        return ["Nessun dato KPI disponibile per pianificare l'allenamento."]

    avg_dist = df["distance_m"].mean() if "distance_m" in df.columns else np.nan
    avg_maxs = df["max_speed_kmh"].mean() if "max_speed_kmh" in df.columns else np.nan
    avg_avgs = df["avg_speed_kmh"].mean() if "avg_speed_kmh" in df.columns else np.nan

    under_dist = []
    if np.isfinite(avg_dist) and avg_dist > 0 and "distance_m" in df.columns:
        under_dist = df[df["distance_m"] < 0.90 * avg_dist]["player_id"].astype(str).tolist()

    under_maxs = []
    if np.isfinite(avg_maxs) and avg_maxs > 0 and "max_speed_kmh" in df.columns:
        under_maxs = df[df["max_speed_kmh"] < 0.90 * avg_maxs]["player_id"].astype(str).tolist()

    low_quality = []
    if "avg_quality" in df.columns:
        low_quality = df[df["avg_quality"] < 55]["player_id"].astype(str).tolist()

    # Zone hints (optional)
    if zone_df is not None and not zone_df.empty and {"player_id", "zone", "percentage"}.issubset(zone_df.columns):
        z3 = zone_df[zone_df["zone"] == "3-Point"].sort_values("percentage", ascending=False)
        zp = zone_df[zone_df["zone"] == "Paint"].sort_values("percentage", ascending=False)
        if not z3.empty:
            bullets.append(f"Spaziature: usa {z3.iloc[0]['player_id']} come spacer (3PT {float(z3.iloc[0]['percentage']):.1f}%).")
        if not zp.empty:
            bullets.append(f"Paint presence: coinvolgi {zp.iloc[0]['player_id']} in tagli/roll (Paint {float(zp.iloc[0]['percentage']):.1f}%).")

    if under_dist:
        bullets.append(f"Conditioning: 12–18 min small-sided 3v3/4v4 a tempo; target +10% distanza per {', '.join(under_dist)}.")
    else:
        bullets.append("Conditioning: 10–12 min mantenimento (no extra volume).")

    if under_maxs:
        bullets.append(f"Speed/repeated sprint: 2–3 set di 6×(10–20 m) con recupero breve; focus per {', '.join(under_maxs)}.")
    else:
        bullets.append("Speed: 4–6 sprint ‘quality’ con recuperi lunghi (mantenimento picchi).")

    if np.isfinite(avg_avgs):
        bullets.append("Ritmo: 2 blocchi da 4 min ‘stop&go’ + cambi direzione, intensità costante.")

    if low_quality:
        bullets.append(f"Data check: quality bassa per {', '.join(low_quality)} → verifica NLOS/ancore o aumenta min_q prima di cambiare carico.")

    bullets.append("Prossimo step: confronta Δ% su distance_m e max_speed_kmh nella prossima sessione (stessi filtri).")
    return bullets


def build_individual_training_plan(player_id, kpi_df, zone_df=None):
    if kpi_df is None or kpi_df.empty:
        return {"summary": "Nessun KPI disponibile.", "plan": []}

    row = kpi_df[kpi_df["player_id"].astype(str) == str(player_id)]
    if row.empty:
        return {"summary": f"Nessun KPI per {player_id}.", "plan": []}

    r = row.iloc[0]
    team = kpi_df.copy()

    def safe_mean(col):
        return float(team[col].mean()) if col in team.columns and team[col].notna().any() else np.nan

    def safe_val(sr, col):
        try:
            v = float(sr[col])
            return v if np.isfinite(v) else np.nan
        except Exception:
            return np.nan

    def pct_vs(v, baseline):
        if not np.isfinite(v) or not np.isfinite(baseline) or baseline == 0:
            return None
        return (v / baseline - 1) * 100

    avg_dist = safe_mean("distance_m")
    avg_avgspd = safe_mean("avg_speed_kmh")
    avg_maxspd = safe_mean("max_speed_kmh")
    avg_q = safe_mean("avg_quality")

    p_dist = safe_val(r, "distance_m")
    p_avgspd = safe_val(r, "avg_speed_kmh")
    p_maxspd = safe_val(r, "max_speed_kmh")
    p_q = safe_val(r, "avg_quality")

    dist_pct = pct_vs(p_dist, avg_dist)
    avgspd_pct = pct_vs(p_avgspd, avg_avgspd)
    maxspd_pct = pct_vs(p_maxspd, avg_maxspd)

    def gap_steps(pct, cap=3):
        if pct is None or pct >= -10:
            return 0
        return int(min(cap, np.ceil(abs(pct) / 10)))

    base_cond_min = 12
    base_stopgo_blocks = 2
    base_sprint_sets = 2
    base_sprints_per_set = 6

    cond_steps = gap_steps(dist_pct, cap=3)
    speed_steps = gap_steps(maxspd_pct, cap=3)
    rhythm_steps = gap_steps(avgspd_pct, cap=2)

    cond_min = int(np.clip(base_cond_min + 3 * cond_steps, 10, 24))
    stopgo_blocks = int(np.clip(base_stopgo_blocks + rhythm_steps, 1, 4))
    sprint_sets = int(np.clip(base_sprint_sets + speed_steps, 2, 4))
    sprints_per_set = int(np.clip(base_sprints_per_set + speed_steps, 6, 8))

    summary_parts = []
    if dist_pct is not None: summary_parts.append(f"Distanza {dist_pct:+.0f}% vs media")
    if avgspd_pct is not None: summary_parts.append(f"Avg speed {avgspd_pct:+.0f}% vs media")
    if maxspd_pct is not None: summary_parts.append(f"Max speed {maxspd_pct:+.0f}% vs media")
    if np.isfinite(p_q) and np.isfinite(avg_q): summary_parts.append(f"Quality {p_q:.0f}/100 (team {avg_q:.0f})")

    plan = []
    if np.isfinite(p_q) and p_q < 55:
        plan.append("Nota dati: quality bassa → prima verifica NLOS/ancore/filtri (alza min_q) e poi rivaluta i target.")

    plan.append(f"Condizionamento: {cond_min} min small-sided 3v3/4v4 a tempo (densità modulata sul tuo gap distanza).")
    plan.append(f"Speed/repeated sprint: {sprint_sets} set × {sprints_per_set} sprint (10–20 m), recupero breve; 2–3 min tra set.")
    plan.append(f"Ritmo/consistenza: {stopgo_blocks} blocchi × 4 min ‘stop&go’ + cambi direzione, intensità costante.")

    if zone_df is not None and not zone_df.empty and {"player_id", "zone", "percentage"}.issubset(zone_df.columns):
        z = zone_df[zone_df["player_id"].astype(str) == str(player_id)]
        if not z.empty:
            z = z.sort_values("percentage", ascending=False)
            top_zone = str(z.iloc[0]["zone"])
            top_pct = float(z.iloc[0]["percentage"])
            if top_zone == "3-Point":
                plan.append(f"Skill after fatigue: 8–12 min catch&shoot + relocation (top zone 3PT {top_pct:.1f}%).")
            elif top_zone == "Paint":
                plan.append(f"Skill: 8–12 min finishing con contatto + tagli/roll (top zone Paint {top_pct:.1f}%).")
            else:
                plan.append(f"Skill: 8–12 min pull-up + closeout reads (top zone Mid {top_pct:.1f}%).")

    plan.append("Target seduta: ridurre gap vs media di ~5–10% su distance/avg/max speed senza peggiorare quality.")
    return {"summary": " | ".join(summary_parts) if summary_parts else "Sintesi non disponibile.", "plan": plan}


def generate_ai_insights(kpi_df, zone_df=None):
    """Return ALWAYS (html_text, is_ai_active)."""
    if kpi_df is None or kpi_df.empty:
        return "<b>Nessun dato KPI disponibile</b> (dopo i filtri).", False

    # --- Try Groq if available ---
    try:
        import groq
        api_key = st.secrets.get("GROQ_API_KEY", None)
        if api_key:
            client = groq.Groq(api_key=api_key)

            zone_summary = ""
            if zone_df is not None and not zone_df.empty:
                zone_summary = "\n\nZone (%):\n" + zone_df.to_string(index=False)

            prompt = f"""
Sei un performance analyst di basket.
Genera un report in italiano in 7 sezioni con titoli + un piano allenamento (team) per la prossima seduta.

Sezioni obbligatorie:
1) Overview squadra con benchmark completi
2) Top 3 performers con % confronto vs media squadra
3) Area miglioramento con gap specifici
4) Analisi velocità (picchi, range, consistenza)
5) Raccomandazioni tattiche per coach
6) Anomalie rilevate (quality critica, distanze anomale)
7) Prossimi step actionable (includi anche "Prossimo allenamento" con esercizi e dosi)

KPI:
{kpi_df.to_string(index=False)}
{zone_summary}
"""
            response = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=900,
                temperature=0.6,
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                text = "AI attiva ma risposta vuota (riprovare)."
            return "<br>".join(text.splitlines()), True
    except Exception:
        pass

    # --- Rule-based fallback (7 sezioni + piano team) ---
    df = kpi_df.copy()

    def safe_mean(col):
        return float(df[col].mean()) if col in df.columns and df[col].notna().any() else np.nan

    def safe_max_row(col):
        if col in df.columns and df[col].notna().any():
            return df.loc[df[col].idxmax()]
        return None

    def safe_min_row(col):
        if col in df.columns and df[col].notna().any():
            return df.loc[df[col].idxmin()]
        return None

    team_avg_dist = safe_mean("distance_m")
    team_avg_avgspd = safe_mean("avg_speed_kmh")
    team_avg_maxspd = safe_mean("max_speed_kmh")
    team_avg_q = safe_mean("avg_quality")

    overview = []
    if np.isfinite(team_avg_dist): overview.append(f"Media distanza: {team_avg_dist:.0f} m")
    if np.isfinite(team_avg_avgspd): overview.append(f"Media velocità: {team_avg_avgspd:.1f} km/h")
    if np.isfinite(team_avg_maxspd): overview.append(f"Media picchi: {team_avg_maxspd:.1f} km/h")
    if np.isfinite(team_avg_q): overview.append(f"Quality medio: {team_avg_q:.0f}/100")
    if not overview: overview.append("Dati non sufficienti per benchmark squadra.")

    top3_html = "<i>KPI distanza non disponibile.</i>"
    if "distance_m" in df.columns and df["distance_m"].notna().any() and np.isfinite(team_avg_dist) and team_avg_dist > 0:
        top3 = df.sort_values("distance_m", ascending=False).head(3)
        rows = []
        for _, r in top3.iterrows():
            pct = (float(r["distance_m"]) / team_avg_dist - 1) * 100
            rows.append(f"<li>{r['player_id']}: {float(r['distance_m']):.0f} m ({pct:+.0f}% vs media)</li>")
        top3_html = "<ul>" + "".join(rows) + "</ul>"

    improve = []
    worst_dist = safe_min_row("distance_m")
    best_dist = safe_max_row("distance_m")
    if worst_dist is not None and best_dist is not None:
        gap = float(best_dist["distance_m"]) - float(worst_dist["distance_m"])
        improve.append(f"Gap distanza: <b>{worst_dist['player_id']}</b> è a -{gap:.0f} m rispetto a {best_dist['player_id']}.")

    worst_q = safe_min_row("avg_quality")
    if worst_q is not None and float(worst_q["avg_quality"]) < 60:
        improve.append(f"Qualità bassa: <b>{worst_q['player_id']}</b> avg_quality {float(worst_q['avg_quality']):.0f}/100 (possibile NLOS).")

    if not improve:
        improve.append("Nessun gap critico evidente con i filtri correnti.")

    speed = []
    best_max = safe_max_row("max_speed_kmh")
    worst_avg = safe_min_row("avg_speed_kmh")
    if best_max is not None:
        speed.append(f"Picco migliore: <b>{best_max['player_id']}</b> {float(best_max['max_speed_kmh']):.1f} km/h.")
    if worst_avg is not None and np.isfinite(team_avg_avgspd) and team_avg_avgspd > 0:
        pct = (float(worst_avg["avg_speed_kmh"]) / team_avg_avgspd - 1) * 100
        speed.append(f"Ritmo basso: <b>{worst_avg['player_id']}</b> avg {float(worst_avg['avg_speed_kmh']):.1f} km/h ({pct:+.0f}% vs media).")
    if not speed:
        speed.append("Metriche velocità non disponibili.")

    tactics = []
    if zone_df is not None and not zone_df.empty and {"player_id", "zone", "percentage"}.issubset(zone_df.columns):
        z3 = zone_df[zone_df["zone"] == "3-Point"].sort_values("percentage", ascending=False)
        zp = zone_df[zone_df["zone"] == "Paint"].sort_values("percentage", ascending=False)
        if not z3.empty:
            tactics.append(f"Spaziature: usa <b>{z3.iloc[0]['player_id']}</b> per allargare (3PT {float(z3.iloc[0]['percentage']):.1f}%).")
        if not zp.empty:
            tactics.append(f"Paint presence: coinvolgi <b>{zp.iloc[0]['player_id']}</b> in tagli/roll (Paint {float(zp.iloc[0]['percentage']):.1f}%).")
    if not tactics:
        tactics.append("Usa Zone Analysis per raccomandazioni più mirate (Paint/3PT).")

    anomalies = []
    if "avg_quality" in df.columns:
        lowq = df[df["avg_quality"] < 50]
        if not lowq.empty:
            anomalies.append("Quality critica (<50): " + ", ".join(map(str, lowq["player_id"].tolist())) + ".")
    if "distance_m" in df.columns and df["distance_m"].notna().any():
        mu = float(df["distance_m"].mean())
        sigma = float(df["distance_m"].std()) if df["distance_m"].std() == df["distance_m"].std() else 0.0
        if sigma > 0:
            hi = df[df["distance_m"] > mu + 2*sigma]
            lo = df[df["distance_m"] < mu - 2*sigma]
            if not hi.empty:
                anomalies.append("Distanze molto alte (outlier): " + ", ".join(map(str, hi["player_id"].tolist())) + ".")
            if not lo.empty:
                anomalies.append("Distanze molto basse (outlier): " + ", ".join(map(str, lo["player_id"].tolist())) + ".")
    if not anomalies:
        anomalies.append("Nessuna anomalia evidente con i filtri correnti.")

    next_steps = build_next_training_plan(kpi_df, zone_df)

    html = f"""
<h4>1) Overview Squadra</h4>
<ul>{"".join([f"<li>{x}</li>" for x in overview])}</ul>

<h4>2) Top 3 Performers</h4>
{top3_html}

<h4>3) Area Miglioramento</h4>
<ul>{"".join([f"<li>{x}</li>" for x in improve])}</ul>

<h4>4) Analisi Velocità</h4>
<ul>{"".join([f"<li>{x}</li>" for x in speed])}</ul>

<h4>5) Raccomandazioni Tattiche</h4>
<ul>{"".join([f"<li>{x}</li>" for x in tactics])}</ul>

<h4>6) Anomalie</h4>
<ul>{"".join([f"<li>{x}</li>" for x in anomalies])}</ul>

<h4>7) Prossimo allenamento (team)</h4>
<ul>{"".join([f"<li>{x}</li>" for x in next_steps])}</ul>
""".strip()

    return html, False


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("## CoachTrack")
    st.caption("Tracking + AI + Coaching")

    st.markdown("---")
    st.markdown("### Dati")
    use_sample = st.toggle("Usa sample inclusi", value=True)
    uwb_file = None
    imu_file = None
    if not use_sample:
        uwb_file = st.file_uploader("UWB CSV", type=["csv"])
        imu_file = st.file_uploader("IMU CSV", type=["csv"])

    st.markdown("### Periodo")
    quarter_labels = [
        "Intera Partita",
        "1° Quarto (0-10 min)",
        "2° Quarto (10-20 min)",
        "3° Quarto (20-30 min)",
        "4° Quarto (30-40 min)",
    ]
    quarter = st.selectbox("Seleziona periodo", quarter_labels, index=0)

    st.markdown("### Filtri")
    min_q = st.slider("Quality minima", 0, 100, 50, 1)
    max_speed_clip = st.slider("Clip velocità (km/h)", 10, 40, 30, 1)

    st.markdown("### Features")
    enable_ai = st.toggle("AI Insights", value=True)
    show_comparison = st.toggle("Confronto Heatmap Multi", value=False)
    show_animation = st.toggle("Animazione Heatmap", value=False)

# =========================
# Load and validate
# =========================
if use_sample:
    uwb, imu = load_sample()
else:
    if uwb_file is None:
        st.info("Carica almeno un file UWB per continuare.")
        st.stop()
    uwb, imu = load_uploaded(uwb_file, imu_file)

required = ["timestamp_s", "player_id", "x_m", "y_m", "quality_factor"]
missing = [c for c in required if c not in uwb.columns]
if missing:
    st.error(f"UWB CSV: colonne mancanti: {missing}. Colonne disponibili: {list(uwb.columns)}")
    st.stop()

uwb = uwb.sort_values(["player_id", "timestamp_s"]).copy()

# Quarter filter
if quarter != "Intera Partita":
    quarter_map = {
        "1° Quarto (0-10 min)": (0, 600),
        "2° Quarto (10-20 min)": (600, 1200),
        "3° Quarto (20-30 min)": (1200, 1800),
        "4° Quarto (30-40 min)": (1800, 2400),
    }
    t_min, t_max = quarter_map[quarter]
    uwb = uwb[(uwb["timestamp_s"] >= t_min) & (uwb["timestamp_s"] < t_max)].copy()

# Filters
uwb = uwb[uwb["quality_factor"] >= min_q].copy()

# Derived metrics
uwb["dx"] = uwb.groupby("player_id")["x_m"].diff()
uwb["dy"] = uwb.groupby("player_id")["y_m"].diff()
uwb["dt"] = uwb.groupby("player_id")["timestamp_s"].diff()
uwb["step_m"] = np.sqrt(uwb["dx"]**2 + uwb["dy"]**2)
uwb["speed_ms_calc"] = np.where((uwb["dt"] > 0) & uwb["dt"].notna(), uwb["step_m"] / uwb["dt"], np.nan)
uwb["speed_kmh_calc"] = (uwb["speed_ms_calc"] * 3.6).clip(upper=max_speed_clip)

# Zone
uwb["zone"] = uwb.apply(lambda row: classify_zone(row["x_m"], row["y_m"]), axis=1)

# KPI
kpi = calculate_kpi(uwb.copy())
zone_for_ai = calculate_zone_stats_for_ai(uwb[["player_id", "zone"]].copy())
all_players = sorted(uwb["player_id"].astype(str).unique())

# =========================
# Header + Top KPIs
# =========================
st.markdown(
    f"""
<div class="ct-box">
  <div style="display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap;">
    <div>
      <div style="font-size:1.35rem;font-weight:800;color:#111827;">Dashboard Sessione</div>
      <div style="color: rgba(17,24,39,0.70); margin-top:2px;">
        Periodo: <b>{quarter}</b> · Quality min: <b>{min_q}</b> · Clip speed: <b>{max_speed_clip}</b> km/h
      </div>
    </div>
    <div style="color: rgba(17,24,39,0.55); font-size:0.9rem;">
      CoachTrack
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

team_dist = float(kpi["distance_m"].mean()) if (not kpi.empty and "distance_m" in kpi.columns) else np.nan
team_avgspd = float(kpi["avg_speed_kmh"].mean()) if (not kpi.empty and "avg_speed_kmh" in kpi.columns) else np.nan
team_maxspd = float(kpi["max_speed_kmh"].max()) if (not kpi.empty and "max_speed_kmh" in kpi.columns) else np.nan
team_q = float(kpi["avg_quality"].mean()) if (not kpi.empty and "avg_quality" in kpi.columns) else np.nan

m1, m2, m3, m4 = st.columns(4)
m1.metric("Distanza media (m)", f"{team_dist:.0f}" if np.isfinite(team_dist) else "—")
m2.metric("Velocità media (km/h)", f"{team_avgspd:.1f}" if np.isfinite(team_avgspd) else "—")
m3.metric("Max speed sessione (km/h)", f"{team_maxspd:.1f}" if np.isfinite(team_maxspd) else "—")
m4.metric("Quality medio", f"{team_q:.0f}/100" if np.isfinite(team_q) else "—")

# =========================
# Tabs
# =========================
tab_dash, tab_heat, tab_zone, tab_imu = st.tabs(["Dashboard", "Heatmap", "Zone", "IMU"])

# -------------------------
# Dashboard
# -------------------------
with tab_dash:
    section_title("KPI per giocatore", "Tabella KPI + AI report + piano allenamento individuale")
    st.dataframe(kpi, use_container_width=True, hide_index=True)

    col_ai, col_ind = st.columns([1.2, 0.8])

    with col_ai:
        section_title("AI Insights", "Report strutturato in 7 sezioni")
        if enable_ai:
            with st.spinner("Analisi in corso..."):
                insights_html, is_ai_active = generate_ai_insights(kpi, zone_for_ai)
            st.markdown(f"<div class='ai-insight'>{insights_html}</div>", unsafe_allow_html=True)
            st.caption("AI attiva" if is_ai_active else "Fallback automatico (rule-based).")
        else:
            st.info("AI disattivata (toggle in sidebar).")

    with col_ind:
        section_title("Piano individuale", "Dose-based (minuti/set) basata sui gap vs media")
        if not kpi.empty:
            sel_player = st.selectbox("Giocatore", options=sorted(kpi["player_id"].astype(str).unique()), key="individual_player")
            plan = build_individual_training_plan(sel_player, kpi, zone_for_ai)
            st.markdown(f"<div class='ct-box'><b>Sintesi</b><br>{plan['summary']}</div>", unsafe_allow_html=True)
            st.markdown("<div class='ct-box'><b>Prossimo allenamento</b><br>" + "<br>".join([f"• {x}" for x in plan["plan"]]) + "</div>", unsafe_allow_html=True)
        else:
            st.info("Nessun KPI disponibile (controlla filtri/periodo).")

# -------------------------
# Heatmap
# -------------------------
with tab_heat:
    section_title("Heatmap & Traiettorie", "Vista singola + confronto multi-giocatore")

    player_filter = st.multiselect(
        "Giocatori da visualizzare (vuoto = tutti)",
        options=all_players,
        default=all_players,
        key="player_filter_heat"
    )
    uwb_filtered = uwb[uwb["player_id"].astype(str).isin(player_filter)].copy() if player_filter else uwb.copy()

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("<div class='ct-box'><b>Traiettorie</b></div>", unsafe_allow_html=True)
        show_all = st.checkbox("Mostra tutti", value=True, key="traj_all")
        if not show_all:
            traj_player = st.selectbox("Giocatore singolo", all_players, key="traj_player")

        marker_size = st.slider("Dimensione marker", 2, 10, 4, key="traj_size")
        marker_opacity = st.slider("Opacità marker", 0.1, 1.0, 0.5, 0.1, key="traj_opacity")

        fig_traj = go.Figure()
        plot_data = uwb_filtered if show_all else uwb_filtered[uwb_filtered["player_id"].astype(str) == str(traj_player)]
        if len(plot_data) > 6000:
            plot_data = plot_data.iloc[::max(1, len(plot_data)//6000)]

        for p in plot_data["player_id"].astype(str).unique():
            d = plot_data[plot_data["player_id"].astype(str) == str(p)]
            fig_traj.add_trace(go.Scatter(
                x=d["x_m"], y=d["y_m"],
                mode="markers",
                name=str(p),
                opacity=marker_opacity,
                marker=dict(size=marker_size),
            ))

        fig_traj.update_layout(
            shapes=draw_basketball_court(),
            xaxis=dict(range=[0, 28], showgrid=False, zeroline=False, title=""),
            yaxis=dict(range=[0, 15], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, title=""),
            plot_bgcolor="rgba(34,139,34,0.18)",
            height=520,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    with c2:
        st.markdown("<div class='ct-box'><b>Heatmap densità</b></div>", unsafe_allow_html=True)
        heat_all = st.checkbox("Tutti i giocatori", value=True, key="heat_all")
        if not heat_all:
            heat_player = st.selectbox("Giocatore singolo", all_players, key="heat_player")

        nbins_x = st.slider("Risoluzione X", 20, 100, 60, 5, key="heat_binsx")
        nbins_y = st.slider("Risoluzione Y", 10, 60, 32, 2, key="heat_binsy")

        heat_data = uwb_filtered if heat_all else uwb_filtered[uwb_filtered["player_id"].astype(str) == str(heat_player)]

        fig_heat = go.Figure()
        fig_heat.add_trace(go.Histogram2d(
            x=heat_data["x_m"],
            y=heat_data["y_m"],
            colorscale="Plasma",
            nbinsx=nbins_x,
            nbinsy=nbins_y,
            colorbar=dict(title="Densità"),
        ))
        fig_heat.update_layout(
            shapes=draw_basketball_court(),
            xaxis=dict(range=[0, 28], showgrid=False, zeroline=False, title=""),
            yaxis=dict(range=[0, 15], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, title=""),
            plot_bgcolor="rgba(34,139,34,0.18)",
            height=520,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    if show_comparison and len(all_players) >= 2:
        st.markdown("---")
        section_title("Confronto Heatmap Multi (overlay)", "Fino a 5 giocatori, colori distinti")

        default_players = all_players[:2]

        # Prefer max_selections when available [web:164]; fallback for older versions [web:166]
        try:
            cmp_players = st.multiselect(
                "Seleziona fino a 5 giocatori",
                options=all_players,
                default=default_players,
                max_selections=5,
                key="cmp_players"
            )
        except TypeError:
            cmp_players = st.multiselect(
                "Seleziona fino a 5 giocatori (limite manuale)",
                options=all_players,
                default=default_players,
                key="cmp_players"
            )
            if len(cmp_players) > 5:
                st.warning("Hai selezionato più di 5 giocatori: verranno usati solo i primi 5.")
                cmp_players = cmp_players[:5]

        if cmp_players:
            palette = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6"]  # 🔴 🔵 🟢 🟡 🟣
            color_map = {p: palette[i % len(palette)] for i, p in enumerate(cmp_players)}

            nbinsx_cmp = st.slider("Risoluzione X (overlay)", 20, 120, 60, 5, key="cmp_binsx")
            nbinsy_cmp = st.slider("Risoluzione Y (overlay)", 10, 80, 32, 2, key="cmp_binsy")
            ncontours = st.slider("Numero contorni", 5, 25, 12, 1, key="cmp_contours")
            line_w = st.slider("Spessore linee", 1, 6, 3, 1, key="cmp_linew")
            alpha = st.slider("Opacità linee", 0.1, 1.0, 0.65, 0.05, key="cmp_alpha")

            # Legend
            st.markdown("<div class='ct-box'><b>Legenda</b><br>" +
                        "<br>".join([f"<span style='display:inline-block;width:12px;height:12px;background:{color_map[p]};border-radius:3px;margin-right:8px;'></span>{p}"
                                     for p in cmp_players]) +
                        "</div>", unsafe_allow_html=True)

            fig_cmp = go.Figure()
            for p in cmp_players:
                d = uwb_filtered[uwb_filtered["player_id"].astype(str) == str(p)]
                if d.empty:
                    continue
                # Density contour overlay (lines) [web:169]
                fig_cmp.add_trace(go.Histogram2dContour(
                    x=d["x_m"],
                    y=d["y_m"],
                    nbinsx=nbinsx_cmp,
                    nbinsy=nbinsy_cmp,
                    ncontours=ncontours,
                    contours=dict(coloring="lines"),
                    line=dict(color=color_map[p], width=line_w),
                    opacity=alpha,
                    showscale=False,
                    name=str(p),
                ))

            fig_cmp.update_layout(
                shapes=draw_basketball_court(),
                xaxis=dict(range=[0, 28], showgrid=False, zeroline=False, title=""),
                yaxis=dict(range=[0, 15], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, title=""),
                plot_bgcolor="rgba(34,139,34,0.18)",
                height=560,
                margin=dict(l=10, r=10, t=40, b=10),
                title="Overlay Heatmap (Contour Lines) - Giocatori selezionati",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            st.markdown("<div class='ct-box'><b>Tabella comparativa KPI</b></div>", unsafe_allow_html=True)
            kpi_cmp = kpi[kpi["player_id"].astype(str).isin([str(x) for x in cmp_players])].copy()
            st.dataframe(kpi_cmp, use_container_width=True, hide_index=True)

    if show_animation:
        st.markdown("---")
        section_title("Animazione Heatmap", "Evoluzione temporale (per giocatore)")
        anim_player = st.selectbox("Giocatore", all_players, key="anim_player")
        time_window = st.slider("Finestra (secondi)", 30, 300, 60, 30)

        anim_data = uwb_filtered[uwb_filtered["player_id"].astype(str) == str(anim_player)].copy()
        if not anim_data.empty:
            min_time = anim_data["timestamp_s"].min()
            max_time = anim_data["timestamp_s"].max()
            time_bins = np.arange(min_time, max_time, time_window)
            if len(time_bins) >= 2:
                anim_data["time_bin"] = pd.cut(
                    anim_data["timestamp_s"],
                    bins=time_bins,
                    labels=[f"{int(t)}-{int(t+time_window)}s" for t in time_bins[:-1]],
                )
                fig_anim = px.density_heatmap(
                    anim_data,
                    x="x_m", y="y_m",
                    animation_frame="time_bin",
                    range_x=[0, 28], range_y=[0, 15],
                    nbinsx=40, nbinsy=20,
                    color_continuous_scale="Plasma",
                    title=f"Evoluzione Posizionale - {anim_player}",
                )
                fig_anim.update_layout(height=560)
                st.plotly_chart(fig_anim, use_container_width=True)
            else:
                st.info("Intervallo tempo troppo corto per creare i bin.")
        else:
            st.info("Nessun dato disponibile per questo giocatore.")

# -------------------------
# Zone
# -------------------------
with tab_zone:
    section_title("Zone analysis", "Distribuzione Paint / Mid-Range / 3-Point")
    zone_stats = calculate_zone_stats(uwb[["player_id", "zone"]].copy())

    if not zone_stats.empty:
        zone_player = st.selectbox("Giocatore", sorted(zone_stats["player_id"].astype(str).unique()), key="zone_player_select")
        player_zones = zone_stats[zone_stats["player_id"].astype(str) == str(zone_player)].copy()

        z1, z2 = st.columns([1, 1])
        with z1:
            st.markdown("<div class='ct-box'><b>Tabella zone</b></div>", unsafe_allow_html=True)
            st.dataframe(player_zones[["zone", "count", "percentage"]], use_container_width=True, hide_index=True)
        with z2:
            st.markdown("<div class='ct-box'><b>Distribuzione %</b></div>", unsafe_allow_html=True)
            fig_pie = px.pie(
                player_zones,
                values="percentage",
                names="zone",
                title=f"Zone distribution - {zone_player}",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Zone stats non disponibili.")

# -------------------------
# IMU
# -------------------------
with tab_imu:
    section_title("IMU", "Accel Z + salti (se presenti)")

    if imu is None:
        st.info("Nessun file IMU caricato (ok per test UWB-only).")
    else:
        if "timestamp_s" not in imu.columns or "accel_z_ms2" not in imu.columns:
            st.warning(f"IMU CSV: colonne richieste mancanti. Colonne disponibili: {list(imu.columns)}")
        else:
            if quarter != "Intera Partita":
                imu = imu[(imu["timestamp_s"] >= t_min) & (imu["timestamp_s"] < t_max)].copy()

            jumps = int((imu.get("jump_detected", pd.Series([0] * len(imu))) == 1).sum()) if "jump_detected" in imu.columns else 0
            st.metric("Salti rilevati (periodo)", jumps)

            imu_players = sorted(imu["player_id"].astype(str).unique())
            psel = st.selectbox("Giocatore IMU", imu_players, key="imu_player")

            imu_p = imu[imu["player_id"].astype(str) == str(psel)].sort_values("timestamp_s")
            fig4 = px.line(
                imu_p, x="timestamp_s", y="accel_z_ms2",
                title=f"Accel Z (m/s²) - {psel} - {quarter}",
                labels={"timestamp_s": "Tempo (s)", "accel_z_ms2": "Accel Z (m/s²)"},
            )
            if "jump_detected" in imu_p.columns:
                jump_points = imu_p[imu_p["jump_detected"] == 1]
                if not jump_points.empty:
                    fig4.add_scatter(
                        x=jump_points["timestamp_s"],
                        y=jump_points["accel_z_ms2"],
                        mode="markers",
                        marker=dict(color="#ef4444", size=10, symbol="star"),
                        name="Salti",
                    )
            st.plotly_chart(fig4, use_container_width=True)

with st.expander("⚙️ Setup AI (Groq)"):
    st.markdown("""
1) Crea `.streamlit/secrets.toml`  
2) Inserisci: `GROQ_API_KEY = "..."`  
3) Redeploy / riavvio
""")
