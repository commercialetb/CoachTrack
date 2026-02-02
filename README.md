# 🏀 Basketball Tracking MVP (Test Realistico)

Questa app Streamlit simula condizioni **realistiche** per UWB indoor basketball tracking:

## Caratteristiche Dataset

### UWB (Ultra-Wideband)
- **6 giocatori** tracciati per 10 minuti (600s)
- **Frequenza:** 10 Hz
- **Dropout:** ~6% packet loss (simula condizioni reali indoor)
- **Outlier NLOS:** ~0.8% punti con bias positivo (fino a 4m)
- **Quality factor:** 0-100 (simula MDEK1001/DWM1001)

### IMU (Inertial Measurement Unit)
- **3 giocatori** con IMU
- **Frequenza:** 100 Hz
- **Rumore:** Bias drift su accelerometro e giroscopio
- **Dropout:** ~1.5% packet loss
- **Jump detection:** ~320 salti rilevati (virtuali)

## Come Usare

### Opzione 1: Test Locale

```bash
# Installa dipendenze
pip install -r requirements.txt

# Avvia app
streamlit run app.py
```

### Opzione 2: Deploy su Streamlit Cloud

1. Crea un repo GitHub
2. Carica questi file:
   - `app.py`
   - `requirements.txt`
   - `data/virtual_uwb_realistic.csv`
   - `data/virtual_imu_realistic.csv`
3. Vai su [Streamlit Community Cloud](https://streamlit.io/cloud)
4. Connetti il tuo repo
5. Imposta **Main file path:** `app.py`
6. Deploy! ✅

## Struttura File

```
.
├── app.py                          # Streamlit dashboard
├── requirements.txt                # Dipendenze Python
├── README.md                       # Questo file
├── data/
│   ├── virtual_uwb_realistic.csv  # Dataset UWB (6 giocatori, 10 min)
│   └── virtual_imu_realistic.csv  # Dataset IMU (3 giocatori, 10 min)
└── .gitignore
```

## Colonne Dataset

### UWB CSV
| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| `timestamp_s` | float | Tempo in secondi |
| `player_id` | str | ID giocatore (P1_Guard, P2_Forward, etc.) |
| `x_m` | float | Posizione X in metri (0-28m campo) |
| `y_m` | float | Posizione Y in metri (0-15m campo) |
| `speed_kmh` | float | Velocità in km/h |
| `accel_ms2` | float | Accelerazione in m/s² |
| `quality_factor` | int | Qualità segnale 0-100 |
| `intensity` | str | Categoria attività (stationary, walking, jogging, running, sprinting) |

### IMU CSV
| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| `timestamp_s` | float | Tempo in secondi |
| `player_id` | str | ID giocatore |
| `accel_x_ms2` | float | Accelerazione asse X |
| `accel_y_ms2` | float | Accelerazione asse Y |
| `accel_z_ms2` | float | Accelerazione asse Z |
| `gyro_x_rads` | float | Giroscopio asse X (rad/s) |
| `gyro_y_rads` | float | Giroscopio asse Y (rad/s) |
| `gyro_z_rads` | float | Giroscopio asse Z (rad/s) |
| `jump_detected` | int | 0/1 salto rilevato |
| `jump_height_cm` | float | Altezza salto stimata (cm) |

## Filtri Dashboard

### Quality Factor Filter
- Default: **50** (scarta misure bassa qualità)
- Range: 0-100
- Usa per: Rimuovere outlier NLOS e misure inaffidabili

### Speed Clip Filter
- Default: **30 km/h** (taglia spike velocità)
- Range: 10-40 km/h
- Usa per: Mitigare outlier NLOS che causano salti di posizione

## Problemi Realistici Simulati

### 1. Dropout (Packet Loss)
- **Simulato:** ~6% punti mancanti
- **Reale:** 5-8% tipico in indoor sports
- **Causa:** Interferenza, occlusion, limiti hardware
- **Soluzione:** Interpolazione, filtri robusti

### 2. Outlier NLOS
- **Simulato:** ~0.8% punti con bias +0.5 a +4m
- **Reale:** NLOS (Non-Line-Of-Sight) causa range errors
- **Causa:** Multipath, riflessioni, ostacoli
- **Soluzione:** Quality factor filter, speed clip, Kalman filter

### 3. Noise & Bias (IMU)
- **Simulato:** Drift accelerometro/giroscopio
- **Reale:** Sensori MEMS hanno bias drift nel tempo
- **Causa:** Temperatura, vibrazione, aging
- **Soluzione:** Sensor fusion (UWB + IMU), calibrazione

## KPI Visualizzati

- **Points:** Numero campioni dopo filtri
- **Distance (m):** Distanza totale percorsa
- **Avg Speed (km/h):** Velocità media
- **Max Speed (km/h):** Velocità massima (picco)
- **Avg Quality:** Quality factor medio (0-100)

## Grafici

1. **Traiettorie:** Scatter plot posizioni (noterai "buchi" da dropout)
2. **Heatmap:** Densità posizioni (zone più frequentate)
3. **Velocità nel tempo:** Line chart (noterai spike da outlier NLOS)
4. **IMU Accel Z:** Picchi corrispondono a salti

## Prossimi Step

Per analisi più avanzate:
1. **Kalman Filter:** Smooth traiettorie e rimuovi outlier
2. **Sensor Fusion:** Combina UWB + IMU per accuracy
3. **Machine Learning:** Classifica azioni (sprint, jump, change direction)
4. **Time Sync:** Gestisci desincronizzazione UWB vs IMU
5. **Multi-anchor optimization:** Trilaterazione robusta

## Riferimenti Tecnici

- Quality factor MDEK1001: [Qorvo DWM1001 Docs](https://www.qorvo.com/products/p/DWM1001)
- UWB ranging accuracy: IEEE 802.15.4a standard
- NLOS error characterization: Sports tech research papers

---

**Nota:** Questo è un MVP per test. Per produzione serve:
- Hardware calibrato (anchor positioning)
- Time synchronization precisa
- Filtri avanzati (Kalman, particle filter)
- Validation con ground truth (Vicon, OptiTrack)

**License:** MIT  
**Creato:** Febbraio 2026