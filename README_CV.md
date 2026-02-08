# CoachTrack Elite AI v3.0 - Computer Vision Module

## 📋 Overview

Sistema di Computer Vision per tracking automatico giocatori basket da Action Cam o PTZ camera.

## 🎯 Features

- ✅ **Player Detection**: YOLO-based detection di tutti i giocatori
- ✅ **Multi-Object Tracking**: Assegnazione ID persistenti con Simple Tracker
- ✅ **Court Calibration**: Trasformazione coordinate immagine → campo reale
- ✅ **Ball Detection**: Rilevamento palla basket
- ✅ **Real-time Processing**: Processing live da camera o offline da video
- ✅ **JSON Export**: Dati strutturati per integrazione con CoachTrack Elite
- ✅ **Streamlit Integration**: UI completa per gestione CV

## 🛠️ Hardware Supportato

### Action Cam (Budget: $50-300)
- GoPro Hero 9/10/11/12
- DJI Osmo Action 3/4
- AKASO V50 Elite
- Apeman A100
- Qualsiasi action cam con USB webcam mode o WiFi streaming

### PTZ Camera (Mid-range: $300-800)
- FEELWORLD 4K12X
- Tenveo VLoop 4K
- SMTAV 20X AI
- Moertek MC8420B

### Compute
- **Minimum**: Raspberry Pi 4 8GB (15-20 fps con YOLOv8-nano)
- **Recommended**: Intel NUC i5/i7 (30-60 fps con YOLOv8-medium)
- **Optimal**: Mini PC con GPU dedicata (60+ fps con YOLOv8-large)

## 📦 Installation

```bash
# Clone repo
cd coachtrack-elite

# Install CV dependencies
pip install -r requirements_cv.txt

# Download YOLO model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Test installation
python cv_processor.py --source 0 --duration 10
```

## 🚀 Quick Start

### 1. Calibrazione Campo

```bash
python cv_processor.py --source 0 --calibrate
```

Nella finestra che si apre:
1. Clicca angolo **Basso-Sinistra** del campo
2. Clicca angolo **Basso-Destra**
3. Clicca angolo **Alto-Destra**
4. Clicca angolo **Alto-Sinistra**
5. Premi 'q'

Salva automaticamente `camera_calibration.json`.

### 2. Live Tracking

```bash
# From USB camera
python cv_processor.py --source 0 --output live_data.json

# From WiFi camera (RTSP)
python cv_processor.py --source rtsp://192.168.1.100/stream --output live_data.json

# Durata limitata (60 secondi)
python cv_processor.py --source 0 --duration 60 --output test.json
```

### 3. Process Video File

```bash
python cv_processor.py --source game_video.mp4 --output game_data.json
```

### 4. Integrazione Streamlit

Nell'app principale `app.py`:

```python
from streamlit_cv_integration import add_computer_vision_tab

# Aggiungi tab CV
tab1, tab2, ..., tab_cv = st.tabs([..., "🎥 Computer Vision"])

with tab_cv:
    add_computer_vision_tab()
```

## 📊 Output Format

File JSON generato:

```json
{
  "metadata": {
    "total_frames": 1800,
    "fps": 30.2,
    "duration_sec": 59.6,
    "calibrated": true,
    "generated_at": "2026-02-08T12:00:00"
  },
  "frames": [
    {
      "timestamp": 1707393600.123,
      "frame_number": 0,
      "processing_time_ms": 33.2,
      "fps": 30.1,
      "players": [
        {
          "player_id": 1,
          "x": 14.2,
          "y": 7.8,
          "conf": 0.92
        },
        ...
      ],
      "ball": {
        "center": [1920, 1080],
        "radius": 20,
        "conf": 0.75
      }
    },
    ...
  ]
}
```

## 🎛️ Configuration

### YOLOv8 Models

Scegli in base a hardware disponibile:

| Model | Size | Speed | mAP | Use Case |
|-------|------|-------|-----|----------|
| yolov8n.pt | 6MB | 60+ fps | 37.3 | Raspberry Pi, real-time |
| yolov8s.pt | 22MB | 40-50 fps | 44.9 | Mini PC, balanced |
| yolov8m.pt | 52MB | 25-35 fps | 50.2 | GPU, high accuracy |
| yolov8l.pt | 87MB | 15-25 fps | 52.9 | Offline processing |

### Tracker Parameters

In `cv_tracking.py`:

```python
tracker = SimpleTracker(
    max_age=30,        # Frame max senza detection
    min_hits=3,        # Hit minimi per confermare track
    iou_threshold=0.3  # IoU minimo per matching
)
```

## 🔧 Troubleshooting

### Camera non si connette

```bash
# List available cameras
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"

# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.read()[0])"
```

### FPS basso

1. Usa modello più leggero (`yolov8n.pt`)
2. Riduci risoluzione input (1080p invece di 4K)
3. Disabilita visualizzazione (`--no-viz`)
4. Usa GPU se disponibile

### Calibrazione imprecisa

- Assicurati che tutto il campo sia visibile
- Clicca esattamente sugli angoli delle linee
- Usa frame con campo ben illuminato
- Ri-calibra se cambi posizione camera

## 🎯 Roadmap

- [ ] Ball detection con YOLO custom trained
- [ ] Advanced tracking (ByteTrack, BoT-SORT)
- [ ] Team assignment automatico (jersey color detection)
- [ ] Shot detection e classification
- [ ] Player pose estimation (biomeccanica)
- [ ] Multi-camera fusion
- [ ] Real-time streaming a CoachTrack cloud

## 📝 License

Part of CoachTrack Elite AI v3.0
© 2026
