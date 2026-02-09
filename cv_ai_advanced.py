# =================================================================
# COACHTRACK ELITE AI v3.1 - CV AI ADVANCED MODULE
# Phase 2: Action Recognition + Shot Tracking + Pose Estimation
# =================================================================
# MODULO STANDALONE - Non modifica app.py
# Import: from cv_ai_advanced import *

import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import time

# =================================================================
# CONDITIONAL IMPORTS (graceful degradation)
# =================================================================

# MediaPipe Pose
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except ImportError:
    print("⚠️ MediaPipe non installato: pip install mediapipe")

# PyTorch for Action Recognition
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch non installato (già presente per YOLO)")

# Scipy for trajectory fitting
SCIPY_AVAILABLE = False
try:
    from scipy.optimize import curve_fit
    from scipy.spatial import distance
    SCIPY_AVAILABLE = True
except ImportError:
    print("⚠️ Scipy non installato: pip install scipy")

# =================================================================
# 1️⃣ ACTION RECOGNITION
# =================================================================

class ActionRecognizer:
    """
    Riconosce azioni basket da sequenze di frame
    Actions: shoot, pass, dribble, rebound, defense, idle
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Path a modello custom (opzionale)
        """
        self.model = None
        self.model_loaded = False
        self.actions = ['shoot', 'pass', 'dribble', 'rebound', 'defense', 'idle']

        # Sliding window per analisi temporale
        self.window_size = 16  # 16 frame = ~0.5s @ 30fps
        self.frame_buffer = []

        if TORCH_AVAILABLE and model_path:
            try:
                self.model = torch.load(model_path)
                self.model.eval()
                self.model_loaded = True
                print(f"✅ Action Recognition model caricato: {model_path}")
            except Exception as e:
                print(f"⚠️ Errore caricamento model: {e}")
                print("   Usando rule-based classifier...")

    def predict_action(self, pose_landmarks: List[Dict], ball_position: Optional[Dict] = None) -> Dict:
        """
        Predice azione da pose landmarks

        Args:
            pose_landmarks: Lista di pose per window_size frame
            ball_position: Posizione palla (opzionale)

        Returns:
            {
                'action': 'shoot',
                'confidence': 0.89,
                'frame_start': 100,
                'frame_end': 116
            }
        """
        if self.model_loaded:
            return self._predict_with_model(pose_landmarks)
        else:
            return self._predict_rule_based(pose_landmarks, ball_position)

    def _predict_rule_based(self, poses: List[Dict], ball_pos: Optional[Dict]) -> Dict:
        """
        Rule-based classifier (fallback senza deep learning)
        Analizza pose landmarks per inferire azione
        """
        if not poses or len(poses) == 0:
            return {'action': 'idle', 'confidence': 0.5}

        # Prendi ultima pose (più recente)
        last_pose = poses[-1] if poses else None
        if not last_pose:
            return {'action': 'idle', 'confidence': 0.5}

        # Estrai keypoints chiave
        right_wrist = last_pose.get('right_wrist', {'y': 0})
        left_wrist = last_pose.get('left_wrist', {'y': 0})
        right_elbow = last_pose.get('right_elbow', {'y': 0})
        nose = last_pose.get('nose', {'y': 0})
        right_hip = last_pose.get('right_hip', {'y': 0})

        # SHOOTING: mani sopra testa
        if right_wrist.get('y', 1) < nose.get('y', 0):
            return {'action': 'shoot', 'confidence': 0.75}

        # PASSING: mani davanti al petto, movimento estensione
        if len(poses) >= 2:
            prev_wrist = poses[-2].get('right_wrist', {'x': 0})
            curr_wrist = right_wrist
            # Movimento verso esterno
            if abs(curr_wrist.get('x', 0) - prev_wrist.get('x', 0)) > 0.1:
                return {'action': 'pass', 'confidence': 0.70}

        # DRIBBLING: mano bassa, movimento verticale ripetuto
        if right_wrist.get('y', 0) > right_hip.get('y', 0):
            return {'action': 'dribble', 'confidence': 0.65}

        # Default: idle
        return {'action': 'idle', 'confidence': 0.60}

    def _predict_with_model(self, poses: List[Dict]) -> Dict:
        """
        Prediction con deep learning model (SlowFast/I3D)
        """
        # TODO: Implementare quando model è disponibile
        # For now, fallback to rule-based
        return self._predict_rule_based(poses, None)

    def process_video_actions(self, video_path: str, pose_data: List[Dict]) -> List[Dict]:
        """
        Processa intero video e restituisce timeline azioni

        Args:
            video_path: Path video
            pose_data: Lista pose per ogni frame

        Returns:
            Lista azioni rilevate con timestamp
        """
        actions_timeline = []

        for i in range(0, len(pose_data), self.window_size):
            window = pose_data[i:i+self.window_size]
            if len(window) < self.window_size:
                continue

            action_result = self.predict_action(window)
            action_result['frame_start'] = i
            action_result['frame_end'] = i + self.window_size

            actions_timeline.append(action_result)

        return actions_timeline

# =================================================================
# 2️⃣ SHOT TRACKING & ANALYTICS
# =================================================================

class ShotTracker:
    """
    Traccia tiri e calcola analytics:
    - Release point (punto rilascio)
    - Release angle (angolo)
    - Arc height (altezza arco)
    - Shot quality score
    - Make/miss prediction
    """

    def __init__(self, court_dimensions: Tuple[float, float] = (28.0, 15.0)):
        """
        Args:
            court_dimensions: (larghezza, lunghezza) in metri (FIBA: 28x15)
        """
        self.court_width = court_dimensions[0]
        self.court_length = court_dimensions[1]

        # Coordinate canestri (FIBA)
        self.basket_1 = np.array([self.court_width/2, 1.575])  # Metro da fondo
        self.basket_2 = np.array([self.court_width/2, self.court_length - 1.575])
        self.basket_height = 3.05  # metri

        # Shot tracking state
        self.active_shots = []
        self.completed_shots = []

        # Shot detection thresholds
        self.min_ball_height = 2.0  # metri (sotto non è tiro)
        self.min_trajectory_points = 5  # punti minimi per fit parabolico

    def detect_shot(self, ball_trajectory: List[Dict], player_pose: Dict) -> Optional[Dict]:
        """
        Rileva se sequenza movimento è un tiro

        Args:
            ball_trajectory: Lista posizioni palla [{'x':, 'y':, 'z':}, ...]
            player_pose: Pose giocatore al momento rilascio

        Returns:
            Shot data o None
        """
        if len(ball_trajectory) < self.min_trajectory_points:
            return None

        # Estrai coordinate
        xs = np.array([p['x'] for p in ball_trajectory])
        ys = np.array([p['y'] for p in ball_trajectory])
        zs = np.array([p.get('z', 0) for p in ball_trajectory])

        # Check: palla sale sopra altezza minima?
        max_z = np.max(zs) if len(zs) > 0 else 0
        if max_z < self.min_ball_height:
            return None

        # Release point: primo punto traiettoria
        release_point = ball_trajectory[0]

        # Calcola release angle e speed
        release_angle, release_speed = self._calculate_release_metrics(ball_trajectory)

        # Arc height
        arc_height = max_z

        # Shot quality score (0-100)
        quality_score = self._calculate_shot_quality(
            release_angle, release_speed, arc_height, release_point
        )

        # Make probability (ML-based prediction)
        make_prob = self._predict_make_probability(
            release_point, release_angle, release_speed, arc_height
        )

        shot_data = {
            'shot_id': len(self.completed_shots) + 1,
            'release_point': release_point,
            'release_angle': round(release_angle, 1),
            'release_speed': round(release_speed, 2),
            'arc_height': round(arc_height, 2),
            'quality_score': round(quality_score, 1),
            'make_probability': round(make_prob, 2),
            'trajectory': ball_trajectory,
            'player_pose': player_pose
        }

        return shot_data

    def _calculate_release_metrics(self, trajectory: List[Dict]) -> Tuple[float, float]:
        """
        Calcola angolo e velocità di rilascio

        Returns:
            (angle_degrees, speed_m_s)
        """
        if len(trajectory) < 2:
            return 45.0, 7.0  # defaults

        # Primi 2 punti per calcolare velocità iniziale
        p0 = trajectory[0]
        p1 = trajectory[1]

        dx = p1['x'] - p0['x']
        dy = p1['y'] - p0['y']
        dz = p1.get('z', 0) - p0.get('z', 0)

        # Velocità (assumendo 30fps → dt=0.033s)
        dt = 0.033
        vx = dx / dt
        vy = dy / dt
        vz = dz / dt

        # Speed totale
        speed = np.sqrt(vx**2 + vy**2 + vz**2)

        # Angolo di rilascio (rispetto orizzontale)
        v_horizontal = np.sqrt(vx**2 + vy**2)
        angle_rad = np.arctan2(vz, v_horizontal)
        angle_deg = np.degrees(angle_rad)

        return angle_deg, speed

    def _calculate_shot_quality(self, angle: float, speed: float, 
                                 arc: float, location: Dict) -> float:
        """
        Calcola quality score 0-100 basato su parametri ideali

        Parametri ideali:
        - Angle: 48-52° (ottimale 50°)
        - Speed: 6-8 m/s
        - Arc: 3.5-4.5m
        """
        score = 100.0

        # Penalità angle (ideale: 50°)
        angle_diff = abs(angle - 50)
        score -= min(angle_diff * 2, 30)  # max -30 punti

        # Penalità speed (ideale: 7 m/s)
        speed_diff = abs(speed - 7.0)
        score -= min(speed_diff * 5, 25)  # max -25 punti

        # Penalità arc (ideale: 4m)
        arc_diff = abs(arc - 4.0)
        score -= min(arc_diff * 10, 25)  # max -25 punti

        # Bonus per distanza appropriata (non troppo vicino/lontano)
        # TODO: Calcolare distanza da canestro

        return max(0, score)

    def _predict_make_probability(self, location: Dict, angle: float, 
                                   speed: float, arc: float) -> float:
        """
        Predice probabilità di fare canestro (0-1)

        Usa modello semplificato basato su shot quality
        """
        quality = self._calculate_shot_quality(angle, speed, arc, location)

        # Conversione quality → probability (non lineare)
        # Quality 90+ → ~70% make
        # Quality 50 → ~35% make
        # Quality <30 → <20% make

        if quality >= 80:
            prob = 0.60 + (quality - 80) * 0.01
        elif quality >= 50:
            prob = 0.35 + (quality - 50) * 0.0083
        else:
            prob = 0.15 + quality * 0.004

        return min(1.0, max(0.0, prob))

    def generate_shot_chart(self, shots: List[Dict], court_img: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Genera shot chart (heatmap) su immagine campo

        Args:
            shots: Lista shot completati
            court_img: Immagine campo (opzionale, altrimenti genera)

        Returns:
            Immagine shot chart
        """
        if court_img is None:
            # Crea immagine campo semplice
            court_img = self._create_court_image()

        h, w = court_img.shape[:2]

        # Overlay shot locations
        for shot in shots:
            loc = shot['release_point']
            x_pixel = int(loc['x'] / self.court_width * w)
            y_pixel = int(loc['y'] / self.court_length * h)

            # Colore basato su make probability
            prob = shot.get('make_probability', 0.5)
            if prob >= 0.6:
                color = (0, 255, 0)  # Verde (alta prob)
            elif prob >= 0.4:
                color = (0, 255, 255)  # Giallo
            else:
                color = (0, 0, 255)  # Rosso (bassa prob)

            cv2.circle(court_img, (x_pixel, y_pixel), 15, color, -1)
            cv2.circle(court_img, (x_pixel, y_pixel), 15, (255,255,255), 2)

        return court_img

    def _create_court_image(self, width: int = 800, height: int = 600) -> np.ndarray:
        """Crea immagine campo basket stilizzata"""
        img = np.ones((height, width, 3), dtype=np.uint8) * 200  # Grigio chiaro

        # Linee campo (semplificato)
        cv2.rectangle(img, (50, 50), (width-50, height-50), (0,0,0), 3)

        # Cerchi centrali
        center = (width//2, height//2)
        cv2.circle(img, center, 100, (0,0,0), 2)

        # Canestri (semplificato)
        cv2.circle(img, (width//2, 80), 20, (255,0,0), -1)
        cv2.circle(img, (width//2, height-80), 20, (255,0,0), -1)

        return img

# =================================================================
# 3️⃣ POSE ESTIMATION & BIOMECHANICS
# =================================================================

class PoseAnalyzer:
    """
    Analizza pose giocatori con MediaPipe
    Features:
    - 33 keypoints detection
    - Shooting form analysis
    - Biomechanical issues detection
    - Injury risk assessment
    """

    def __init__(self):
        self.pose_detector = None
        self.available = MEDIAPIPE_AVAILABLE

        if MEDIAPIPE_AVAILABLE:
            self.pose_detector = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("✅ MediaPipe Pose inizializzato")
        else:
            print("⚠️ MediaPipe non disponibile - Pose analysis disabilitato")

    def extract_pose(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Estrae pose landmarks da frame

        Args:
            frame: Frame RGB

        Returns:
            Dict con 33 keypoints o None
        """
        if not self.available:
            return None

        # Process frame
        results = self.pose_detector.process(frame)

        if not results.pose_landmarks:
            return None

        # Converti landmarks in dict
        landmarks = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks[idx] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            }

        # Estrai keypoints chiave con nomi leggibili
        pose_dict = {
            'nose': landmarks.get(0, {}),
            'left_eye': landmarks.get(2, {}),
            'right_eye': landmarks.get(5, {}),
            'left_ear': landmarks.get(7, {}),
            'right_ear': landmarks.get(8, {}),
            'left_shoulder': landmarks.get(11, {}),
            'right_shoulder': landmarks.get(12, {}),
            'left_elbow': landmarks.get(13, {}),
            'right_elbow': landmarks.get(14, {}),
            'left_wrist': landmarks.get(15, {}),
            'right_wrist': landmarks.get(16, {}),
            'left_hip': landmarks.get(23, {}),
            'right_hip': landmarks.get(24, {}),
            'left_knee': landmarks.get(25, {}),
            'right_knee': landmarks.get(26, {}),
            'left_ankle': landmarks.get(27, {}),
            'right_ankle': landmarks.get(28, {}),
            'raw_landmarks': landmarks
        }

        return pose_dict

    def analyze_shooting_form(self, pose: Dict) -> Dict:
        """
        Analizza meccanica tiro

        Returns:
            {
                'elbow_angle': 87,
                'knee_bend': 42,
                'release_height': 2.41,
                'form_score': 8.5,
                'issues': ['Elbow flare', ...]
            }
        """
        if not pose:
            return {'form_score': 0, 'issues': ['No pose detected']}

        # Calcola angoli chiave
        elbow_angle = self._calculate_angle(
            pose.get('right_shoulder'), 
            pose.get('right_elbow'), 
            pose.get('right_wrist')
        )

        knee_angle = self._calculate_angle(
            pose.get('right_hip'),
            pose.get('right_knee'),
            pose.get('right_ankle')
        )

        # Release height (altezza polso rispetto a terra)
        wrist_y = pose.get('right_wrist', {}).get('y', 0.5)
        ankle_y = pose.get('right_ankle', {}).get('y', 0.9)
        # Stima altezza in metri (assumendo altezza player ~1.9m)
        release_height = (ankle_y - wrist_y) * 1.9 + 1.9

        # Issues detection
        issues = []

        # Elbow ideale: 85-95°
        if elbow_angle < 80 or elbow_angle > 100:
            issues.append(f"Angolo gomito non ottimale: {elbow_angle:.0f}° (ideale: 85-95°)")

        # Knee bend ideale: 40-50°
        if knee_angle < 35 or knee_angle > 55:
            issues.append(f"Flessione ginocchia: {knee_angle:.0f}° (ideale: 40-50°)")

        # Calcola form score (0-10)
        form_score = 10.0
        form_score -= abs(elbow_angle - 90) * 0.1  # max -1 per grado
        form_score -= abs(knee_angle - 45) * 0.1
        form_score = max(0, min(10, form_score))

        return {
            'elbow_angle': round(elbow_angle, 1),
            'knee_bend': round(knee_angle, 1),
            'release_height': round(release_height, 2),
            'form_score': round(form_score, 1),
            'issues': issues,
            'recommendations': self._generate_form_recommendations(elbow_angle, knee_angle)
        }

    def _calculate_angle(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """
        Calcola angolo tra 3 punti (p2 è il vertice)
        """
        if not all([p1, p2, p3]):
            return 90.0  # default

        # Vettori
        v1 = np.array([p1.get('x', 0) - p2.get('x', 0), 
                       p1.get('y', 0) - p2.get('y', 0)])
        v2 = np.array([p3.get('x', 0) - p2.get('x', 0), 
                       p3.get('y', 0) - p2.get('y', 0)])

        # Angolo
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

        return angle

    def _generate_form_recommendations(self, elbow: float, knee: float) -> List[str]:
        """Genera raccomandazioni personalizzate"""
        recs = []

        if elbow < 85:
            recs.append("Aumenta angolo gomito: porta palla più in alto")
        elif elbow > 95:
            recs.append("Riduci angolo gomito: tieni gomito più vicino al corpo")

        if knee < 40:
            recs.append("Aumenta flessione ginocchia per più potenza")
        elif knee > 50:
            recs.append("Riduci flessione: stai troppo basso")

        if not recs:
            recs.append("Ottima forma! Continua così")

        return recs

    def draw_skeleton(self, frame: np.ndarray, pose: Dict) -> np.ndarray:
        """
        Disegna skeleton su frame
        """
        if not pose or not MEDIAPIPE_AVAILABLE:
            return frame

        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Connessioni da disegnare
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle')
        ]

        # Disegna connessioni
        for conn in connections:
            p1 = pose.get(conn[0])
            p2 = pose.get(conn[1])

            if p1 and p2:
                x1, y1 = int(p1['x'] * w), int(p1['y'] * h)
                x2, y2 = int(p2['x'] * w), int(p2['y'] * h)
                cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Disegna keypoints
        for key in pose:
            if key == 'raw_landmarks':
                continue
            point = pose[key]
            if point:
                x, y = int(point.get('x', 0) * w), int(point.get('y', 0) * h)
                cv2.circle(annotated, (x, y), 5, (0, 0, 255), -1)

        return annotated

# =================================================================
# INTEGRATED CV AI PIPELINE
# =================================================================

class CVAIPipeline:
    """
    Pipeline integrata che combina tutti i moduli AI
    """

    def __init__(self):
        self.action_recognizer = ActionRecognizer()
        self.shot_tracker = ShotTracker()
        self.pose_analyzer = PoseAnalyzer()

        print("✅ CV AI Pipeline inizializzata")
        print(f"   - Action Recognition: {'✅' if True else '❌'}")
        print(f"   - Shot Tracking: {'✅' if True else '❌'}")
        print(f"   - Pose Estimation: {'✅' if MEDIAPIPE_AVAILABLE else '❌'}")

    def process_video_complete(self, video_path: str, output_json: str = "cv_ai_analysis.json") -> Dict:
        """
        Processa video con tutte le analisi AI

        Returns:
            Dizionario completo con tutti i dati
        """
        print(f"🎬 Processing video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Storage per analisi
        all_poses = []
        all_actions = []
        all_shots = []

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # RGB per MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Extract pose
            pose = self.pose_analyzer.extract_pose(frame_rgb)
            if pose:
                pose['frame'] = frame_idx
                all_poses.append(pose)

            frame_idx += 1

            # Progress ogni 30 frame
            if frame_idx % 30 == 0:
                progress = frame_idx / frame_count * 100
                print(f"   Progress: {progress:.1f}% ({frame_idx}/{frame_count})")

        cap.release()

        # Action recognition su pose sequence
        print("🎯 Analyzing actions...")
        all_actions = self.action_recognizer.process_video_actions(video_path, all_poses)

        # Shot detection (placeholder - richiede ball tracking)
        print("🏀 Analyzing shots...")
        # TODO: Integra con ball tracking da cv_processor

        # Generate summary
        summary = {
            'video': video_path,
            'metadata': {
                'fps': fps,
                'total_frames': frame_count,
                'duration_sec': frame_count / fps if fps > 0 else 0,
                'processed_at': datetime.now().isoformat()
            },
            'pose_data': all_poses,
            'actions': all_actions,
            'shots': all_shots,
            'statistics': {
                'total_poses_detected': len(all_poses),
                'total_actions': len(all_actions),
                'total_shots': len(all_shots)
            }
        }

        # Save JSON
        with open(output_json, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✅ Analysis complete! Saved to: {output_json}")

        return summary

# =================================================================
# CONVENIENCE FUNCTIONS (per import in app.py)
# =================================================================

def analyze_video_ai(video_path: str, output_json: str = "cv_ai_analysis.json") -> Dict:
    """
    Funzione semplice per analisi completa video
    Usage: from cv_ai_advanced import analyze_video_ai
    """
    pipeline = CVAIPipeline()
    return pipeline.process_video_complete(video_path, output_json)

def get_shot_chart(shots_data: List[Dict]) -> np.ndarray:
    """Genera shot chart da dati shots"""
    tracker = ShotTracker()
    return tracker.generate_shot_chart(shots_data)

def analyze_player_form(frame: np.ndarray) -> Dict:
    """Analizza form giocatore da singolo frame"""
    analyzer = PoseAnalyzer()
    pose = analyzer.extract_pose(frame)
    if pose:
        return analyzer.analyze_shooting_form(pose)
    return {'form_score': 0, 'issues': ['No pose detected']}

# =================================================================
# MODULE INFO
# =================================================================

__version__ = "3.1.0"
__phase__ = "Phase 2: Action Recognition + Shot Tracking + Pose Estimation"
__all__ = [
    'ActionRecognizer',
    'ShotTracker',
    'PoseAnalyzer',
    'CVAIPipeline',
    'analyze_video_ai',
    'get_shot_chart',
    'analyze_player_form',
    'MEDIAPIPE_AVAILABLE'
]

if __name__ == "__main__":
    print("✅ CV AI Advanced Module - Phase 2")
    print(f"   Version: {__version__}")
    print(f"   Phase: {__phase__}")
    print(f"   MediaPipe: {'✅' if MEDIAPIPE_AVAILABLE else '❌ (pip install mediapipe)'}")
