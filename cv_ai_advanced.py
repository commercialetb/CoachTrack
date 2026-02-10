# =================================================================
# COACHTRACK ELITE AI v3.3 FINAL - MediaPipe 0.10.9
# Compatible with mp.solutions API
# =================================================================

import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import time

# =================================================================
# MEDIAPIPE IMPORT - 0.10.9 (solutions API)
# =================================================================

MEDIAPIPE_AVAILABLE = False
mp_pose = None
mp_drawing = None

try:
    import mediapipe as mp

    # MediaPipe 0.10.9 ha mp.solutions
    if hasattr(mp, 'solutions'):
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        MEDIAPIPE_AVAILABLE = True
        print(f"✅ MediaPipe {mp.__version__} caricato (solutions API)", flush=True)
    else:
        print(f"⚠️ MediaPipe {mp.__version__} - solutions not found", flush=True)

except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"⚠️ MediaPipe non disponibile: {e}", flush=True)

except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"⚠️ Errore MediaPipe: {e}", flush=True)

# Scipy
SCIPY_AVAILABLE = False
try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

print(f"🔧 CV AI Module Status:", flush=True)
print(f"   - MediaPipe: {'✅' if MEDIAPIPE_AVAILABLE else '❌'}", flush=True)
print(f"   - Scipy: {'✅' if SCIPY_AVAILABLE else '❌'}", flush=True)


# =================================================================
# ACTION RECOGNIZER
# =================================================================

class ActionRecognizer:
    """Riconosce azioni basket con rule-based system"""

    def __init__(self):
        self.actions = ['shoot', 'pass', 'dribble', 'rebound', 'defense', 'idle']
        self.window_size = 16
        print("✅ ActionRecognizer inizializzato (rule-based)", flush=True)

    def predict_action(self, pose_landmarks: List[Dict], ball_position=None) -> Dict:
        """Predice azione da pose landmarks"""
        if not pose_landmarks or len(pose_landmarks) == 0:
            return {'action': 'idle', 'confidence': 0.5}

        last_pose = pose_landmarks[-1]

        # Estrai keypoints
        right_wrist = last_pose.get('right_wrist', {'y': 0.5})
        left_wrist = last_pose.get('left_wrist', {'y': 0.5})
        nose = last_pose.get('nose', {'y': 0.3})
        right_hip = last_pose.get('right_hip', {'y': 0.6})

        # SHOOTING: mani sopra la testa
        if right_wrist.get('y', 1) < nose.get('y', 0) or left_wrist.get('y', 1) < nose.get('y', 0):
            return {'action': 'shoot', 'confidence': 0.75, 'frame': last_pose.get('frame', 0)}

        # DRIBBLING: mano bassa
        if right_wrist.get('y', 0) > right_hip.get('y', 0):
            return {'action': 'dribble', 'confidence': 0.65, 'frame': last_pose.get('frame', 0)}

        return {'action': 'idle', 'confidence': 0.60, 'frame': last_pose.get('frame', 0)}

    def process_video_actions(self, pose_data: List[Dict]) -> List[Dict]:
        """Processa timeline azioni da pose data"""
        actions_timeline = []

        if not pose_data:
            return actions_timeline

        for i in range(0, len(pose_data), self.window_size):
            window = pose_data[i:min(i+self.window_size, len(pose_data))]

            if len(window) < 3:
                continue

            action_result = self.predict_action(window)
            action_result['frame_start'] = i
            action_result['frame_end'] = i + len(window)
            actions_timeline.append(action_result)

        return actions_timeline


# =================================================================
# SHOT TRACKER
# =================================================================

class ShotTracker:
    """Traccia tiri e calcola analytics"""

    def __init__(self):
        self.completed_shots = []
        print("✅ ShotTracker inizializzato", flush=True)

    def detect_shot(self, ball_trajectory: List[Dict], player_pose: Dict) -> Optional[Dict]:
        """Rileva tiro da traiettoria palla"""
        if len(ball_trajectory) < 5:
            return None

        # Altezza massima traiettoria
        max_height = max([p.get('y', 0) for p in ball_trajectory])

        if max_height < 0.7:
            return None

        # Quality score semplificato
        quality_score = 75.0
        make_prob = 0.55

        shot_data = {
            'shot_id': len(self.completed_shots) + 1,
            'arc_height': round(max_height, 2),
            'quality_score': round(quality_score, 1),
            'make_probability': round(make_prob, 2),
            'trajectory_length': len(ball_trajectory)
        }

        self.completed_shots.append(shot_data)
        return shot_data


# =================================================================
# POSE ANALYZER
# =================================================================

class PoseAnalyzer:
    """Analizza pose giocatori con MediaPipe 0.10.9"""

    def __init__(self):
        self.pose_detector = None
        self.available = False

        print("🔍 Inizializzazione PoseAnalyzer...", flush=True)
        print(f"   MEDIAPIPE_AVAILABLE: {MEDIAPIPE_AVAILABLE}", flush=True)

        if MEDIAPIPE_AVAILABLE and mp_pose:
            try:
                self.pose_detector = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.available = True
                print("✅ MediaPipe Pose detector inizializzato!", flush=True)
            except Exception as e:
                print(f"❌ Errore init Pose: {e}", flush=True)
                self.available = False
        else:
            print("⚠️ MediaPipe Pose non disponibile", flush=True)

    def extract_pose(self, frame: np.ndarray) -> Optional[Dict]:
        """Estrae pose landmarks da frame"""
        if not self.available or not self.pose_detector:
            return None

        try:
            # RGB per MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process
            results = self.pose_detector.process(rgb_frame)

            if not results.pose_landmarks:
                return None

            # Converti landmarks in dict
            landmarks = results.pose_landmarks.landmark

            pose_dict = {
                'nose': self._landmark_to_dict(landmarks[0]),
                'left_shoulder': self._landmark_to_dict(landmarks[11]),
                'right_shoulder': self._landmark_to_dict(landmarks[12]),
                'left_elbow': self._landmark_to_dict(landmarks[13]),
                'right_elbow': self._landmark_to_dict(landmarks[14]),
                'left_wrist': self._landmark_to_dict(landmarks[15]),
                'right_wrist': self._landmark_to_dict(landmarks[16]),
                'left_hip': self._landmark_to_dict(landmarks[23]),
                'right_hip': self._landmark_to_dict(landmarks[24]),
                'left_knee': self._landmark_to_dict(landmarks[25]),
                'right_knee': self._landmark_to_dict(landmarks[26]),
                'left_ankle': self._landmark_to_dict(landmarks[27]),
                'right_ankle': self._landmark_to_dict(landmarks[28])
            }

            return pose_dict

        except Exception as e:
            print(f"⚠️ Errore extract_pose: {e}", flush=True)
            return None

    def _landmark_to_dict(self, landmark) -> Dict:
        """Converte landmark in dict"""
        return {
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z,
            'visibility': landmark.visibility
        }

    def analyze_shooting_form(self, pose: Dict) -> Dict:
        """Analizza meccanica tiro"""
        if not pose:
            return {
                'form_score': 0,
                'issues': ['No pose detected'],
                'recommendations': ['Riprova il rilevamento']
            }

        # Calcola angoli
        elbow_angle = self._calculate_angle(
            pose.get('right_shoulder', {}),
            pose.get('right_elbow', {}),
            pose.get('right_wrist', {})
        )

        knee_angle = self._calculate_angle(
            pose.get('right_hip', {}),
            pose.get('right_knee', {}),
            pose.get('right_ankle', {})
        )

        issues = []

        # Valutazione gomito (ideale: 85-95°)
        if elbow_angle < 80 or elbow_angle > 100:
            issues.append(f"Angolo gomito: {elbow_angle:.0f}° (ideale: 85-95°)")

        # Valutazione ginocchia (ideale: 40-50°)
        if knee_angle < 35 or knee_angle > 55:
            issues.append(f"Flessione ginocchia: {knee_angle:.0f}° (ideale: 40-50°)")

        # Calcola form score (0-10)
        form_score = 10.0
        form_score -= abs(elbow_angle - 90) * 0.1
        form_score -= abs(knee_angle - 45) * 0.1
        form_score = max(0, min(10, form_score))

        return {
            'elbow_angle': round(elbow_angle, 1),
            'knee_bend': round(knee_angle, 1),
            'form_score': round(form_score, 1),
            'issues': issues if issues else ['Ottima forma!'],
            'recommendations': self._generate_recommendations(elbow_angle, knee_angle)
        }

    def _calculate_angle(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """Calcola angolo tra 3 punti"""
        if not all([p1, p2, p3]):
            return 90.0

        # Vettori
        v1 = np.array([p1.get('x', 0) - p2.get('x', 0),
                       p1.get('y', 0) - p2.get('y', 0)])
        v2 = np.array([p3.get('x', 0) - p2.get('x', 0),
                       p3.get('y', 0) - p2.get('y', 0)])

        # Angolo
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

        return angle

    def _generate_recommendations(self, elbow: float, knee: float) -> List[str]:
        """Genera raccomandazioni"""
        recs = []

        if elbow < 85:
            recs.append("Aumenta l'angolo del gomito")
        elif elbow > 95:
            recs.append("Riduci l'angolo del gomito")

        if knee < 40:
            recs.append("Aumenta la flessione delle ginocchia")
        elif knee > 50:
            recs.append("Riduci la flessione")

        if not recs:
            recs.append("Meccanica ottima! Continua così")

        return recs

    def __del__(self):
        """Cleanup"""
        if self.pose_detector:
            try:
                self.pose_detector.close()
            except:
                pass


# =================================================================
# INTEGRATED CV AI PIPELINE
# =================================================================

class CVAIPipeline:
    """Pipeline integrata per analisi video basket"""

    def __init__(self):
        self.action_recognizer = ActionRecognizer()
        self.shot_tracker = ShotTracker()
        self.pose_analyzer = PoseAnalyzer()

        print("="*60, flush=True)
        print("✅ CV AI Pipeline inizializzata", flush=True)
        print(f"   - Action Recognition: ✅", flush=True)
        print(f"   - Shot Tracking: ✅", flush=True)
        print(f"   - Pose Estimation: {'✅' if self.pose_analyzer.available else '❌'}", flush=True)
        print("="*60, flush=True)

    def process_video_complete(self, video_path: str, output_json: str = "cv_ai_analysis.json") -> Dict:
        """Processa video completo con AI"""
        print(f"\n🎬 Processing video: {video_path}", flush=True)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {
                'error': 'Cannot open video',
                'statistics': {
                    'total_poses_detected': 0,
                    'total_actions': 0,
                    'total_shots': 0
                }
            }

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_poses = []
        frame_idx = 0
        processed_frames = 0

        print(f"📊 Video: {frame_count} frames @ {fps} fps", flush=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process ogni 5 frames per performance
            if frame_idx % 5 == 0:
                pose = self.pose_analyzer.extract_pose(frame)
                if pose:
                    pose['frame'] = frame_idx
                    pose['timestamp'] = frame_idx / fps
                    all_poses.append(pose)
                    processed_frames += 1

            frame_idx += 1

            # Progress log
            if frame_idx % 100 == 0:
                progress = (frame_idx / frame_count * 100) if frame_count > 0 else 0
                print(f"   Progress: {progress:.1f}% - Poses: {len(all_poses)}", flush=True)

        cap.release()

        print(f"\n🎯 Analyzing {len(all_poses)} poses...", flush=True)

        # Actions
        all_actions = self.action_recognizer.process_video_actions(all_poses)
        print(f"✅ Found {len(all_actions)} actions", flush=True)

        # Summary
        summary = {
            'video': video_path,
            'metadata': {
                'fps': fps,
                'total_frames': frame_count,
                'processed_frames': processed_frames,
                'duration_sec': frame_count / fps if fps > 0 else 0,
                'processed_at': datetime.now().isoformat()
            },
            'pose_data': all_poses[:100],  # Primi 100 per dimensione file
            'actions': all_actions,
            'shots': [],
            'statistics': {
                'total_poses_detected': len(all_poses),
                'total_actions': len(all_actions),
                'total_shots': 0
            }
        }

        # Salva JSON
        with open(output_json, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Analysis complete! Saved: {output_json}", flush=True)
        return summary


# =================================================================
# CONVENIENCE FUNCTIONS
# =================================================================

def analyze_video_ai(video_path: str, output_json: str = "cv_ai_analysis.json") -> Dict:
    """Analisi video completa"""
    try:
        pipeline = CVAIPipeline()
        return pipeline.process_video_complete(video_path, output_json)
    except Exception as e:
        print(f"❌ Errore analyze_video_ai: {e}", flush=True)
        return {
            'error': str(e),
            'statistics': {
                'total_poses_detected': 0,
                'total_actions': 0,
                'total_shots': 0
            }
        }


def analyze_player_form(frame: np.ndarray) -> Dict:
    """Analizza shooting form da singolo frame"""
    analyzer = PoseAnalyzer()
    pose = analyzer.extract_pose(frame)
    if pose:
        return analyzer.analyze_shooting_form(pose)
    return {
        'form_score': 0,
        'issues': ['No pose detected'],
        'recommendations': ['Riprova il rilevamento']
    }


# =================================================================
# MODULE INFO
# =================================================================

__version__ = "3.3.0"
__status__ = "MediaPipe 0.10.9 (solutions API)"
__all__ = [
    'ActionRecognizer',
    'ShotTracker',
    'PoseAnalyzer',
    'CVAIPipeline',
    'analyze_video_ai',
    'analyze_player_form',
    'MEDIAPIPE_AVAILABLE'
]

if __name__ == "__main__":
    print("="*60)
    print("✅ CV AI Advanced Module v3.3")
    print(f"   Version: {__version__}")
    print(f"   Status: {__status__}")
    print(f"   MediaPipe: {'✅' if MEDIAPIPE_AVAILABLE else '❌'}")
    print("="*60)
