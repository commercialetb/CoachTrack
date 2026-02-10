# =================================================================
# COACHTRACK ELITE AI v4.0 - MediaPipe Tasks API 0.10.32+
# =================================================================

import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import time
import os

# =================================================================
# MEDIAPIPE TASKS API - 0.10.32+
# =================================================================

MEDIAPIPE_AVAILABLE = False

try:
    from mediapipe import Image
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe Tasks API 0.10.32+ caricato")
except ImportError as e:
    print(f"⚠️ MediaPipe Tasks non disponibile: {e}")
except Exception as e:
    print(f"⚠️ Errore MediaPipe Tasks: {e}")

# Scipy for trajectory
SCIPY_AVAILABLE = False
try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

print(f"🔧 CV AI Status:")
print(f"   - MediaPipe Tasks: {'✅' if MEDIAPIPE_AVAILABLE else '❌'}")
print(f"   - Scipy: {'✅' if SCIPY_AVAILABLE else '❌'}")

# =================================================================
# POSE LANDMARKER CLASS
# =================================================================

class PoseLandmarker:
    """Pose detection con MediaPipe Tasks API"""

    def __init__(self, model_path: str = "pose_landmarker_lite.task"):
        self.landmarker = None
        self.model_path = model_path

        if MEDIAPIPE_AVAILABLE and os.path.exists(model_path):
            try:
                BaseOptions = python.BaseOptions
                PoseLandmarker = vision.PoseLandmarker
                PoseLandmarkerOptions = vision.PoseLandmarkerOptions

                base_options = BaseOptions(model_asset_path=model_path)
                options = PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_segmentation_masks=False
                )

                self.landmarker = PoseLandmarker.create_from_options(options)
                print(f"✅ PoseLandmarker caricato: {model_path}")

            except Exception as e:
                print(f"⚠️ Errore PoseLandmarker: {e}")
                self.landmarker = None
        else:
            print(f"⚠️ Modello non trovato: {model_path}")

    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        """Rileva pose su frame"""
        if not self.landmarker:
            return None

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=Image.Format.SRGB, data=rgb_frame)

            detection_result = self.landmarker.detect(mp_image)

            if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
                return None

            landmarks = detection_result.pose_landmarks[0]

            # Converti in formato standard (33 keypoints)
            pose_dict = {}
            for i, landmark in enumerate(landmarks):
                pose_dict[i] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'presence': landmark.presence
                }

            # Mappa nomi principali
            keypoint_names = {
                0: 'nose',
                11: 'left_shoulder', 12: 'right_shoulder',
                13: 'left_elbow', 14: 'right_elbow',
                15: 'left_wrist', 16: 'right_wrist',
                23: 'left_hip', 24: 'right_hip',
                25: 'left_knee', 26: 'right_knee',
                27: 'left_ankle', 28: 'right_ankle'
            }

            named_pose = {name: pose_dict.get(idx, {}) for idx, name in keypoint_names.items()}
            named_pose['raw_landmarks'] = pose_dict

            return named_pose

        except Exception as e:
            print(f"⚠️ Errore detect: {e}")
            return None

    def __del__(self):
        if self.landmarker:
            try:
                self.landmarker.close()
            except:
                pass

# =================================================================
# POSE ANALYZER
# =================================================================

class PoseAnalyzer:
    """Analizza pose giocatori"""

    def __init__(self):
        self.landmarker = PoseLandmarker()
        print("✅ PoseAnalyzer inizializzato (Tasks API)")

    def extract_pose(self, frame: np.ndarray) -> Optional[Dict]:
        """Estrae pose da frame"""
        return self.landmarker.detect(frame)

    def analyze_shooting_form(self, pose: Dict) -> Dict:
        """Analizza meccanica tiro"""
        if not pose:
            return {'form_score': 0, 'issues': ['No pose detected']}

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

        if elbow_angle < 80 or elbow_angle > 100:
            issues.append(f"Gomito: {elbow_angle:.0f}° (ideale: 85-95°)")

        if knee_angle < 35 or knee_angle > 55:
            issues.append(f"Ginocchia: {knee_angle:.0f}° (ideale: 40-50°)")

        form_score = 10.0 - abs(elbow_angle - 90) * 0.1 - abs(knee_angle - 45) * 0.1
        form_score = max(0, min(10, form_score))

        return {
            'elbow_angle': round(elbow_angle, 1),
            'knee_bend': round(knee_angle, 1),
            'form_score': round(form_score, 1),
            'issues': issues,
            'recommendations': self._generate_recommendations(elbow_angle, knee_angle)
        }

    def _calculate_angle(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """Calcola angolo tra 3 punti"""
        if not all([p1, p2, p3]):
            return 90.0

        v1 = np.array([p1.get('x', 0) - p2.get('x', 0), 
                       p1.get('y', 0) - p2.get('y', 0)])
        v2 = np.array([p3.get('x', 0) - p2.get('x', 0), 
                       p3.get('y', 0) - p2.get('y', 0)])

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        return angle

    def _generate_recommendations(self, elbow: float, knee: float) -> List[str]:
        recs = []
        if elbow < 85:
            recs.append("Aumenta angolo gomito")
        elif elbow > 95:
            recs.append("Riduci angolo gomito")

        if knee < 40:
            recs.append("Aumenta flessione ginocchia")
        elif knee > 50:
            recs.append("Riduci flessione")

        if not recs:
            recs.append("Ottima forma!")
        return recs

# =================================================================
# ACTION RECOGNIZER & SHOT TRACKER (stessi di prima)
# =================================================================

class ActionRecognizer:
    def __init__(self):
        self.actions = ['shoot', 'pass', 'dribble', 'rebound', 'defense', 'idle']
        self.window_size = 16
        print("✅ ActionRecognizer inizializzato")

    def predict_action(self, pose_landmarks: List[Dict], ball_position=None) -> Dict:
        if not pose_landmarks or len(pose_landmarks) == 0:
            return {'action': 'idle', 'confidence': 0.5}

        last_pose = pose_landmarks[-1]
        right_wrist = last_pose.get('right_wrist', {'y': 0.5})
        left_wrist = last_pose.get('left_wrist', {'y': 0.5})
        nose = last_pose.get('nose', {'y': 0.3})
        right_hip = last_pose.get('right_hip', {'y': 0.6})

        # SHOOTING
        if right_wrist.get('y', 1) < nose.get('y', 0) or left_wrist.get('y', 1) < nose.get('y', 0):
            return {'action': 'shoot', 'confidence': 0.75, 'frame': last_pose.get('frame', 0)}

        # DRIBBLING
        if right_wrist.get('y', 0) > right_hip.get('y', 0):
            return {'action': 'dribble', 'confidence': 0.65, 'frame': last_pose.get('frame', 0)}

        return {'action': 'idle', 'confidence': 0.60, 'frame': last_pose.get('frame', 0)}

class ShotTracker:
    def __init__(self):
        print("✅ ShotTracker inizializzato")

    def detect_shot(self, ball_trajectory: List[Dict], player_pose: Dict) -> Optional[Dict]:
        if len(ball_trajectory) < 5:
            return None

        max_height = max([p.get('y', 0) for p in ball_trajectory])
        if max_height < 0.7:
            return None

        quality_score = 75.0  # Default
        return {
            'shot_id': 1,
            'arc_height': round(max_height, 2),
            'quality_score': round(quality_score, 1),
            'make_probability': 0.55,
            'trajectory_length': len(ball_trajectory)
        }

# =================================================================
# PIPELINE COMPLETA
# =================================================================

class CVAIPipeline:
    def __init__(self):
        self.action_recognizer = ActionRecognizer()
        self.shot_tracker = ShotTracker()
        self.pose_analyzer = PoseAnalyzer()
        print("✅ CV AI Pipeline v4.0 (Tasks API)")

    def process_video_complete(self, video_path: str, output_json: str = "cv_ai_analysis.json") -> Dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': 'Cannot open video'}

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        all_poses = []

        print(f"📊 Processing {frame_count} frames @ {fps} fps")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 5 == 0:
                pose = self.pose_analyzer.extract_pose(frame)
                if pose:
                    pose['frame'] = frame_idx
                    pose['timestamp'] = frame_idx / fps
                    all_poses.append(pose)

            frame_idx += 1

        cap.release()

        actions = self.action_recognizer.process_video_actions(all_poses)

        summary = {
            'video': video_path,
            'metadata': {
                'fps': fps,
                'total_frames': frame_count,
                'poses_detected': len(all_poses),
                'actions': len(actions),
                'processed_at': datetime.now().isoformat()
            },
            'statistics': {
                'total_poses_detected': len(all_poses),
                'total_actions': len(actions)
            }
        }

        with open(output_json, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✅ Analysis completa: {len(all_poses)} poses, {len(actions)} actions")
        return summary

# =================================================================
# EXPORTS
# =================================================================

def analyze_video_ai(video_path: str):
    try:
        pipeline = CVAIPipeline()
        return pipeline.process_video_complete(video_path)
    except Exception as e:
        return {'error': str(e)}

def analyze_player_form(frame: np.ndarray):
    analyzer = PoseAnalyzer()
    pose = analyzer.extract_pose(frame)
    if pose:
        return analyzer.analyze_shooting_form(pose)
    return {'form_score': 0, 'issues': ['No pose detected']}

__version__ = "4.0.0"
__status__ = "MediaPipe Tasks API 0.10.32+"
__all__ = ['PoseLandmarker', 'PoseAnalyzer', 'ActionRecognizer', 'ShotTracker', 'CVAIPipeline', 'analyze_video_ai', 'analyze_player_form', 'MEDIAPIPE_AVAILABLE']

if __name__ == "__main__":
    print(f"✅ CV AI v{__version__} - {__status__}")
    print(f"   MediaPipe Tasks: {'✅' if MEDIAPIPE_AVAILABLE else '❌'}")
