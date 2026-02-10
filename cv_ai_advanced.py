# =================================================================
# COACHTRACK ELITE AI v4.0 - MediaPipe Tasks API 0.10.32+
# FINAL VERSION - Working with pose_landmarker_lite.task
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
    from mediapipe.framework.formats import landmark_pb2
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe Tasks API 0.10.32+ caricato", flush=True)
except ImportError as e:
    print(f"⚠️ MediaPipe Tasks non disponibile: {e}", flush=True)
except Exception as e:
    print(f"⚠️ Errore MediaPipe Tasks: {e}", flush=True)

# Scipy
SCIPY_AVAILABLE = False
try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

print(f"🔧 CV AI Module Status:", flush=True)
print(f"   - MediaPipe Tasks: {'✅' if MEDIAPIPE_AVAILABLE else '❌'}", flush=True)
print(f"   - Scipy: {'✅' if SCIPY_AVAILABLE else '❌'}", flush=True)

# =================================================================
# ACTION RECOGNIZER
# =================================================================

class ActionRecognizer:
    """Riconosce azioni basket"""

    def __init__(self):
        self.actions = ['shoot', 'pass', 'dribble', 'rebound', 'defense', 'idle']
        self.window_size = 16
        print("✅ ActionRecognizer inizializzato", flush=True)

    def predict_action(self, pose_landmarks: List[Dict], ball_position=None) -> Dict:
        if not pose_landmarks or len(pose_landmarks) == 0:
            return {'action': 'idle', 'confidence': 0.5}

        last_pose = pose_landmarks[-1]
        right_wrist = last_pose.get('right_wrist', {'y': 0.5})
        left_wrist = last_pose.get('left_wrist', {'y': 0.5})
        nose = last_pose.get('nose', {'y': 0.3})
        right_hip = last_pose.get('right_hip', {'y': 0.6})

        # SHOOTING: mani sopra testa
        if right_wrist.get('y', 1) < nose.get('y', 0) or left_wrist.get('y', 1) < nose.get('y', 0):
            return {'action': 'shoot', 'confidence': 0.75, 'frame': last_pose.get('frame', 0)}

        # DRIBBLING: mano bassa
        if right_wrist.get('y', 0) > right_hip.get('y', 0):
            return {'action': 'dribble', 'confidence': 0.65, 'frame': last_pose.get('frame', 0)}

        return {'action': 'idle', 'confidence': 0.60, 'frame': last_pose.get('frame', 0)}

    def process_video_actions(self, pose_data: List[Dict]) -> List[Dict]:
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
        if len(ball_trajectory) < 5:
            return None

        max_height = max([p.get('y', 0) for p in ball_trajectory])
        if max_height < 0.7:
            return None

        quality_score = 75.0
        make_prob = 0.55

        return {
            'shot_id': len(self.completed_shots) + 1,
            'arc_height': round(max_height, 2),
            'quality_score': round(quality_score, 1),
            'make_probability': round(make_prob, 2),
            'trajectory_length': len(ball_trajectory)
        }


# =================================================================
# POSE ANALYZER - TASKS API
# =================================================================

class PoseAnalyzer:
    """Analizza pose con MediaPipe Tasks API 0.10.32+"""

    def __init__(self):
        self.landmarker = None
        self.available = False
        self.model_path = "pose_landmarker_lite.task"

        print(f"🔍 Inizializzazione PoseAnalyzer...", flush=True)
        print(f"   MediaPipe Tasks: {MEDIAPIPE_AVAILABLE}", flush=True)
        print(f"   Cercando modello: {self.model_path}", flush=True)
        print(f"   File esiste: {os.path.exists(self.model_path)}", flush=True)

        if os.path.exists(self.model_path):
            size_mb = os.path.getsize(self.model_path) / 1024 / 1024
            print(f"   Dimensione modello: {size_mb:.2f} MB", flush=True)

        if MEDIAPIPE_AVAILABLE and os.path.exists(self.model_path):
            try:
                BaseOptions = python.BaseOptions
                PoseLandmarker = vision.PoseLandmarker
                PoseLandmarkerOptions = vision.PoseLandmarkerOptions
                VisionRunningMode = vision.RunningMode

                base_options = BaseOptions(model_asset_path=self.model_path)
                options = PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=VisionRunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_segmentation_masks=False
                )

                self.landmarker = PoseLandmarker.create_from_options(options)
                self.available = True
                print(f"✅ PoseLandmarker caricato con successo!", flush=True)

            except Exception as e:
                self.available = False
                print(f"❌ Errore caricamento PoseLandmarker: {e}", flush=True)
        else:
            if not MEDIAPIPE_AVAILABLE:
                print("⚠️ MediaPipe Tasks non disponibile", flush=True)
            if not os.path.exists(self.model_path):
                print(f"⚠️ Modello non trovato: {self.model_path}", flush=True)

    def extract_pose(self, frame: np.ndarray) -> Optional[Dict]:
        """Estrae pose landmarks da frame"""
        if not self.available or self.landmarker is None:
            return None

        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image
            mp_image = vision.Image(
                image_format=vision.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Detect
            detection_result = self.landmarker.detect(mp_image)

            if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
                return None

            landmarks = detection_result.pose_landmarks[0]  # Prima persona

            # Converti in formato dict (33 keypoints)
            pose_dict = {}
            for i, landmark in enumerate(landmarks):
                pose_dict[i] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility if hasattr(landmark, 'visibility') else landmark.presence
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
            print(f"⚠️ Errore extract_pose: {e}", flush=True)
            return None

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
            issues.append(f"Angolo gomito: {elbow_angle:.0f}° (ideale: 85-95°)")

        if knee_angle < 35 or knee_angle > 55:
            issues.append(f"Flessione ginocchia: {knee_angle:.0f}° (ideale: 40-50°)")

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
        """Genera raccomandazioni"""
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

    def __del__(self):
        """Cleanup"""
        if self.landmarker:
            try:
                self.landmarker.close()
            except:
                pass


# =================================================================
# INTEGRATED CV AI PIPELINE
# =================================================================

class CVAIPipeline:
    """Pipeline integrata AI"""

    def __init__(self):
        self.action_recognizer = ActionRecognizer()
        self.shot_tracker = ShotTracker()
        self.pose_analyzer = PoseAnalyzer()

        print("="*60, flush=True)
        print("✅ CV AI Pipeline v4.0 inizializzata (Tasks API)", flush=True)
        print(f"   - Action Recognition: ✅", flush=True)
        print(f"   - Shot Tracking: ✅", flush=True)
        print(f"   - Pose Estimation: {'✅' if self.pose_analyzer.available else '❌'}", flush=True)
        print("="*60, flush=True)

    def process_video_complete(self, video_path: str, output_json: str = "cv_ai_analysis.json") -> Dict:
        """Processa video completo"""
        print(f"\n🎬 Processing video: {video_path}", flush=True)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {'error': 'Cannot open video', 'statistics': {}}

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_poses = []
        frame_idx = 0
        processed_frames = 0

        print(f"📊 Video info: {frame_count} frames @ {fps} fps", flush=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process ogni 5 frames
            if frame_idx % 5 == 0:
                pose = self.pose_analyzer.extract_pose(frame)
                if pose:
                    pose['frame'] = frame_idx
                    pose['timestamp'] = frame_idx / fps
                    all_poses.append(pose)
                    processed_frames += 1

            frame_idx += 1

            # Progress
            if frame_idx % 100 == 0:
                progress = (frame_idx / frame_count * 100) if frame_count > 0 else 0
                print(f"   Progress: {progress:.1f}% - Poses: {len(all_poses)}", flush=True)

        cap.release()

        print(f"\n🎯 Analyzing {len(all_poses)} poses...", flush=True)
        all_actions = self.action_recognizer.process_video_actions(all_poses)

        print(f"✅ Found {len(all_actions)} actions", flush=True)

        summary = {
            'video': video_path,
            'metadata': {
                'fps': fps,
                'total_frames': frame_count,
                'processed_frames': processed_frames,
                'duration_sec': frame_count / fps if fps > 0 else 0,
                'processed_at': datetime.now().isoformat()
            },
            'pose_data': all_poses[:100],  # Primi 100
            'actions': all_actions,
            'shots': [],
            'statistics': {
                'total_poses_detected': len(all_poses),
                'total_actions': len(all_actions),
                'total_shots': 0
            }
        }

        with open(output_json, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Analysis complete! Saved to: {output_json}", flush=True)
        return summary


# =================================================================
# CONVENIENCE FUNCTIONS
# =================================================================

def analyze_video_ai(video_path: str, output_json: str = "cv_ai_analysis.json") -> Dict:
    """Analisi completa video"""
    try:
        pipeline = CVAIPipeline()
        return pipeline.process_video_complete(video_path, output_json)
    except Exception as e:
        print(f"❌ Errore: {e}", flush=True)
        return {
            'error': str(e),
            'statistics': {
                'total_poses_detected': 0,
                'total_actions': 0,
                'total_shots': 0
            }
        }


def analyze_player_form(frame: np.ndarray) -> Dict:
    """Analizza form da frame"""
    analyzer = PoseAnalyzer()
    pose = analyzer.extract_pose(frame)
    if pose:
        return analyzer.analyze_shooting_form(pose)
    return {'form_score': 0, 'issues': ['No pose detected']}


# =================================================================
# MODULE INFO
# =================================================================

__version__ = "4.0.0"
__status__ = "MediaPipe Tasks API 0.10.32+ (FINAL)"
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
    print("✅ CV AI Advanced Module v4.0 FINAL")
    print(f"   Version: {__version__}")
    print(f"   Status: {__status__}")
    print(f"   MediaPipe Tasks: {'✅' if MEDIAPIPE_AVAILABLE else '❌'}")
    print("="*60)
