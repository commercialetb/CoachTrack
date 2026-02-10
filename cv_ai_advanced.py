# =================================================================
# COACHTRACK ELITE AI v3.2 - CV AI ADVANCED MODULE
# Compatible with MediaPipe 0.10.30+ (NEW API)
# =================================================================

import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import time

# =================================================================
# MEDIAPIPE IMPORT - NEW API (0.10.30+)
# =================================================================

MEDIAPIPE_AVAILABLE = False
mp = None

try:
    import mediapipe as mp
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2

    # Check if solutions module has pose
    if hasattr(solutions, 'pose'):
        MEDIAPIPE_AVAILABLE = True
        print(f"✅ MediaPipe {mp.__version__} caricato (NEW API)")
    else:
        print(f"⚠️ MediaPipe {mp.__version__} importato ma pose non disponibile")

except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"⚠️ MediaPipe non installato: {e}")
    print("   Comando: pip install mediapipe")
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"⚠️ Errore import MediaPipe: {e}")

# Scipy for trajectory fitting
SCIPY_AVAILABLE = False
try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

print(f"🔧 CV AI Module Status:")
print(f"   - MediaPipe: {'✅' if MEDIAPIPE_AVAILABLE else '❌'}")
print(f"   - Scipy: {'✅' if SCIPY_AVAILABLE else '❌'}")

# =================================================================
# 1️⃣ ACTION RECOGNITION
# =================================================================

class ActionRecognizer:
    """
    Riconosce azioni basket da sequenze di frame
    Actions: shoot, pass, dribble, rebound, defense, idle
    """

    def __init__(self):
        self.actions = ['shoot', 'pass', 'dribble', 'rebound', 'defense', 'idle']
        self.window_size = 16
        self.frame_buffer = []
        print("✅ ActionRecognizer inizializzato (rule-based)")

    def predict_action(self, pose_landmarks: List[Dict], ball_position: Optional[Dict] = None) -> Dict:
        """Predice azione da pose landmarks"""
        return self._predict_rule_based(pose_landmarks, ball_position)

    def _predict_rule_based(self, poses: List[Dict], ball_pos: Optional[Dict]) -> Dict:
        """Rule-based classifier"""
        if not poses or len(poses) == 0:
            return {'action': 'idle', 'confidence': 0.5}

        last_pose = poses[-1] if poses else None
        if not last_pose:
            return {'action': 'idle', 'confidence': 0.5}

        # Extract key points
        right_wrist = last_pose.get('right_wrist', {'y': 0.5})
        left_wrist = last_pose.get('left_wrist', {'y': 0.5})
        nose = last_pose.get('nose', {'y': 0.3})
        right_hip = last_pose.get('right_hip', {'y': 0.6})

        # SHOOTING: mani sopra testa
        if right_wrist.get('y', 1) < nose.get('y', 0) or left_wrist.get('y', 1) < nose.get('y', 0):
            return {'action': 'shoot', 'confidence': 0.75, 'frame': last_pose.get('frame', 0)}

        # PASSING: movimento laterale rapido
        if len(poses) >= 2:
            prev_wrist = poses[-2].get('right_wrist', {'x': 0})
            curr_wrist = right_wrist
            if abs(curr_wrist.get('x', 0) - prev_wrist.get('x', 0)) > 0.15:
                return {'action': 'pass', 'confidence': 0.70, 'frame': last_pose.get('frame', 0)}

        # DRIBBLING: mano bassa
        if right_wrist.get('y', 0) > right_hip.get('y', 0):
            return {'action': 'dribble', 'confidence': 0.65, 'frame': last_pose.get('frame', 0)}

        return {'action': 'idle', 'confidence': 0.60, 'frame': last_pose.get('frame', 0)}

    def process_video_actions(self, pose_data: List[Dict]) -> List[Dict]:
        """Processa intero video e restituisce timeline azioni"""
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
            action_result['timestamp'] = window[0].get('frame', 0) / 30.0  # Assume 30fps
            actions_timeline.append(action_result)

        return actions_timeline


# =================================================================
# 2️⃣ SHOT TRACKING & ANALYTICS
# =================================================================

class ShotTracker:
    """Traccia tiri e calcola analytics"""

    def __init__(self):
        self.court_width = 28.0
        self.court_length = 15.0
        self.basket_height = 3.05
        self.completed_shots = []
        print("✅ ShotTracker inizializzato")

    def detect_shot(self, ball_trajectory: List[Dict], player_pose: Dict) -> Optional[Dict]:
        """Rileva se sequenza movimento è un tiro"""
        if len(ball_trajectory) < 5:
            return None

        max_height = max([p.get('y', 0) for p in ball_trajectory])

        if max_height < 0.7:  # Normalized height threshold
            return None

        release_point = ball_trajectory[0]
        release_angle = 45.0  # Default
        release_speed = 7.0   # Default m/s

        quality_score = self._calculate_shot_quality(release_angle, release_speed, max_height)
        make_prob = self._predict_make_probability(quality_score)

        return {
            'shot_id': len(self.completed_shots) + 1,
            'release_point': release_point,
            'release_angle': round(release_angle, 1),
            'release_speed': round(release_speed, 2),
            'arc_height': round(max_height, 2),
            'quality_score': round(quality_score, 1),
            'make_probability': round(make_prob, 2),
            'trajectory_length': len(ball_trajectory)
        }

    def _calculate_shot_quality(self, angle: float, speed: float, arc: float) -> float:
        """Calcola quality score 0-100"""
        score = 100.0
        score -= min(abs(angle - 50) * 2, 30)
        score -= min(abs(speed - 7.0) * 5, 25)
        score -= min(abs(arc - 0.75) * 50, 25)
        return max(0, min(100, score))

    def _predict_make_probability(self, quality: float) -> float:
        """Predice probabilità di fare canestro"""
        if quality >= 80:
            return 0.60 + (quality - 80) * 0.01
        elif quality >= 50:
            return 0.35 + (quality - 50) * 0.0083
        else:
            return 0.15 + quality * 0.004


# =================================================================
# 3️⃣ POSE ESTIMATION - NEW API
# =================================================================

class PoseAnalyzer:
    """Analizza pose giocatori con MediaPipe 0.10.30+"""

    def __init__(self):
        self.pose_detector = None
        self.available = False

        print(f"🔍 Inizializzazione PoseAnalyzer...")
        print(f"   MEDIAPIPE_AVAILABLE: {MEDIAPIPE_AVAILABLE}")

        if MEDIAPIPE_AVAILABLE and mp is not None:
            try:
                # Use solutions API (NEW)
                self.pose_detector = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.available = True
                print("✅ MediaPipe Pose detector inizializzato (NEW API)")

            except Exception as e:
                self.available = False
                print(f"⚠️ Errore inizializzazione Pose: {e}")
        else:
            print("⚠️ MediaPipe non disponibile")

    def extract_pose(self, frame: np.ndarray) -> Optional[Dict]:
        """Estrae pose landmarks da frame"""
        if not self.available or self.pose_detector is None:
            return None

        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(frame_rgb)

            if not results.pose_landmarks:
                return None

            # Extract landmarks
            landmarks = {}
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks[idx] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }

            # Map to named landmarks
            pose_dict = {
                'nose': landmarks.get(0, {}),
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

        except Exception as e:
            print(f"⚠️ Errore extract_pose: {e}")
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
        if self.pose_detector:
            try:
                self.pose_detector.close()
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

        print("="*60)
        print("✅ CV AI Pipeline inizializzata")
        print(f"   - Action Recognition: ✅")
        print(f"   - Shot Tracking: ✅")
        print(f"   - Pose Estimation: {'✅' if MEDIAPIPE_AVAILABLE else '❌'}")
        print("="*60)

    def process_video_complete(self, video_path: str, output_json: str = "cv_ai_analysis.json") -> Dict:
        """Processa video con tutte le analisi AI"""
        print(f"\n🎬 Processing video: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {'error': 'Cannot open video', 'statistics': {}}

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_poses = []
        frame_idx = 0
        processed_frames = 0

        print(f"📊 Video info: {frame_count} frames @ {fps} fps")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every 5th frame for speed
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
                print(f"   Progress: {progress:.1f}% ({frame_idx}/{frame_count}) - Poses: {len(all_poses)}")

        cap.release()

        print(f"\n🎯 Analyzing {len(all_poses)} poses...")
        all_actions = self.action_recognizer.process_video_actions(all_poses)

        print(f"✅ Found {len(all_actions)} actions")

        summary = {
            'video': video_path,
            'metadata': {
                'fps': fps,
                'total_frames': frame_count,
                'processed_frames': processed_frames,
                'duration_sec': frame_count / fps if fps > 0 else 0,
                'processed_at': datetime.now().isoformat()
            },
            'pose_data': all_poses[:100],  # Limit to first 100 for JSON size
            'actions': all_actions,
            'shots': [],
            'statistics': {
                'total_poses_detected': len(all_poses),
                'total_actions': len(all_actions),
                'total_shots': 0
            }
        }

        # Save to JSON
        with open(output_json, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Analysis complete! Saved to: {output_json}")
        print(f"📊 Summary: {len(all_poses)} poses, {len(all_actions)} actions")

        return summary


# =================================================================
# CONVENIENCE FUNCTIONS
# =================================================================

def analyze_video_ai(video_path: str, output_json: str = "cv_ai_analysis.json") -> Dict:
    """Funzione semplice per analisi completa video"""
    try:
        pipeline = CVAIPipeline()
        return pipeline.process_video_complete(video_path, output_json)
    except Exception as e:
        print(f"❌ Errore in analyze_video_ai: {e}")
        return {
            'error': str(e),
            'statistics': {
                'total_poses_detected': 0,
                'total_actions': 0,
                'total_shots': 0
            }
        }


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

__version__ = "3.2.0"
__status__ = "Compatible with MediaPipe 0.10.30+"
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
    print("✅ CV AI Advanced Module v3.2")
    print(f"   Version: {__version__}")
    print(f"   Status: {__status__}")
    print(f"   MediaPipe: {'✅' if MEDIAPIPE_AVAILABLE else '❌'}")
    print("="*60)
