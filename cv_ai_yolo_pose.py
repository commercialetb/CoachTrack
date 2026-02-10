# =================================================================
# COACHTRACK ELITE AI v5.0 - YOLOv8 Pose (NO MediaPipe!)
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
# YOLOv8 POSE ESTIMATION
# =================================================================

YOLO_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLOv8 Pose caricato!", flush=True)
except ImportError:
    print("⚠️ Installa: pip install ultralytics", flush=True)

# Scipy
SCIPY_AVAILABLE = False
try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except:
    SCIPY_AVAILABLE = False

print(f"🔧 CV AI Status:", flush=True)
print(f"   - YOLOv8 Pose: {'✅' if YOLO_AVAILABLE else '❌'}", flush=True)
print(f"   - Scipy: {'✅' if SCIPY_AVAILABLE else '❌'}", flush=True)

# =================================================================
# YOLO POSE ANALYZER
# =================================================================

class YOLOPoseAnalyzer:
    """Pose estimation con YOLOv8n-pose (17 keypoints)"""

    def __init__(self):
        self.model = None
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO('yolov8n-pose.pt')  # Auto-download ~6MB
                print("✅ YOLOv8n-pose model loaded!", flush=True)
            except Exception as e:
                print(f"⚠️ YOLO model error: {e}", flush=True)
                self.model = None

    def extract_pose(self, frame: np.ndarray) -> Optional[Dict]:
        """Estrae pose da frame"""
        if not self.model:
            return None

        try:
            # Inference
            results = self.model(frame, verbose=False)

            # Prendi prima persona
            if results[0].keypoints is not None and len(results[0].keypoints) > 0:
                keypoints = results[0].keypoints.xy[0].cpu().numpy()
                confidences = results[0].keypoints.conf[0].cpu().numpy()

                if keypoints.shape[0] == 17:
                    # Mappa YOLO keypoints (COCO format)
                    pose_dict = {
                        'nose': self._kp_to_dict(keypoints[0], confidences[0]),
                        'left_eye': self._kp_to_dict(keypoints[1], confidences[1]),
                        'right_eye': self._kp_to_dict(keypoints[2], confidences[2]),
                        'left_ear': self._kp_to_dict(keypoints[3], confidences[3]),
                        'right_ear': self._kp_to_dict(keypoints[4], confidences[4]),
                        'left_shoulder': self._kp_to_dict(keypoints[5], confidences[5]),
                        'right_shoulder': self._kp_to_dict(keypoints[6], confidences[6]),
                        'left_elbow': self._kp_to_dict(keypoints[7], confidences[7]),
                        'right_elbow': self._kp_to_dict(keypoints[8], confidences[8]),
                        'left_wrist': self._kp_to_dict(keypoints[9], confidences[9]),
                        'right_wrist': self._kp_to_dict(keypoints[10], confidences[10]),
                        'left_hip': self._kp_to_dict(keypoints[11], confidences[11]),
                        'right_hip': self._kp_to_dict(keypoints[12], confidences[12]),
                        'left_knee': self._kp_to_dict(keypoints[13], confidences[13]),
                        'right_knee': self._kp_to_dict(keypoints[14], confidences[14]),
                        'left_ankle': self._kp_to_dict(keypoints[15], confidences[15]),
                        'right_ankle': self._kp_to_dict(keypoints[16], confidences[16])
                    }

                    return pose_dict

        except Exception as e:
            print(f"⚠️ YOLO pose error: {e}", flush=True)

        return None

    def _kp_to_dict(self, kp: np.ndarray, conf: float) -> Dict:
        """Converte YOLO keypoint in dict"""
        if kp[0] is None or kp[1] is None:
            return {'x': 0, 'y': 0, 'confidence': 0}
        return {
            'x': float(kp[0]),
            'y': float(kp[1]),
            'confidence': float(conf)
        }

    def analyze_shooting_form(self, pose: Dict) -> Dict:
        """Analizza meccanica tiro"""
        if not pose:
            return {'form_score': 0, 'issues': ['No pose detected']}

        # Angolo gomito destro
        elbow_angle = self._calculate_angle(
            pose.get('right_shoulder', {}),
            pose.get('right_elbow', {}),
            pose.get('right_wrist', {})
        )

        # Angolo ginocchia destro
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

        form_score = 10.0 - abs(elbow_angle - 90)*0.1 - abs(knee_angle - 45)*0.1
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
# ACTION RECOGNIZER (rule-based)
# =================================================================

class ActionRecognizer:
    def __init__(self):
        print("✅ ActionRecognizer inizializzato", flush=True)

    def predict_action(self, pose: Dict) -> Dict:
        if not pose:
            return {'action': 'idle', 'confidence': 0.5}

        right_wrist = pose.get('right_wrist', {'y': 0.5})
        nose = pose.get('nose', {'y': 0.3})
        right_hip = pose.get('right_hip', {'y': 0.6})

        # Shoot
        if right_wrist.get('y', 1) < nose.get('y', 0):
            return {'action': 'shoot', 'confidence': 0.75}

        # Dribble
        if right_wrist.get('y', 0) > right_hip.get('y', 0):
            return {'action': 'dribble', 'confidence': 0.65}

        return {'action': 'idle', 'confidence': 0.6}

# =================================================================
# MAIN PIPELINE
# =================================================================

class CVAIPipeline:
    def __init__(self):
        self.pose_analyzer = YOLOPoseAnalyzer()
        self.action_recognizer = ActionRecognizer()
        print("✅ CV AI Pipeline v5.0 (YOLOv8 Pose)", flush=True)

    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """Analisi frame singolo"""
        pose = self.pose_analyzer.extract_pose(frame)
        action = self.action_recognizer.predict_action(pose)
        form = self.pose_analyzer.analyze_shooting_form(pose)

        return {
            'pose': pose,
            'action': action,
            'form_analysis': form
        }

# =================================================================
# EXPORTS
# =================================================================

def analyze_frame_ai(frame: np.ndarray) -> Dict:
    pipeline = CVAIPipeline()
    return pipeline.analyze_frame(frame)

__version__ = "5.0.0"
print(f"✅ YOLOv8 Pose CV AI v{__version__} loaded!", flush=True)
