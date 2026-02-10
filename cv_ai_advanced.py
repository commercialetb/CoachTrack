# =================================================================
# COACHTRACK ELITE AI v5.0 - YOLOv8 Pose (NO MediaPipe!)
# =================================================================

import cv2
import numpy as np
from ultralytics import YOLO
from scipy.signal import butter, filtfilt

class CVAIPipeline:
    """YOLOv8 Pose + Action Recognition per basket indoor"""

    def __init__(self):
        self.model = None
        self.pose_history = []

    def initialize(self):
        """Carica YOLOv8n-pose (11MB)"""
        try:
            self.model = YOLO('yolov8n-pose.pt')
            return True
        except Exception as e:
            print(f"❌ YOLOv8 init error: {e}")
            return False

    def process_frame(self, frame):
        """
        Process frame con YOLOv8 Pose
        Returns: dict con pose keypoints + action
        """
        if self.model is None:
            return None

        try:
            results = self.model(frame, verbose=False)

            if len(results) == 0 or results[0].keypoints is None:
                return None

            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            confidence = results[0].keypoints.conf[0].cpu().numpy()

            # YOLOv8 keypoints (17 COCO format):
            # 0: nose, 5: left_shoulder, 6: right_shoulder
            # 7: left_elbow, 8: right_elbow
            # 9: left_wrist, 10: right_wrist
            # 11: left_hip, 12: right_hip
            # 13: left_knee, 14: right_knee
            # 15: left_ankle, 16: right_ankle

            pose_data = {
                'nose': keypoints[0],
                'left_shoulder': keypoints[5],
                'right_shoulder': keypoints[6],
                'left_elbow': keypoints[7],
                'right_elbow': keypoints[8],
                'left_wrist': keypoints[9],
                'right_wrist': keypoints[10],
                'left_hip': keypoints[11],
                'right_hip': keypoints[12],
                'left_knee': keypoints[13],
                'right_knee': keypoints[14],
                'left_ankle': keypoints[15],
                'right_ankle': keypoints[16],
                'confidence': confidence
            }

            # Analisi shooting form
            shooting_metrics = self._analyze_shooting_form(pose_data)

            # Action recognition
            action = self._detect_action(pose_data)

            result = {
                'pose': pose_data,
                'shooting_form': shooting_metrics,
                'action': action
            }

            self.pose_history.append(result)
            if len(self.pose_history) > 60:  # Keep last 2s @ 30fps
                self.pose_history.pop(0)

            return result

        except Exception as e:
            print(f"⚠️ Frame processing error: {e}")
            return None

    def _analyze_shooting_form(self, pose_data):
        """Analizza forma di tiro (angoli gomito/ginocchia)"""
        try:
            # Calcola angolo gomito destro
            shoulder = pose_data['right_shoulder']
            elbow = pose_data['right_elbow']
            wrist = pose_data['right_wrist']

            elbow_angle = self._calculate_angle(shoulder, elbow, wrist)

            # Calcola angolo ginocchio destro
            hip = pose_data['right_hip']
            knee = pose_data['right_knee']
            ankle = pose_data['right_ankle']

            knee_angle = self._calculate_angle(hip, knee, ankle)

            # Valutazione
            form_score = 0
            if 70 <= elbow_angle <= 110:  # Follow-through ottimale
                form_score += 50
            if 90 <= knee_angle <= 140:  # Flessione gambe corretta
                form_score += 50

            return {
                'elbow_angle': elbow_angle,
                'knee_angle': knee_angle,
                'form_score': form_score
            }
        except:
            return {'elbow_angle': 0, 'knee_angle': 0, 'form_score': 0}

    def _calculate_angle(self, p1, p2, p3):
        """Calcola angolo tra 3 punti"""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    def _detect_action(self, pose_data):
        """Riconosce azione: shoot/dribble/idle"""
        try:
            wrist_y = pose_data['right_wrist'][1]
            shoulder_y = pose_data['right_shoulder'][1]
            hip_y = pose_data['right_hip'][1]

            # Shooting: polso sopra spalla
            if wrist_y < shoulder_y - 30:
                return 'shooting'

            # Dribbling: polso vicino a anche
            if abs(wrist_y - hip_y) < 50:
                return 'dribbling'

            return 'idle'
        except:
            return 'unknown'

    def draw_skeleton(self, frame, result):
        """Disegna skeleton su frame"""
        if result is None or 'pose' not in result:
            return frame

        pose = result['pose']

        # Connessioni skeleton
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

        # Draw connections
        for conn in connections:
            pt1 = tuple(pose[conn[0]].astype(int))
            pt2 = tuple(pose[conn[1]].astype(int))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Draw keypoints
        for key in ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 
                    'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 
                    'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']:
            pt = tuple(pose[key].astype(int))
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)

        # Draw action
        action = result.get('action', 'unknown')
        cv2.putText(frame, f"Action: {action}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw shooting form
        if 'shooting_form' in result:
            form = result['shooting_form']
            cv2.putText(frame, f"Form: {form['form_score']}/100", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        return frame

    def get_session_stats(self):
        """Statistiche sessione"""
        if not self.pose_history:
            return {}

        actions = [r.get('action', 'unknown') for r in self.pose_history]
        shooting_frames = [r for r in self.pose_history if r.get('action') == 'shooting']

        stats = {
            'total_frames': len(self.pose_history),
            'shooting_frames': len(shooting_frames),
            'dribbling_frames': actions.count('dribbling'),
            'avg_form_score': np.mean([r['shooting_form']['form_score'] 
                                       for r in shooting_frames]) if shooting_frames else 0
        }

        return stats

# Test status
def get_cv_status():
    """Check se tutto funziona"""
    status = {
        'yolov8': False,
        'scipy': False
    }

    try:
        from ultralytics import YOLO
        status['yolov8'] = True
    except:
        pass

    try:
        import scipy
        status['scipy'] = True
    except:
        pass

    return status

if __name__ == "__main__":
    print("🔧 CV AI Status:")
    status = get_cv_status()
    for key, val in status.items():
        print(f"   - {key}: {'✅' if val else '❌'}")

    if status['yolov8']:
        pipeline = CVAIPipeline()
        if pipeline.initialize():
            print("✅ YOLOv8 Pose caricato!")
        else:
            print("❌ Errore caricamento YOLOv8")
