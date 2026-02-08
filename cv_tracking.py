# =================================================================
# COACHTRACK ELITE AI v3.0 - TRACKING MODULE
# YOLO Detection + ByteTrack Multi-Object Tracking
# =================================================================

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import time

# Placeholder per import condizionali (installare se necessario)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Ultralytics YOLO non installato. Usa: pip install ultralytics")


class PlayerDetector:
    """Rileva giocatori usando YOLO"""

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5):
        """
        Args:
            model_path: Path al modello YOLO (yolov8n.pt, yolov8s.pt, custom trained)
            conf_threshold: Soglia confidenza minima
        """
        self.conf_threshold = conf_threshold
        self.model = None
        self.model_loaded = False

        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                self.model_loaded = True
                print(f"✅ YOLO model caricato: {model_path}")
            except Exception as e:
                print(f"❌ Errore caricamento YOLO: {e}")
        else:
            print("⚠️ Usando detector mock (per testing)")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Rileva persone nel frame
        Returns: Lista di detections [{bbox, conf, class_id}, ...]
        """
        if not self.model_loaded:
            return self._mock_detections(frame)

        # YOLO inference
        results = self.model(frame, conf=self.conf_threshold, classes=[0])  # class 0 = person

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                class_id = int(box.cls[0])

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'conf': conf,
                    'class_id': class_id,
                    'center': [int((x1+x2)/2), int((y1+y2)/2)]
                })

        return detections

    def _mock_detections(self, frame: np.ndarray) -> List[Dict]:
        """Detections fake per testing senza YOLO"""
        h, w = frame.shape[:2]
        # Simula 10 giocatori random
        detections = []
        for i in range(10):
            x = np.random.randint(w//4, 3*w//4)
            y = np.random.randint(h//4, 3*h//4)
            bbox = [x-50, y-100, x+50, y+100]
            detections.append({
                'bbox': bbox,
                'conf': 0.85,
                'class_id': 0,
                'center': [x, y]
            })
        return detections


class SimpleTracker:
    """Tracker semplificato per assegnare ID persistenti"""

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Args:
            max_age: Frame max senza detection prima di eliminare track
            min_hits: Hit minimi prima di confermare track
            iou_threshold: IoU minimo per associare detection a track
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks = []  # Lista di tracks attivi
        self.next_id = 1
        self.frame_count = 0

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Aggiorna tracks con nuove detections
        Returns: Lista tracks con ID [{track_id, bbox, ...}, ...]
        """
        self.frame_count += 1

        # Se non ci sono tracks, inizializzali
        if len(self.tracks) == 0 and len(detections) > 0:
            for det in detections:
                self._initiate_track(det)
            return self._get_confirmed_tracks()

        # Associa detections a tracks esistenti
        if len(detections) > 0:
            self._associate_detections_to_tracks(detections)

        # Rimuovi tracks vecchi
        self.tracks = [t for t in self.tracks if t['age'] < self.max_age]

        return self._get_confirmed_tracks()

    def _initiate_track(self, detection: Dict):
        """Inizia nuovo track"""
        track = {
            'track_id': self.next_id,
            'bbox': detection['bbox'],
            'center': detection['center'],
            'conf': detection['conf'],
            'age': 0,
            'hits': 1,
            'last_update': self.frame_count
        }
        self.tracks.append(track)
        self.next_id += 1

    def _associate_detections_to_tracks(self, detections: List[Dict]):
        """Associa detections a tracks con Hungarian algorithm semplificato"""
        # Calcola IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))

        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(track['bbox'], det['bbox'])

        # Greedy assignment (per semplicità, invece di Hungarian)
        matched_tracks = set()
        matched_dets = set()

        while np.max(iou_matrix) > self.iou_threshold:
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)

            # Update track
            self.tracks[i]['bbox'] = detections[j]['bbox']
            self.tracks[i]['center'] = detections[j]['center']
            self.tracks[i]['conf'] = detections[j]['conf']
            self.tracks[i]['age'] = 0
            self.tracks[i]['hits'] += 1
            self.tracks[i]['last_update'] = self.frame_count

            matched_tracks.add(i)
            matched_dets.add(j)
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0

        # Tracks non matchati: incrementa age
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track['age'] += 1

        # Detections non matchate: crea nuovi tracks
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._initiate_track(det)

    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calcola IoU tra due bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calcola intersezione
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calcola union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _get_confirmed_tracks(self) -> List[Dict]:
        """Ritorna solo tracks confermati (hits >= min_hits)"""
        return [t for t in self.tracks if t['hits'] >= self.min_hits]


class BallDetector:
    """Rileva palla basket"""

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Path modello custom per ball detection (opzionale)
        """
        self.model_loaded = False
        # TODO: Implementa con YOLO custom trained su basketball
        print("⚠️ Ball detector non implementato, usa mock")

    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        """Rileva palla nel frame"""
        # Mock detection
        h, w = frame.shape[:2]
        return {
            'center': [w//2, h//2],
            'radius': 20,
            'conf': 0.7
        }
