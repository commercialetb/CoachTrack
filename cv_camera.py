# =================================================================
# COACHTRACK ELITE AI v3.0 - CAMERA MODULE
# Action Cam / PTZ Camera Integration
# =================================================================

import cv2
import numpy as np
import time
from typing import Optional, Dict, Tuple
import json

class CameraManager:
    """Gestisce connessione e acquisizione da Action Cam o PTZ"""

    def __init__(self, source: str = "0", camera_type: str = "actioncam"):
        """
        Args:
            source: '0' per USB webcam, 'rtsp://ip/stream' per WiFi, o path video file
            camera_type: 'actioncam' o 'ptz'
        """
        self.source = source if source.startswith('rtsp') or source.endswith('.mp4') else int(source)
        self.camera_type = camera_type
        self.cap = None
        self.fps = 30
        self.frame_width = 3840  # 4K
        self.frame_height = 2160

        # Parametri calibrazione (aggiornati dopo calibrazione)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.homography_matrix = None

    def connect(self) -> bool:
        """Connette alla camera"""
        try:
            self.cap = cv2.VideoCapture(self.source)

            if not self.cap.isOpened():
                print(f"❌ Errore apertura camera: {self.source}")
                return False

            # Imposta risoluzione 4K se disponibile
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Leggi parametri effettivi
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            print(f"✅ Camera connessa: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
            return True

        except Exception as e:
            print(f"❌ Errore connessione: {e}")
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Legge frame dalla camera"""
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        return ret, frame

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """Corregge distorsione wide-angle"""
        if self.camera_matrix is None or self.dist_coeffs is None:
            return frame

        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

    def load_calibration(self, calibration_file: str = "camera_calibration.json"):
        """Carica parametri calibrazione da file"""
        try:
            with open(calibration_file, 'r') as f:
                calib = json.load(f)

            self.camera_matrix = np.array(calib['camera_matrix'])
            self.dist_coeffs = np.array(calib['dist_coeffs'])
            self.homography_matrix = np.array(calib['homography_matrix'])

            print(f"✅ Calibrazione caricata da {calibration_file}")
            return True
        except Exception as e:
            print(f"⚠️ Calibrazione non trovata, uso default: {e}")
            self._set_default_calibration()
            return False

    def _set_default_calibration(self):
        """Imposta calibrazione default per action cam wide-angle 170°"""
        # Parametri tipici GoPro/DJI Osmo Action
        fx = fy = self.frame_width * 0.9
        cx, cy = self.frame_width / 2, self.frame_height / 2

        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Distorsione tipica wide-angle (k1, k2, p1, p2, k3)
        self.dist_coeffs = np.array([-0.3, 0.1, 0, 0, 0], dtype=np.float32)

    def release(self):
        """Rilascia risorse camera"""
        if self.cap:
            self.cap.release()
            print("🔌 Camera disconnessa")


class CourtCalibrator:
    """Calibra campo basket per homography transform"""

    def __init__(self):
        self.court_points_image = []  # 4 punti nell'immagine
        self.court_points_real = np.array([
            [0, 0],      # Angolo in basso a sinistra
            [28, 0],     # Angolo in basso a destra (28m larghezza campo FIBA)
            [28, 15],    # Angolo in alto a destra (15m lunghezza)
            [0, 15]      # Angolo in alto a sinistra
        ], dtype=np.float32)

    def calibrate_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Calibra campo da un singolo frame
        User deve cliccare 4 angoli del campo nell'ordine: BL, BR, TR, TL
        """
        self.court_points_image = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(self.court_points_image) < 4:
                self.court_points_image.append([x, y])
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                cv2.imshow('Calibrazione Campo', frame)

        cv2.namedWindow('Calibrazione Campo')
        cv2.setMouseCallback('Calibrazione Campo', mouse_callback)
        cv2.imshow('Calibrazione Campo', frame)

        print("📍 Clicca 4 angoli campo: 1)Basso-SX 2)Basso-DX 3)Alto-DX 4)Alto-SX")
        print("Premi 'q' quando finito")

        while len(self.court_points_image) < 4:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        if len(self.court_points_image) == 4:
            pts_img = np.array(self.court_points_image, dtype=np.float32)
            homography = cv2.getPerspectiveTransform(pts_img, self.court_points_real)
            print("✅ Homography calcolata")
            return homography
        else:
            print("❌ Calibrazione fallita: servono 4 punti")
            return None

    def auto_detect_court(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Rileva automaticamente campo da linee bianche"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Rileva linee
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                minLineLength=100, maxLineGap=10)

        if lines is None:
            print("⚠️ Nessuna linea rilevata")
            return None

        # TODO: Logica per identificare 4 angoli da linee rilevate
        # Per ora usa calibrazione manuale
        print("⚠️ Auto-detect non implementato, usa calibrazione manuale")
        return None

    def save_calibration(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                        homography: np.ndarray, filename: str = "camera_calibration.json"):
        """Salva calibrazione su file"""
        calib = {
            'camera_matrix': camera_matrix.tolist(),
            'dist_coeffs': dist_coeffs.tolist(),
            'homography_matrix': homography.tolist()
        }

        with open(filename, 'w') as f:
            json.dump(calib, f, indent=2)

        print(f"💾 Calibrazione salvata: {filename}")
