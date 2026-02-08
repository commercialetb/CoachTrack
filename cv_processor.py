# =================================================================
# COACHTRACK ELITE AI v3.0 - PROCESSING MODULE
# Main Processing Pipeline
# =================================================================

import cv2
import numpy as np
from typing import Dict, List, Optional
import time
import json
from datetime import datetime

from cv_camera import CameraManager, CourtCalibrator
from cv_tracking import PlayerDetector, SimpleTracker, BallDetector


class CoachTrackVisionProcessor:
    """Pipeline principale Computer Vision"""

    def __init__(self, camera_source: str = "0", output_mode: str = "json"):
        """
        Args:
            camera_source: '0' USB, 'rtsp://...' WiFi, 'video.mp4' file
            output_mode: 'json' per solo dati, 'video' per video annotato
        """
        self.camera = CameraManager(camera_source)
        self.detector = PlayerDetector(model_path="yolov8n.pt", conf_threshold=0.5)
        self.tracker = SimpleTracker(max_age=30, min_hits=3)
        self.ball_detector = BallDetector()
        self.calibrator = CourtCalibrator()

        self.output_mode = output_mode
        self.is_calibrated = False
        self.running = False

        # Stats
        self.frame_count = 0
        self.start_time = None
        self.fps = 0

    def initialize(self, calibration_file: Optional[str] = "camera_calibration.json") -> bool:
        """Inizializza sistema"""
        print("🚀 Inizializzazione CoachTrack Vision...")

        # Connetti camera
        if not self.camera.connect():
            return False

        # Carica calibrazione
        if calibration_file:
            self.is_calibrated = self.camera.load_calibration(calibration_file)

        print("✅ Sistema pronto")
        return True

    def calibrate_court(self) -> bool:
        """Esegue calibrazione campo"""
        print("📐 Avvio calibrazione campo...")

        ret, frame = self.camera.read_frame()
        if not ret:
            print("❌ Errore lettura frame per calibrazione")
            return False

        # Undistort frame
        frame_undist = self.camera.undistort_frame(frame)

        # Calibra
        homography = self.calibrator.calibrate_from_frame(frame_undist)

        if homography is not None:
            self.camera.homography_matrix = homography

            # Salva calibrazione
            self.calibrator.save_calibration(
                self.camera.camera_matrix,
                self.camera.dist_coeffs,
                homography
            )

            self.is_calibrated = True
            print("✅ Calibrazione completata")
            return True

        return False

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Processa singolo frame
        Returns: Dizionario con tracking data
        """
        process_start = time.time()

        # 1. Undistort
        frame_undist = self.camera.undistort_frame(frame)

        # 2. Detect players
        detections = self.detector.detect(frame_undist)

        # 3. Track players
        tracks = self.tracker.update(detections)

        # 4. Detect ball
        ball = self.ball_detector.detect(frame_undist)

        # 5. Transform to court coordinates
        court_positions = []
        if self.is_calibrated and self.camera.homography_matrix is not None:
            for track in tracks:
                court_pos = self._image_to_court(track['center'])
                court_positions.append({
                    'player_id': track['track_id'],
                    'x': float(court_pos[0]),
                    'y': float(court_pos[1]),
                    'conf': track['conf']
                })

        # 6. Calculate metrics
        processing_time = time.time() - process_start

        return {
            'timestamp': time.time(),
            'frame_number': self.frame_count,
            'processing_time_ms': processing_time * 1000,
            'players': court_positions if self.is_calibrated else [],
            'ball': ball,
            'raw_tracks': tracks,
            'fps': self.fps
        }

    def _image_to_court(self, point: List[int]) -> np.ndarray:
        """Trasforma punto immagine a coordinate campo"""
        if self.camera.homography_matrix is None:
            return np.array(point)

        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.camera.homography_matrix)
        return transformed[0][0]

    def run_realtime(self, output_file: str = "tracking_data.json", 
                     visualize: bool = True, duration: int = 0):
        """
        Esegue processing real-time
        Args:
            output_file: File output per dati JSON
            visualize: Mostra finestra con video annotato
            duration: Durata in secondi (0 = infinito)
        """
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0

        all_data = []

        print(f"🎬 Avvio processing real-time...")
        print(f"📊 Output: {output_file}")
        print(f"👁️ Visualizzazione: {'ON' if visualize else 'OFF'}")
        print("Premi 'q' per fermare\n")

        try:
            while self.running:
                # Check duration
                if duration > 0 and (time.time() - self.start_time) > duration:
                    break

                # Read frame
                ret, frame = self.camera.read_frame()
                if not ret:
                    print("❌ Fine stream")
                    break

                # Process
                frame_data = self.process_frame(frame)
                all_data.append(frame_data)

                # Update stats
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                self.fps = self.frame_count / elapsed if elapsed > 0 else 0
                frame_data['fps'] = self.fps

                # Visualize
                if visualize:
                    annotated_frame = self._annotate_frame(frame, frame_data)
                    cv2.imshow('CoachTrack Vision', annotated_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n🛑 Stopped by user")
                        break

                # Print stats ogni 30 frames
                if self.frame_count % 30 == 0:
                    print(f"Frame {self.frame_count} | "
                          f"FPS: {self.fps:.1f} | "
                          f"Players: {len(frame_data['players'])} | "
                          f"Process: {frame_data['processing_time_ms']:.1f}ms")

        except KeyboardInterrupt:
            print("\n⚠️ Interrotto da tastiera")

        finally:
            # Save data
            self._save_tracking_data(all_data, output_file)

            # Cleanup
            if visualize:
                cv2.destroyAllWindows()
            self.camera.release()

            print(f"\n✅ Processing completato")
            print(f"📊 Frame processati: {self.frame_count}")
            print(f"⏱️ Durata: {time.time() - self.start_time:.1f}s")
            print(f"📈 FPS medio: {self.fps:.1f}")

    def _annotate_frame(self, frame: np.ndarray, data: Dict) -> np.ndarray:
        """Annota frame con bounding boxes e info"""
        annotated = frame.copy()

        # Draw tracks
        for track in data['raw_tracks']:
            bbox = track['bbox']
            track_id = track['track_id']
            conf = track['conf']

            # Bounding box
            cv2.rectangle(annotated, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         (0, 255, 0), 2)

            # Label
            label = f"ID:{track_id} {conf:.2f}"
            cv2.putText(annotated, label, 
                       (bbox[0], bbox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw ball
        if data['ball']:
            center = data['ball']['center']
            radius = data['ball']['radius']
            cv2.circle(annotated, tuple(center), radius, (0, 0, 255), 3)

        # Draw stats
        stats_text = [
            f"Frame: {data['frame_number']}",
            f"FPS: {data['fps']:.1f}",
            f"Players: {len(data['players'])}",
            f"Process: {data['processing_time_ms']:.1f}ms"
        ]

        y_offset = 30
        for text in stats_text:
            cv2.putText(annotated, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30

        return annotated

    def _save_tracking_data(self, data: List[Dict], filename: str):
        """Salva tracking data su file JSON"""
        output = {
            'metadata': {
                'total_frames': len(data),
                'fps': self.fps,
                'duration_sec': time.time() - self.start_time if self.start_time else 0,
                'calibrated': self.is_calibrated,
                'generated_at': datetime.now().isoformat()
            },
            'frames': data
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"💾 Dati salvati: {filename}")

    def process_video_file(self, video_path: str, output_json: str, 
                          output_video: Optional[str] = None):
        """
        Processa video file offline
        Args:
            video_path: Path video input
            output_json: Path output JSON
            output_video: Path output video annotato (opzionale)
        """
        # Re-init camera con video file
        self.camera = CameraManager(video_path)
        if not self.camera.connect():
            return False

        # Load calibration
        self.camera.load_calibration()

        # Setup video writer se richiesto
        writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video, fourcc, self.camera.fps,
                                    (self.camera.frame_width, self.camera.frame_height))

        # Process
        self.run_realtime(output_file=output_json, visualize=False, duration=0)

        if writer:
            writer.release()

        return True


# =================================================================
# MAIN ENTRY POINT
# =================================================================

def main():
    """Entry point per testing standalone"""
    import argparse

    parser = argparse.ArgumentParser(description='CoachTrack Vision Processor')
    parser.add_argument('--source', default='0', help='Camera source (0, rtsp://..., video.mp4)')
    parser.add_argument('--calibrate', action='store_true', help='Run calibration')
    parser.add_argument('--output', default='tracking_data.json', help='Output JSON file')
    parser.add_argument('--duration', type=int, default=0, help='Duration in seconds (0=infinite)')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')

    args = parser.parse_args()

    # Initialize processor
    processor = CoachTrackVisionProcessor(camera_source=args.source)

    if not processor.initialize():
        print("❌ Initialization failed")
        return

    # Calibration if requested
    if args.calibrate:
        if not processor.calibrate_court():
            print("❌ Calibration failed")
            return

    # Run processing
    processor.run_realtime(
        output_file=args.output,
        visualize=not args.no_viz,
        duration=args.duration
    )


if __name__ == "__main__":
    main()
