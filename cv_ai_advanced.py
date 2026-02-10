from ultralytics import YOLO
model = YOLO('yolov8n-pose.pt')  # Auto-download 6MB

def extract_pose_yolo(frame):
    """Pose analysis con YOLOv8"""
    results
