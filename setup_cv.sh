#!/bin/bash
# Setup script per Computer Vision module

echo "🚀 Installing CoachTrack Elite CV Dependencies..."

# Update pip
pip install --upgrade pip

# Install requirements
pip install -r requirements_cv.txt

# Download YOLO model
echo "📦 Downloading YOLOv8 nano model..."
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

echo "✅ Setup completato!"
echo ""
echo "📝 Per avviare:"
echo "   python cv_processor.py --source 0 --calibrate"
