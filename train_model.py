"""
YOLO Model Training
- Train YOLOv8 on VisDrone dataset
- Detect humans and cars
"""

from ultralytics import YOLO
import os

print("🚀 Starting YOLO Model Training...")

# Load pretrained YOLOv8 nano model (fastest for CPU)
model = YOLO('yolov8n.pt')

print("✅ Model loaded!")

# Train the model
results = model.train(
    data='data.yaml',
    epochs=5,          # Start with 5 epochs (quick training)
    imgsz=640,         # Image size
    batch=8,           # Batch size (smaller for CPU)
    patience=3,        # Early stopping
    device='cpu',          # GPU if available, CPU otherwise
    save=True,
    verbose=True
)

print("✅ Training complete!")
print(f"Best model saved at: runs/detect/train/weights/best.pt")