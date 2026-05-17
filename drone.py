# Test if all required libraries are installed correctly

print("✅ Python working!")

import cv2
print("✅ OpenCV installed!")

import numpy as np
print("✅ NumPy installed!")

from ultralytics import YOLO
print("✅ YOLO/Ultralytics installed!")

import torch
print("✅ PyTorch installed!")

print("\n🎉 All libraries installed successfully!")
print(f"PyTorch is using: {'GPU' if torch.cuda.is_available() else 'CPU'}")