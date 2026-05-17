"""
Quick Detection using Pre-trained YOLOv8
Skip training, use pretrained model directly
"""

from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path

print("🚀 Starting Quick Detection with Pre-trained Model...")

# Use pretrained model (no training needed)
model = YOLO('yolov8n.pt')

print("✅ Pre-trained model loaded!")

# Detection path
test_images_path = "archive/VisDrone_Dataset/VisDrone2019-DET-test-dev/images"

# Get first 10 images for quick demo
test_images = os.listdir(test_images_path)[:10]

os.makedirs("detection_results", exist_ok=True)

print("\n🔍 Running Detection on Test Images...")

total_humans = 0
total_cars = 0

for idx, img_name in enumerate(test_images, 1):
    img_path = os.path.join(test_images_path, img_name)
    img = cv2.imread(img_path)
    
    if img is None:
        continue
    
    # Run detection
    results = model(img, conf=0.5)
    
    # Count and draw
    annotated_img = img.copy()
    human_count = 0
    car_count = 0
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            
            # Classes: 0=person, 2=car, etc.
            if class_id == 0:  # person
                human_count += 1
                color = (0, 255, 0)  # Green
                label = f"Human {confidence:.2f}"
            elif class_id == 2:  # car
                car_count += 1
                color = (0, 0, 255)  # Red
                label = f"Car {confidence:.2f}"
            else:
                continue
            
            # Draw box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_img, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    total_humans += human_count
    total_cars += car_count
    
    # Add count text
    cv2.putText(annotated_img, f"Humans: {human_count}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_img, f"Cars: {car_count}", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Save result
    output_path = os.path.join("detection_results", f"detected_{img_name}")
    cv2.imwrite(output_path, annotated_img)
    
    print(f"✅ {idx}. {img_name}: {human_count} humans, {car_count} cars")

print(f"\n📊 Total Summary:")
print(f"   Total Humans Detected: {total_humans}")
print(f"   Total Cars Detected: {total_cars}")
print(f"   Detection Results Saved: detection_results/")

# Visualize results
results_images = os.listdir("detection_results")[:6]
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, img_name in enumerate(results_images):
    ax = axes[idx // 3, idx % 3]
    img = cv2.imread(os.path.join("detection_results", img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_title(img_name, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig("detection_results/summary.png", dpi=100, bbox_inches='tight')
print("✅ Summary visualization saved: detection_results/summary.png")
plt.close()

print("\n✅ Detection Complete!")