"""
Dataset Exploration Script for VisDrone Dataset
Purpose: Analyze dataset structure and visualize sample images
"""

import os
import cv2
import matplotlib.pyplot as plt

# ============================================
# 1. Dataset Path
# ============================================
dataset_path = "archive/VisDrone_Dataset/VisDrone2019-DET-train/images"

# Check if path exists
if not os.path.exists(dataset_path):
    print(f"❌ Path not found: {dataset_path}")
    exit()

# ============================================
# 2. Count total images
# ============================================
all_images = os.listdir(dataset_path)
print(f"📊 Total images in training set: {len(all_images)}")
print(f"📸 First 5 images: {all_images[:5]}")

# ============================================
# 3. Display sample images
# ============================================
os.makedirs("output_exploration", exist_ok=True)

sample_images = all_images[:6]  # First 6 images

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, img_name in enumerate(sample_images):
    img_path = os.path.join(dataset_path, img_name)
    
    # Read image
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"⚠️ Could not read: {img_name}")
        continue
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get dimensions
    height, width = img.shape[:2]
    
    # Plot
    row = idx // 3
    col = idx % 3
    axes[row, col].imshow(img_rgb)
    axes[row, col].set_title(f"{img_name}\nSize: {width}x{height}")
    axes[row, col].axis('off')
    
    print(f"✅ Image {idx+1}: {img_name} - Size: {width}x{height}")

plt.tight_layout()
plt.savefig("output_exploration/sample_images.png", dpi=100, bbox_inches='tight')
print("\n✅ Sample visualization saved: output_exploration/sample_images.png")
plt.close()

# ============================================
# 4. Dataset Info
# ============================================
print("""
✅ Dataset Exploration Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 Dataset: VisDrone (Drone/Aerial Images)
📌 Task: Detect humans and cars
📌 Image Types: Drone/aerial view photographs
📌 Total Training Images: """ + str(len(all_images)) + """
📌 Classes: Pedestrian, People, Bicycle, Car, Van, Truck, etc.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print("✅ Step 1 (Dataset Exploration) Complete!")