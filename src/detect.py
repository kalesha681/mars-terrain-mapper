from ultralytics import YOLO
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# Load pretrained YOLOv8 model (already downloaded)
model = YOLO("yolov8n.pt")

# Use ultralytics built-in test image (no internet needed)
from ultralytics.utils import ASSETS
test_image = str(ASSETS / "bus.jpg")

# Run inference
results = model(test_image)

# Show results
img = results[0].plot()
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 7))
plt.imshow(img_rgb)
plt.title("YOLOv8 Inference — Confirming Pipeline Works")
plt.axis("off")
plt.tight_layout()
plt.show()

print(f"Objects detected: {len(results[0].boxes)}")
for box in results[0].boxes:
    cls = results[0].names[int(box.cls)]
    conf = float(box.conf)
    print(f"  → {cls}: {conf:.2f} confidence")