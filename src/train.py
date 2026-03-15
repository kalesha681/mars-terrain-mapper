from ultralytics import YOLO

# Load pretrained YOLOv8 nano model (already downloaded)
model = YOLO("yolov8n.pt")

# Train on Mars dataset
results = model.train(
    data="data/raw/mars-dataset/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    name="mars_detector",
    project="results",
    device="cpu",
    patience=10,
    verbose=True
)

print("\nTraining complete!")
print(f"Best mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")