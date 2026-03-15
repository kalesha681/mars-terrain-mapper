"""
Evaluation script — runs trained model on test set
and reports detection metrics.
"""
import sys
sys.path.insert(0, '.')

from ultralytics import YOLO
import yaml

def evaluate(weights_path, data_yaml, conf=0.3):
    print(f"Loading model: {weights_path}")
    model = YOLO(weights_path)

    print("Running evaluation on test set...")
    metrics = model.val(
        data=data_yaml,
        split='test',
        conf=conf,
        verbose=True
    )

    print("\n── Evaluation Results ───────────────────")
    print(f"mAP50        : {metrics.box.map50:.4f}")
    print(f"mAP50-95     : {metrics.box.map:.4f}")
    print(f"Precision    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")
    print("─────────────────────────────────────────")
    return metrics

if __name__ == "__main__":
    evaluate(
        weights_path="runs/detect/results/mars_detector/weights/best.pt",
        data_yaml="data/raw/mars-dataset/data.yaml"
    )