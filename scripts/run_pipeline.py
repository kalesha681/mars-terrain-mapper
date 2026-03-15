"""
Full end-to-end pipeline demo.
Simulates autonomous Mars arena survey:
    1. Load trained YOLOv8 detector
    2. Run detection on sample images
    3. Feed detections into EKF tracker
    4. Build and save arena map
"""
import sys
sys.path.insert(0, '.')

import os
import yaml
import numpy as np
from ultralytics import YOLO
from src.ekf_tracker import EKFTracker
from src.mapper import ArenaMapper

def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def run_pipeline():
    print("=" * 50)
    print("  Mars Terrain Mapper — Full Pipeline Demo")
    print("=" * 50)

    # ── Load config ───────────────────────────────────
    cfg = load_config()
    print(f"\n[1/4] Config loaded")

    # ── Load trained model ────────────────────────────
    weights = cfg['model']['trained_weights']
    if not os.path.exists(weights):
        print(f"[ERROR] Trained weights not found at {weights}")
        print("Run scripts/train.py first.")
        return

    model = YOLO(weights)
    print(f"[2/4] Model loaded: {weights}")

    # ── Run detection on test images ──────────────────
    test_dir = "data/raw/mars-dataset/test/images"
    if not os.path.exists(test_dir):
        print(f"[ERROR] Test images not found at {test_dir}")
        return

    images = [os.path.join(test_dir, f)
              for f in os.listdir(test_dir)
              if f.endswith(('.jpg', '.png'))][:10]  # first 10

    print(f"[3/4] Running detection on {len(images)} test images...")

    # ── Initialize mapper ─────────────────────────────
    mapper_cfg = cfg['mapper']
    mapper = ArenaMapper(
        arena_size=mapper_cfg['arena_size'],
        cell_size=mapper_cfg['cell_size']
    )

    # Simulate drone moving across arena
    positions = [(x, y)
                 for y in np.linspace(-2, 2, len(images))
                 for x in [np.random.uniform(-2, 2)]]

    total_detections = 0
    for i, (img_path, (dx, dy)) in enumerate(zip(images, positions)):
        results = model(img_path,
                       conf=cfg['model']['confidence_threshold'],
                       verbose=False)
        n_det = len(results[0].boxes)
        total_detections += n_det

        # Add each detection to mapper
        for box in results[0].boxes:
            conf = float(box.conf)
            mapper.add_detection(dx, dy, conf)
        mapper.drone_path.append((dx, dy))

        print(f"  Image {i+1}/{len(images)}: "
              f"{n_det} detections at pos ({dx:.2f}, {dy:.2f})")

    # ── Generate map ──────────────────────────────────
    print(f"\n[4/4] Generating arena map...")
    mapper.stats()
    mapper.show(title="Mars Arena Survey — Full Pipeline Demo")

    print("\n Pipeline complete!")
    print(f"   Total detections: {total_detections}")
    print(f"   Map saved to: results/plots/arena_map.png")

if __name__ == "__main__":
    run_pipeline()