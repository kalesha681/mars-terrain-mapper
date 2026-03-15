# 🚀 Mars Terrain Mapper

![Python](https://img.shields.io/badge/Python-3.13-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![ISRO](https://img.shields.io/badge/ISRO-ASCEND_2025-red)

> Real-time Mars geological feature detection and arena mapping pipeline  
> Built for **ISRO ASCEND Round 2** — autonomous planetary survey drone

---

## 📽️ Demo Output

### Arena Survey Map
![Arena Map](results/plots/arena_map.png)

*Lawnmower survey pattern with EKF-tracked feature detections.  
Left: Detection density heatmap. Right: Drone path with confidence-coded detections.*

---

## 🏗️ System Architecture
```
Drone Camera (Intel RealSense)
        ↓
YOLOv8n — Fine-tuned on Mars imagery
        ↓
Feature Detection (bounding boxes + confidence)
        ↓
EKF Tracker — IMU + Optical Flow fusion
        ↓
Stable 3D world-frame coordinates
        ↓
ArenaMapper — 2D survey map generation
```

---

## 📊 Results

| Component | Metric | Value |
|---|---|---|
| YOLOv8n Detection | mAP50 | **0.593** |
| YOLOv8n Detection | mAP50-95 | **0.337** |
| EKF Position Estimation | RMSE | **0.156 m** |
| EKF vs Dead Reckoning | Improvement | **95.2%** |
| Inference Speed (CPU) | Per Frame | **70ms** |
| Training Time (CPU) | 30 epochs | **0.461 hrs** |

---

## 🛰️ Competition Context

This project was built for **ISRO ASCEND 2025** (Autonomous Systems Challenge for Engineering and Navigation Drones).

- ✅ Qualified Round 1
- 🔄 Round 2: Autonomous Mars terrain survey with feature detection
- **Hardware:** Pixhawk 6x + Jetson Nano + Intel RealSense
- **Base Station:** Jetson Orin NX + GStreamer video pipeline

---

## 📁 Project Structure
```
mars-terrain-mapper/
├── src/
│   ├── ekf_tracker.py        # EKF sensor fusion (IMU + optical flow)
│   ├── train.py              # YOLOv8 fine-tuning on Mars dataset
│   ├── detect.py             # Real-time inference pipeline
│   ├── mapper.py             # 2D arena survey map generation
│   └── download_dataset.py   # Dataset setup via Roboflow
├── data/
│   └── raw/                  # Mars detection dataset (not tracked)
├── models/
│   └── weights/              # Trained model weights (not tracked)
├── results/
│   └── plots/                # Output visualizations
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup
```bash
# Clone the repo
git clone https://github.com/kalesha681/mars-terrain-mapper.git
cd mars-terrain-mapper

# Install dependencies
pip install -r requirements.txt

# Download dataset
python src/download_dataset.py

# Train model
python src/train.py

# Run mapper simulation
python src/mapper.py
```

---

## 🧠 Key Components

### 1. Extended Kalman Filter (`src/ekf_tracker.py`)
Custom EKF implementation from scratch fusing IMU acceleration and optical flow velocity.
- **State vector:** `[position, velocity]`
- **95.2% RMSE reduction** vs dead reckoning baseline
- Reusable class — drop-in for real RealSense + Pixhawk pipeline

### 2. Mars Feature Detector (`src/train.py`)
YOLOv8n fine-tuned on Mars surface rock detection dataset.
- **Dataset:** 593 Mars surface images via Roboflow
- **mAP50: 0.593** after 30 epochs on CPU
- Deployable on Jetson Nano with TensorRT optimization

### 3. Arena Mapper (`src/mapper.py`)
Builds real-time 2D survey map from detections + EKF position estimates.
- Lawnmower coverage pattern simulation
- Detection density heatmap
- Confidence-coded scatter map with drone trajectory

---

## 🔧 Hardware Stack

| Component | Hardware |
|---|---|
| Flight Controller | Pixhawk 6x |
| Onboard Computer | Jetson Nano |
| Camera | Intel RealSense D series |
| Base Station | Jetson Orin NX |
| Comms | UDP + GStreamer |
| Competition | ISRO ASCEND 2025 |

---

## 📦 Dependencies
```
ultralytics==8.4.21
opencv-python==4.13.0
torch==2.10.0
numpy==2.4.3
matplotlib==3.10.8
scipy==1.17.1
roboflow
```

---

## 👤 Author

**Kalesha Shaik**  
Undergraduate Researcher — Aerial Robotics & Autonomous Drones  
RGUKT Nuzvid, EEE (2023–2027)

📧 kalesha681@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/kalesha681)  
💻 [GitHub](https://github.com/kalesha681)

---

## 📄 License

MIT License — feel free to use and build on this work.