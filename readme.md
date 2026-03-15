# Mars Terrain Mapper 🚀

Real-time geological feature detection and mapping pipeline 
built for ISRO ASCEND Round 2 — autonomous planetary survey drone.

## Overview
Fine-tuned YOLOv8n on Mars rock detection dataset to identify 
geological features from aerial drone footage. EKF (Extended Kalman Filter) 
tracks detected feature positions across frames to build a stable 
2D survey map despite IMU drift and sensor noise.

Built entirely on CPU (Intel i5-13600) — no GPU required.

## Pipeline
```
Drone Camera Feed
      ↓
YOLOv8n (fine-tuned on Mars imagery)
      ↓
Rock/Feature Detection (bounding boxes + confidence)
      ↓
EKFTracker (IMU + Optical Flow fusion)
      ↓
Stable 3D feature coordinates
      ↓
2D Arena Survey Map
```

## Results
| Component | Metric | Value |
|---|---|---|
| YOLOv8n Detection | mAP50 | 0.593 (30 epochs, CPU) |
| EKF Position Estimation | RMSE | 0.156 m |
| EKF vs Dead Reckoning | Improvement | 95.2% |
| Inference Speed | CPU | ~52ms/frame |

## Dataset
- **Source:** Mars Computer Vision Dataset (Roboflow)
- **Images:** 593 Mars surface images (640×640)
- **Split:** 193 train / 60 valid / 29 test
- **Classes:** Mars rock detection

## Tech Stack
- Python 3.13
- YOLOv8 (Ultralytics 8.4.21)
- OpenCV 4.13
- NumPy / SciPy
- Custom EKF implementation (from scratch)

## Key Components

### 1. EKF Sensor Fusion (`src/ekf_tracker.py`)
Custom Extended Kalman Filter fusing IMU acceleration 
and optical flow velocity measurements.
- Reduces position error by 95.2% vs dead reckoning
- RMSE: 0.156m over 10-second simulated flight

### 2. Mars Feature Detection (`src/train.py`, `src/detect.py`)
YOLOv8n fine-tuned on Mars surface imagery.
- Detects geological features in real-time
- Deployable on Jetson Nano with TensorRT optimization

## Hardware Target
- **Drone:** Pixhawk 6x + Jetson Nano
- **Camera:** Intel RealSense D series (RGB + Depth)
- **Base Station:** Jetson Orin NX
- **Competition:** ISRO ASCEND 2025 (Qualified Round 1)

## Author
**Kalesha Shaik** — RGUKT Nuzvid, EEE (2023–2027)  
ISRO ASCEND Team | Aerial Robotics Researcher  
📧 kalesha681@gmail.com | 
🔗 [LinkedIn](https://linkedin.com/in/kalesha681) | 
💻 [GitHub](https://github.com/kalesha681)