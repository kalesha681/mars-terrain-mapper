# Mars Terrain Mapper

Real-time geological feature detection and mapping pipeline built for ISRO ASCEND Round 2.

## Overview
Fine-tuned YOLOv8 on NASA AI4Mars dataset to detect Mars surface features 
(soil, bedrock, sand, rock). EKF tracks detected features across frames 
to build a stable 2D survey map.

## Pipeline
```
NASA AI4Mars Images → YOLOv8 Detection → EKF Feature Tracking → Arena Map
```

## Results
*(to be updated as project progresses)*

## Dataset
NASA AI4Mars: https://data.nasa.gov/Space-Science/AI4Mars-A-Dataset-for-Terrain-Aware-Autonomous-Dr/cykx-2qix

## Tech Stack
- Python 3.13
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy / SciPy
- EKF (custom implementation)

## Author
Kalesha Shaik — RGUKT Nuzvid
ISRO ASCEND 2025 Team
