# Robotic Bin-Picking System

[English](README.md) | [简体中文](README.zh-CN.md)

An intelligent robotic vision system for automated bin-picking tasks. This project implements a complete pipeline for 3D object detection, pose estimation, and grasp planning using RGB-D point cloud data.

## About This Project

This system addresses the challenge of automated object grasping in cluttered environments, a key problem in warehouse automation, manufacturing, and logistics. By combining traditional computer vision techniques with intelligent decision-making algorithms, it achieves real-time performance while maintaining high reliability.

### Problem Statement

In modern warehouses and manufacturing facilities, robots need to pick objects from bins containing multiple items in random orientations. The challenge includes:
- **Detection**: Identifying individual objects in cluttered 3D scenes
- **Pose Estimation**: Determining precise position and orientation of each object
- **Grasp Planning**: Selecting the optimal gripper and approach strategy
- **Collision Avoidance**: Ensuring the gripper doesn't collide with bin walls

### Solution Approach

This project implements a modular pipeline that:
1. Processes RGB-D point cloud data to segment individual objects
2. Estimates 6D poses using Principal Component Analysis
3. Intelligently selects between finger and suction grippers based on object properties
4. Performs collision detection to ensure safe grasp execution

### Real-World Applications

- **E-commerce Fulfillment**: Automated order picking in warehouses
- **Manufacturing**: Parts feeding and assembly automation
- **Recycling**: Sorting mixed materials
- **Food Industry**: Handling irregular-shaped items

## Features

- **Point Cloud Processing**: Efficient downsampling, noise removal, and plane segmentation
- **Object Detection**: DBSCAN-based clustering for robust object segmentation
- **6D Pose Estimation**: PCA-based orientation estimation with bounding box analysis
- **Intelligent Gripper Selection**: Automatic selection between finger and suction grippers with collision avoidance
- **Modular Architecture**: Clean separation of concerns for easy customization and extension

## Performance

- **Latency**: ~110-140ms per detection cycle (CPU)
- **Throughput**: 7-9 fps
- **Optimization Potential**: <100ms with GPU acceleration (CUDA PCL)

## Project Structure

```
bin_picking/
├── src/
│   ├── __init__.py
│   ├── clustering.py          # Point cloud preprocessing and clustering
│   ├── pose_estimation.py     # 6D pose estimation
│   ├── grasp_planner.py       # Gripper selection logic
│   └── action_generator.py    # Complete detection pipeline
├── notebooks/
│   └── demo.ipynb             # Interactive demo
├── data/                       # Dataset 
├── README.md
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.7+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd bin_picking
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add your dataset to the `data/` directory:
```
data/
├── Color_3840x2160.png
├── pcd.ply
├── rgb_intrinsic.yaml
├── depth_intrinsic.yaml
└── rgb_distortion.yaml
```

## Quick Start

### Using the Python API

```python
import numpy as np
import open3d as o3d
from src import BinPickingSystem

# Initialize system
system = BinPickingSystem(
    voxel_size=2.0,
    dbscan_eps=40.0,
    dbscan_min_points=100
)

# Load your data
rgb_image = ...  # Load RGB image
point_cloud = ...  # Load point cloud
camera_params = ...  # Load camera parameters

# Process scene
detected_objects, grasp_plans, bin_dims = system.process_scene(
    rgb_image=rgb_image,
    point_cloud=point_cloud,
    camera_params=camera_params
)

# Get top grasp candidate
if grasp_plans:
    obj, decision = grasp_plans[0]
    print(f"Selected Gripper: {decision['gripper']}")
    print(f"Confidence: {decision['confidence']:.2%}")
    print(f"Grasp Pose: {decision['grasp_pose']}")
```

### Using the Jupyter Notebook

Run the interactive demo:
```bash
jupyter notebook notebooks/demo.ipynb
```

## Module Documentation

### clustering.py

Handles point cloud preprocessing and object segmentation.

**Key Components:**
- `PointCloudProcessor`: Main class for point cloud operations
  - Voxel downsampling
  - Statistical outlier removal
  - RANSAC plane segmentation
  - DBSCAN clustering

**Example:**
```python
from src.clustering import PointCloudProcessor

processor = PointCloudProcessor(
    voxel_size=2.0,
    dbscan_eps=40.0,
    dbscan_min_points=100
)

# Preprocess point cloud
pcd_processed, plane_model = processor.preprocess_point_cloud(pcd)

# Extract clusters
clusters, labels = processor.extract_clusters(pcd_processed)
```

### pose_estimation.py

Estimates 6D poses (position + orientation) for detected objects.

**Key Components:**
- `PoseEstimator`: Static methods for pose estimation
  - PCA-based orientation estimation
  - Bounding box computation
  - Confidence scoring
  - Surface quality estimation

**Example:**
```python
from src.pose_estimation import PoseEstimator

# Estimate pose for a point cluster
pose_data = PoseEstimator.estimate_pose(
    points=cluster_points,
    bin_center_x=bin_cx,
    bin_center_y=bin_cy,
    bin_half_width=bin_hw,
    bin_half_depth=bin_hd
)

print(f"Position: {pose_data['position']}")
print(f"Orientation: {pose_data['orientation']}")
```

### grasp_planner.py

Selects appropriate gripper based on object properties and collision constraints.

**Key Components:**
- `GraspPlanner`: Gripper selection logic
  - Finger gripper evaluation
  - Suction gripper evaluation
  - Collision detection
  - Confidence scoring

**Gripper Specifications:**
```python
DEFAULT_GRIPPER_SPECS = {
    "finger_gripper": {
        "max_width": 145,          # mm
        "jaw_depth": 50,           # mm
        "approach_clearance": 20,  # mm
        "grip_height": 30          # mm
    },
    "suction_gripper": {
        "cup_diameter": 40,        # mm
        "min_flat_area": 625,      # mm²
    }
}
```

**Example:**
```python
from src.grasp_planner import GraspPlanner

planner = GraspPlanner()

decision = planner.choose_gripper(
    obj=detected_object,
    bin_dims={'width': 500, 'depth': 400, 'height': 300}
)

print(f"Gripper: {decision['gripper']}")
print(f"Reason: {decision['reason']}")
```

### action_generator.py

Integrates all modules into a complete bin-picking pipeline.

**Key Components:**
- `BinPickingSystem`: Main system class
  - `detect_objects_and_poses()`: Object detection
  - `plan_grasps()`: Grasp planning
  - `process_scene()`: Complete pipeline
  - `analyze_results()`: Performance analysis

## Algorithm Overview

### Detection Pipeline

1. **Preprocessing** (~15ms)
   - Voxel downsampling (2mm grid)
   - Statistical outlier removal
   - RANSAC plane segmentation

2. **Clustering** (~80ms)
   - DBSCAN with ε=40mm, min_points=100
   - Heuristic filtering for bin walls

3. **Pose Estimation** (~10ms per object)
   - PCA for orientation
   - Axis-aligned bounding box
   - Surface quality classification

4. **Grasp Planning** (~5ms per object)
   - Collision detection
   - Gripper feasibility check
   - Confidence scoring

### Gripper Selection Logic

**Finger Gripper:**
- **Size Constraint**: 10mm < object_width < 145mm
- **Clearance Required**: 70mm (jaw_depth + approach_clearance)
- **Approach Directions**: X or Y axis
- **Preferred For**: Medium-sized, compact objects

**Suction Gripper:**
- **Surface**: Smooth or rough (not porous)
- **Area**: ≥625mm² flat surface
- **Accessibility**: Top must be reachable
- **Preferred For**: Large, flat objects or when finger gripper has collisions

## Performance Optimization

### Current Bottlenecks
1. DBSCAN clustering: ~60-80ms
2. Outlier removal: ~25-35ms
3. RANSAC: ~12-18ms

### Optimization Strategies
1. **GPU Acceleration**: Use CUDA PCL for DBSCAN and outlier removal
2. **Algorithmic Tuning**: Reduce RANSAC iterations or use Euclidean clustering
3. **Hardware Preprocessing**: Leverage camera onboard filtering

### Estimated Performance (GPU)
- Total latency: **70-90ms**
- Throughput: **11-14 fps**
- Hardware: NVIDIA Jetson AGX Orin

## Limitations

1. **Fixed Parameters**: DBSCAN parameters may not work for all bin configurations
2. **Geometry-Only Surface Classification**: ~30% mislabeling rate without RGB fusion
3. **No Occlusion Modeling**: Pose errors >20mm for heavily occluded objects
4. **Fixed Coordinate Frame**: Assumes bin-aligned coordinate system
5. **No Grasp Stability Verification**: May select low-quality grasps

## Future Improvements

1. **RGB-D Fusion**: Add ResNet-18 for material classification (~10ms)
2. **Adaptive Clustering**: Replace DBSCAN with HDBSCAN
3. **Grasp Simulation**: Integrate contact point analysis
4. **Dynamic Parameter Tuning**: Auto-adjust based on scene complexity
5. **Deep Learning Integration**: Add PointNet++ for improved segmentation

## License

MIT License

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{bin_picking_system,
  title = {Robotic Bin-Picking System: Intelligent Grasp Planning with Collision Avoidance},
  author = {Wu, Tao (Michael)},
  year = {2025},
  url = {https://github.com/WuTao1103/bin_picking}
}
```

## Author

**Michael Wu (吴涛)**
- GitHub: [@WuTao1103](https://github.com/WuTao1103)
- Website: [TaoWu.me](https:/taowu.me)

## Acknowledgments

- **Open3D** - Excellent library for 3D data processing
- **OpenCV** - Computer vision tools
- Thanks to the open-source robotics community for inspiration and resources
