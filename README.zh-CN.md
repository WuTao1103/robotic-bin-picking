# 机器人料箱拣选系统

[English](README.md) | [简体中文](README.zh-CN.md)

一个用于自动化料箱拣选任务的智能机器人视觉系统。本项目实现了完整的 3D 物体检测、姿态估计和抓取规划管道，使用 RGB-D 点云数据。

## 关于本项目

本系统解决了杂乱环境中自动化物体抓取的挑战，这是仓储自动化、制造业和物流领域的关键问题。通过结合传统计算机视觉技术与智能决策算法，实现了实时性能和高可靠性。

### 问题描述

在现代仓库和制造设施中，机器人需要从包含多个随机摆放物体的料箱中拣选物品。主要挑战包括：
- **检测**：在杂乱的3D场景中识别单个物体
- **姿态估计**：确定每个物体的精确位置和方向
- **抓取规划**：选择最优的抓手和接近策略
- **碰撞避免**：确保抓手不会与料箱壁碰撞

### 解决方案

本项目实现了一个模块化管道：
1. 处理 RGB-D 点云数据以分割单个物体
2. 使用主成分分析（PCA）估计 6D 姿态
3. 基于物体属性智能选择夹爪或吸盘
4. 执行碰撞检测以确保安全抓取

### 实际应用场景

- **电商履单**：仓库中的自动化订单拣选
- **制造业**：零件供给和装配自动化
- **回收**：混合材料分拣
- **食品行业**：处理不规则形状物品

## 功能特性

- **点云处理**：高效的降采样、噪声去除和平面分割
- **物体检测**：基于 DBSCAN 的鲁棒物体分割
- **6D 姿态估计**：基于 PCA 的方向估计和边界框分析
- **智能抓手选择**：在夹爪和吸盘之间自动选择，具有碰撞避免功能
- **模块化架构**：清晰的职责分离，便于定制和扩展

## 性能指标

- **延迟**：每个检测周期约 110-140ms（CPU）
- **吞吐量**：7-9 fps
- **优化潜力**：使用 GPU 加速（CUDA PCL）可达 <100ms

## 项目结构

```
bin_picking/
├── src/
│   ├── __init__.py
│   ├── clustering.py          # 点云预处理和聚类
│   ├── pose_estimation.py     # 6D 姿态估计
│   ├── grasp_planner.py       # 抓手选择逻辑
│   └── action_generator.py    # 完整检测管道
├── notebooks/
│   └── demo.ipynb             # 交互式演示
├── data/                       # 数据集目录
├── README.md
└── requirements.txt
```

## 安装

### 前置要求

- Python 3.7+
- pip

### 设置步骤

1. 克隆仓库：
```bash
git clone https://github.com/WuTao1103/robotic-bin-picking.git
cd robotic-bin-picking
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 将数据集添加到 `data/` 目录：
```
data/
├── Color_3840x2160.png
├── pcd.ply
├── rgb_intrinsic.yaml
├── depth_intrinsic.yaml
└── rgb_distortion.yaml
```

## 快速开始

### 使用 Python API

```python
import numpy as np
import open3d as o3d
from src import BinPickingSystem

# 初始化系统
system = BinPickingSystem(
    voxel_size=2.0,
    dbscan_eps=40.0,
    dbscan_min_points=100
)

# 加载您的数据
rgb_image = ...  # 加载 RGB 图像
point_cloud = ...  # 加载点云
camera_params = ...  # 加载相机参数

# 处理场景
detected_objects, grasp_plans, bin_dims = system.process_scene(
    rgb_image=rgb_image,
    point_cloud=point_cloud,
    camera_params=camera_params
)

# 获取最佳抓取候选
if grasp_plans:
    obj, decision = grasp_plans[0]
    print(f"选择的抓手: {decision['gripper']}")
    print(f"置信度: {decision['confidence']:.2%}")
    print(f"抓取姿态: {decision['grasp_pose']}")
```

### 使用 Jupyter Notebook

运行交互式演示：
```bash
jupyter notebook notebooks/demo.ipynb
```

## 模块文档

### clustering.py

处理点云预处理和物体分割。

**核心组件：**
- `PointCloudProcessor`：点云操作的主类
  - 体素降采样
  - 统计离群点去除
  - RANSAC 平面分割
  - DBSCAN 聚类

**示例：**
```python
from src.clustering import PointCloudProcessor

processor = PointCloudProcessor(
    voxel_size=2.0,
    dbscan_eps=40.0,
    dbscan_min_points=100
)

# 预处理点云
pcd_processed, plane_model = processor.preprocess_point_cloud(pcd)

# 提取聚类
clusters, labels = processor.extract_clusters(pcd_processed)
```

### pose_estimation.py

估计检测到的物体的 6D 姿态（位置 + 方向）。

**核心组件：**
- `PoseEstimator`：姿态估计的静态方法
  - 基于 PCA 的方向估计
  - 边界框计算
  - 置信度评分
  - 表面质量估计

**示例：**
```python
from src.pose_estimation import PoseEstimator

# 为点簇估计姿态
pose_data = PoseEstimator.estimate_pose(
    points=cluster_points,
    bin_center_x=bin_cx,
    bin_center_y=bin_cy,
    bin_half_width=bin_hw,
    bin_half_depth=bin_hd
)

print(f"位置: {pose_data['position']}")
print(f"方向: {pose_data['orientation']}")
```

### grasp_planner.py

基于物体属性和碰撞约束选择合适的抓手。

**核心组件：**
- `GraspPlanner`：抓手选择逻辑
  - 夹爪评估
  - 吸盘评估
  - 碰撞检测
  - 置信度评分

**抓手规格：**
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

**示例：**
```python
from src.grasp_planner import GraspPlanner

planner = GraspPlanner()

decision = planner.choose_gripper(
    obj=detected_object,
    bin_dims={'width': 500, 'depth': 400, 'height': 300}
)

print(f"抓手: {decision['gripper']}")
print(f"原因: {decision['reason']}")
```

### action_generator.py

将所有模块集成到完整的料箱拣选管道中。

**核心组件：**
- `BinPickingSystem`：主系统类
  - `detect_objects_and_poses()`：物体检测
  - `plan_grasps()`：抓取规划
  - `process_scene()`：完整管道
  - `analyze_results()`：性能分析

## 算法概览

### 检测管道

1. **预处理**（约15ms）
   - 体素降采样（2mm 网格）
   - 统计离群点去除
   - RANSAC 平面分割

2. **聚类**（约80ms）
   - DBSCAN（ε=40mm，min_points=100）
   - 料箱壁的启发式过滤

3. **姿态估计**（每个物体约10ms）
   - PCA 方向估计
   - 轴对齐边界框
   - 表面质量分类

4. **抓取规划**（每个物体约5ms）
   - 碰撞检测
   - 抓手可行性检查
   - 置信度评分

### 抓手选择逻辑

**夹爪：**
- **尺寸约束**：10mm < 物体宽度 < 145mm
- **所需间隙**：70mm（爪深 + 接近间隙）
- **接近方向**：X 或 Y 轴
- **适用于**：中等尺寸、紧凑型物体

**吸盘：**
- **表面**：光滑或粗糙（非多孔）
- **面积**：≥625mm² 平面
- **可达性**：顶部必须可达
- **适用于**：大型、平面物体或夹爪有碰撞时

## 性能优化

### 当前瓶颈
1. DBSCAN 聚类：约60-80ms
2. 离群点去除：约25-35ms
3. RANSAC：约12-18ms

### 优化策略
1. **GPU 加速**：使用 CUDA PCL 进行 DBSCAN 和离群点去除
2. **算法调优**：减少 RANSAC 迭代次数或使用欧几里得聚类
3. **硬件预处理**：利用相机板载过滤

### 预估性能（GPU）
- 总延迟：**70-90ms**
- 吞吐量：**11-14 fps**
- 硬件：NVIDIA Jetson AGX Orin

## 局限性

1. **固定参数**：DBSCAN 参数可能不适用于所有料箱配置
2. **仅几何表面分类**：没有 RGB 融合时误标率约 30%
3. **无遮挡建模**：严重遮挡物体的姿态误差 >20mm
4. **固定坐标系**：假设料箱对齐的坐标系统
5. **无抓取稳定性验证**：可能选择低质量的抓取

## 未来改进

1. **RGB-D 融合**：添加 ResNet-18 进行材料分类（约10ms）
2. **自适应聚类**：用 HDBSCAN 替换 DBSCAN
3. **抓取仿真**：集成接触点分析
4. **动态参数调整**：根据场景复杂度自动调整
5. **深度学习集成**：添加 PointNet++ 以改进分割

## 许可证

MIT License

## 引用

如果您觉得本项目有用，请考虑引用：

```bibtex
@software{robotic_bin_picking_system,
  title = {Robotic Bin-Picking System: Intelligent Grasp Planning with Collision Avoidance},
  author = {Wu, Tao (Michael)},
  year = {2025},
  url = {https://github.com/WuTao1103/robotic-bin-picking}
}
```

## 作者

**吴涛 (Michael Wu)**
- GitHub: [@WuTao1103](https://github.com/WuTao1103)
- Website: [TaoWu.me](https://taowu.me)

## 致谢

- **Open3D** - 优秀的 3D 数据处理库
- **OpenCV** - 计算机视觉工具
- 感谢开源机器人社区的灵感和资源

---

[English](README.md) | 简体中文

