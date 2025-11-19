"""
Robotic Bin-Picking System

A modular system for object detection, pose estimation, and grasp planning
in robotic bin-picking applications.
"""

from .clustering import PointCloudProcessor, debug_scan_clustering
from .pose_estimation import PoseEstimator
from .grasp_planner import GraspPlanner, DEFAULT_GRIPPER_SPECS
from .action_generator import BinPickingSystem

__version__ = "1.0.0"

__all__ = [
    "PointCloudProcessor",
    "PoseEstimator",
    "GraspPlanner",
    "BinPickingSystem",
    "DEFAULT_GRIPPER_SPECS",
    "debug_scan_clustering"
]
