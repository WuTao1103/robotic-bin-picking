"""
Action Generator Module

This module orchestrates the complete detection and grasp planning pipeline.
It integrates point cloud processing, object detection, pose estimation,
and gripper selection to generate actionable grasp plans.
"""

import numpy as np
import open3d as o3d
from typing import List, Dict, Tuple, Optional

from .clustering import PointCloudProcessor
from .pose_estimation import PoseEstimator
from .grasp_planner import GraspPlanner, DEFAULT_GRIPPER_SPECS


class BinPickingSystem:
    """
    Complete robotic bin-picking system integrating perception and planning.

    Processes RGB-D data to detect objects, estimate poses, and plan grasps
    while avoiding collisions with bin walls.
    """

    def __init__(
        self,
        voxel_size: float = 2.0,
        dbscan_eps: float = 40.0,
        dbscan_min_points: int = 100,
        gripper_specs: Dict = None
    ):
        """
        Initialize the bin-picking system.

        Args:
            voxel_size: Voxel size for downsampling (mm)
            dbscan_eps: DBSCAN epsilon parameter (mm)
            dbscan_min_points: DBSCAN minimum points parameter
            gripper_specs: Gripper specifications (uses defaults if None)
        """
        self.processor = PointCloudProcessor(
            voxel_size=voxel_size,
            dbscan_eps=dbscan_eps,
            dbscan_min_points=dbscan_min_points
        )
        self.pose_estimator = PoseEstimator()
        self.grasp_planner = GraspPlanner(gripper_specs or DEFAULT_GRIPPER_SPECS)

    def detect_objects_and_poses(
        self,
        rgb_image: np.ndarray,
        point_cloud: o3d.geometry.PointCloud,
        camera_params: Dict,
        debug: bool = False
    ) -> List[Dict]:
        """
        Detect objects and estimate their 6D poses from RGB-D data.

        Args:
            rgb_image: RGB image (HxWx3)
            point_cloud: Open3D point cloud with colors
            camera_params: Camera intrinsic parameters
            debug: Enable debug output

        Returns:
            List of detected objects with structure:
                {
                    'id': int,
                    'position': np.array([x, y, z]),  # mm, from bin center
                    'orientation': np.array([roll, pitch, yaw]),  # radians
                    'bounding_box': {'width', 'height', 'depth'},  # mm
                    'confidence': float,  # [0-1]
                    'surface_quality': str,  # 'smooth', 'rough', 'porous'
                    'distance_to_walls': {'left', 'right', 'front', 'back'},  # mm
                    'point_indices': list
                }
        """
        if point_cloud is None or len(point_cloud.points) == 0:
            return []

        # Get full point cloud for bin geometry estimation
        full_points = np.asarray(point_cloud.points)

        x_min, x_max = full_points[:, 0].min(), full_points[:, 0].max()
        y_min, y_max = full_points[:, 1].min(), full_points[:, 1].max()

        bin_center_x = 0.5 * (x_min + x_max)
        bin_center_y = 0.5 * (y_min + y_max)

        bin_half_width = 0.5 * (x_max - x_min)
        bin_half_depth = 0.5 * (y_max - y_min)

        # Preprocess point cloud
        pcd_processed, plane_model = self.processor.preprocess_point_cloud(
            o3d.geometry.PointCloud(point_cloud),
            remove_plane=True
        )

        if len(pcd_processed.points) == 0:
            if debug:
                print("[DEBUG] No points remaining after preprocessing")
            return []

        # Extract clusters
        clusters, cluster_labels = self.processor.extract_clusters(
            pcd_processed,
            min_cluster_size=80
        )

        if not clusters:
            if debug:
                print("[DEBUG] No valid clusters found")
            return []

        if debug:
            print(f"[DEBUG] Found {len(clusters)} clusters")

        # Process each cluster
        objects = []
        for obj_id, (points, label) in enumerate(zip(clusters, cluster_labels)):
            # Apply heuristic filters
            bbox_temp = self.pose_estimator.compute_bounding_box(points)
            x, y, z = bbox_temp['width'], bbox_temp['height'], bbox_temp['depth']

            # Size filters
            if (x > 400 or y > 400 or z > 200 or (x * y) > (300 * 300)):
                continue

            if max(x, y, z) < 30 or (x * y * z) < (30 ** 3):
                continue

            if (z < 25 and (x > 250 or y > 250)):
                continue

            if max(x, y, z) > 300 and min(x, y, z) < 40:
                continue

            # Estimate pose
            pose_data = self.pose_estimator.estimate_pose(
                points,
                bin_center_x,
                bin_center_y,
                bin_half_width,
                bin_half_depth,
                plane_model
            )

            # Create object dictionary
            obj = {
                'id': int(obj_id),
                'position': pose_data['position'],
                'orientation': pose_data['orientation'],
                'bounding_box': pose_data['bounding_box'],
                'confidence': pose_data['confidence'],
                'surface_quality': pose_data['surface_quality'],
                'distance_to_walls': pose_data['distance_to_walls'],
                'point_indices': []  # Would need mapping back to original cloud
            }

            objects.append(obj)

        # Sort by confidence
        objects.sort(key=lambda o: o['confidence'], reverse=True)

        return objects

    def plan_grasps(
        self,
        detected_objects: List[Dict],
        bin_dims: Dict,
        gripper_specs: Dict = None
    ) -> List[Tuple[Dict, Dict]]:
        """
        Plan grasps for all detected objects.

        Args:
            detected_objects: List of detected objects from detect_objects_and_poses
            bin_dims: Bin dimensions {'width', 'depth', 'height'}
            gripper_specs: Optional gripper specifications

        Returns:
            List of tuples (object, grasp_decision) sorted by grasp confidence
        """
        if gripper_specs is None:
            gripper_specs = self.grasp_planner.gripper_specs

        results = []

        for obj in detected_objects:
            decision = self.grasp_planner.choose_gripper(
                obj, bin_dims, gripper_specs
            )

            results.append((obj, decision))

        # Sort by grasp confidence
        results.sort(key=lambda x: x[1]['confidence'], reverse=True)

        return results

    def process_scene(
        self,
        rgb_image: np.ndarray,
        point_cloud: o3d.geometry.PointCloud,
        camera_params: Dict,
        debug: bool = False
    ) -> Tuple[List[Dict], List[Tuple[Dict, Dict]], Dict]:
        """
        Complete scene processing: detection + grasp planning.

        Args:
            rgb_image: RGB image
            point_cloud: Point cloud
            camera_params: Camera parameters
            debug: Enable debug output

        Returns:
            Tuple of (detected_objects, grasp_plans, bin_dims)
        """
        # Detect objects
        detected_objects = self.detect_objects_and_poses(
            rgb_image, point_cloud, camera_params, debug
        )

        if not detected_objects:
            return [], [], {}

        # Estimate bin dimensions from point cloud
        full_points = np.asarray(point_cloud.points)
        bin_dims = {
            "width": float(full_points[:, 0].max() - full_points[:, 0].min()),
            "depth": float(full_points[:, 1].max() - full_points[:, 1].min()),
            "height": float(full_points[:, 2].max() - full_points[:, 2].min())
        }

        # Plan grasps
        grasp_plans = self.plan_grasps(detected_objects, bin_dims)

        return detected_objects, grasp_plans, bin_dims

    def analyze_results(self, grasp_plans: List[Tuple[Dict, Dict]]) -> Dict:
        """
        Analyze gripper selection statistics.

        Args:
            grasp_plans: List of (object, decision) tuples

        Returns:
            Dictionary with analysis statistics
        """
        if not grasp_plans:
            return {
                'total_objects': 0,
                'finger_count': 0,
                'suction_count': 0,
                'high_confidence_count': 0
            }

        finger_count = sum(
            1 for _, dec in grasp_plans
            if dec['gripper'] == 'finger_gripper'
        )
        suction_count = len(grasp_plans) - finger_count

        finger_confs = [
            dec['confidence'] for _, dec in grasp_plans
            if dec['gripper'] == 'finger_gripper'
        ]
        suction_confs = [
            dec['confidence'] for _, dec in grasp_plans
            if dec['gripper'] == 'suction_gripper'
        ]

        high_conf_count = sum(
            1 for _, dec in grasp_plans
            if dec['confidence'] > 0.7
        )

        collision_count = sum(
            1 for _, dec in grasp_plans
            if 'blocked' in dec['reason'].lower() or 'collide' in dec['reason'].lower()
        )

        return {
            'total_objects': len(grasp_plans),
            'finger_count': finger_count,
            'suction_count': suction_count,
            'finger_percentage': finger_count / len(grasp_plans) * 100,
            'suction_percentage': suction_count / len(grasp_plans) * 100,
            'finger_mean_conf': np.mean(finger_confs) if finger_confs else 0.0,
            'suction_mean_conf': np.mean(suction_confs) if suction_confs else 0.0,
            'high_confidence_count': high_conf_count,
            'high_confidence_percentage': high_conf_count / len(grasp_plans) * 100,
            'collision_count': collision_count
        }
