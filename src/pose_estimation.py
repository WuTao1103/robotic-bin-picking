"""
Pose Estimation Module

This module provides 6D pose estimation (position + orientation) for detected objects
using Principal Component Analysis (PCA) and bounding box computation.
"""

import numpy as np
from typing import Dict, Tuple


class PoseEstimator:
    """
    Estimates 6D poses of objects from point cloud clusters.

    Uses PCA-based orientation estimation and bounding box analysis
    to determine object position and orientation.
    """

    @staticmethod
    def rotation_matrix_to_rpy(R: np.ndarray) -> np.ndarray:
        """
        Convert 3x3 rotation matrix to roll, pitch, yaw (ZYX Euler angles).

        Args:
            R: 3x3 rotation matrix

        Returns:
            Array of [roll, pitch, yaw] in radians
        """
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0.0

        return np.array([roll, pitch, yaw], dtype=np.float64)

    @staticmethod
    def compute_bounding_box(points: np.ndarray) -> Dict[str, float]:
        """
        Compute axis-aligned bounding box dimensions.

        Args:
            points: Nx3 array of 3D points

        Returns:
            Dictionary with 'width', 'height', 'depth' (X, Y, Z dimensions in mm)
        """
        min_xyz = points.min(axis=0)
        max_xyz = points.max(axis=0)
        bbox_sizes = max_xyz - min_xyz

        return {
            'width': float(bbox_sizes[0]),
            'height': float(bbox_sizes[1]),
            'depth': float(bbox_sizes[2])
        }

    @staticmethod
    def estimate_orientation_pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate object orientation using Principal Component Analysis.

        Args:
            points: Nx3 array of 3D points

        Returns:
            Tuple of (rotation matrix, [roll, pitch, yaw])
        """
        # Center points
        centroid = points.mean(axis=0)
        points_centered = points - centroid

        # Compute covariance and eigenvectors
        cov = np.cov(points_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort by eigenvalue (largest first)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        R = eigvecs[:, order]

        # Ensure right-handed coordinate system
        if np.linalg.det(R) < 0:
            R[:, -1] *= -1

        # Convert to roll-pitch-yaw
        rpy = PoseEstimator.rotation_matrix_to_rpy(R)

        return R, rpy

    @staticmethod
    def compute_confidence_score(
        points: np.ndarray,
        bbox: Dict[str, float],
        eigvals: np.ndarray,
        plane_model: np.ndarray = None
    ) -> float:
        """
        Compute detection confidence score based on multiple heuristics.

        Args:
            points: Nx3 array of object points
            bbox: Bounding box dictionary
            eigvals: PCA eigenvalues
            plane_model: Optional plane model [a, b, c, d] for distance scoring

        Returns:
            Confidence score in range [0, 1]
        """
        num_points = float(len(points))
        x, y, z = bbox['width'], bbox['height'], bbox['depth']

        # 1. Size score: more points = higher confidence
        size_score = float(np.clip(num_points / 1000.0, 0.0, 1.0))

        # 2. Shape score: penalize extreme aspect ratios
        ratios = np.array([
            x / (y + 1e-6),
            x / (z + 1e-6),
            y / (z + 1e-6)
        ])
        max_ratio = float(ratios.max())
        if max_ratio <= 4.0:
            shape_score = 1.0
        elif max_ratio >= 8.0:
            shape_score = 0.0
        else:
            shape_score = (8.0 - max_ratio) / 4.0

        # 3. Compactness score: based on PCA eigenvalue ratio
        eigvals = np.maximum(eigvals, 1e-6)
        thickness_ratio = float(eigvals[2] / eigvals[0])
        compact_score = float(np.clip((thickness_ratio - 0.01) / 0.19, 0.0, 1.0))

        # 4. Plane height score: objects should be above the plane
        plane_height_score = 0.5
        if plane_model is not None:
            a, b, c, d = plane_model
            plane_norm = np.sqrt(a * a + b * b + c * c) + 1e-6
            centroid = points.mean(axis=0)
            plane_dist = abs(a * centroid[0] + b * centroid[1] + c * centroid[2] + d) / plane_norm
            plane_height_score = float(np.clip(plane_dist / 5.0, 0.0, 1.0))

        # Weighted combination
        raw_conf = (
            0.40 * size_score +
            0.25 * shape_score +
            0.25 * compact_score +
            0.10 * plane_height_score
        )

        return float(np.clip(0.2 + 0.8 * raw_conf, 0.05, 0.99))

    @staticmethod
    def estimate_surface_quality(points: np.ndarray, bbox: Dict[str, float]) -> str:
        """
        Estimate surface quality based on geometric features.

        Args:
            points: Nx3 array of object points
            bbox: Bounding box dictionary

        Returns:
            Surface quality: 'smooth', 'rough', or 'porous'
        """
        width = bbox['width']
        height = bbox['height']
        depth = bbox['depth']

        dims = sorted([width, height, depth])
        min_dim, mid_dim, max_dim = dims

        z_std = np.std(points[:, 2])

        # Porous: very thin or irregular objects
        if min_dim < 12:
            return "porous"

        if max_dim > 200 and min_dim < 40 and z_std > 15:
            return "porous"

        if z_std > 35 and min_dim < 50:
            return "porous"

        # Smooth: large flat objects with low Z variation
        if min_dim < 60 and max_dim > 180 and z_std < 7:
            return "smooth"

        if min_dim < 50 and mid_dim > 80 and z_std < 5:
            return "smooth"

        if max_dim < 120 and min_dim > 40 and z_std < 4:
            return "smooth"

        # Default to rough
        return "rough"

    @staticmethod
    def compute_distance_to_walls(
        position: np.ndarray,
        bin_half_width: float,
        bin_half_depth: float
    ) -> Dict[str, float]:
        """
        Compute distances from object position to bin walls.

        Args:
            position: Object position [x, y, z] relative to bin center
            bin_half_width: Half-width of bin in X direction
            bin_half_depth: Half-depth of bin in Y direction

        Returns:
            Dictionary with 'left', 'right', 'front', 'back' distances (mm)
        """
        return {
            'left': float(bin_half_width + position[0]),
            'right': float(bin_half_width - position[0]),
            'front': float(bin_half_depth + position[1]),
            'back': float(bin_half_depth - position[1]),
        }

    @staticmethod
    def estimate_pose(
        points: np.ndarray,
        bin_center_x: float,
        bin_center_y: float,
        bin_half_width: float,
        bin_half_depth: float,
        plane_model: np.ndarray = None
    ) -> Dict:
        """
        Estimate complete 6D pose and properties for an object cluster.

        Args:
            points: Nx3 array of object points
            bin_center_x: X coordinate of bin center
            bin_center_y: Y coordinate of bin center
            bin_half_width: Half-width of bin
            bin_half_depth: Half-depth of bin
            plane_model: Optional plane model for confidence scoring

        Returns:
            Dictionary containing pose estimation results
        """
        # Compute centroid
        centroid = points.mean(axis=0)

        # Convert to bin-centered coordinates
        position_bin = np.array([
            centroid[0] - bin_center_x,
            centroid[1] - bin_center_y,
            centroid[2]
        ], dtype=np.float64)

        # Estimate orientation
        R, rpy = PoseEstimator.estimate_orientation_pca(points)

        # Compute bounding box
        bbox = PoseEstimator.compute_bounding_box(points)

        # Compute confidence
        points_centered = points - centroid
        cov = np.cov(points_centered.T)
        eigvals = np.linalg.eigh(cov)[0]
        eigvals = eigvals[np.argsort(eigvals)[::-1]]

        confidence = PoseEstimator.compute_confidence_score(
            points, bbox, eigvals, plane_model
        )

        # Estimate surface quality
        surface_quality = PoseEstimator.estimate_surface_quality(points, bbox)

        # Compute distances to walls
        distance_to_walls = PoseEstimator.compute_distance_to_walls(
            position_bin, bin_half_width, bin_half_depth
        )

        return {
            'position': position_bin,
            'orientation': rpy,
            'bounding_box': bbox,
            'confidence': confidence,
            'surface_quality': surface_quality,
            'distance_to_walls': distance_to_walls,
            'rotation_matrix': R
        }
