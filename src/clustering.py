"""
Point Cloud Clustering Module

This module handles point cloud preprocessing, segmentation, and clustering
for robotic bin-picking applications.
"""

import numpy as np
import open3d as o3d
from typing import Tuple, List, Optional


class PointCloudProcessor:
    """
    Processes point clouds for object detection and segmentation.

    This class provides methods for downsampling, noise removal, plane segmentation,
    and clustering of 3D point cloud data.
    """

    def __init__(
        self,
        voxel_size: float = 2.0,
        outlier_nb_neighbors: int = 20,
        outlier_std_ratio: float = 2.0,
        plane_distance_threshold: float = 3.0,
        plane_ransac_n: int = 3,
        plane_num_iterations: int = 1000,
        dbscan_eps: float = 40.0,
        dbscan_min_points: int = 100
    ):
        """
        Initialize the point cloud processor with configurable parameters.

        Args:
            voxel_size: Size of voxel grid for downsampling (mm)
            outlier_nb_neighbors: Number of neighbors for outlier removal
            outlier_std_ratio: Standard deviation ratio for outlier removal
            plane_distance_threshold: Distance threshold for RANSAC plane segmentation (mm)
            plane_ransac_n: Number of points to sample for RANSAC
            plane_num_iterations: Number of RANSAC iterations
            dbscan_eps: Maximum distance between points in a cluster (mm)
            dbscan_min_points: Minimum points required to form a cluster
        """
        self.voxel_size = voxel_size
        self.outlier_nb_neighbors = outlier_nb_neighbors
        self.outlier_std_ratio = outlier_std_ratio
        self.plane_distance_threshold = plane_distance_threshold
        self.plane_ransac_n = plane_ransac_n
        self.plane_num_iterations = plane_num_iterations
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_points = dbscan_min_points

    def downsample(self, point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Downsample point cloud using voxel grid filtering.

        Args:
            point_cloud: Input point cloud

        Returns:
            Downsampled point cloud
        """
        return point_cloud.voxel_down_sample(voxel_size=self.voxel_size)

    def remove_outliers(
        self,
        point_cloud: o3d.geometry.PointCloud
    ) -> Tuple[o3d.geometry.PointCloud, List[int]]:
        """
        Remove statistical outliers from point cloud.

        Args:
            point_cloud: Input point cloud

        Returns:
            Tuple of (filtered point cloud, inlier indices)
        """
        return point_cloud.remove_statistical_outlier(
            nb_neighbors=self.outlier_nb_neighbors,
            std_ratio=self.outlier_std_ratio
        )

    def segment_plane(
        self,
        point_cloud: o3d.geometry.PointCloud
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Segment the dominant plane (typically bin floor) using RANSAC.

        Args:
            point_cloud: Input point cloud

        Returns:
            Tuple of (plane model [a, b, c, d], inlier indices)
        """
        plane_model, inliers = point_cloud.segment_plane(
            distance_threshold=self.plane_distance_threshold,
            ransac_n=self.plane_ransac_n,
            num_iterations=self.plane_num_iterations
        )
        return np.array(plane_model), inliers

    def cluster_dbscan(
        self,
        point_cloud: o3d.geometry.PointCloud
    ) -> np.ndarray:
        """
        Cluster point cloud using DBSCAN algorithm.

        Args:
            point_cloud: Input point cloud

        Returns:
            Array of cluster labels (-1 for noise points)
        """
        labels = np.array(
            point_cloud.cluster_dbscan(
                eps=self.dbscan_eps,
                min_points=self.dbscan_min_points,
                print_progress=False
            )
        )
        return labels

    def preprocess_point_cloud(
        self,
        point_cloud: o3d.geometry.PointCloud,
        remove_plane: bool = True
    ) -> Tuple[o3d.geometry.PointCloud, Optional[np.ndarray]]:
        """
        Complete preprocessing pipeline: downsample, denoise, and optionally remove plane.

        Args:
            point_cloud: Input point cloud
            remove_plane: Whether to remove the dominant plane

        Returns:
            Tuple of (processed point cloud, plane model if removed)
        """
        # Downsample
        pcd_down = self.downsample(point_cloud)

        if len(pcd_down.points) == 0:
            return pcd_down, None

        # Remove outliers
        pcd_down, _ = self.remove_outliers(pcd_down)

        if len(pcd_down.points) == 0:
            return pcd_down, None

        # Optionally remove plane
        plane_model = None
        if remove_plane and len(pcd_down.points) >= 50:
            plane_model, inliers = self.segment_plane(pcd_down)
            pcd_down = pcd_down.select_by_index(inliers, invert=True)

        return pcd_down, plane_model

    def extract_clusters(
        self,
        point_cloud: o3d.geometry.PointCloud,
        min_cluster_size: int = 80
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Extract individual object clusters from point cloud.

        Args:
            point_cloud: Input point cloud (after preprocessing)
            min_cluster_size: Minimum number of points to consider as valid cluster

        Returns:
            Tuple of (list of point arrays for each cluster, list of cluster labels)
        """
        if len(point_cloud.points) == 0:
            return [], []

        # Cluster
        labels = self.cluster_dbscan(point_cloud)

        if labels.size == 0 or labels.max() < 0:
            return [], []

        # Extract valid clusters
        all_points = np.asarray(point_cloud.points)
        unique_labels = [l for l in np.unique(labels) if l >= 0]

        clusters = []
        cluster_labels = []

        for label in unique_labels:
            idx = np.where(labels == label)[0]
            if idx.size >= min_cluster_size:
                clusters.append(all_points[idx, :])
                cluster_labels.append(int(label))

        return clusters, cluster_labels


def debug_scan_clustering(
    point_cloud: o3d.geometry.PointCloud,
    eps_values: List[float] = [25.0, 30.0, 35.0, 40.0, 50.0],
    min_points_values: List[int] = [60, 80, 100]
) -> None:
    """
    Helper function for tuning DBSCAN parameters.

    Args:
        point_cloud: Input point cloud
        eps_values: List of epsilon values to test
        min_points_values: List of min_points values to test
    """
    print("\n[DEBUG] Clustering parameter scan:")

    for eps in eps_values:
        for min_pts in min_points_values:
            labels = np.array(
                point_cloud.cluster_dbscan(
                    eps=eps,
                    min_points=min_pts,
                    print_progress=False
                )
            )

            if labels.size == 0 or labels.max() < 0:
                n_clusters = 0
            else:
                n_clusters = (np.unique(labels) >= 0).sum()

            print(f"  eps={eps:4.1f}, min_points={min_pts:3d} -> clusters={n_clusters}")
