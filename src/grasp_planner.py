"""
Grasp Planning Module

This module implements gripper selection logic for robotic bin-picking systems.
It decides between finger gripper and suction gripper based on object properties
and collision avoidance constraints.
"""

import numpy as np
from typing import Dict


# Default gripper specifications
DEFAULT_GRIPPER_SPECS = {
    "finger_gripper": {
        "max_width": 145,          # mm - maximum jaw opening
        "jaw_depth": 50,           # mm - how far jaws extend when gripping
        "approach_clearance": 20,  # mm - minimum clearance needed for approach
        "grip_height": 30          # mm - height of gripper jaws
    },
    "suction_gripper": {
        "cup_diameter": 40,        # mm
        "min_flat_area": 625,      # mm² - minimum flat surface (25mm x 25mm)
        "max_surface_roughness": "smooth"
    }
}


class GraspPlanner:
    """
    Plans grasp strategy by selecting appropriate gripper for each object.

    Considers object size, surface quality, position, and potential collisions
    with bin walls to determine the best gripper and grasp pose.
    """

    def __init__(self, gripper_specs: Dict = None):
        """
        Initialize grasp planner with gripper specifications.

        Args:
            gripper_specs: Dictionary of gripper specifications
                          (uses defaults if not provided)
        """
        self.gripper_specs = gripper_specs or DEFAULT_GRIPPER_SPECS

    def choose_gripper(
        self,
        obj: Dict,
        bin_dims: Dict,
        gripper_specs: Dict = None
    ) -> Dict:
        """
        Select the best gripper for a given object.

        Args:
            obj: Object dictionary containing:
                - id: Object identifier
                - position: [x, y, z] position relative to bin center (mm)
                - orientation: [roll, pitch, yaw] in radians
                - bounding_box: {'width', 'height', 'depth'} in mm
                - confidence: Detection confidence [0-1]
                - surface_quality: 'smooth', 'rough', or 'porous'
                - distance_to_walls: {'left', 'right', 'front', 'back'} in mm
            bin_dims: Bin dimensions {'width', 'depth', 'height'} in mm
            gripper_specs: Optional gripper specs (uses self.gripper_specs if None)

        Returns:
            Dictionary containing:
                - gripper: 'finger_gripper' or 'suction_gripper'
                - confidence: Grasp confidence [0-1]
                - reason: Explanation for the choice
                - grasp_pose: [x, y, z, roll, pitch, yaw] grasp pose
        """
        if gripper_specs is None:
            gripper_specs = self.gripper_specs

        # Extract object properties
        position = np.array(obj["position"], dtype=float)
        orientation = np.array(obj["orientation"], dtype=float)
        bbox = obj["bounding_box"]
        surface = obj.get("surface_quality", "rough")
        obj_confidence = float(obj.get("confidence", 0.5))

        # Bin dimensions
        bin_half_width = bin_dims["width"] / 2.0
        bin_half_depth = bin_dims["depth"] / 2.0
        bin_height = bin_dims["height"]

        # Distances to bin walls
        d2w = obj.get("distance_to_walls", {})
        if all(k in d2w for k in ("left", "right", "front", "back")):
            left_clearance = float(d2w["left"])
            right_clearance = float(d2w["right"])
            front_clearance = float(d2w["front"])
            back_clearance = float(d2w["back"])
        else:
            left_clearance = bin_half_width + position[0]
            right_clearance = bin_half_width - position[0]
            front_clearance = bin_half_depth + position[1]
            back_clearance = bin_half_depth - position[1]

        # Gripper specs
        finger_specs = gripper_specs["finger_gripper"]
        max_grip_width = finger_specs["max_width"]
        jaw_depth = finger_specs["jaw_depth"]
        approach_clearance = finger_specs["approach_clearance"]
        grip_height = finger_specs["grip_height"]

        suction_specs = gripper_specs["suction_gripper"]
        cup_diameter = suction_specs["cup_diameter"]
        min_flat_area = suction_specs["min_flat_area"]

        # Evaluate finger gripper feasibility
        finger_result = self._evaluate_finger_gripper(
            position, orientation, bbox, obj_confidence,
            left_clearance, right_clearance, front_clearance, back_clearance,
            max_grip_width, jaw_depth, approach_clearance, grip_height,
            bin_height
        )

        # Evaluate suction gripper feasibility
        suction_result = self._evaluate_suction_gripper(
            position, orientation, bbox, obj_confidence, surface,
            cup_diameter, min_flat_area, bin_height
        )

        # Make decision
        return self._make_decision(
            finger_result, suction_result, position, orientation
        )

    def _evaluate_finger_gripper(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        bbox: Dict,
        obj_confidence: float,
        left_clearance: float,
        right_clearance: float,
        front_clearance: float,
        back_clearance: float,
        max_grip_width: float,
        jaw_depth: float,
        approach_clearance: float,
        grip_height: float,
        bin_height: float
    ) -> Dict:
        """Evaluate finger gripper feasibility and confidence."""
        obj_dims = [bbox["width"], bbox["height"], bbox["depth"]]
        min_dim = float(min(obj_dims))
        max_dim = float(max(obj_dims))

        # Size constraint
        size_ok = (10.0 < min_dim < max_grip_width)

        # Required clearance for approach
        min_side_clearance = jaw_depth + approach_clearance  # 70mm

        # Check X-axis approach
        x_feasible = (left_clearance >= min_side_clearance) or \
                     (right_clearance >= min_side_clearance)

        # Check Y-axis approach
        y_feasible = (front_clearance >= min_side_clearance) or \
                     (back_clearance >= min_side_clearance)

        # Vertical clearance
        z_clearance_needed = grip_height + approach_clearance  # 50mm
        z_feasible = position[2] >= z_clearance_needed

        # Overall collision check
        collision_free = z_feasible and (x_feasible or y_feasible)

        # Build collision reason if blocked
        collision_reason = ""
        if not collision_free:
            if not z_feasible:
                collision_reason = (
                    f"Object too low for finger gripper "
                    f"(z={position[2]:.0f}mm < {z_clearance_needed:.0f}mm needed)"
                )
            elif not x_feasible and not y_feasible:
                collision_reason = (
                    f"Finger gripper blocked in X and Y "
                    f"(X: L{left_clearance:.0f}/R{right_clearance:.0f}, "
                    f"Y: F{front_clearance:.0f}/B{back_clearance:.0f}, "
                    f"need ≥{min_side_clearance:.0f}mm on one side)"
                )

        # Compute confidence if feasible
        confidence = 0.0
        grasp_pose = np.zeros(6, dtype=float)

        if size_ok and collision_free:
            # Size score
            size_score = 1.0 - (min_dim / max_grip_width)

            # Aspect ratio score
            aspect_ratio = max_dim / (min_dim + 1e-6)
            aspect_score = 1.0 / (1.0 + np.exp(aspect_ratio - 5.0))

            # Height score
            height_score = np.exp(
                -((position[2] - 0.5 * bin_height) / (0.3 * bin_height)) ** 2
            )

            # Combined confidence
            confidence = (
                0.4 * obj_confidence +
                0.2 * size_score +
                0.2 * aspect_score +
                0.2 * height_score
            )
            confidence = float(np.clip(confidence, 0.35, 0.95))

            # Determine grasp pose based on approach direction
            if x_feasible:
                grasp_pose = np.array([
                    position[0], position[1], position[2],
                    orientation[0], orientation[1], orientation[2]
                ], dtype=float)
            else:
                # Y-approach: rotate yaw by 90°
                grasp_pose = np.array([
                    position[0], position[1], position[2],
                    orientation[0], orientation[1], orientation[2] + np.pi / 2.0
                ], dtype=float)

        return {
            'feasible': size_ok and collision_free,
            'confidence': confidence,
            'grasp_pose': grasp_pose,
            'collision_reason': collision_reason
        }

    def _evaluate_suction_gripper(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        bbox: Dict,
        obj_confidence: float,
        surface: str,
        cup_diameter: float,
        min_flat_area: float,
        bin_height: float
    ) -> Dict:
        """Evaluate suction gripper feasibility and confidence."""
        # Surface suitability
        surface_suitable = surface in ["smooth", "rough"]

        # Area check
        estimated_top_area = bbox["width"] * bbox["height"]
        area_sufficient = estimated_top_area >= min_flat_area

        # Top accessibility
        top_clearance = bin_height - (position[2] + bbox["depth"] / 2.0)
        top_accessible = top_clearance > (cup_diameter / 2.0 + 10.0)

        # Compute confidence if feasible
        confidence = 0.0
        grasp_pose = np.zeros(6, dtype=float)

        if surface_suitable and area_sufficient and top_accessible:
            surface_score = 1.0 if surface == "smooth" else 0.7
            area_score = float(np.clip(
                estimated_top_area / (min_flat_area * 4.0),
                0.5, 1.0
            ))
            height_score = float(np.clip(
                position[2] / (0.3 * bin_height),
                0.3, 1.0
            ))

            confidence = (
                0.3 * obj_confidence +
                0.3 * surface_score +
                0.2 * area_score +
                0.2 * height_score
            )
            confidence = float(np.clip(confidence, 0.25, 0.9))

            # Grasp from top
            grasp_pose = np.array([
                position[0],
                position[1],
                position[2] + bbox["depth"] / 2.0,
                0.0,
                0.0,
                orientation[2]
            ], dtype=float)

        return {
            'feasible': surface_suitable and area_sufficient and top_accessible,
            'confidence': confidence,
            'grasp_pose': grasp_pose,
            'surface': surface
        }

    def _make_decision(
        self,
        finger_result: Dict,
        suction_result: Dict,
        position: np.ndarray,
        orientation: np.ndarray
    ) -> Dict:
        """Make final gripper selection decision."""
        finger_feasible = finger_result['feasible']
        finger_conf = finger_result['confidence']
        suction_feasible = suction_result['feasible']
        suction_conf = suction_result['confidence']

        # Prefer finger if feasible and competitive
        if finger_feasible and finger_conf >= suction_conf and finger_conf > 0.4:
            return {
                "gripper": "finger_gripper",
                "confidence": float(finger_conf),
                "reason": "Finger gripper feasible with sufficient clearance",
                "grasp_pose": finger_result['grasp_pose']
            }

        # Use suction if valid
        if suction_feasible and suction_conf > 0.3:
            reason = finger_result['collision_reason'] if finger_result['collision_reason'] \
                     else f"Suction feasible on {suction_result['surface']} surface"
            return {
                "gripper": "suction_gripper",
                "confidence": float(suction_conf),
                "reason": reason,
                "grasp_pose": suction_result['grasp_pose']
            }

        # Fallback: choose less risky option with low confidence
        base_conf = 0.3

        if finger_result['collision_reason']:
            return {
                "gripper": "suction_gripper",
                "confidence": base_conf,
                "reason": f"Fallback to suction: {finger_result['collision_reason']}",
                "grasp_pose": np.array(
                    [position[0], position[1], position[2], 0.0, 0.0, 0.0],
                    dtype=float
                )
            }
        else:
            return {
                "gripper": "finger_gripper",
                "confidence": base_conf,
                "reason": "Fallback: no clear preference, defaulting to finger gripper",
                "grasp_pose": np.array(
                    [position[0], position[1], position[2], 0.0, 0.0, 0.0],
                    dtype=float
                )
            }
