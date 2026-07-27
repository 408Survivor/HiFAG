"""Hand-crafted region-level descriptors for HiFAG's coarse facial graph.

For each frame and each of the 9 facial regions (AFGNN's LANDMARK_GROUPS_68)
we compute a fixed-size, zero-parameter feature vector from the landmark
coordinates. These descriptors become the node features of the coarse
9-node-per-frame region graph.

Feature layout per region node (FEATURE_DIM = 10):
    0: cx             region centroid x
    1: cy             region centroid y
    2: area           polygon area (closed regions only: eyes/mouths; else 0)
    3: spread         mean distance of region landmarks to centroid
    4: vx             mean velocity x of region landmarks
    5: vy             mean velocity y of region landmarks
    6: speed          mean |v| of region landmarks
    7: motion_var     mean squared deviation of landmark velocities from the
                      region mean velocity (intra-region motion disagreement)
    8: sym_centroid   |c_left - c_right| for symmetric pairs (brows/eyes), else 0
    9: sym_velocity   |v_left - v_right| for symmetric pairs, else 0

Velocity follows AFGNN's convention: first-order temporal difference with
zero padding at the first frame.
"""

from typing import Dict, List

import numpy as np

# Region definitions, identical to AFGNN's LANDMARK_GROUPS_68
# (AFGNN/src/models/graph_utils.py:19-29). Duplicated here so this module
# works without importing AFGNN (and stays correct if used standalone).
REGION_GROUPS: Dict[str, List[int]] = {
    "face_contour": list(range(0, 17)),
    "left_eyebrow": list(range(17, 22)),
    "right_eyebrow": list(range(22, 27)),
    "nose_bridge": list(range(27, 31)),
    "nose_bottom": list(range(31, 36)),
    "left_eye": list(range(36, 42)),
    "right_eye": list(range(42, 48)),
    "outer_mouth": list(range(48, 60)),
    "inner_mouth": list(range(60, 68)),
}

REGION_NAMES: List[str] = list(REGION_GROUPS.keys())
NUM_REGIONS: int = len(REGION_NAMES)  # 9

# Regions whose landmarks form a closed polygon (area is meaningful).
CLOSED_REGIONS = {"left_eye", "right_eye", "outer_mouth", "inner_mouth"}

# Symmetric left/right region pairs (region ids into REGION_NAMES).
SYMMETRY_PAIRS = [
    (REGION_NAMES.index("left_eyebrow"), REGION_NAMES.index("right_eyebrow")),
    (REGION_NAMES.index("left_eye"), REGION_NAMES.index("right_eye")),
]

FEATURE_NAMES: List[str] = [
    "cx", "cy", "area", "spread",
    "vx", "vy", "speed", "motion_var",
    "sym_centroid", "sym_velocity",
]
FEATURE_DIM: int = len(FEATURE_NAMES)  # 10


def compute_velocity(coords: np.ndarray) -> np.ndarray:
    """First-order temporal difference, zero-padded at the first frame.

    Args:
        coords: (T, N, 2)
    Returns:
        velocity: (T, N, 2)
    """
    velocity = np.zeros_like(coords)
    if coords.shape[0] > 1:
        velocity[1:] = coords[1:] - coords[:-1]
    return velocity


def _polygon_area(points: np.ndarray) -> np.ndarray:
    """Shoelace formula for polygon area.

    Args:
        points: (T, M, 2) polygon vertices in order.
    Returns:
        area: (T,) absolute polygon area per frame.
    """
    x = points[..., 0]
    y = points[..., 1]
    x_next = np.roll(x, -1, axis=1)
    y_next = np.roll(y, -1, axis=1)
    return 0.5 * np.abs(np.sum(x * y_next - x_next * y, axis=1))


def compute_region_features(coords: np.ndarray) -> np.ndarray:
    """Compute per-frame, per-region descriptors from landmark coordinates.

    Args:
        coords: landmark coordinates, shape (T, 68, 2).

    Returns:
        features: shape (T, NUM_REGIONS, FEATURE_DIM), float32.
    """
    coords = np.asarray(coords, dtype=np.float64)
    T = coords.shape[0]
    velocity = compute_velocity(coords)  # (T, 68, 2)

    feats = np.zeros((T, NUM_REGIONS, FEATURE_DIM), dtype=np.float64)

    centroids = np.zeros((T, NUM_REGIONS, 2), dtype=np.float64)
    mean_vels = np.zeros((T, NUM_REGIONS, 2), dtype=np.float64)

    for r, name in enumerate(REGION_NAMES):
        idx = REGION_GROUPS[name]
        pts = coords[:, idx, :]      # (T, M, 2)
        vel = velocity[:, idx, :]    # (T, M, 2)

        centroid = pts.mean(axis=1)          # (T, 2)
        mean_vel = vel.mean(axis=1)          # (T, 2)
        centroids[:, r, :] = centroid
        mean_vels[:, r, :] = mean_vel

        # Geometry
        feats[:, r, 0:2] = centroid
        if name in CLOSED_REGIONS:
            feats[:, r, 2] = _polygon_area(pts)
        feats[:, r, 3] = np.linalg.norm(pts - centroid[:, None, :], axis=-1).mean(axis=1)

        # Motion
        feats[:, r, 4:6] = mean_vel
        feats[:, r, 6] = np.linalg.norm(vel, axis=-1).mean(axis=1)
        feats[:, r, 7] = ((vel - mean_vel[:, None, :]) ** 2).sum(axis=-1).mean(axis=1)

    # Symmetry (attached to both nodes of each pair, 0 elsewhere).
    for left, right in SYMMETRY_PAIRS:
        feats[:, left, 8] = feats[:, right, 8] = np.linalg.norm(
            centroids[:, left, :] - centroids[:, right, :], axis=-1
        )
        feats[:, left, 9] = feats[:, right, 9] = np.linalg.norm(
            mean_vels[:, left, :] - mean_vels[:, right, :], axis=-1
        )

    return feats.astype(np.float32)


def compute_region_features_batch(coords: np.ndarray) -> np.ndarray:
    """Batched wrapper over compute_region_features.

    Args:
        coords: (B, T, 68, 2)
    Returns:
        features: (B, T, NUM_REGIONS, FEATURE_DIM)
    """
    return np.stack([compute_region_features(c) for c in coords], axis=0)


def feature_labels() -> List[str]:
    """Flat names for the (region, feature) dimensions, for reporting."""
    return [f"{region}.{feat}" for region in REGION_NAMES for feat in FEATURE_NAMES]
