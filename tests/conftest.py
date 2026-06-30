"""
Shared pytest fixtures for the 3D tongue-tip tracking test suite.
"""

import json
import os
import tempfile

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Camera geometry helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def camera_matrix():
    """A simple pinhole camera intrinsic matrix (640×480 sensor)."""
    return np.array(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


@pytest.fixture
def dist_coeffs():
    """Near-zero distortion coefficients."""
    return np.zeros((1, 5), dtype=np.float64)


@pytest.fixture
def three_projection_matrices(camera_matrix):
    """
    Three (3,4) projection matrices for a 3-camera rig.

    Camera 2 (mid) is the reference.  Cameras 1 and 3 are offset ±100 mm
    along the X axis.
    """
    K = camera_matrix
    R = np.eye(3, dtype=np.float64)

    P1 = K @ np.hstack([R, np.array([[-100.0], [0.0], [0.0]])])  # left
    P2 = K @ np.hstack([R, np.zeros((3, 1))])  # mid (reference)
    P3 = K @ np.hstack([R, np.array([[100.0], [0.0], [0.0]])])  # right
    return [P1, P2, P3]


@pytest.fixture
def known_3d_point():
    """A 3-D world point that projects well inside all three camera views."""
    return np.array([10.0, 5.0, 1000.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Synthetic image / frame fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_gray_frame():
    """200×200 random grayscale frame — provides enough texture for KLT."""
    rng = np.random.default_rng(42)
    return rng.integers(30, 220, (200, 200), dtype=np.uint8)


@pytest.fixture
def synthetic_bgr_frame(synthetic_gray_frame):
    """BGR colour version of the synthetic grayscale frame."""
    return cv2.cvtColor(synthetic_gray_frame, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Synthetic video fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_video_path(tmp_path):
    """
    Write a tiny 20-frame AVI video with a white dot moving diagonally.
    The moving dot creates detectable optical flow.
    """
    video_path = str(tmp_path / "test_video.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    w, h = 200, 200
    writer = cv2.VideoWriter(video_path, fourcc, 10.0, (w, h))

    for i in range(20):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cx, cy = 50 + i * 3, 50 + i * 2  # moving dot
        cv2.circle(frame, (cx, cy), 8, (255, 255, 255), -1)
        writer.write(frame)

    writer.release()
    return video_path


# ---------------------------------------------------------------------------
# Camera-poses JSON fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def camera_poses_json(tmp_path, camera_matrix, dist_coeffs):
    """
    Write a minimal camera_poses.json with three cameras and return its path.
    """
    K = camera_matrix.tolist()
    dist = dist_coeffs.ravel().tolist()
    R_id = np.eye(3).tolist()

    poses = {
        "cameras": [
            {
                "id": "left",
                "K": K,
                "dist": dist,
                "R": R_id,
                "t": [[-100.0], [0.0], [0.0]],
            },
            {
                "id": "mid",
                "K": K,
                "dist": dist,
                "R": R_id,
                "t": [[0.0], [0.0], [0.0]],
            },
            {
                "id": "right",
                "K": K,
                "dist": dist,
                "R": R_id,
                "t": [[100.0], [0.0], [0.0]],
            },
        ]
    }
    path = str(tmp_path / "camera_poses.json")
    with open(path, "w") as f:
        json.dump(poses, f)
    return path


# ---------------------------------------------------------------------------
# Tracked-points CSV fixtures
# ---------------------------------------------------------------------------


def _write_csv(path: str, points: np.ndarray) -> None:
    """Write (N,2) array to a CSV with header frame,x,y."""
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x", "y"])
        for i, (x, y) in enumerate(points):
            w.writerow([i + 1, float(x), float(y)])


@pytest.fixture
def tracked_points_csvs(tmp_path, three_projection_matrices, known_3d_point):
    """
    Create three tracked-points CSV files whose 2-D projections are consistent
    with *known_3d_point* (perfect, noise-free data).
    """
    n = 30

    def project(P, X):
        h = P @ np.append(X, 1.0)
        return h[:2] / h[2]

    # Points slightly varying around known_3d_point
    rng = np.random.default_rng(0)
    xyz_all = known_3d_point + rng.standard_normal((n, 3)) * 5.0

    paths = []
    for P in three_projection_matrices:
        pts = np.array([project(P, X) for X in xyz_all])
        csv_path = str(tmp_path / f"view_{len(paths)}.csv")
        _write_csv(csv_path, pts)
        paths.append(csv_path)

    return paths  # [left_csv, mid_csv, right_csv]
