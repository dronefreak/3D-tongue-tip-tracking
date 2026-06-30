"""
Unit tests for tracking_in_3d.py

Covers:
  - triangulate_point_dlt      (DLT triangulation, 2 and 3 views)
  - triangulate_all            (batch triangulation)
  - _project                   (3-D → 2-D projection)
  - _ba_residuals              (residual vector shape and zero-error case)
  - bundle_adjust              (convergence on noisy data)
  - compute_reprojection_error (perfect and noisy data)
  - load_tracked_points        (CSV round-trip)
  - load_camera_poses          (JSON round-trip, projection matrix shape)
"""

import csv
import json
import os

import numpy as np
import pytest

from tracking_in_3d import (
    _ba_residuals,
    _project,
    bundle_adjust,
    compute_reprojection_error,
    load_camera_poses,
    load_tracked_points,
    triangulate_all,
    triangulate_point_dlt,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _proj(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    h = P @ np.append(X, 1.0)
    return h[:2] / h[2]


# ---------------------------------------------------------------------------
# triangulate_point_dlt
# ---------------------------------------------------------------------------


class TestTriangulatePointDlt:
    def test_two_view_exact_recovery(self, three_projection_matrices, known_3d_point):
        """DLT with 2 views must recover the known point to floating-point precision."""
        P1, P2 = three_projection_matrices[:2]
        p1 = _proj(P1, known_3d_point)
        p2 = _proj(P2, known_3d_point)

        X = triangulate_point_dlt([p1, p2], [P1, P2])
        assert np.linalg.norm(X - known_3d_point) < 1e-4

    def test_three_view_exact_recovery(self, three_projection_matrices, known_3d_point):
        """DLT with 3 views must recover the known point with negligible error."""
        pts = [_proj(P, known_3d_point) for P in three_projection_matrices]
        X = triangulate_point_dlt(pts, three_projection_matrices)
        assert np.linalg.norm(X - known_3d_point) < 1e-4

    def test_returns_3d_array(self, three_projection_matrices, known_3d_point):
        pts = [_proj(P, known_3d_point) for P in three_projection_matrices]
        X = triangulate_point_dlt(pts, three_projection_matrices)
        assert X.shape == (3,)

    def test_output_dtype_is_float64(self, three_projection_matrices, known_3d_point):
        pts = [_proj(P, known_3d_point) for P in three_projection_matrices]
        X = triangulate_point_dlt(pts, three_projection_matrices)
        assert X.dtype == np.float64

    def test_different_depths_recovered_correctly(self, three_projection_matrices):
        """Multiple depth values must all triangulate accurately."""
        for z in [500.0, 1000.0, 2000.0]:
            X_true = np.array([0.0, 0.0, z])
            pts = [_proj(P, X_true) for P in three_projection_matrices]
            X = triangulate_point_dlt(pts, three_projection_matrices)
            assert np.linalg.norm(X - X_true) < 1.0, f"failed at depth z={z}"


# ---------------------------------------------------------------------------
# triangulate_all
# ---------------------------------------------------------------------------


class TestTriangulateAll:
    def test_output_shape(self, three_projection_matrices, known_3d_point):
        n = 10
        rng = np.random.default_rng(1)
        xyz_true = known_3d_point + rng.standard_normal((n, 3)) * 5.0

        pts_views = [np.array([_proj(P, X) for X in xyz_true]) for P in three_projection_matrices]
        xyz = triangulate_all(pts_views, three_projection_matrices)
        assert xyz.shape == (n, 3)

    def test_all_points_recovered(self, three_projection_matrices, known_3d_point):
        n = 5
        xyz_true = known_3d_point + np.arange(n).reshape(-1, 1) * np.array([1, 2, 3])
        pts_views = [np.array([_proj(P, X) for X in xyz_true]) for P in three_projection_matrices]
        xyz = triangulate_all(pts_views, three_projection_matrices)

        for i in range(n):
            assert np.linalg.norm(xyz[i] - xyz_true[i]) < 1e-3


# ---------------------------------------------------------------------------
# _project
# ---------------------------------------------------------------------------


class TestProject:
    def test_known_projection(self, camera_matrix, known_3d_point):
        """Project a point at depth 1000 and check the pixel position."""
        K = camera_matrix
        P = K @ np.hstack([np.eye(3), np.zeros((3, 1))])  # P = K [I|0]
        px = _project(P, known_3d_point)

        # Expected: (fx*X/Z + cx, fy*Y/Z + cy)
        X, Y, Z = known_3d_point
        expected_x = K[0, 0] * X / Z + K[0, 2]
        expected_y = K[1, 1] * Y / Z + K[1, 2]

        assert px == pytest.approx([expected_x, expected_y], abs=1e-6)

    def test_output_shape(self, three_projection_matrices, known_3d_point):
        px = _project(three_projection_matrices[0], known_3d_point)
        assert px.shape == (2,)


# ---------------------------------------------------------------------------
# _ba_residuals
# ---------------------------------------------------------------------------


class TestBaResiduals:
    def test_zero_residuals_on_perfect_data(self, three_projection_matrices, known_3d_point):
        """With perfect 2-D observations, residuals must be (near) zero."""
        n = 3
        xyz = known_3d_point + np.arange(n).reshape(-1, 1)
        pts_views = [np.array([_proj(P, X) for X in xyz]) for P in three_projection_matrices]
        res = _ba_residuals(xyz.ravel(), pts_views, three_projection_matrices)
        assert np.abs(res).max() < 1e-6

    def test_residual_vector_length(self, three_projection_matrices, known_3d_point):
        """Residual length must be n_points × n_views × 2."""
        n, m = 5, len(three_projection_matrices)
        xyz = known_3d_point + np.arange(n).reshape(-1, 1)
        pts_views = [np.array([_proj(P, X) for X in xyz]) for P in three_projection_matrices]
        res = _ba_residuals(xyz.ravel(), pts_views, three_projection_matrices)
        assert res.shape == (n * m * 2,)

    def test_nonzero_residuals_on_perturbed_points(self, three_projection_matrices, known_3d_point):
        n = 4
        xyz_true = known_3d_point + np.arange(n).reshape(-1, 1)
        pts_views = [np.array([_proj(P, X) for X in xyz_true]) for P in three_projection_matrices]
        xyz_noisy = xyz_true + 10.0  # perturb 3D points
        res = _ba_residuals(xyz_noisy.ravel(), pts_views, three_projection_matrices)
        assert np.abs(res).max() > 0.5


# ---------------------------------------------------------------------------
# bundle_adjust
# ---------------------------------------------------------------------------


class TestBundleAdjust:
    def test_reduces_reprojection_error(self, three_projection_matrices, known_3d_point):
        n = 10
        rng = np.random.default_rng(42)
        xyz_true = known_3d_point + rng.standard_normal((n, 3)) * 5.0
        pts_views = [np.array([_proj(P, X) for X in xyz_true]) for P in three_projection_matrices]
        xyz_noisy = xyz_true + rng.standard_normal((n, 3)) * 2.0

        err_before = compute_reprojection_error(xyz_noisy, pts_views, three_projection_matrices)
        xyz_refined, err_after = bundle_adjust(xyz_noisy, pts_views, three_projection_matrices)
        assert err_after < err_before

    def test_output_shape_preserved(self, three_projection_matrices, known_3d_point):
        n = 5
        rng = np.random.default_rng(0)
        xyz = known_3d_point + rng.standard_normal((n, 3)) * 3.0
        pts_views = [np.array([_proj(P, X) for X in xyz]) for P in three_projection_matrices]
        xyz_ref, _ = bundle_adjust(xyz, pts_views, three_projection_matrices)
        assert xyz_ref.shape == (n, 3)

    def test_convergence_on_perfect_initial_estimate(
        self, three_projection_matrices, known_3d_point
    ):
        """Starting from perfect points, BA must give near-zero reprojection error."""
        n = 5
        xyz_true = known_3d_point + np.arange(n).reshape(-1, 1)
        pts_views = [np.array([_proj(P, X) for X in xyz_true]) for P in three_projection_matrices]
        _, err = bundle_adjust(xyz_true.copy(), pts_views, three_projection_matrices)
        assert err < 1e-4


# ---------------------------------------------------------------------------
# compute_reprojection_error
# ---------------------------------------------------------------------------


class TestComputeReprojectionError:
    def test_zero_error_on_perfect_projection(self, three_projection_matrices, known_3d_point):
        n = 5
        xyz = known_3d_point + np.arange(n).reshape(-1, 1)
        pts_views = [np.array([_proj(P, X) for X in xyz]) for P in three_projection_matrices]
        err = compute_reprojection_error(xyz, pts_views, three_projection_matrices)
        assert err == pytest.approx(0.0, abs=1e-6)

    def test_error_positive_on_noisy_data(self, three_projection_matrices, known_3d_point):
        n = 5
        xyz = known_3d_point + np.arange(n).reshape(-1, 1)
        pts_views = [
            np.array([_proj(P, X) for X in xyz]) + np.random.randn(n, 2) * 3.0
            for P in three_projection_matrices
        ]
        err = compute_reprojection_error(xyz, pts_views, three_projection_matrices)
        assert err > 0.5


# ---------------------------------------------------------------------------
# load_tracked_points
# ---------------------------------------------------------------------------


class TestLoadTrackedPoints:
    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "pts.csv")
        pts_expected = np.array([[1.5, 2.7], [3.1, 4.9], [0.0, 100.0]])
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "x", "y"])
            for i, (x, y) in enumerate(pts_expected):
                w.writerow([i + 1, x, y])

        pts = load_tracked_points(path)
        np.testing.assert_allclose(pts, pts_expected)

    def test_output_shape(self, tmp_path):
        path = str(tmp_path / "pts.csv")
        n = 50
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "x", "y"])
            for i in range(n):
                w.writerow([i + 1, float(i), float(i * 2)])
        pts = load_tracked_points(path)
        assert pts.shape == (n, 2)

    def test_output_dtype_float64(self, tmp_path):
        path = str(tmp_path / "pts.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "x", "y"])
            w.writerow([1, 1.0, 2.0])
        pts = load_tracked_points(path)
        assert pts.dtype == np.float64


# ---------------------------------------------------------------------------
# load_camera_poses
# ---------------------------------------------------------------------------


class TestLoadCameraPoses:
    def test_loads_three_cameras(self, camera_poses_json):
        cameras = load_camera_poses(camera_poses_json)
        assert len(cameras) == 3

    def test_camera_ids_present(self, camera_poses_json):
        cameras = load_camera_poses(camera_poses_json)
        ids = [cam["id"] for cam in cameras]
        assert ids == ["left", "mid", "right"]

    def test_K_shape(self, camera_poses_json):
        cameras = load_camera_poses(camera_poses_json)
        for cam in cameras:
            assert cam["K"].shape == (3, 3)

    def test_R_shape(self, camera_poses_json):
        cameras = load_camera_poses(camera_poses_json)
        for cam in cameras:
            assert cam["R"].shape == (3, 3)

    def test_t_shape(self, camera_poses_json):
        cameras = load_camera_poses(camera_poses_json)
        for cam in cameras:
            assert cam["t"].shape == (3, 1)

    def test_projection_matrix_shape(self, camera_poses_json):
        cameras = load_camera_poses(camera_poses_json)
        for cam in cameras:
            assert cam["P"].shape == (3, 4)

    def test_projection_matrix_correct(self, camera_poses_json):
        """P must equal K @ [R | t]."""
        cameras = load_camera_poses(camera_poses_json)
        for cam in cameras:
            P_expected = cam["K"] @ np.hstack([cam["R"], cam["t"]])
            np.testing.assert_allclose(cam["P"], P_expected, atol=1e-10)

    def test_dist_is_1d_array(self, camera_poses_json):
        cameras = load_camera_poses(camera_poses_json)
        for cam in cameras:
            assert cam["dist"].ndim == 1
