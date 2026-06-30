#!/usr/bin/env python3
"""
3D tongue-tip reconstruction from multi-view tracked points.

Python replacement for tracking_in_3d.m (MATLAB Computer Vision Toolbox).

Pipeline
--------
1. Load 2D tracked points from three camera views (CSV output of tracking_tongue.py).
2. Load camera intrinsics (K, dist) and extrinsics (R, t) from a JSON file
   produced by the stereo-calibration mode of calib-camera.py.
3. Build 3×4 projection matrices  P_i = K_i · [R_i | t_i].
4. Triangulate each frame's tongue-tip position with the Direct Linear Transform
   (DLT) across all three views — equivalent to MATLAB's triangulateMultiview.
5. Refine the 3-D points by minimising total reprojection error with
   scipy.optimize.least_squares (simplified bundle adjustment: camera poses fixed).
   Equivalent to MATLAB's bundleAdjustment.
6. Visualise and optionally save the 3-D point cloud.

Camera-poses JSON format (produced by  calib-camera.py --stereo-folder)
------------------------------------------------------------------------
{
  "cameras": [
    {
      "id": "left",
      "K":    [[fx,0,cx],[0,fy,cy],[0,0,1]],
      "dist": [k1, k2, p1, p2, k3],
      "R":    [[...3×3 rotation matrix...]],
      "t":    [[tx],[ty],[tz]]
    },
    { "id": "mid",   ... },
    { "id": "right", ... }
  ]
}

Usage
-----
  python tracking_in_3d.py \\
      --cameras  camera_poses.json \\
      --left-csv left.csv \\
      --mid-csv  mid.csv  \\
      --right-csv right.csv \\
      --output-csv xyz.csv \\
      --save-plot reconstruction.png

  # Skip bundle adjustment (faster, less accurate)
  python tracking_in_3d.py ... --no-ba

  # Headless (no interactive window)
  python tracking_in_3d.py ... --no-display
"""

import argparse
import csv
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# Camera utilities
# ---------------------------------------------------------------------------


def load_camera_poses(json_path: str) -> list[dict]:
    """
    Load camera intrinsics and extrinsics from a JSON file.

    Returns a list of dicts, each containing:
      'id'   : str
      'K'    : (3,3) ndarray — intrinsic matrix
      'dist' : (1,5) ndarray — distortion coefficients
      'R'    : (3,3) ndarray — rotation (world → camera)
      't'    : (3,1) ndarray — translation (world → camera)
      'P'    : (3,4) ndarray — projection matrix K·[R|t]
    """
    with open(json_path) as f:
        data = json.load(f)

    cameras = []
    for cam in data["cameras"]:
        K = np.array(cam["K"], dtype=np.float64)
        dist = np.array(cam["dist"], dtype=np.float64).ravel()
        R = np.array(cam["R"], dtype=np.float64)
        t = np.array(cam["t"], dtype=np.float64).reshape(3, 1)
        Rt = np.hstack([R, t])
        P = K @ Rt
        cameras.append({"id": cam["id"], "K": K, "dist": dist, "R": R, "t": t, "P": P})
    return cameras


def load_tracked_points(csv_path: str) -> np.ndarray:
    """
    Load 2D tracked points from a CSV file produced by tracking_tongue.py.

    Returns an (N, 2) float64 array of (x, y) coordinates.
    """
    pts = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts.append([float(row["x"]), float(row["y"])])
    return np.array(pts, dtype=np.float64)


# ---------------------------------------------------------------------------
# Triangulation (DLT — equivalent to MATLAB triangulateMultiview)
# ---------------------------------------------------------------------------


def triangulate_point_dlt(
    pts_2d: list[tuple[float, float]],
    proj_mats: list[np.ndarray],
) -> np.ndarray:
    """
    Direct Linear Transform triangulation from N ≥ 2 views.

    Each view contributes two equations.  The solution is the right singular
    vector corresponding to the smallest singular value of A.

    Parameters
    ----------
    pts_2d    : list of (x, y) image coordinates, one per view
    proj_mats : list of (3,4) projection matrices, one per view

    Returns
    -------
    (3,) ndarray — triangulated 3-D point in world coordinates
    """
    A = []
    for (x, y), P in zip(pts_2d, proj_mats):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def triangulate_all(
    pts_views: list[np.ndarray],
    proj_mats: list[np.ndarray],
) -> np.ndarray:
    """
    Triangulate N tongue-tip positions from M camera views.

    Parameters
    ----------
    pts_views  : list of M arrays, each (N,2) — one per camera view
    proj_mats  : list of M (3,4) projection matrices

    Returns
    -------
    (N,3) ndarray of 3-D world points
    """
    n = len(pts_views[0])
    xyz = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        pts_2d = [(pts[i, 0], pts[i, 1]) for pts in pts_views]
        xyz[i] = triangulate_point_dlt(pts_2d, proj_mats)
        if (i + 1) % 100 == 0:
            print(f"  Triangulated {i + 1}/{n} points", end="\r")
    print()
    return xyz


# ---------------------------------------------------------------------------
# Bundle adjustment (scipy — equivalent to MATLAB bundleAdjustment)
# ---------------------------------------------------------------------------


def _project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project 3-D point X with (3,4) projection matrix P → (2,) image point."""
    h = P @ np.append(X, 1.0)
    return h[:2] / h[2]


def _ba_residuals(
    xyz_flat: np.ndarray,
    pts_views: list[np.ndarray],
    proj_mats: list[np.ndarray],
) -> np.ndarray:
    """
    Reprojection-error residuals for scipy.optimize.least_squares.

    Parameters are the flattened 3-D points (N×3).  Camera poses are fixed.
    Returns a (N × M × 2,) vector of (projected − observed) pixel errors.
    """
    n = xyz_flat.size // 3
    xyz = xyz_flat.reshape(n, 3)
    res = []
    for i, X in enumerate(xyz):
        for pts, P in zip(pts_views, proj_mats):
            proj = _project(P, X)
            res.append(proj[0] - pts[i, 0])
            res.append(proj[1] - pts[i, 1])
    return np.array(res, dtype=np.float64)


def bundle_adjust(
    xyz: np.ndarray,
    pts_views: list[np.ndarray],
    proj_mats: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    """
    Refine 3-D points by minimising reprojection error (camera poses fixed).

    Uses a robust Huber loss to reduce the influence of tracking outliers.
    Equivalent to MATLAB's bundleAdjustment (simplified: poses not refined).

    Parameters
    ----------
    xyz        : (N,3) initial 3-D points from DLT
    pts_views  : list of M (N,2) observed 2-D points
    proj_mats  : list of M (3,4) projection matrices

    Returns
    -------
    (xyz_refined, mean_reprojection_error_px)
    """
    print("  Running bundle adjustment…")
    result = least_squares(
        _ba_residuals,
        xyz.ravel(),
        args=(pts_views, proj_mats),
        method="trf",
        loss="huber",
        f_scale=1.0,  # Huber threshold in pixels
        max_nfev=500,
        verbose=0,
    )
    xyz_refined = result.x.reshape(-1, 3)

    # Compute mean reprojection error
    n = len(xyz_refined)
    m = len(proj_mats)
    residuals = result.fun.reshape(n, m, 2)
    mean_error = float(np.linalg.norm(residuals, axis=2).mean())
    return xyz_refined, mean_error


# ---------------------------------------------------------------------------
# Reprojection error (without BA — for diagnostics)
# ---------------------------------------------------------------------------


def compute_reprojection_error(
    xyz: np.ndarray,
    pts_views: list[np.ndarray],
    proj_mats: list[np.ndarray],
) -> float:
    """Return mean reprojection error across all views and points (pixels)."""
    errors = []
    for pts, P in zip(pts_views, proj_mats):
        for X, obs in zip(xyz, pts):
            proj = _project(P, X)
            errors.append(np.linalg.norm(proj - obs))
    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# Visualisation (replaces MATLAB pcshow / plotCamera)
# ---------------------------------------------------------------------------


def visualise_3d(
    xyz: np.ndarray,
    cameras: list[dict] | None = None,
    title: str = "3D Tongue-Tip Reconstruction",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Scatter-plot the 3-D tongue-tip trajectory.

    Points are coloured by frame index (early → late = blue → yellow).
    Camera centres are drawn as red triangles if *cameras* is provided.

    Equivalent to MATLAB's pcshow + plotCamera.
    """
    matplotlib.use("Agg" if not show else matplotlib.get_backend())

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Tongue-tip trajectory
    n = len(xyz)
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=15, alpha=0.7, depthshade=True)
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "b-", alpha=0.2, linewidth=0.8)

    # Camera centres
    if cameras:
        for cam in cameras:
            # Centre in world coords = -R^T · t
            centre = (-cam["R"].T @ cam["t"]).ravel()
            ax.scatter(*centre, s=150, marker="^", color="red", zorder=5)
            ax.text(centre[0], centre[1], centre[2], f"  {cam['id']}", fontsize=9)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(title)
    ax.invert_yaxis()  # match MATLAB 'VerticalAxisDir', 'down'

    # Colour bar for frame index
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=1, vmax=n))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.5, label="Frame index")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot  : {save_path}")

    if show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser for tracking_in_3d."""
    parser = argparse.ArgumentParser(
        description=(
            "3D tongue-tip reconstruction from multi-view tracked points. "
            "Python replacement for tracking_in_3d.m."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cameras",
        required=True,
        metavar="camera_poses.json",
        help=(
            "JSON file with camera intrinsics and extrinsics for all views "
            "(produced by: python calib-camera.py ... --stereo-folder ...)"
        ),
    )
    parser.add_argument(
        "--left-csv",
        required=True,
        metavar="FILE",
        help="CSV of tracked 2-D points from the left-view camera",
    )
    parser.add_argument(
        "--mid-csv",
        required=True,
        metavar="FILE",
        help="CSV of tracked 2-D points from the mid-view camera",
    )
    parser.add_argument(
        "--right-csv",
        required=True,
        metavar="FILE",
        help="CSV of tracked 2-D points from the right-view camera",
    )
    parser.add_argument(
        "--no-ba", action="store_true", help="Skip bundle adjustment (faster, less accurate)"
    )
    parser.add_argument(
        "--output-csv", metavar="FILE", help="Save 3-D coordinates to CSV (columns: frame, x, y, z)"
    )
    parser.add_argument(
        "--save-plot", metavar="FILE", help="Save 3-D scatter plot to an image file (PNG/PDF/…)"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Do not open an interactive plot window"
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the 3-D reconstruction pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # --- Validate inputs --------------------------------------------------
    for path, label in [
        (args.cameras, "camera-poses JSON"),
        (args.left_csv, "left CSV"),
        (args.mid_csv, "mid CSV"),
        (args.right_csv, "right CSV"),
    ]:
        if not os.path.exists(path):
            print(f"Error: {label} not found: {path}")
            sys.exit(1)

    # --- Load data --------------------------------------------------------
    print("Loading camera parameters…")
    cameras = load_camera_poses(args.cameras)
    if len(cameras) < 3:
        print(f"Error: need ≥ 3 cameras in {args.cameras}, found {len(cameras)}.")
        sys.exit(1)

    # Expect cameras ordered as left, mid, right (or use first three)
    proj_mats = [cam["P"] for cam in cameras[:3]]
    cam_ids = [cam["id"] for cam in cameras[:3]]
    print(f"  Camera order: {cam_ids}")

    print("Loading tracked 2-D points…")
    pts_left = load_tracked_points(args.left_csv)
    pts_mid = load_tracked_points(args.mid_csv)
    pts_right = load_tracked_points(args.right_csv)

    n_pts = len(pts_mid)
    if len(pts_left) != n_pts or len(pts_right) != n_pts:
        print(
            f"Error: point-count mismatch — "
            f"left={len(pts_left)}, mid={n_pts}, right={len(pts_right)}.\n"
            f"All three CSV files must have the same number of rows."
        )
        sys.exit(1)

    print(f"Starting 3-D reconstruction with {n_pts} points…")

    pts_views = [pts_left, pts_mid, pts_right]

    # --- Triangulate ------------------------------------------------------
    print("Triangulating (DLT)…")
    xyz = triangulate_all(pts_views, proj_mats)

    init_error = compute_reprojection_error(xyz, pts_views, proj_mats)
    print(f"  Mean reprojection error (before BA): {init_error:.4f} px")

    # --- Bundle adjustment ------------------------------------------------
    if not args.no_ba:
        xyz, mean_error = bundle_adjust(xyz, pts_views, proj_mats)
        print(f"  Mean reprojection error (after  BA): {mean_error:.4f} px")
    else:
        print("  Bundle adjustment skipped (--no-ba).")
        mean_error = init_error

    print("3-D reconstruction complete.")
    print(f"  X range: [{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}] mm")
    print(f"  Y range: [{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}] mm")
    print(f"  Z range: [{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}] mm")

    # --- Save CSV ---------------------------------------------------------
    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "x_mm", "y_mm", "z_mm"])
            for i, (x, y, z) in enumerate(xyz):
                writer.writerow([i + 1, round(float(x), 4), round(float(y), 4), round(float(z), 4)])
        print(f"Saved CSV   : {args.output_csv}")

    # --- Visualise --------------------------------------------------------
    show_window = not args.no_display
    if args.save_plot or show_window:
        visualise_3d(
            xyz,
            cameras=cameras[:3],
            title=f"3D Tongue-Tip Reconstruction  (mean reproj. error: {mean_error:.3f} px)",
            save_path=args.save_plot,
            show=show_window,
        )


if __name__ == "__main__":
    main()
