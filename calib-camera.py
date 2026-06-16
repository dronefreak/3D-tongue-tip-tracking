#!/usr/bin/env python

"""
Camera calibration using a checkerboard pattern.

Reference:
  https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

Usage:
  python calib-camera.py <folder> <image_type> <num_rows> <num_cols> <cell_dimension_mm>

Example:
  python calib-camera.py ./camera_01 jpg 8 8 20

  --help for all options.
"""

import argparse
import numpy as np
import cv2
import glob
import sys
import os

# Number of iterations for subpixel corner refinement
MAX_CORNER_ITERATIONS = 30
# Minimum images required for a valid calibration
MIN_IMAGES_REQUIRED = 9


def initialize_arg_parser():
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Calibrate a camera using a checkerboard pattern.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("folder", type=str,
                        help="Path to folder containing calibration images")
    parser.add_argument("image_type", type=str,
                        help="Image file extension (e.g. jpg, png)")
    parser.add_argument("rows", type=int,
                        help="Number of internal corners along the checkerboard rows")
    parser.add_argument("cols", type=int,
                        help="Number of internal corners along the checkerboard columns")
    parser.add_argument("dimension", type=float,
                        help="Physical size of each checkerboard square in mm")
    parser.add_argument(
        "--stereo-folder", type=str, default=None, metavar="FOLDER",
        help=(
            "Path to a second camera's calibration images for stereo calibration. "
            "Image pairs are matched by filename (or sort order). "
            "Produces R, T, E, F matrices and a camera_poses.json file."
        ),
    )
    parser.add_argument(
        "--output-prefix", type=str, default="stereo",
        help="Filename prefix for stereo output files (e.g. 'left_mid' → left_mid_R.txt …)",
    )
    return parser


def validate_inputs(folder, image_type, rows, cols, dimension):
    """Validate calibration parameters and return them (raises ValueError on bad input)."""
    if rows < 2:
        raise ValueError(f"rows must be >= 2, got {rows}")
    if cols < 2:
        raise ValueError(f"cols must be >= 2, got {cols}")
    if dimension <= 0:
        raise ValueError(f"dimension must be > 0, got {dimension}")

    # Strip a leading dot from the extension if supplied (e.g. ".jpg" → "jpg")
    image_type = image_type.lstrip(".")
    if not image_type:
        raise ValueError("image_type must not be empty")

    # Normalise folder path but do not restrict to relative paths
    folder = os.path.normpath(folder)

    return folder, image_type, rows, cols, dimension

import json


# ---------------------------------------------------------------------------
# Core calibration helpers
# ---------------------------------------------------------------------------

def _find_corners(
    images: list[str],
    n_cols: int,
    n_rows: int,
    objp: np.ndarray,
    criteria: tuple,
    interactive: bool = True,
) -> tuple[list, list, tuple | None, str | None, int]:
    """
    Detect checkerboard corners in *images* and (optionally) let the user
    accept/reject each detected pattern interactively.

    Returns
    -------
    (objpoints, imgpoints, image_size, last_rejected_path, n_accepted)
    """
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    img_size: tuple | None = None
    img_not_good: str | None = images[0] if images else None
    n_accepted = 0

    for fname in images:
        if "calibresult" in fname:
            continue

        img = cv2.imread(fname)
        if img is None:
            print(f"  Warning: could not read {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]   # (width, height)

        print(f"  Processing: {fname}")
        ret, corners = cv2.findChessboardCorners(gray, (n_cols, n_rows), None)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            if interactive:
                cv2.drawChessboardCorners(img, (n_cols, n_rows), corners2, ret)
                cv2.imshow("Checkerboard – ESC to skip, ENTER to accept", img)
                print("  Pattern found!  Press ENTER to accept or ESC to skip.")
                k = cv2.waitKey(0) & 0xFF
                if k == 27:   # ESC
                    print("  Skipped.")
                    img_not_good = fname
                    continue

            n_accepted += 1
            objpoints.append(objp)
            imgpoints.append(corners2)
            print(f"  Accepted ({n_accepted} so far).")
        else:
            img_not_good = fname
            print("  Checkerboard not found.")

    if interactive:
        cv2.destroyAllWindows()

    return objpoints, imgpoints, img_size, img_not_good, n_accepted


def _save_undistorted(img_path: str, mtx: np.ndarray, dist: np.ndarray, out_folder: str) -> None:
    """Save an undistorted sample image into *out_folder*."""
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    x, y, rw, rh = roi
    dst = dst[y: y + rh, x: x + rw]
    out_path = os.path.join(out_folder, "calibresult.png")
    cv2.imwrite(out_path, dst)
    print(f"  Undistorted sample: {out_path}")


def _reprojection_error(
    objpoints: list,
    imgpoints: list,
    rvecs: list,
    tvecs: list,
    mtx: np.ndarray,
    dist: np.ndarray,
) -> float:
    """Compute mean reprojection error across all calibration images."""
    total = 0.0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        total += cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
    return total / len(objpoints)


# ---------------------------------------------------------------------------
# Single-camera calibration
# ---------------------------------------------------------------------------

def calibrate_single(
    folder: str,
    image_type: str,
    n_rows: int,
    n_cols: int,
    objp: np.ndarray,
    criteria: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calibrate a single camera from checkerboard images in *folder*.

    Saves cameraMatrix.txt, cameraDistortion.txt, and calibresult.png to *folder*.

    Returns
    -------
    (camera_matrix, dist_coeffs)
    """
    pattern = os.path.join(folder, f"*.{image_type}")
    images = sorted(glob.glob(pattern))

    if len(images) < MIN_IMAGES_REQUIRED:
        raise ValueError(
            f"Found only {len(images)} image(s) in '{folder}'; "
            f"need at least {MIN_IMAGES_REQUIRED}."
        )

    print(f"Found {len(images)} image(s) in {folder}")
    obj_pts, img_pts, img_size, img_not_good, n_accepted = _find_corners(
        images, n_cols, n_rows, objp, criteria, interactive=True
    )

    if n_accepted < 2:
        raise ValueError(
            f"Only {n_accepted} pattern(s) accepted; need at least 2 for calibration."
        )

    print(f"\nCalibrating from {n_accepted} image(s)…")
    _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, img_size, None, None
    )

    mean_err = _reprojection_error(obj_pts, img_pts, rvecs, tvecs, mtx, dist)
    print(f"\nCamera matrix:\n{mtx}")
    print(f"\nDistortion coefficients:\n{dist.ravel()}")
    print(f"\nMean reprojection error: {mean_err:.4f} px")

    np.savetxt(os.path.join(folder, "cameraMatrix.txt"), mtx, delimiter=",")
    np.savetxt(os.path.join(folder, "cameraDistortion.txt"), dist, delimiter=",")
    print(f"\nCalibration files saved to: {folder}/")

    if img_not_good:
        _save_undistorted(img_not_good, mtx, dist, folder)

    return mtx, dist


# ---------------------------------------------------------------------------
# Stereo calibration
# ---------------------------------------------------------------------------

def calibrate_stereo(
    folder1: str,
    folder2: str,
    image_type: str,
    n_rows: int,
    n_cols: int,
    objp: np.ndarray,
    criteria: tuple,
    output_prefix: str = "stereo",
) -> tuple:
    """
    Stereo-calibrate two cameras using matched checkerboard image pairs.

    Images in *folder1* and *folder2* are matched by filename (or sort order if
    filenames differ).  Both cameras must detect the pattern for a pair to be used.

    Outputs
    -------
    {output_prefix}_R.txt, _T.txt, _E.txt, _F.txt
    {output_prefix}_camera_poses.json   ← ready for tracking_in_3d.py

    Returns
    -------
    (mtx1, dist1, mtx2, dist2, R, T, E, F)
    """
    images1 = sorted(glob.glob(os.path.join(folder1, f"*.{image_type}")))
    images2 = sorted(glob.glob(os.path.join(folder2, f"*.{image_type}")))

    if not images1 or not images2:
        raise ValueError(
            f"No '{image_type}' images found in one or both folders:\n"
            f"  {folder1} ({len(images1)} images)\n"
            f"  {folder2} ({len(images2)} images)"
        )

    # Match by filename; fall back to positional pairing
    fnames1 = {os.path.basename(p): p for p in images1}
    fnames2 = {os.path.basename(p): p for p in images2}
    common_names = sorted(fnames1.keys() & fnames2.keys())

    if common_names:
        pairs = [(fnames1[n], fnames2[n]) for n in common_names]
        print(f"Matched {len(pairs)} image pair(s) by filename.")
    else:
        n = min(len(images1), len(images2))
        pairs = list(zip(images1[:n], images2[:n]))
        print(f"Warning: no matching filenames — pairing by sort order ({n} pair(s)).")

    obj_pts_all: list = []
    img_pts1_all: list = []
    img_pts2_all: list = []
    img_size: tuple | None = None
    n_good = 0

    for p1, p2 in pairs:
        img1 = cv2.imread(p1)
        img2 = cv2.imread(p2)
        if img1 is None or img2 is None:
            print(f"  ✗ Could not read pair: {os.path.basename(p1)}")
            continue

        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img_size = g1.shape[::-1]

        ret1, c1 = cv2.findChessboardCorners(g1, (n_cols, n_rows), None)
        ret2, c2 = cv2.findChessboardCorners(g2, (n_cols, n_rows), None)

        if ret1 and ret2:
            c1 = cv2.cornerSubPix(g1, c1, (11, 11), (-1, -1), criteria)
            c2 = cv2.cornerSubPix(g2, c2, (11, 11), (-1, -1), criteria)
            obj_pts_all.append(objp)
            img_pts1_all.append(c1)
            img_pts2_all.append(c2)
            n_good += 1
            print(f"  ✓ {os.path.basename(p1)}")
        else:
            print(f"  ✗ {os.path.basename(p1)} (pattern not found in both views)")

    if n_good < MIN_IMAGES_REQUIRED:
        raise ValueError(
            f"Only {n_good} valid pair(s); need at least {MIN_IMAGES_REQUIRED}."
        )

    print(f"\nCalibrating individual cameras from {n_good} pairs…")
    _, mtx1, dist1, _, _ = cv2.calibrateCamera(obj_pts_all, img_pts1_all, img_size, None, None)
    _, mtx2, dist2, _, _ = cv2.calibrateCamera(obj_pts_all, img_pts2_all, img_size, None, None)

    print("Running stereo calibration (fixed intrinsics)…")
    rms, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
        obj_pts_all, img_pts1_all, img_pts2_all,
        mtx1, dist1, mtx2, dist2, img_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    print(f"  Stereo RMS reprojection error: {rms:.4f} px")

    # Save raw matrices
    for name, mat in [("R", R), ("T", T), ("E", E), ("F", F)]:
        np.savetxt(f"{output_prefix}_{name}.txt", mat, delimiter=",")
    print(f"  Saved R, T, E, F → {output_prefix}_{{R,T,E,F}}.txt")

    # Build camera_poses.json (camera 1 = world reference: R=I, t=0)
    poses = {
        "cameras": [
            {
                "id": os.path.basename(folder1),
                "K":    mtx1.tolist(),
                "dist": dist1.ravel().tolist(),
                "R":    np.eye(3).tolist(),
                "t":    [[0.0], [0.0], [0.0]],
            },
            {
                "id": os.path.basename(folder2),
                "K":    mtx2.tolist(),
                "dist": dist2.ravel().tolist(),
                "R":    R.tolist(),
                "t":    T.tolist(),
            },
        ]
    }
    poses_path = f"{output_prefix}_camera_poses.json"
    with open(poses_path, "w") as f:
        json.dump(poses, f, indent=2)
    print(f"  Camera poses JSON → {poses_path}")
    print(
        f"\nTip: for a 3-camera setup, run stereo calibration twice:\n"
        f"  python calib-camera.py left jpg {n_rows} {n_cols} <dim> --stereo-folder mid --output-prefix left_mid\n"
        f"  python calib-camera.py mid  jpg {n_rows} {n_cols} <dim> --stereo-folder right --output-prefix mid_right\n"
        f"Then combine the two JSON files into a single camera_poses.json for tracking_in_3d.py."
    )

    return mtx1, dist1, mtx2, dist2, R, T, E, F


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = initialize_arg_parser()
    args = parser.parse_args()

    try:
        folder, image_type, n_rows, n_cols, dimension = validate_inputs(
            args.folder, args.image_type, args.rows, args.cols, args.dimension
        )
    except ValueError as e:
        print(f"Invalid input: {e}")
        sys.exit(1)

    if not os.path.exists(folder):
        print(f"Error: folder not found: {folder}")
        sys.exit(1)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        MAX_CORNER_ITERATIONS,
        0.001,
    )
    objp = np.zeros((n_rows * n_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:n_cols, 0:n_rows].T.reshape(-1, 2)

    if args.stereo_folder:
        stereo_folder = os.path.normpath(args.stereo_folder)
        if not os.path.exists(stereo_folder):
            print(f"Error: stereo folder not found: {stereo_folder}")
            sys.exit(1)
        try:
            calibrate_stereo(
                folder, stereo_folder, image_type, n_rows, n_cols, objp, criteria,
                output_prefix=args.output_prefix,
            )
        except ValueError as e:
            print(f"Stereo calibration failed: {e}")
            sys.exit(1)
    else:
        try:
            calibrate_single(folder, image_type, n_rows, n_cols, objp, criteria)
        except ValueError as e:
            print(f"Calibration failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
