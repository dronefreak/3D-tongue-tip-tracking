#!/usr/bin/env python3
"""
2D tongue-tip tracking via optical flow.

Python replacement for tracking_tongue.m (MATLAB Computer Vision Toolbox).

Algorithm
---------
1. Dense Farnebäck optical flow is estimated on every frame.
2. The pixel with the highest flow magnitude gives the raw tongue-tip candidate.
3. A pyramidal Lucas-Kanade (KLT) tracker refines the position between frames.
4. When the peak flow magnitude exceeds *flow_threshold* the tracker is
   re-initialised at the new optical-flow maximum — mirroring the threshold
   logic (hardcoded at 6 in the original MATLAB script).

Optical-flow parameters are chosen to match MATLAB's opticalFlowFarneback
defaults.  KLT parameters match vision.PointTracker('MaxBidirectionalError',1).

Outputs
-------
  * CSV  : frame, x, y  (one row per frame processed)
  * JSON : full metadata + per-frame coordinate list

Usage
-----
  # Named view with preset ROI
  python tracking_tongue.py -v video.avi --view mid --output-csv mid.csv

  # Custom ROI (x y w h) and JSON output
  python tracking_tongue.py -v video.avi --roi 367 350 361 365 --output-json mid.json

  # Headless batch processing
  python tracking_tongue.py -v video.avi --view left --no-display --output-csv left.csv
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Pre-defined ROIs for the three camera views used in the original study.
# Format: (x, y, w, h) — top-left corner + size (same as OpenCV ROI convention).
# These match the values hardcoded in tracking_tongue.m.
PRESET_ROIS: dict[str, tuple[int, int, int, int]] = {
    "mid":   (367, 350, 361, 365),
    "left":  (135, 479, 367, 261),
    "right": (595, 431, 412, 302),
}

# Farnebäck optical-flow parameters — mirror MATLAB opticalFlowFarneback defaults.
FARNEBACK_PARAMS: dict = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)

# KLT tracker parameters — mirror vision.PointTracker defaults.
KLT_WIN_SIZE = (15, 15)
KLT_MAX_LEVEL = 3
KLT_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
# Matches vision.PointTracker('MaxBidirectionalError', 1)
KLT_MAX_BIDIRECTIONAL_ERROR: float = 1.0

# Flow magnitude threshold for tracker re-initialisation (was hardcoded as 6 in MATLAB).
DEFAULT_FLOW_THRESHOLD: float = 6.0


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def imsharpen(gray: np.ndarray, amount: float = 0.8, sigma: float = 2.5) -> np.ndarray:
    """
    Unsharp-mask sharpening — equivalent to MATLAB's imsharpen with default params.

    Parameters
    ----------
    gray   : single-channel uint8 image
    amount : sharpening strength (MATLAB default: 0.8)
    sigma  : Gaussian blur radius  (MATLAB default: 2.5)
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)


def crop_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Crop *frame* to *roi* = (x, y, w, h) — equivalent to MATLAB's imcrop."""
    x, y, w, h = roi
    return frame[y: y + h, x: x + w]


# ---------------------------------------------------------------------------
# KLT tracking
# ---------------------------------------------------------------------------

def klt_track(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    point: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """
    Track a single point using pyramidal Lucas-Kanade with bidirectional validation.

    Mirrors MATLAB vision.PointTracker with MaxBidirectionalError = 1.

    Parameters
    ----------
    prev_gray : previous frame (grayscale)
    gray      : current frame (grayscale)
    point     : (2,) float32 array — (x, y) to track

    Returns
    -------
    (new_point, is_valid) — new_point is the original point if tracking failed.
    """
    pts = point.reshape(1, 1, 2).astype(np.float32)

    new_pts, fwd_status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, pts, None,
        winSize=KLT_WIN_SIZE,
        maxLevel=KLT_MAX_LEVEL,
        criteria=KLT_CRITERIA,
    )

    if fwd_status is None or fwd_status[0, 0] == 0:
        return point, False

    # Bidirectional check: track back and measure round-trip drift
    back_pts, bwd_status, _ = cv2.calcOpticalFlowPyrLK(
        gray, prev_gray, new_pts, None,
        winSize=KLT_WIN_SIZE,
        maxLevel=KLT_MAX_LEVEL,
        criteria=KLT_CRITERIA,
    )

    if bwd_status is None or bwd_status[0, 0] == 0:
        return point, False

    drift = float(np.linalg.norm(pts[0, 0] - back_pts[0, 0]))
    if drift > KLT_MAX_BIDIRECTIONAL_ERROR:
        return point, False

    return new_pts[0, 0], True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "2D tongue-tip tracking via optical flow. "
            "Python replacement for tracking_tongue.m."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v", "--video", required=True,
        help="Path to input video file (.avi, .mp4, …)",
    )

    roi_group = parser.add_mutually_exclusive_group()
    roi_group.add_argument(
        "--view", choices=list(PRESET_ROIS.keys()),
        help="Named camera view (uses preset ROI from the original study)",
    )
    roi_group.add_argument(
        "--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
        help="Custom region of interest: top-left x y, then width height",
    )

    parser.add_argument(
        "--flow-threshold", type=float, default=DEFAULT_FLOW_THRESHOLD,
        help="Optical-flow magnitude threshold for tracker re-initialisation",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable live preview window (faster batch processing)",
    )
    parser.add_argument(
        "--output-csv", type=str, metavar="FILE",
        help="Save tracked coordinates to CSV (columns: frame, x, y)",
    )
    parser.add_argument(
        "--output-json", type=str, metavar="FILE",
        help="Save tracked coordinates and metadata to JSON",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # --- Validate inputs ---------------------------------------------------
    if not os.path.exists(args.video):
        print(f"Error: video file not found: {args.video}")
        sys.exit(1)

    roi: tuple[int, int, int, int] | None
    if args.roi:
        roi = tuple(args.roi)
    elif args.view:
        roi = PRESET_ROIS[args.view]
    else:
        roi = None  # use full frame

    # --- Open video -------------------------------------------------------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: could not open video: {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    print(f"Video      : {args.video}")
    print(f"Frames     : {total_frames}   FPS: {fps:.2f}")
    print(f"ROI        : {roi if roi else 'full frame'}")
    print(f"Flow thr.  : {args.flow_threshold}")

    # --- Bootstrap: read first frame --------------------------------------
    ret, frame = cap.read()
    if not ret:
        print("Error: could not read the first frame.")
        cap.release()
        sys.exit(1)

    if roi:
        frame = crop_roi(frame, roi)
        if frame.size == 0:
            print(f"Error: ROI {roi} produces an empty frame on the first frame "
                  f"(video is {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
                  f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}).")
            cap.release()
            sys.exit(1)

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialise tracker at the frame centre (same implicit behaviour as MATLAB)
    h0, w0 = prev_gray.shape
    tracked_pt = np.array([w0 // 2, h0 // 2], dtype=np.float32)

    # --- Pre-allocate result arrays --------------------------------------
    xs = np.zeros(total_frames, dtype=np.float32)
    ys = np.zeros(total_frames, dtype=np.float32)
    frame_idx = 0

    print(f"\nTracking {total_frames} frames…  (press q to stop preview)")

    # --- Main tracking loop ----------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if roi:
            frame = crop_roi(frame, roi)
            if frame.size == 0:
                # ROI is outside the video frame — skip this frame
                frame_idx += 1
                xs[frame_idx - 1] = tracked_pt[0]
                ys[frame_idx - 1] = tracked_pt[1]
                prev_gray = prev_gray   # unchanged
                continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = imsharpen(gray)

        # Dense optical flow (Farnebäck — mirrors MATLAB opticalFlowFarneback)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **FARNEBACK_PARAMS)
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        max_mag = float(magnitude.max())

        if max_mag >= args.flow_threshold:
            # High motion: re-initialise at the pixel of maximum flow magnitude
            flat_idx = int(magnitude.argmax())
            row, col = np.unravel_index(flat_idx, magnitude.shape)
            tracked_pt = np.array([col, row], dtype=np.float32)
        else:
            # Low motion: refine with KLT point tracker
            new_pt, valid = klt_track(prev_gray, gray, tracked_pt)
            if valid:
                tracked_pt = new_pt

        xs[frame_idx] = tracked_pt[0]
        ys[frame_idx] = tracked_pt[1]
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total_frames} frames processed", end="\r")

        # Live preview
        if not args.no_display:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cx_i, cy_i = int(tracked_pt[0]), int(tracked_pt[1])
            cv2.circle(vis, (cx_i, cy_i), 10, (0, 0, 255), 2)
            cv2.putText(vis, f"({cx_i},{cy_i})", (cx_i + 12, cy_i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.imshow("Tongue Tracking – press q to stop", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nUser interrupted.")
                break

        prev_gray = gray

    cap.release()
    # Some OpenCV builds (headless / no-GUI) raise on destroyAllWindows — ignore it
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    # Trim to actual frame count
    xs = xs[:frame_idx]
    ys = ys[:frame_idx]

    print(f"\nTracking complete: {frame_idx} frames processed.")
    print(f"X range: [{xs.min():.1f}, {xs.max():.1f}]")
    print(f"Y range: [{ys.min():.1f}, {ys.max():.1f}]")

    # --- Export ----------------------------------------------------------
    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "x", "y"])
            for i in range(frame_idx):
                writer.writerow([i + 1, float(xs[i]), float(ys[i])])
        print(f"Saved CSV  : {args.output_csv}")

    if args.output_json:
        data = {
            "video": args.video,
            "view": args.view or "custom",
            "roi": list(roi) if roi else None,
            "total_frames": frame_idx,
            "fps": fps,
            "flow_threshold": args.flow_threshold,
            "points": [
                {"frame": i + 1, "x": float(xs[i]), "y": float(ys[i])}
                for i in range(frame_idx)
            ],
        }
        with open(args.output_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved JSON : {args.output_json}")

    if not args.output_csv and not args.output_json:
        print("Tip: add --output-csv or --output-json to save the tracked coordinates.")


if __name__ == "__main__":
    main()
