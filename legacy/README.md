# Legacy MATLAB Scripts

The two MATLAB scripts in this directory are **superseded** by their Python equivalents and are kept here for historical reference only.

| MATLAB script | Python replacement | Notes |
|---|---|---|
| `tracking_tongue.m` | `../tracking_tongue.py` | Optical flow 2D tracking |
| `tracking_in_3d.m` | `../tracking_in_3d.py` | Multi-view triangulation + bundle adjustment |

## Why we replaced them

- **MATLAB is not required.** All algorithms (`opticalFlowFarneback`, KLT tracking, DLT triangulation, bundle adjustment) are available in `opencv-python` and `scipy`, both of which are already in `requirements.txt`.
- **Camera calibration** (previously needing the MATLAB `cameraCalibrator` GUI) is now fully handled by `calib-camera.py --stereo-folder`.
- The Python pipeline is cross-platform, open-source, Docker-compatible, and CI-testable.

## What to use instead

```bash
# Step 1 — Camera calibration (single camera)
python calib-camera.py ./camera_left jpg 8 8 20

# Step 1b — Stereo calibration (produces camera_poses.json)
python calib-camera.py ./camera_left jpg 8 8 20 --stereo-folder ./camera_mid --output-prefix left_mid

# Step 2 — 2D tongue-tip tracking (one run per camera view)
python tracking_tongue.py --video video_left.avi  --view left  --output-csv left.csv
python tracking_tongue.py --video video_mid.avi   --view mid   --output-csv mid.csv
python tracking_tongue.py --video video_right.avi --view right --output-csv right.csv

# Step 3 — 3D reconstruction
python tracking_in_3d.py \
    --cameras camera_poses.json \
    --left-csv left.csv --mid-csv mid.csv --right-csv right.csv \
    --output-csv xyz.csv --save-plot reconstruction.png
```

See the main [README](../README.md) for full documentation.
