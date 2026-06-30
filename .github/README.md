# Optical Flow based Tongue Tip Tracking in 3D

A novel method for tracking the tip of the tongue in 3-dimensions for medical applications, using optical flow and facial landmark detection. The repository includes a detailed [research report](ipcv_report.pdf) explaining the algorithm and its motivation.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Full Pipeline](#full-pipeline)
- [Advanced Usage](#advanced-usage)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## Features

- 🎯 **Full Python pipeline** — no MATLAB required
- 🚀 **Optimized tracking** with Farnebäck optical flow + KLT refinement
- 📐 **3D reconstruction** via Direct Linear Transform + bundle adjustment
- 🎥 **Multi-view support** — left, mid, and right camera presets
- 💾 **Data export** to CSV and JSON formats
- ⚡ **Batch / headless mode** for server-side processing
- 🧪 **129 automated tests** covering the full pipeline
- 🖥️ **Optional GUI** (`tongue_tracking_gui.py`) for desktop use

## Requirements

- Python 3.9 or higher
- Dependencies in `requirements.txt`:
  - `opencv-python >= 4.5.0`
  - `numpy >= 1.19.0`
  - `matplotlib >= 3.3.0`
  - `dlib >= 19.21.0`
  - `imutils >= 0.5.4`
  - `scipy >= 1.5.0`

> **Note:** `dlib` requires a C++ compiler. See [Troubleshooting](#dlib-installation-fails) below.

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/dronefreak/3D-tongue-tip-tracking.git
cd 3D-tongue-tip-tracking

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the facial landmark model (~40 MB)
curl -L https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks_GTX.dat.bz2 \
  -o shape_predictor_68_face_landmarks_GTX.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks_GTX.dat.bz2
```

The GTX shape predictor is the official improved dlib model — smaller, faster, and more accurate than previous versions.

## Quick Start

### 2D landmark detection from video

```bash
python src/facial_landmarks_video.py \
  --shape-predictor shape_predictor_68_face_landmarks_GTX.dat \
  --video your_video.avi \
  --export-csv results.csv
```

### 2D tongue-tip tracking (optical flow)

```bash
python src/tracking_tongue.py \
  --video your_video.avi \
  --view mid \
  --output-csv mid.csv
```

### 3D reconstruction from three views

```bash
python src/tracking_in_3d.py \
  --cameras camera_poses.json \
  --left-csv left.csv \
  --mid-csv mid.csv \
  --right-csv right.csv \
  --output-csv xyz.csv \
  --save-plot reconstruction.png
```

## Full Pipeline

### Step 1 — Camera Calibration

Calibrate each camera individually, then run stereo calibration to get the
relative pose between cameras:

```bash
# Single-camera calibration (repeat for each camera)
python src/calib-camera.py ./camera_01 jpg 8 8 20

# Stereo calibration (produces camera_poses.json)
python src/calib-camera.py ./stereo_images jpg 8 8 20 \
  --stereo-folder ./stereo_pairs \
  --output-prefix camera_poses
```

Outputs:
- `cameraMatrix.txt` — intrinsic parameters
- `cameraDistortion.txt` — lens distortion coefficients
- `camera_poses.json` — extrinsics for all views (input to Step 3)

### Step 2 — 2D Tracking (per camera view)

```bash
# Track each view separately
python src/tracking_tongue.py -v video_mid.avi   --view mid   --no-display --output-csv mid.csv
python src/tracking_tongue.py -v video_left.avi  --view left  --no-display --output-csv left.csv
python src/tracking_tongue.py -v video_right.avi --view right --no-display --output-csv right.csv
```

Custom ROI (x y w h):
```bash
python src/tracking_tongue.py -v video.avi --roi 367 350 361 365 --output-csv mid.csv
```

### Step 3 — 3D Reconstruction

```bash
python src/tracking_in_3d.py \
  --cameras  camera_poses.json \
  --left-csv left.csv \
  --mid-csv  mid.csv \
  --right-csv right.csv \
  --output-csv xyz.csv \
  --save-plot reconstruction.png
```

The script triangulates 3D tongue-tip coordinates using DLT across all three
views and refines them with bundle adjustment (skip with `--no-ba` for speed).

**Visualization examples:**

![Shape Detector](image.png)

![3D](3d.png)

![3D-R](10.png)

![3D-R](9.png)

## Advanced Usage

### Facial landmark detection options

```bash
facial_landmarks_video.py [-h] -p SHAPE_PREDICTOR [-v VIDEO]
                          [--no-display] [--skip-frames N]
                          [--export-csv FILE] [--export-json FILE]
                          [--output-video FILE]

Required:
  -p, --shape-predictor  Path to shape predictor model (.dat)

Optional:
  -v, --video            Input video (default: proefpersoon 2_M.avi)
  --no-display           Disable live preview (faster)
  --skip-frames N        Process every Nth frame (default: 1)
  --export-csv FILE      Save mouth coordinates to CSV
  --export-json FILE     Save coordinates + metadata to JSON
  --output-video FILE    Save annotated video
```

### Performance tips

```bash
# 2× faster: process every other frame
python src/facial_landmarks_video.py -p model.dat -v video.avi \
    --no-display --skip-frames 2

# 5× faster: sample every 5th frame
python src/facial_landmarks_video.py -p model.dat -v video.avi \
    --no-display --skip-frames 5
```

### Batch processing

```bash
# Process multiple videos automatically
python examples/batch_process.py
```

Edit `examples/batch_process.py` to set `INPUT_DIR`, `OUTPUT_DIR`, and
`MODEL_PATH` for your dataset.

### JSON output format

```json
{
  "video_file": "input.avi",
  "total_frames": 1000,
  "frames_processed": 1000,
  "detections": 950,
  "skip_frames": 1,
  "coordinates": [
    {"frame": 1, "mouth_x": 245.3, "mouth_y": 312.7}
  ]
}
```

## Project Structure

```
3D-tongue-tip-tracking/
├── src/                            # All Python source scripts
│   ├── calib-camera.py             # Camera calibration (single + stereo)
│   ├── facial_landmarks_video.py   # 2D landmark tracking from video
│   ├── facial_landmarks_webcam.py  # Real-time webcam tracking
│   ├── tongue_tracking_gui.py      # Tkinter GUI wrapper
│   ├── tracking_tongue.py          # 2D optical-flow tracker (replaces .m)
│   └── tracking_in_3d.py           # 3D reconstruction (replaces .m)
├── tests/                          # 129 automated tests
│   ├── conftest.py                 # Shared fixtures
│   ├── test_calib.py
│   ├── test_tracking_tongue.py
│   ├── test_tracking_3d.py
│   ├── test_facial_landmarks_video.py
│   ├── test_integration.py
│   ├── test_data_processing.py
│   └── test_validation.py
├── examples/
│   └── batch_process.py            # Batch processing example
├── legacy/                         # Archived MATLAB scripts
│   ├── tracking_tongue.m
│   ├── tracking_in_3d.m
│   └── README.md
├── pyproject.toml                  # Build config + tool settings
├── requirements.txt                # Runtime dependencies
├── requirements-dev.txt            # Dev dependencies
├── pytest.ini                      # Test runner config
├── .pre-commit-config.yaml         # Pre-commit hooks
├── CHANGELOG.md
├── CONTRIBUTING.md
└── SECURITY.md
```

## Testing

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Fast run (skip integration tests)
pytest -m "not integration"

# With coverage
pytest --cov=src --cov-report=term-missing
```

## Troubleshooting

### "Shape predictor file not found"
```bash
curl -L https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks_GTX.dat.bz2 \
  -o shape_predictor_68_face_landmarks_GTX.dat.bz2 && \
bzip2 -d shape_predictor_68_face_landmarks_GTX.dat.bz2
```
Check the filename matches exactly: `shape_predictor_68_face_landmarks_GTX.dat`.

### dlib installation fails

dlib requires a C++ compiler.

```bash
# Ubuntu / Debian
sudo apt-get install build-essential cmake
pip install dlib

# macOS
brew install cmake
pip install dlib
```

On Windows, use pre-built wheels or Anaconda.

### "No mouth coordinates detected"
- Ensure the video contains clearly visible faces
- Check video quality and lighting conditions
- Try adjusting the camera angle or distance

### Performance issues
- Use `--no-display` to skip video rendering
- Use `--skip-frames N` to sample frames
- Process shorter video segments for testing

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{tongue-tracking-3d,
  author = {Kumar, Navaneeth},
  title  = {Optical Flow based Tongue Tip Tracking in 3D},
  year   = {2019},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/dronefreak/3D-tongue-tip-tracking}}
}
```

## Contact

- **Issues / questions:** [GitHub Issues](https://github.com/dronefreak/3D-tongue-tip-tracking/issues)
- **Email:** kumaar324@gmail.com

## License

MIT — see [LICENSE](LICENSE) for details.

---

> **Research code notice:** Test thoroughly before using in production medical applications.

