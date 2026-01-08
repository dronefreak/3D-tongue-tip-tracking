# Optical Flow based Tongue Tip Tracking in 3D

We present a novel method for tracking the tip of tongues in 3-dimensions for medical applications. The repository also includes a report that explains the algorithm in detail and the motivation behind the project.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Python Implementation](#python-implementation)
- [MATLAB Implementation](#matlab-implementation)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Contact](#contact)
- [License](#license)

## Features

✨ **New in latest version:**
- 🚀 **3-4x faster processing** with optimized algorithms
- 💾 **Data export** to CSV and JSON formats
- 🎥 **Annotated video output** with tracking visualization
- ⚡ **Batch processing mode** for headless environments
- 📊 **Progress indicators** during processing
- 🛡️ **Comprehensive error handling** and validation

## Requirements

### Python Implementation
- Python 3.6 or higher
- Dependencies listed in `requirements.txt`:
  - opencv-python >= 4.5.0
  - numpy >= 1.19.0
  - matplotlib >= 3.3.0
  - dlib >= 19.21.0
  - imutils >= 0.5.4
  - scipy >= 1.5.0

### MATLAB Implementation
- MATLAB 2018b or higher (may work with other versions)
- Computer Vision Toolbox
- Image Processing Toolbox

## Installation

### Python Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dronefreak/3D-tongue-tip-tracking.git
   cd 3D-tongue-tip-tracking
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the facial landmark model:**

   Download the finetuned shape predictor model from [Google Drive](https://drive.google.com/file/d/1kEOn0SsyToOCGr45UDygxnkDo4uxlWeh/view?usp=sharing) and place it in the project directory.

### MATLAB Setup

1. Ensure MATLAB 2018b or higher is installed
2. Install required toolboxes:
   - Computer Vision Toolbox
   - Image Processing Toolbox

## Quick Start

### Python - Basic Usage

```bash
python facial_landmarks_video.py \
  --shape-predictor shape_predictor_68_face_landmarks_finetuned.dat \
  --video your_video.avi
```

### Python - Fast Processing with Data Export

```bash
python facial_landmarks_video.py \
  --shape-predictor shape_predictor_68_face_landmarks_finetuned.dat \
  --video your_video.avi \
  --no-display \
  --skip-frames 2 \
  --export-csv results.csv \
  --export-json results.json \
  --output-video annotated.avi
```

## Python Implementation

### Camera Calibration

Before tracking, calibrate your camera using the provided script:

```bash
python calib-camera.py <folder> <image_type> <num_rows> <num_cols> <cell_dimension>
```

Example:
```bash
python calib-camera.py ./camera_01 jpg 8 8 20
```

This will generate:
- `cameraMatrix.txt` - Camera intrinsic parameters
- `cameraDistortion.txt` - Lens distortion coefficients
- `calibresult.png` - Sample undistorted image

### Facial Landmark Detection

The system uses Constrained Local Neural Fields (CLNF) to detect faces and predict face orientation. This helps evaluate a 3D box around the face, the coordinates of which are later used for localizing the tongue in 3D.

**Research Reference:** [Constrained Local Neural Fields for Robust Facial Landmark Detection](https://arxiv.org/pdf/1611.08657.pdf)

**Model Download:** [Finetuned Shape Predictor (68 landmarks)](https://drive.google.com/file/d/1kEOn0SsyToOCGr45UDygxnkDo4uxlWeh/view?usp=sharing)

### Command-Line Options

```bash
facial_landmarks_video.py [-h] -p SHAPE_PREDICTOR [-v VIDEO]
                          [--no-display] [--skip-frames N]
                          [--export-csv FILE] [--export-json FILE]
                          [--output-video FILE]

Required arguments:
  -p, --shape-predictor  Path to facial landmark predictor model

Optional arguments:
  -v, --video           Path to input video file (default: proefpersoon 2_M.avi)
  --no-display          Disable video display for faster batch processing
  --skip-frames N       Process every Nth frame (default: 1, process all)
  --export-csv FILE     Export mouth coordinates to CSV file
  --export-json FILE    Export mouth coordinates to JSON file
  --output-video FILE   Save annotated video with tracking overlays
```

### Output Files

The script generates several output files:

1. **plot_x.png** - Graph of relative X-coordinate motion
2. **plot_y.png** - Graph of relative Y-coordinate motion
3. **results.csv** (if --export-csv specified) - Frame-by-frame coordinates
4. **results.json** (if --export-json specified) - Complete metadata and coordinates
5. **annotated.avi** (if --output-video specified) - Video with tracking visualization

![Shape Detector](image.png)

A 3D visualization is shown below. This is what we use later.

![3D](3d.png)

## MATLAB Implementation

### Camera Calibration

Before beginning, since this is a 2D-3D transformation project, you need to calibrate your camera. Use the `cameraCalibrator` app provided in MATLAB's Computer Vision Toolbox.

### Tracking Pipeline

There are two MATLAB scripts in this repository that must be executed in order:

1. **tracking_tongue.m** - Tracks tongue tip in video using optical flow
2. **tracking_in_3d.m** - Reconstructs 3D coordinates from multiple views

#### Step 1: Configure Video File

Edit `tracking_tongue.m` and set the video file path:
```matlab
videoFile = 'path/to/your/video.avi';
```

You can also adjust the Region of Interest (ROI) for different camera views:
```matlab
r = [367.5 350.5 361 365];  % midtest (default)
% r = [135.5 479.5 367 261];  % ltest (uncomment for left view)
% r = [595.5 431.5 412 302];  % rtest (uncomment for right view)
```

#### Step 2: Run Tracking

```matlab
>> tracking_tongue.m
```

This tracks the tongue tip using Gunnar Farnebäck's Optical Flow Method and stores results in the `mpoints` variable.

#### Step 3: 3D Reconstruction

After running tracking for all three views (left, middle, right), run:

```matlab
>> tracking_in_3d.m
```

This requires:
- `cameraParams` - Camera calibration parameters
- `camPoses` - Camera pose information for all three views
- `mpoints` - Middle view tracking results
- `lpoints` - Left view tracking results
- `rpoints` - Right view tracking results

The script performs:
1. Multi-view triangulation
2. Bundle adjustment for refinement
3. 3D point cloud visualization

![3D-R](10.png)

The 3D visualization should look something like this. These are the 3D coordinates of the tongue tip:

![3D-R](9.png)

## Advanced Usage

### Performance Optimization

For large video files or batch processing:

```bash
# Process every 2nd frame with no display (2-3x faster)
python facial_landmarks_video.py -p model.dat -v video.avi --no-display --skip-frames 2

# Process every 5th frame (5x faster, good for quick analysis)
python facial_landmarks_video.py -p model.dat -v video.avi --no-display --skip-frames 5
```

### Integration with Analysis Pipeline

Export data in your preferred format for further analysis:

```bash
# Export to CSV for Excel/Pandas
python facial_landmarks_video.py -p model.dat -v video.avi --export-csv data.csv

# Export to JSON for Python/JavaScript/R
python facial_landmarks_video.py -p model.dat -v video.avi --export-json data.json
```

JSON output includes metadata:
```json
{
  "video_file": "input.avi",
  "total_frames": 1000,
  "frames_processed": 1000,
  "detections": 950,
  "skip_frames": 1,
  "coordinates": [
    {"frame": 1, "mouth_x": 245.3, "mouth_y": 312.7},
    ...
  ]
}
```

## Troubleshooting

### Common Issues

#### "Shape predictor file not found"
- Download the model from the [Google Drive link](https://drive.google.com/file/d/1kEOn0SsyToOCGr45UDygxnkDo4uxlWeh/view?usp=sharing)
- Ensure the file is in the same directory as the script
- Check the filename matches exactly: `shape_predictor_68_face_landmarks_finetuned.dat`

#### "No mouth coordinates detected"
- Ensure the video contains clearly visible faces
- Check video quality and lighting conditions
- Try adjusting the camera angle or distance
- Verify the shape predictor model is the correct version

#### "dlib installation fails"
dlib requires a C++ compiler. On Ubuntu/Debian:
```bash
sudo apt-get install build-essential cmake
pip install dlib
```

On macOS:
```bash
brew install cmake
pip install dlib
```

On Windows, consider using pre-built wheels or Anaconda.

#### MATLAB "Video file not found"
- Edit `tracking_tongue.m` and update the `videoFile` variable
- Use absolute paths if relative paths don't work
- Ensure the video format is supported (.avi, .mp4, .mov)

#### MATLAB "Variable not found" in tracking_in_3d.m
- Ensure you've run `tracking_tongue.m` for all three views first
- Variables must be in workspace: `cameraParams`, `camPoses`, `mpoints`, `lpoints`, `rpoints`
- Check that all point arrays have the same length

### Performance Issues

If processing is slow:
- Use `--no-display` to disable video rendering
- Use `--skip-frames N` to sample frames
- Process shorter video segments for testing
- Ensure your system has adequate RAM (8GB+ recommended)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{tongue-tracking-3d,
  author = {Kumar, Navaneeth},
  title = {Optical Flow based Tongue Tip Tracking in 3D},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/dronefreak/3D-tongue-tip-tracking}}
}
```

## Contact

For questions, issues, or collaboration opportunities:

- **Create an issue**: [GitHub Issues](https://github.com/dronefreak/3D-tongue-tip-tracking/issues)
- **Email**: kumaar324@gmail.com or navaneeth.94@gmail.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This is research code. While we've added error handling and optimizations, please test thoroughly before using in production medical applications.
