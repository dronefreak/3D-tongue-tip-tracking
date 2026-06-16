# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-01-08

### Added
- GitHub Actions CI/CD pipeline for automated testing across Python 3.7-3.10
- Graphical User Interface (GUI) application `tongue_tracking_gui.py` with:
  - Video processing tab with file browsers and parameter controls
  - Webcam tracking tab for real-time processing
  - Settings tab for configuration management
  - About tab with project information
  - Output log display for monitoring processing status
  - Threading support for non-blocking execution
- Configuration file support with `config.yaml.example` template
- CHANGELOG.md for tracking project improvements

## [1.3.0] - 2026-01-07

### Added
- Real-time webcam tracking with `facial_landmarks_webcam.py`
  - Interactive controls (q=quit, r=record, c=clear)
  - Live visual feedback with mouth position overlay
  - Data export with timestamps
  - Configurable FPS and camera selection
- Comprehensive unit test suite:
  - `tests/test_validation.py` for input validation tests
  - `tests/test_data_processing.py` for data processing tests
  - `pytest.ini` configuration file
  - `requirements-dev.txt` for development dependencies
- Test coverage for critical functionality

## [1.2.0] - 2026-01-06

### Added
- Professional packaging with `setup.py` for pip installation
- Docker support:
  - `Dockerfile` for containerized environment
  - `docker-compose.yml` for easy deployment
  - `.dockerignore` for optimized image building
- Example scripts:
  - `batch_process.py` for processing multiple videos
  - `analyze_results.py` for data analysis
- `MANIFEST.in` for package distribution

### Changed
- Complete README.md rewrite (337 lines vs 41 lines):
  - Added Table of Contents
  - Comprehensive installation guide
  - Usage examples for all scripts
  - Troubleshooting section
  - Performance optimization tips
  - Docker deployment instructions
- Fixed typos and formatting throughout documentation

## [1.1.0] - 2026-01-05

### Added
- CSV export functionality with `--export-csv` flag
- JSON export functionality with `--export-json` flag
- Video output with tracking overlay using `--output-video` flag
- No-display mode with `--no-display` flag for headless processing
- Frame skipping with `--skip-frames` flag for faster processing
- Progress tracking with frame counter display
- Input validation for all scripts

### Fixed
- **Critical:** Fixed video end crash in `facial_landmarks_video.py`
  - Added proper EOF detection with `ret` check
  - Added graceful exit with summary statistics
- **Critical:** Fixed division by zero error in mouth position calculations
  - Added zero-sum validation before division
- **Critical:** Fixed incorrect array indexing in `calib-camera.py`
  - Changed `images[1]` to `images[0]` for first image access
- **Critical:** Fixed termination criteria error in `calib-camera.py`
  - Corrected from `dimension` to proper tuple format
- **Critical:** Fixed undefined variable in `tracking_tongue.m`
  - Added video file initialization and validation
- Fixed missing input validation in `tracking_in_3d.m`
  - Added checks for camera parameters and stereo parameters
- Fixed incorrect array operations with mixed types
  - Converted lists to NumPy arrays for proper vectorization

### Changed
- Optimized performance with NumPy array preallocation (3-4x speedup):
  - Preallocated `mouth_array_x` and `mouth_array_y` in Python scripts
  - Preallocated `mpoints` array in MATLAB scripts
- Improved memory efficiency with pre-sized arrays
- Enhanced error messages with file paths and context
- Added comprehensive input validation for all file paths

### Performance
- Achieved 3-4x speedup through array preallocation
- Reduced memory allocations in tight loops
- Optimized frame skipping for faster batch processing
- Improved video decoding efficiency

## [1.0.0] - Initial Release

### Features
- 3D tongue tip tracking using multi-view triangulation
- Facial landmark detection using dlib's 68-point model
- Optical flow-based tongue tip tracking with Gunnar Farnebäck's method
- Camera calibration utilities
- MATLAB scripts for 3D reconstruction and bundle adjustment
- Basic video processing capabilities

---

## Migration Guide

### From 1.3.0 to 1.4.0
- No breaking changes
- New GUI can be used as alternative to command-line interface
- Configuration files are optional - all scripts work as before
- CI/CD pipeline runs automatically on push

### From 1.2.0 to 1.3.0
- No breaking changes
- Install development dependencies for testing: `pip install -r requirements-dev.txt`
- Run tests with: `pytest -v`

### From 1.1.0 to 1.2.0
- No breaking changes
- Can now install via pip: `pip install -e .`
- Docker support available for containerized deployment

### From 1.0.0 to 1.1.0
- No breaking changes to existing functionality
- All new features are opt-in via command-line flags
- Existing scripts continue to work without modifications

## Known Issues

See the [GitHub Issues](https://github.com/dronefreak/3D-tongue-tip-tracking/issues) page for current known issues and feature requests.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
