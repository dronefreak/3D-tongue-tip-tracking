# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] — 2026-06-30

### Added
- **`src/tracking_tongue.py`** — pure-Python replacement for `tracking_tongue.m`.
  Uses `cv2.calcOpticalFlowFarneback` (matches MATLAB `opticalFlowFarneback` defaults)
  and pyramidal Lucas-Kanade KLT tracking with bidirectional error validation
  (`MaxBidirectionalError = 1.0 px`, matching MATLAB `vision.PointTracker`).
  CLI: `--video`, `--view {mid,left,right}`, `--roi X Y W H`,
  `--flow-threshold`, `--no-display`, `--output-csv`, `--output-json`.
- **`src/tracking_in_3d.py`** — pure-Python replacement for `tracking_in_3d.m`.
  Implements Direct Linear Transform (DLT) multi-view triangulation and
  `scipy.optimize.least_squares` bundle adjustment with Huber loss (equivalent
  to MATLAB `triangulateMultiview` + `bundleAdjustment`).
  CLI: `--cameras JSON`, `--left-csv`, `--mid-csv`, `--right-csv`,
  `--no-ba`, `--output-csv`, `--save-plot`, `--no-display`.
- **Stereo calibration mode** in `src/calib-camera.py` (`--stereo-folder`,
  `--output-prefix`). Produces `camera_poses.json` consumed by `tracking_in_3d.py`.
- **`legacy/`** directory with archived `tracking_tongue.m` and
  `tracking_in_3d.m` plus a `README.md` explaining the migration.
- **Test suite** — 129 tests across 7 files (128 pass, 1 skipped on
  environments without dlib/imutils):
  - `tests/conftest.py` — shared fixtures (synthetic video, camera poses, tracked CSVs)
  - `tests/test_calib.py` — 20 unit tests for `calib-camera.py`
  - `tests/test_tracking_tongue.py` — 29 unit tests for `tracking_tongue.py`
  - `tests/test_tracking_3d.py` — 30 unit tests for `tracking_in_3d.py`
  - `tests/test_facial_landmarks_video.py` — 21 unit tests (AST-based, no dlib/imutils required)
  - `tests/test_integration.py` — 11 end-to-end integration tests
- **`pyproject.toml`** — replaces `setup.py`; defines build system, dependencies,
  entry-point scripts, ruff and isort config, and pytest settings.
- **`.pre-commit-config.yaml`** — pre-commit hooks: `pre-commit-hooks` (file
  hygiene), `ruff` (lint + fix), `ruff-format` (formatting).
- **`requirements-dev.txt`** — development dependencies (pytest, ruff, pre-commit).
- **`src/`** directory — all Python scripts moved from the project root to `src/`.

### Changed
- **Model reference** updated from a broken Google Drive link to the official
  dlib GTX model (`shape_predictor_68_face_landmarks_GTX.dat`), which is
  smaller (40 MB vs 99 MB), faster, and more accurate than the original.
  Updated in: `README.md`, `tongue_tracking_gui.py`, `examples/batch_process.py`, and all documentation.
- **`src/calib-camera.py`** — all module-level code wrapped into functions;
  `MAX_CORNER_ITERATIONS = 30` constant replaces the bug where the board
  `dimension` (20 mm float) was passed as `maxCount` to `cv2.cornerSubPix`.
- **`src/calib-camera.py`** — implemented `initialize_arg_parser()` and
  `validate_inputs()` which were called but never defined (would raise
  `NameError` at runtime).
- **`src/tongue_tracking_gui.py`** — subprocess paths now resolved relative to
  `__file__` so the GUI works regardless of the current working directory.
- **`pytest.ini`** — added `pythonpath = src` so test imports resolve correctly
  after the `src/` restructure.

### Fixed
- **`src/facial_landmarks_video.py`**: Orphaned `finally:` block (was a
  `SyntaxError` that prevented the script from running at all).
- **`src/facial_landmarks_video.py`**: Added `FRAME_WIDTH = 500` constant
  (was used implicitly via `imutils.resize` but never defined, causing a
  mismatch between display size and `VideoWriter` dimensions).
- **`src/facial_landmarks_video.py`**: `VideoWriter` zero-division guard —
  skips writer initialisation when `resize_ratio` is 0.
- **`src/facial_landmarks_video.py`**: Preallocated arrays now sized by
  `frames_to_process` (not `total_frames`) so index never exceeds array
  bounds when `--skip-frames` is used.
- **`src/calib-camera.py`**: Termination criteria `maxCount` was incorrectly
  set to `dimension` (the board cell size in mm, typically 20) instead of the
  `MAX_CORNER_ITERATIONS` constant.
- **`src/tongue_tracking_gui.py`**: `start_webcam()` blocked the Tkinter event
  loop via a bare `subprocess.run()` call — moved to a daemon thread via
  `_run_webcam_cmd()`.
- **`src/tracking_tongue.py`**: `cv2.destroyAllWindows()` wrapped in
  `try/except cv2.error` so the script works on headless OpenCV builds.
- **`src/tracking_tongue.py`**: Empty-frame guard added at bootstrap and in the
  main loop when the ROI falls entirely outside the video frame.

### Removed
- **`setup.py`** — superseded by `pyproject.toml`.
- **`facial_landmarks_video.py.bak`** — stale backup file removed.
- **Docker files removed** — `Dockerfile`, `docker-compose.yml`, `.dockerignore`
  removed; plain `pip install` is sufficient for this research project.

### Community files added
- `CONTRIBUTING.md` — development setup, coding standards, PR process
- `SECURITY.md` — vulnerability reporting policy
- `.github/ISSUE_TEMPLATE/bug_report.md` — structured bug reports
- `.github/ISSUE_TEMPLATE/feature_request.md` — feature proposals
- `.github/PULL_REQUEST_TEMPLATE.md` — PR checklist

---

## [1.0.0] — 2019 (original release)

### Added
- Initial Python implementation: `facial_landmarks_video.py`,
  `facial_landmarks_webcam.py`, `tongue_tracking_gui.py`, `calib-camera.py`.
- MATLAB pipeline: `tracking_tongue.m`, `tracking_in_3d.m`.
- `requirements.txt`, initial `Dockerfile` (since removed — no longer needed).
- Research report (`ipcv_report.pdf`) detailing the algorithm and motivation.

[1.1.0]: https://github.com/dronefreak/3D-tongue-tip-tracking/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/dronefreak/3D-tongue-tip-tracking/releases/tag/v1.0.0
