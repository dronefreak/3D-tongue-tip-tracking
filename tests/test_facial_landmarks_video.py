"""
Tests for facial_landmarks_video.py logic.

Because this script has no if-__name__-__main__ guard and requires dlib/imutils
(not installed in CI), we test its logic in isolation rather than importing
the module directly.  This covers:

  - FRAME_WIDTH constant value
  - Frame-skip counter logic
  - Array normalisation (zero-sum guard)
  - Preallocated-array overflow guard
  - CSV export format
  - JSON export format
  - VideoWriter dimension calculation (zero-width guard)
  - Script --help exits 0 (proves argparse is correct)
"""

import ast
import csv
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load relevant constants / logic from the script via AST + exec
# We isolate only the top-level constant assignments, not the executable code.
# ---------------------------------------------------------------------------

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "facial_landmarks_video.py")


def _read_source():
    with open(SCRIPT_PATH) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Syntax
# ---------------------------------------------------------------------------

def test_script_parses_without_syntax_error():
    """The entire script must be valid Python 3."""
    src = _read_source()
    ast.parse(src)   # raises SyntaxError if broken


# ---------------------------------------------------------------------------
# FRAME_WIDTH constant
# ---------------------------------------------------------------------------

def test_frame_width_constant_is_500():
    """FRAME_WIDTH must be 500 so VideoWriter and imutils.resize stay in sync."""
    src = _read_source()
    # Extract the constant via AST
    tree = ast.parse(src)
    frame_width = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "FRAME_WIDTH":
                    if isinstance(node.value, ast.Constant):
                        frame_width = node.value.value
    assert frame_width == 500, f"FRAME_WIDTH expected 500, got {frame_width}"


def test_frame_width_used_in_resize():
    """imutils.resize must reference FRAME_WIDTH, not a hardcoded literal 500."""
    src = _read_source()
    assert "imutils.resize(image, width=FRAME_WIDTH)" in src


def test_frame_width_used_in_video_writer():
    """VideoWriter size tuple must use FRAME_WIDTH, not a hardcoded literal."""
    src = _read_source()
    assert "FRAME_WIDTH" in src
    # The old hardcoded (500, int(cap.get...)) pattern must be gone
    assert "(500, int(cap.get" not in src


# ---------------------------------------------------------------------------
# Frame-skip counter logic
# ---------------------------------------------------------------------------

class TestSkipFramesLogic:
    """Mirror the skip-counter logic from facial_landmarks_video.py."""

    @staticmethod
    def _simulate_skip(total_frames: int, skip_frames: int) -> int:
        """Return the number of frames that pass the skip filter."""
        skip_counter = 0
        processed = 0
        for _ in range(total_frames):
            skip_counter += 1
            if skip_counter < skip_frames:
                continue
            skip_counter = 0
            processed += 1
        return processed

    def test_skip1_processes_all_frames(self):
        assert self._simulate_skip(100, 1) == 100

    def test_skip2_halves_frames(self):
        assert self._simulate_skip(100, 2) == 50

    def test_skip3_thirds_frames(self):
        assert self._simulate_skip(99, 3) == 33

    def test_skip_never_exceeds_total(self):
        for total in [1, 10, 100]:
            for skip in [1, 2, 3, 5]:
                result = self._simulate_skip(total, skip)
                assert result <= total

    def test_single_frame_skip1(self):
        assert self._simulate_skip(1, 1) == 1

    def test_zero_frames(self):
        assert self._simulate_skip(0, 1) == 0


# ---------------------------------------------------------------------------
# Array normalisation (zero-sum guard)
# ---------------------------------------------------------------------------

class TestNormalisationLogic:
    """Mirror the x-normalisation guard in facial_landmarks_video.py."""

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        x_sum = np.sum(arr)
        return arr if x_sum == 0 else arr / x_sum

    def test_normal_array_normalises_to_one(self):
        arr = np.array([100.0, 200.0, 300.0])
        result = self._normalize(arr)
        assert np.sum(result) == pytest.approx(1.0)

    def test_zero_array_returns_unchanged(self):
        arr = np.zeros(5, dtype=np.float32)
        result = self._normalize(arr)
        np.testing.assert_array_equal(result, arr)

    def test_single_value_normalises(self):
        arr = np.array([50.0])
        result = self._normalize(arr)
        assert result[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Preallocated-array overflow guard
# ---------------------------------------------------------------------------

def test_preallocation_uses_total_frames_not_frames_to_process():
    """
    Arrays must be sized by total_frames, not frames_to_process (floor division),
    to avoid silent data loss when multiple faces are detected per frame.
    """
    src = _read_source()
    # Confirm total_frames is used for preallocation
    assert "np.zeros(total_frames, dtype=np.float32)" in src
    # Confirm the old frames_to_process-sized preallocation is gone
    assert "np.zeros(frames_to_process, dtype=np.float32)" not in src


# ---------------------------------------------------------------------------
# VideoWriter zero-width guard
# ---------------------------------------------------------------------------

def test_video_writer_guards_against_zero_width():
    """The orig_w > 0 guard must be present to avoid ZeroDivisionError."""
    src = _read_source()
    assert "orig_w > 0" in src


# ---------------------------------------------------------------------------
# CSV export logic
# ---------------------------------------------------------------------------

class TestCsvExport:
    """Test the CSV export logic independently from the script."""

    @staticmethod
    def _export_csv(path: str, frame_arr: np.ndarray,
                    x_arr: np.ndarray, y_arr: np.ndarray) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "mouth_x", "mouth_y"])
            for i in range(len(x_arr)):
                writer.writerow([frame_arr[i], x_arr[i], y_arr[i]])

    def test_csv_header(self, tmp_path):
        p = str(tmp_path / "out.csv")
        self._export_csv(p,
                         np.array([1, 2, 3]),
                         np.array([10.0, 20.0, 30.0]),
                         np.array([5.0, 15.0, 25.0]))
        with open(p) as f:
            header = f.readline().strip()
        assert header == "frame,mouth_x,mouth_y"

    def test_csv_row_count(self, tmp_path):
        p = str(tmp_path / "out.csv")
        n = 7
        self._export_csv(p, np.arange(1, n + 1),
                         np.ones(n), np.ones(n))
        with open(p) as f:
            rows = list(csv.reader(f))
        assert len(rows) == n + 1   # header + n data rows

    def test_csv_values_round_trip(self, tmp_path):
        p = str(tmp_path / "out.csv")
        frames = np.array([1, 2])
        xs = np.array([123.45, 678.90])
        ys = np.array([11.11, 22.22])
        self._export_csv(p, frames, xs, ys)

        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert float(rows[0]["mouth_x"]) == pytest.approx(123.45)
        assert float(rows[1]["mouth_y"]) == pytest.approx(22.22)


# ---------------------------------------------------------------------------
# JSON export logic
# ---------------------------------------------------------------------------

class TestJsonExport:

    @staticmethod
    def _export_json(path: str, video: str, n: int,
                     frames: np.ndarray, xs: np.ndarray,
                     ys: np.ndarray, skip: int) -> None:
        data = {
            "video_file": video,
            "total_frames": n,
            "frames_processed": n,
            "detections": n,
            "skip_frames": skip,
            "coordinates": [
                {"frame": int(frames[i]), "mouth_x": float(xs[i]), "mouth_y": float(ys[i])}
                for i in range(n)
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def test_json_has_required_keys(self, tmp_path):
        p = str(tmp_path / "out.json")
        self._export_json(p, "v.avi", 3,
                          np.array([1, 2, 3]),
                          np.array([1.0, 2.0, 3.0]),
                          np.array([4.0, 5.0, 6.0]), 1)
        with open(p) as f:
            data = json.load(f)
        for key in ("video_file", "total_frames", "detections", "coordinates"):
            assert key in data, f"Missing key: {key}"

    def test_json_coordinates_length(self, tmp_path):
        p = str(tmp_path / "out.json")
        n = 5
        self._export_json(p, "v.avi", n,
                          np.arange(1, n + 1),
                          np.ones(n), np.ones(n), 1)
        with open(p) as f:
            data = json.load(f)
        assert len(data["coordinates"]) == n

    def test_json_coordinate_entry_structure(self, tmp_path):
        p = str(tmp_path / "out.json")
        self._export_json(p, "v.avi", 1,
                          np.array([42]),
                          np.array([7.5]),
                          np.array([8.5]), 1)
        with open(p) as f:
            data = json.load(f)
        entry = data["coordinates"][0]
        assert entry["frame"] == 42
        assert entry["mouth_x"] == pytest.approx(7.5)
        assert entry["mouth_y"] == pytest.approx(8.5)


# ---------------------------------------------------------------------------
# CLI --help (proves argparse is intact, no SyntaxError at load time)
# ---------------------------------------------------------------------------

def test_help_flag_exits_zero():
    """
    Running the script with --help must exit 0.
    This also verifies the script has no top-level syntax errors that
    would prevent argparse from even loading.
    Skipped when optional dependencies (dlib, imutils) are not installed.
    """
    import importlib
    if importlib.util.find_spec("imutils") is None or importlib.util.find_spec("dlib") is None:
        pytest.skip("imutils/dlib not installed in this environment")

    result = subprocess.run(
        [sys.executable, SCRIPT_PATH, "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"--help exited {result.returncode}\n{result.stderr}"
    )
    assert "--shape-predictor" in result.stdout
