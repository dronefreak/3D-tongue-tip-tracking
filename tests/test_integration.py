"""
Integration / smoke tests for the full Python pipeline.

These tests exercise the real code paths end-to-end using synthetic data —
no real video footage or camera hardware is required.

Pipeline under test
-------------------
  1. tracking_tongue.py  — 2-D optical-flow tracking on a synthetic video
  2. tracking_in_3d.py   — DLT triangulation + bundle adjustment from
                           synthetic tracked-point CSVs and a known camera rig

Marked with @pytest.mark.integration so they can be skipped in fast runs:
  pytest -m "not integration"
"""

import csv
import json
import os
import sys
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from tracking_in_3d import main as reconstruct_main
from tracking_tongue import main as tongue_main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_with_args(main_fn, argv: list[str]) -> None:
    """Invoke a CLI main() function with a patched sys.argv."""
    with patch.object(sys, "argv", ["script"] + argv):
        main_fn()


def _write_tracked_csv(path: str, n: int, x_base: float = 100.0, y_base: float = 100.0) -> None:
    """Write a simple n-row tracked-points CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x", "y"])
        for i in range(n):
            w.writerow([i + 1, x_base + i * 0.5, y_base + i * 0.3])


# ---------------------------------------------------------------------------
# tracking_tongue.py integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTrackingTongueIntegration:
    def test_full_video_tracking_produces_csv(self, synthetic_video_path, tmp_path):
        """
        Run tracking_tongue.py on a synthetic video and verify the CSV output
        has the correct number of rows and the expected columns.
        """
        out_csv = str(tmp_path / "tracked.csv")
        _run_with_args(
            tongue_main,
            [
                "--video",
                synthetic_video_path,
                "--no-display",
                "--output-csv",
                out_csv,
            ],
        )

        assert os.path.exists(out_csv), "CSV output file was not created"

        with open(out_csv, newline="") as f:
            rows = list(csv.DictReader(f))

        # Synthetic video has 20 frames; first frame is used to bootstrap
        # so 19 frames are processed in the loop
        assert len(rows) >= 15, f"Expected ≥15 rows, got {len(rows)}"
        assert set(rows[0].keys()) == {"frame", "x", "y"}

    def test_full_video_tracking_produces_json(self, synthetic_video_path, tmp_path):
        """Run tracking_tongue.py and verify the JSON output structure."""
        out_json = str(tmp_path / "tracked.json")
        _run_with_args(
            tongue_main,
            [
                "--video",
                synthetic_video_path,
                "--no-display",
                "--output-json",
                out_json,
            ],
        )

        assert os.path.exists(out_json), "JSON output file was not created"

        with open(out_json) as f:
            data = json.load(f)

        assert "video" in data
        assert "points" in data
        assert "total_frames" in data
        assert isinstance(data["points"], list)
        assert len(data["points"]) >= 15

    def test_tracking_with_preset_roi_does_not_crash(self, synthetic_video_path, tmp_path):
        """
        Using --view mid with a small 200×200 synthetic video — the preset ROI
        starts at (367, 350) which is outside the frame.  The script must exit
        cleanly (sys.exit) rather than raising an unhandled exception.
        """
        with pytest.raises(SystemExit) as exc:
            _run_with_args(
                tongue_main,
                [
                    "--video",
                    synthetic_video_path,
                    "--view",
                    "mid",
                    "--no-display",
                ],
            )
        # Must exit with a non-zero code (informative error, not a traceback)
        assert exc.value.code != 0

    def test_tracking_with_custom_roi(self, synthetic_video_path, tmp_path):
        """--roi must restrict tracking to approximately the given region."""
        out_csv = str(tmp_path / "roi.csv")
        _run_with_args(
            tongue_main,
            [
                "--video",
                synthetic_video_path,
                "--roi",
                "0",
                "0",
                "100",
                "100",
                "--no-display",
                "--output-csv",
                out_csv,
            ],
        )
        assert os.path.exists(out_csv)
        with open(out_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        # KLT can extrapolate slightly outside the crop; allow ±5 px slack
        for row in rows:
            assert float(row["x"]) >= -5, f"x={row['x']} too far outside ROI"
            assert float(row["y"]) >= -5, f"y={row['y']} too far outside ROI"

    def test_missing_video_exits_nonzero(self, tmp_path):
        """Non-existent video must cause a sys.exit(1)."""
        with pytest.raises(SystemExit) as exc:
            _run_with_args(
                tongue_main,
                [
                    "--video",
                    str(tmp_path / "does_not_exist.avi"),
                    "--no-display",
                ],
            )
        assert exc.value.code != 0


# ---------------------------------------------------------------------------
# tracking_in_3d.py integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTrackingIn3dIntegration:
    def test_reconstruction_produces_csv(self, tmp_path, camera_poses_json, tracked_points_csvs):
        """Full 3-D reconstruction pipeline must produce a well-formed CSV."""
        out_csv = str(tmp_path / "xyz.csv")
        left_csv, mid_csv, right_csv = tracked_points_csvs

        _run_with_args(
            reconstruct_main,
            [
                "--cameras",
                camera_poses_json,
                "--left-csv",
                left_csv,
                "--mid-csv",
                mid_csv,
                "--right-csv",
                right_csv,
                "--output-csv",
                out_csv,
                "--no-display",
            ],
        )

        assert os.path.exists(out_csv), "3-D CSV output was not created"

        with open(out_csv, newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 30  # matches fixture n=30
        assert set(rows[0].keys()) == {"frame", "x_mm", "y_mm", "z_mm"}

    def test_reconstruction_values_are_numeric(
        self, tmp_path, camera_poses_json, tracked_points_csvs
    ):
        """Every cell in the CSV must be parseable as a float."""
        out_csv = str(tmp_path / "xyz.csv")
        left_csv, mid_csv, right_csv = tracked_points_csvs

        _run_with_args(
            reconstruct_main,
            [
                "--cameras",
                camera_poses_json,
                "--left-csv",
                left_csv,
                "--mid-csv",
                mid_csv,
                "--right-csv",
                right_csv,
                "--output-csv",
                out_csv,
                "--no-display",
            ],
        )

        with open(out_csv, newline="") as f:
            for row in csv.DictReader(f):
                for key in ("x_mm", "y_mm", "z_mm"):
                    float(row[key])  # raises ValueError if not numeric

    def test_reconstruction_without_bundle_adjustment(
        self, tmp_path, camera_poses_json, tracked_points_csvs
    ):
        """--no-ba flag must still produce a complete output CSV."""
        out_csv = str(tmp_path / "xyz_no_ba.csv")
        left_csv, mid_csv, right_csv = tracked_points_csvs

        _run_with_args(
            reconstruct_main,
            [
                "--cameras",
                camera_poses_json,
                "--left-csv",
                left_csv,
                "--mid-csv",
                mid_csv,
                "--right-csv",
                right_csv,
                "--output-csv",
                out_csv,
                "--no-display",
                "--no-ba",
            ],
        )

        with open(out_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 30

    def test_mismatched_csv_lengths_exit_nonzero(self, tmp_path, camera_poses_json):
        """Mismatched point-count between views must cause sys.exit(1)."""

        def _write(name, n):
            p = str(tmp_path / name)
            _write_tracked_csv(p, n)
            return p

        left_csv = _write("l.csv", 10)
        mid_csv = _write("m.csv", 10)
        right_csv = _write("r.csv", 5)  # mismatch!

        with pytest.raises(SystemExit) as exc:
            _run_with_args(
                reconstruct_main,
                [
                    "--cameras",
                    camera_poses_json,
                    "--left-csv",
                    left_csv,
                    "--mid-csv",
                    mid_csv,
                    "--right-csv",
                    right_csv,
                    "--no-display",
                ],
            )
        assert exc.value.code != 0

    def test_missing_camera_json_exits_nonzero(self, tmp_path, tracked_points_csvs):
        """A non-existent camera-poses JSON must cause sys.exit(1)."""
        left_csv, mid_csv, right_csv = tracked_points_csvs

        with pytest.raises(SystemExit) as exc:
            _run_with_args(
                reconstruct_main,
                [
                    "--cameras",
                    str(tmp_path / "no_such_file.json"),
                    "--left-csv",
                    left_csv,
                    "--mid-csv",
                    mid_csv,
                    "--right-csv",
                    right_csv,
                    "--no-display",
                ],
            )
        assert exc.value.code != 0

    def test_saved_plot_file_created(self, tmp_path, camera_poses_json, tracked_points_csvs):
        """--save-plot must write an image file."""
        plot_path = str(tmp_path / "reconstruction.png")
        left_csv, mid_csv, right_csv = tracked_points_csvs

        _run_with_args(
            reconstruct_main,
            [
                "--cameras",
                camera_poses_json,
                "--left-csv",
                left_csv,
                "--mid-csv",
                mid_csv,
                "--right-csv",
                right_csv,
                "--no-display",
                "--save-plot",
                plot_path,
            ],
        )

        assert os.path.exists(plot_path), "Plot file was not saved"
        assert os.path.getsize(plot_path) > 1000, "Plot file is suspiciously small"
