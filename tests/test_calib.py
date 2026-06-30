"""
Unit tests for calib-camera.py

Covers:
  - initialize_arg_parser  (argument parsing)
  - validate_inputs        (input validation)
  - _reprojection_error    (reprojection error math)
  - _find_corners          (corner detection, non-interactive)
  - _save_undistorted      (undistortion save path)
  - calibrate_single       (full calibration with mocked cv2)
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock, patch, call

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import calib-camera.py via importlib (hyphen in filename prevents normal import)
# ---------------------------------------------------------------------------

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "calib-camera.py")

spec = importlib.util.spec_from_file_location("calib_camera", _SCRIPT)
calib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calib)


# ---------------------------------------------------------------------------
# initialize_arg_parser
# ---------------------------------------------------------------------------

class TestInitializeArgParser:

    def test_parses_required_positional_args(self):
        parser = calib.initialize_arg_parser()
        args = parser.parse_args(["./cam", "jpg", "8", "8", "20"])
        assert args.folder == "./cam"
        assert args.image_type == "jpg"
        assert args.rows == 8
        assert args.cols == 8
        assert args.dimension == 20.0

    def test_stereo_folder_defaults_to_none(self):
        parser = calib.initialize_arg_parser()
        args = parser.parse_args(["./cam", "jpg", "8", "8", "20"])
        assert args.stereo_folder is None

    def test_stereo_folder_accepted(self):
        parser = calib.initialize_arg_parser()
        args = parser.parse_args(["./cam1", "jpg", "8", "8", "20", "--stereo-folder", "./cam2"])
        assert args.stereo_folder == "./cam2"

    def test_output_prefix_default(self):
        parser = calib.initialize_arg_parser()
        args = parser.parse_args(["./cam", "jpg", "8", "8", "20"])
        assert args.output_prefix == "stereo"

    def test_output_prefix_custom(self):
        parser = calib.initialize_arg_parser()
        args = parser.parse_args(["./cam", "jpg", "8", "8", "20", "--output-prefix", "left_mid"])
        assert args.output_prefix == "left_mid"

    def test_dimension_accepts_float(self):
        parser = calib.initialize_arg_parser()
        args = parser.parse_args(["./cam", "jpg", "8", "8", "25.4"])
        assert args.dimension == pytest.approx(25.4)

    def test_help_exits_zero(self, capsys):
        parser = calib.initialize_arg_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0


# ---------------------------------------------------------------------------
# validate_inputs
# ---------------------------------------------------------------------------

class TestValidateInputs:

    def test_valid_inputs_pass_through(self):
        folder, img_type, rows, cols, dim = calib.validate_inputs(
            "./camera_01", "jpg", 8, 8, 20.0
        )
        assert folder == os.path.normpath("./camera_01")
        assert img_type == "jpg"
        assert rows == 8
        assert cols == 8
        assert dim == 20.0

    def test_leading_dot_stripped_from_extension(self):
        _, img_type, _, _, _ = calib.validate_inputs("./cam", ".png", 8, 8, 20.0)
        assert img_type == "png"

    def test_rows_less_than_2_raises(self):
        with pytest.raises(ValueError, match="rows"):
            calib.validate_inputs("./cam", "jpg", 1, 8, 20.0)

    def test_cols_less_than_2_raises(self):
        with pytest.raises(ValueError, match="cols"):
            calib.validate_inputs("./cam", "jpg", 8, 1, 20.0)

    def test_negative_dimension_raises(self):
        with pytest.raises(ValueError, match="dimension"):
            calib.validate_inputs("./cam", "jpg", 8, 8, -5.0)

    def test_zero_dimension_raises(self):
        with pytest.raises(ValueError, match="dimension"):
            calib.validate_inputs("./cam", "jpg", 8, 8, 0.0)

    def test_empty_image_type_raises(self):
        with pytest.raises(ValueError, match="image_type"):
            calib.validate_inputs("./cam", "", 8, 8, 20.0)

    def test_image_type_only_dot_raises(self):
        with pytest.raises(ValueError, match="image_type"):
            calib.validate_inputs("./cam", ".", 8, 8, 20.0)

    def test_absolute_path_accepted(self):
        """validate_inputs must not reject absolute paths."""
        folder, _, _, _, _ = calib.validate_inputs("/tmp/cam", "jpg", 8, 8, 20.0)
        assert os.path.isabs(folder)


# ---------------------------------------------------------------------------
# _reprojection_error
# ---------------------------------------------------------------------------

class TestReprojectionError:

    def _make_perfect_data(self, camera_matrix, n=5):
        """Return objpoints, imgpoints, rvecs, tvecs with zero reprojection error."""
        K = camera_matrix
        objp = np.zeros((6 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.array([[0.0], [0.0], [500.0]])

        img_pts, _ = cv2.projectPoints(objp, rvec, tvec, K, np.zeros(5))

        objpoints = [objp] * n
        imgpoints = [img_pts] * n
        rvecs = [rvec] * n
        tvecs = [tvec] * n
        return objpoints, imgpoints, rvecs, tvecs

    def test_perfect_projection_gives_zero_error(self, camera_matrix):
        objpoints, imgpoints, rvecs, tvecs = self._make_perfect_data(camera_matrix)
        err = calib._reprojection_error(objpoints, imgpoints, rvecs, tvecs,
                                        camera_matrix, np.zeros(5))
        assert err == pytest.approx(0.0, abs=1e-6)

    def test_error_increases_with_noise(self, camera_matrix):
        objpoints, imgpoints, rvecs, tvecs = self._make_perfect_data(camera_matrix)
        # cv2.norm requires both arrays to have the same dtype (float32)
        # Use large noise (10 px) to reliably exceed the 0.5-px threshold
        noisy = [pts + np.random.randn(*pts.shape).astype(np.float32) * 10.0
                 for pts in imgpoints]
        err = calib._reprojection_error(objpoints, noisy, rvecs, tvecs,
                                        camera_matrix, np.zeros(5))
        assert err > 0.5


# ---------------------------------------------------------------------------
# _find_corners  (non-interactive, mocked cv2)
# ---------------------------------------------------------------------------

class TestFindCorners:

    def _make_fake_image(self, h=480, w=640):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def _make_fake_corners(self, n=6 * 8):
        return np.random.rand(n, 1, 2).astype(np.float32)

    def test_all_patterns_accepted_non_interactive(self, tmp_path):
        """In non-interactive mode every detected pattern should be accepted."""
        fake_images = [str(tmp_path / f"img_{i:02d}.jpg") for i in range(3)]
        for p in fake_images:
            open(p, "w").close()  # create empty placeholder

        fake_corners = self._make_fake_corners()
        objp = np.zeros((6 * 8, 3), np.float32)

        with (
            patch("cv2.imread", return_value=self._make_fake_image()),
            patch("cv2.cvtColor", return_value=np.zeros((480, 640), np.uint8)),
            patch("cv2.findChessboardCorners", return_value=(True, fake_corners)),
            patch("cv2.cornerSubPix", return_value=fake_corners),
        ):
            obj_pts, img_pts, img_size, _, n = calib._find_corners(
                fake_images, 8, 6, objp,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001),
                interactive=False,
            )

        assert n == 3
        assert len(obj_pts) == 3
        assert len(img_pts) == 3

    def test_unreadable_images_skipped(self, tmp_path):
        """Images that cv2.imread cannot read should be silently skipped."""
        path = str(tmp_path / "bad.jpg")
        open(path, "w").close()
        objp = np.zeros((6 * 8, 3), np.float32)

        with patch("cv2.imread", return_value=None):
            obj_pts, img_pts, _, _, n = calib._find_corners(
                [path], 8, 6, objp,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001),
                interactive=False,
            )
        assert n == 0
        assert obj_pts == []

    def test_calibresult_images_skipped(self, tmp_path):
        """Images containing 'calibresult' in their name must be skipped."""
        path = str(tmp_path / "calibresult.jpg")
        open(path, "w").close()
        objp = np.zeros((6 * 8, 3), np.float32)

        with patch("cv2.imread") as mock_read:
            calib._find_corners(
                [path], 8, 6, objp,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001),
                interactive=False,
            )
        mock_read.assert_not_called()


# ---------------------------------------------------------------------------
# _save_undistorted
# ---------------------------------------------------------------------------

class TestSaveUndistorted:

    def test_does_not_crash_on_unreadable_image(self, tmp_path, camera_matrix):
        """If cv2.imread returns None, _save_undistorted must return silently."""
        with patch("cv2.imread", return_value=None):
            calib._save_undistorted("nonexistent.jpg", camera_matrix, np.zeros(5),
                                    str(tmp_path))
        # No exception — test passes

    def test_writes_calibresult_png(self, tmp_path, camera_matrix):
        """A valid image should produce calibresult.png in out_folder."""
        fake_img = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_map = np.zeros((480, 640), dtype=np.float32)
        with (
            patch("cv2.imread", return_value=fake_img),
            patch("cv2.getOptimalNewCameraMatrix",
                  return_value=(camera_matrix, (0, 0, 640, 480))),
            patch("cv2.initUndistortRectifyMap",
                  return_value=(fake_map, fake_map)),
            patch("cv2.remap", return_value=fake_img),
            patch("cv2.imwrite") as mock_write,
        ):
            calib._save_undistorted("img.jpg", camera_matrix, np.zeros(5), str(tmp_path))

        assert mock_write.called
        saved_path = mock_write.call_args[0][0]
        assert saved_path.endswith("calibresult.png")
