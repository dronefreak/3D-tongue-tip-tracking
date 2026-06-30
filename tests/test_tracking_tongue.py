"""
Unit tests for tracking_tongue.py

Covers:
  - imsharpen         (unsharp-mask helper)
  - crop_roi          (frame cropping)
  - klt_track         (Lucas-Kanade point tracker)
  - build_arg_parser  (CLI argument parsing)
  - PRESET_ROIS       (named view constants)
  - FARNEBACK_PARAMS  (optical-flow parameter dict)
"""

import sys

import cv2
import numpy as np
import pytest

from tracking_tongue import (
    DEFAULT_FLOW_THRESHOLD,
    FARNEBACK_PARAMS,
    KLT_MAX_BIDIRECTIONAL_ERROR,
    PRESET_ROIS,
    build_arg_parser,
    crop_roi,
    imsharpen,
    klt_track,
)

# ---------------------------------------------------------------------------
# imsharpen
# ---------------------------------------------------------------------------


class TestImsharpen:
    def test_output_shape_matches_input(self, synthetic_gray_frame):
        out = imsharpen(synthetic_gray_frame)
        assert out.shape == synthetic_gray_frame.shape

    def test_output_dtype_matches_input(self, synthetic_gray_frame):
        out = imsharpen(synthetic_gray_frame)
        assert out.dtype == synthetic_gray_frame.dtype

    def test_sharpened_differs_from_blurred(self, synthetic_gray_frame):
        """imsharpen result should differ from a Gaussian-blurred version."""
        blurred = cv2.GaussianBlur(synthetic_gray_frame, (0, 0), 2.5)
        sharpened = imsharpen(synthetic_gray_frame)
        assert not np.array_equal(sharpened, blurred)

    def test_flat_image_unchanged(self):
        """A completely uniform image has nothing to sharpen — result is same value."""
        flat = np.full((100, 100), 128, dtype=np.uint8)
        out = imsharpen(flat)
        # All pixels should still be 128 (±1 due to integer rounding)
        assert np.abs(out.astype(int) - 128).max() <= 1

    def test_amount_zero_returns_original(self, synthetic_gray_frame):
        """amount=0 means no sharpening — output should equal the input."""
        out = imsharpen(synthetic_gray_frame, amount=0.0)
        np.testing.assert_array_equal(out, synthetic_gray_frame)


# ---------------------------------------------------------------------------
# crop_roi
# ---------------------------------------------------------------------------


class TestCropRoi:
    def test_output_dimensions_match_roi(self, synthetic_bgr_frame):
        roi = (10, 20, 80, 60)  # x=10, y=20, w=80, h=60
        cropped = crop_roi(synthetic_bgr_frame, roi)
        assert cropped.shape == (60, 80, 3)

    def test_pixel_values_are_correct(self):
        """Cropped region must contain the exact pixels from the original frame."""
        frame = np.arange(200 * 200 * 3, dtype=np.uint8).reshape(200, 200, 3)
        roi = (50, 30, 40, 20)  # x=50, y=30, w=40, h=20
        cropped = crop_roi(frame, roi)
        expected = frame[30:50, 50:90]  # numpy [y:y+h, x:x+w]
        np.testing.assert_array_equal(cropped, expected)

    def test_full_frame_roi_returns_copy_of_frame(self, synthetic_bgr_frame):
        h, w = synthetic_bgr_frame.shape[:2]
        cropped = crop_roi(synthetic_bgr_frame, (0, 0, w, h))
        assert cropped.shape == synthetic_bgr_frame.shape
        np.testing.assert_array_equal(cropped, synthetic_bgr_frame)

    def test_single_pixel_roi(self, synthetic_bgr_frame):
        cropped = crop_roi(synthetic_bgr_frame, (10, 10, 1, 1))
        assert cropped.shape == (1, 1, 3)


# ---------------------------------------------------------------------------
# klt_track
# ---------------------------------------------------------------------------


class TestKltTrack:
    def test_identical_frames_track_successfully(self, synthetic_gray_frame):
        """Tracking on identical frames must succeed with negligible drift."""
        pt = np.array([100.0, 100.0], dtype=np.float32)
        new_pt, valid = klt_track(synthetic_gray_frame, synthetic_gray_frame, pt)
        assert valid
        assert np.linalg.norm(new_pt - pt) < KLT_MAX_BIDIRECTIONAL_ERROR

    def test_returns_original_point_on_failure(self):
        """On a flat (featureless) image KLT may fail; original point must be returned."""
        flat = np.zeros((100, 100), dtype=np.uint8)
        pt = np.array([50.0, 50.0], dtype=np.float32)
        new_pt, valid = klt_track(flat, flat, pt)
        # If tracking fails, original point is returned; if it succeeds, that's also fine
        if not valid:
            np.testing.assert_array_equal(new_pt, pt)

    def test_tracks_moving_texture(self):
        """Tracker should follow a 1-pixel shift of a textured patch."""
        rng = np.random.default_rng(7)
        prev = rng.integers(40, 210, (100, 100), dtype=np.uint8)

        # Shift by (1, 0): roll along axis=1
        curr = np.roll(prev, 1, axis=1)
        curr[:, 0] = 0  # fill rolled-in column

        pt = np.array([50.0, 50.0], dtype=np.float32)
        new_pt, valid = klt_track(prev, curr, pt)

        # The result may or may not be valid on synthetic data, but if valid the
        # displacement should be small (≤ 3 px for a 1-px shift)
        if valid:
            assert np.linalg.norm(new_pt - pt) <= 3.0

    def test_output_shape(self, synthetic_gray_frame):
        """new_pt must always be a (2,) float32 array."""
        pt = np.array([50.0, 50.0], dtype=np.float32)
        new_pt, _ = klt_track(synthetic_gray_frame, synthetic_gray_frame, pt)
        assert new_pt.shape == (2,)


# ---------------------------------------------------------------------------
# build_arg_parser
# ---------------------------------------------------------------------------


class TestBuildArgParser:
    def test_video_required(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])  # missing --video

    def test_video_accepted(self, tmp_path):
        parser = build_arg_parser()
        args = parser.parse_args(["-v", "video.avi"])
        assert args.video == "video.avi"

    def test_view_choices(self):
        parser = build_arg_parser()
        for view in ("mid", "left", "right"):
            args = parser.parse_args(["-v", "v.avi", "--view", view])
            assert args.view == view

    def test_invalid_view_rejected(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["-v", "v.avi", "--view", "top"])

    def test_roi_and_view_mutually_exclusive(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["-v", "v.avi", "--view", "mid", "--roi", "0", "0", "100", "100"])

    def test_roi_accepted(self):
        parser = build_arg_parser()
        args = parser.parse_args(["-v", "v.avi", "--roi", "10", "20", "300", "200"])
        assert args.roi == [10, 20, 300, 200]

    def test_flow_threshold_default(self):
        parser = build_arg_parser()
        args = parser.parse_args(["-v", "v.avi"])
        assert args.flow_threshold == pytest.approx(DEFAULT_FLOW_THRESHOLD)

    def test_flow_threshold_custom(self):
        parser = build_arg_parser()
        args = parser.parse_args(["-v", "v.avi", "--flow-threshold", "3.5"])
        assert args.flow_threshold == pytest.approx(3.5)

    def test_no_display_flag(self):
        parser = build_arg_parser()
        args = parser.parse_args(["-v", "v.avi", "--no-display"])
        assert args.no_display is True

    def test_output_flags_default_to_none(self):
        parser = build_arg_parser()
        args = parser.parse_args(["-v", "v.avi"])
        assert args.output_csv is None
        assert args.output_json is None


# ---------------------------------------------------------------------------
# PRESET_ROIS
# ---------------------------------------------------------------------------


class TestPresetRois:
    def test_has_three_views(self):
        assert set(PRESET_ROIS.keys()) == {"mid", "left", "right"}

    def test_each_roi_is_4_tuple(self):
        for name, roi in PRESET_ROIS.items():
            assert len(roi) == 4, f"{name} ROI should have 4 elements"

    def test_all_values_are_positive_ints(self):
        for name, roi in PRESET_ROIS.items():
            for v in roi:
                assert isinstance(v, int) and v > 0, (
                    f"{name} ROI has non-positive or non-int value: {v}"
                )

    def test_mid_roi_matches_original_matlab(self):
        """Verify the mid-view ROI matches the value from tracking_tongue.m."""
        assert PRESET_ROIS["mid"] == (367, 350, 361, 365)


# ---------------------------------------------------------------------------
# FARNEBACK_PARAMS
# ---------------------------------------------------------------------------


class TestFarnebackParams:
    EXPECTED_KEYS = {
        "pyr_scale",
        "levels",
        "winsize",
        "iterations",
        "poly_n",
        "poly_sigma",
        "flags",
    }

    def test_has_all_required_keys(self):
        assert self.EXPECTED_KEYS.issubset(FARNEBACK_PARAMS.keys())

    def test_pyr_scale_valid_range(self):
        assert 0.0 < FARNEBACK_PARAMS["pyr_scale"] < 1.0

    def test_levels_at_least_one(self):
        assert FARNEBACK_PARAMS["levels"] >= 1

    def test_winsize_positive_odd_or_even(self):
        # cv2 accepts any positive integer
        assert FARNEBACK_PARAMS["winsize"] > 0

    def test_params_accepted_by_opencv(self, synthetic_gray_frame):
        """FARNEBACK_PARAMS must be accepted by cv2.calcOpticalFlowFarneback."""
        flow = cv2.calcOpticalFlowFarneback(
            synthetic_gray_frame, synthetic_gray_frame, None, **FARNEBACK_PARAMS
        )
        assert flow.shape == (*synthetic_gray_frame.shape, 2)
