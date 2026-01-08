"""
Tests for data processing and array operations
"""
import pytest
import numpy as np
from scipy.signal import medfilt, find_peaks


def test_array_preallocation():
    """Test numpy array preallocation"""
    size = 1000
    arr_x = np.zeros(size, dtype=np.float32)
    arr_y = np.zeros(size, dtype=np.float32)
    frames = np.zeros(size, dtype=np.int32)

    assert arr_x.shape == (size,)
    assert arr_y.shape == (size,)
    assert frames.shape == (size,)
    assert arr_x.dtype == np.float32
    assert frames.dtype == np.int32


def test_array_trimming():
    """Test trimming preallocated arrays to actual size"""
    size = 1000
    actual_count = 750

    arr = np.zeros(size, dtype=np.float32)
    # Fill first 750 elements
    arr[:actual_count] = np.random.randn(actual_count)

    # Trim to actual size
    trimmed = arr[:actual_count]

    assert trimmed.shape == (actual_count,)
    assert len(trimmed) == actual_count


def test_division_by_zero_protection():
    """Test protection against division by zero"""
    arr = np.array([1.0, 2.0, 3.0])
    total = np.sum(arr)

    if total == 0:
        result = arr  # Use raw values
    else:
        result = arr / total

    assert np.sum(result) > 0  # Should have normalized or kept original


def test_zero_sum_handling():
    """Test handling of array with zero sum"""
    arr_zero = np.array([0.0, 0.0, 0.0])
    total_zero = np.sum(arr_zero)

    if total_zero == 0:
        result = arr_zero
    else:
        result = arr_zero / total_zero

    assert np.allclose(result, arr_zero)


def test_median_filter():
    """Test median filtering works correctly"""
    data = np.array([1, 2, 100, 4, 5], dtype=float)  # 100 is outlier
    filtered = medfilt(data, kernel_size=3)

    # Median filter should reduce the outlier effect
    assert filtered[2] < 100  # Middle value should be smoothed
    assert len(filtered) == len(data)


def test_peak_detection():
    """Test peak finding in signal"""
    # Create signal with clear peaks
    x = np.linspace(0, 10, 100)
    signal = np.sin(x) + np.sin(2*x)

    peaks, _ = find_peaks(signal)

    assert len(peaks) > 0  # Should find some peaks
    assert all(0 <= p < len(signal) for p in peaks)  # Valid indices


def test_coordinate_normalization():
    """Test coordinate normalization"""
    coords = np.array([100, 200, 300, 400, 500], dtype=float)
    total = np.sum(coords)

    normalized = coords / total

    assert np.isclose(np.sum(normalized), 1.0)  # Should sum to 1
    assert np.all(normalized >= 0)  # All values should be positive
    assert np.all(normalized <= 1)  # All values should be <= 1


def test_empty_array_handling():
    """Test that operations handle empty arrays gracefully"""
    empty_arr = np.array([], dtype=float)

    assert len(empty_arr) == 0
    assert empty_arr.size == 0

    # Check sum of empty array
    total = np.sum(empty_arr)
    assert total == 0


def test_frame_coordinate_alignment():
    """Test that frame numbers and coordinates align properly"""
    num_detections = 100
    frames = np.arange(1, num_detections + 1, dtype=np.int32)
    coords_x = np.random.rand(num_detections).astype(np.float32)
    coords_y = np.random.rand(num_detections).astype(np.float32)

    assert len(frames) == len(coords_x) == len(coords_y)
    assert frames[0] == 1
    assert frames[-1] == num_detections
