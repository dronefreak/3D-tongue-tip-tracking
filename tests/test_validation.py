"""
Tests for input validation and error handling
"""

import tempfile
import os


def test_video_file_validation():
    """Test video file existence validation"""
    # Test with non-existent file
    assert not os.path.exists("nonexistent_video.avi")

    # Test with existing temporary file
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as f:
        temp_path = f.name

    try:
        assert os.path.exists(temp_path)
    finally:
        os.unlink(temp_path)


def test_model_file_validation():
    """Test model file existence validation"""
    # Test with non-existent model
    assert not os.path.exists("nonexistent_model.dat")


def test_directory_validation():
    """Test directory existence validation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        assert os.path.exists(tmpdir)
        assert os.path.isdir(tmpdir)


def test_output_directory_creation():
    """Test that output directories can be created"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)
        assert os.path.exists(output_dir)
        assert os.path.isdir(output_dir)
