#!/usr/bin/env python3
"""
Example: Batch process multiple videos with tongue tracking

This script demonstrates how to process multiple videos in batch mode
and export results in various formats.
"""

import os
import subprocess
import glob
import json
from pathlib import Path

# Configuration
INPUT_DIR = "./videos"
OUTPUT_DIR = "./results"
MODEL_PATH = "./shape_predictor_68_face_landmarks_finetuned.dat"

# Processing options
SKIP_FRAMES = 2  # Process every 2nd frame for faster processing
NO_DISPLAY = True  # Disable display for batch mode

def process_video(video_path, output_dir):
    """
    Process a single video file

    Args:
        video_path: Path to input video
        output_dir: Directory for output files
    """
    video_name = Path(video_path).stem
    output_subdir = os.path.join(output_dir, video_name)
    os.makedirs(output_subdir, exist_ok=True)

    # Prepare output paths
    csv_output = os.path.join(output_subdir, f"{video_name}.csv")
    json_output = os.path.join(output_subdir, f"{video_name}.json")
    video_output = os.path.join(output_subdir, f"{video_name}_annotated.avi")

    # Build command
    cmd = [
        "python", "facial_landmarks_video.py",
        "--shape-predictor", MODEL_PATH,
        "--video", video_path,
        "--export-csv", csv_output,
        "--export-json", json_output,
        "--output-video", video_output,
        "--skip-frames", str(SKIP_FRAMES),
    ]

    if NO_DISPLAY:
        cmd.append("--no-display")

    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"{'='*60}")

    try:
        # Run the processing
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        # Load and display summary from JSON
        if os.path.exists(json_output):
            with open(json_output, 'r') as f:
                data = json.load(f)
                print(f"\nResults Summary:")
                print(f"  Total frames: {data['total_frames']}")
                print(f"  Processed: {data['frames_processed']}")
                print(f"  Detections: {data['detections']}")
                print(f"  Detection rate: {data['detections']/data['frames_processed']*100:.1f}%")

        return True

    except subprocess.CalledProcessError as e:
        print(f"Error processing {video_name}: {e.stderr}")
        return False

def main():
    """Main batch processing function"""

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found: {MODEL_PATH}")
        print("Please download from: https://drive.google.com/file/d/1kEOn0SsyToOCGr45UDygxnkDo4uxlWeh/view?usp=sharing")
        return

    # Find all video files
    video_patterns = ["*.avi", "*.mp4", "*.mov"]
    video_files = []
    for pattern in video_patterns:
        video_files.extend(glob.glob(os.path.join(INPUT_DIR, pattern)))

    if not video_files:
        print(f"No video files found in {INPUT_DIR}")
        print(f"Looking for: {', '.join(video_patterns)}")
        return

    print(f"Found {len(video_files)} video(s) to process")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each video
    success_count = 0
    for video_path in video_files:
        if process_video(video_path, OUTPUT_DIR):
            success_count += 1

    # Print summary
    print(f"\n{'='*60}")
    print(f"Batch Processing Complete")
    print(f"{'='*60}")
    print(f"Successfully processed: {success_count}/{len(video_files)} videos")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
