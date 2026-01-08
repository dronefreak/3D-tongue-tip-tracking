#!/usr/bin/env python3
"""
Example: Analyze exported tracking results

This script demonstrates how to load and analyze the exported
CSV and JSON data from tongue tracking.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_tracking_data(json_path=None, csv_path=None):
    """
    Load tracking data from JSON or CSV

    Args:
        json_path: Path to JSON export file
        csv_path: Path to CSV export file

    Returns:
        DataFrame with tracking data
    """
    if json_path and Path(json_path).exists():
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Convert to DataFrame
        df = pd.DataFrame(data['coordinates'])
        print(f"Loaded {len(df)} detections from JSON")
        print(f"Video: {data['video_file']}")
        print(f"Total frames: {data['total_frames']}")
        print(f"Detection rate: {len(df)/data['frames_processed']*100:.1f}%")

        return df

    elif csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} detections from CSV")
        return df

    else:
        raise FileNotFoundError("No valid data file found")

def compute_statistics(df):
    """Compute basic statistics on tracking data"""

    stats = {
        'mean_x': df['mouth_x'].mean(),
        'std_x': df['mouth_x'].std(),
        'min_x': df['mouth_x'].min(),
        'max_x': df['mouth_x'].max(),
        'range_x': df['mouth_x'].max() - df['mouth_x'].min(),

        'mean_y': df['mouth_y'].mean(),
        'std_y': df['mouth_y'].std(),
        'min_y': df['mouth_y'].min(),
        'max_y': df['mouth_y'].max(),
        'range_y': df['mouth_y'].max() - df['mouth_y'].min(),
    }

    # Compute velocity (change per frame)
    df['velocity_x'] = df['mouth_x'].diff()
    df['velocity_y'] = df['mouth_y'].diff()
    df['velocity_magnitude'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)

    stats['mean_velocity'] = df['velocity_magnitude'].mean()
    stats['max_velocity'] = df['velocity_magnitude'].max()

    return stats

def plot_trajectory(df, output_path='trajectory.png'):
    """Plot the tongue tip trajectory over time"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # X coordinate over time
    axes[0, 0].plot(df['frame'], df['mouth_x'], linewidth=0.5)
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('X Coordinate (pixels)')
    axes[0, 0].set_title('Horizontal Position Over Time')
    axes[0, 0].grid(True, alpha=0.3)

    # Y coordinate over time
    axes[0, 1].plot(df['frame'], df['mouth_y'], linewidth=0.5, color='orange')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Y Coordinate (pixels)')
    axes[0, 1].set_title('Vertical Position Over Time')
    axes[0, 1].grid(True, alpha=0.3)

    # 2D trajectory
    axes[1, 0].plot(df['mouth_x'], df['mouth_y'], linewidth=0.5, alpha=0.7)
    axes[1, 0].scatter(df['mouth_x'].iloc[0], df['mouth_y'].iloc[0],
                       c='green', s=100, label='Start', zorder=5)
    axes[1, 0].scatter(df['mouth_x'].iloc[-1], df['mouth_y'].iloc[-1],
                       c='red', s=100, label='End', zorder=5)
    axes[1, 0].set_xlabel('X Coordinate (pixels)')
    axes[1, 0].set_ylabel('Y Coordinate (pixels)')
    axes[1, 0].set_title('2D Trajectory')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].invert_yaxis()  # Invert Y for image coordinates

    # Velocity magnitude
    if 'velocity_magnitude' in df.columns:
        axes[1, 1].plot(df['frame'], df['velocity_magnitude'], linewidth=0.5, color='green')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Velocity (pixels/frame)')
        axes[1, 1].set_title('Movement Velocity')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nTrajectory plot saved to: {output_path}")

def main():
    """Main analysis function"""

    # Example usage - modify paths as needed
    json_path = "results.json"
    csv_path = "results.csv"

    # Load data
    try:
        df = load_tracking_data(json_path=json_path, csv_path=csv_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run facial_landmarks_video.py with --export-json or --export-csv first")
        return

    # Compute statistics
    print("\n" + "="*60)
    print("Tracking Statistics")
    print("="*60)

    stats = compute_statistics(df)

    print(f"\nPosition Statistics:")
    print(f"  X: {stats['mean_x']:.2f} ± {stats['std_x']:.2f} pixels")
    print(f"     Range: [{stats['min_x']:.2f}, {stats['max_x']:.2f}] ({stats['range_x']:.2f} pixels)")
    print(f"  Y: {stats['mean_y']:.2f} ± {stats['std_y']:.2f} pixels")
    print(f"     Range: [{stats['min_y']:.2f}, {stats['max_y']:.2f}] ({stats['range_y']:.2f} pixels)")

    print(f"\nMovement Statistics:")
    print(f"  Mean velocity: {stats['mean_velocity']:.2f} pixels/frame")
    print(f"  Max velocity: {stats['max_velocity']:.2f} pixels/frame")

    # Create visualizations
    plot_trajectory(df)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == "__main__":
    main()
