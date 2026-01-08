#!/usr/bin/env python3
"""
Real-time tongue tip tracking using webcam

This script performs real-time facial landmark detection and tongue tracking
using a webcam feed. Press 'q' to quit, 'r' to start/stop recording.
"""
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
import sys
import json
import time
from datetime import datetime

def main():
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser(description="Real-time tongue tip tracking with webcam")
    ap.add_argument("-p", "--shape-predictor", required=True,
        help="path to facial landmark predictor")
    ap.add_argument("-c", "--camera", type=int, default=0,
        help="camera device index (default: 0)")
    ap.add_argument("-w", "--width", type=int, default=640,
        help="frame width (default: 640)")
    ap.add_argument("-r", "--record", action="store_true",
        help="enable recording mode")
    ap.add_argument("--export-csv", type=str,
        help="export mouth coordinates to CSV file")
    ap.add_argument("--export-json", type=str,
        help="export mouth coordinates to JSON file")
    ap.add_argument("--fps", type=int, default=30,
        help="target FPS for recording (default: 30)")
    args = vars(ap.parse_args())

    # Validate model file
    if not os.path.exists(args["shape_predictor"]):
        print(f"Error: Shape predictor file not found: {args['shape_predictor']}")
        print("Please download the model from the link provided in the README")
        sys.exit(1)

    # Initialize dlib's face detector and shape predictor
    print("Loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    # Initialize webcam
    print(f"Initializing camera {args['camera']}...")
    cap = cv2.VideoCapture(args["camera"])

    if not cap.isOpened():
        print(f"Error: Could not open camera {args['camera']}")
        print("Try a different camera index with --camera N")
        sys.exit(1)

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args["width"])
    cap.set(cv2.CAP_PROP_FPS, args["fps"])

    # Get actual camera properties
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
    print("\nControls:")
    print("  'q' - Quit")
    print("  'r' - Start/Stop recording")
    print("  'c' - Clear recorded data")
    print("\nPress any key to start...")

    cv2.waitKey(0)

    # Arrays to store tracking data
    mouth_array_x = []
    mouth_array_y = []
    timestamp_arr = []
    frame_count_arr = []

    # Recording state
    is_recording = args["record"]
    recording_started = False
    frame_count = 0
    start_time = time.time()

    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0

    print("\nTracking started. Press 'q' to quit.")

    try:
        while True:
            # Capture frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to grab frame")
                break

            frame_count += 1
            current_time = time.time() - start_time

            # Resize frame for faster processing
            frame = imutils.resize(frame, width=args["width"])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            rects = detector(gray, 1)

            # Process detected faces
            for (i, rect) in enumerate(rects):
                # Get facial landmarks
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # Extract mouth coordinates (landmark 48 is left corner of mouth)
                mouth_x, mouth_y = shape[48]

                # Store data if recording
                if is_recording:
                    mouth_array_x.append(mouth_x)
                    mouth_array_y.append(mouth_y)
                    timestamp_arr.append(current_time)
                    frame_count_arr.append(frame_count)

                    if not recording_started:
                        recording_started = True
                        print("Recording started!")

                # Draw bounding box
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw face number
                cv2.putText(frame, f"Face #{i + 1}", (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw all facial landmarks
                for (lx, ly) in shape:
                    cv2.circle(frame, (lx, ly), 2, (0, 0, 255), -1)

                # Highlight mouth landmark
                cv2.circle(frame, (mouth_x, mouth_y), 5, (255, 0, 0), -1)

                # Draw mouth position text
                cv2.putText(frame, f"Mouth: ({mouth_x}, {mouth_y})",
                    (10, actual_height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Calculate FPS
            fps_frame_count += 1
            if time.time() - fps_start_time > 1.0:
                current_fps = fps_frame_count / (time.time() - fps_start_time)
                fps_frame_count = 0
                fps_start_time = time.time()

            # Draw status information
            status_text = "REC" if is_recording else "PAUSED"
            status_color = (0, 0, 255) if is_recording else (0, 255, 255)
            cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(frame, f"Frames: {frame_count}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(frame, f"Detections: {len(mouth_array_x)}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display frame
            cv2.imshow('Tongue Tip Tracking (Webcam)', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('r'):
                is_recording = not is_recording
                if is_recording:
                    print("Recording resumed" if recording_started else "Recording started")
                else:
                    print("Recording paused")
            elif key == ord('c'):
                mouth_array_x.clear()
                mouth_array_y.clear()
                timestamp_arr.clear()
                frame_count_arr.clear()
                frame_count = 0
                start_time = time.time()
                recording_started = False
                print("Data cleared")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Export data if requested
        if len(mouth_array_x) > 0:
            print(f"\nRecorded {len(mouth_array_x)} data points")

            # Export to CSV
            if args["export_csv"]:
                import csv
                with open(args["export_csv"], 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['frame', 'timestamp', 'mouth_x', 'mouth_y'])
                    for i in range(len(mouth_array_x)):
                        writer.writerow([
                            frame_count_arr[i],
                            timestamp_arr[i],
                            mouth_array_x[i],
                            mouth_array_y[i]
                        ])
                print(f"Exported data to CSV: {args['export_csv']}")

            # Export to JSON
            if args["export_json"]:
                data = {
                    'recording_date': datetime.now().isoformat(),
                    'camera_index': args['camera'],
                    'frame_width': args['width'],
                    'target_fps': args['fps'],
                    'total_frames': frame_count,
                    'detections': len(mouth_array_x),
                    'duration_seconds': timestamp_arr[-1] if timestamp_arr else 0,
                    'coordinates': [
                        {
                            'frame': int(frame_count_arr[i]),
                            'timestamp': float(timestamp_arr[i]),
                            'mouth_x': float(mouth_array_x[i]),
                            'mouth_y': float(mouth_array_y[i])
                        }
                        for i in range(len(mouth_array_x))
                    ]
                }
                with open(args["export_json"], 'w') as jsonfile:
                    json.dump(data, jsonfile, indent=2)
                print(f"Exported data to JSON: {args['export_json']}")

            print("\nSession complete!")
        else:
            print("\nNo data recorded")

if __name__ == "__main__":
    main()
