# USAGE
# python facial_landmarks_video.py --shape-predictor \
#   shape_predictor_68_face_landmarks_finetuned.dat --video input_video.avi
# For faster batch processing: add --no-display
# To skip frames: add --skip-frames N (e.g., --skip-frames 2 processes every other frame)
# To export data: add --export-csv output.csv or --export-json output.json
# To save annotated video: add --output-video output.avi
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
import sys
import json
import matplotlib.pyplot as plt
from scipy.signal import medfilt, find_peaks

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-p", "--shape-predictor", required=True, help="path to facial landmark predictor"
)
ap.add_argument(
    "-v",
    "--video",
    default="proefpersoon 2_M.avi",
    help="path to input video file (default: proefpersoon 2_M.avi)",
)
ap.add_argument(
    "--no-display",
    action="store_true",
    help="disable video display for faster batch processing",
)
ap.add_argument(
    "--skip-frames",
    type=int,
    default=1,
    help="process every Nth frame (default: 1, process all frames)",
)
ap.add_argument("--export-csv", type=str, help="export mouth coordinates to CSV file")
ap.add_argument("--export-json", type=str, help="export mouth coordinates to JSON file")
ap.add_argument(
    "--output-video", type=str, help="save annotated video to file (e.g., output.avi)"
)
args = vars(ap.parse_args())

# Validate input files
if not os.path.exists(args["shape_predictor"]):
    print(f"Error: Shape predictor file not found: {args['shape_predictor']}")
    print("Please download the model from the link provided in the README")
    sys.exit(1)

if not os.path.exists(args["video"]):
    print(f"Error: Video file not found: {args['video']}")
    sys.exit(1)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
except Exception as e:
    print(f"Error initializing face detector or predictor: {e}")
    exit(1)

cap = cv2.VideoCapture(args["video"])

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file: {args['video']}")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames_to_process = total_frames // args["skip_frames"]

# Initialize video writer if output video requested
video_writer = None
if args["output_video"]:
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # Video will be resized to width=500, so calculate proportional height
    video_writer = cv2.VideoWriter(
        args["output_video"],
        fourcc,
        fps / args["skip_frames"],
        (
            500,
            int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                * 500
                / cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            ),
        ),
    )
    print(f"Saving annotated video to: {args['output_video']}")

# Preallocate arrays for better performance (avoid repeated list.append())
# We'll trim these later to actual detections
mouth_array_x = np.zeros(frames_to_process, dtype=np.float32)
mouth_array_y = np.zeros(frames_to_process, dtype=np.float32)
frame_count_arr = np.zeros(frames_to_process, dtype=np.int32)

print(f"Processing {total_frames} frames (every {args['skip_frames']} frame(s))...")
if args["no_display"]:
    print("Display disabled for faster processing")

processed_frames = 0
detection_count = 0
skip_counter = 0

try:
    while True:
        # Capture image-by-image
        ret, image = cap.read()

        # Check if video ended
        if not ret:
            print(
                f"\nVideo processing complete. Processed {processed_frames} "
                f"frames, detected {detection_count} mouth positions."
            )
            break

        processed_frames += 1

        # Skip frames if requested
        skip_counter += 1
        if skip_counter < args["skip_frames"]:
            continue
        skip_counter = 0

        # Progress indicator
        if processed_frames % 100 == 0:
            print(
                f"Processed {processed_frames}/{total_frames} frames "
                f"({detection_count} detections)",
                end="\r",
            )

        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = detector(gray, 1)

        # Process detected faces
        for i, rect in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Extract mouth coordinates (landmark 48 is the left corner of the mouth)
            for x, y in shape[48:49]:
                if detection_count < len(mouth_array_x):
                    mouth_array_x[detection_count] = x
                    mouth_array_y[detection_count] = y
                    frame_count_arr[detection_count] = processed_frames
                    detection_count += 1

            # Draw annotations if display enabled OR video output requested
            if not args["no_display"] or video_writer:
                # convert dlib's rectangle to a OpenCV-style bounding box
                # [i.e., (x, y, w, h)], then draw the face bounding box
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # show the face number
                cv2.putText(
                    image,
                    f"Face #{i + 1}",
                    (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw them on the image
                for x, y in shape:
                    cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

        # Write frame to output video if requested
        if video_writer:
            video_writer.write(image)

        # Only show display if not in no-display mode
        if not args["no_display"]:
            cv2.imshow("image", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nUser interrupted processing.")
                break

finally:
    # When everything done, release the capture
    cap.release()
    if video_writer:
        video_writer.release()
        print(f"Saved annotated video: {args['output_video']}")
    cv2.destroyAllWindows()

    # Trim arrays to actual detection count
    mouth_array_x = mouth_array_x[:detection_count]
    mouth_array_y = mouth_array_y[:detection_count]
    frame_count_arr = frame_count_arr[:detection_count]

    print(f"\nTotal detections: {detection_count}")
    print(f"Mouth X coordinates: {len(mouth_array_x)}")
    print(f"Mouth Y coordinates: {len(mouth_array_y)}")

# Plotting the results for estimation

# Check if we have valid data to plot
if detection_count == 0:
    print("\nError: No mouth coordinates detected in the video.")
    print("Please ensure:")
    print("  1. The video contains visible faces")
    print("  2. The shape predictor model is correct")
    print("  3. The video quality is sufficient for detection")
    sys.exit(1)

# Check for zero sum to avoid division by zero
x_sum = np.sum(mouth_array_x)
if x_sum == 0:
    print(
        "\nWarning: Sum of X coordinates is zero. Using raw values instead of normalized."
    )
    x = mouth_array_x
else:
    x = mouth_array_x / x_sum

y = mouth_array_y

peak_estimates = find_peaks(x)
print(f"\nPeak estimates: {peak_estimates[0]}")
array_len = len(peak_estimates[0])

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(frame_count_arr, medfilt(x), label="Relative Motion of X-Coordinates")
plt.title("Graphical Representation")
ax.legend()
fig.savefig("plot_x.png")
plt.close(fig)
print("Saved plot_x.png")

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(frame_count_arr, medfilt(y), label="Relative Motion of Y-Coordinates")
plt.title("Graphical Representation")
ax.legend()
fig.savefig("plot_y.png")
plt.close(fig)
print("Saved plot_y.png")

# Export data to CSV if requested
if args["export_csv"]:
    import csv

    with open(args["export_csv"], "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "mouth_x", "mouth_y"])
        for i in range(detection_count):
            writer.writerow([frame_count_arr[i], mouth_array_x[i], mouth_array_y[i]])
    print(f"Exported data to CSV: {args['export_csv']}")

# Export data to JSON if requested
if args["export_json"]:
    data = {
        "video_file": args["video"],
        "total_frames": total_frames,
        "frames_processed": processed_frames,
        "detections": detection_count,
        "skip_frames": args["skip_frames"],
        "coordinates": [
            {
                "frame": int(frame_count_arr[i]),
                "mouth_x": float(mouth_array_x[i]),
                "mouth_y": float(mouth_array_y[i]),
            }
            for i in range(detection_count)
        ],
    }
    with open(args["export_json"], "w") as jsonfile:
        json.dump(data, jsonfile, indent=2)
    print(f"Exported data to JSON: {args['export_json']}")

print("\nProcessing complete!")
