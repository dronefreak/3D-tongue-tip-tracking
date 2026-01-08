# USAGE
# python facial_landmarks_video.py --shape-predictor shape_predictor_68_face_landmarks_finetuned.dat --video input_video.avi
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import medfilt, find_peaks

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", default="proefpersoon 2_M.avi",
	help="path to input video file (default: proefpersoon 2_M.avi)")
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
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# define the arrays for further appending the coordinates
mouth_array_x = []
mouth_array_y = []

cap = cv2.VideoCapture(args["video"])

# Check if video opened successfully
if not cap.isOpened():
	print(f"Error: Could not open video file: {args['video']}")
	sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Processing {total_frames} frames...")

frame_count = 0
frame_count_arr = []
processed_frames = 0

while(True):
	# Capture image-by-image
	ret, image = cap.read()

	# Check if video ended
	if not ret:
		print(f"\nVideo processing complete. Processed {processed_frames} frames.")
		break

	processed_frames += 1
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# detect faces in the grayscale image
	rects = detector(gray, 1)

	# Process detected faces
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# Extract mouth coordinates (landmark 48 is the left corner of the mouth)
		for (x, y) in shape[48:49]:
			mouth_array_x.append(x)
			mouth_array_y.append(y)
			frame_count_arr.append(processed_frames)

		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# show the face number
		cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
	cv2.imwrite('image.png',image)
	cv2.imshow('image',image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

print(f"\nTotal detections: {len(frame_count_arr)}")
print(f"Mouth X coordinates: {len(mouth_array_x)}")
print(f"Mouth Y coordinates: {len(mouth_array_y)}")

# Plotting the results for estimation

# Check if we have valid data to plot
if len(mouth_array_x) == 0 or len(mouth_array_y) == 0:
	print("\nError: No mouth coordinates detected in the video.")
	print("Please ensure:")
	print("  1. The video contains visible faces")
	print("  2. The shape predictor model is correct")
	print("  3. The video quality is sufficient for detection")
	sys.exit(1)

# Convert to numpy arrays for processing
mouth_array_x = np.array(mouth_array_x)
mouth_array_y = np.array(mouth_array_y)
frame_count_arr = np.array(frame_count_arr)

# Check for zero sum to avoid division by zero
x_sum = np.sum(mouth_array_x)
if x_sum == 0:
	print("\nWarning: Sum of X coordinates is zero. Using raw values instead of normalized.")
	x = mouth_array_x
else:
	x = mouth_array_x / x_sum

y = mouth_array_y

peak_estimates = find_peaks(x)
print(f"\nPeak estimates: {peak_estimates[0]}")
array_len = len(peak_estimates[0])

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(frame_count_arr, medfilt(x), label='Relative Motion of X-Coordinates')
plt.title('Graphical Representation')
ax.legend()
fig.savefig('plot_x.png')
plt.close(fig)
print("Saved plot_x.png")

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(frame_count_arr, medfilt(y), label='Relative Motion of Y-Coordinates')
plt.title('Graphical Representation')
ax.legend()
fig.savefig('plot_y.png')
plt.close(fig)
print("Saved plot_y.png")

print("\nProcessing complete!")