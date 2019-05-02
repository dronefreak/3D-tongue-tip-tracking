# USAGE
# python facial_landmarks_video.py --shape-predictor shape_predictor_68_face_landmarks_finetuned.dat
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# define the arrays for further appending the coordinates
mouth_array_x = []
mouth_array_y = []

cap = cv2.VideoCapture('proefpersoon 2_M.avi')
frame_count = 0
frame_count_arr = []
while(True):
	# Capture image-by-image
	ret, image = cap.read()
	image = imutils.resize(image, width=500)
	# Our operations on the image come here
	#image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# detect faces in the grayscale image
	rects = detector(gray, 1)

	for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		for (x, y) in shape[48:49]:
			mouth_array_x.append(x)
			frame_count = frame_count + 1
			frame_count_arr.append(frame_count)
			mouth_array_y.append(y)

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

print(len(frame_count_arr))
print(len(mouth_array_x))

# Plotting the results for estimation

import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import medfilt, find_peaks

y = mouth_array_y
x = mouth_array_x/np.sum(mouth_array_x)
peak_estimates = find_peaks(x)

print (peak_estimates[0])
array_len = len(peak_estimates[0])

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(frame_count_arr,medfilt(x), label='Relative Motion of X-Coordinates')
plt.title('Graphical Representation')
ax.legend()
fig.savefig('plot_x.png')

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(frame_count_arr,medfilt(y), label='Relative Motion of Y-Coordinates')
plt.title('Graphical Representation')
ax.legend()
fig.savefig('plot_y.png')