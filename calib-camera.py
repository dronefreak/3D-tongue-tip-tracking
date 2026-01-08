#!/usr/bin/env python

"""
From https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration

Calling:
cameracalib.py  <folder> <image type> <num rows> <num cols> <cell dimension>

like cameracalib.py folder_name png

--h for help
"""

import numpy as np
import cv2
import glob
import sys
import os

#---------------------- SET THE PARAMETERS
nRows = 8
nCols = 8
dimension = 20 #- mm (checkerboard square size)


workingFolder   = "./camera_01"
imageType       = 'jpg'
#------------------------------------------

# termination criteria for corner refinement
# (type, max_iterations, epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nRows*nCols,3), np.float32)
objp[:,:2] = np.mgrid[0:nCols,0:nRows].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

if len(sys.argv) < 6:
        print("\n Not enough inputs are provided. Using the default values.\n\n" \
              " type -h for help")
else:
    workingFolder   = sys.argv[1]
    imageType       = sys.argv[2]
    nRows           = int(sys.argv[3])
    nCols           = int(sys.argv[4])
    dimension       = float(sys.argv[5])

if '-h' in sys.argv or '--h' in sys.argv:
    print("\n IMAGE CALIBRATION GIVEN A SET OF IMAGES")
    print(" call: python cameracalib.py <folder> <image type> <num rows (9)> <num cols (6)> <cell dimension (25)>")
    print("\n The script will look for every image in the provided folder and will show the pattern found." \
          " User can skip the image pressing ESC or accepting the image with RETURN. " \
          " At the end the end the following files are created:" \
          "  - cameraDistortion.txt" \
          "  - cameraMatrix.txt \n\n")

    sys.exit()

# Validate working folder exists
if not os.path.exists(workingFolder):
    print(f"Error: Working folder does not exist: {workingFolder}")
    print("Please create the folder and add calibration images.")
    sys.exit(1)

# Find the images files
filename    = workingFolder + "/*." + imageType
images      = glob.glob(filename)

print(f"Found {len(images)} images in {workingFolder}")
if len(images) < 9:
    print("Not enough images were found: at least 9 shall be provided!!!")
    sys.exit(1)

# Process images
nPatternFound = 0
imgNotGood = images[0]  # Use first image as fallback

for fname in images:
    if 'calibresult' in fname:
        continue

    # Read the file and convert to greyscale
    img = cv2.imread(fname)

    # Validate image was read successfully
    if img is None:
        print(f"Warning: Could not read image: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"Reading image: {fname}")

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (nCols, nRows), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print("Pattern found! Press ESC to skip or ENTER to accept")
        # Refine corner positions for better accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nCols, nRows), corners2, ret)
        cv2.imshow('img', img)

        k = cv2.waitKey(0) & 0xFF
        if k == 27:  # ESC Button
            print("Image Skipped")
            imgNotGood = fname
            continue

        print("Image accepted")
        nPatternFound += 1
        objpoints.append(objp)
        imgpoints.append(corners2)
    else:
        imgNotGood = fname


cv2.destroyAllWindows()

if (nPatternFound > 1):
    print(f"Found {nPatternFound} good images")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Undistort an image
    img = cv2.imread(imgNotGood)

    # Validate image was read successfully
    if img is None:
        print(f"Error: Could not read image for undistortion: {imgNotGood}")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"Image to undistort: {imgNotGood}")
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistort
    mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
    dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    print(f"ROI: x={x}, y={y}, w={w}, h={h}")

    output_path = workingFolder + "/calibresult.png"
    cv2.imwrite(output_path, dst)
    print(f"Calibrated picture saved as {output_path}")
    print("\nCalibration Matrix:")
    print(mtx)
    print("\nDistortion Coefficients:")
    print(dist)

    # Save calibration results
    filename = workingFolder + "/cameraMatrix.txt"
    np.savetxt(filename, mtx, delimiter=',')
    filename = workingFolder + "/cameraDistortion.txt"
    np.savetxt(filename, dist, delimiter=',')
    print(f"\nCalibration files saved to {workingFolder}/")

    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print(f"Mean reprojection error: {mean_error/len(objpoints):.4f} pixels")

else:
    print("In order to calibrate you need at least 9 good pictures... try again")
