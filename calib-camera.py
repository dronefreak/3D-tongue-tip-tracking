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
import argparse

#---------------------- DEFAULT PARAMETERS
DEFAULT_N_ROWS = 8
DEFAULT_N_COLS = 8
DEFAULT_DIMENSION = 20  # mm
DEFAULT_WORKING_FOLDER = "./camera_01"
DEFAULT_IMAGE_TYPE = 'jpg'
MIN_IMAGES_REQUIRED = 9
MIN_PATTERNS_REQUIRED = 9
#------------------------------------------

def sanitize_path(path):
    """Sanitize file path to prevent directory traversal"""
    # Remove any '..' patterns that could lead to directory traversal
    path = os.path.normpath(path)
    # Ensure path doesn't contain potentially dangerous characters
    if '..' in path or path.startswith('/') or ':\\' in path:
        raise ValueError("Invalid path provided")
    return path

def validate_inputs(folder, img_type, rows, cols, dim):
    """Validate input parameters"""
    # Validate folder path
    try:
        folder = sanitize_path(folder)
    except ValueError as e:
        print(f"Invalid folder path: {e}")
        sys.exit(1)
    
    # Validate image type
    valid_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    if img_type.lower() not in valid_extensions:
        print(f"Invalid image type: {img_type}. Valid types: {valid_extensions}")
        sys.exit(1)
    
    # Validate numeric values
    if rows <= 0 or cols <= 0:
        print("Number of rows and columns must be positive integers")
        sys.exit(1)
    
    if dim <= 0:
        print("Cell dimension must be a positive number")
        sys.exit(1)
    
    return folder, img_type, rows, cols, dim

def initialize_arg_parser():
    """Initialize and return argument parser with proper help messages"""
    parser = argparse.ArgumentParser(
        description="Camera calibration using chessboard patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calib-camera.py ./images jpg 8 8 20
  python calib-camera.py -h
        """
    )
    parser.add_argument("folder", nargs="?", default=DEFAULT_WORKING_FOLDER,
                        help=f"Folder containing calibration images (default: {DEFAULT_WORKING_FOLDER})")
    parser.add_argument("image_type", nargs="?", default=DEFAULT_IMAGE_TYPE,
                        help=f"Image file extension (default: {DEFAULT_IMAGE_TYPE})")
    parser.add_argument("rows", nargs="?", type=int, default=DEFAULT_N_ROWS,
                        help=f"Number of internal corner rows in the chessboard (default: {DEFAULT_N_ROWS})")
    parser.add_argument("cols", nargs="?", type=int, default=DEFAULT_N_COLS,
                        help=f"Number of internal corner columns in the chessboard (default: {DEFAULT_N_COLS})")
    parser.add_argument("dimension", nargs="?", type=float, default=DEFAULT_DIMENSION,
                        help=f"Size of each square in the chessboard in mm (default: {DEFAULT_DIMENSION})")
    return parser

# Initialize with defaults
nRows = DEFAULT_N_ROWS
nCols = DEFAULT_N_COLS
dimension = DEFAULT_DIMENSION
workingFolder = DEFAULT_WORKING_FOLDER
imageType = DEFAULT_IMAGE_TYPE

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nRows*nCols,3), np.float32)
objp[:,:2] = np.mgrid[0:nCols,0:nRows].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Parse command line arguments using argparse
parser = initialize_arg_parser()
args = parser.parse_args()

# Use argparse values if provided
workingFolder = args.folder
imageType = args.image_type
nRows = args.rows
nCols = args.cols
dimension = args.dimension

# Validate all inputs
try:
    workingFolder, imageType, nRows, nCols, dimension = validate_inputs(
        workingFolder, imageType, nRows, nCols, dimension
    )
    
    # Update termination criteria and object points with validated values
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)
    objp = np.zeros((nRows*nCols,3), np.float32)
    objp[:,:2] = np.mgrid[0:nCols,0:nRows].T.reshape(-1,2)
    
except ValueError as e:
    print(f"Invalid input: {e}")
    sys.exit(1)

# Since argparse handles help, we don't need the manual help section
# The script will automatically exit if -h is provided

# Find the images files
filename = os.path.join(workingFolder, "*." + imageType)
images = glob.glob(filename)

print(f"Found {len(images)} images")
if len(images) < MIN_IMAGES_REQUIRED:
    print(f"Not enough images were found: at least {MIN_IMAGES_REQUIRED} shall be provided!!!")
    sys.exit(1)


else:
    nPatternFound = 0
    imgNotGood = None  # Initialize to None instead of images[1]

    for fname in images:
        if 'calibresult' in fname:
            continue
            
        try:
            #-- Read the file and convert in greyscale
            img = cv2.imread(fname)
            if img is None:
                print(f"Warning: Could not read image {fname}, skipping...")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            print(f"Reading image {fname}")

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (nCols,nRows), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                print("Pattern found! Press ESC to skip or ENTER to accept")
                #--- Sometimes, Harris corners fails with crappy pictures, so
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nCols,nRows), corners2, ret)
                cv2.imshow('img', img)
                # cv2.waitKey(0)
                k = cv2.waitKey(0) & 0xFF
                if k == 27:  #-- ESC Button
                    print("Image Skipped")
                    imgNotGood = fname
                    continue

                print("Image accepted")
                nPatternFound += 1
                objpoints.append(objp)
                imgpoints.append(corners2)

                # cv2.waitKey(0)
            else:
                imgNotGood = fname
                print(f"No pattern found in {fname}")
                
        except cv2.error as e:
            print(f"OpenCV error processing {fname}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue

cv2.destroyAllWindows()

if (nPatternFound >= MIN_PATTERNS_REQUIRED):
    print(f"Found {nPatternFound} good images")
    if len(imgpoints) == 0 or len(objpoints) == 0:
        print("No valid image points or object points found!")
        sys.exit(1)
        
    # Get image dimensions from the first valid image
    first_img = cv2.imread(images[0])
    if first_img is None:
        print("Could not read any valid image to get dimensions")
        sys.exit(1)
    h, w = first_img.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w,h), None, None)

    # Undistort an image - use the last processed image if available
    if imgNotGood:
        img = cv2.imread(imgNotGood)
        if img is not None:
            h, w = img.shape[:2]
            print(f"Image to undistort: {imgNotGood}")
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

            # undistort
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

            # crop the image
            if roi:
                x, y, w_roi, h_roi = roi
                dst = dst[y:y+h_roi, x:x+w_roi]
                print(f"ROI: {x} {y} {w_roi} {h_roi}")

            cv2.imwrite(os.path.join(workingFolder, "calibresult.png"), dst)
            print("Calibrated picture saved as calibresult.png")
            print("Calibration Matrix: ")
            print(mtx)
            print("Distortion: ", dist)

            #--------- Save result
            camera_matrix_path = os.path.join(workingFolder, "cameraMatrix.txt")
            np.savetxt(camera_matrix_path, mtx, delimiter=',')
            camera_distortion_path = os.path.join(workingFolder, "cameraDistortion.txt")
            np.savetxt(camera_distortion_path, dist, delimiter=',')

            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                mean_error += error

            print("total error: ", mean_error/len(objpoints))
        else:
            print(f"Could not read the image for undistortion: {imgNotGood}")
    else:
        print("No image available for undistortion")
        
else:
    print(f"In order to calibrate you need at least {MIN_PATTERNS_REQUIRED} good pictures... try again")





