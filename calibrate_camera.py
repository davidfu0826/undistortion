import glob
import argparse

import os
import cv2
import numpy as np

# Directory with all calibration images
#CALIBRATION_DIR = "test/"

# Chessboard grid size
#GRIDSIZE = (5, 8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calibrate camera by computing the camera matrix and distortion coefficients with calibration images.')
    parser.add_argument('--img-dir', type=str, required=True,
                        help='Path to directory with calibration images (OpenCV chessboards)')
    parser.add_argument('--gridsize', nargs='+', type=int, required=True,
                        help='The shape of the chessboard (e.g. 5x8, hint:search for OpenCV Chessboard)')
    args = parser.parse_args()

    CALIBRATION_DIR = args.img_dir
    if len(args.gridsize) != 2:
        print("Please enter valid --gridsize, e.g. --gridsize 5 8.")
        exit(1)
    else:
        GRIDSIZE = (args.gridsize[0], args.gridsize[1])

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, e.g. (0,0,0), (1,0,0), ..., (6,5,0)
    objp = np.zeros((GRIDSIZE[0]*GRIDSIZE[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:GRIDSIZE[0], 0:GRIDSIZE[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the calibration images.
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane

    # Get path to all
    images = list()
    for suffix in ['.png', '.jpg']:
        images += glob.glob(os.path.join(CALIBRATION_DIR, '*.png'))

    for img_path in images:
        img = cv2.imread(img_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, GRIDSIZE, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria) # Feature detection
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, GRIDSIZE, corners2, ret)
            cv2.imshow("Chessboard", img)
            cv2.waitKey(500)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    save_path = os.path.abspath("calbration_data.npz")

    np.savez('calbration_data.npz', name1=mtx, name2=dist)

    print("The distortion coefficients and camera matrix has been saved.")
    print(f"The save path: {save_path}")