import glob
import argparse

import os
import cv2
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Undistort any media with OpenCV (Only video supported currently).')
    parser.add_argument('--calibration-data', type=str, default='calbration_data.npz',
                        help='Path to calibration data (.npz)')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to the file to undistort.')
    args = parser.parse_args()

    # Load calibration data
    data = np.load(args.calibration_data)
    mtx = data['name1']
    dist = data['name2']
    
    # Instantiate VideoCapture object and read metadata
    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CV_CAP_PROP_FPS)
    width  = cap.get(cv2.CV_CAP_PROP_FRAME_WIDTH)  
    height = cap.get(cv2.CV_CAP_PROP_FRAME_HEIGHT) 

    # Instantiate VideoWriter object for writing to .mp4
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width,height))

    # Obtain new camera matrix from calibration data
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width,height), 1, (width,height))

    # Start undistortion of video
    count = 0
    while(True):

        # Read next frame
        ret, frame = cap.read()

        # Stop if no more frames
        if not ret:
            break

        # Undistort frame, display frame and write the frame
        if frame is not None:
            frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
            cv2.imshow('frame', frame)
            c = cv2.waitKey(1)
            out.write(frame)
        print("\r", count, end="")
        count += 1

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
