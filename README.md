# About this repo
Undistort any vidoe (.mp4) and save the results! This repo uses OpenCV and removes the radial- and tangential-distortion with OpenCV ([Read more here](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html)).

## Prerequisites
1. You need to record images of a **chessboard**, taken with the camera you are calibrating. For instance, see this [link](https://markhedleyjones.com/projects/calibration-checkerboard-collection).
2. Put all the images in the same directory.
3. Python 3 with OpenCV, Numpy are required.

## Usage
In the terminal:
- To compute the camera matrix and distortion coefficients:
  - `python3 calibrate_camera.py --img-dir <path to calibration images> --gridsize <two integers specifying grid size>`
  - Example: `python3 calibrate_camera.py --img-dir calibration/images --gridsize 5 8`

- To undistort a video (.mp4)
  - `python3 undistort.py --source <path to mp4 video>`
  - Example: `python3 undistort.py --source test_video.mp4`
