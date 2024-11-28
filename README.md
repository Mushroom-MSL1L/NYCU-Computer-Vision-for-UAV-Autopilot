# NYCU-Computer-Vision-for-UAV-Autopilot

# environment setup
* python=3.9
```bash
# install all the required python packages
pip install -r requirements.txt
```
* pytorch [link](https://pytorch.org/get-started/locally/)

# Labs 
- Lab1
    - Environment Setup (conda virtual environment)
    - OpenCV Basics
    - Image Processing
    - Image Filtering
    - Interpolation
    - Edge Detection

- Lab2
    - Color picture histogram equalization
    - Optimized Otsu Threshold algorithm 
    - Connected Component by Two-Pass Algorithm with disjoint set data structure

- Lab3 
    - Camera calibartion for camera intrinsic parameters 
        - Implement the Zhang's method for camera calibration
    - Wrap perspective transformation for real time camera capture
        - Camera will be mapped to the embedded space of the image
        - Utilizing homography matrix.

- Lab4
    - Marker detection
        - Show the detected markers on the image
        - Show distance between markers and the camera

- Lab5
    - Utilize PID to control the drone
        - Implement the PID controller and keyboard controller for the drone
        - PID controller will help adjust speed to reach the marker

- Lab6
    - Implement the drone autopilot
        - Go to the marker
        - Go right
        - Go to the second marker
        - Go left
        - Land

- Midterm
    - Implement the drone autopilot
        - Marker tracing 
        - Marker following
        - Auto rotation, forward, left, right, up, down
        - Auto Land

- Lab7
    - Object detection
        - Using HOG to detect pedestrian
        - Using Haar cascade to detect face
        - Solve PnP with 4 points to get the distance between camera and object