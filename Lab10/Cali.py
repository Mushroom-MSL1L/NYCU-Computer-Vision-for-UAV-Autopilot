import cv2
import numpy as np
from djitellopy import Tello

checkerboard = (9, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# 總共有9*6個點，每個點有三維(x, y, z)座標
objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)

# 設置好object point (先分別x, y座標，轉置後得到(x,y)配對，最後轉為二維)
objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)


objpoints = []  # real world space
imgpoints = []  # image plane

# connect to drone
drone = Tello()
drone.connect()
drone.streamon()
frame_read = drone.get_frame_read()

while(True):
    frame = frame_read.frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corner = cv2.findChessboardCorners(gray, checkerboard, None)

    if ret:
        corner2 = cv2.cornerSubPix(gray, corner, (11, 11), (-1,-1), criteria)
        objpoints.append(objp)
        imgpoints.append(corner2)
        drawn_frame = cv2.drawChessboardCorners(frame, checkerboard, corner2, ret)

        cv2.imshow('frame', drawn_frame)
        cv2.waitKey(5)
    if len(imgpoints) > 50:
        break

cv2.destroyAllWindows()

ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, checkerboard, None, None)

# save parameters as xml file
f = cv2.FileStorage("params.xml", cv2.FILE_STORAGE_WRITE)
f.write("intrinsic", cameraMatrix)  
f.write("distortion", distCoeffs) 
f.release()