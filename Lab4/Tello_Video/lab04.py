import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID

"""
test if the drone is linked to the computer
"""
def main():
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    calibration_file = "./calibration_output.xml"
    fs = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)
    if not fs.isOpened() :
        print("error : cannot open calibration.xml")
        return None
    intrinsic = fs.getNode("intrinsic").mat()
    distortion = fs.getNode("distortion").mat()
    
    # Tello
    drone = Tello()
    drone.connect()
    #time.sleep(10)
    drone.streamon()
    frame_read = drone.get_frame_read()

    while (1) :
        frame = frame_read.frame
        
        if intrinsic is None or distortion is None :
            print("error : cannot read camera parameters")
            break
        corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        try : 
            rotated_vectors, translation_vectors, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 15, intrinsic, distortion)
            frame = cv2.aruco.drawAxis(frame, intrinsic, distortion, rotated_vectors, translation_vectors, 15)
            for i in range(len(ids)):
                c = corners[i][0]
                center_x = int(c[:, 0].mean())
                center_y = int(c[:, 1].mean())
                
                t_vec = translation_vectors[i][0]  
                x, y, z = t_vec[0], t_vec[1], t_vec[2]
                text = f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}"
                cv2.putText(frame, text, (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                

        except Exception as e :
            pass 
        
        cv2.imshow("frame", frame)
        if cv2.waitKey(33) & 0xFF == ord('q') :
            cv2.destroyAllWindows()
            break
    
    #cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

