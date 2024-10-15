import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID

def MAX (value, max_value) :
    if value > max_value : 
        return max_value
    elif value < -max_value :
        return -max_value
    
def keyboard(self, key):
    #global is_flying
    print("key:", key)
    fb_speed = 40
    lf_speed = 40
    ud_speed = 50
    degree = 30
    if key == ord('1'):
        self.takeoff()
        #is_flying = True
    if key == ord('2'):
        self.land()
        #is_flying = False
    if key == ord('3'):
        self.send_rc_control(0, 0, 0, 0)
        print("stop!!!!")
    if key == ord('w'):
        self.send_rc_control(0, fb_speed, 0, 0)
        print("forward!!!!")
    if key == ord('s'):
        self.send_rc_control(0, (-1) * fb_speed, 0, 0)
        print("backward!!!!")
    if key == ord('a'):
        self.send_rc_control((-1) * lf_speed, 0, 0, 0)
        print("left!!!!")
    if key == ord('d'):
        self.send_rc_control(lf_speed, 0, 0, 0)
        print("right!!!!")
    if key == ord('z'):
        self.send_rc_control(0, 0, ud_speed, 0)
        print("down!!!!")
    if key == ord('x'):
        self.send_rc_control(0, 0, (-1) *ud_speed, 0)
        print("up!!!!")
    if key == ord('c'):
        self.send_rc_control(0, 0, 0, degree)
        print("rotate!!!!")
    if key == ord('v'):
        self.send_rc_control(0, 0, 0, (-1) *degree)
        print("counter rotate!!!!")
    if key == ord('5'):
        height = self.get_height()
        print(height)
    if key == ord('6'):
        battery = self.get_battery()
        print (battery)

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
    drone.connect()#time.sleep(10)
    drone.streamon()
    frame_read = drone.get_frame_read()
    # Tello speeds 
    max_speed = 50
        
    x_pid   = PID(kP=0.7, kI=0.0001, kD=0.1)
    z_pid   = PID(kP=0.7, kI=0.0001, kD=0.1)
    y_pid   = PID(kP=0.7, kI=0.0001, kD=0.1)
    yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    
    yaw_pid.initialize()
    z_pid.initialize()
    y_pid.initialize()
    x_pid.initialize()

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
                
                rotation_matrix = np.zeros((3, 3))
                rotation_matrix = cv2.Rodrigues(rotated_vectors[i][0]) # transform the rotation vector into a rotation matrix
                z_axis = np.array([0, 0, 1])
                z_prime = rotation_matrix @ z_axis
                
                z_prime_x = z_prime[0][0]
                z_prime_z = z_prime[2][0]
                angle_rad = math.atan2(z_prime_z, z_prime_x) # v_vector = np.array([z_prime_x, z_prime_z])
                angle_deg = math.degrees(angle_rad)
                
                # update speed PID
                x_update = x - 0
                x_update = MAX(x_pid.update(x_update, sleep=0), max_speed)
                y_update = y - 0 
                y_update = MAX(y_pid.update(y_update, sleep=0), max_speed)
                z_update = z - 0
                z_update = MAX(z_pid.update(z_update, sleep=0), max_speed)
                yaw_update = angle_deg * 1 
                yaw_update = MAX(yaw_pid.update(yaw_update, sleep=0), max_speed)
                
                drone.send_rc_control(0, int (z_update // 2), int (y_update), int (yaw_update))
                print(x_update, y_update, z_update, yaw_update)
            text = f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}"
            cv2.putText(frame, text, (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  
            if ids is None :
                drone.send_rc_control(0, 0, 0, 0)
        except Exception as e :
            pass 
        
        key = cv2.waitKey(1)
        
        if key != -1:
            keyboard(drone, key)
        
        cv2.imshow("frame", frame)
        if key & 0xFF == ord('q') :
            cv2.destroyAllWindows()
            break
    
if __name__ == "__main__" : 
    main() 