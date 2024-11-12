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
    return value 
    
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
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters_create()
calibration_file = "./calibration_output.xml"
fs = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)
if not fs.isOpened() :
    print("error : cannot open calibration.xml")
    exit()
intrinsic = fs.getNode("intrinsic").mat()
distortion = fs.getNode("distortion").mat()
if intrinsic is None or distortion is None :
    print("error : cannot read camera parameters")
    
# Tello
drone = Tello()
drone.connect()#time.sleep(10)
drone.streamon()
frame_read = drone.get_frame_read()
# Tello speeds 
max_speed = 50
    
# 1. 先把 I, D 設為 0
# 2. P : 無人機會停在你設定的距離附近
# 3. I : 無人機會在設定的距離附近抖動
# 4. D : 停止抖動
x_pid   = PID(kP=0.7, kI=0.01, kD=0.1)
z_pid   = PID(kP=0.7, kI=0.01, kD=0.1)
y_pid   = PID(kP=1.5, kI=0.01, kD=0.1)
yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)

x_pid.initialize()
y_pid.initialize()
z_pid.initialize()
yaw_pid.initialize()

## Lab6
task_index = 0 # default 0

## for id1
height_counter = 0       # default 0 
up_or_down = 0           # default 0 means go up 
height_threshold = 150

## for id3
right_counter = 0        # default 0 
right_or_left = 0        # default 0 means go right 
right_threshold = 20

## midterm 
not_find_id1 = 1   # default 1
not_find_id3 = 1   # default 1
not_find_id4 = 1   # default 1
not_find_id5 = 1   # default 1
not_find_id6 = 1   # default 1

stable_counter5 = 0
stable_threshold5 = 100

stable_counter6 = 0
stable_threshold6 = 500

while (1) :
    frame = frame_read.frame
    corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    print(f"task_index : {task_index}")
    print(f"not_find_id1 : {not_find_id1}, not_find_id3 : {not_find_id3},\n not_find_id4 : {not_find_id4}, not_find_id5 : {not_find_id5}, not_find_id6 : {not_find_id6}")
    print("battery: ", drone.get_battery())
    # if ids is not None and has_foward == 1 :
    if ids is not None :
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        if (len(ids) > 0) : 
            print ("ids: ", ids) ##
        try : 
            for i in range(len(ids)):
                rotated_vectors, translation_vectors, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 15, intrinsic, distortion)
                frame = cv2.aruco.drawAxis(frame, intrinsic, distortion, rotated_vectors, translation_vectors, 15)
                c = corners[i][0]
                center_x = int(c[:, 0].mean())
                center_y = int(c[:, 1].mean())
                print("origin current id = ", ids[i])
                
                x, y, z = translation_vectors[0,0,0], translation_vectors[0,0,1], abs(translation_vectors[0,0,2])  
                print("x, y, z : ", x, y, z )
                rotation_matrix = np.zeros((3, 3))
                cv2.Rodrigues(rotated_vectors[i], rotation_matrix) # transform the rotation vector into a rotation matrix
                yaw_pitch_roll = cv2.RQDecomp3x3(rotation_matrix)[0] # yaw : vertical, pitch : horizontal, roll : perpendicular
                yaw = yaw_pitch_roll[1]                
                print("yaw : ", yaw)
                
                ## number calibration on the drone detected coordinates
                x_update = (x * 2 + 6)  # + is right, - is left
                y_update = -(y * 3)     # + is down, - is up
                z_update = (z - 50)     # + is forward, - is backward
                yaw_update = yaw * 1 - 4
                # yaw_update = angle_deg * 1 
                
                ## original Lab6 parts 
                if ids[i] == 1 and task_index == 0 :
                    not_find_id1 = 0
                    if z is None : ## slowly forward
                        for ii in range (1) :
                            drone.send_rc_control(0, 20, 0, 0)
                            time.sleep(0.4)
                    if z >= 60 : # before go right
                        print("in the first iter", z)
                        z_update += 5 # 50 - 5 cm
                        x_update = MAX(x_pid.update(x_update, sleep=0), max_speed)
                        y_update = MAX(y_pid.update(y_update, sleep=0), max_speed)
                        z_update = MAX(z_pid.update(z_update, sleep=0), max_speed)
                        yaw_update = MAX(yaw_pid.update(yaw_update, sleep=0), max_speed)
                        drone.send_rc_control(int(x_update), int (z_update), int(y_update), int(yaw_update))
                    else : # z < 60  
                        ## go right, between id1 and id2 
                        task_index += 1 
                        for ii in range (4) :
                            drone.send_rc_control(24, 0, 0, 0)
                            time.sleep(0.4)
                elif ids[i] == 2 and task_index == 1 : # forward to id2
                    if z >= 55 : 
                        z_update += 15 # 50 - 15 cm
                        x_update = MAX(x_pid.update(x_update, sleep=0), max_speed)
                        y_update = MAX(y_pid.update(y_update, sleep=0), max_speed)
                        z_update = MAX(z_pid.update(z_update, sleep=0), max_speed)
                        yaw_update = MAX(yaw_pid.update(yaw_update, sleep=0), max_speed)
                        drone.send_rc_control(int(x_update), int (z_update), int(y_update), int(yaw_update))
                    else : # z < 50
                        ## go left, between id2 and id3 
                        task_index += 1 
                        for ii in range (4) :
                            drone.send_rc_control(-30, 0, 0, 0)
                            time.sleep(0.5)
                            
                        ## midterm parts 
                        ## go forward, between id2 and id3 
                        for ii in range (4) :
                            drone.send_rc_control(0, 25, -40, 0)
                            time.sleep(0.5)
                        x_pid.initialize()
                        y_pid.initialize()
                        z_pid.initialize()
                        yaw_pid.initialize()
                ## midtem parts
                elif ids[i] == 3 and task_index == 2 : # forward to id3
                    not_find_id3 = 0 
                    if z is None : ## slowly forward 
                        for ii in range (1) :
                            drone.send_rc_control(0, 15, 0, 0)
                            time.sleep(0.4)
                    if z >= 55 : 
                        z_update += 20 # 50 - 20 cm
                        x_update = MAX(x_pid.update(x_update, sleep=0), max_speed)
                        y_update = MAX(y_pid.update(y_update, sleep=0), max_speed)
                        z_update = MAX(z_pid.update(z_update, sleep=0), max_speed)
                        yaw_update = MAX(yaw_pid.update(yaw_update, sleep=0), max_speed)
                        drone.send_rc_control(int(x_update), int (z_update), int(y_update), int(yaw_update))
                    else : # z < 50
                        ## go down, after id3 
                        task_index += 1 
                        drone.send_rc_control(0, 0, 0, 0)
                        for ii in range (6) :
                            drone.send_rc_control(0, 0, -80, 0)
                            time.sleep(0.5)
                        ## go forward, after id3 
                        drone.send_rc_control(0, 0, 0, 0)
                        for ii in range (7) :
                            drone.send_rc_control(5, 50, 0, 0)
                            time.sleep(0.5)
                        drone.send_rc_control(0, 0, 0, 0)
                elif ids[i] == 0 and task_index == 3 and not_find_id4 : # follow people
                    z_update -= 0 # 50 + 10 cm
                    x_update = MAX(x_pid.update(x_update, sleep=0), max_speed)
                    y_update = MAX(y_pid.update(y_update, sleep=0), max_speed)
                    z_update = MAX(z_pid.update(z_update, sleep=0), max_speed)
                    yaw_update = MAX(yaw_pid.update(yaw_update, sleep=0), max_speed)
                    drone.send_rc_control(int(x_update), int (z_update), int(y_update), int(yaw_update))
                elif ids[i] == 4 and task_index == 3 and not_find_id4 : # id4 is in the certain place
                    if z < 160 : 
                        not_find_id4 = 0 
                elif ids[i] == 4 and task_index == 3 and not_find_id4 == 0: # forward to id4
                    if z >= 60 : 
                        z_update += 10 # 50 - 10 cm
                        x_update = MAX(x_pid.update(x_update, sleep=0), max_speed)
                        y_update = MAX(y_pid.update(y_update, sleep=0), max_speed)
                        z_update = MAX(z_pid.update(z_update, sleep=0), max_speed)
                        yaw_update = MAX(yaw_pid.update(yaw_update, sleep=0), max_speed)
                        drone.send_rc_control(int(x_update), int (z_update), int(y_update), int(yaw_update))
                    else : # z < 60
                        ## go rotate to right
                        task_index += 1 
                        for  ii in range (7) : 
                            drone.send_rc_control(0, 0, 0, 48) # + is rotate right
                            time.sleep(0.4)
                        x_pid.initialize()
                        y_pid.initialize()
                        z_pid.initialize()
                        yaw_pid.initialize()
                elif ids[i] == 5 and task_index == 4 : # forward to id5
                    not_find_id5 = 0 
                    if z >= 50 or x >= 5 or x <= -5 or y >= 5 or y <= -5 : 
                        stable_counter5 += 1
                        z_update += 15 # 50 - 15 cm
                        x_update = MAX(x_pid.update(x_update, sleep=0), max_speed)
                        y_update = MAX(y_pid.update(y_update, sleep=0), max_speed)
                        z_update = MAX(z_pid.update(z_update, sleep=0), max_speed)
                        yaw_update = MAX(yaw_pid.update(yaw_update, sleep=0), max_speed)
                        drone.send_rc_control(int(x_update), int (z_update), int(y_update), int(yaw_update))
                    if z < 50 and stable_counter5 > stable_threshold5 : # z < 50
                        ## go left 
                        task_index += 1 
                        ## moving code is below 
                elif ids[i] == 6 and task_index == 5 : # land by id6
                    not_find_id6 = 0 
                    if z < 120 or x >= 5 or x <= -5 or y >= 5 or y <= -5 : 
                        z_update -= 70 # 50 + 70 cm
                        yaw_update += 6
                        stable_counter6 += 1
                        x_update = MAX(x_pid.update(x_update, sleep=0), max_speed)
                        y_update = MAX(y_pid.update(y_update, sleep=0), max_speed)
                        z_update = MAX(z_pid.update(z_update, sleep=0), max_speed)
                        yaw_update = MAX(yaw_pid.update(yaw_update, sleep=0), max_speed)
                        drone.send_rc_control(int(x_update), int (z_update), int(y_update), int(yaw_update))
                    if z >= 120 and stable_counter6 > stable_threshold6  : # z >= 120
                        ## landing end
                        task_index += 1 
                        for ii in range (1) :
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(0.4)
                        drone.land()
                    ### end of midterm 
                ## using PID to control the speed
                print("current id = ", ids[i])
                print("after update x, y, z") 
                print(f"x_update, y_update, z_update, yaw_update : {x_update} \t {y_update} \t {z_update} \t {yaw_update}")
                text = f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}, yaw: {yaw:.4f}"
                cv2.putText(frame, text, (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  
            ### end of id iteration
        ### end of try catch 
        except Exception as e :
            pass 
    ### end of ids is none 
    elif drone.is_flying and not_find_id1 and task_index == 0 :
        ### before find id1
        print("height_counter = ", height_counter)
        if height_counter > height_threshold : # too height
            up_or_down = 1
        elif height_counter < 0 : # too low
            up_or_down = 0 
        if up_or_down == 0 : ## up 
            height_counter += 1 
            drone.send_rc_control(0, 0, 15, 0)
            time.sleep(0.4)
        else : ## down
            height_counter -= 1 
            drone.send_rc_control(0, 0, -15, 0)
            time.sleep(0.4)
    elif drone.is_flying and not_find_id3 and task_index == 2 :
        ### before id3
        print("right_counter = ", right_counter)
        if right_counter > right_threshold : # too right
            right_or_left = 1
        elif right_counter < 0 : # too left
            right_or_left = 0 
        if right_or_left == 0 : ## go right 
            right_counter += 1 
            drone.send_rc_control(10, 0, 0, 0)
            time.sleep(0.3)
        else : ## go left
            right_counter -= 1 
            drone.send_rc_control(-15, 0, 0, 0)
            time.sleep(0.2)
    # elif drone.is_flying and not_find_id5 and task_index == 4 : 
    #     drone.send_rc_control(0, 0, 0, 25) # + is rotate right
    #     time.sleep(0.3)
    elif drone.is_flying and not_find_id6 and task_index == 5 :
        ### between id5 and id6
        drone.send_rc_control(-25, 0, 0, 0)
        time.sleep(0.3)
    else : ## if no marker is detected, ids is None
        drone.send_rc_control(0, 0, 0, 0)
    key = cv2.waitKey(1)
    
    if key != -1:
        keyboard(drone, key)
    
    cv2.imshow("frame", frame)
    if key & 0xFF == ord('q') :
        cv2.destroyAllWindows()
        break
    