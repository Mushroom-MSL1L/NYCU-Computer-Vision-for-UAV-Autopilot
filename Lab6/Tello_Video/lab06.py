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
x_pid   = PID(kP=2, kI=0.1, kD=0.1)
z_pid   = PID(kP=1.5, kI=0.01, kD=0.1)
y_pid   = PID(kP=1.5, kI=0.01, kD=0.1)
yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)

x_pid.initialize()
y_pid.initialize()
z_pid.initialize()
yaw_pid.initialize()

## Lab6
start_time = 0
wait_time = 0 
task_index= 0 
timing = 0 

## midterm 
crowl_flag = 0 

while (1) :
    frame = frame_read.frame
    
    corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    if ids is not None :
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        if (len(ids) > 0) : 
            print ("ids: ", ids) ##
        try : 
            rotated_vectors, translation_vectors, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 15, intrinsic, distortion)
            frame = cv2.aruco.drawAxis(frame, intrinsic, distortion, rotated_vectors, translation_vectors, 15)
            for i in range(len(ids)):
                c = corners[i][0]
                center_x = int(c[:, 0].mean())
                center_y = int(c[:, 1].mean())
                
                x, y, z = translation_vectors[0,0,0], translation_vectors[0,0,1], abs(translation_vectors[0,0,2])          
                print("x, y, z : ", x, y, z )
                # rotation_matrix = np.zeros((3, 3))
                # rotation_matrix = cv2.Rodrigues(rotated_vectors[i][0]) # transform the rotation vector into a rotation matrix
                # z_axis = np.array([0, 0, 1]) # old z-axis
                # z_prime = rotation_matrix @ z_axis
                
                # z_prime_x = z_prime[0][0] # new z map to x-axis
                # z_prime_z = z_prime[2][0] # new z map to z-axis
                # angle_rad = math.atan2(z_prime_z, z_prime_x) # v_vector = np.array([z_prime_x, z_prime_z])
                # angle_deg = math.degrees(angle_rad)
                
                rotation_matrix = np.zeros((3, 3))
                cv2.Rodrigues(rotated_vectors[i], rotation_matrix) # transform the rotation vector into a rotation matrix
                yaw_pitch_roll = cv2.RQDecomp3x3(rotation_matrix)[0] # yaw : vertical, pitch : horizontal, roll : perpendicular
                yaw = yaw_pitch_roll[1]                
                print("yaw : ", yaw)
                
                ## number calibration on the drone detected coordinates
                x_update = (x * 2 + 6)  # + is right, - is left
                y_update = -(y * 3)     # + is down, - is up
                z_update = (z - 50)     # + is forward, - is backward
                yaw_update = yaw * 1
                # yaw_update = angle_deg * 1 
                
                print(f"task_index : {task_index}")
                ## Lab6 parts 
                if ids[i] == 1 and task_index == 0 :
                    if z >= 60 and z != None: # before go right
                        print("in the first iter", z)
                        z_update -= 10 # 50 - 10 cm
                        x_update = MAX(x_pid.update(x_update, sleep=0), max_speed)
                        y_update = MAX(y_pid.update(y_update, sleep=0), max_speed)
                        z_update = MAX(z_pid.update(z_update, sleep=0), max_speed)
                        yaw_update = MAX(yaw_pid.update(yaw_update, sleep=0), max_speed)
                        drone.send_rc_control(int(x_update), int (z_update), int(y_update), int(yaw_update))
                    elif z < 60 : # z < 60
                        task_index += 1 
                    else:
                        print("in fuction 1 z is not detected.")
                        continue
                elif task_index == 1 : # go right, between id1 and id2 
                    x_speed = 40 # 40 cm
                    x_update = MAX(x_pid.update(x_update, sleep=0), max_speed)
                    y_update = MAX(y_pid.update(y_update, sleep=0), max_speed)
                    z_update = MAX(z_pid.update(z_update, sleep=0), max_speed)
                    yaw_update = MAX(yaw_pid.update(yaw_update, sleep=0), max_speed)
                    drone.send_rc_control(int(x_speed), int (z_update), int(y_update), int(yaw_update))
                    # time.sleep( 2 )
                    if ids[i] == 2 :
                        task_index += 1 
                elif ids[i] == 2 and task_index == 2 : # forward to id2
                    if z >= 60 : 
                        z_update -= 10 # 50 - 10 cm
                        x_update = MAX(x_pid.update(x_update, sleep=0), max_speed)
                        y_update = MAX(y_pid.update(y_update, sleep=0), max_speed)
                        z_update = MAX(z_pid.update(z_update, sleep=0), max_speed)
                        yaw_update = MAX(yaw_pid.update(yaw_update, sleep=0), max_speed)
                        drone.send_rc_control(int(x_update), int (z_update), int(y_update), int(yaw_update))
                    elif z < 60: # z < 60
                        task_index += 1 
                    else:
                        continue
                elif task_index == 3 : # go left, between id2 and id3
                    x_speed = -40 # 40 cm
                    x_update = MAX(x_pid.update(x_update, sleep=0), max_speed)
                    y_update = MAX(y_pid.update(y_update, sleep=0), max_speed)
                    z_update = MAX(z_pid.update(z_update, sleep=0), max_speed)
                    yaw_update = MAX(yaw_pid.update(yaw_update, sleep=0), max_speed)
                    drone.send_rc_control(int(x_speed), int (z_update), int(y_update), int(yaw_update))

                    # time.sleep( 3 ) # wait for fly to the left
                    task_index+= 1 
                elif task_index == 4 : # stop and landing 
                    drone.land()
                    break
                
                ## midtem parts
                # if left_flag == 1 : # between id2 and id3
                #     x_speed = 40 # 40 cm
                #     drone.send_rc_control(int(x_speed), 0 ,0 ,0)
                
                # if ids[i] == 3 and crowl_flag == 0 :
                #     left_flag = 0 
                #     if z >= 50 : # before go down
                #         z_update += 20 # 50 - 20 cm
                #     else : # z < 50
                #         start_time = time.time()
                #         wait_time = 3
                #         crowl_flag = 1
                
                # if (crowl_flag == 1) and (time.time() - start_time >= wait_time): # after id3
                #     y_speed = 40 # 40 cm
                #     z_speed = 20 # 20 cm
                #     yaw_speed = 10
                #     drone.send_rc_control(0, int(z_speed), int(y_speed), int(yaw_speed))
                    
                # if (crowl_flag == 1) and (time.time() - start_time >= wait_time): # before orbit
                    
                
                ## using PID to control the speed

                print("after update x, y, z") 
                print(f"x_update, y_update, z_update, yaw_update : {x_update} \t {y_update} \t {z_update} \t {yaw_update}")
                text = f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}"
                cv2.putText(frame, text, (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  
        except Exception as e :
            pass 
    else : ## if no marker is detected, ids is None
        drone.send_rc_control(0, 0, 0, 0)
    key = cv2.waitKey(1)
    
    if key != -1:
        keyboard(drone, key)
    
    cv2.imshow("frame", frame)
    if key & 0xFF == ord('q') :
        cv2.destroyAllWindows()
        break