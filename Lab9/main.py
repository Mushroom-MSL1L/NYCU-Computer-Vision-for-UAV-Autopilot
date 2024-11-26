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
        # current_height = self.get_height()
        # if current_height < 120 :
        #     self.send_rc_control(0, 0, 30, 0)
        #     time.sleep(1.8)
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

def get_pattern(img, thres_rate = 0.5, pattern_size=(3, 3)):
    # thres_rate should be a value between 0 ~ 1
    # img should be binary image
    height = img.shape[0]
    width = img.shape[1]
    height_step = height // pattern_size[0]
    width_step = width // pattern_size[1]
    thres = (1 - thres_rate) * height_step * width_step
    pattern = np.zeros(pattern_size)
    for h in range(pattern_size[0]):
        for w in range(pattern_size[1]):
            pattern[h, w] = (np.sum(img[h * height_step : (h + 1) * height_step, 
                                        w * width_step : (w + 1) * width_step]) 
                             < thres)
    return pattern

def horizontal(pattern) : 
    match = np.array_equal(pattern, np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]))
    return match
def vertical(pattern) : 
    match = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    return np.array_equal(pattern, match)
def left_up(pattern) : 
    match = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]])
    return np.array_equal(pattern, match)
def right_down(pattern) :
    match = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]])
    return np.array_equal(pattern, match)
def left_down(pattern) :
    match = np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]])
    return np.array_equal(pattern, match)
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
x_pid   = PID(kP=0.72, kI=0.0001, kD=0.1)
z_pid   = PID(kP=0.7, kI=0.0001, kD=0.1)
y_pid   = PID(kP=0.7, kI=0.0001, kD=0.1)
yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)

x_pid.initialize()
y_pid.initialize()
z_pid.initialize()
yaw_pid.initialize()

task_index = -1     # default -1
find_id1 = 0        # default 0
threshold = 100     # range 0~255, for 黑白圖片
thres_rate = 0.2    # range 0~1, for 九宮格
go_back_id1 = 0     # default 0

move_speed = 20 
sleep_time = 0.2

## for id1
height_counter = 0       # default 0 
up_or_down = 0           # default 0 means go up 
height_threshold = 150

while (1) :
    frame = frame_read.frame
    corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    print(f"task_index : {task_index}, find_id1 : {find_id1}")
    print("battery: ", drone.get_battery())
    
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    dilation = cv2.dilate(binary, np.ones((3,3), np.uint8), iterations = 1) # 加粗
    # erosion = cv2.erode(image, kernel, iterations = 1) # 變細
    cv2.imshow("dilation", dilation)
    cv2.namedWindow("dilation", 0)
    cv2.resizeWindow("dilation", 600, 600)

    
    binary01 = dilation // 255
    pattern = get_pattern(binary01, thres_rate)
    text = "pattern: " + str(pattern)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    
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
                y_update = -y     # + is down, - is up
                z_update = (z - 50) + 15    # + is forward, - is backward
                yaw_update = yaw * 1 - 10
                # yaw_update = angle_deg * 1 
                
                ## original Lab6 parts 
                if ids[i] == 1 and task_index == -1 :
                    find_id1 = 1
                    if z is None : ## slowly forward
                        for ii in range (1) :
                            drone.send_rc_control(0, 20, 0, 0)
                            time.sleep(0.4)
                    if z >= 35 and -10 < y and y < 10 : # before go right for line
                        z_update += 15 # 50 - 15 cm = 35
                        x_update = MAX(x_pid.update(x_update, sleep=0), max_speed)
                        y_update = MAX(y_pid.update(y_update, sleep=0), max_speed)
                        z_update = MAX(z_pid.update(z_update, sleep=0), max_speed)
                        yaw_update = MAX(yaw_pid.update(yaw_update, sleep=0), max_speed)
                        drone.send_rc_control(int(x_update), int (z_update / 2), int(y_update / 2), int(yaw_update))
                    else : # z < 30  
                        ## go right for line
                        task_index += 1 
                        drone.send_rc_control(0, 0, 0, 0)
                if ids[i] == 1 and go_back_id1 == 1 :
                    ## lane before code
                    drone.land()
                    ### end of lab9 trace 
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
    elif drone.is_flying and find_id1 == 0 :
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
    elif drone.is_flying and find_id1 == 1 :
        if task_index == 0 :
            task_index += 1
        elif task_index == 1 : 
            # go right
            drone.send_rc_control(move_speed, 0, 0, 0) 
            time.sleep(sleep_time)

        if task_index == 1 and left_up(pattern) :
            task_index += 1
        elif task_index == 2 : 
            # go up
            drone.send_rc_control(0, move_speed, 0, 0)
            time.sleep(sleep_time)
            
        if task_index == 2 and right_down(pattern) :
            task_index += 1
        elif task_index == 3 : 
            # go right
            drone.send_rc_control(move_speed, 0, 0, 0) 
            time.sleep(sleep_time)
            
        if task_index == 3 and left_up(pattern) :
            task_index += 1
        elif task_index == 4 : 
            # go up
            drone.send_rc_control(0, move_speed, 0, 0) 
            time.sleep(sleep_time)
            
        if task_index == 4 and left_down(pattern) :
            task_index += 1
        elif task_index == 5 : 
            # go left
            drone.send_rc_control(-move_speed, 0, 0, 0) 
            time.sleep(sleep_time)

        if task_index == 5 and right_down(pattern) :
            task_index += 1
        elif task_index == 6 : 
            # go down
            drone.send_rc_control(0, -move_speed, 0, 0) 
            time.sleep(sleep_time)
            go_back_id1 = 1
    ## end of trace line
    else : ## if no marker is detected, ids is None
        drone.send_rc_control(0, 0, 0, 0)
    key = cv2.waitKey(1)
    
    if key != -1:
        keyboard(drone, key)
    
    cv2.namedWindow("frame", 0)
    cv2.resizeWindow("frame", 600, 600)
    cv2.imshow("frame", frame)
    if key & 0xFF == ord('q') :
        cv2.destroyAllWindows()
        break
    