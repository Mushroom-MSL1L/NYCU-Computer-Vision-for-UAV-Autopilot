import cv2
import numpy as np
import time
import math
import torch
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard, is_flying, is_start

from torchvision import transforms
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, scale_coords
from utils.plots import plot_one_box

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters_create()
z_base = np.array([[0],[0],[1]])

def thres(x, max_speed_threshold = 40):
    if x > max_speed_threshold:
        x = max_speed_threshold
    elif x < -max_speed_threshold:
        x = -max_speed_threshold
    return x

def calibrate():
    fs = cv2.FileStorage("params.xml", cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("intrinsic").mat()
    distortion = fs.getNode("distortion").mat()
    fs.release()
    return intrinsic, distortion

def battery_dis(drone):
    battery = drone.get_battery()
    print("\n!! Now battery: {}\n".format(battery))

def distinguish_doll(frame):
def load_model():
    WEIGHT = './best.pt'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = attempt_load(WEIGHT, map_location=device)
    if device == "cuda":
        model = model.half().to(device)
    else:
        model = model.float().to(device)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
def main():
    load_model()
    

    face1 = False
    face2 = False
    line_finish = False
    finish4 = False
    land = False

    drone = Tello()
    drone.connect()
    drone.streamon()
    intrinsic, distortion = calibrate()
    
    battery_dis(drone)
    x_pid = PID(kP=0.72, kI=0.0001, kD=0.1)  # Use tvec_x (tvec[i,0,0]) ----> control left and right
    z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)  # Use tvec_z (tvec[i,0,2])----> control forward and backward
    y_pid = PID(kP=0.7, kI=0.0001, kD=0.1)  # Use tvec_y (tvec[i,0,1])----> control upward and downward
    yaw_pid = PID(kP=0.7,kI=0.0001, kD=0.1)
    
    x_pid.initialize()
    z_pid.initialize()
    y_pid.initialize()
    yaw_pid.initialize()

    while True:
        frame_read = drone.get_frame_read()
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        x_update = 0
        y_update = 0
        z_update = 0
        yaw_update = 0
        angle_diff = 0
        if markerIds is not None:
            # pose estimation for single markers
            frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)
            print("marker not None!!\n")

            for i in range(rvec.shape[0]):
                PID_state = {}
                PID_state["pid_x"] = ''
                PID_state["pid_y"] = ''
                PID_state["pid_z"] = ''
                PID_state["pid_yaw"] = ''
                current_marker_id = markerIds[i][0]
                frame = cv2.aruco.drawAxis(frame, intrinsic, distortion, rvec[i], tvec[i], 10)

                # Step 1-1: 偵測人臉1 ===============================================================
                if current_marker_id == 1:
                    print("marker is 1\n")
                    rvec_3x3,_ = cv2.Rodrigues(rvec[i])
                    rvec_zbase = rvec_3x3.dot(np.array([[0],[0],[1]]))
                    rx_project = rvec_zbase[0]
                    rz_project = rvec_zbase[2]
                    angle_diff= math.atan2(float(rz_project), float(rx_project))*180/math.pi + 90 

                    # update yaw, x, y, z 經過調整誤差後的無人機位置距離(無人機的實際位置數據)
                    yaw_update = (-1)* (angle_diff + 15) 
                    x_update = tvec[i,0,0] - 10 * (tvec[i,0,2]/100)
                    y_update = (tvec[i,0,1] - 10) * (-1) 
                    z_update = tvec[i,0,2] - 75
                    if z_update <= 10 :
                        print("對到人臉1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                        face1 = True
                        drone.send_rc_control(0, 0, 25, 0) # 向上
                        time.sleep(2)
                        drone.send_rc_control(0, 25, 0, 0) # 向前
                        time.sleep(3)

                    x_update = thres(x_pid.update(x_update, sleep=0))
                    y_update = thres(y_pid.update(y_update, sleep=0))
                    z_update = thres(z_pid.update(z_update, sleep=0))
                    yaw_update = thres(yaw_pid.update(yaw_update, sleep=0))
                    PID_state["pid_x"] = str(x_update)
                    PID_state["pid_y"] = str(y_update)
                    PID_state["pid_z"] = str(z_update)
                    PID_state["pid_yaw"] = str(yaw_update)
                
                #Step 1-2: 偵測人臉2 =================================================================
                elif current_marker_id == 2 and face1 :
                    print("marker is 2\n")
                    rvec_3x3,_ = cv2.Rodrigues(rvec[i])
                    rvec_zbase = rvec_3x3.dot(np.array([[0],[0],[1]]))
                    rx_project = rvec_zbase[0]
                    rz_project = rvec_zbase[2]
                    angle_diff= math.atan2(float(rz_project), float(rx_project))*180/math.pi + 90 
    
                    # update yaw, x, y, z 經過調整誤差後的無人機位置距離(無人機的實際位置數據)
                    yaw_update = (-1)* (angle_diff + 15) 
                    x_update = tvec[i,0,0] - 10 * (tvec[i,0,2]/100)
                    y_update = (tvec[i,0,1] - 10) * (-1) 
                    z_update = tvec[i,0,2] - 70
                    if z_update <= 10:
                        print("偵測到人臉2!!!!!!!!!!!!!!!!!!!!!!!\n")
                        face2 = True
                        drone.send_rc_control(0, 0, -25, 0) # 向上
                        time.sleep(2)
                        drone.send_rc_control(0, 25, 0, 0) # 向前
                        time.sleep(4)

                    x_update = thres(x_pid.update(x_update, sleep=0))
                    y_update = thres(y_pid.update(y_update, sleep=0))
                    z_update = thres(z_pid.update(z_update, sleep=0))
                    yaw_update = thres(yaw_pid.update(yaw_update, sleep=0))
                    PID_state["pid_x"] = str(x_update)
                    PID_state["pid_y"] = str(y_update)
                    PID_state["pid_z"] = str(z_update)
                    PID_state["pid_yaw"] = str(yaw_update)
                
                # 偵測 carna 還是 melody
                
                #Step 2-1: 偵測到carna =================================================================
                elif current_marker_id == 2 and face2 and doll_label = "carna": 
                    print("Step 2 is CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n")
                    # 開始追線
                    line_finish = True

                #Step 2-2: 偵測到melody =================================================================
                elif current_marker_id == 2 and face2 and doll_label = "melody": 
                    print("Step 2 is MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM\n")
                    # 開始追線
                    line_finish = True
                
                # Step 4: 追線完成 =================================================================                 
                elif line_finish and current_marker_id == 3: 
                    print("追線完成！正在對準marker3!!!!!!!!!!!!!!!!!!!\n")
                    rvec_3x3,_ = cv2.Rodrigues(rvec[i])
                    rvec_zbase = rvec_3x3.dot(np.array([[0],[0],[1]]))
                    rx_project = rvec_zbase[0]
                    rz_project = rvec_zbase[2]
                    angle_diff= math.atan2(float(rz_project), float(rx_project))*180/math.pi + 90 
    
                    # update yaw, x, y, z 經過調整誤差後的無人機位置距離(無人機的實際位置數據)
                    yaw_update = (-1)* (angle_diff + 15) 
                    x_update = tvec[i,0,0] - 10 * (tvec[i,0,2]/100)
                    y_update = (tvec[i,0,1] - 10) * (-1) 
                    z_update = tvec[i,0,2] - 60
                    if z_update <= 15:
                        print("marker_id 3對準了！準備轉彎!!!!!!!!!!!!!!!!!!!!!!!\n\n\n\n")
                        drone.send_rc_control(0, 0, 0, -180) # 向左180度
                        time.sleep(1)
                        drone.send_rc_control(0, 30, 0, 0) # 向前
                        time.sleep(4)
                        finish4 = True

                    x_update = thres(x_pid.update(x_update, sleep=0))
                    y_update = thres(y_pid.update(y_update, sleep=0))
                    z_update = thres(z_pid.update(z_update, sleep=0))
                    yaw_update = thres(yaw_pid.update(yaw_update, sleep=0))
                    PID_state["pid_x"] = str(x_update)
                    PID_state["pid_y"] = str(y_update)
                    PID_state["pid_z"] = str(z_update)
                    PID_state["pid_yaw"] = str(yaw_update)
                
                # Step 5-1: 偵測 carna 或 melody =================================================================                 
                elif finish4 and doll_label = "carna" :
                    print("step 5_0 : CCCCCCCCCCCCCCCCCCCCCC!!!!!!!!!!!\n")
                    drone.send_rc_control(-25, 0, 0, 0) # 向左
                    time.sleep(2)
                    drone.send_rc_control(0, 25, 0, 0) # 向前
                    time.sleep(1)
                    land = True
                
                    x_update = thres(x_pid.update(x_update, sleep=0))
                    y_update = thres(y_pid.update(y_update, sleep=0))
                    z_update = thres(z_pid.update(z_update, sleep=0))
                    yaw_update = thres(yaw_pid.update(yaw_update, sleep=0))
                    PID_state["pid_x"] = str(x_update)
                    PID_state["pid_y"] = str(y_update)
                    PID_state["pid_z"] = str(z_update)
                    PID_state["pid_yaw"] = str(yaw_update)
                              
                elif finish4 and doll_label = "melody" :
                    print("step 5_1 : MMMMMMMMMMMMMMMMMMMMMM!!!!!!!!!!!\n")
                    drone.send_rc_control(25, 0, 0, 0) # 向右
                    time.sleep(2)
                    drone.send_rc_control(0, 25, 0, 0) # 向前
                    time.sleep(1)
                    land = True
                
                    x_update = thres(x_pid.update(x_update, sleep=0))
                    y_update = thres(y_pid.update(y_update, sleep=0))
                    z_update = thres(z_pid.update(z_update, sleep=0))
                    yaw_update = thres(yaw_pid.update(yaw_update, sleep=0))
                    PID_state["pid_x"] = str(x_update)
                    PID_state["pid_y"] = str(y_update)
                    PID_state["pid_z"] = str(z_update)
                    PID_state["pid_yaw"] = str(yaw_update)

                # Step 5-2: 偵測 marker id 並降落 =================================================================                 
                elif land and current_marker_id == 5:
                    print("---------------------marker is 5 cccccccccccccc---------------------\n\n\n\n")
                    rvec_3x3,_ = cv2.Rodrigues(rvec[i])
                    rvec_zbase = rvec_3x3.dot(np.array([[0],[0],[1]]))
                    rx_project = rvec_zbase[0]
                    rz_project = rvec_zbase[2]
                    angle_diff= math.atan2(float(rz_project), float(rx_project))*180/math.pi + 90 
    
                    # update yaw, x, y, z 經過調整誤差後的無人機位置距離(無人機的實際位置數據)
                    yaw_update = (-1)* (angle_diff + 15) 
                    x_update = tvec[i,0,0] - 10 * (tvec[i,0,2]/100) + 30
                    y_update = (tvec[i,0,1] - 10) * (-1) 
                    z_update = tvec[i,0,2] - 165
                    if abs(z_update) <= 10 and abs(x_update) <= 10:
                        time.sleep(2)
                        if abs(z_update) <= 10 and abs(x_update) <= 10:
                            print("---------------準備降落!!!!!!!!!!!!!!!!!!!!!!!-------------\n\n\n\n")
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.land()
                            drone.send_rc_control(0, 0, 0, 0)
    
                    x_update = thres(x_pid.update(x_update, sleep=0))
                    y_update = thres(y_pid.update(y_update, sleep=0))
                    z_update = thres(z_pid.update(z_update, sleep=0))
                    yaw_update = thres(yaw_pid.update(yaw_update, sleep=0))
                    PID_state["pid_x"] = str(x_update)
                    PID_state["pid_y"] = str(y_update)
                    PID_state["pid_z"] = str(z_update)
                    PID_state["pid_yaw"] = str(yaw_update)


                drone.send_rc_control(int(x_update//1), int(z_update//2), int(y_update//1), int(yaw_update//1))
                # display x, y, z coordinates on the frame
                # 無人機鏡頭偵測到的位置距離
                cv2.putText(frame, "x = {:.3f}  y = {:.3f}  z = {:.3f}  yaw = {:.3f}".format(tvec[i][0][0], tvec[i][0][1], tvec[i][0][2], angle_diff), (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                # 經過PID控制器調整後的位置
                cv2.putText(frame, "x = {:.3f}  y = {:.3f}  z = {:.3f}  yaw = {:.3f}".format(x_update, y_update,z_update,yaw_update), (10, 128), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                print("--------------------------------------------")
                print("MarkerIDs: {}".format(markerIds))
                print("tvec: {}||{}||{}||{}".format(tvec[i,0,0], tvec[i,0,1], tvec[i,0,2], angle_diff))
                print("PID: {}||{}||{}||{}".format(PID_state["pid_x"],PID_state["pid_y"],PID_state["pid_z"],PID_state["pid_yaw"]))
                print("--------------------------------------------")
                
        else:
            if is_flying == True:
                drone.send_rc_control(0, 0, 0, 0)
                print("no command")
        
        cv2.imshow("drone", frame)
        key = cv2.waitKey(1)

        if key != -1:
            keyboard(drone, key)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            drone.send_rc_control(0, 0, 0, 0)
            break
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()