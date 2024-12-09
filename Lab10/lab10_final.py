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
import random 

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

def detect_face(frame, face_objectPoints, intrinsic, distortion):
    face_cascade = cv2.CascadeClassifier('Haarcascade_Frontal_Face.xml')
    ScaleFactor = 1.1
    minNeighbers = 20
    minSize = (30, 30)
    estimated_distance = -1 

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    face_rects = face_cascade.detectMultiScale(
        frame,
        ScaleFactor,    #每次搜尋方塊減少的比例
        minNeighbers,   #每個目標至少檢測到幾次以上，才可被認定是真數據。
        0,
        minSize         #設定數據搜尋的最小尺寸 ，如 minSize=(40,40)
    )
            
    for (x, y, w, h) in face_rects:
        face_imagePoints = np.array([
            # left-top, right-top, right-bottom, left-bottom
            [x, y], [x+w, y], [x+w, y+h], [x, y+h]
        ], dtype=np.float32)
        _, _, tvec = cv2.solvePnP(
            face_objectPoints,
            face_imagePoints,
            intrinsic, 
            distortion
        )
        estimated_distance = tvec[2]
        if estimated_distance > 200 :
            continue
        cv2.rectangle(
            frame, 
            (int(x),   int(y)),      # upper left
            (int(x+w), int(y+h)),    # lower right
            color=(0, 255, 0),
            thickness=2
        )
        text = "Face distance: " + str(np.round(estimated_distance, 4)) + " cm"
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return frame, estimated_distance

def distinguish_doll(image, device, model, names, colors):
    image_orig = image.copy()
    image = letterbox(image, (640, 640), stride=64, auto=True)[0]
    if device == "cuda":
        image = transforms.ToTensor()(image).to(device).half().unsqueeze(0)
    else:
        image = transforms.ToTensor()(image).to(device).float().unsqueeze(0)
    with torch.no_grad():
        output = model(image)[0]
    output = non_max_suppression_kpt(output, 0.25, 0.65)[0]
    
    ## Draw label and confidence on the image
    output[:, :4] = scale_coords(image.shape[2:], output[:, :4], image_orig.shape).round()
    for *xyxy, conf, cls in output:
        label = f'{names[int(cls)]} {conf:.2f}'
        return names[int(cls)], conf, output 
    
    output[:, :4] = scale_coords(image.shape[2:], output[:, :4], image_orig.shape).round()
    for *xyxy, conf, cls in output:
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, image_orig, label=label, color=colors[int(cls)], line_thickness=1)
        print(label)

    cv2.imshow("Detected", image_orig)
    cv2.waitKey(1)

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
    # load_model()
    

    face1 = False
    face2 = False
    line_finish = False
    finish4 = False
    land = False

    drone = Tello()
    drone.connect()
    drone.streamon()
    frame_read = drone.get_frame_read()
    intrinsic, distortion = calibrate()
    face_width = 15     # cm of real face width
    face_height = 18    # cm of real face height
    face_objectPoints = np.array([
        # left-top, right-top, right-bottom, left-bottom
        [0, 0, 0], [face_width, 0, 0], [face_width, face_height, 0], [0, face_height, 0]
    ], dtype=np.float32)
    
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
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        frame, face_distance = detect_face(frame, face_objectPoints, intrinsic, distortion)
        x_update = 0
        y_update = 0
        z_update = 0
        yaw_update = 0
        angle_diff = 0
        
        # if markerIds is not None:
        #     # pose estimation for single markers
        #     frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        #     rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)
        #     print("marker not None!!\n")

        #     for i in range(rvec.shape[0]):
        #         PID_state = {}
        #         PID_state["pid_x"] = ''
        #         PID_state["pid_y"] = ''
        #         PID_state["pid_z"] = ''
        #         PID_state["pid_yaw"] = ''
        #         current_marker_id = markerIds[i][0]
        #         frame = cv2.aruco.drawAxis(frame, intrinsic, distortion, rvec[i], tvec[i], 10)

                
        #         # 偵測 carna 還是 melody
                
        #         #Step 2-1: 偵測到carna =================================================================
        #         if current_marker_id == 2 and face2 and doll_label == "carna": 
        #             print("Step 2 is CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n")
        #             # 開始追線
        #             line_finish = True

        #         #Step 2-2: 偵測到melody =================================================================
        #         elif current_marker_id == 2 and face2 and doll_label == "melody": 
        #             print("Step 2 is MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM\n")
        #             # 開始追線
        #             line_finish = True
                
        #         # Step 4: 追線完成 =================================================================                 
        #         elif line_finish and current_marker_id == 3: 
        #             print("追線完成！正在對準marker3!!!!!!!!!!!!!!!!!!!\n")
        #             rvec_3x3,_ = cv2.Rodrigues(rvec[i])
        #             rvec_zbase = rvec_3x3.dot(np.array([[0],[0],[1]]))
        #             rx_project = rvec_zbase[0]
        #             rz_project = rvec_zbase[2]
        #             angle_diff= math.atan2(float(rz_project), float(rx_project))*180/math.pi + 90 
    
        #             # update yaw, x, y, z 經過調整誤差後的無人機位置距離(無人機的實際位置數據)
        #             yaw_update = (-1)* (angle_diff + 15) 
        #             x_update = tvec[i,0,0] - 10 * (tvec[i,0,2]/100)
        #             y_update = (tvec[i,0,1] - 10) * (-1) 
        #             z_update = tvec[i,0,2] - 60
        #             if z_update <= 15:
        #                 print("marker_id 3對準了！準備轉彎!!!!!!!!!!!!!!!!!!!!!!!\n\n\n\n")
        #                 drone.send_rc_control(0, 0, 0, -180) # 向左180度
        #                 time.sleep(1)
        #                 drone.send_rc_control(0, 30, 0, 0) # 向前
        #                 time.sleep(4)
        #                 finish4 = True

        #             x_update = thres(x_pid.update(x_update, sleep=0))
        #             y_update = thres(y_pid.update(y_update, sleep=0))
        #             z_update = thres(z_pid.update(z_update, sleep=0))
        #             yaw_update = thres(yaw_pid.update(yaw_update, sleep=0))
        #             PID_state["pid_x"] = str(x_update)
        #             PID_state["pid_y"] = str(y_update)
        #             PID_state["pid_z"] = str(z_update)
        #             PID_state["pid_yaw"] = str(yaw_update)
                
        #         # Step 5-1: 偵測 carna 或 melody =================================================================                 
        #         elif finish4 and doll_label = "carna" :
        #             print("step 5_0 : CCCCCCCCCCCCCCCCCCCCCC!!!!!!!!!!!\n")
        #             drone.send_rc_control(-25, 0, 0, 0) # 向左
        #             time.sleep(2)
        #             drone.send_rc_control(0, 25, 0, 0) # 向前
        #             time.sleep(1)
        #             land = True
                
        #             x_update = thres(x_pid.update(x_update, sleep=0))
        #             y_update = thres(y_pid.update(y_update, sleep=0))
        #             z_update = thres(z_pid.update(z_update, sleep=0))
        #             yaw_update = thres(yaw_pid.update(yaw_update, sleep=0))
        #             PID_state["pid_x"] = str(x_update)
        #             PID_state["pid_y"] = str(y_update)
        #             PID_state["pid_z"] = str(z_update)
        #             PID_state["pid_yaw"] = str(yaw_update)
                              
        #         elif finish4 and doll_label = "melody" :
        #             print("step 5_1 : MMMMMMMMMMMMMMMMMMMMMM!!!!!!!!!!!\n")
        #             drone.send_rc_control(25, 0, 0, 0) # 向右
        #             time.sleep(2)
        #             drone.send_rc_control(0, 25, 0, 0) # 向前
        #             time.sleep(1)
        #             land = True
                
        #             x_update = thres(x_pid.update(x_update, sleep=0))
        #             y_update = thres(y_pid.update(y_update, sleep=0))
        #             z_update = thres(z_pid.update(z_update, sleep=0))
        #             yaw_update = thres(yaw_pid.update(yaw_update, sleep=0))
        #             PID_state["pid_x"] = str(x_update)
        #             PID_state["pid_y"] = str(y_update)
        #             PID_state["pid_z"] = str(z_update)
        #             PID_state["pid_yaw"] = str(yaw_update)

        #         # Step 5-2: 偵測 marker id 並降落 =================================================================                 
        #         elif land and current_marker_id == 5:
        #             print("---------------------marker is 5 cccccccccccccc---------------------\n\n\n\n")
        #             rvec_3x3,_ = cv2.Rodrigues(rvec[i])
        #             rvec_zbase = rvec_3x3.dot(np.array([[0],[0],[1]]))
        #             rx_project = rvec_zbase[0]
        #             rz_project = rvec_zbase[2]
        #             angle_diff= math.atan2(float(rz_project), float(rx_project))*180/math.pi + 90 
    
        #             # update yaw, x, y, z 經過調整誤差後的無人機位置距離(無人機的實際位置數據)
        #             yaw_update = (-1)* (angle_diff + 15) 
        #             x_update = tvec[i,0,0] - 10 * (tvec[i,0,2]/100) + 30
        #             y_update = (tvec[i,0,1] - 10) * (-1) 
        #             z_update = tvec[i,0,2] - 165
        #             if abs(z_update) <= 10 and abs(x_update) <= 10:
        #                 time.sleep(2)
        #                 if abs(z_update) <= 10 and abs(x_update) <= 10:
        #                     print("---------------準備降落!!!!!!!!!!!!!!!!!!!!!!!-------------\n\n\n\n")
        #                     drone.send_rc_control(0, 0, 0, 0)
        #                     time.sleep(1)
        #                     drone.land()
        #                     drone.send_rc_control(0, 0, 0, 0)
    
        #             x_update = thres(x_pid.update(x_update, sleep=0))
        #             y_update = thres(y_pid.update(y_update, sleep=0))
        #             z_update = thres(z_pid.update(z_update, sleep=0))
        #             yaw_update = thres(yaw_pid.update(yaw_update, sleep=0))
        #             PID_state["pid_x"] = str(x_update)
        #             PID_state["pid_y"] = str(y_update)
        #             PID_state["pid_z"] = str(z_update)
        #             PID_state["pid_yaw"] = str(yaw_update)


        #         drone.send_rc_control(int(x_update//1), int(z_update//2), int(y_update//1), int(yaw_update//1))
        #         # display x, y, z coordinates on the frame
        #         # 無人機鏡頭偵測到的位置距離
        #         cv2.putText(frame, "x = {:.3f}  y = {:.3f}  z = {:.3f}  yaw = {:.3f}".format(tvec[i][0][0], tvec[i][0][1], tvec[i][0][2], angle_diff), (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        #         # 經過PID控制器調整後的位置
        #         cv2.putText(frame, "x = {:.3f}  y = {:.3f}  z = {:.3f}  yaw = {:.3f}".format(x_update, y_update,z_update,yaw_update), (10, 128), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
        #         print("--------------------------------------------")
        #         print("MarkerIDs: {}".format(markerIds))
        #         print("tvec: {}||{}||{}||{}".format(tvec[i,0,0], tvec[i,0,1], tvec[i,0,2], angle_diff))
        #         print("PID: {}||{}||{}||{}".format(PID_state["pid_x"],PID_state["pid_y"],PID_state["pid_z"],PID_state["pid_yaw"]))
        #         print("--------------------------------------------")

        # Step 1-1: 偵測人臉1 ===============================================================
        if not face1 and drone.is_flying :
            face1_distance = 50
            if face_distance > 0 and face_distance < face1_distance :
                print("偵測到人臉1!!!!!!!!!!!!!!!!!!!!!!!\n")
                face1 = True
                ## up 
                drone.send_rc_control(0, 0, 50, 0)
                time.sleep(2)
                ## forward 
                drone.send_rc_control(0, 0, 50, 0)
                time.sleep(0.5)
                ## down
                drone.send_rc_control(0, 0, -50, 0)
                time.sleep(2)
            elif face_distance >= face1_distance:
                print("人臉距離太遠1，前進!!!!!!!!!!!!!!!!!!!!!!!\n")
                drone.send_rc_control(0, 25, 0, 0)
                time.sleep(0.5)
            elif face_distance <= 0:
                print("沒看到人臉1，向上!!!!!!!!!!!!!!!!!!!!!!!\n")
                drone.send_rc_control(0, 0, 30, 0)
                time.sleep(0.5)
        #Step 1-2: 偵測人臉2 =================================================================
        elif face1 and not face2 and drone.is_flying :
            face2_distance = 50
            if face_distance > 0 and face_distance < face2_distance :
                print("偵測到人臉2!!!!!!!!!!!!!!!!!!!!!!!\n")
                face2 = True
                ## down
                drone.send_rc_control(0, 0, -50, 0)
                time.sleep(1)
                ## forward 
                drone.send_rc_control(0, 0, 50, 0)
                time.sleep(0.5)
                ## up
                drone.send_rc_control(0, 0, 50, 0)
                time.sleep(1)
            elif face_distance >= face2_distance:
                print("人臉2距離太遠，前進!!!!!!!!!!!!!!!!!!!!!!!\n")
                drone.send_rc_control(0, 25, 0, 0)
                time.sleep(0.5)
            elif face_distance <= 0:
                print("沒看到人臉2，向下!!!!!!!!!!!!!!!!!!!!!!!\n")
                drone.send_rc_control(0, 0, -50, 0)
                time.sleep(0.5)

        else:
            if drone.is_flying == True:
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