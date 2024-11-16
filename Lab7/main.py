import cv2
import NMS
import numpy as np

"""
• 利用 HOG 行人檢測及 Haar-cascade 臉部偵測框出 人 ( 25%) 與 人臉 (25%)
• 利用任一方法算出與其的距離
• Demo 時為即時影像並用尺量 人 (25%) 與 人臉 (25%) 距離準確度
• Demo 誤差：人 (50cm)、人臉 (10cm)

• 不限定方法
1. 已知物體大小及相機焦距, 用物體在畫面中占的pixel計算
物件的框會有留白, 可以自行判斷要乘多少比例才是物體實際pixel大小
2. 假設人或人臉為平面, 已知大小解SolvePnP
• cv2.solvePnP(objp, imgPoints, intrinsic, distortion)→ retval, rvec, tvec
    objp的部分要用真實的長度單位, 非(0,0), (0,1), (1,0), (1,1)
"""

def init_camera_params (filename) :
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    if not fs.isOpened() :
        print("error : cannot open calibration.xml")
        return None
    intri = fs.getNode("intrinsic").mat()
    disto = fs.getNode("distortion").mat()
    return intri, disto


def detect_face_and_people(filename, face_objectPoints, people_objectPoints):
    # get the camera
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: webcam failed 1 ")
        
    # get the camera parameters
    intrinsic, distortion = init_camera_params(filename)

    # initialize the HOG descriptor/person detector
    # https://pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    winStride = (4,4) # n*n pixel 
    scale = 1.5 # usually 1.05-1.5

    # https://hackmd.io/@yillkid/ByQ7ySDT8/%2F%40DsnbUcX9Tyi-Vn5MNyBM7g%2FH1eMbtjZD
    face_cascade = cv2.CascadeClassifier('Haarcascade_Frontal_Face.xml')
    ScaleFactor = 1.1
    minNeighbers = 20
    minSize = (30, 30)

    while True:
        try:
            retur, frame = cap.read()
            if not retur:
                print("Error: webcam failed 3 ")
                raise Exception
        except:
            print("Error: webcam failed 2 ")
            input("Please connect the camera and press Enter")
            continue
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        pedestrian_rects, weights = hog.detectMultiScale( 
            blurred_frame,              #輸入圖
            winStride=winStride,        #在圖上抓取特徵時窗口的移動大小
            scale=scale,                #抓取不同scale (越小就要做越多次)
            useMeanshiftGrouping=False
        )
        ### If there're multiple boxes for the same person, use NMS to remove the redundant boxes. 
        # rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in pedestrian_rects])
        # rects = NMS.non_max_suppression_fast(rects, 0.65)
        # for (rect, weight) in zip(rects, weights):
        #     x, y, x2, y2 = rect
        #     if weight > 0.5:
        #         cv2.rectangle(
        #             frame, 
        #             (int(x),   int(y)),       # left-top
        #             (int(x2), int(y2)),       # right-bottom
        #             color=(0, 255, 255),
        #             thickness=4
        #         )
        for (rect, weight) in zip(pedestrian_rects, weights):
            x, y, w, h = rect
            if weight > 0.8:
                cv2.rectangle(
                    frame, 
                    (int(x),   int(y)),       # left-top
                    (int(x+w), int(y+h)),     # right-bottom
                    color=(0, 0, 255),
                    thickness=4
                )
            people_imagePoints = np.array([
                # left-top, right-top, right-bottom, left-bottom
                [x, y], [x+w, y], [x+w, y+h], [x, y+h]
            ], dtype=np.float32)
            # https://mapostech.com/ros-opencv-solvepnp/
            _, _, tvec = cv2.solvePnP(
                people_objectPoints,
                people_imagePoints,
                intrinsic,
                distortion
            )
            estimated_distance = tvec[2]
            text = "People distance: " + str(np.round(estimated_distance * 0.01, 4)) + " m"
            cv2.putText(frame, text, (0,150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8, cv2.LINE_AA)
        
        
        face_rects = face_cascade.detectMultiScale(
            frame,
            ScaleFactor,    #每次搜尋方塊減少的比例
            minNeighbers,   #每個目標至少檢測到幾次以上，才可被認定是真數據。
            0,
            minSize         #設定數據搜尋的最小尺寸 ，如 minSize=(40,40)
        )
        
        for (x, y, w, h) in face_rects:
            cv2.rectangle(
                frame, 
                (int(x),   int(y)),        # upper left
                (int(x+w), int(y+h)),    # lower right
                color=(0, 255, 0),
                thickness=2
            )
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
            text = "Face distance: " + str(np.round(estimated_distance, 4)) + " cm"
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_width = 15 # cm of face width
    face_objectPoints = np.array([
        # left-top, right-top, right-bottom, left-bottom
        [0, 0, 0], [face_width, 0, 0], [face_width, face_width, 0], [0, face_width, 0]
    ], dtype=np.float32)
    body_width = 60 * 1.8   # cm of body width,     real width * ratio of box
    body_height = 170 * 1.2 # cm of body height     real height * ratio of box
    people_objectPoints = np.array([
        # left-top, right-top, right-bottom, left-bottom
        [0, 0, 0], [body_width, 0, 0], [body_width, body_height, 0], [0, body_height, 0]
    ], dtype=np.float32)
    calib_file = 'calibration_output.xml'
    detect_face_and_people(calib_file, face_objectPoints, people_objectPoints)