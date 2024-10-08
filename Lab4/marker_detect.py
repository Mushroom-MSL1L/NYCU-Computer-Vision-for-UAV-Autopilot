import cv2 as cv 

class MarkerDetector :
    def __init__(self, filename="calibration.xml") :
        self.dictionary = cv.aruco.Dictionary_get(cv.aryco.DICT_6X6_250)
        # other parameters https://docs.opencv.org/3.4/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html#aca7a04c0d23b3e1c575e11af697d506c
        self.parameters = cv.aruce.DetectorParameters_create()
        self.intrinsic, self.distortion = self.get_camera_params(filename)
        self.capture = self.open_camera()

    def open_camera (self) :
        cap = cv.VideoCapture(0)
        if not cap.isOpened() :
            print("error : cannot open camera")
            return None
        return cap
    def get_camera_params (self, filename="calibration.xml") :
        if self.intrinsic is not None and self.distortion is not None :
            return self.intrinsic, self.distortion
        fs = cv.FileStorage(filename, cv.FILE_STORAGE_READ)
        if not fs.isOpened() :
            print("error : cannot open calibration.xml")
            return None
        intrinsic = fs.getNode("intrinsic").mat()
        distortion = fs.getNode("distortion").mat()
        return intrinsic, distortion
        
    def detect (self) :
        while (1) :
            retur, frame = self.capture.read()
            if not retur : 
                print("error : cannot read frame")
                break
            corners, ids, _ = cv.aruco.detectMarkers(frame, self.dictionary, parameters=self.parameters)
            frame = cv.aruco.drawDetectedMarkers(frame, corners, ids)
            rotated_vectors, translation_vectors, object_points = cv.aruco.estimatePoseSingleMarkers(corners, 15, self.intrinsic, self.distortion)
            frame = cv.aruco.drawAxis(frame, self.intrinsic, self.distortion, rotated_vectors, translation_vectors, 0.1)
            cv.imshow("frame", frame)
            if cv.waitKey(33) & 0xFF == ord('q') :
                cv.destroyAllWindows()
                break
if __name__ == "__main__" :
    calibration_file = "../Lab3/output_0/calibration_output.xml"
    md = MarkerDetector(calibration_file)
