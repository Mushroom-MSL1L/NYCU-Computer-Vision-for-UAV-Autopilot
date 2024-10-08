import cv2 as cv 

class MarkerDetector :
    def __init__(self, filename="calibration.xml") :
        self.dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
        # other parameters https://docs.opencv.org/3.4/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html#aca7a04c0d23b3e1c575e11af697d506c
        self.parameters = cv.aruco.DetectorParameters_create()
        self.capture = self._open_camera()
        self.intrinsic, self.distortion = self._init_camera_params(filename)

    def _open_camera (self) :
        cap = cv.VideoCapture(0)
        if not cap.isOpened() :
            print("error : cannot open camera")
            return None
        return cap
    def _init_camera_params (self, filename) :
        fs = cv.FileStorage(filename, cv.FILE_STORAGE_READ)
        if not fs.isOpened() :
            print("error : cannot open calibration.xml")
            return None
        intri = fs.getNode("intrinsic").mat()
        disto = fs.getNode("distortion").mat()
        return intri, disto
        
    def detect (self) :
        while (1) :
            retur, frame = self.capture.read()
            if not retur : 
                print("error : cannot read frame")
                break
            
            if self.intrinsic is None or self.distortion is None :
                print("error : cannot read camera parameters")
                break
            corners, ids, _ = cv.aruco.detectMarkers(frame, self.dictionary, parameters=self.parameters)
            frame = cv.aruco.drawDetectedMarkers(frame, corners, ids)
            try : 
                rotated_vectors, translation_vectors, _ = cv.aruco.estimatePoseSingleMarkers(corners, 15, self.intrinsic, self.distortion)
                frame = cv.aruco.drawAxis(frame, self.intrinsic, self.distortion, rotated_vectors, translation_vectors, 0.1)
            except Exception as e :
                pass 
            cv.imshow("frame", frame)
            if cv.waitKey(33) & 0xFF == ord('q') :
                cv.destroyAllWindows()
                break

if __name__ == "__main__" :
    calibration_file = "../Lab3/output_0/calibration_output.xml"
    md = MarkerDetector(calibration_file)
    md.detect()