import cv2 as cv 
import numpy as np 
import os

def get_points (frame, pattern_size):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    retur, corners = cv.findChessboardCorners(frame, pattern_size, None)
    if not retur:
        return False, None
    precised_corners = cv.cornerSubPix(
        image=frame,
        corners=corners, 
        winSize=(11, 11), 
        zeroZone=(-1, -1), # detect all 
        criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    )
    return True, precised_corners

def calibrate_and_store (shape, image_points, object_points, dist, output_file, save_parameters):  
    retur, intrinsic_matrix, distortion_coefficient, rotation_matrix, translation_matrix = cv.calibrateCamera(
        objectPoints=object_points,
        imagePoints=image_points,
        imageSize=(shape[1], shape[0]),
        cameraMatrix=None,
        distCoeffs=None
    )
    if not retur:
        print("Error: calibration failed")
        print("Warning: please recapture the images")
        image_points = []
        object_points = []
        return False
    print("calibration successful")
    print("\tintrinsic_matrix: \n", intrinsic_matrix)
    print("\tdistortion_coefficient: \n", distortion_coefficient)
    print("\trotation_matrix: \n", rotation_matrix)
    print("\ttranslation_matrix: \n", translation_matrix)
    if save_parameters:
        os.makedirs(dist, exist_ok=True)
        file = cv.FileStorage(output_file, cv.FILE_STORAGE_WRITE)
        file.write("intrinsic", intrinsic_matrix)
        file.write("distortion", distortion_coefficient)
        file.release()
    return True

def camera_calibration (picture_numbers, dist, output_file, save_parameters, import_images, export_images):
    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        print("Error: webcam not found")
        return
    image_points = []
    object_points = []
    pattern_size = (9, 6)
    grid_size = 1 
    object_point = np.zeros((pattern_size[0] * pattern_size[1], 3) , np.float32)
    object_point[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    object_point = object_point * grid_size
    shape = cap.read()[1].shape
    
    while (1):
        # import images
        if import_images:
            for file in os.listdir(dist):
                if not file.endswith(".jpg"):
                    continue
                frame = cv.imread(dist + "/" + file)
                retur, points = get_points(frame, pattern_size)
                if not retur:
                    print("Error: chessboard not found")
                    continue
                image_points.append(points.copy())
                object_points.append(object_point.copy())
                print("captured image: ", len(image_points))
            
        # Capture frame-by-frame
        retur, frame = cap.read()
        if not retur:
            print("Error: webcam failed")
            break
        cv.imshow('frame', frame) 
        
        # calculate and store the parameters
        if len(image_points) > picture_numbers :
            if calibrate_and_store(shape, image_points, object_points, dist, output_file, save_parameters):
                break
            image_points = []
            object_points = []
        
        # take a picture on 'Enter' key
        if cv.waitKey(33) & 0xFF == 13:
            retur, points = get_points(frame, pattern_size)
            if not retur:
                print("Error: chessboard not found")
                continue
            image_points.append(points.copy())
            object_points.append(object_point.copy())
            if export_images:
                os.makedirs(dist, exist_ok=True)
                cv.imwrite(dist + "/image_" + str(len(image_points)) + ".jpg", frame)
            print("captured image: ", len(image_points))
        
        # Exit on 'q' key
        if cv.waitKey(33) & 0xFF == ord('q'): 
            # 30 fps == 33 ms delay between frames
            break

if __name__ == '__main__':
    picture_numbers = 20
    output_file = 'output/calibration_output.xml'
    dist = 'output'
    save_parameters = True
    import_images = False
    export_images = True
    camera_calibration(
        picture_numbers=picture_numbers,
        dist = dist, 
        output_file = output_file, 
        save_parameters=save_parameters,
        import_images=import_images, 
        export_images=export_images
    )