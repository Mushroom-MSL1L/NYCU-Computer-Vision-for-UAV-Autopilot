import cv2
import numpy as np
import time
from djitellopy import Tello

# Initialize the Tello drone
drone = Tello()
drone.connect()
drone.streamon()

# Define chessboard size and prepare object points
chessboard_size = (9, 6)  # Chessboard pattern size (number of inner corners)
square_size = 1  # Size of each square in real-world units (meters or millimeters)

# Prepare object points based on real-world size of the chessboard
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all images
objpoints = []  # 3d points in real-world space
imgpoints = []  # 2d points in image plane

# Number of images needed for calibration
num_images_needed = 20

def capture_calibration_images():
    while len(objpoints) < num_images_needed:
        # Capture frame from drone
        frame = drone.get_frame_read().frame
        h, w = frame.shape[:2]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray_frame, chessboard_size, None)

        # If found, refine the corners and add them to the image points
        if ret:
            print(f"Chessboard detected - {len(objpoints)+1}/{num_images_needed}")
            corners_refined = cv2.cornerSubPix(
                gray_frame, corners, (11, 11), (-1, -1), 
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            objpoints.append(objp.copy())
            imgpoints.append(corners_refined)

            # Draw and display the corners
            drawn_frame = cv2.drawChessboardCorners(frame, chessboard_size, corners_refined, ret)
            cv2.imshow("Drone Calibration - Chessboard Detected", drawn_frame)
            cv2.waitKey(500)  # Wait for half a second

        # Show the current frame (whether chessboard is detected or not)
        cv2.imshow("Drone View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Calibration aborted.")
            break

    cv2.destroyAllWindows()

def perform_camera_calibration(h, w):
    # Perform camera calibration using collected points
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    
    # Check if calibration was successful
    if ret:
        print("Calibration successful!")
        print(f"Camera Matrix:\n{cameraMatrix}")
        print(f"Distortion Coefficients:\n{distCoeffs}")
        
        # Save the calibration results to a file
        save_calibration(cameraMatrix, distCoeffs)
    else:
        print("Calibration failed.")

def save_calibration(cameraMatrix, distCoeffs):
    # Save the calibration results to an XML file
    file = cv2.FileStorage("calibration_results.xml", cv2.FILE_STORAGE_WRITE)
    file.write("intrinsic", cameraMatrix)
    file.write("distortion", distCoeffs)
    file.release()
    print("Calibration data saved to 'calibration_results.xml'")

def main():
    try:
        # Step 1: Capture chessboard images from the drone for calibration
        print("Starting calibration image capture...")
        capture_calibration_images()

        # Step 2: Perform calibration
        frame = drone.get_frame_read().frame
        h, w = frame.shape[:2]
        print("Performing camera calibration...")
        perform_camera_calibration(h, w)

    finally:
        # Clean up and release resources
        drone.streamoff()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()