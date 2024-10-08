import cv2 as cv
import numpy as np

def self_defined_warp_perspective(M, frame, original_height, original_width):
    warpped_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)
    M_inv = np.linalg.inv(M)
    
    # M * captured_corners(camera frame) = embedded_corners(part of AD)
    # source(camera) = M_inv * original(whole AD)
    # then map the source to the original image
    for y in range(original_height):
        for x in range(original_width):
            homogeneous_coord = np.array([x, y, 1])
            source_coord = M_inv @ homogeneous_coord
            source_coord /= source_coord[2]
            src_x, src_y = source_coord[0], source_coord[1]

            # check if the source coordinates are within bounds
            # beware !!!!!!! frame, shape = [y, x] !!!!!!!
            if 0 <= src_x < frame.shape[1] and 0 <= src_y < frame.shape[0]:
                # source coordinates are not discrete 
                # using bilinear interpolation
                x0, y0 = int(src_x), int(src_y) # lower bound 
                x1, y1 = min(x0 + 1, frame.shape[1] - 1), min(y0 + 1, frame.shape[0] - 1) # upper bound 
                a, b = src_x - x0, src_y - y0 # ratio 
                pixel_value = (1 - a) * (1 - b) * frame[y0, x0] + \
                              a * (1 - b) * frame[y0, x1] + \
                              (1 - a) * b * frame[y1, x0] + \
                              a * b * frame[y1, x1]
                warpped_image[y, x] = pixel_value
    return warpped_image

def get_map_list (lux, luy, rux, ruy, rdx, rdy, ldx, ldy):
    # get minimum bounding box of embedded space
    max_x, min_x, max_y, min_y = max(lux, rux, rdx, ldx), min(lux, rux, rdx, ldx), max(luy, ruy, rdy, ldy), min(luy, ruy, rdy, ldy)
    # only take points of bounded area of embedded space from original image 
    map_list = []
    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            at_left_up    = (luy - ldy) / (lux - ldx) * (x - ldx) > y - ldy
            at_right_up   = (luy - ruy) / (lux - rux) * (x - rux) > y - ruy
            at_right_down = (ruy - rdy) / (rux - rdx) * (x - rdx) > y - rdy
            at_left_down  = (ldy - rdy) / (ldx - rdx) * (x - rdx) < y - rdy
            if at_left_up or at_right_up or at_right_down or at_left_down:
                continue
            map_list.append([y, x])
    return map_list

def get_capture_size():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: webcam not found")
        return None
    retur, frame = cap.read()
    if not retur:
        print("Error: webcam failed")
        return None
    frame_size = frame.shape[:2]
    print("frame_size : ", frame_size)
    cap.release()
    return frame_size

def wrap_2_picture(image_path, lux, luy, rux, ruy, rdx, rdy, ldx, ldy):
    frame_size = get_capture_size()
    if frame_size is None:
        print ("Error: Captured frame size not found.")
        return
    height, width = frame_size

    original_image = cv.imread(image_path)
    result_image = original_image.copy()
    if original_image is None:
        print("Error: Image not found.")
        return
    original_height, original_width = original_image.shape[:2]

    # relative positions of two sets of corners must be the same
    left_up, right_up, right_down, left_down = (lux, luy), (rux, ruy), (rdx, rdy), (ldx, ldy)
    embedded_corners = np.float32([left_up, right_up, right_down, left_down])
    captured_corners = np.float32([(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)])
    
    # M * captured_corners = embedded_corners
    M = cv.getPerspectiveTransform(captured_corners, embedded_corners)
    print("homography M : \n", M)
    
    # get pixel coordinates as mask
    y_map, x_map = zip(*get_map_list(lux, luy, rux, ruy, rdx, rdy, ldx, ldy))

    # Open webcam and show wrapped image
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: webcam not found")
        return

    while True:
        retur, frame = cap.read()
        if not retur:
            print("Error: webcam failed")
            break

        # get the transformed image (background is 0)
        # and place it on top of the original frame (frame is not colored picture)
        warpped_image = self_defined_warp_perspective(M, frame, original_height, original_width)

        # merge the original image and warped frame
        result_image[y_map, x_map] = warpped_image[y_map, x_map]

        cv.imshow('frame', result_image)
        if cv.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    # using /LAb3/pts.py to get the four corners of the embedded space
    
    lux, luy, rux, ruy, rdx, rdy, ldx, ldy = \
        415, 870, 1641, 220, 1656, 1257, 334, 1407  # Four corners of embedded space
    image_path = "src/screen.jpg"
    wrap_2_picture(image_path, lux, luy, rux, ruy, rdx, rdy, ldx, ldy)
