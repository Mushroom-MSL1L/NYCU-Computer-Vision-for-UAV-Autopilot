import cv2 as cv 
import numpy as np
import math

"""
aborted now
"""

# def biLinear_interpolation(original_image, ratio=3):
#     old_row, old_col, channels = original_image.shape
#     new_row = old_row * ratio
#     new_col = old_col * ratio
#     row_p = old_row-1
#     col_p = old_col-1
#     new_row_ratio = new_row/row_p
#     new_col_ratio = new_col/col_p
#     new_image = np.zeros((new_row, new_col, channels), dtype=np.float32)

#     # print("original_shape before bilinear_interpolation : ", original_image.shape)
#     # print("need to wait about 20 seconds for ratio=3 ...")
#     for row in range(new_row):
#         for col in range(new_col):
#             x_i = int(row//new_row_ratio)
#             y_i = int(col//new_col_ratio)
#             x = row/new_row_ratio
#             y = col/new_col_ratio
#             x_rate = x - x_i
#             y_rate = y - y_i
#             x_i1 = x_i+1
#             y_i1 = y_i+1
#             if x_rate == 0 and y_rate == 0:
#                 new_image[row][col] = original_image[x_i][y_i]
#             elif x_rate == 0:
#                 new_image[row][col] = ((y - y_i) / (y_i1 - y_i)) * original_image[x_i][y_i1] + ((y_i1 - y) / (y_i1 - y_i)) * original_image[x_i][y_i]
#             elif y_rate == 0:
#                 new_image[row][col] = ((x - x_i) / (x_i1 - x_i)) * original_image[x_i1][y_i] + ((x_i1 - x) / (x_i1 - x_i)) * original_image[x_i][y_i]
#             else:
#                 temp_x1 = ((x - x_i) / (x_i1 - x_i)) * original_image[x_i1][y_i] + ((x_i1 - x) / (x_i1 - x_i)) * original_image[x_i][y_i]
#                 temp_x2 = ((x - x_i) / (x_i1 - x_i)) * original_image[x_i1][y_i1] + ((x_i1 - x) / (x_i1 - x_i)) * original_image[x_i][y_i1]
#                 new_image[row][col] = ((y - y_i) / (y_i1 - y_i)) * temp_x2 + ((y_i1 - y) / (y_i1 - y_i)) * temp_x1
#     return new_image

def biLinear_interpolation(img,dstH,dstW):
    scrH, scrW = img.shape[:2]
    img = np.pad(img,((0,1),(0,1),(0,0)),'edge')
    #print(img.shape)
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=(i+1)*(scrH/dstH)-1
            scry=(j+1)*(scrW/dstW)-1
            x=math.floor(scrx)
            y=math.floor(scry)
            u=scrx-x
            v=scry-y
            retimg[i,j]=(1-u)*(1-v)*img[x,y]+u*(1-v)*img[x+1,y]+(1-u)*v*img[x,y+1]+u*v*img[x+1,y+1]
    
    return retimg

def self_defined_warp_perspective (M, frame, height, width):
    warpped_image = np.zeros((height, width, 3), dtype=np.uint8)
    frame_y, frame_x = frame.shape[:2]
    # apply M to each pixel of frame
    for y in range(frame_y):
        for x in range(frame_x):
            projected_y = int((M[1,0]*x+M[1,1]*y+M[1,2])/(M[2,0]*x+M[2,1]*y+M[2,2]))
            projected_x = int((M[0,0]*x+M[0,1]*y+M[0,2])/(M[2,0]*x+M[2,1]*y+M[2,2]))
            # warpped_image[projected_y, projected_x, :] = frame[y, x, :]
            if 0 <= projected_x < width and 0 <= projected_y < height:
                warpped_image[projected_y, projected_x, :] = frame[y, x, :]
    size_ratio = 1
    optimized_warpped_image = cv.resize(warpped_image, (math.floor(frame_x*size_ratio), math.floor(frame_y*size_ratio)), interpolation=cv.INTER_NEAREST)
    # optimized_warpped_image = biLinear_interpolation(warpped_image, math.floor(height*size_ratio), math.floor(width*size_ratio))
    # optimized_warpped_image = biLinear_interpolation(warpped_image, ratio=size_ratio)
    return optimized_warpped_image

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

def get_capture_size ():
    cap = cv.VideoCapture(0)
    frame_size = 0
    if not cap.isOpened():
        print("Error: webcam not found")
        return
    retur, frame = cap.read()
    if not retur : 
        print("Error: webcam failed")
    frame_size = frame.shape[:2]
    print("frame_size : ", frame_size)
    cap.release()
    return frame_size

def wrap_2_picture (image_path, lux, luy, rux, ruy, rdx, rdy, ldx, ldy):
    frame_size = get_capture_size()
    height, width = frame_size
    # get the four corners of embedded space
    left_up, right_up, right_down, left_down = (lux, luy), (rux, ruy), (rdx, rdy), (ldx, ldy)
    embedded_corners = np.float32([left_up, right_up, right_down, left_down])

    # get the four corners of the original image
    original_image = cv.imread(image_path)
    print("original_image shape : ", original_image.shape)
    new_image = original_image.copy()
    captured_corners = np.float32([(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)])

    # embeded = M * captured
    M = cv.getPerspectiveTransform(captured_corners, embedded_corners)
    print("M : \n",M)
    # only take points of bounded area of embedded space from original image 
    map_list = get_map_list(lux, luy, rux, ruy, rdx, rdy, ldx, ldy)

    # open webcam and show wrapped image
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: webcam not found")
        return
    while (1) :
        retur, frame = cap.read()
        if not retur : 
            print("Error: webcam failed")
            break
        wrapped_image = self_defined_warp_perspective(M, frame, height, width)
        for y, x in map_list:
            new_image[y, x] = wrapped_image[y, x]
        cv.imshow('frame', wrapped_image)
        if cv.waitKey(33) & 0xFF == ord('q') :
            cap.release()
            break

if __name__ == '__main__' :
    lux, luy, rux, ruy, rdx, rdy, ldx, ldy = \
        415, 870, 1641, 220, 1656, 1257, 334, 1407 # four corners of embedded space
    image_path = "src/screen.jpg"
    wrap_2_picture(image_path, lux, luy, rux, ruy, rdx, rdy, ldx, ldy)