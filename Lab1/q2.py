import cv2 as cv
import os 
import numpy as np

""" 
Interpolation - 雙線性內插法 (40%)
根據輸出影像的像素位置，找到輸入影像中最鄰近的四個點 ,再利用雙線性內插法求出輸出影像的像素強度。
將照片放大3倍
以參數方式輸入影像以及倍率
● 學會使用 OpenCV API (10%) 自行實作雙線性內插法 (40%)
"""

def exercise_2(input_file_path, output_file_path, directory, ratio=3, show_image=False, save_image=True):
    print("exercise_2")
    original_image = cv.imread(input_file_path)

    old_row, old_col, channels = original_image.shape
    new_row = old_row * ratio
    new_col = old_col * ratio
    row_p = old_row-1
    col_p = old_col-1
    new_row_ratio = new_row/row_p
    new_col_ratio = new_col/col_p
    new_image = np.zeros((new_row, new_col, channels), dtype=np.uint8)

    print("original_shape : ", original_image.shape)
    print("need to wait about 20 seconds for ratio=3 ...")
    for row in range(new_row):
        for col in range(new_col):
            x_i = int(row//new_row_ratio)
            y_i = int(col//new_col_ratio)
            x = row/new_row_ratio
            y = col/new_col_ratio
            x_rate = x - x_i
            y_rate = y - y_i
            x_i1 = x_i+1
            y_i1 = y_i+1
            if x_rate == 0 and y_rate == 0:
                new_image[row][col] = original_image[x_i][y_i]
            elif x_rate == 0:
                new_image[row][col] = ((y - y_i) / (y_i1 - y_i)) * original_image[x_i][y_i1] + ((y_i1 - y) / (y_i1 - y_i)) * original_image[x_i][y_i]
            elif y_rate == 0:
                new_image[row][col] = ((x - x_i) / (x_i1 - x_i)) * original_image[x_i1][y_i] + ((x_i1 - x) / (x_i1 - x_i)) * original_image[x_i][y_i]
            else:
                temp_x1 = ((x - x_i) / (x_i1 - x_i)) * original_image[x_i1][y_i] + ((x_i1 - x) / (x_i1 - x_i)) * original_image[x_i][y_i]
                temp_x2 = ((x - x_i) / (x_i1 - x_i)) * original_image[x_i1][y_i1] + ((x_i1 - x) / (x_i1 - x_i)) * original_image[x_i][y_i1]
                new_image[row][col] = ((y - y_i) / (y_i1 - y_i)) * temp_x2 + ((y_i1 - y) / (y_i1 - y_i)) * temp_x1
    
    print("new_image shape : ", new_image.shape)
    
    if show_image:
        cv.imshow("new_image", new_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    if save_image:
        os.makedirs(directory, exist_ok=True)
        cv.imwrite(output_file_path, new_image)

if __name__ == "__main__":
    input_file_path = "./src/ive.jpg"
    output_file_path = "./output/Lab1_2.jpg"
    directory = "output"
    ratio = 3
    show_image = True
    save_image = True
    exercise_2(
        input_file_path, 
        output_file_path, 
        directory, 
        ratio, 
        show_image, 
        save_image
    )