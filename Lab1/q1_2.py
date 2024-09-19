import cv2 as cv
import os 
import numpy as np
""" 
對比與亮度(10%)
更改原始圖片中的「藍點與黃點」像素的對比與亮度，其餘像素保持原樣
Hint: (B + G) * 0.3 > R
new_image = (old_image - 127) × (contrast/127 + 1) + 127 + brightness
Hint: 記得注意overflow的問題
可能會用到的函式: np.array(img, dtype=np.int32)、np.clip(img, 0, 255)、np.array(img, dtype=np.uint8)
"""
def exercise_1_2(input_file_path, output_file_path, directory, contrast=100, brightness=40, show_image=False, save_image=True):
    print("exercise_1_2")
    original_image = cv.imread(input_file_path)

    new_image = original_image.astype(np.int32)
    b, g, r = cv.split(original_image)
    shape = original_image.shape

    for row in range(shape[0]):
        for col in range(shape[1]):
            bb = int(b[row][col]) # avoid overflow
            gg = int(g[row][col]) 
            rr = int(r[row][col])
            is_yellow_point = (
                (bb + gg) * 0.3 > rr
            )
            is_blue_point = (
                bb > 100 \
                and bb * 0.6 > gg  \
                and bb * 0.6 > rr
            )
            if is_yellow_point or is_blue_point:
                for channel in range(3):
                    new_image[row][col][channel] = (original_image[row][col][channel]-127) * (contrast/127 + 1) + 127 + brightness
            else :
                new_image[row][col] = original_image[row][col]
    new_image = np.clip(new_image, 0, 255)
    new_image = np.array(new_image, dtype=np.uint8)

    if show_image:
        cv.imshow("new_image", new_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    if save_image:
        os.makedirs(directory, exist_ok=True)
        cv.imwrite(output_file_path, new_image)

if __name__ == "__main__" :
    input_file_path = "./src/test.jpg"
    output_file_path = "./output/Lab1_1_2.jpg"
    directory = "output"
    contrast = 100
    brightness = 40
    show_image = True
    save_image = True
    exercise_1_2(
        input_file_path, 
        output_file_path,
        directory, 
        contrast, 
        brightness, 
        show_image, 
        save_image
    )

