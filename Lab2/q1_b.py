import cv2 as cv
import os 
import numpy as np
"""
先將圖片轉成HSV格式後對V做直方圖等化
● 計算輸入圖的直方圖
● 計算直方圖的累計表
● 用直方圖累計表完成各強度的映射
"""
def exercise_1_b(input_file_path, output_file_path, directory, show_image=False, save_image=True):
    print("exercise_1_b")
    original_image = cv.imread(input_file_path)
    hsv = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)
    shape = hsv.shape
    h, s, v = cv.split(hsv)
    new_value = v.copy()
    new_image = hsv.copy()
    
    number_of_pixels = np.zeros(256)
    aggregate_number_of_pixels = np.zeros(256)
    quantized_values = np.zeros(256)
    for pixel in v.flatten():
        number_of_pixels[pixel] += 1
        
    aggregate_number_of_pixels[0] = number_of_pixels[0]
    for i in range(1, 256):
        aggregate_number_of_pixels[i] = aggregate_number_of_pixels[i-1] + number_of_pixels[i]
        
    for i in range(256):
        quantized_values[i] = round(aggregate_number_of_pixels[i] * 255 / aggregate_number_of_pixels[255])
        
    for row in range(shape[0]):
        for col in range(shape[1]):
            new_value[row][col] = quantized_values[v[row][col]]
    new_image = cv.merge([h, s, new_value])
    new_image = cv.cvtColor(new_image, cv.COLOR_HSV2BGR)
            
    if show_image:
        cv.imshow("new_image", new_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    if save_image:
        os.makedirs(directory, exist_ok=True)
        cv.imwrite(output_file_path, new_image)
        
        

if __name__ == "__main__":
    input_file_path = "./src/histogram.jpg"
    output_file_path = "./output/Lab2_1_b.jpg"
    directory = "output"
    show_image = True
    save_image = True
    exercise_1_b(
        input_file_path, 
        output_file_path, 
        directory, 
        show_image, 
        save_image
    )
