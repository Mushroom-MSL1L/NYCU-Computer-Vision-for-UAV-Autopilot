import cv2 as cv
import os 
import numpy as np
"""
BGR 彩色圖片直方圖等化
● 計算輸入圖的直方圖
● 計算直方圖的累計表
● 用直方圖累計表完成各強度的映射
"""
def exercise_1_a(input_file_path, output_file_path, directory, show_image=False, save_image=True):
    print("exercise_1_a")
    original_image = cv.imread(input_file_path)

    new_image = original_image.copy()
    b, g, r = cv.split(original_image)
    
    b = equalize_histogram_2_one_channel(b, "blue channel")
    g = equalize_histogram_2_one_channel(g, "green channel")
    r = equalize_histogram_2_one_channel(r, "red channel")
    new_image = cv.merge([b, g, r])

    if show_image:
        cv.imshow("new_image", new_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    if save_image:
        os.makedirs(directory, exist_ok=True)
        cv.imwrite(output_file_path, new_image)
        
def equalize_histogram_2_one_channel(image, name):
    print("eqaulize_histogram_2_one_channel: " + name)
    number_of_pixels = np.zeros(256)
    aggregate_number_of_pixels = np.zeros(256)
    quantized_values = np.zeros(256)
    shape = image.shape
    new_image = image.copy()
    original_image = image.copy()
    image = image.flatten()
    for pixel in image:
        number_of_pixels[pixel] += 1
        
    aggregate_number_of_pixels[0] = number_of_pixels[0]
    for i in range(1, 256):
        aggregate_number_of_pixels[i] = aggregate_number_of_pixels[i-1] + number_of_pixels[i]
        
    for i in range(256):
        quantized_values[i] = round(aggregate_number_of_pixels[i] * 255 / aggregate_number_of_pixels[255])
        
    for row in range(shape[0]):
        for col in range(shape[1]):
            new_image[row][col] = quantized_values[original_image[row][col]]
        
    return new_image.astype(np.uint8)

if __name__ == "__main__":
    input_file_path = "./src/histogram.jpg"
    output_file_path = "./output/Lab2_1_a.jpg"
    directory = "output"
    show_image = True
    save_image = True
    exercise_1_a(
        input_file_path, 
        output_file_path, 
        directory, 
        show_image, 
        save_image
    )
