import cv2 as cv
import os 
import numpy as np
"""
Otsu Threshold
● 先計算影像的直方圖。
● 把直方圖強度大於閾值的像素分成一組,把小於閾值的像素分成另一組。
● 分別計算這兩組的組內變異數,並把兩個組內變異數相加。
● 將 0 ~ 255 依序當作閾值來計算組內變異數和,總和值最小的就是結果閾值。
"""
def exercise_2(input_file_path, output_file_path, directory, show_image=False, save_image=True):
    print("exercise_2")
    original_image = cv.imread(input_file_path)
    gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    new_image = gray_image.copy()
    
    number_of_pixels = np.zeros(256)
    for pixel in gray_image.flatten():
        number_of_pixels[pixel] += 1
    
    threshold = 1
    n1 = np.sum(number_of_pixels[:threshold])
    n2 = np.sum(number_of_pixels[threshold:])
    u1 = (np.sum([i * number_of_pixels[i] for i in range(threshold)]) / n1)
    u2 = (np.sum([i * number_of_pixels[i] for i in range(threshold, 256)]) / n2)
    v_between_group = (n1 * n2 * (u1 - u2) ** 2)
    previous_n1, previous_n2, previous_u1, previous_u2 = n1, n2, u1, u2
    best_threshold = 1
    max_v_between_group = v_between_group
    for threshold in range (2, 256):
        n1 = previous_n1 + number_of_pixels[threshold]
        n2 = previous_n2 - number_of_pixels[threshold]
        if n1 == 0 or n2 == 0:
            continue
        u1 = ((previous_n1 * previous_u1 + threshold * number_of_pixels[threshold]) / n1)
        u2 = ((previous_n2 * previous_u2 - threshold * number_of_pixels[threshold]) / n2)
        v_between_group = (n1 * n2 * (u1 - u2) ** 2)
        previous_n1, previous_n2, previous_u1, previous_u2 = n1, n2, u1, u2
        if max_v_between_group < v_between_group:
            max_v_between_group = v_between_group
            best_threshold = threshold
    _, new_image = cv.threshold(gray_image, best_threshold, 255, cv.THRESH_BINARY)
    
    if show_image:
        cv.imshow("new_image", new_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    if save_image:
        os.makedirs(directory, exist_ok=True)
        cv.imwrite(output_file_path, new_image)
        
    return new_image
        
        

if __name__ == "__main__":
    input_file_path = "./src/input.jpg"
    output_file_path = "./output/Lab2_2.jpg"
    directory = "output"
    show_image = True
    save_image = True
    exercise_2(
        input_file_path, 
        output_file_path, 
        directory, 
        show_image, 
        save_image
    )
