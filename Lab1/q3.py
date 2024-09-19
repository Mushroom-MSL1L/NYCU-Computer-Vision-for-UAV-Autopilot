import cv2 as cv
import os 
import numpy as np

""" 
邊緣偵測(filtering & Sobel Operator) (30%)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 將圖片轉為灰階
img = cv2.GaussianBlur(img, (5, 5), 0) 對灰階圖做高斯模糊(去雜訊)
cv2.filter2D(img, -1, kernel)
### Do NOT use cv2.Sobel() directly
"""

sobel_x = np.array([[  1,  0, -1], 
                    [  2,  0, -2], 
                    [  1,  0, -1]])
sobel_y = np.array([[  1,  2,  1], 
                    [  0,  0,  0], 
                    [ -1, -2, -1]])
inverse_sobel_x = np.array([
                    [ -1,  0,  1], 
                    [ -2,  0,  2], 
                    [ -1,  0,  1]])
inverse_sobel_y = np.array(
                    [[ -1, -2, -1], 
                    [  0,  0,  0], 
                    [  1,  2,  1]])

def exercise_3(input_file_path, output_file_path, directory, show_image=False, save_image=True):
    print("exercise_3")
    original_image = cv.imread(input_file_path)

    gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    blur_image = cv.GaussianBlur(gray_image, (5, 5), 0)

    g_x = cv.filter2D(blur_image, -1, inverse_sobel_x)
    g_y = cv.filter2D(blur_image, -1, inverse_sobel_y)
    g = cv.addWeighted(g_x, 1, g_y, 1, 0)
    # g = np.sqrt(g_x**2 + g_y**2).astype(np.uint8)
    g = cv.normalize(g, None, 0, 255, cv.NORM_MINMAX)
    # g = cv.convertScaleAbs(g)

    if show_image:
        cv.imshow("new_image", g)
        cv.waitKey(0)
        cv.destroyAllWindows()
    if save_image:
        os.makedirs(directory, exist_ok=True)
        cv.imwrite(output_file_path, g)
    
def exercise_3_hint_method(input_file_path, output_file_path, directory, show_image=False, save_image=True):
    print("exercise_3")
    original_image = cv.imread(input_file_path)

    gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    blur_image = cv.GaussianBlur(gray_image, (5, 5), 0)

    g_x = cv.filter2D(blur_image, -1, sobel_x).astype(np.int32)
    g_y = cv.filter2D(blur_image, -1, sobel_y).astype(np.int32)
    g = np.sqrt(g_x**2 + g_y**2).astype(np.uint8)
    g = cv.normalize(g, None, 0, 255, cv.NORM_MINMAX)
    g = cv.convertScaleAbs(g)

    if show_image:
        cv.imshow("new_image", g)
        cv.waitKey(0)
        cv.destroyAllWindows()
    if save_image:
        os.makedirs(directory, exist_ok=True)
        cv.imwrite(output_file_path, g)
    

if __name__ == "__main__":
    input_file_path = "./src/ive.jpg"
    output_file_path = "./output/Lab1_3.jpg"
    directory = "output"
    show_image = True
    save_image = True
    ## both ok, but different implementations
    exercise_3(input_file_path, output_file_path, directory, show_image, save_image)
    # exercise_3_hint_method(input_file_path, output_file_path, directory, show_image, save_image)
