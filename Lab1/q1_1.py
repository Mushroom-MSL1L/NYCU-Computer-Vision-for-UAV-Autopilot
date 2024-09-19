import cv2 as cv
import os 
"""
灰階與顏色濾鏡(20%)
將原始圖片中的「藍點」予以保留，並把其餘的點改為灰階。
Hint: B > 100 and B * 0.6 > G and B * 0.6 > R
"""
def exercise_1_1(input_file_path, output_file_path, directory, show_image=False, save_image=True):
    print("exercise_1_1")
    original_image = cv.imread(input_file_path)

    new_image = original_image.copy()
    gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    b, g, r = cv.split(original_image)
    shape = original_image.shape

    for row in range(shape[0]):
        for col in range(shape[1]):
            if b[row][col] > 100 \
            and b[row][col] * 0.6 > g[row][col] \
            and b[row][col] * 0.6 > r[row][col]:
                new_image[row][col] = original_image[row][col]
            else :
                new_image[row][col] = gray_image[row][col]

    if show_image:
        cv.imshow("new_image", new_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    if save_image:
        os.makedirs(directory, exist_ok=True)
        cv.imwrite(output_file_path, new_image)

if __name__ == "__main__":
    input_file_path = "./src/test.jpg"
    output_file_path = "./output/Lab1_1_1.jpg"
    directory = "output"
    show_image = True
    save_image = True
    exercise_1_1(
        input_file_path, 
        output_file_path, 
        directory, 
        show_image, 
        save_image
    )