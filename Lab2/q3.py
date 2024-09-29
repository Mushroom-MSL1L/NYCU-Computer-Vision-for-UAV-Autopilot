import cv2 as cv
import os 
import numpy as np
from q2 import exercise_2
import random
"""
Connected Component (50%)
● Two-Pass Algorithm:
Pass 1:
• Perform label assignment and label propagation. 
• Construct the equivalence relations between labels when two different labels propagate to the 
same pixel.
• Apply resolve function to find the transitive closure of all equivalence relations. 
Pass 2: 
• Perform label translation
"""
def generate_color_image(shape):
    random.seed(5)
    color_image = np.zeros(shape, dtype=np.uint8)
    for row in range(shape[0]):
        for col in range(shape[1]):
            color_image[row][col] = [random.randrange(0, 256) for _ in range(3)]
    return color_image

class DisjointSet:
    def __init__ (self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0 for i in range(n)]
    def find_root(self, x):
        if self.parent[x] == x :
            return x 
        self.parent[x] = self.find_root(self.parent[x])
        return self.parent[x]
    def union(self, x, y) : # x <- y
        root_x = self.find_root(x)
        root_y = self.find_root(y)
        if root_x == root_y :
            return
        if self.rank[root_x] > self.rank[root_y] :
            self.parent[root_y] = root_x
        elif self.rank[root_x] < self.rank[root_y] :
            self.parent[root_x] = root_y
        else :
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
def exercise_3(binary_image, output_file_path, directory, show_image=False, save_image=True):
    print("exercise_3")
    image = binary_image
    length = image.copy().flatten().shape[0]
    shape = image.shape
    new_image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    new_shape = (shape[0], shape[1], 3)
    ds = DisjointSet(length)
    color_image = generate_color_image(new_shape)
    # using joint set to find the connected component
    for row in range(shape[0]):
        for col in range(shape[1]):
            if row > 0 and  image[row][col] == image[row-1][col] :
                now = row * shape[1] + col
                up = (row-1) * shape[1] + col
                ds.union(now, up)
            if col > 0 and image[row][col] == image[row][col-1] :
                now = row * shape[1] + col
                left = row * shape[1] + col - 1
                ds.union(now, left)
    # connect pixels and assign colors 
    for row in range(shape[0]):
        for col in range(shape[1]):
            index = row * shape[1] + col
            root_pixel = ds.find_root(index)
            if image[row][col] == 0:
                continue
            new_image[row][col] = color_image[root_pixel // shape[1]][root_pixel % shape[1]]
    
    if show_image:
        cv.imshow("new_image", new_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    if save_image:
        os.makedirs(directory, exist_ok=True)
        cv.imwrite(output_file_path, new_image)
        
        

if __name__ == "__main__":
    first_input_file_path = "./src/input.jpg"
    first_output_file_path = "./output/Lab2_2.jpg"
    output_file_path = "./output/Lab2_3.jpg"
    directory = "output"
    show_image = True
    save_image = True
    binary_image = exercise_2(
        first_input_file_path, 
        first_output_file_path, 
        directory, 
        False, 
        False
    )
    exercise_3(
        binary_image,
        output_file_path, 
        directory, 
        show_image, 
        save_image
    )
