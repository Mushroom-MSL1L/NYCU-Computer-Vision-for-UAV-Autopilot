from q1_a import *
from q1_b import *
from q2 import *
from q3 import *

if __name__ == "__main__":
    exercise_1_a(
        input_file_path = "./src/histogram.jpg",
        output_file_path = "./output/Lab2_1_a.jpg",
        directory = "output",
        show_image = True,
        save_image = True
    )
    exercise_1_b(
        input_file_path = "./src/histogram.jpg",
        output_file_path = "./output/Lab2_1_b.jpg",
        directory = "output",
        show_image = True,
        save_image = True
    )
    binary_image = exercise_2(
        input_file_path = "./src/input.jpg",
        output_file_path = "./output/Lab2_2.jpg",
        directory = "output",
        show_image = True,
        save_image = True
    )
    exercise_3(
        binary_image,
        output_file_path = "./output/Lab2_3.jpg",
        directory = "output",
        show_image = True,
        save_image = True
    )