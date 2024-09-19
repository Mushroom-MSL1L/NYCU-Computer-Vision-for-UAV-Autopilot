from q1_1 import exercise_1_1
from q1_2 import exercise_1_2
from q2 import exercise_2
from q3 import exercise_3, exercise_3_hint_method

if __name__ == "__main__":
    show_image = True
    save_image = True
    exercise_1_1(
        input_file_path = "./src/test.jpg",
        output_file_path = "./output/Lab1_1_1.jpg",
        directory = "output",
        show_image =  show_image,
        save_image = save_image
    )
    exercise_1_2(
        input_file_path = "./src/test.jpg",
        output_file_path = "./output/Lab1_1_2.jpg",
        directory = "output",
        contrast = 100,
        brightness = 40,
        show_image = show_image,
        save_image = save_image
    )
    exercise_2(
        input_file_path = "./src/ive.jpg",
        output_file_path = "./output/Lab1_2.jpg",
        directory = "output",
        ratio = 3,
        show_image = show_image,
        save_image = save_image
    )
    exercise_3(
        input_file_path = "./src/ive.jpg",
        output_file_path = "./output/Lab1_3.jpg",
        directory = "output",
        show_image = show_image,
        save_image = save_image
    )
    # exercise_3_hint_method(
    #     input_file_path = "./src/ive.jpg",
    #     output_file_path = "./output/Lab1_3_hint.jpg",
    #     directory = "output",
    #     show_image = show_image,
    #     save_image = save_image
    # )    

