import cv2

# Initialize a list to store the clicked points
clicked_points = []

# Mouse callback function to get the points
def get_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button down
        clicked_points.append((x, y))  # Add the point to the list
        print(f"Point selected: ({x}, {y})")

# Load the image
image_path = 'src/screen.jpg'  # Change this to your image path
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Unable to load image.")
else:
    cv2.imshow('Click on Image', image)  # Show the image
    cv2.setMouseCallback('Click on Image', get_points)  # Set the mouse callback function

    # Wait until the user presses 'q' to exit
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cv2.destroyAllWindows()  # Close all OpenCV windows

# Print all selected points after closing the window
print("Selected points:", clicked_points)
