import numpy as np
import time
import cv2 

class LineTrace:
    def __init__(self):
        self.threshold = 100     # range 0~255, for 黑白圖片
        self.thres_rate = 0.2    # range 0~1, for 九宮格

        self.move_speed = 20         # default 20, for line trace control
        self.sleep_time = 0.1        # default 0.1, for line trace control
        self.distance_rate = 0.5     # range 0~1, for line trace distance adjustment
        self.distance_durance = 0.1  # range 0~1, for line trace distance adjustment

    def process_frame(self, frame):
        temp = frame.copy()
        gray = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        dilation = cv2.dilate(binary, np.ones((3,3), np.uint8), iterations = 1)
        return dilation

    def get_pattern(self, img, pattern_size=(3, 3)):
        # thres_rate should be a value between 0 ~ 1
        # img should be binary image
        thres_rate = self.thres_rate
        height = img.shape[0]
        width = img.shape[1]
        height_step = height // pattern_size[0]
        width_step = width // pattern_size[1]
        thres = (1 - thres_rate) * height_step * width_step
        pattern = np.zeros(pattern_size)
        for h in range(pattern_size[0]):
            for w in range(pattern_size[1]):
                pattern[h, w] = (np.sum(img[h * height_step : (h + 1) * height_step, 
                                            w * width_step : (w + 1) * width_step]) 
                                < thres)
        return pattern

    def right_empty(self, pattern):
        return np.sum(pattern[:, 2]) == 0
    def left_empty(self, pattern):
        return np.sum(pattern[:, 0]) == 0
    def up_empty(self, pattern):
        return np.sum(pattern[0, :]) == 0
    def down_empty(self, pattern):
        return np.sum(pattern[2, :]) == 0

    def determine_line_distance(self, img):
        distance_rate = self.distance_rate
        distance_tolerance = self.distance_durance
        # distance_rate should be a value between 0 ~ 1
        # distance_tolerance should be a value between 0 ~ 1
        # img should be binary image
        
        height = img.shape[0]
        width = img.shape[1]
        upper_rate = min(1, distance_rate + distance_tolerance)
        lower_rate = max(0, distance_rate - distance_tolerance)
        upper_thres = (1 - upper_rate) * height * width
        lower_thres = (1 - lower_rate) * height * width
        current_sum = np.sum(img)
        
        if current_sum > upper_thres:
            return -1 # too close
        elif current_sum < lower_thres:
            return 1 # too far
        return 0 # don't adjust

    def vertical_up_line_tracing (self, drone, pattern, distance=0):
        move_speed, sleep_time = self.move_speed, self.sleep_time
        
        pattern3 = np.logical_or.reduce([pattern[0], pattern[1], pattern[2]])
        print("pattern3 : ", pattern3)
        # x, z, y
        # move_speed > 0 : means toward up
        # move_speed < 0 : means toward down
        # distance < 0 : means too close, move backward
        # distance > 0 : means too far, move forward
        absolute_speed = abs(move_speed)
        adjust_distance = absolute_speed * distance
        if np.equal(pattern, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])).all() :
            ## too close
            drone.send_rc_control(0, -absolute_speed, 0, 0)
            time.sleep(sleep_time)
            return "ver : too close"
        if np.all(np.equal(pattern3, [0, 1, 0])):
            drone.send_rc_control(0, adjust_distance, move_speed, 0)
            time.sleep(sleep_time)
            return "ver : up"
        elif np.all(np.equal(pattern3, [0, 0, 1])):
            drone.send_rc_control(absolute_speed//2, adjust_distance, 0, 0)
            time.sleep(sleep_time)
            return "ver : right"
        elif np.all(np.equal(pattern3, [1, 0, 0])):
            drone.send_rc_control(-absolute_speed//2, adjust_distance, 0, 0)
            time.sleep(sleep_time)
            return "ver : left"
        elif np.all(np.equal(pattern3, [0, 1, 1])):
            drone.send_rc_control(absolute_speed//2, adjust_distance, move_speed//2, 0)
            time.sleep(sleep_time//2)
            return "ver : little right"
        elif np.all(np.equal(pattern3, [1, 1, 0])):
            drone.send_rc_control(-absolute_speed//2, adjust_distance, move_speed//2, 0)
            time.sleep(sleep_time//2)
            return "ver : little left"
        elif np.all(np.equal(pattern3, [0, 0, 0])):
            return "ver : 0, 0, 0"
        elif np.all(np.equal(pattern3, [1, 1, 1])):
            drone.send_rc_control(0, adjust_distance, move_speed//2, 0)
            time.sleep(sleep_time)
            return "ver : 1, 1, 1"
        else :
            return "ver : 1, 0, 1"

    def vertical_down_line_tracing (self, drone, pattern, distance=0):
        move_speed, sleep_time = -self.move_speed, self.sleep_time
        
        pattern3 = np.logical_or.reduce([pattern[0], pattern[1], pattern[2]])
        print("pattern3 : ", pattern3)
        # x, z, y
        # move_speed > 0 : means toward up
        # move_speed < 0 : means toward down
        # distance < 0 : means too close, move backward
        # distance > 0 : means too far, move forward
        absolute_speed = abs(move_speed)
        adjust_distance = absolute_speed * distance
        if np.equal(pattern, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])).all() :
            ## too close
            drone.send_rc_control(0, -absolute_speed, 0, 0)
            time.sleep(sleep_time)
            return "ver : too close"
        if np.all(np.equal(pattern3, [0, 1, 0])):
            drone.send_rc_control(0, adjust_distance, move_speed, 0)
            time.sleep(sleep_time)
            return "ver : down"
        elif np.all(np.equal(pattern3, [0, 0, 1])):
            drone.send_rc_control(absolute_speed//2, adjust_distance, 0, 0)
            time.sleep(sleep_time)
            return "ver : right"
        elif np.all(np.equal(pattern3, [1, 0, 0])):
            drone.send_rc_control(-absolute_speed//2, adjust_distance, 0, 0)
            time.sleep(sleep_time)
            return "ver : left"
        elif np.all(np.equal(pattern3, [0, 1, 1])):
            drone.send_rc_control(absolute_speed//2, adjust_distance, move_speed//2, 0)
            time.sleep(sleep_time//2)
            return "ver : little right"
        elif np.all(np.equal(pattern3, [1, 1, 0])):
            drone.send_rc_control(-absolute_speed//2, adjust_distance, move_speed//2, 0)
            time.sleep(sleep_time//2)
            return "ver : little left"
        elif np.all(np.equal(pattern3, [0, 0, 0])):
            return "ver : 0, 0, 0"
        elif np.all(np.equal(pattern3, [1, 1, 1])):
            drone.send_rc_control(0, adjust_distance, move_speed//2, 0)
            time.sleep(sleep_time)
            return "ver : 1, 1, 1"
        else :
            return "ver : 1, 0, 1"

    def horizontal_right_line_tracing(self, drone, pattern, distance=0):
        move_speed, sleep_time = self.move_speed, self.sleep_time
        
        pattern3 = np.logical_or.reduce([pattern[0], pattern[1], pattern[2]], axis=1)
        print("pattern3 : ", pattern3)
        
        absolute_speed = abs(move_speed)
        adjust_distance = absolute_speed * distance
        if np.equal(pattern, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])).all() :
            ## too close
            drone.send_rc_control(0, -absolute_speed, 0, 0)
            time.sleep(sleep_time)
            return "ver : too close"

        # x, z, y
        if np.all(np.equal(pattern3, [0, 1, 0])):
            drone.send_rc_control(move_speed, adjust_distance, 0, 0)
            time.sleep(sleep_time)
            return "hor r : forward to right"
        elif np.all(np.equal(pattern3, [0, 0, 1])):
            drone.send_rc_control(0, adjust_distance, -move_speed, 0)
            time.sleep(sleep_time)
            return "hor r : move down"
        elif np.all(np.equal(pattern3, [1, 0, 0])):
            drone.send_rc_control(0, adjust_distance, move_speed, 0)
            time.sleep(sleep_time)
            return "hor r : move up"
        elif np.all(np.equal(pattern3, [0, 1, 1])):
            drone.send_rc_control(0, adjust_distance, -move_speed, 0)
            time.sleep(sleep_time // 2)
            return "hor r : move little down"
        elif np.all(np.equal(pattern3, [1, 1, 0])):
            drone.send_rc_control(0, adjust_distance, move_speed, 0)
            time.sleep(sleep_time // 2)
            return "hor r : move little up"
        elif np.all(np.equal(pattern3, [0, 0, 0])):
            return "hor r : 0, 0, 0"
        elif np.all(np.equal(pattern3, [1, 1, 1])):
            drone.send_rc_control(move_speed, adjust_distance, 0, 0)
            time.sleep(sleep_time)
            return "hor r : 1, 1, 1"
        else:
            return "hor r : 1, 0, 1"

    def horizontal_left_line_tracing(self, drone, pattern, distance=0):
        move_speed, sleep_time = self.move_speed, self.sleep_time
        
        pattern3 = np.logical_or.reduce([pattern[0], pattern[1], pattern[2]], axis=1)
        print("pattern3 : ", pattern3)

        absolute_speed = abs(move_speed)
        adjust_distance = absolute_speed * distance
        if np.equal(pattern, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])).all() :
            ## too close
            drone.send_rc_control(0, -absolute_speed, 0, 0)
            time.sleep(sleep_time)
            return "ver : too close"

        # x, z, y
        if np.all(np.equal(pattern3, [0, 1, 0])):
            drone.send_rc_control(-move_speed, adjust_distance, 0, 0)
            time.sleep(sleep_time)
            return "hor l : forward to left"
        elif np.all(np.equal(pattern3, [0, 0, 1])):
            drone.send_rc_control(0, adjust_distance, -move_speed, 0)
            time.sleep(sleep_time)
            return "hor l : move down"
        elif np.all(np.equal(pattern3, [1, 0, 0])):
            drone.send_rc_control(0, adjust_distance, move_speed, 0)
            time.sleep(sleep_time)
            return "hor l : move up"
        elif np.all(np.equal(pattern3, [0, 1, 1])):
            drone.send_rc_control(0, adjust_distance, -move_speed, 0)
            time.sleep(sleep_time // 2)
            return "hor l : move little down"
        elif np.all(np.equal(pattern3, [1, 1, 0])):
            drone.send_rc_control(0, adjust_distance, move_speed, 0)
            time.sleep(sleep_time // 2)
            return "hor l : move little up"
        elif np.all(np.equal(pattern3, [0, 0, 0])):
            return "hor l : 0, 0, 0"
        elif np.all(np.equal(pattern3, [1, 1, 1])):
            drone.send_rc_control(-move_speed, adjust_distance, 0, 0)
            time.sleep(sleep_time)
            return "hor l : 1, 1, 1"
        else:
            return "hor l : 1, 0, 1"

"""
Sample usage
"""

if __name__ == "__main__":
    find_id = 0
    task_index = 0
    
    lt = LineTrace()
    frame = cv2.imread("line_trace.jpg")
    dilation = lt.process_frame(frame)
    cv2.imshow("dilation", dilation)
    cv2.namedWindow("dilation", 0)
    cv2.resizeWindow("dilation", 800, 600)
    state = ""

    binary01 = dilation // 255
    pattern = lt.get_pattern(binary01)
    text = "pattern: \n" + str(pattern[0]) + "\n" + str(pattern[1]) + "\n" + str(pattern[2])
    print(text)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if  find_id == 1 :
        distance = lt.determine_line_distance(binary01)
        if task_index == 0 :
            ## intentionally stop
            task_index += 1
            drone.send_rc_control(0, 0, 0, 0)
        elif task_index == 1 : 
            # go right
            state = lt.horizontal_right_line_tracing(drone, pattern, distance)
            if lt.right_empty(pattern) :
                task_index += 1
                ## forcely go up
                drone.send_rc_control(0, 0, 30, 0)
                time.sleep(1.5)
        elif task_index == 2 : 
            # go up
            state = lt.vertical_up_line_tracing(drone, pattern, distance)
            if lt.up_empty(pattern) :
                task_index += 1    
                ## forcely go right
                drone.send_rc_control(20, 0, 0, 0)
                time.sleep(1.5)
        elif task_index == 3 : 
            # go right
            state  = lt.horizontal_right_line_tracing(drone, pattern, distance)
            if lt.right_empty(pattern) :
                task_index += 1    
                ## forcely go up 
                drone.send_rc_control(0, 0, 20, 0)
                time.sleep(1.5)
        elif task_index == 4 : 
            # go up
            state = lt.vertical_up_line_tracing(drone, pattern, distance)
            if lt.up_empty(pattern) :
                task_index += 1
                ## forcely go left
                drone.send_rc_control(-20, 0, 0, 0)
                time.sleep(1.5)
        elif task_index == 5 : 
            # go left
            state = lt.horizontal_left_line_tracing(drone, pattern, distance)
            if lt.left_empty(pattern) :
                task_index += 1
                ## forcely go down
                drone.send_rc_control(0, 0, -20, 0)
                time.sleep(1.5)
        elif task_index == 6 : 
            # go down
            # move_speed should be negative
            state = lt.vertical_down_line_tracing(drone, pattern, distance)
            go_back_id1 = 1

    ## end of trace line
    else : ## if no marker is detected, ids is None
        drone.send_rc_control(0, 0, 0, 0)
    key = cv2.waitKey(1)

    text = "state: " + state
    cv2.putText(frame, text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    print(text)

    cv2.namedWindow("frame", 0)
    cv2.resizeWindow("frame", 800, 600)
    cv2.imshow("frame", frame)
    if key & 0xFF == ord('q') :
        cv2.destroyAllWindows()
