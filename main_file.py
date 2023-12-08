from utilities import *
from calculations import *



eye_aspect_ratio_threshold = 0.19 #0.19
eye_aspect_ratio_frame_count = 15
wink_aspect_ratio_difference_threshold = 0.04
wink_aspect_ratio_close_threshold = 0.19
wink_frame_count = 10

# Initializing some key variables
input_active = False
pivot = (0, 0)
mouth_event_count = 0
eye_event_count = 0
wink_event_count = 0

scroll_active = False
red = (0, 0, 255)
# Getting the current directory of the file
pwd = os.path.dirname(__file__)






# Using Dlib's shape predictor to create the facial landmark predictor and 
# initializing the frontal face detector
shape_predictor = dlib.shape_predictor(pwd + "/model/shape_predictor_68_face_landmarks.dat")
frontal_face_detector = dlib.get_frontal_face_detector()



# Getting the facial land mark indices of left eye, right eye, nose and mouth
(mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(nose_start, nose_end) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Setting up the video capture
cam_capture = cv2.VideoCapture(0)

# Recording FPS
fps = FPS().start()

# Infinite loop to continuously calculate the aspect ratios, 
# tracking the facial movements and performing mouse cursor operations
while True:

    x, cv2_frame = cam_capture.read()
    cv2_frame = imutils.resize(cv2_frame, width=640, height=480)
    cv2_frame = cv2.flip(cv2_frame, 1)

    # Facial Detection in grayscale frame
    detected_faces = frontal_face_detector(cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2GRAY), 0)

    if len(detected_faces) == 0:
        cv2.imshow("Camera Feed1", cv2_frame)
        key = cv2.waitKey(1) & 0xFF
        continue
    else:
        face_detected = detected_faces[0]
        

    # Calculating the facial landmarks within the facial region
    facial_landmarks = shape_predictor(cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2GRAY), face_detected)

    # Converting the coordinates of these facial landmarks into a NumPy array
    facial_landmarks = face_utils.shape_to_np(facial_landmarks)

    # Getting the coordinates of the left and right eyes, and 
    # using these coordinates to calculate the eye aspect ratio for both eyes
    mouth_landmarks = facial_landmarks[mouth_start:mouth_end]
    left_eye_landmarks = facial_landmarks[left_eye_start:left_eye_end]
    right_eye_landmarks = facial_landmarks[right_eye_start:right_eye_end]
    nose_landmarks = facial_landmarks[nose_start:nose_end]

    # Swapping left_eye and right_eye since the frame is flipped
    var = left_eye_landmarks
    left_eye_landmarks = right_eye_landmarks
    right_eye_landmarks = var

    # Calculatig the aspect ratios for mouth, left eye and right eye
    mouth_aspect_ratio = calculate_mar(mouth_landmarks)
    left_eye_aspect_ratio = calculate_ear(left_eye_landmarks)
    right_eye_aspect_ratio = calculate_ear(right_eye_landmarks)

    # Calculating the average eye aspect ratio for both eyes
    eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0

    # Difference in the eye aspect ratios for both eyes
    diff_ear = abs(left_eye_aspect_ratio - right_eye_aspect_ratio)

    nose_point = (nose_landmarks[3, 0], nose_landmarks[3, 1])

    font = cv2.FONT_HERSHEY_COMPLEX

    # Displaying the rounded MAR and EAR values in real-time
    cv2.putText(cv2_frame, f'MAR: {round(mouth_aspect_ratio, 6)}', (10, 60), font , 0.6, (255, 0, 0), 1)
    cv2.putText(cv2_frame, f'LEFT EAR: {round(left_eye_aspect_ratio, 6)}', (10, 80), font, 0.6, (255, 0, 0), 1)
    cv2.putText(cv2_frame, f'RIGHT EAR: {round(right_eye_aspect_ratio, 6)}', (10, 100), font, 0.6, (255, 0, 0), 1)
    cv2.putText(cv2_frame, f'EAR: {round(eye_aspect_ratio, 6)}', (10, 120), font, 0.6, (255, 0, 0), 1)
    # cv2.putText(cv2_frame, f'{fps}', (10, 120), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)

    # Calculating the convex hull for both the left and right eyes
    convex_hull_left_eye = cv2.convexHull(left_eye_landmarks)
    convex_hull_right_eye = cv2.convexHull(right_eye_landmarks)
    convex_hull_mouth = cv2.convexHull(mouth_landmarks)
    

    # Displaying each of the eyes visually.
    cv2.drawContours(cv2_frame, [convex_hull_left_eye], -1, (0, 255, 255), 1)
    cv2.drawContours(cv2_frame, [convex_hull_right_eye], -1, (0, 255, 255), 1)
    cv2.drawContours(cv2_frame, [convex_hull_mouth], -1, (0, 255, 255), 1)
    


    # Toggling the input activation based on mouth aspect ratio and consecutive frame count.

    if diff_ear > wink_aspect_ratio_difference_threshold:

        if left_eye_aspect_ratio < right_eye_aspect_ratio:
            if left_eye_aspect_ratio < eye_aspect_ratio_threshold:
                wink_event_count += 1

                if wink_event_count > wink_frame_count:
                    pyautogui.click(button='left')

                    wink_event_count = 0

        elif left_eye_aspect_ratio > right_eye_aspect_ratio:
            if right_eye_aspect_ratio < eye_aspect_ratio_threshold:
                wink_event_count += 1

                if wink_event_count > wink_frame_count:
                    pyautogui.click(button='right')

                    wink_event_count = 0
        else:
            wink_event_count = 0
    else:
        if eye_aspect_ratio <= eye_aspect_ratio_threshold:
            eye_event_count += 1

            if eye_event_count > eye_aspect_ratio_frame_count:
                scroll_active = not scroll_active
                # INPUT_MODE = not INPUT_MODE
                eye_event_count = 0

                # nose point to draw a bounding box around it

        else:
            eye_event_count = 0
            wink_event_count = 0

    #--------------------------------------------------------------------------#

    if mouth_aspect_ratio <= 0.35:
        mouth_event_count = 0
        
    else:
        mouth_event_count += 1

        if mouth_event_count >= 20:
            input_active = not input_active
            mouth_event_count = 0
            pivot = nose_point

    

    if input_active:
        cv2.putText(cv2_frame, "INPUT MODE ACTIVE", (300, 30), font, 0.6, red, 1)
        mult = 1
        drag_value = 25 #18
        pivot_x, pivot_y = pivot
        nose_x, nose_y = nose_point
        rectangle_width, rectangle_height = 50, 35 #60, 35
        direction = determine_direction(nose_point, pivot, rectangle_width, rectangle_height)
        cv2.rectangle(cv2_frame, (pivot_x - rectangle_width, pivot_y - rectangle_height), (pivot_x + rectangle_width, pivot_y + rectangle_height), (0, 255, 0), 2)
        cv2.line(cv2_frame, pivot, nose_point, (255, 0, 0), 2)

        cv2.putText(cv2_frame, direction.upper(), (10, 140), font, 0.6, (0, 0, 255), 1)
        if direction == 'right':
            # pyautogui.moveRel(drag_value, 0)
            macmouse.move(drag_value, 0, absolute=False, duration=0.2)

        elif direction == 'left':
            # pyautogui.moveRel(-drag_value, 0)
            macmouse.move(-drag_value, 0, absolute=False, duration=0.2)
        elif direction == 'up':
            if scroll_active:
                # pyautogui.scroll(1000)
                macmouse.wheel(1)
            else:
                macmouse.move(0, -drag_value, absolute=False, duration=0.2)
            
        elif direction == 'down':
            if scroll_active:
                # pyautogui.scroll(-1000)
                macmouse.wheel(-1)
            else:
                macmouse.move(0, drag_value, absolute=False, duration=0.2)
            
    if scroll_active:
        cv2.putText(cv2_frame, 'SCROLL MODE IS ON!', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)
    

    for (x, y) in np.concatenate((mouth_landmarks, left_eye_landmarks, right_eye_landmarks), axis=0):
        cv2.circle(cv2_frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Camera Feed", cv2_frame)
    key = cv2.waitKey(1) & 0xFF

    # Breaking from the loop if Esc key is pressed
    if key == 27:
        break
    fps.update()


fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
cam_capture.release()