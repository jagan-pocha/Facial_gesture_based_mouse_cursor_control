
from utilities import np

# Calculate EAR (Eye Aspect Ratio) from eye landmarks
def calculate_ear(eye_landmarks):
    landmarks_1_5 = eye_landmarks[1] - eye_landmarks[5]
    landmarks_2_4 = eye_landmarks[2] - eye_landmarks[4]
    landmarks_0_3 = eye_landmarks[0] - eye_landmarks[3]

    dist_1 = np.linalg.norm(landmarks_1_5)
    dist_2 = np.linalg.norm(landmarks_2_4)
    dist_3 = np.linalg.norm(landmarks_0_3)

    ear_value = (dist_1 + dist_2) / (2.0 * dist_3)

    return ear_value

# Calculate MAR (Mouth Aspect Ratio) from mouth landmarks
def calculate_mar(mouth_landmarks):
    landmarks_13_19 = mouth_landmarks[13] - mouth_landmarks[19]
    landmarks_14_18 = mouth_landmarks[14] - mouth_landmarks[18]
    landmarks_15_17 = mouth_landmarks[15] - mouth_landmarks[17]
    landmarks_12_16 = mouth_landmarks[12] - mouth_landmarks[16]

    dist_1 = np.linalg.norm(landmarks_13_19)
    dist_2 = np.linalg.norm(landmarks_14_18)
    dist_3 = np.linalg.norm(landmarks_15_17)
    dist_4 = np.linalg.norm(landmarks_12_16)

    mar_value = (dist_1 + dist_2 + dist_3) / (2 * dist_4)

    return mar_value

# Determine direction based on nose and anchor points
def determine_direction(nose_pt, anchor_pt, width, height, multiplier=1):
    nose_x, nose_y = nose_pt
    anchor_x, anchor_y = anchor_pt

    if nose_x > anchor_x + multiplier * width:
        return 'right'
    elif nose_x < anchor_x - multiplier * width:
        return 'left'

    if nose_y > anchor_y + multiplier * height:
        return 'down'
    elif nose_y < anchor_y - multiplier * height:
        return 'up'

    return 'none'
