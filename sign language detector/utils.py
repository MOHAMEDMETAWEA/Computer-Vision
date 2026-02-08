import numpy as np
import cv2

# Configuration Constants
HAND_LANDMARKER_PATH = 'hand_landmarker.task'
DATA_DIR = './data'
DATA_PICKLE_PATH = './data.pickle'
MODEL_PATH = './model.p'

# Labels Dictionary
# Update these based on your actual sign language gestures
LABELS_DICT = {0: 'A', 1: 'B', 2: 'L'}

def extract_features(hand_landmarks):
    """
    Extracts normalized coordinates and distance features from hand landmarks.
    
    Args:
        hand_landmarks: A list of NormalizedLandmark objects.
        
    Returns:
        data_aux: List of extracted features (normalized coords + distances).
        x_: List of x coordinates (for bounding box calculation).
        y_: List of y coordinates (for bounding box calculation).
    """
    data_aux = []
    x_ = []
    y_ = []

    # Extract raw x, y coordinates
    for landmark in hand_landmarks:
        x = landmark.x
        y = landmark.y
        x_.append(x)
        y_.append(y)

    # Normalize coordinates relative to bounding box
    min_x, max_x = min(x_), max(x_)
    min_y, max_y = min(y_), max(y_)

    # Add normalized landmark positions (42 features)
    for landmark in hand_landmarks:
        x = landmark.x
        y = landmark.y
        data_aux.append((x - min_x) / (max_x - min_x + 1e-6))
        data_aux.append((y - min_y) / (max_y - min_y + 1e-6))

    # Add distances between key landmarks for better feature representation (5 features)
    # Distance from wrist (0) to each fingertip (4, 8, 12, 16, 20)
    wrist_x, wrist_y = hand_landmarks[0].x, hand_landmarks[0].y
    fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

    for tip_idx in fingertips:
        tip_x = hand_landmarks[tip_idx].x
        tip_y = hand_landmarks[tip_idx].y
        distance = np.sqrt((tip_x - wrist_x)**2 + (tip_y - wrist_y)**2)
        data_aux.append(distance)
        
    return data_aux, x_, y_

def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Draws hand landmarks and connections on the image.
    
    Args:
        rgb_image: The input image (can be BGR or RGB).
        detection_result: The detection result from MediaPipe HandLandmarker.
        
    Returns:
        annotated_image: The image with landmarks drawn.
    """
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        
        # Draw the landmarks
        for landmark in hand_landmarks:
            x = int(landmark.x * annotated_image.shape[1])
            y = int(landmark.y * annotated_image.shape[0])
            cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)
            
        # Draw the connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_point = (int(hand_landmarks[start_idx].x * annotated_image.shape[1]),
                           int(hand_landmarks[start_idx].y * annotated_image.shape[0]))
            end_point = (int(hand_landmarks[end_idx].x * annotated_image.shape[1]),
                         int(hand_landmarks[end_idx].y * annotated_image.shape[0]))
            
            cv2.line(annotated_image, start_point, end_point, (255, 255, 255), 2)
            
    return annotated_image
