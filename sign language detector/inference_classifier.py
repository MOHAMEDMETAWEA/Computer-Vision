import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque


try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: model.p not found. Please run train_classifier.py first.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
    
# Set resolution for better quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

import utils

# Initialize MediaPipe Hands using the new Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Create HandLandmarker options
base_options = python.BaseOptions(model_asset_path=utils.HAND_LANDMARKER_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5
)

# Create the hand landmarker
hands = vision.HandLandmarker.create_from_options(options)

# Labels
labels_dict = utils.LABELS_DICT

# Prediction smoothing using a queue
prediction_queue = deque(maxlen=5)  # Keep last 5 predictions for smoothing

print("Starting sign language detection...")
print("Press 'q' to quit")
print("="*50)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    # Detect hand landmarks
    results = hands.detect(mp_image)
    
    # Visualize the results
    if results.hand_landmarks:
        frame = utils.draw_landmarks_on_image(frame, results)
        
        for hand_landmarks in results.hand_landmarks:
            # Feature Extraction using shared utility
            data_aux, x_, y_ = utils.extract_features(hand_landmarks)

            # Bounding Box Calculation
            x1 = int(min(x_) * W) - 20
            y1 = int(min(y_) * H) - 20
            x2 = int(max(x_) * W) + 20
            y2 = int(max(y_) * H) + 20

            # Prediction with confidence
            try:
                prediction = model.predict([np.asarray(data_aux)])
                prediction_proba = model.predict_proba([np.asarray(data_aux)])
                
                predicted_class = int(prediction[0])
                confidence = np.max(prediction_proba) * 100
                
                # Add to prediction queue for smoothing
                prediction_queue.append(predicted_class)
                
                # Use most common prediction in queue (voting)
                if len(prediction_queue) >= 3:
                    smoothed_prediction = max(set(prediction_queue), key=prediction_queue.count)
                    predicted_character = labels_dict.get(smoothed_prediction, "?")
                else:
                    predicted_character = labels_dict.get(predicted_class, "?")
                
                # Color based on confidence
                if confidence > 80:
                    color = (0, 255, 0)  # Green for high confidence
                elif confidence > 60:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 0, 255)  # Red for low confidence
                
            except Exception as e:
                predicted_character = "?"
                confidence = 0
                color = (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw prediction text with background
            text = f"{predicted_character} ({confidence:.1f}%)"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            
            # Background rectangle for text
            cv2.rectangle(frame, 
                         (x1, y1 - text_size[1] - 20), 
                         (x1 + text_size[0] + 10, y1), 
                         color, -1)
            
            # Text
            cv2.putText(frame, text, (x1 + 5, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display FPS
    if frame_count % 30 == 0:
        fps_text = f"Frame: {frame_count}"
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Instructions
    cv2.putText(frame, "Press 'q' to quit", (10, H - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Sign Language Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nDetection stopped.")
