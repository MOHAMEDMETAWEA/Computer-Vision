import pickle
import cv2
import os
import time
from collections import deque
import numpy as np
from utils import FaceLandmarksDetector

# Labels matching the model's training order
EMOTIONS = ['ANGRY', 'HAPPY', 'SAD', 'SURPRISED']
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)] # BGR

# Load the trained model
model_path = './model'
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found. Please run train_model.py first.")
    exit()

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Initialize detector
detector = FaceLandmarksDetector(static_image_mode=False)

# Smoothing buffer for probabilities (more stable than labels)
proba_buffer = deque(maxlen=8)

# Initialize camera
cap = cv2.VideoCapture(0)
prev_time = time.time()

print("Starting Professional Emotion Recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror effect (more natural for users)
    frame = cv2.flip(frame, 1)
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Darken background slightly for UI contrast
    overlay = frame.copy()
    
    # Extract landmarks
    face_landmarks = detector.get_landmarks(frame, draw=True)
    bbox = detector.get_bbox(frame)

    # 1. Dashboard UI setup
    h, w, _ = frame.shape
    dashboard_w = 220
    cv2.rectangle(frame, (0, 0), (dashboard_w, h), (40, 40, 40), -1)
    cv2.line(frame, (dashboard_w, 0), (dashboard_w, h), (120, 120, 120), 1)

    if face_landmarks and bbox:
        x, y, bw, bh = bbox
        # Draw stylish corner brackets for face detection
        length = 30
        t = 2
        # Top Left
        cv2.line(frame, (x, y), (x + length, y), (255, 255, 255), t)
        cv2.line(frame, (x, y), (x, y + length), (255, 255, 255), t)
        # Top Right
        cv2.line(frame, (x + bw, y), (x + bw - length, y), (255, 255, 255), t)
        cv2.line(frame, (x + bw, y), (x + bw, y + length), (255, 255, 255), t)
        # Bottom Left
        cv2.line(frame, (x, y + bh), (x + length, y + bh), (255, 255, 255), t)
        cv2.line(frame, (x, y + bh), (x, y + bh - length), (255, 255, 255), t)
        # Bottom Right
        cv2.line(frame, (x + bw, y + bh), (x + bw - length, y + bh), (255, 255, 255), t)
        cv2.line(frame, (x + bw, y + bh), (x + bw, y + bh - length), (255, 255, 255), t)

        # Get probabilities for all classes
        probas = model.predict_proba([face_landmarks])[0]
        proba_buffer.append(probas)
        
        # Average probabilities over the buffer for stability
        avg_probas = np.mean(list(proba_buffer), axis=0)
        max_idx = np.argmax(avg_probas)
        current_emotion = EMOTIONS[max_idx]
        confidence = avg_probas[max_idx]

        # Draw Emotion Indicator
        cv2.putText(frame, f"EMOTION: {current_emotion}", (dashboard_w + 20, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, COLORS[max_idx], 2)
        cv2.putText(frame, f"CONF: {confidence*100:.1f}%", (dashboard_w + 20, 90),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Draw Confidence Bars on Dashboard
        for i, (emotion, proba) in enumerate(zip(EMOTIONS, avg_probas)):
            y_pos = 150 + i * 70
            bar_max_w = 180
            bar_w = int(proba * bar_max_w)
            
            # Label
            cv2.putText(frame, emotion, (20, y_pos - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            # Background Bar
            cv2.rectangle(frame, (20, y_pos), (20 + bar_max_w, y_pos + 15), (60, 60, 60), -1)
            # Colored Bar
            cv2.rectangle(frame, (20, y_pos), (20 + bar_w, y_pos + 15), COLORS[i], -1)
            # Percentage
            cv2.putText(frame, f"{proba*100:.0f}%", (165, y_pos - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    else:
        proba_buffer.clear()
        cv2.putText(frame, "WAITING FOR FACE...", (dashboard_w + 20, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (100, 100, 255), 2)
        
        # Static text for dashboard when no face
        cv2.putText(frame, "System: Ready", (20, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

    # FPS and System Info
    cv2.putText(frame, f"FPS: {int(fps)}", (w - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Real-World Emotion Recognition Pro', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()