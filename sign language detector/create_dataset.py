import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

import utils

# Initialize MediaPipe Hands using the new Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Create HandLandmarker options
base_options = python.BaseOptions(model_asset_path=utils.HAND_LANDMARKER_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,  # Focus on one hand at a time
    min_hand_detection_confidence=0.7,  # Higher confidence threshold
    min_hand_presence_confidence=0.5
)

# Create the hand landmarker
hands = vision.HandLandmarker.create_from_options(options)

DATA_DIR = utils.DATA_DIR

data = []
labels = []
skipped_images = 0
processed_images = 0

print(f"Starting dataset creation from {DATA_DIR}...")
for dir_ in os.listdir(DATA_DIR):
    img_dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(img_dir_path):
        continue

    print(f"\nProcessing class {dir_}...")
    class_processed = 0
    
    for img_path in os.listdir(img_dir_path):
        try:
            img = cv2.imread(os.path.join(img_dir_path, img_path))
            if img is None:
                skipped_images += 1
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to MediaPipe Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            # Detect hand landmarks
            results = hands.detect(mp_image)
            
            if results.hand_landmarks:
                # Only use the first hand detected to ensure consistent feature length
                hand_landmarks = results.hand_landmarks[0]
                
                # Extract features using shared utility function
                data_aux, _, _ = utils.extract_features(hand_landmarks)

                data.append(data_aux)
                labels.append(dir_)
                processed_images += 1
                class_processed += 1
            else:
                skipped_images += 1
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            skipped_images += 1
    
    print(f"Class {dir_}: {class_processed} images processed")

print(f"\n{'='*50}")
print(f"Dataset creation complete!")
print(f"Total images processed: {processed_images}")
print(f"Total images skipped: {skipped_images}")
print(f"Number of classes: {len(set(labels))}")
print(f"{'='*50}\n")

# Save the dataset
if len(data) > 0:
    f = open('data.pickle', 'wb')
    pickle.dump({'data': data, 'labels': labels}, f)
    f.close()
    print("Dataset saved to 'data.pickle'")
else:
    print("ERROR: No data was processed. Please check your images and try again.")

