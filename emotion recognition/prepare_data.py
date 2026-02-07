import os
import cv2
import numpy as np
from utils import FaceLandmarksDetector

data_dir = './data'
output = []

# Initialize the detector once for all images
detector = FaceLandmarksDetector(static_image_mode=True)

print(f"Processing images in {data_dir}...")

for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
    emotion_path = os.path.join(data_dir, emotion)
    if not os.path.isdir(emotion_path):
        continue
        
    print(f"  Processing emotion: {emotion} (index: {emotion_indx})")
    
    count = 0
    for image_name in os.listdir(emotion_path):
        image_path = os.path.join(emotion_path, image_name)
        image = cv2.imread(image_path)
        
        if image is None:
            continue

        try:
            face_landmarks = detector.get_landmarks(image)

            # 478 landmarks * 3 coordinates (x, y, z) = 1434 (MediaPipe Tasks API)
            # 468 landmarks * 3 coordinates (x, y, z) = 1404 (Legacy API)
            if len(face_landmarks) in [1404, 1434]:
                face_landmarks.append(int(emotion_indx))
                output.append(face_landmarks)
                count += 1
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            
    print(f"  Processed {count} valid samples for {emotion}")

# Save the processed data
try:
    if output:
        output_file = 'data.txt'
        np.savetxt(output_file, np.asarray(output))
        print(f"Data preparation complete. Saved {len(output)} samples to {output_file}.")
    else:
        print("No samples found! Check images and detector.")
finally:
    if detector:
        detector.close()


