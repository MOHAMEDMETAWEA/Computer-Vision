import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceLandmarksDetector:
    def __init__(self, static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5):
        """
        Initializes the MediaPipe Face Landmarker (Tasks API).
        """
        model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence
        )
        
        # Set running mode
        if static_image_mode:
            options.running_mode = vision.RunningMode.IMAGE
        else:
            # Note: VIDEO mode requires timestamps. For simplicity in this wrapper,
            # we might use IMAGE mode if we don't strictly need tracking state,
            # or we need to manage timestamps. 
            # Given test_model.py loops and calls it, IMAGE mode is safer unless we track timestamps.
            # However, for video consistency, VIDEO mode is better.
            # To avoid timestamp management complexity for the user loop, we'll use IMAGE mode
            # which works for frame-by-frame anyway, just slightly less temporal smoothing.
            options.running_mode = vision.RunningMode.IMAGE 
        
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def get_landmarks(self, image, draw=False):
        """
        Extracts normalized face landmarks from an image.
        Returns flattened list of (x, y, z) normalized coordinates.
        """
        if image is None:
            return []
            
        # Convert the BGR image to RGB
        image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_input_rgb)
        
        # Detect
        # If we used VIDEO mode, we'd need: self.landmarker.detect_for_video(mp_image, valid_timestamp_ms)
        detection_result = self.landmarker.detect(mp_image)

        image_landmarks = []

        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0] # List of NormalizedLandmark
            
            if draw:
                # Manual drawing since solutions.drawing_utils is missing
                h, w, _ = image.shape
                for lm in face_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 1, (255, 255, 0), -1)

            # Get coordinates
            xs = [lm.x for lm in face_landmarks]
            ys = [lm.y for lm in face_landmarks]
            zs = [lm.z for lm in face_landmarks]

            # Better Normalization:
            # 1. Center the landmarks around the mean (translation invariance)
            mean_x, mean_y, mean_z = sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)
            
            # 2. Scale landmarks by the maximum distance from mean (scale invariance)
            max_dist = 0
            for x, y, z in zip(xs, ys, zs):
                dist = ((x - mean_x)**2 + (y - mean_y)**2 + (z - mean_z)**2)**0.5
                if dist > max_dist:
                    max_dist = dist
            
            if max_dist == 0: max_dist = 1 # Avoid division by zero

            for x, y, z in zip(xs, ys, zs):
                image_landmarks.append((x - mean_x) / max_dist)
                image_landmarks.append((y - mean_y) / max_dist)
                image_landmarks.append((z - mean_z) / max_dist)

        return image_landmarks

    def get_bbox(self, image):
        """Returns the bounding box (x, y, w, h) in pixels."""
        if image is None: return None
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        detection_result = self.landmarker.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return None
            
        h, w, _ = image.shape
        landmarks = detection_result.face_landmarks[0]
        px = [int(lm.x * w) for lm in landmarks]
        py = [int(lm.y * h) for lm in landmarks]
        
        return (min(px), min(py), max(px) - min(px), max(py) - min(py))

    def close(self):
        """Releases the underlying resources."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()

# For backward compatibility
def get_face_landmarks(image, draw=False, static_image_mode=True):
    detector = FaceLandmarksDetector(static_image_mode=static_image_mode)
    try:
        return detector.get_landmarks(image, draw=draw)
    finally:
        detector.close()