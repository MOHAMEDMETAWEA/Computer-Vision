import cv2
import numpy as np
import os

class FaceAnonymizer:
    def __init__(self, min_detection_confidence=0.5):
        """
        Initializes the FaceAnonymizer using OpenCV Haar Cascades.
        """
        self.min_confidence = min_detection_confidence
        
        # Load multiple cascades for better detection
        self.frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        if self.frontal_cascade.empty():
            raise RuntimeError("Could not load Frontal Face Haar Cascade")

    def _get_face_coords(self, frame):
        """
        Detects faces using multiple Haar Cascades and returns their bounding boxes.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect frontal faces
        # We use minNeighbors to approximate confidence (higher = more confident)
        min_neighbors = int(self.min_confidence * 10) + 2
        faces_frontal = self.frontal_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(30, 30))
        
        # Detect profile faces
        faces_profile = self.profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(30, 30))
        
        # Combine and remove overlaps
        all_faces = []
        if len(faces_frontal) > 0: all_faces.extend(faces_frontal)
        if len(faces_profile) > 0: all_faces.extend(faces_profile)
        
        if not all_faces:
            return []

        # Simple overlap removal (Non-Maximum Suppression-ish)
        all_faces = sorted(all_faces, key=lambda x: x[2]*x[3], reverse=True)
        final_faces = []
        for (x, y, w, h) in all_faces:
            is_overlap = False
            for (fx, fy, fw, fh) in final_faces:
                # Check if center of current face is inside an already accepted face
                if (x + w/2 > fx and x + w/2 < fx + fw and
                    y + h/2 > fy and y + h/2 < fy + fh):
                    is_overlap = True
                    break
            if not is_overlap:
                final_faces.append((x, y, w, h))

        face_coords = []
        ih, iw = frame.shape[:2]
        for (x, y, w, h) in final_faces:
            # Add some padding
            padding_w = int(0.15 * w)
            padding_h = int(0.15 * h)
            
            x1 = max(0, x - padding_w)
            y1 = max(0, y - padding_h)
            x2 = min(iw, x + w + padding_w)
            y2 = min(ih, y + h + padding_h)
            
            face_coords.append((x1, y1, x2, y2))
            
        return face_coords

    def apply_blur(self, frame, x1, y1, x2, y2, intensity=99):
        """Applies Gaussian blur to the face region."""
        if x2 <= x1 or y2 <= y1: return frame
        kernel_size = max(1, intensity if intensity % 2 != 0 else intensity + 1)
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0: return frame
        
        blurred_face = cv2.GaussianBlur(face_roi, (kernel_size, kernel_size), 0)
        frame[y1:y2, x1:x2] = blurred_face
        return frame

    def apply_blur_oval(self, frame, x1, y1, x2, y2, intensity=99):
        """Applies Gaussian blur within an oval region to look more natural."""
        if x2 <= x1 or y2 <= y1: return frame
        
        face_roi = frame[y1:y2, x1:x2].copy()
        if face_roi.size == 0: return frame
        
        kernel_size = max(1, intensity if intensity % 2 != 0 else intensity + 1)
        blurred_roi = cv2.GaussianBlur(face_roi, (kernel_size, kernel_size), 0)
        
        # Create an oval mask
        mask = np.zeros(face_roi.shape[:2], dtype=np.uint8)
        center = ((x2-x1)//2, (y2-y1)//2)
        axes = ((x2-x1)//2, (y2-y1)//2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # Apply mask
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        frame[y1:y2, x1:x2] = (face_roi * (1 - mask_3ch) + blurred_roi * mask_3ch).astype(np.uint8)
        return frame

    def apply_pixelate(self, frame, x1, y1, x2, y2, intensity=99):
        """Applies pixelation effect to the face region."""
        if x2 <= x1 or y2 <= y1: return frame
        face_roi = frame[y1:y2, x1:x2]
        h, w = face_roi.shape[:2]
        if h == 0 or w == 0: return frame
        
        # Scale: higher intensity = lower scale = larger pixels
        scale = max(0.005, 10.0 / (intensity + 5))
        
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        temp = cv2.resize(face_roi, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pixelated_face = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        
        frame[y1:y2, x1:x2] = pixelated_face
        return frame

    def apply_blackout(self, frame, x1, y1, x2, y2):
        """Covers the face region with a black rectangle."""
        if x2 <= x1 or y2 <= y1: return frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
        return frame

    def process_frame(self, frame, mode='blur', intensity=99):
        """
        Main entry point for processing a single frame.
        """
        coords = self._get_face_coords(frame)
        for (x1, y1, x2, y2) in coords:
            if mode == 'blur':
                frame = self.apply_blur(frame, x1, y1, x2, y2, intensity)
            elif mode == 'blur_oval':
                frame = self.apply_blur_oval(frame, x1, y1, x2, y2, intensity)
            elif mode == 'pixelate':
                frame = self.apply_pixelate(frame, x1, y1, x2, y2, intensity)
            elif mode == 'blackout':
                frame = self.apply_blackout(frame, x1, y1, x2, y2)
        return frame, len(coords)

    def __del__(self):
        pass
