import cv2
from PIL import Image
from util import get_limits

# Configuration
COLOR_TO_DETECT = [0, 255, 255]  # Yellow in BGR format
DETECTION_CONFIDENCE_THRESHOLD = 50  # Minimum pixel count to consider detection valid
WINDOW_NAME = 'Color Detection'

def initialize_camera():
    """Initialize and validate camera capture."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video capture.")
    # Set optimal camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

def process_frame(frame, lower_limit, upper_limit):
    """Process a single frame and detect color."""
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for color range
    mask = cv2.inRange(hsv_image, lower_limit, upper_limit)
    
    # Apply morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Get bounding box
    pil_mask = Image.fromarray(mask)
    bbox = pil_mask.getbbox()
    
    return mask, bbox

def draw_detection(frame, bbox):
    """Draw bounding box on frame if detection is valid."""
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Display detection info
        text = f"Detected at ({x1}, {y1})"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def main():
    """Main execution function."""
    try:
        # Initialize
        lower_limit, upper_limit = get_limits(COLOR_TO_DETECT)
        cap = initialize_camera()
        
        print("Starting color detection. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame.")
                break
            
            # Process frame
            mask, bbox = process_frame(frame, lower_limit, upper_limit)
            
            # Draw results
            frame = draw_detection(frame, bbox)
            
            # Display
            cv2.imshow(WINDOW_NAME, frame)
            # Also show mask for debugging
            cv2.imshow('Mask', mask)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")

if __name__ == "__main__":
    main()