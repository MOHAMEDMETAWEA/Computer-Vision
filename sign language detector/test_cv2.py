import sys
print(f"Python executable: {sys.executable}")
print(f"Sys path: {sys.path}")
try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
    print(f"OpenCV file: {cv2.__file__}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
