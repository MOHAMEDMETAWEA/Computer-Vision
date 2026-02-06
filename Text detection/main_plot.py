"""
OCR Visualization Tool using EasyOCR
------------------------------------
This script provides a simple way to detect text within an image and visualize 
the results by drawing bounding boxes and labels using OpenCV and Matplotlib.

Author: Antigravity AI
Date: 2024
"""

import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import os

def detect_and_plot(image_path, languages=['en'], threshold=0.25):
    """
    Detects text in an image using EasyOCR and plots the results with bounding boxes.

    Args:
        image_path (str): Path to the input image file.
        languages (list): List of language codes for OCR (default: ['en']).
        threshold (float): Confidence score threshold to filter detections (default: 0.25).
    """
    # Check if the image file exists before proceeding
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # 1. Initialize the EasyOCR reader
    # Note: gpu=False ensures it runs on CPU if no GPU is available
    print(f"Initializing EasyOCR with languages: {languages}")
    reader = easyocr.Reader(languages, gpu=False)

    # 2. Read the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # 3. Perform text detection
    # readtext() returns a list of results: (bbox, text, confidence_score)
    print(f"Detecting text in {image_path}...")
    results = reader.readtext(img)

    # 4. Process and draw results
    for (bbox, text, score) in results:
        # Only show results above the confidence threshold
        if score > threshold:
            print(f"Detected: '{text}' with confidence {score:.4f}")
            
            # Extract coordinates for the bounding box (bbox is a list of 4 points)
            # We use the top-left (point 0) and bottom-right (point 2)
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            
            # Draw a green rectangle around the detected text
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            
            # Put the detected text label in red above the box
            # Subtract 10 from Y to place text slightly above the rectangle
            cv2.putText(img, text, (top_left[0], top_left[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 5. Display the result using Matplotlib
    plt.figure(figsize=(10, 8))
    # Convert BGR (OpenCV default) to RGB for correct display in Matplotlib
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"OCR Results for {os.path.basename(image_path)}")
    plt.axis('off') # Hide axes
    plt.show()

# Entry point of the script
if __name__ == "__main__":
    # Define the path to the test image
    test_image = 'test.png'
    
    # Run the detection if the test image exists
    if os.path.exists(test_image):
        detect_and_plot(test_image)
    else:
        # Inform the user how to run the script if test.png is missing
        print(f"Please provide a valid image path. '{test_image}' not found.")
        print("Usage: detect_and_plot('path_to_your_image.jpg')")

