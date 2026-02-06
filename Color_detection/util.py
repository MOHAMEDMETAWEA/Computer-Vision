import numpy as np
import cv2

# Function to get the lower and upper limits of a color in HSV color space
# The input color is expected to be in BGR format (as used by OpenCV)
# The function returns the lower and upper limits of the color in HSV format, which can be used for color detection in images.
# The limits are calculated by taking the hue value of the input color and creating a range of ±10 around it, while keeping the saturation and value at their maximum ranges (100-255 for saturation and 100-255 for value) to ensure a wide detection range for the specified color.
# detect color in HSV color space, which is more robust to lighting changes compared to RGB color space.

def get_limits(color):

    c = np.uint8([[color]])  # Convert the color to a NumPy array
    hsv_color = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[0][0]  # Convert the color to HSV

    lower_limit = np.array([hsv_color[0] - 10, 100, 100])  # Lower limit of the color range
    upper_limit = np.array([hsv_color[0] + 10, 255, 255])  # Upper limit of the color range

    return lower_limit, upper_limit