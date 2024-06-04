import cv2
from cv2 import aruco
import numpy as np

def detect_aruco_markers(image_path, dictionary=aruco.DICT_6X6_250):
    """
    Detect ArUco markers in an image and display the result.

    Parameters:
    - image_path: str, path to the input image file.
    - dictionary: aruco dictionary type (default is aruco.DICT_6X6_250).
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the specified dictionary
    aruco_dict = aruco.getPredefinedDictionary(dictionary)
    
    # Initialize the detector parameters using default values
    parameters = aruco.DetectorParameters()
    
    # Detect the markers
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    # Check if any markers were detected
    if ids is not None:
        # Draw detected markers
        detected_image = aruco.drawDetectedMarkers(image.copy(), corners, ids)
        # Display the result
        cv2.imshow('Detected ArUco markers', detected_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No ArUco markers detected")

# Example usage
if __name__ == "__main__":
    # Path to the input image
    image_path = 'glossy.jpg'
    detect_aruco_markers(image_path)
